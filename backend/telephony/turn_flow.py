"""Turn-based (non-streaming) call flow for Twilio trial accounts.

Trial accounts strip the <Start><Stream> verb from custom TwiML (it is replaced
with a static "not available" message), so the real-time Media Streams agent
cannot run while the account is in trial mode. Instead this module drives the
call with a <Gather input="speech"> turn loop, which the trial supports:

  * Twilio transcribes each utterance with its own Speech Recognition
    (150 free trial units) and POSTs the transcript to our webhook,
  * we run the same ConversationEngine over the transcript,
  * the reply is synthesized with Sarvam TTS (150 free trial units),
  * the audio is hosted at a public URL and handed back to Twilio as
    <Play> + <Gather>, so the caller hears the agent and then speaks again.

Only <Call minutes> (75 free trial units) are consumed per call, plus the
TTS/Speech units listed above. The real-time streaming path is kept intact for
accounts that have upgraded.
"""

from __future__ import annotations

import asyncio
import html
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import Settings
from backend.conversation import ConversationEngine
from backend.database import session_scope
from backend.errors import AppError
from backend.metrics import TurnTimings, persist_turn_telemetry
from backend.call_profile import build_call_profile
from backend.models import Lead, Message, Session
from backend.language_utils import (
    detect_explicit_language_switch,
    infer_script_language,
    resolve_turn_language,
)
from backend.pipeline_trace import trace
from backend.providers.sarvam_client import SarvamClient
from backend.streaming_json import FirstSpeechChunkBuffer
from backend.telephony.call_manager import (
    CallRecord,
    CallRegistry,
    fallback_text,
)
from backend.telephony.twilio_service import TwilioService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

TTS_DIR_NAME = "tts"
MAX_HOSTED_FILES = 50
HOSTED_FILE_TTL_SECONDS = 2 * 3600

# Streaming TTS: how long we wait for the first audio chunk before falling back
# to the buffered endpoint, and how long a finished stream stays consumable.
TTS_FIRST_CHUNK_TIMEOUT_SECONDS = 2.0
TTS_STREAM_TTL_SECONDS = 120

# Twilio call-processing webhooks hard-cap at ~15s. Heavy LLM+TTS work must
# NOT run inside that request - we return a short hold + Redirect to
# /turn-result and finish there so Twilio never plays "could not reach TwiML".
# Defaults favor Gather latency; Settings can override per deployment.
TURN_INLINE_BUDGET_SECONDS = 5.0
TURN_POLL_WAIT_SECONDS = 12.0
TURN_POLL_PAUSE_SECONDS = 0
MAX_TURN_POLLS = 6
PENDING_TURN_TTL_SECONDS = 180

# default_language -> <Gather language> attribute for Twilio Speech Recognition
GATHER_LANGUAGES = {
    "en": "en-IN",
    "en-in": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hinglish": "hi-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "gujarati": "gu-IN",
    "gujlish": "gu-IN",
}


def _gather_language(default_language: str) -> str:
    key = (default_language or "").strip().lower()
    return GATHER_LANGUAGES.get(key, "en-IN")


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


@dataclass
class TtsStream:
    """In-memory buffer for one streaming TTS synthesis.

    The background TTS task pushes raw audio chunks into ``queue`` while the
    /api/calls/audio/{token} endpoint drains them for Twilio <Play>.
    """

    audio_type: str = "audio/wav"
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None
    timings: TurnTimings | None = None


@dataclass
class PendingTurn:
    """In-flight LLM+TTS work that may outlive one Twilio webhook request."""

    call_sid: str
    token: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    twiml: str | None = None
    error: Exception | None = None
    polls: int = 0
    language: str | None = None
    task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.monotonic)
    timings: TurnTimings | None = None
    early_ready: asyncio.Event = field(default_factory=asyncio.Event)
    early_twiml: str | None = None
    early_text: str = ""
    early_delivered: bool = False


class TurnFlow:
    """Runs the <Gather input="speech"> turn loop for one call at a time."""

    def __init__(
        self,
        settings: Settings,
        session_factory,
        engine: ConversationEngine,
        sarvam: SarvamClient,
        twilio: TwilioService,
        registry: CallRegistry,
    ) -> None:
        self._settings = settings
        self._factory = session_factory
        self._engine = engine
        self._sarvam = sarvam
        self._twilio = twilio
        self._registry = registry
        self._tts_streams: dict[str, TtsStream] = {}
        self._hosted_timings: dict[str, TurnTimings] = {}
        self._pending_turns: dict[str, PendingTurn] = {}
        # lang -> (cached_at, twiml, greeting_text)
        self._greeting_cache: dict[str, tuple[float, str, str]] = {}
        self._greeting_refreshing: set[str] = set()
        self._greeting_lock = asyncio.Lock()

    # ---------- TwiML builders ----------

    def _inline_budget_seconds(self) -> float:
        try:
            configured = float(self._settings.gather_inline_budget_seconds)
        except (TypeError, ValueError, AttributeError):
            configured = TURN_INLINE_BUDGET_SECONDS
        # Tests monkeypatch the module constant to a tiny value.
        if TURN_INLINE_BUDGET_SECONDS < 1.0:
            return float(TURN_INLINE_BUDGET_SECONDS)
        return configured

    def _poll_pause_seconds(self) -> int:
        try:
            return max(0, int(self._settings.gather_poll_pause_seconds))
        except (TypeError, ValueError, AttributeError):
            return max(0, int(TURN_POLL_PAUSE_SECONDS))

    def _gather_attrs(self, turn_url: str, language: str | None = None) -> str:
        lang = _gather_language(language or self._settings.default_language)
        speech_timeout = (
            getattr(self._settings, "gather_speech_timeout", None) or "1"
        ).strip() or "1"
        try:
            gather_timeout = int(
                getattr(self._settings, "gather_timeout_seconds", 5) or 5
            )
        except (TypeError, ValueError):
            gather_timeout = 5
        gather_timeout = max(1, min(gather_timeout, 15))
        return (
            'input="speech" action="{url}" method="POST" '
            'speechTimeout="{speech_timeout}" timeout="{timeout}" '
            'language="{lang}" actionOnEmptyResult="true"'
        ).format(
            url=_escape(turn_url),
            lang=lang,
            speech_timeout=_escape(speech_timeout),
            timeout=gather_timeout,
        )

    def _gather_twiml(
        self,
        prompt_url: str | None,
        prompt_text: str | None = None,
        language: str | None = None,
    ) -> str:
        """TwiML that plays the agent prompt (or says it) and listens for speech."""
        turn_url = self._twilio.turn_url()
        if prompt_url:
            inner = f"<Play>{_escape(prompt_url)}</Play>"
        elif prompt_text:
            inner = f"<Say>{_escape(prompt_text)}</Say>"
        else:
            inner = ""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Gather {self._gather_attrs(turn_url, language)}>{inner}</Gather></Response>'
        )

    def _pending_redirect_twiml(self, pending: PendingTurn) -> str:
        """Keep the call alive while LLM+TTS finishes in the background.

        Use a silent Pause (not <Say>). Twilio's default Say voice cannot speak
        Gujarati/Hindi script and turns hold phrases like "એક સેકન્ડ" into
        garbled one-word noise ("start", etc.).
        """
        if pending.timings is not None:
            pending.timings.mark("redirect_issued")
        url = self._twilio.turn_result_url(pending.call_sid, pending.token)
        pause_s = self._poll_pause_seconds()
        # Pause length 0 wastes a full second of perceived latency; skip it.
        pause_xml = f'<Pause length="{pause_s}"/>' if pause_s > 0 else ""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"{pause_xml}"
            f'<Redirect method="POST">{_escape(url)}</Redirect>'
            "</Response>"
        )

    def _early_reply_twiml(self, pending: PendingTurn, audio_url: str) -> str:
        """Play the first generated sentence, then fetch the completed turn."""
        result_url = self._twilio.turn_result_url(
            pending.call_sid, pending.token
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Play>{_escape(audio_url)}</Play>"
            f'<Redirect method="POST">{_escape(result_url)}</Redirect>'
            "</Response>"
        )

    async def _wait_for_pending_progress(
        self, pending: PendingTurn, timeout: float
    ) -> str:
        """Wait until early audio, the full turn, or the webhook budget wins."""
        done_wait = asyncio.create_task(pending.done.wait())
        early_wait = (
            asyncio.create_task(pending.early_ready.wait())
            if not pending.early_delivered
            else None
        )
        waiters = {done_wait}
        if early_wait is not None:
            waiters.add(early_wait)
        try:
            finished, _ = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not finished:
                return "timeout"
            if (
                pending.early_ready.is_set()
                and pending.early_twiml
                and not pending.early_delivered
            ):
                return "early"
            return "done" if pending.done.is_set() else "timeout"
        finally:
            if early_wait is not None:
                early_wait.cancel()
            done_wait.cancel()

    @staticmethod
    def _take_early_twiml(pending: PendingTurn) -> str | None:
        if (
            pending.early_delivered
            or not pending.early_ready.is_set()
            or not pending.early_twiml
        ):
            return None
        pending.early_delivered = True
        return pending.early_twiml

    def _say_gather_twiml(self, text: str, language: str | None = None) -> str:
        """Gather after a short English <Say>.

        Twilio Polly cannot reliably speak Indic script, so any path that must
        talk without Sarvam TTS uses English only.
        """
        return self._gather_twiml(None, text, language=language)

    def _farewell_twiml(
        self,
        reply_url: str | None,
        reply_text: str,
    ) -> str:
        """TwiML that plays a short goodbye and hangs up (caller ended the call)."""
        parts: list[str] = []
        if reply_url:
            parts.append(f"<Play>{_escape(reply_url)}</Play>")
        elif reply_text:
            parts.append(f"<Say>{_escape(reply_text)}</Say>")
        parts.append("<Hangup/>")
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response>{''.join(parts)}</Response>"
        )

    def _final_twiml(
        self,
        reply_url: str | None,
        goodbye_url: str | None,
        reply_text: str,
        goodbye_text: str,
    ) -> str:
        """TwiML that plays the final reply + goodbye and hangs up."""
        parts: list[str] = []
        if reply_url:
            parts.append(f"<Play>{_escape(reply_url)}</Play>")
        elif reply_text:
            parts.append(f"<Say>{_escape(reply_text)}</Say>")
        if goodbye_url:
            parts.append(f"<Play>{_escape(goodbye_url)}</Play>")
        elif goodbye_text:
            parts.append(f"<Say>{_escape(goodbye_text)}</Say>")
        parts.append("<Hangup/>")
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response>{''.join(parts)}</Response>"
        )

    # ---------- TTS audio hosting ----------

    def _tts_dir(self) -> Path:
        return self._settings.resolved_temp_dir / TTS_DIR_NAME

    def _sweep_tts_dir(self) -> None:
        """Keep the hosted-audio folder from growing without bound."""
        try:
            directory = self._tts_dir()
            if not directory.is_dir():
                return
            files = sorted(
                directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if len(files) <= MAX_HOSTED_FILES:
                return
            now = time.time()
            for path in files[MAX_HOSTED_FILES:]:
                try:
                    if now - path.stat().st_mtime > HOSTED_FILE_TTL_SECONDS:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue
        except OSError:
            pass

    async def _host_tts(
        self,
        text: str,
        detected_language: str | None,
        timings: TurnTimings | None = None,
    ) -> tuple[str | None, int]:
        """Synthesize with Sarvam and return a public URL for <Play>."""
        try:
            audio, _, tts_ms = await self._sarvam.synthesize(text, detected_language)
        except AppError as exc:
            logger.warning("Turn TTS failed: %s", exc.code)
            return None, 0
        if not audio:
            return None, 0
        if timings is not None:
            timings.retry_count += max(
                0, getattr(self._sarvam, "last_attempt_count", 1) - 1
            )
            timings.tts_provider = "sarvam"
            if timings.tts_mode != "streaming_to_buffered":
                timings.tts_mode = "buffered"
            timings.tts_attempted = True
            timings.tts_latency_ms = tts_ms
            timings.mark("tts_first_audio")
        self._sweep_tts_dir()
        file_id = uuid.uuid4().hex
        path = self._tts_dir() / f"{file_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        if timings is not None:
            self._hosted_timings[file_id] = timings
        return self._twilio.audio_url(file_id), tts_ms

    async def _stream_tts(
        self,
        text: str,
        detected_language: str | None,
        timings: TurnTimings | None = None,
    ) -> tuple[str | None, int]:
        """Start streaming TTS and return a public <Play> URL for it.

        Returns once the first audio chunk has arrived (so the /audio route can
        already serve data), then the stream keeps filling in the background.
        Returns (None, 0) when streaming is disabled or produces no audio in
        time, so callers can fall back to the buffered file path.
        """
        if not self._settings.sarvam_tts_streaming:
            return await self._host_tts(text, detected_language, timings)
        token = uuid.uuid4().hex
        stream = TtsStream(timings=timings)
        self._tts_streams[token] = stream
        if timings is not None:
            timings.tts_provider = "sarvam"
            timings.tts_mode = "streaming"
            timings.tts_attempted = True
        task = asyncio.create_task(
            self._run_tts_stream(token, stream, text, detected_language)
        )
        ready_task = asyncio.create_task(stream.ready.wait())
        done_task = asyncio.create_task(stream.done.wait())
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                asyncio.wait(
                    {ready_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                ),
                timeout=TTS_FIRST_CHUNK_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            self._tts_streams.pop(token, None)
            logger.warning("Streaming TTS produced no audio in time; using buffered TTS.")
            if timings is not None:
                timings.fallback_count += 1
                timings.tts_mode = "streaming_to_buffered"
            return await self._host_tts(text, detected_language, timings)
        finally:
            ready_task.cancel()
            done_task.cancel()
        if not stream.ready.is_set():
            task.cancel()
            self._tts_streams.pop(token, None)
            if timings is not None:
                timings.fallback_count += 1
                timings.tts_mode = "streaming_to_buffered"
            return await self._host_tts(text, detected_language, timings)
        first_chunk_ms = int((time.monotonic() - started) * 1000)
        if timings is not None:
            timings.tts_latency_ms = first_chunk_ms
        return self._twilio.audio_url(token), first_chunk_ms

    async def _run_tts_stream(
        self,
        token: str,
        stream: TtsStream,
        text: str,
        detected_language: str | None,
    ) -> None:
        """Background task: pull Sarvam chunks into the in-memory stream queue."""
        try:
            async for chunk in self._sarvam.stream_synthesize(text, detected_language):
                stream.queue.put_nowait(chunk)
                if stream.timings is not None:
                    stream.timings.mark("tts_first_audio")
                stream.ready.set()
        except AppError as exc:
            logger.warning("Streaming TTS failed: %s", exc.code)
            stream.error = exc
        finally:
            stream.done.set()
            asyncio.get_running_loop().call_later(
                TTS_STREAM_TTL_SECONDS, self.drop_tts_stream, token
            )

    def get_tts_stream(self, token: str) -> TtsStream | None:
        return self._tts_streams.get(token)

    def drop_tts_stream(self, token: str) -> None:
        self._tts_streams.pop(token, None)

    def mark_audio_outbound(self, token: str) -> None:
        """Record when the first response bytes leave for Twilio's <Play>."""
        stream = self._tts_streams.get(token)
        timings = stream.timings if stream is not None else self._hosted_timings.pop(
            token, None
        )
        if timings is None:
            return
        if "first_outbound_audio" in timings.phase_elapsed_ms:
            return
        timings.mark("first_outbound_audio")
        self._persist_timings(timings)
        timings.log(logger, event="first_outbound_audio")

    def _persist_timings(self, timings: TurnTimings) -> None:
        if not timings.session_id:
            return
        try:
            with session_scope(self._factory) as db:
                persist_turn_telemetry(db, timings.session_id, timings)
                if timings.assistant_message_id is not None:
                    message = db.get(Message, timings.assistant_message_id)
                    caller_latency = timings.caller_perceived()
                    if message is not None and caller_latency is not None:
                        message.total_turn_latency_ms = caller_latency
                        db.add(message)
        except Exception:
            logger.warning(
                "Could not persist turn timings (audio playback continues)",
                exc_info=True,
            )

    # ---------- entry points ----------

    async def greeting_twiml(self, call_sid: str | None = None) -> str:
        """TwiML for the answered call: play the greeting and listen.

        The opening line is generated by the LLM so the agent does not speak the
        same hard-coded sentence on every call, but generation + TTS is far too
        slow to run inside Twilio's webhook budget. The cached greeting is
        served instantly and a fresh one is generated in the background, so
        every call gets a new opening line while this webhook stays fast.

        When ``call_sid`` is provided, the spoken greeting is written into a
        new DB session so the first user turn does not re-introduce the agent.
        """
        lang = self._settings.default_language
        entry = self._greeting_cache.get(lang)
        if entry is not None:
            asyncio.create_task(self._refresh_greeting(lang))
            _, twiml, text = entry
            self._seed_greeting_session(call_sid, text, lang)
            return twiml
        if lang in self._greeting_refreshing:
            # Cold-start race only: a greeting is being generated, so serve the
            # static fallback instead of blocking the webhook on the LLM.
            logger.warning("Greeting still generating; serving static fallback.")
            text = fallback_text("greeting", lang, settings=self._settings)
            # Avoid Twilio <Say> on Indic script - use English for this rare path.
            say_text = (
                text if lang in {"en", "en-in", "en-IN"} else fallback_text("greeting", "en", settings=self._settings)
            )
            self._seed_greeting_session(call_sid, text, lang)
            return self._gather_twiml(None, say_text, language=lang)
        async with self._greeting_lock:
            entry = self._greeting_cache.get(lang)
            if entry is not None:
                _, twiml, text = entry
                self._seed_greeting_session(call_sid, text, lang)
                return twiml
            twiml = await self._build_and_cache_greeting(lang)
            entry = self._greeting_cache.get(lang)
            text = entry[2] if entry else fallback_text("greeting", lang, settings=self._settings)
            self._seed_greeting_session(call_sid, text, lang)
            return twiml

    def _seed_greeting_session(
        self, call_sid: str | None, greeting_text: str, language: str
    ) -> None:
        """Persist the played opening so the first LLM turn continues, not re-greets."""
        if not call_sid or not (greeting_text or "").strip():
            return
        record = self._registry.get(call_sid)
        if record is None:
            record = CallRecord(call_sid=call_sid, to_number="", from_number="")
            self._registry.add(record)
        if record.status not in {"in-progress", "ringing", "answered"}:
            self._registry.update(call_sid, status="in-progress")
        if record.session_id:
            return
        with session_scope(self._factory) as db:
            session = Session(
                language=language or self._settings.default_language,
                current_state="collecting_identity",
                status="active",
            )
            db.add(session)
            db.flush()
            db.add(
                Message(
                    session_id=session.id,
                    role="assistant",
                    content=greeting_text.strip(),
                )
            )
            profile = build_call_profile(
                self._settings,
                lead_overrides=record.lead_overrides,
                phone_number=record.to_number or None,
            )
            self._engine.set_call_profile(profile)
            lead = Lead(session_id=session.id)
            self._engine.apply_known_lead_fields(lead)
            if record.to_number and not lead.phone_number:
                lead.phone_number = record.to_number
            db.add(lead)
            self._registry.update(call_sid, session_id=session.id)
            logger.info(
                "Seeded greeting session %s for call %s", session.id, call_sid
            )

    async def warm_greeting(self) -> None:
        """Pre-build and cache the greeting TwiML (fire-and-forget at startup)."""
        await self._refresh_greeting(self._settings.default_language)

    async def _refresh_greeting(self, lang: str) -> None:
        if lang in self._greeting_refreshing:
            return
        self._greeting_refreshing.add(lang)
        try:
            async with self._greeting_lock:
                await self._build_and_cache_greeting(lang)
        finally:
            self._greeting_refreshing.discard(lang)

    async def _build_and_cache_greeting(self, lang: str) -> str:
        timings = TurnTimings(settings=self._settings)
        text = fallback_text("greeting", lang, settings=self._settings)
        try:
            greeting, _, _ = await self._engine.generate_greeting(timings, language=lang)
            if greeting and greeting.strip():
                text = greeting.strip()
        except AppError:
            logger.warning("Dynamic greeting generation failed; using fallback.")
        # File-hosted (not streaming) so the audio URL outlives the cache.
        url, _ = await self._host_tts(text, lang)
        # If TTS failed, fall back to English <Say> (Twilio cannot speak Gujarati).
        say_fallback = (
            text
            if (url is not None or lang in {"en", "en-in", "en-IN"})
            else fallback_text("greeting", "en", settings=self._settings)
        )
        content = self._gather_twiml(
            url, say_fallback if url is None else None, language=lang
        )
        self._greeting_cache[lang] = (time.monotonic(), content, text)
        return content

    def _drop_pending(self, call_sid: str) -> None:
        pending = self._pending_turns.pop(call_sid, None)
        if pending and pending.task and not pending.task.done():
            pending.task.cancel()

    def _sweep_pending(self) -> None:
        now = time.monotonic()
        for call_sid, pending in list(self._pending_turns.items()):
            if now - pending.created_at > PENDING_TURN_TTL_SECONDS:
                self._drop_pending(call_sid)

    async def process_webhook(self, call_sid: str, speech_result: str) -> str:
        """One <Gather> turn: process the transcript and return the next TwiML.

        Fast path (empty speech / quick LLM): return Play+Gather inline.
        Slow path: start background work and Redirect to /turn-result so Twilio's
        ~15s webhook budget is never exceeded.
        """
        text = (speech_result or "").strip()

        record = self._registry.get(call_sid)
        if record is None:
            record = CallRecord(call_sid=call_sid, to_number="", from_number="")
            self._registry.add(record)
        if record.status not in {"in-progress", "ringing", "answered"}:
            self._registry.update(call_sid, status="in-progress")

        # Empty SpeechResult is common while the caller is still starting to talk
        # (Gather timeout / noise / barge-in). Stay silent for the first miss so
        # we do not barge in; after N consecutive empties, ask if they are still
        # there / if we missed something.
        if not text:
            detected_language = self._settings.default_language
            with session_scope(self._factory) as db:
                session = db.get(Session, record.session_id) if record.session_id else None
                if session is None:
                    session = Session(language=self._settings.default_language)
                    db.add(session)
                    db.flush()
                    self._registry.update(call_sid, session_id=session.id)
                detected_language = session.language or self._settings.default_language
            empty_count = int(getattr(record, "empty_listen_count", 0) or 0) + 1
            self._registry.update(call_sid, empty_listen_count=empty_count)
            try:
                prompt_after = int(
                    getattr(self._settings, "gather_empty_prompt_after", 3) or 3
                )
            except (TypeError, ValueError):
                prompt_after = 3
            prompt_after = max(1, min(prompt_after, 5))

            if empty_count < prompt_after:
                logger.info(
                    "Empty SpeechResult for %s; silent re-listen #%s (lang=%s)",
                    call_sid,
                    empty_count,
                    detected_language,
                )
                trace(
                    "gather.listen.empty",
                    call_sid=call_sid,
                    session_id=record.session_id,
                    language=detected_language,
                    action="silent_relisten",
                    empty_listen_count=empty_count,
                )
                return self._gather_twiml(None, None, language=detected_language)

            prompt = fallback_text("still_there", detected_language)
            logger.info(
                "Empty SpeechResult for %s; still-there prompt #%s (lang=%s)",
                call_sid,
                empty_count,
                detected_language,
            )
            trace(
                "gather.listen.empty",
                call_sid=call_sid,
                session_id=record.session_id,
                language=detected_language,
                action="still_there_prompt",
                empty_listen_count=empty_count,
            )
            url, _ = await self._host_tts(prompt, detected_language)
            lang_key = (detected_language or "").strip().lower()
            if url is not None:
                return self._gather_twiml(url, None, language=detected_language)
            # Twilio <Say> cannot speak Indic script — fall back to English.
            say_text = (
                prompt
                if lang_key in {"en", "en-in", "en-IN"}
                else fallback_text("still_there", "en")
            )
            return self._gather_twiml(None, say_text, language=detected_language)

        if getattr(record, "empty_listen_count", 0):
            self._registry.update(call_sid, empty_listen_count=0)

        self._sweep_pending()
        self._drop_pending(call_sid)
        pending = PendingTurn(
            call_sid=call_sid,
            token=uuid.uuid4().hex,
            language=self._settings.default_language,
            timings=TurnTimings(
                settings=self._settings,
                transport="twilio_gather",
                stt_provider="twilio",
            ),
        )
        pending.timings.transcript_char_count = len(text)
        pending.timings.mark("transcript_received")
        # Seed language from the live session so the hold filler matches.
        if record.session_id:
            with session_scope(self._factory) as db:
                session = db.get(Session, record.session_id)
                if session and session.language:
                    pending.language = session.language
        trace(
            "gather.transcript.received",
            call_sid=call_sid,
            turn_id=pending.timings.turn_id if pending.timings else None,
            session_id=record.session_id,
            stt_provider="twilio",
            language=pending.language,
            transcript=text,
            transcript_chars=len(text),
            inline_budget_s=self._inline_budget_seconds(),
        )
        self._pending_turns[call_sid] = pending
        pending.task = asyncio.create_task(
            self._run_turn_job(pending, text), name=f"turn-{call_sid[-8:]}"
        )
        # Keep the Twilio webhook tiny. If the reply is ready almost immediately
        # we return it inline; otherwise Redirect before Twilio's ~15s cap.
        inline_budget = self._inline_budget_seconds()
        progress = await self._wait_for_pending_progress(pending, inline_budget)
        if progress == "timeout":
            content = self._pending_redirect_twiml(pending)
            if pending.timings is not None:
                pending.timings.log(logger, event="turn_redirected")
            logger.warning(
                "Turn for %s still running after %.1fs; deferring via Redirect",
                call_sid,
                inline_budget,
            )
            trace(
                "gather.turn.redirect",
                call_sid=call_sid,
                turn_id=pending.timings.turn_id if pending.timings else None,
                waited_ms=int(inline_budget * 1000),
                pause_s=self._poll_pause_seconds(),
            )
            return content

        early_twiml = self._take_early_twiml(pending)
        if early_twiml is not None:
            trace(
                "gather.turn.speak_early",
                call_sid=call_sid,
                turn_id=pending.timings.turn_id if pending.timings else None,
                text=pending.early_text,
            )
            return early_twiml

        self._pending_turns.pop(call_sid, None)
        if pending.timings is not None:
            pending.timings.log(logger)
        if pending.error is not None or not pending.twiml:
            lang = pending.language or self._settings.default_language
            repeat = fallback_text("repeat", "en")
            trace(
                "gather.turn.error_fallback",
                call_sid=call_sid,
                turn_id=pending.timings.turn_id if pending.timings else None,
                error=str(pending.error) if pending.error else "empty_twiml",
                fallback=repeat,
            )
            return self._say_gather_twiml(repeat, language=lang)
        trace(
            "gather.turn.inline_ok",
            call_sid=call_sid,
            turn_id=pending.timings.turn_id if pending.timings else None,
            language=pending.language,
            twiml_chars=len(pending.twiml or ""),
            total_elapsed_ms=pending.timings.total() if pending.timings else None,
        )
        return pending.twiml

    async def poll_pending_turn(self, call_sid: str, pending_token: str) -> str:
        """Serve deferred turn TwiML for a prior /turn Redirect."""
        pending = self._pending_turns.get(call_sid)
        if pending is None or pending.token != pending_token:
            logger.warning(
                "turn-result miss call_sid=%s token_ok=%s",
                call_sid,
                bool(pending and pending.token == pending_token),
            )
            lang = self._settings.default_language
            record = self._registry.get(call_sid)
            if record and record.session_id:
                with session_scope(self._factory) as db:
                    session = db.get(Session, record.session_id)
                    if session and session.language:
                        lang = session.language
            repeat = fallback_text("repeat", "en")
            return self._say_gather_twiml(repeat, language=lang)

        pending.polls += 1
        poll_started = time.monotonic()
        if pending.timings is not None:
            pending.timings.transport_retry_count = pending.polls
            pending.timings.mark("redirect_poll_received")
            redirect_elapsed = pending.timings.duration_between(
                "redirect_issued", "redirect_poll_received"
            )
            if redirect_elapsed is not None and "redirect_transit_delay" not in (
                pending.timings.phase_durations_ms
            ):
                pending.timings.add_duration(
                    "redirect_transit_delay", redirect_elapsed
                )
        logger.info(
            "turn-result poll=%s call_sid=%s done=%s",
            pending.polls,
            call_sid,
            pending.done.is_set(),
        )
        progress = await self._wait_for_pending_progress(
            pending, TURN_POLL_WAIT_SECONDS
        )
        if progress == "timeout":
            if pending.timings is not None:
                pending.timings.add_duration(
                    "redirect_poll_delay",
                    int((time.monotonic() - poll_started) * 1000),
                )
                pending.timings.log(logger, event="turn_poll_timeout")
            if pending.polls >= MAX_TURN_POLLS:
                logger.error(
                    "Turn for %s exceeded max polls; falling back", call_sid
                )
                if pending.timings is not None:
                    self._persist_timings(pending.timings)
                self._drop_pending(call_sid)
                lang = pending.language or self._settings.default_language
                repeat = fallback_text("repeat", "en")
                return self._say_gather_twiml(repeat, language=lang)
            return self._pending_redirect_twiml(pending)

        if pending.timings is not None:
            pending.timings.add_duration(
                "redirect_poll_delay",
                int((time.monotonic() - poll_started) * 1000),
            )
        early_twiml = self._take_early_twiml(pending)
        if early_twiml is not None:
            trace(
                "gather.turn.speak_early",
                call_sid=call_sid,
                turn_id=pending.timings.turn_id if pending.timings else None,
                text=pending.early_text,
                poll=pending.polls,
            )
            return early_twiml

        self._pending_turns.pop(call_sid, None)
        if pending.timings is not None:
            pending.timings.log(logger)
        if pending.error is not None or not pending.twiml:
            lang = pending.language or self._settings.default_language
            repeat = fallback_text("repeat", "en")
            return self._say_gather_twiml(repeat, language=lang)
        return pending.twiml

    async def _run_turn_job(self, pending: PendingTurn, text: str) -> None:
        """Background LLM+TTS for one Gather turn; fills pending.twiml."""
        # Yield so the webhook's wait_for timeout can be scheduled immediately.
        await asyncio.sleep(0)
        try:
            pending.twiml = await self._build_turn_twiml(pending.call_sid, text, pending)
        except Exception as exc:
            logger.exception("Turn job failed for %s: %s", pending.call_sid, exc)
            pending.error = exc
        finally:
            pending.done.set()

    async def _build_turn_twiml(
        self, call_sid: str, text: str, pending: PendingTurn | None = None
    ) -> str:
        """Run conversation + TTS and return the next TwiML document."""
        record = self._registry.get(call_sid)
        if record is None:
            record = CallRecord(call_sid=call_sid, to_number="", from_number="")
            self._registry.add(record)

        reply = ""
        detected_language = self._settings.default_language
        listen_language = self._settings.default_language
        completed = False
        abandoned = False
        reply_url: str | None = None
        remaining_reply = ""

        with session_scope(self._factory) as db:
            session = db.get(Session, record.session_id) if record.session_id else None
            if session is None:
                session = Session(language=self._settings.default_language)
                db.add(session)
                db.flush()
                self._registry.update(call_sid, session_id=session.id)
            detected_language = session.language or self._settings.default_language
            if pending is not None:
                pending.language = detected_language

            timings = (
                pending.timings
                if pending is not None and pending.timings is not None
                else TurnTimings(
                    settings=self._settings,
                    transport="twilio_gather",
                    stt_provider="twilio",
                )
            )
            timings.session_id = session.id
            timings.transcript_char_count = len(text)
            timings.mark("transcript_received")
            first_chunk = FirstSpeechChunkBuffer()
            prior_lang = (session.language or "").strip().lower()
            expected_lang = resolve_turn_language(
                text, prior_language=prior_lang
            )
            switching = bool(detect_explicit_language_switch(text)) or bool(
                expected_lang
                and prior_lang
                and expected_lang != prior_lang
            )
            use_speak_early = bool(
                pending is not None
                and self._settings.llm_streaming_enabled
                and self._settings.sarvam_tts_streaming
                and not switching
            )

            async def speak_early(fragment: str) -> None:
                if pending is None or pending.early_text:
                    return
                sentence = first_chunk.feed(fragment)
                if not sentence:
                    return
                tts_language = (
                    infer_script_language(sentence) or detected_language
                )
                trace(
                    "gather.tts.early_start",
                    call_sid=call_sid,
                    turn_id=timings.turn_id,
                    language=tts_language,
                    text=sentence,
                )
                audio_url, tts_ms = await self._stream_tts(
                    sentence, tts_language, timings
                )
                if not audio_url:
                    return
                pending.early_text = sentence
                pending.early_twiml = self._early_reply_twiml(
                    pending, audio_url
                )
                pending.early_ready.set()
                trace(
                    "gather.tts.early_ready",
                    call_sid=call_sid,
                    turn_id=timings.turn_id,
                    tts_ms=tts_ms,
                    audio_url=audio_url,
                )

            process_kwargs = {}
            if use_speak_early:
                process_kwargs["on_assistant_chunk"] = speak_early
            _, parsed = await self._engine.process_turn(
                db, session, text, timings, **process_kwargs
            )
            reply = parsed.assistant_message or ""
            remaining_reply = reply
            if pending is not None and pending.early_text:
                prefix_at = reply.find(
                    pending.early_text, 0, len(pending.early_text) + 8
                )
                if prefix_at >= 0:
                    remaining_reply = reply[
                        prefix_at + len(pending.early_text) :
                    ].lstrip()
                else:
                    logger.warning(
                        "Early spoken text did not match completed reply; "
                        "playing the validated reply in full."
                    )
            timings.response_char_count = len(reply)
            listen_language = (
                session.language
                or getattr(parsed, "detected_language", None)
                or detected_language
            )
            tts_language = (
                infer_script_language(reply)
                or getattr(parsed, "detected_language", None)
                or listen_language
            )
            detected_language = tts_language
            if pending is not None:
                pending.language = listen_language
            completed = session.status == "completed"
            abandoned = session.status == "abandoned"
            trace(
                "gather.llm.done",
                call_sid=call_sid,
                turn_id=timings.turn_id,
                session_id=session.id,
                state=session.current_state,
                language=listen_language,
                reply=reply,
                llm_latency_ms=timings.llm_latency_ms,
                completed=completed,
                abandoned=abandoned,
            )

            if not completed and not abandoned:
                tts_ms = 0
                if remaining_reply:
                    trace(
                        "gather.tts.start",
                        call_sid=call_sid,
                        turn_id=timings.turn_id,
                        language=detected_language,
                        text=remaining_reply,
                    )
                    reply_url, tts_ms = await self._stream_tts(
                        remaining_reply,
                        detected_language,
                        None if pending and pending.early_text else timings,
                    )
                    trace(
                        "gather.tts.done",
                        call_sid=call_sid,
                        turn_id=timings.turn_id,
                        tts_ms=tts_ms,
                        audio_url=reply_url,
                        mode=timings.tts_mode,
                    )
                if reply_url or (pending is not None and pending.early_text):
                    if not pending or not pending.early_text:
                        timings.tts_latency_ms = tts_ms
                    last_assistant = (
                        db.query(Message)
                        .filter(
                            Message.session_id == session.id,
                            Message.role == "assistant",
                        )
                        .order_by(Message.id.desc())
                        .first()
                    )
                    if last_assistant is not None:
                        timings.assistant_message_id = last_assistant.id
                        last_assistant.tts_latency_ms = timings.tts_latency_ms
                        last_assistant.total_turn_latency_ms = timings.total()
                        last_assistant.estimated_provider_cost = (
                            (last_assistant.estimated_provider_cost or 0.0)
                            + timings.estimated_tts_cost()
                        )
                        db.add(last_assistant)
                persist_turn_telemetry(db, session.id, timings)

        if completed:
            final_url, _ = (
                await self._stream_tts(
                    remaining_reply,
                    detected_language,
                    None if pending and pending.early_text else timings,
                )
                if remaining_reply
                else (None, 0)
            )
            goodbye = fallback_text("goodbye", detected_language)
            goodbye_url, _ = await self._host_tts(goodbye, detected_language)
            self._persist_timings(timings)
            return self._final_twiml(
                final_url, goodbye_url, remaining_reply, goodbye
            )

        if abandoned:
            farewell = remaining_reply
            if not farewell and not (pending and pending.early_text):
                farewell = fallback_text("goodbye", detected_language)
            farewell_url, _ = (
                await self._stream_tts(
                    farewell,
                    detected_language,
                    None if pending and pending.early_text else timings,
                )
                if farewell
                else (None, 0)
            )
            self._persist_timings(timings)
            return self._farewell_twiml(farewell_url, farewell)

        return self._gather_twiml(
            reply_url,
            remaining_reply if reply_url is None else None,
            language=listen_language,
        )
