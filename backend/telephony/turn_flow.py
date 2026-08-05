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
from backend.metrics import TurnTimings
from backend.models import Message, Session
from backend.providers.sarvam_client import SarvamClient
from backend.telephony.call_manager import (
    GREETING,
    GOODBYE,
    REPEAT_MESSAGE,
    CallRecord,
    CallRegistry,
)
from backend.telephony.twilio_service import TwilioService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

TTS_DIR_NAME = "tts"
MAX_HOSTED_FILES = 50
HOSTED_FILE_TTL_SECONDS = 2 * 3600

# Streaming TTS: how long we wait for the first audio chunk before falling back
# to the buffered endpoint, and how long a finished stream stays consumable.
TTS_FIRST_CHUNK_TIMEOUT_SECONDS = 3.0
TTS_STREAM_TTL_SECONDS = 120

# default_language -> <Gather language> attribute for Twilio Speech Recognition
GATHER_LANGUAGES = {
    "en": "en-IN",
    "en-in": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hinglish": "hi-IN",
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

    # ---------- TwiML builders ----------

    def _gather_attrs(self, turn_url: str) -> str:
        lang = _gather_language(self._settings.default_language)
        return (
            'input="speech" action="{url}" method="POST" speechTimeout="auto" '
            'timeout="4" language="{lang}" actionOnEmptyResult="true"'
        ).format(url=_escape(turn_url), lang=lang)

    def _gather_twiml(
        self, prompt_url: str | None, prompt_text: str | None = None
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
            f'<Response><Gather {self._gather_attrs(turn_url)}>{inner}</Gather></Response>'
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
        self, text: str, detected_language: str | None
    ) -> tuple[str | None, int]:
        """Synthesize with Sarvam and return a public URL for <Play>."""
        try:
            audio, _, tts_ms = await self._sarvam.synthesize(text, detected_language)
        except AppError as exc:
            logger.warning("Turn TTS failed: %s", exc.code)
            return None, 0
        if not audio:
            return None, 0
        self._sweep_tts_dir()
        file_id = uuid.uuid4().hex
        path = self._tts_dir() / f"{file_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return self._twilio.audio_url(file_id), tts_ms

    async def _stream_tts(
        self, text: str, detected_language: str | None
    ) -> tuple[str | None, int]:
        """Start streaming TTS and return a public <Play> URL for it.

        Returns once the first audio chunk has arrived (so the /audio route can
        already serve data), then the stream keeps filling in the background.
        Returns (None, 0) when streaming is disabled or produces no audio in
        time, so callers can fall back to the buffered file path.
        """
        if not self._settings.sarvam_tts_streaming:
            return await self._host_tts(text, detected_language)
        token = uuid.uuid4().hex
        stream = TtsStream()
        self._tts_streams[token] = stream
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
            return await self._host_tts(text, detected_language)
        finally:
            ready_task.cancel()
            done_task.cancel()
        if not stream.ready.is_set():
            task.cancel()
            self._tts_streams.pop(token, None)
            return await self._host_tts(text, detected_language)
        first_chunk_ms = int((time.monotonic() - started) * 1000)
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

    # ---------- entry points ----------

    async def greeting_twiml(self) -> str:
        """TwiML for the answered call: play the greeting and listen.

        The opening line is generated by the LLM so the agent does not speak the
        same hard-coded sentence on every call. GREETING is kept as a fallback if
        generation or TTS fails.
        """
        lang = self._settings.default_language
        timings = TurnTimings(settings=self._settings)
        text = GREETING
        try:
            greeting, _, _ = await self._engine.generate_greeting(timings)
            if greeting and greeting.strip():
                text = greeting.strip()
        except AppError:
            logger.warning("Dynamic greeting generation failed; using fallback.")
        url, _ = await self._stream_tts(text, lang)
        return self._gather_twiml(url, text if url is None else None)

    async def process_webhook(self, call_sid: str, speech_result: str) -> str:
        """One <Gather> turn: process the transcript and return the next TwiML."""
        text = (speech_result or "").strip()

        record = self._registry.get(call_sid)
        if record is None:
            record = CallRecord(call_sid=call_sid, to_number="", from_number="")
            self._registry.add(record)
        if record.status not in {"in-progress", "ringing", "answered"}:
            self._registry.update(call_sid, status="in-progress")

        reply = ""
        detected_language = self._settings.default_language
        completed = False
        reply_url: str | None = None

        with session_scope(self._factory) as db:
            session = db.get(Session, record.session_id) if record.session_id else None
            if session is None:
                session = Session(language=self._settings.default_language)
                db.add(session)
                db.flush()
                self._registry.update(call_sid, session_id=session.id)
            detected_language = session.language or self._settings.default_language

            if not text:
                prompt_url, _ = await self._host_tts(REPEAT_MESSAGE, detected_language)
                return self._gather_twiml(
                    prompt_url, REPEAT_MESSAGE if prompt_url is None else None
                )

            timings = TurnTimings(settings=self._settings)
            timings.transcript_char_count = len(text)
            _, parsed = await self._engine.process_turn(db, session, text, timings)
            reply = parsed.assistant_message or ""
            timings.response_char_count = len(reply)
            if getattr(parsed, "detected_language", None):
                detected_language = parsed.detected_language
            completed = session.status == "completed"

            if not completed:
                reply_url, tts_ms = await self._stream_tts(reply, detected_language)
                if reply_url:
                    timings.tts_attempted = True
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
                        last_assistant.tts_latency_ms = timings.tts_latency_ms
                        last_assistant.total_turn_latency_ms = timings.total()
                        last_assistant.estimated_provider_cost = (
                            (last_assistant.estimated_provider_cost or 0.0)
                            + timings.estimated_tts_cost()
                        )
                        db.add(last_assistant)

        if completed:
            final_url, _ = await self._stream_tts(reply, detected_language) if reply else (None, 0)
            goodbye_url, _ = await self._host_tts(GOODBYE, detected_language)
            return self._final_twiml(
                final_url, goodbye_url, reply, GOODBYE
            )

        return self._gather_twiml(reply_url, reply if reply_url is None else None)
