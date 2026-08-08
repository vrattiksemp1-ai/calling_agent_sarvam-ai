"""Concurrent real-time call sessions for bidirectional carrier streams.

A CallSession owns one phone call:
  * receives provider-normalized 8 kHz PCM from a transport adapter,
  * detects speech/silence (simple energy VAD),
  * sends each utterance through STT -> LLM -> TTS,
  * paces 20 ms reply frames back over the socket,
  * hangs up when the lead conversation completes.
"""

from __future__ import annotations

import asyncio
import struct
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from backend.config import Settings, tts_language_code_for
from backend.conversation import ConversationEngine
from backend.database import session_scope
from backend.errors import AppError
from backend.metrics import TurnTimings, persist_turn_telemetry
from backend.models import Message, Session
from backend.language_utils import map_stt_language_code
from backend.providers.sarvam_client import SarvamClient
from backend.streaming_json import StreamingTextChunker
from backend.telephony import mulaw
from backend.telephony.endpointing import SemanticEndpointing
from backend.telephony.transport import (
    CORE_FRAME_BYTES,
    FRAME_DURATION_SECONDS,
    AudioReceived,
    CallTransport,
    ControlEvent,
    PlaybackMarked,
    StreamStarted,
    StreamStopped,
    TwilioMediaStreamTransport,
)
from backend.telephony.twilio_service import TwilioService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Emergency-only lines when LLM/STT fails. Normal conversation wording is
# always LLM-generated from the call profile. Each key has multiple variants.
FALLBACK_MESSAGES = {'repeat': {'en': ["Sorry, I didn't catch that. Could you say that again?",
                   'I missed that — mind repeating in a few words?',
                   'Could you please repeat that?'],
            'hi': ['माफ़ कीजिए, मैं समझ नहीं पाया. क्या आप दोबारा कह सकते हैं?',
                   'सुनाई नहीं दिया — थोड़े शब्दों में दोबारा बोलेंगे?',
                   'एक बार फिर कह सकते हैं क्या?'],
            'gu': ['સોરી, સમજાયું નહીં. ફરી એક વાર કહો તો?',
                   'સંભળાયું નહીં — ટૂંકમાં ફરી કહો?',
                   'ફરી એક વાર કહી શકો?']},
 'goodbye': {'en': ['Thank you for your time. Your details are saved. Goodbye!',
                    "Thanks for talking — I've noted your details. Bye for now!",
                    "Appreciate your time. We'll take it from here. Goodbye!"],
             'hi': ['समय देने के लिए धन्यवाद. आपकी जानकारी सुरक्षित हो गई है. नमस्ते!',
                    'बात करने के लिए शुक्रिया. डिटेल्स सेव हो गई हैं. अलविदा!',
                    'धन्यवाद. हम आगे बढ़ते हैं. नमस्ते!'],
             'gu': ['ટાઈમ આપ્યો તેનો આભાર. તમારી વિગતો સેવ થઈ ગઈ છે. આવજો!',
                    'વાત કરવા બદલ આભાર. ડિટેલ્સ નોંધી લીધી. આવજો!',
                    'ટાઈમ આપ્યો એટલે આભાર. અમે આગળ વધીશું. આવજો!']},
 'hold': {'en': ['One moment.', 'Just a second.', 'Hang on a moment.'],
          'hi': ['एक सेकंड।', 'ज़रा रुकिए।', 'एक पल।'],
          'gu': ['એક સેકન્ડ.', 'જરા રોકો.', 'એક પળ.']}}




def fallback_text(
    name: str,
    language: str | None,
    *,
    settings=None,
    call_profile=None,
) -> str:
    """Emergency fallback only — random variant, never the primary script."""
    import random

    lang = (language or "en").strip().lower()
    if name == "greeting":
        if call_profile is not None:
            return call_profile.opening_greeting(lang)
        if settings is not None:
            from backend.call_profile import build_call_profile

            return build_call_profile(settings).opening_greeting(lang)
        from backend.call_profile import CallProfile

        return CallProfile().opening_greeting(lang)
    bucket = (
        "gu"
        if lang in {"gu", "gu-in", "gujarati", "gujlish"}
        else "hi"
        if lang in {"hi", "hi-in", "hinglish"}
        else "en"
    )
    options = FALLBACK_MESSAGES[name][bucket]
    if isinstance(options, list):
        return random.choice(options)
    return options

# VAD tuning (all in 20 ms mu-law chunks = 160 samples each)
SPEECH_ENERGY = 500
MIN_UTTERANCE_CHUNKS = 8
SILENCE_CHUNKS = 30
MAX_UTTERANCE_CHUNKS = 750
PLAYBACK_MARK_TIMEOUT_SECONDS = 15.0


@dataclass
class CallRecord:
    call_sid: str
    to_number: str
    from_number: str
    provider: str = "twilio"
    status: str = "initiated"
    error: str | None = None
    session_id: str | None = None
    lead_overrides: dict | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


ACTIVE_CALL_STATUSES = {
    "queued",
    "initiated",
    "ringing",
    "answered",
    "in-progress",
}


class CallRegistry:
    """In-memory registry of active/recent calls (single-user MVP grade)."""

    def __init__(self) -> None:
        self._calls: dict[str, CallRecord] = {}

    def add(self, record: CallRecord) -> None:
        self._calls[record.call_sid] = record

    def update(self, call_sid: str, **changes) -> None:
        record = self._calls.get(call_sid)
        if record is None:
            return
        for key, value in changes.items():
            setattr(record, key, value)
        record.updated_at = time.time()

    def get(self, call_sid: str) -> CallRecord | None:
        return self._calls.get(call_sid)

    def remove(self, call_sid: str) -> None:
        self._calls.pop(call_sid, None)

    def active_calls(self) -> list[CallRecord]:
        """Calls that are still ringing or connected (not terminal)."""
        return [
            record
            for record in self._calls.values()
            if (record.status or "").lower() in ACTIVE_CALL_STATUSES
        ]


class CallSession:
    """Concurrent provider-neutral real-time conversation session.

    The receive loop only parses provider events and feeds VAD. Greeting,
    STT/LLM/TTS and paced playback run in a cancellable task, which lets caller
    speech interrupt generation as well as already-buffered provider audio.
    """

    def __init__(
        self,
        settings: Settings,
        session_factory,
        engine: ConversationEngine,
        sarvam: SarvamClient,
        twilio: TwilioService | None,
        registry: CallRegistry,
        ws: WebSocket | None = None,
        *,
        transport: CallTransport | None = None,
        call_service=None,
    ) -> None:
        self._settings = settings
        self._factory = session_factory
        self._engine = engine
        self._sarvam = sarvam
        self._call_service = call_service or twilio
        self._registry = registry
        self._ws = ws
        if transport is None:
            if ws is None:
                raise ValueError("CallSession requires a transport or WebSocket")
            transport = TwilioMediaStreamTransport(ws)
        self._transport = transport

        self._stream_sid: str | None = None
        self._call_sid: str | None = None
        self._session_id: str | None = None
        # Retained for compatibility/diagnostics; it no longer gates inbound
        # audio. Provider receive stays active while this is true.
        self._busy = False
        self._playing = False
        self._completed = False
        self._utterance = bytearray()
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._ready_utterance: bytes | None = None
        self._response_task: asyncio.Task | None = None
        self._pending_marks: dict[str, TurnTimings | None] = {}
        self._pending_mark_events: dict[str, asyncio.Event] = {}
        self._playback_mark_name: str | None = None
        self._playback_timings: TurnTimings | None = None
        self._realtime_stt = None
        self._stt_receive_task: asyncio.Task | None = None
        self._stt_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=250)
        self._realtime_timings: TurnTimings | None = None
        self._active_tts_session = None
        self._tts_connection_count = 0
        self._semantic_endpointing = SemanticEndpointing(
            settings.sarvam_realtime_stt_silence_ms,
            settings.sarvam_semantic_fast_silence_ms,
            settings.sarvam_semantic_slow_silence_ms,
        )
        self._semantic_silence_ms = settings.sarvam_realtime_stt_silence_ms

    # ---------- media streaming ----------

    async def _stream_wav(
        self, wav_bytes: bytes, timings: TurnTimings | None = None
    ) -> None:
        """Convert TTS WAV to core PCM and send one frame every 20 ms."""

        mulaw_bytes = mulaw.tts_wav_to_mulaw(wav_bytes)
        if not mulaw_bytes:
            return
        pcm16 = mulaw.decode_mulaw(mulaw_bytes)
        self._playing = True
        self._playback_mark_name = None
        self._playback_timings = timings
        first_chunk_sent = False
        loop = asyncio.get_running_loop()
        last_send_at: float | None = None
        for frame_index, offset in enumerate(range(0, len(pcm16), CORE_FRAME_BYTES)):
            if last_send_at is not None:
                await asyncio.sleep(
                    max(
                        0.0,
                        FRAME_DURATION_SECONDS - (loop.time() - last_send_at),
                    )
                )
            if not self._playing:
                return
            frame = pcm16[offset : offset + CORE_FRAME_BYTES]
            if len(frame) < CORE_FRAME_BYTES:
                frame += b"\x00" * (CORE_FRAME_BYTES - len(frame))
            audio_emitted = await self._transport.send_audio(frame)
            last_send_at = loop.time()
            if timings is not None and audio_emitted and not first_chunk_sent:
                timings.mark("first_outbound_audio")
                self._persist_timings(timings)
                timings.log(logger, event="first_outbound_audio")
                first_chunk_sent = True
        if self._playing:
            audio_emitted = await self._transport.flush_audio()
            if timings is not None and audio_emitted and not first_chunk_sent:
                timings.mark("first_outbound_audio")
                self._persist_timings(timings)
                timings.log(logger, event="first_outbound_audio")
            mark_name = f"reply_done:{uuid.uuid4().hex[:12]}"
            self._pending_marks[mark_name] = timings
            self._playback_mark_name = mark_name
            await self._transport.send_mark(mark_name)

    async def _speak(
        self, text: str, timings: TurnTimings | None = None
    ) -> None:
        try:
            audio, _, tts_ms = await self._sarvam.synthesize(text, None)
        except AppError as exc:
            logger.warning("Call TTS failed: %s", exc.code)
            return
        if timings is not None:
            timings.retry_count += max(
                0, getattr(self._sarvam, "last_attempt_count", 1) - 1
            )
            timings.tts_provider = "sarvam"
            timings.tts_mode = "buffered"
            timings.tts_attempted = True
            timings.tts_latency_ms = tts_ms
            timings.mark("tts_first_audio")
        await self._stream_wav(audio, timings)

    async def _stream_tts_socket(self, tts_session, timings: TurnTimings) -> None:
        """Pace provider-native 8 kHz chunks directly to the call transport."""
        buffer = bytearray()
        frame_index = 0
        first_audio = False
        first_outbound = False
        request_started = asyncio.get_running_loop().time()
        last_send_at: float | None = None
        self._playing = True
        self._playback_timings = timings
        while True:
            event = await tts_session.receive()
            if event.type == "error":
                raise RuntimeError(f"Sarvam realtime TTS error: {event.payload}")
            if event.type == "complete":
                break
            if event.type != "audio" or not event.audio:
                continue
            if not first_audio:
                timings.mark("tts_first_audio")
                timings.tts_latency_ms = int(
                    (asyncio.get_running_loop().time() - request_started) * 1000
                )
                first_audio = True
            audio = event.audio
            if tts_session.codec == "mulaw":
                audio = mulaw.decode_mulaw(audio)
            buffer.extend(audio)
            while len(buffer) >= CORE_FRAME_BYTES:
                if last_send_at is not None:
                    await asyncio.sleep(
                        max(
                            0.0,
                            FRAME_DURATION_SECONDS
                            - (asyncio.get_running_loop().time() - last_send_at),
                        )
                    )
                if not self._playing:
                    return
                frame = bytes(buffer[:CORE_FRAME_BYTES])
                del buffer[:CORE_FRAME_BYTES]
                emitted = await self._transport.send_audio(frame)
                last_send_at = asyncio.get_running_loop().time()
                if emitted and not first_outbound:
                    timings.mark("first_outbound_audio")
                    timings.log(logger, event="first_outbound_audio")
                    first_outbound = True
                frame_index += 1
        if buffer and self._playing:
            frame = bytes(buffer) + b"\x00" * (CORE_FRAME_BYTES - len(buffer))
            emitted = await self._transport.send_audio(frame)
            if emitted and not first_outbound:
                timings.mark("first_outbound_audio")
                timings.log(logger, event="first_outbound_audio")
        if self._playing:
            emitted = await self._transport.flush_audio()
            if emitted and not first_outbound:
                timings.mark("first_outbound_audio")
                timings.log(logger, event="first_outbound_audio")
            mark_name = f"reply_done:{uuid.uuid4().hex[:12]}"
            self._pending_marks[mark_name] = timings
            self._playback_mark_name = mark_name
            played = asyncio.Event()
            self._pending_mark_events[mark_name] = played
            await self._transport.send_mark(mark_name)
            try:
                await asyncio.wait_for(
                    played.wait(), timeout=PLAYBACK_MARK_TIMEOUT_SECONDS
                )
            finally:
                self._pending_mark_events.pop(mark_name, None)

    def _persist_timings(self, timings: TurnTimings) -> None:
        if not timings.session_id:
            return
        with session_scope(self._factory) as db:
            persist_turn_telemetry(db, timings.session_id, timings)
            if timings.assistant_message_id is not None:
                message = db.get(Message, timings.assistant_message_id)
                caller_latency = timings.caller_perceived()
                if message is not None and caller_latency is not None:
                    message.total_turn_latency_ms = caller_latency
                    db.add(message)

    # ---------- turn processing ----------

    async def _handle_utterance(self, pcm16: bytes | None = None) -> None:
        timings = TurnTimings(
            settings=self._settings,
            transport=self._transport.name,
            stt_provider="sarvam",
        )
        timings.mark("utterance_end")
        if pcm16 is None:
            pcm16 = self._ready_utterance or bytes(self._utterance)
            self._ready_utterance = None
            self._utterance.clear()
            self._speech_chunks = 0
            self._silence_chunks = 0
        if not pcm16:
            return

        duration_ms = int(len(pcm16) / 2 / mulaw.SAMPLE_RATE * 1000)
        wav = mulaw.pcm16_to_wav(pcm16, mulaw.SAMPLE_RATE)
        tmp = self._settings.resolved_temp_dir / f"call_{uuid.uuid4().hex}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(wav)
        try:
            transcript, stt_ms, stt_language = await self._sarvam.transcribe(
                str(tmp), duration_ms
            )
        except AppError as exc:
            logger.warning("Call STT failed: %s", exc.code)
            await self._speak(fallback_text("repeat", self._settings.default_language))
            return
        finally:
            tmp.unlink(missing_ok=True)

        timings.stt_latency_ms = stt_ms
        timings.retry_count += max(
            0, getattr(self._sarvam, "last_attempt_count", 1) - 1
        )
        timings.stt_language = stt_language
        timings.transcript_char_count = len(transcript)
        timings.audio_duration_ms = duration_ms
        timings.mark("transcript_received")
        await self._handle_transcript(transcript, stt_language, timings)

    async def _handle_transcript(
        self,
        transcript: str,
        stt_language: str | None,
        timings: TurnTimings,
    ) -> None:
        """Generate, validate, apply, and speak one final transcript.

        The database transaction remains open until streaming playback finishes.
        A barge-in cancellation therefore rolls back extraction/state and avoids
        recording an interrupted assistant turn as completed.
        """
        reply = ""
        completed = False
        abandoned = False
        tts_bytes: bytes | None = None
        detected_language = self._settings.default_language
        streamed_audio = False
        keep_tts_session = False
        try:
            with session_scope(self._factory) as db:
                session = db.get(Session, self._session_id)
                if session is None:
                    return
                timings.session_id = session.id
                use_streaming = (
                    self._settings.llm_streaming_enabled
                    and self._settings.sarvam_realtime_tts_enabled
                )
                tts_session = None
                tts_consumer = None
                chunker = StreamingTextChunker()
                if use_streaming:
                    try:
                        target_language = tts_language_code_for(
                            stt_language or session.language,
                            self._settings.sarvam_tts_language_code,
                        )
                        if (
                            self._active_tts_session is not None
                            and self._active_tts_session.language_code
                            != target_language
                        ):
                            await self._active_tts_session.close()
                            self._active_tts_session = None
                        tts_session = self._active_tts_session
                        if tts_session is None:
                            tts_session = await self._sarvam.open_realtime_tts(
                                stt_language or session.language
                            )
                            self._active_tts_session = tts_session
                            if self._tts_connection_count:
                                timings.mark("tts_reconnect")
                                timings.log(logger, event="tts_reconnect")
                            self._tts_connection_count += 1
                        timings.tts_provider = "sarvam"
                        timings.tts_mode = "websocket"
                        timings.tts_attempted = True
                        tts_consumer = asyncio.create_task(
                            self._stream_tts_socket(tts_session, timings)
                        )
                    except Exception:
                        timings.fallback_count += 1
                        timings.tts_mode = "websocket_to_buffered"
                        logger.exception("Realtime TTS connect failed; using buffered TTS")
                        tts_session = None

                async def speak_chunk(text: str) -> None:
                    if tts_session is None:
                        return
                    for chunk in chunker.feed(text):
                        await tts_session.send_text(chunk)

                _, parsed = await self._engine.process_turn(
                    db,
                    session,
                    transcript,
                    timings,
                    stt_language=stt_language,
                    on_assistant_chunk=speak_chunk if tts_session else None,
                )
                reply = parsed.assistant_message
                detected_language = session.language or self._settings.default_language
                timings.response_char_count = len(reply)

                if tts_session is not None:
                    remainder = chunker.flush()
                    if remainder:
                        await tts_session.send_text(remainder)
                    await tts_session.flush()
                    await tts_consumer
                    streamed_audio = True
                    keep_tts_session = True
                elif reply:
                    try:
                        tts_bytes, _, tts_ms = await self._sarvam.synthesize(
                            reply, parsed.detected_language
                        )
                        timings.tts_latency_ms = tts_ms
                        timings.retry_count += max(
                            0, getattr(self._sarvam, "last_attempt_count", 1) - 1
                        )
                        timings.tts_attempted = True
                        timings.tts_provider = "sarvam"
                        if timings.tts_mode != "websocket_to_buffered":
                            timings.tts_mode = "buffered"
                        timings.mark("tts_first_audio")
                    except AppError as exc:
                        logger.warning("Call TTS failed: %s", exc.code)
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
                        + timings.estimated_stt_cost()
                        + timings.estimated_tts_cost()
                    )
                    db.add(last_assistant)
                persist_turn_telemetry(db, session.id, timings)
                completed = session.status == "completed"
                abandoned = session.status == "abandoned"
        finally:
            if not keep_tts_session:
                active, self._active_tts_session = self._active_tts_session, None
                if active is not None:
                    await active.close()

        if tts_bytes and not streamed_audio:
            await self._stream_wav(tts_bytes, timings)
        elif reply and not streamed_audio and not abandoned:
            timings.fallback_count += 1
            await self._speak(fallback_text("repeat", detected_language), timings)

        if completed:
            await self._speak(fallback_text("goodbye", detected_language))
            self._completed = True
        elif abandoned:
            if not tts_bytes:
                await self._speak(fallback_text("goodbye", detected_language))
            self._completed = True

    # ---------- VAD ----------

    @staticmethod
    def _energy(pcm16: bytes) -> int:
        if not pcm16:
            return 0
        count = len(pcm16) // 2
        values = struct.unpack(f"<{count}h", pcm16)
        return max(abs(v) for v in values)

    async def _feed(self, payload_b64: str) -> bool:
        """Compatibility helper for tests and Twilio-formatted audio."""

        ready = await self._feed_pcm(mulaw.base64_to_mulaw(payload_b64))
        if ready is not None:
            self._ready_utterance = ready
            return True
        return False

    async def _feed_pcm(self, pcm16: bytes) -> bytes | None:
        """Feed one 20 ms core PCM frame and return a completed utterance.

        Speech cancels both in-flight generation and playback. Unlike the old
        ``_busy`` gate, every inbound frame is inspected while provider calls
        are running.
        """
        energy = self._energy(pcm16)
        response_active = (
            self._response_task is not None and not self._response_task.done()
        )
        if (
            energy > SPEECH_ENERGY
            and not self._utterance
            and (self._playing or response_active)
        ):
            await self._cancel_response(clear_provider=True)

        if energy > SPEECH_ENERGY:
            self._utterance.extend(pcm16)
            self._speech_chunks += 1
            self._silence_chunks = 0
        elif self._utterance:
            self._silence_chunks += 1
            if self._speech_chunks < MIN_UTTERANCE_CHUNKS and self._silence_chunks >= 90:
                self._utterance.clear()
                self._speech_chunks = 0
                self._silence_chunks = 0
        if self._speech_chunks >= MIN_UTTERANCE_CHUNKS and (
            self._silence_chunks >= SILENCE_CHUNKS
            or self._speech_chunks >= MAX_UTTERANCE_CHUNKS
        ):
            self._busy = True
            ready = bytes(self._utterance)
            self._utterance.clear()
            self._speech_chunks = 0
            self._silence_chunks = 0
            return ready
        return None

    async def _send_realtime_stt_audio(self, stt_session) -> None:
        while True:
            await stt_session.send_audio(await self._stt_audio_queue.get())

    async def _run_realtime_stt(self) -> None:
        """Keep one Saaras session alive and turn final events into turns."""
        sender = None
        try:
            stt_session = await self._sarvam.open_realtime_stt()
            self._realtime_stt = stt_session
            logger.info("realtime_stt_connected call_id=%s", self._call_sid)
            sender = asyncio.create_task(self._send_realtime_stt_audio(stt_session))
            async for event in stt_session:
                if event.type == "session.begin":
                    logger.info("realtime_stt_session_begin call_id=%s", self._call_sid)
                elif event.type == "vad.speech_start":
                    timings = TurnTimings(
                        settings=self._settings,
                        transport=self._transport.name,
                        stt_provider="sarvam_realtime",
                    )
                    timings.session_id = self._session_id
                    timings.mark("vad_speech_start")
                    self._realtime_timings = timings
                    await self._cancel_response(clear_provider=True)
                    timings.log(logger, event="vad_speech_start")
                elif event.type == "vad.speech_end":
                    timings = self._realtime_timings
                    if timings is not None:
                        timings.mark("vad_speech_end")
                        timings.mark("utterance_end")
                        timings.log(logger, event="vad_speech_end")
                elif event.type == "transcript.partial":
                    timings = self._realtime_timings
                    if timings is not None and event.transcript:
                        timings.mark("stt_first_partial")
                        if self._settings.sarvam_semantic_endpointing_enabled:
                            silence_ms = self._semantic_endpointing.recommend(
                                event.transcript
                            )
                            if silence_ms != self._semantic_silence_ms:
                                await stt_session.update_config(
                                    silence_ms=silence_ms
                                )
                                self._semantic_silence_ms = silence_ms
                                timings.mark("semantic_endpoint_adjustment")
                        timings.log(logger, event="stt_first_partial")
                elif event.type == "transcript.final" and event.transcript.strip():
                    timings = self._realtime_timings or TurnTimings(
                        settings=self._settings,
                        transport=self._transport.name,
                        stt_provider="sarvam_realtime",
                    )
                    timings.session_id = self._session_id
                    timings.mark("stt_first_final")
                    timings.mark("transcript_received")
                    timings.transcript_char_count = len(event.transcript.strip())
                    self._realtime_timings = None
                    self._ready_utterance = None
                    self._utterance.clear()
                    language = map_stt_language_code(event.language_code)
                    timings.stt_language = language
                    timings.log(logger, event="stt_first_final")
                    await self._replace_response(
                        self._handle_transcript(
                            event.transcript.strip(), language, timings
                        ),
                        label="realtime-turn",
                    )
                elif event.type == "error":
                    raise RuntimeError(f"Sarvam realtime STT error: {event.payload}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("realtime_stt_fallback call_id=%s", self._call_sid)
            timings = self._realtime_timings
            if timings is not None:
                timings.fallback_count += 1
                timings.log(logger, event="stt_fallback")
            if (
                self._settings.sarvam_realtime_stt_fallback_enabled
                and self._ready_utterance
            ):
                ready, self._ready_utterance = self._ready_utterance, None
                await self._replace_response(
                    self._handle_utterance(ready), label="stt-fallback"
                )
        finally:
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            stt_session, self._realtime_stt = self._realtime_stt, None
            if stt_session is not None:
                await stt_session.close()

    async def _replace_response(self, coroutine, *, label: str) -> None:
        await self._cancel_response(clear_provider=True)

        async def run_response() -> None:
            self._busy = True
            try:
                await coroutine
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Call response task failed: %s", label)
            finally:
                if self._response_task is asyncio.current_task():
                    self._busy = False

        self._response_task = asyncio.create_task(
            run_response(), name=f"call-{label}-{(self._call_sid or 'unknown')[-8:]}"
        )

    async def _cancel_response(self, *, clear_provider: bool) -> None:
        task = self._response_task
        self._response_task = None
        task_was_active = task is not None and not task.done()
        if task_was_active:
            logger.info(
                "call_generation_cancelled call_id=%s transport=%s",
                self._call_sid,
                self._transport.name,
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._busy = False
        if clear_provider and (task_was_active or self._playing):
            await self._clear_playback(force=True)

    async def _clear_playback(self, *, force: bool = False) -> None:
        """Abort provider-buffered audio and reset playback state for barge-in."""
        if self._playing or force:
            started = time.monotonic()
            timings = self._playback_timings
            self._playing = False
            self._playback_mark_name = None
            self._playback_timings = None
            self._pending_marks.clear()
            self._pending_mark_events.clear()
            active, self._active_tts_session = self._active_tts_session, None
            if active is not None:
                await active.close()
            await self._transport.clear()
            if timings is not None:
                timings.mark("interruption_clear")
                timings.add_duration(
                    "interruption_clear_delay",
                    int((time.monotonic() - started) * 1000),
                )
                self._persist_timings(timings)
                timings.log(logger, event="interruption_clear")

    def _handle_playback_mark(self, name: str) -> None:
        """Record Twilio's acknowledgement after buffered audio was played."""
        timings = self._pending_marks.pop(name, None)
        played = self._pending_mark_events.get(name)
        if played is not None:
            played.set()
        if name == self._playback_mark_name:
            self._playing = False
            self._playback_mark_name = None
            self._playback_timings = None
        if timings is None:
            return
        timings.mark("playback_mark")
        playback_ms = timings.duration_between(
            "first_outbound_audio", "playback_mark"
        )
        if playback_ms is not None:
            timings.add_duration("playback_duration", playback_ms)
        self._persist_timings(timings)
        timings.log(logger, event="playback_mark")

    # ---------- main loop ----------

    async def run(self) -> None:
        await self._transport.accept()
        logger.info("%s connected: waiting for start event", self._transport.name)
        try:
            while True:
                event = await self._transport.receive()
                if isinstance(event, StreamStarted):
                    await self._handle_start(event)
                elif isinstance(event, AudioReceived):
                    for offset in range(0, len(event.pcm16), CORE_FRAME_BYTES):
                        frame = event.pcm16[offset : offset + CORE_FRAME_BYTES]
                        if not frame:
                            continue
                        if self._settings.sarvam_realtime_stt_enabled:
                            if self._stt_audio_queue.full():
                                self._stt_audio_queue.get_nowait()
                            self._stt_audio_queue.put_nowait(frame)
                        ready = await self._feed_pcm(frame)
                        if ready is not None:
                            realtime_running = (
                                self._stt_receive_task is not None
                                and not self._stt_receive_task.done()
                            )
                            if realtime_running:
                                self._ready_utterance = ready
                            else:
                                await self._replace_response(
                                    self._handle_utterance(ready), label="turn"
                                )
                elif isinstance(event, PlaybackMarked):
                    self._handle_playback_mark(event.name)
                elif isinstance(event, StreamStopped):
                    break
                elif isinstance(event, ControlEvent):
                    continue
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Call stream error")
        finally:
            if self._stt_receive_task is not None:
                self._stt_receive_task.cancel()
                await asyncio.gather(
                    self._stt_receive_task, return_exceptions=True
                )
            await self._cancel_response(clear_provider=False)
            await self._cleanup()

    async def _handle_start(self, event: StreamStarted) -> None:
        self._stream_sid = event.stream_id
        self._call_sid = event.call_id
        logger.info(
            "Stream start: provider=%s call_id=%s stream_id=%s",
            self._transport.name,
            self._call_sid,
            self._stream_sid,
        )
        if self._call_sid and self._registry.get(self._call_sid) is None:
            self._registry.add(
                CallRecord(
                    call_sid=self._call_sid,
                    to_number="",
                    from_number="",
                    provider=(
                        "exotel"
                        if self._transport.name == "exotel_agent_stream"
                        else "twilio"
                    ),
                )
            )
        self._session_id = await self._create_session()
        if (
            self._settings.sarvam_realtime_stt_enabled
            and self._sarvam is not None
        ):
            self._stt_receive_task = asyncio.create_task(
                self._run_realtime_stt(),
                name=f"stt-{(self._call_sid or 'unknown')[-8:]}",
            )
        if self._call_sid:
            self._registry.update(self._call_sid, status="in-progress")
        greeting = ""
        try:
            from backend.metrics import TurnTimings

            greeting, _, _ = await self._engine.generate_greeting(
                TurnTimings(settings=self._settings),
                language=self._settings.default_language,
            )
        except Exception:  # noqa: BLE001 - emergency shell if LLM unavailable
            logger.warning("LLM greeting failed; using emergency profile shell")
        if not (greeting or "").strip():
            greeting = fallback_text(
                "greeting",
                self._settings.default_language,
                settings=self._settings,
                call_profile=getattr(self._engine, "call_profile", None),
            )
        await self._replace_response(
            self._speak(greeting),
            label="greeting",
        )

    async def _create_session(self) -> str:
        from backend.models import Lead

        with session_scope(self._factory) as db:
            session = Session(
                language=self._settings.default_language,
                current_state="collecting_identity",
            )
            db.add(session)
            db.flush()
            lead = Lead(session_id=session.id)
            if hasattr(self._engine, "apply_known_lead_fields"):
                self._engine.apply_known_lead_fields(lead)
            record = self._registry.get(self._call_sid) if self._call_sid else None
            phone = (record.to_number if record else "") or ""
            if phone and not lead.phone_number:
                lead.phone_number = phone
            db.add(lead)
            session_id = session.id
        if self._call_sid:
            self._registry.update(self._call_sid, session_id=session_id)
        return session_id

    async def _cleanup(self) -> None:
        if self._completed and self._call_sid and self._call_service is not None:
            await self._call_service.complete_call(self._call_sid)
        if self._call_sid:
            self._registry.update(self._call_sid, status="completed")
        active, self._active_tts_session = self._active_tts_session, None
        if active is not None:
            await active.close()
        await self._transport.close()
