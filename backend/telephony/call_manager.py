"""Real-time call session handling for the Twilio Media Streams bridge.

A CallSession owns one phone call:
  * receives inbound mu-law audio chunks from Twilio,
  * detects speech/silence (simple energy VAD),
  * sends each utterance through STT -> LLM -> TTS,
  * streams the mu-law reply back over the socket,
  * hangs up when the lead conversation completes.
"""

from __future__ import annotations

import struct
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket

from backend.config import Settings
from backend.conversation import ConversationEngine
from backend.database import session_scope
from backend.errors import AppError
from backend.metrics import TurnTimings
from backend.models import Message, Session
from backend.providers.sarvam_client import SarvamClient
from backend.telephony import mulaw
from backend.telephony.twilio_client import TwilioClient
from backend.utils.logging import get_logger

logger = get_logger(__name__)

GREETING = (
    "Hi there! I'm the lead qualification assistant. To get you the right "
    "information quickly, may I ask your name first?"
)
REPEAT_MESSAGE = "I'm sorry, I didn't catch that. Could you say that again?"
GOODBYE = "Thank you for your time. Your details are saved and this call is ending. Goodbye!"

# VAD tuning (all in 20 ms mu-law chunks = 160 samples each)
SPEECH_ENERGY = 500
MIN_UTTERANCE_CHUNKS = 8
SILENCE_CHUNKS = 30
MAX_UTTERANCE_CHUNKS = 750


@dataclass
class CallRecord:
    call_sid: str
    to_number: str
    from_number: str
    status: str = "initiated"
    error: str | None = None
    session_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


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


class CallSession:
    def __init__(
        self,
        settings: Settings,
        session_factory,
        engine: ConversationEngine,
        sarvam: SarvamClient,
        twilio: TwilioClient,
        registry: CallRegistry,
        ws: WebSocket,
    ) -> None:
        self._settings = settings
        self._factory = session_factory
        self._engine = engine
        self._sarvam = sarvam
        self._twilio = twilio
        self._registry = registry
        self._ws = ws

        self._stream_sid: str | None = None
        self._call_sid: str | None = None
        self._session_id: str | None = None
        self._busy = False
        self._completed = False
        self._utterance = bytearray()
        self._speech_chunks = 0
        self._silence_chunks = 0

    # ---------- media streaming ----------

    async def _send(self, payload: dict) -> None:
        await self._ws.send_json(payload)

    async def _send_media(self, payload_mulaw: bytes) -> None:
        await self._send(
            {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": mulaw.mulaw_to_base64(payload_mulaw)},
            }
        )

    async def _send_mark(self, name: str) -> None:
        await self._send(
            {"event": "mark", "streamSid": self._stream_sid, "mark": {"name": name}}
        )

    async def _stream_wav(self, wav_bytes: bytes) -> None:
        mulaw_bytes = mulaw.tts_wav_to_mulaw(wav_bytes)
        if not mulaw_bytes:
            return
        for i in range(0, len(mulaw_bytes), 160):
            await self._send_media(mulaw_bytes[i:i + 160])
        await self._send_mark("reply_done")

    async def _speak(self, text: str) -> None:
        try:
            audio, _, _ = await self._sarvam.synthesize(text, None)
        except AppError as exc:
            logger.warning("Call TTS failed: %s", exc.code)
            return
        await self._stream_wav(audio)

    # ---------- turn processing ----------

    async def _handle_utterance(self) -> None:
        pcm16 = bytes(self._utterance)
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
            transcript, stt_ms = await self._sarvam.transcribe(str(tmp), duration_ms)
        except AppError as exc:
            logger.warning("Call STT failed: %s", exc.code)
            await self._speak(REPEAT_MESSAGE)
            return
        finally:
            tmp.unlink(missing_ok=True)

        settings = self._settings
        timings = TurnTimings(settings=settings)
        timings.stt_latency_ms = stt_ms
        timings.transcript_char_count = len(transcript)
        timings.audio_duration_ms = duration_ms

        reply = ""
        completed = False
        tts_bytes: bytes | None = None
        with session_scope(self._factory) as db:
            session = db.get(Session, self._session_id)
            if session is None:
                return
            _, parsed = await self._engine.process_turn(db, session, transcript, timings)
            reply = parsed.assistant_message
            timings.response_char_count = len(reply)
            if session.status != "abandoned":
                try:
                    tts_bytes, _, tts_ms = await self._sarvam.synthesize(
                        reply, parsed.detected_language
                    )
                    timings.tts_latency_ms = tts_ms
                    timings.tts_attempted = True
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
                last_assistant.tts_latency_ms = timings.tts_latency_ms
                last_assistant.total_turn_latency_ms = timings.total()
                last_assistant.estimated_provider_cost = (
                    (last_assistant.estimated_provider_cost or 0.0)
                    + timings.estimated_stt_cost()
                    + timings.estimated_tts_cost()
                )
                db.add(last_assistant)
            completed = session.status == "completed"

        if tts_bytes:
            await self._stream_wav(tts_bytes)
        elif reply:
            await self._speak(REPEAT_MESSAGE)

        if completed:
            await self._speak(GOODBYE)
            self._completed = True

    # ---------- VAD ----------

    @staticmethod
    def _energy(pcm16: bytes) -> int:
        if not pcm16:
            return 0
        count = len(pcm16) // 2
        values = struct.unpack(f"<{count}h", pcm16)
        return max(abs(v) for v in values)

    def _feed(self, payload_b64: str) -> None:
        if self._busy:
            self._utterance.clear()
            self._speech_chunks = 0
            self._silence_chunks = 0
            return
        pcm16 = mulaw.base64_to_mulaw(payload_b64)
        if self._energy(pcm16) > SPEECH_ENERGY:
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

    # ---------- main loop ----------

    async def run(self) -> None:
        await self._ws.accept()
        try:
            while True:
                message = await self._ws.receive_json()
                event = message.get("event")
                if event == "start":
                    start = message.get("start") or {}
                    self._stream_sid = start.get("streamSid") or message.get("streamSid")
                    self._call_sid = start.get("callSid") or message.get("callSid")
                    if self._call_sid and self._registry.get(self._call_sid) is None:
                        self._registry.add(
                            CallRecord(
                                call_sid=self._call_sid,
                                to_number="",
                                from_number="",
                            )
                        )
                    self._session_id = await self._create_session()
                    if self._call_sid:
                        self._registry.update(self._call_sid, status="in-progress")
                    self._busy = True
                    await self._speak(GREETING)
                    self._busy = False
                elif event == "media":
                    media = message.get("media") or {}
                    payload = media.get("payload") or ""
                    self._feed(payload)
                    if self._busy and self._utterance:
                        await self._handle_utterance()
                        self._busy = False
                        if self._completed:
                            break
                elif event == "stop":
                    break
        except Exception:
            logger.exception("Call stream error")
        finally:
            await self._cleanup()

    async def _create_session(self) -> str:
        with session_scope(self._factory) as db:
            session = Session(language=self._settings.default_language)
            db.add(session)
            db.flush()
            session_id = session.id
        if self._call_sid:
            self._registry.update(self._call_sid, session_id=session_id)
        return session_id

    async def _cleanup(self) -> None:
        if self._completed and self._call_sid:
            await self._twilio.complete_call(self._call_sid)
        if self._call_sid:
            self._registry.update(self._call_sid, status="completed")
        try:
            await self._ws.close()
        except Exception:
            pass
