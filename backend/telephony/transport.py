"""Provider-neutral bidirectional telephony WebSocket transports.

The conversation core consumes 8 kHz, mono, signed little-endian PCM and does
not know whether the wire protocol is Twilio Media Streams or Exotel
AgentStream.  Each adapter owns the provider event names, identifiers and
audio encoding.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from fastapi import WebSocket

from backend.telephony import mulaw

CORE_SAMPLE_RATE = 8000
PCM_BYTES_PER_SAMPLE = 2
FRAME_DURATION_SECONDS = 0.020
CORE_FRAME_BYTES = int(CORE_SAMPLE_RATE * FRAME_DURATION_SECONDS) * PCM_BYTES_PER_SAMPLE
EXOTEL_MIN_CHUNK_BYTES = 3200


@dataclass(frozen=True)
class StreamStarted:
    stream_id: str
    call_id: str


@dataclass(frozen=True)
class AudioReceived:
    pcm16: bytes


@dataclass(frozen=True)
class PlaybackMarked:
    name: str


@dataclass(frozen=True)
class StreamStopped:
    reason: str = ""


@dataclass(frozen=True)
class ControlEvent:
    """A provider event the conversation core does not currently consume."""

    name: str


TransportEvent: TypeAlias = (
    StreamStarted | AudioReceived | PlaybackMarked | StreamStopped | ControlEvent
)


class CallTransport(Protocol):
    """Wire adapter used by ``CallSession``."""

    name: str

    async def accept(self) -> None: ...

    async def receive(self) -> TransportEvent: ...

    async def send_audio(self, pcm16: bytes) -> bool: ...

    async def flush_audio(self) -> bool: ...

    async def send_mark(self, name: str) -> None: ...

    async def clear(self) -> None: ...

    async def close(self) -> None: ...


class _WebSocketTransport:
    name = "websocket"

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._stream_id: str | None = None
        self._send_lock = asyncio.Lock()

    async def accept(self) -> None:
        await self._ws.accept()

    async def _send(self, payload: dict) -> None:
        async with self._send_lock:
            await self._ws.send_json(payload)

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass


class TwilioMediaStreamTransport(_WebSocketTransport):
    """Official Twilio bidirectional Media Streams JSON and PCMU framing."""

    name = "twilio_media_stream"

    async def receive(self) -> TransportEvent:
        message = await self._ws.receive_json()
        event = message.get("event")
        if event == "start":
            start = message.get("start") or {}
            self._stream_id = start.get("streamSid") or message.get("streamSid") or ""
            return StreamStarted(
                stream_id=self._stream_id,
                call_id=start.get("callSid") or message.get("callSid") or "",
            )
        if event == "media":
            media = message.get("media") or {}
            payload = base64.b64decode(media.get("payload") or "")
            return AudioReceived(mulaw.decode_mulaw(payload))
        if event == "mark":
            return PlaybackMarked((message.get("mark") or {}).get("name") or "")
        if event == "stop":
            return StreamStopped((message.get("stop") or {}).get("reason") or "")
        return ControlEvent(str(event or "unknown"))

    async def send_audio(self, pcm16: bytes) -> bool:
        await self._send(
            {
                "event": "media",
                "streamSid": self._stream_id,
                "media": {
                    "payload": base64.b64encode(mulaw.encode_mulaw(pcm16)).decode("ascii")
                },
            }
        )
        return True

    async def flush_audio(self) -> bool:
        return False

    async def send_mark(self, name: str) -> None:
        await self._send(
            {
                "event": "mark",
                "streamSid": self._stream_id,
                "mark": {"name": name},
            }
        )

    async def clear(self) -> None:
        await self._send({"event": "clear", "streamSid": self._stream_id})


def _resample_pcm16(pcm16: bytes, source_rate: int, target_rate: int) -> bytes:
    """Small integer-ratio telephony resampler.

    AgentStream's supported rates are 8/16/24 kHz, so nearest-neighbour
    decimation/expansion is sufficient at this protocol boundary. Sarvam still
    receives canonical 8 kHz WAV and provider-native output is restored here.
    """

    if not pcm16 or source_rate == target_rate:
        return pcm16
    sample_count = len(pcm16) // PCM_BYTES_PER_SAMPLE
    samples = memoryview(pcm16[: sample_count * PCM_BYTES_PER_SAMPLE]).cast("h")
    if source_rate > target_rate and source_rate % target_rate == 0:
        step = source_rate // target_rate
        return b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples[::step])
    if target_rate > source_rate and target_rate % source_rate == 0:
        repeats = target_rate // source_rate
        return b"".join(
            int(sample).to_bytes(2, "little", signed=True)
            for sample in samples
            for _ in range(repeats)
        )
    raise ValueError(f"Unsupported AgentStream sample rate conversion: {source_rate}->{target_rate}")


class ExotelAgentStreamTransport(_WebSocketTransport):
    """Exotel Voicebot/AgentStream bidirectional JSON framing.

    Exotel normally negotiates raw little-endian PCM. The adapter also accepts
    PCMU start metadata used by the extended Voicebot configuration.
    """

    name = "exotel_agent_stream"

    def __init__(self, ws: WebSocket) -> None:
        super().__init__(ws)
        self._encoding = "raw"
        self._sample_rate = CORE_SAMPLE_RATE
        self._outbound_buffer = bytearray()

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @staticmethod
    def _is_mulaw(encoding: str) -> bool:
        normalized = encoding.lower()
        return any(token in normalized for token in ("mulaw", "mu-law", "pcmu"))

    async def receive(self) -> TransportEvent:
        message = await self._ws.receive_json()
        event = message.get("event")
        if event == "start":
            start = message.get("start") or {}
            media_format = start.get("media_format") or start.get("mediaFormat") or {}
            encoding = str(media_format.get("encoding") or "raw")
            self._encoding = "mulaw" if self._is_mulaw(encoding) else "raw"
            try:
                self._sample_rate = int(
                    media_format.get("sample_rate")
                    or media_format.get("sampleRate")
                    or CORE_SAMPLE_RATE
                )
            except (TypeError, ValueError):
                self._sample_rate = CORE_SAMPLE_RATE
            if self._sample_rate not in {8000, 16000, 24000}:
                raise ValueError("Unsupported AgentStream sample rate")
            self._stream_id = (
                start.get("stream_sid")
                or start.get("streamSid")
                or message.get("stream_sid")
                or message.get("streamSid")
                or ""
            )
            return StreamStarted(
                stream_id=self._stream_id,
                call_id=(
                    start.get("call_sid")
                    or start.get("callSid")
                    or message.get("call_sid")
                    or ""
                ),
            )
        if event == "media":
            payload = base64.b64decode((message.get("media") or {}).get("payload") or "")
            pcm16 = mulaw.decode_mulaw(payload) if self._encoding == "mulaw" else payload
            return AudioReceived(
                _resample_pcm16(pcm16, self._sample_rate, CORE_SAMPLE_RATE)
            )
        if event == "mark":
            return PlaybackMarked((message.get("mark") or {}).get("name") or "")
        if event == "stop":
            return StreamStopped((message.get("stop") or {}).get("reason") or "")
        return ControlEvent(str(event or "unknown"))

    async def _send_audio_payload(self, payload: bytes) -> None:
        await self._send(
            {
                "event": "media",
                "stream_sid": self._stream_id,
                "media": {"payload": base64.b64encode(payload).decode("ascii")},
            }
        )

    async def send_audio(self, pcm16: bytes) -> bool:
        """Buffer core frames into Exotel's minimum 3,200-byte payload."""
        provider_pcm = _resample_pcm16(pcm16, CORE_SAMPLE_RATE, self._sample_rate)
        payload = (
            mulaw.encode_mulaw(provider_pcm)
            if self._encoding == "mulaw"
            else provider_pcm
        )
        self._outbound_buffer.extend(payload)
        sent = False
        while len(self._outbound_buffer) >= EXOTEL_MIN_CHUNK_BYTES:
            chunk = bytes(self._outbound_buffer[:EXOTEL_MIN_CHUNK_BYTES])
            del self._outbound_buffer[:EXOTEL_MIN_CHUNK_BYTES]
            await self._send_audio_payload(chunk)
            sent = True
        return sent

    async def flush_audio(self) -> bool:
        if not self._outbound_buffer:
            return False
        # Exotel requires every outbound chunk to be at least 3,200 bytes and a
        # multiple of 320. Pad only the final partial chunk with silence.
        padding = EXOTEL_MIN_CHUNK_BYTES - len(self._outbound_buffer)
        payload = bytes(self._outbound_buffer) + (b"\x00" * padding)
        self._outbound_buffer.clear()
        await self._send_audio_payload(payload)
        return True

    async def send_mark(self, name: str) -> None:
        await self._send(
            {
                "event": "mark",
                "stream_sid": self._stream_id,
                "mark": {"name": name},
            }
        )

    async def clear(self) -> None:
        self._outbound_buffer.clear()
        await self._send({"event": "clear", "stream_sid": self._stream_id})
