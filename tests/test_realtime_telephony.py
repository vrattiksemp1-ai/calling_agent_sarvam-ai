"""Mocked tests for the concurrent multi-carrier call core."""

import asyncio
import base64
import struct
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import build_app
from backend.telephony import mulaw
from backend.telephony.call_manager import CallRegistry, CallSession
from backend.telephony.exotel_service import ExotelService
from backend.telephony.transport import (
    AudioReceived,
    ExotelAgentStreamTransport,
    PlaybackMarked,
    StreamStarted,
    StreamStopped,
)
from backend.telephony.twilio_service import OutboundCallResult

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    lead_llm_handler,
    make_mock_llm_client,
    make_mock_sarvam_client,
    make_mock_twilio_client,
    make_settings,
    sarvam_handler,
)


def _tone_pcm() -> bytes:
    return struct.pack("<160h", *([12000] * 160))


class QueueTransport:
    name = "test_transport"

    def __init__(self):
        self.events = asyncio.Queue()
        self.sent_audio: list[tuple[float, bytes]] = []
        self.marks: list[str] = []
        self.clears = 0
        self.clear_event = asyncio.Event()
        self.audio_event = asyncio.Event()

    async def accept(self):
        return None

    async def receive(self):
        return await self.events.get()

    async def send_audio(self, pcm16):
        self.sent_audio.append((time.monotonic(), pcm16))
        self.audio_event.set()
        return True

    async def flush_audio(self):
        return False

    async def send_mark(self, name):
        self.marks.append(name)

    async def clear(self):
        self.clears += 1
        self.clear_event.set()

    async def close(self):
        return None


def _session(tmp_path, transport) -> CallSession:
    return CallSession(
        settings=make_settings(tmp_path),
        session_factory=None,
        engine=None,
        sarvam=None,
        twilio=None,
        registry=CallRegistry(),
        transport=transport,
    )


@pytest.mark.asyncio
async def test_receive_loop_cancels_greeting_generation_on_speech(tmp_path, monkeypatch):
    """Inbound media is consumed while greeting/provider work is still awaiting."""

    transport = QueueTransport()
    session = _session(tmp_path, transport)
    greeting_started = asyncio.Event()

    async def fake_create_session():
        return "session-test"

    async def blocked_greeting(text, timings=None):
        greeting_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "_create_session", fake_create_session)
    monkeypatch.setattr(session, "_speak", blocked_greeting)

    await transport.events.put(StreamStarted("MZ-1", "CA-1"))
    run_task = asyncio.create_task(session.run())
    await asyncio.wait_for(greeting_started.wait(), timeout=1)

    await transport.events.put(AudioReceived(_tone_pcm()))
    await asyncio.wait_for(transport.clear_event.wait(), timeout=1)
    assert session._speech_chunks == 1
    assert session._response_task is None

    await transport.events.put(StreamStopped("callended"))
    await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_playback_is_paced_and_cancellable(tmp_path):
    transport = QueueTransport()
    session = _session(tmp_path, transport)
    wav = mulaw.pcm16_to_wav(_tone_pcm() * 6, 8000)

    await session._stream_wav(wav)
    assert len(transport.sent_audio) == 6
    intervals = [
        later[0] - earlier[0]
        for earlier, later in zip(transport.sent_audio, transport.sent_audio[1:])
    ]
    assert all(interval >= 0.012 for interval in intervals)
    assert len(transport.marks) == 1
    session._handle_playback_mark(transport.marks[0])
    assert session._playing is False

    transport.sent_audio.clear()
    transport.audio_event.clear()
    long_wav = mulaw.pcm16_to_wav(_tone_pcm() * 50, 8000)
    playback = asyncio.create_task(session._stream_wav(long_wav))
    session._response_task = playback
    await asyncio.wait_for(transport.audio_event.wait(), timeout=1)
    await session._cancel_response(clear_provider=True)
    assert playback.cancelled()
    assert len(transport.sent_audio) < 50
    assert transport.clears == 1


class FakeWebSocket:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        return await self.incoming.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_exotel_raw_pcm_start_media_mark_clear_encoding():
    pcm = _tone_pcm()
    ws = FakeWebSocket(
        [
            {
                "event": "start",
                "stream_sid": "MZ-raw",
                "start": {
                    "stream_sid": "MZ-raw",
                    "call_sid": "CA-raw",
                    "media_format": {
                        "encoding": "audio/x-raw",
                        "sample_rate": "8000",
                        "bit_rate": "16",
                    },
                },
            },
            {
                "event": "media",
                "stream_sid": "MZ-raw",
                "media": {"payload": base64.b64encode(pcm).decode()},
            },
            {
                "event": "mark",
                "stream_sid": "MZ-raw",
                "mark": {"name": "done-1"},
            },
        ]
    )
    adapter = ExotelAgentStreamTransport(ws)
    start = await adapter.receive()
    assert start == StreamStarted("MZ-raw", "CA-raw")
    audio = await adapter.receive()
    assert audio == AudioReceived(pcm)
    assert await adapter.receive() == PlaybackMarked("done-1")

    assert await adapter.send_audio(pcm) is False
    assert ws.sent == []
    assert await adapter.flush_audio() is True
    await adapter.send_mark("out-1")
    await adapter.clear()
    assert ws.sent[0]["stream_sid"] == "MZ-raw"
    raw_payload = base64.b64decode(ws.sent[0]["media"]["payload"])
    assert len(raw_payload) == 3200
    assert raw_payload.startswith(pcm)
    assert ws.sent[1] == {
        "event": "mark",
        "stream_sid": "MZ-raw",
        "mark": {"name": "out-1"},
    }
    assert ws.sent[2] == {"event": "clear", "stream_sid": "MZ-raw"}


@pytest.mark.asyncio
async def test_exotel_mulaw_message_shape_is_decoded_and_encoded():
    pcm = _tone_pcm()
    encoded = mulaw.encode_mulaw(pcm)
    ws = FakeWebSocket(
        [
            {
                "event": "start",
                "start": {
                    "stream_sid": "MZ-mu",
                    "call_sid": "CA-mu",
                    "media_format": {
                        "encoding": "audio/x-mulaw",
                        "sample_rate": 8000,
                    },
                },
            },
            {
                "event": "media",
                "media": {"payload": base64.b64encode(encoded).decode()},
            },
        ]
    )
    adapter = ExotelAgentStreamTransport(ws)
    await adapter.receive()
    received = await adapter.receive()
    assert received.pcm16 == mulaw.decode_mulaw(encoded)
    assert await adapter.send_audio(pcm) is False
    assert await adapter.flush_audio() is True
    encoded_payload = base64.b64decode(ws.sent[0]["media"]["payload"])
    assert len(encoded_payload) == 3200
    assert encoded_payload.startswith(encoded)


@pytest.mark.asyncio
async def test_exotel_connect_voice_ai_call_creation_is_mocked(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "call": {
                    "sid": "EX-call-1",
                    "status": "in-progress",
                    "from": "+919876543210",
                }
            },
        )

    settings = make_settings(
        tmp_path,
        public_base_url="https://bot.example.com",
        exotel_base_url="https://api.in.exotel.com",
        exotel_account_sid="AC-exotel",
        exotel_api_key="key",
        exotel_api_token="token",
        exotel_caller_id="08012345678",
        exotel_flow_id="",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = ExotelService(settings, http_client=client)
    result = await service.start_call("+919876543210")

    request = captured["request"]
    assert request.url == (
        "https://api.in.exotel.com/v1/accounts/AC-exotel/calls/connect"
    )
    form = parse_qs(request.content.decode())
    assert form["from"] == ["+919876543210"]
    assert form["callerid"] == ["08012345678"]
    assert form["streamtype"] == ["bidirectional"]
    assert form["streamurl"] == [
        "wss://bot.example.com/api/calls/exotel/stream"
    ]
    assert request.headers["authorization"].startswith("Basic ")
    assert result.call_sid == "EX-call-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_exotel_flow_call_uses_voice_v1_canonical_endpoint(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={"Call": {"Sid": "EX-flow-1", "Status": "in-progress"}},
        )

    settings = make_settings(
        tmp_path,
        public_base_url="https://bot.example.com",
        exotel_base_url="https://api.exotel.com",
        exotel_account_sid="vrattiks1",
        exotel_api_key="key",
        exotel_api_token="token",
        exotel_caller_id="09513886367",
        exotel_flow_id="1310646",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = ExotelService(settings, http_client=client)
    result = await service.start_call("+919876543210")

    request = captured["request"]
    assert request.url == (
        "https://api.exotel.com/v1/Accounts/vrattiks1/Calls/connect"
    )
    form = parse_qs(request.content.decode())
    assert form["From"] == ["+919876543210"]
    assert form["CallerId"] == ["09513886367"]
    assert form["Url"] == [
        "https://my.exotel.com/vrattiks1/exoml/start_voice/1310646"
    ]
    assert form["CallType"] == ["trans"]
    assert result.call_sid == "EX-flow-1"
    await client.aclose()


def test_place_call_provider_override_and_twilio_default(tmp_path):
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC-twilio",
        twilio_auth_token="token",
        twilio_phone_number="+15005550006",
        public_base_url="https://bot.example.com",
        twilio_test_phone_number="+919876543210",
        exotel_account_sid="AC-exotel",
        exotel_api_key="key",
        exotel_api_token="token",
        exotel_caller_id="08012345678",
    )

    class FakeExotel:
        def __init__(self):
            self.started = []

        async def start_call(self, to):
            self.started.append(to)
            return OutboundCallResult("EX-selected", "queued", to, "08012345678")

        async def complete_call(self, call_sid):
            return None

        async def aclose(self):
            return None

    exotel = FakeExotel()
    twilio = make_mock_twilio_client(settings)
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=twilio,
        exotel_client=exotel,
    )
    with TestClient(app) as client:
        selected = client.post(
            "/api/calls",
            json={"to": "+919876543210", "provider": "exotel"},
        )
        assert selected.status_code == 200
        assert selected.json()["call_sid"] == "EX-selected"
        assert app.state.call_registry.get("EX-selected").provider == "exotel"

        defaulted = client.post("/api/calls", json={"to": "+919876543210"})
        assert defaulted.status_code == 200
        assert defaulted.json()["call_sid"].startswith("CA-test-call-sid")
        assert app.state.call_registry.get(defaulted.json()["call_sid"]).provider == "twilio"
        assert exotel.started == ["+919876543210"]
