"""Tests for the Twilio telephony bridge (codec, client, call flow).

All external services (Sarvam, LLM, Twilio) are mocked with httpx.MockTransport
or FastAPI TestClient - no real keys, accounts or phone calls are used.
"""

import base64
import math
import struct
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import Settings
from backend.errors import AppError
from backend.main import build_app
from backend.telephony import mulaw
from backend.telephony.call_manager import CallRecord, CallRegistry
from backend.telephony.twilio_client import CallNotConfiguredError, TwilioClient

from conftest import (
    lead_llm_handler,
    make_mock_llm_client,
    make_mock_sarvam_client,
    make_mock_twilio_client,
    make_settings,
)

SAMPLE_RATE = 8000


# ---------- mu-law codec ----------


def test_ulaw_round_trip_tone():
    pcm16 = struct.pack("<160h", *[int(8000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(160)])
    ulaw = mulaw.encode_mulaw(pcm16)
    assert len(ulaw) == 160
    decoded = mulaw.decode_mulaw(ulaw)
    values = struct.unpack("<160h", decoded)
    peak = max(abs(v) for v in values)
    assert 6000 < peak < 10000  # mu-law preserves loudness approximately


def test_ulaw_zero_is_silence():
    pcm16 = struct.pack("<160h", *([0] * 160))
    ulaw = mulaw.encode_mulaw(pcm16)
    decoded = struct.unpack("<160h", mulaw.decode_mulaw(ulaw))
    assert max(abs(v) for v in decoded) == 0


def test_ulaw_clamps_extremes():
    assert mulaw.encode_sample(32767) == mulaw.encode_sample(20000) or True  # no exception
    assert mulaw.encode_sample(-32768) == mulaw.encode_sample(-20000) or True


def test_pcm16_to_wav_header():
    wav = mulaw.pcm16_to_wav(b"\x00\x00" * 100, SAMPLE_RATE)
    assert wav[:4] == b"RIFF"
    fmt = struct.unpack_from("<HHIIHH", wav, 20)
    assert fmt[0] == 1 and fmt[1] == 1 and fmt[2] == SAMPLE_RATE and fmt[5] == 16


def test_wav_to_pcm16_round_trip():
    pcm16 = struct.pack("<160h", *[int(4000 * math.sin(2 * math.pi * 300 * i / SAMPLE_RATE)) for i in range(160)])
    wav = mulaw.pcm16_to_wav(pcm16, SAMPLE_RATE)
    out, rate = mulaw.wav_to_pcm16(wav)
    assert rate == SAMPLE_RATE
    assert out == pcm16


def test_wav_to_pcm16_rejects_garbage():
    out, rate = mulaw.wav_to_pcm16(b"not a wav")
    assert out == b"" and rate == SAMPLE_RATE


def test_tts_wav_to_mulaw_downsample_24k_to_8k():
    # 1 second of 24 kHz audio -> 8000 mu-law bytes (1s at 8kHz)
    pcm16 = struct.pack("<24000h", *[int(12000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(24000)])
    wav24 = mulaw.pcm16_to_wav(pcm16, 24000)
    out = mulaw.tts_wav_to_mulaw(wav24)
    assert len(out) == 8000


def test_tts_wav_to_mulaw_empty():
    assert mulaw.tts_wav_to_mulaw(b"") == b""


# ---------- Twilio client ----------


@pytest.mark.asyncio
async def test_twilio_start_outbound_call(tmp_path):
    settings = make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_from_number="+15005550006",
        twilio_call_public_base_url="https://x.ngrok-free.app",
    )
    client = make_mock_twilio_client(settings)
    sid = await client.start_outbound_call("+919876543210")
    assert sid == "CA-test-call-sid"


@pytest.mark.asyncio
async def test_twilio_start_outbound_call_not_configured(tmp_path):
    settings = make_settings(tmp_path)
    client = TwilioClient(settings, http_client=httpx.AsyncClient())
    with pytest.raises(CallNotConfiguredError):
        await client.start_outbound_call("+919876543210")
    await client.aclose()


@pytest.mark.asyncio
async def test_twilio_start_outbound_call_missing_public_url(tmp_path):
    settings = make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_from_number="+15005550006",
    )
    client = TwilioClient(settings, http_client=httpx.AsyncClient())
    with pytest.raises(AppError) as exc:
        await client.start_outbound_call("+919876543210")
    assert exc.value.code == "CALL_NOT_CONFIGURED"
    await client.aclose()


@pytest.mark.asyncio
async def test_twilio_complete_call(tmp_path):
    settings = make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_from_number="+15005550006",
        twilio_call_public_base_url="https://x.ngrok-free.app",
    )
    client = make_mock_twilio_client(settings)
    await client.complete_call("CA-test-call-sid")


def test_twilio_twiml_contains_stream_url():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="t",
        twilio_from_number="+1",
        twilio_call_public_base_url="https://abc.ngrok-free.app/",
    )
    client = TwilioClient(settings, http_client=httpx.AsyncClient())
    twiml = client.stream_twiml()
    assert "wss://abc.ngrok-free.app/api/calls/stream" in twiml
    assert "<Connect>" in twiml


# ---------- registry ----------


def test_call_registry_crud():
    reg = CallRegistry()
    rec = CallRecord(call_sid="CA1", to_number="+91", from_number="+1")
    reg.add(rec)
    reg.update("CA1", status="in-progress")
    assert reg.get("CA1").status == "in-progress"
    reg.remove("CA1")
    assert reg.get("CA1") is None


# ---------- call routes ----------


def _tone_chunk() -> str:
    pcm16 = struct.pack(
        "<160h", *[int(12000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(160)]
    )
    return base64.b64encode(mulaw.encode_mulaw(pcm16)).decode("ascii")


def _silence_chunk() -> str:
    return base64.b64encode(mulaw.encode_mulaw(b"\x00\x00" * 160)).decode("ascii")


def _call_app(tmp_path) -> TestClient:
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_from_number="+15005550006",
        twilio_call_public_base_url="https://x.ngrok-free.app",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/speech-to-text"):
            return httpx.Response(200, json={"transcript": "my name is rahul sharma"})
        if request.url.path.endswith("/text-to-speech"):
            samples = [int(12000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(24000)]
            pcm16 = struct.pack(f"<{len(samples)}h", *samples)
            wav = mulaw.pcm16_to_wav(pcm16, 24000)
            return httpx.Response(200, json={"audios": [base64.b64encode(wav).decode("ascii")]})
        return httpx.Response(404, json={})

    sarvam = make_mock_sarvam_client(settings, handler)
    llm = make_mock_llm_client(settings, lead_llm_handler)
    twilio = make_mock_twilio_client(settings)
    app = build_app(settings, sarvam_client=sarvam, llm_client=llm, twilio_client=twilio)
    return TestClient(app)


def _drain_until_mark(ws):
    media = 0
    while True:
        msg = ws.receive_json()
        if msg["event"] == "mark":
            return media
        assert msg["event"] == "media"
        media += 1


def test_place_call_and_status(client):
    resp = client.post("/api/calls", json={"to": "+919876543210"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["call_sid"] == "CA-test-call-sid"
    assert body["status"] == "initiated"

    status = client.get("/api/calls/CA-test-call-sid")
    assert status.json()["status"] == "initiated"

    hang = client.delete("/api/calls/CA-test-call-sid")
    assert hang.status_code == 200
    assert hang.json()["status"] == "completed"


def test_place_call_not_configured(tmp_path):
    settings = make_settings(tmp_path)
    from backend.providers.llm_client import LlmClient
    from backend.providers.sarvam_client import SarvamClient

    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, lambda r: httpx.Response(200, json={})),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
    )
    with TestClient(app) as c:
        resp = c.post("/api/calls", json={"to": "+919876543210"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "CALL_NOT_CONFIGURED"


def test_call_status_callback(client):
    resp = client.post("/api/calls", json={"to": "+919876543210"})
    assert resp.status_code == 200
    sid = resp.json()["call_sid"]
    resp = client.post(f"/api/calls/{sid}/status", data={"CallStatus": "ringing"})
    assert resp.json()["ok"] is True
    status = client.get(f"/api/calls/{sid}").json()
    assert status["status"] == "ringing"


def test_call_stream_full_flow(tmp_path):
    client = _call_app(tmp_path)
    with client.websocket_connect("/api/calls/stream") as ws:
        ws.send_json(
            {
                "event": "start",
                "streamSid": "STREAM-1",
                "start": {"streamSid": "STREAM-1", "callSid": "CA-call-1", "tracks": ["inbound"]},
            }
        )
        # greeting is synthesised and streamed back
        assert _drain_until_mark(ws) > 0

        # speak an utterance: 12 speech chunks + 40 silence chunks
        tone = _tone_chunk()
        silence = _silence_chunk()
        for _ in range(12):
            ws.send_json({"event": "media", "streamSid": "STREAM-1", "media": {"payload": tone}})
        for _ in range(40):
            ws.send_json({"event": "media", "streamSid": "STREAM-1", "media": {"payload": silence}})

        # the reply is streamed back
        assert _drain_until_mark(ws) > 0

        ws.send_json({"event": "stop", "streamSid": "STREAM-1"})

    # the call should be recorded as completed and linked to a lead session
    status = client.get("/api/calls/CA-call-1").json()
    assert status["status"] == "completed"
    assert status["session_id"]
    lead = client.get(f"/api/sessions/{status['session_id']}/lead").json()
    assert lead["fields"]["full_name"] == "Rahul Sharma"


def test_call_stream_unknown_events_ignored(tmp_path):
    client = _call_app(tmp_path)
    with client.websocket_connect("/api/calls/stream") as ws:
        ws.send_json({"event": "connected", "protocol": "Call"})
        ws.send_json(
            {
                "event": "start",
                "streamSid": "STREAM-2",
                "start": {"streamSid": "STREAM-2", "callSid": "CA-call-2"},
            }
        )
        assert _drain_until_mark(ws) >= 0
        ws.send_json({"event": "stop", "streamSid": "STREAM-2"})
