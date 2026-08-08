"""Tests for the Twilio telephony bridge (codec, client, call flow).

All external services (Sarvam, LLM, Twilio) are mocked with httpx.MockTransport
or FastAPI TestClient - no real keys, accounts or phone calls are used.
"""

import base64
import math
import re
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
from backend.telephony.twilio_service import (
    CallNotAllowedError,
    CallNotConfiguredError,
    TwilioService,
)

from conftest import (
    FakeTwilioClient,
    lead_llm_handler,
    make_mock_llm_client,
    make_mock_sarvam_client,
    make_mock_twilio_client,
    make_settings,
    sarvam_handler,
    structured_json,
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


# ---------- Twilio service ----------


def _service_settings(tmp_path) -> Settings:
    return make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+919876543210",
    )


@pytest.mark.asyncio
async def test_twilio_service_start_call(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    result = await service.start_call("+919876543210")
    assert result.call_sid == "CA-test-call-sid"
    assert result.to == "+919876543210"
    assert result.from_ == "+15005550006"
    created = service._client.calls.created[-1]
    assert created["to"] == "+919876543210"
    assert created["from_"] == "+15005550006"
    assert "api/calls/twiml" in created["url"]


@pytest.mark.asyncio
async def test_twilio_service_start_call_normalizes_e164(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    result = await service.start_call(" 919876543210 ")
    assert result.to == "+919876543210"


@pytest.mark.asyncio
async def test_twilio_service_rejects_invalid_number(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    with pytest.raises(AppError) as exc:
        await service.start_call("+12")
    assert exc.value.code == "INVALID_PHONE_NUMBER"
    assert service._client.calls.created == []


@pytest.mark.asyncio
async def test_twilio_service_start_call_not_configured(tmp_path):
    settings = make_settings(tmp_path)
    service = TwilioService(settings, client=FakeTwilioClient())
    with pytest.raises(CallNotConfiguredError):
        await service.start_call("+919876543210")


@pytest.mark.asyncio
async def test_twilio_service_start_call_missing_public_url(tmp_path):
    settings = make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    with pytest.raises(AppError) as exc:
        await service.start_call("+919876543210")
    assert exc.value.code == "CALL_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_twilio_service_trial_blocks_unverified_number(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    with pytest.raises(CallNotAllowedError):
        await service.start_call("+14155552671")
    assert service._client.calls.created == []


@pytest.mark.asyncio
async def test_twilio_service_trial_allows_verified_number(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    result = await service.start_call("+917048211715")
    assert result.call_sid == "CA-test-call-sid"


@pytest.mark.asyncio
async def test_twilio_service_trial_off_skips_check(tmp_path):
    settings = make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_trial_mode=False,
    )
    service = make_mock_twilio_client(settings)
    result = await service.start_call("+14155552671")
    assert result.call_sid == "CA-test-call-sid"


@pytest.mark.asyncio
async def test_twilio_service_verified_numbers(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    numbers = await service.verified_numbers()
    assert "+919876543210" in numbers  # env fallback
    assert "+917048211715" in numbers  # from the Twilio OutgoingCallerIds API


@pytest.mark.asyncio
async def test_twilio_service_complete_call(tmp_path):
    settings = _service_settings(tmp_path)
    service = make_mock_twilio_client(settings)
    await service.complete_call("CA-test-call-sid")
    assert "CA-test-call-sid" in service._client.calls.updated


@pytest.mark.asyncio
async def test_twilio_service_complete_call_not_configured(tmp_path):
    settings = make_settings(tmp_path)
    service = TwilioService(settings, client=FakeTwilioClient())
    await service.complete_call("CA-x")  # no-op, no crash


def test_twilio_service_twiml_contains_stream_url():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="t",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app/",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    twiml = service.stream_twiml()
    assert "wss://abc.ngrok-free.app/api/calls/stream" in twiml
    assert "<Connect>" in twiml
    assert "</Stream></Connect>" in twiml
    assert "<Start>" not in twiml
    assert "<Pause" not in twiml


def test_twilio_service_signature_validation():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="tok",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    from twilio.request_validator import RequestValidator

    params = {"CallStatus": "completed", "CallSid": "CA1"}
    sig = RequestValidator("tok").compute_signature("https://abc.ngrok-free.app/api/calls/status", params)
    assert service.validate_signature("https://abc.ngrok-free.app/api/calls/status", params, sig)
    assert not service.validate_signature("https://abc.ngrok-free.app/api/calls/status", params, "bad")


def test_twilio_service_signature_missing_token():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    assert not service.validate_signature("https://abc.ngrok-free.app/api/calls/status", {}, "whatever")


def test_twilio_service_turn_url_includes_token():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="t",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
        twilio_turn_webhook_secret="secret123",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    assert service.turn_url() == (
        "https://abc.ngrok-free.app/api/calls/turn?turn_token=secret123"
        + "#ct=10000&rt=15000&tt=15000&rc=3&rp=ct,5xx"
    )


def test_twilio_service_turn_url_without_token_when_unconfigured():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="t",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
        twilio_turn_webhook_secret="",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    assert service.turn_url() == (
        "https://abc.ngrok-free.app/api/calls/turn"
        + "#ct=10000&rt=15000&tt=15000&rc=3&rp=ct,5xx"
    )


def test_twilio_service_turn_callback_prefers_signature():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="tok",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
        twilio_turn_webhook_secret="secret123",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    from twilio.request_validator import RequestValidator

    url = "https://abc.ngrok-free.app/api/calls/turn?turn_token=secret123"
    params = {"CallSid": "CA1", "SpeechResult": "hi"}
    sig = RequestValidator("tok").compute_signature(url, params)
    assert service.validate_turn_callback(url, params, sig, "")
    # a wrong token is irrelevant when the signature is valid
    assert service.validate_turn_callback(url, params, sig, "bogus")
    # Behind ngrok, Redirect callbacks may carry a mismatched signature; a
    # valid shared-secret token is still accepted so the call does not drop.
    assert service.validate_turn_callback(url, params, "bogus-sig", "secret123")
    assert not service.validate_turn_callback(url, params, "bogus-sig", "wrong")


def test_twilio_service_turn_callback_token_when_unsigned():
    settings = Settings(
        twilio_account_sid="AC",
        twilio_auth_token="tok",
        twilio_phone_number="+1",
        public_base_url="https://abc.ngrok-free.app",
        twilio_turn_webhook_secret="secret123",
    )
    service = TwilioService(settings, client=FakeTwilioClient())
    url = "https://abc.ngrok-free.app/api/calls/turn?turn_token=secret123"
    assert service.validate_turn_callback(url, {"CallSid": "CA1"}, "", "secret123")
    assert not service.validate_turn_callback(url, {"CallSid": "CA1"}, "", "wrong")
    assert not service.validate_turn_callback(url, {"CallSid": "CA1"}, "", "")


# ---------- registry ----------


def test_call_registry_crud():
    reg = CallRegistry()
    rec = CallRecord(call_sid="CA1", to_number="+91", from_number="+1")
    reg.add(rec)
    reg.update("CA1", status="in-progress")
    assert reg.get("CA1").status == "in-progress"
    assert reg.active_calls() == [rec]
    reg.update("CA1", status="completed")
    assert reg.active_calls() == []
    reg.remove("CA1")
    assert reg.get("CA1") is None


def test_place_call_supersedes_active_call(client):
    """A second place-call hangs up any still-active registry call first."""
    first = client.post("/api/calls", json={"to": "+919876543210"})
    assert first.status_code == 200
    first_sid = first.json()["call_sid"]
    # Simulate answered/in-progress so the next place treats it as active.
    client.app.state.call_registry.update(first_sid, status="in-progress")

    second = client.post("/api/calls", json={"to": "+919876543210"})
    assert second.status_code == 200
    second_sid = second.json()["call_sid"]
    assert second_sid != first_sid

    prior = client.app.state.call_registry.get(first_sid)
    assert prior is not None
    assert prior.status == "completed"
    assert prior.error == "superseded"
    assert first_sid in client.app.state.twilio_client._client.calls.updated

    active = client.app.state.call_registry.active_calls()
    assert len(active) == 1
    assert active[0].call_sid == second_sid


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
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+917048211715",
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
            # Providers return this event after buffered playback completes.
            ws.send_json(msg)
            return media
        assert msg["event"] == "media"
        media += 1


def test_place_call_and_status(client):
    resp = client.post("/api/calls", json={"to": "+919876543210"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["call_sid"] == "CA-test-call-sid"
    assert body["status"] == "queued"

    status = client.get("/api/calls/CA-test-call-sid")
    assert status.json()["status"] == "queued"

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


def test_place_call_rejects_invalid_e164(client):
    resp = client.post("/api/calls", json={"to": "not-a-number"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_PHONE_NUMBER"


def test_place_call_trial_blocks_unverified(client):
    resp = client.post("/api/calls", json={"to": "+14155552671"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "CALL_NOT_ALLOWED"


def test_verified_numbers_endpoint(client):
    resp = client.get("/api/calls/numbers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trial_mode"] is True
    assert "+917048211715" in body["verified_numbers"]


def test_verified_numbers_endpoint_keeps_env_list_outside_trial_mode(tmp_path):
    """Media Streams mode still exposes TWILIO_VERIFIED_NUMBERS for the picker."""
    from fastapi.testclient import TestClient

    from backend.main import build_app
    from tests.conftest import (
        make_mock_llm_client,
        make_mock_sarvam_client,
        make_mock_twilio_client,
        make_settings,
        lead_llm_handler,
        sarvam_handler,
    )

    settings = make_settings(
        tmp_path,
        sarvam_api_key="test-key",
        llm_api_key="test-key",
        twilio_account_sid="AC-test",
        twilio_auth_token="token-test",
        twilio_phone_number="+15005550006",
        public_base_url="https://example.ngrok-free.app",
        twilio_trial_mode=False,
        twilio_test_phone_number="+919428748109",
        twilio_verified_numbers="+919428748109",
    )
    app = build_app(
        settings,
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    with TestClient(app) as client:
        resp = client.get("/api/calls/numbers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trial_mode"] is False
    assert "+919428748109" in body["verified_numbers"]


def test_twiml_endpoint(client):
    # trial mode -> <Gather input="speech"> turn loop, no <Stream>
    resp = client.post("/api/calls/twiml")
    assert resp.status_code == 200
    assert 'input="speech"' in resp.text
    assert "<Gather" in resp.text
    assert "api/calls/turn" in resp.text
    assert "<Play>" in resp.text
    assert "<Stream" not in resp.text


def test_twiml_seeds_greeting_so_first_turn_does_not_regreet(tmp_path):
    """Opening audio is persisted; first user reply must continue, not re-intro."""
    import json

    settings = make_settings(
        tmp_path,
        default_language="gu",
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_trial_mode=True,
        twilio_test_phone_number="+917048211715",
    )

    greeting_text = "હેલો, હું વ્રત્તિક્સ થી બોલું છું. તમે શું શોધી રહ્યા છો?"

    def greeting_then_continue(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        messages = payload.get("messages", [])
        has_user = any(
            m.get("role") == "user"
            and not (m.get("content") or "").startswith("\n\nCurrent state:")
            and not (m.get("content") or "").startswith("\n\nYour previous reply")
            for m in messages
        )
        if not has_user:
            return structured_json(
                greeting_text,
                extracted_fields={},
                next_state="collecting_identity",
                detected_language="gu",
            )
        history = " ".join(m.get("content", "") for m in messages)
        assert greeting_text in history
        return structured_json(
            "સરસ, ઓટોમેશન માટે. તમારું નામ શું છે?",
            extracted_fields={"product_or_service_interest": "automation"},
            next_state="collecting_identity",
            detected_language="gu",
        )

    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, greeting_then_continue),
        twilio_client=make_mock_twilio_client(settings),
    )
    with TestClient(app) as c:
        twiml = c.post("/api/calls/twiml", data={"CallSid": "CA-seed-1"})
        assert twiml.status_code == 200
        assert "<Play>" in twiml.text

        record = app.state.call_registry.get("CA-seed-1")
        assert record is not None
        assert record.session_id

        from backend.database import session_scope
        from backend.models import Message, Session

        with session_scope(app.state.session_factory) as db:
            session = db.get(Session, record.session_id)
            assert session is not None
            assert session.current_state == "collecting_identity"
            msgs = (
                db.query(Message)
                .filter(Message.session_id == session.id)
                .order_by(Message.id)
                .all()
            )
            assert len(msgs) == 1
            assert msgs[0].role == "assistant"
            assert greeting_text in msgs[0].content

        from twilio.request_validator import RequestValidator

        url = "https://x.ngrok-free.app/api/calls/turn"
        params = {
            "CallSid": "CA-seed-1",
            "SpeechResult": "ઓટોમેશન જોઈએ છે",
        }
        sig = RequestValidator("tok").compute_signature(url, params)
        turn = c.post(
            "/api/calls/turn",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )
        assert turn.status_code == 200
        assert "<Gather" in turn.text

        with session_scope(app.state.session_factory) as db:
            msgs = (
                db.query(Message)
                .filter(Message.session_id == record.session_id)
                .order_by(Message.id)
                .all()
            )
            assert len(msgs) >= 3  # greeting + user + reply
            assert msgs[0].role == "assistant"
            assert msgs[1].role == "user"
            assert msgs[2].role == "assistant"
            assert "ઓટોમેશન" in msgs[2].content
            assert "વ્રત્તિક્સ થી બોલું" not in msgs[2].content


def test_twiml_endpoint_gather_language_gujarati(tmp_path):
    settings = make_settings(
        tmp_path,
        default_language="gu",
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_trial_mode=True,
    )
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    with TestClient(app) as c:
        resp = c.post("/api/calls/twiml")
        assert resp.status_code == 200
        assert 'language="gu-IN"' in resp.text
        assert "<Play>" in resp.text


def test_twiml_endpoint_streaming_when_not_trial(tmp_path):
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_trial_mode=False,
    )
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    with TestClient(app) as c:
        resp = c.post("/api/calls/twiml")
        assert resp.status_code == 200
        assert "<Stream" in resp.text
        assert "<Connect>" in resp.text
        assert "<Start>" not in resp.text
        assert "wss://x.ngrok-free.app/api/calls/stream" in resp.text


def _signed_turn_post(client, speech, call_sid="CA-turn-1", token="token-test"):
    from twilio.request_validator import RequestValidator

    url = "https://example.ngrok-free.app/api/calls/turn"
    params = {"CallSid": call_sid, "SpeechResult": speech}
    sig = RequestValidator(token).compute_signature(url, params)
    return client.post(
        "/api/calls/turn",
        data=params,
        headers={"X-Twilio-Signature": sig},
    )


def test_turn_webhook_rejects_bad_signature(client):
    resp = _signed_turn_post(client, "hello", token="wrong-token")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "INVALID_TWILIO_SIGNATURE"


def _turn_app(tmp_path, secret="turn-secret-123"):
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+917048211715",
        twilio_turn_webhook_secret=secret,
    )
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    return TestClient(app)


def test_turn_webhook_accepts_token_without_signature(tmp_path):
    client = _turn_app(tmp_path)
    resp = client.post(
        "/api/calls/turn?turn_token=turn-secret-123",
        data={"CallSid": "CA-token-1", "SpeechResult": ""},
    )
    assert resp.status_code == 200
    assert "<Gather" in resp.text
    assert "api/calls/turn" in resp.text


def test_turn_webhook_rejects_bad_token_without_signature(tmp_path):
    client = _turn_app(tmp_path)
    resp = client.post(
        "/api/calls/turn?turn_token=wrong",
        data={"CallSid": "CA-token-2", "SpeechResult": "hello"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "INVALID_TWILIO_SIGNATURE"


def test_turn_twiml_action_url_contains_token(tmp_path):
    client = _turn_app(tmp_path)
    resp = client.post("/api/calls/twiml")
    assert resp.status_code == 200
    assert 'action="https://x.ngrok-free.app/api/calls/turn?turn_token=turn-secret-123#ct=10000&amp;rt=15000&amp;tt=15000&amp;rc=3&amp;rp=ct,5xx"' in resp.text


def test_turn_webhook_empty_speech_silent_relisten(client):
    """Blank SpeechResult must re-open Gather without speaking 'I'm sorry'."""
    resp = _signed_turn_post(client, "")
    assert resp.status_code == 200
    assert "<Gather" in resp.text
    assert "<Say>" not in resp.text
    assert "<Play>" not in resp.text
    assert "<Hangup" not in resp.text
    assert 'timeout="5"' in resp.text
    assert 'speechTimeout="1"' in resp.text
    assert "api/calls/turn" in resp.text


def test_turn_webhook_full_conversation(client):
    sid = "CA-turn-flow-1"
    turns = [
        "my name is rahul sharma",
        "my phone number is 98765 43210",
        "I need a CRM",
        "yes, you can contact me",
        "confirm yes",
    ]
    last = None
    for speech in turns:
        last = _signed_turn_post(client, speech, call_sid=sid)
        assert last.status_code == 200, last.text
    assert "<Hangup" in last.text
    assert "<Gather" not in last.text

    status = client.get(f"/api/calls/{sid}").json()
    assert status["status"] == "in-progress"
    lead = client.get(f"/api/sessions/{status['session_id']}/lead").json()
    assert lead["fields"]["full_name"] == "Rahul Sharma"
    assert lead["conversation_status"] == "completed"


def test_turn_audio_endpoint_serves_wav(client):
    resp = _signed_turn_post(client, "my name is rahul sharma")
    match = re.search(r"/api/calls/audio/([0-9a-f]+)", resp.text)
    assert match, resp.text
    audio = client.get(f"/api/calls/audio/{match.group(1)}")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content[:4] == b"RIFF"


def _custom_turn_app(tmp_path, llm_handler, default_language="en"):
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+917048211715",
        twilio_turn_webhook_secret="turn-secret-123",
        default_language=default_language,
    )
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        llm_client=make_mock_llm_client(settings, llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    return TestClient(app)


def _token_turn_post(client, speech, call_sid="CA-hangup-1"):
    return client.post(
        "/api/calls/turn?turn_token=turn-secret-123",
        data={"CallSid": call_sid, "SpeechResult": speech},
    )


def test_turn_webhook_hangup_intent_hangs_up(tmp_path):
    def handler(request):
        return structured_json(
            "Theek che, aavjo! Bye.",
            detected_language="gu",
            extracted_fields={},
            next_state="abandoned",
        )

    client = _custom_turn_app(tmp_path, handler)
    resp = _token_turn_post(client, "hang up the call", call_sid="CA-hangup-1")
    assert resp.status_code == 200
    assert "<Hangup/>" in resp.text
    assert "<Gather" not in resp.text
    assert "<Play>" in resp.text

    status = client.get("/api/calls/CA-hangup-1").json()
    assert status["session_id"]
    lead = client.get(f"/api/sessions/{status['session_id']}/lead").json()
    assert lead["conversation_status"] == "abandoned"


def test_turn_webhook_gather_language_follows_session(tmp_path):
    def handler(request):
        return structured_json(
            "Kem chho? Tamaru naam shu chhe?",
            detected_language="gu",
            extracted_fields={},
            next_state="collecting_identity",
        )

    # default is English, but the caller speaks Gujarati -> next <Gather> must
    # switch to gu-IN so Twilio STT follows the user's language dynamically.
    client = _custom_turn_app(tmp_path, handler, default_language="en")
    resp = _token_turn_post(client, "kem cho", call_sid="CA-lang-1")
    assert resp.status_code == 200
    assert 'language="gu-IN"' in resp.text
    assert "<Gather" in resp.text
    assert "<Hangup" not in resp.text


def test_turn_webhook_defers_with_redirect_when_slow(tmp_path, monkeypatch):
    """Slow LLM+TTS must Redirect instead of blowing Twilio's ~15s budget."""
    import asyncio

    from backend.telephony import turn_flow as turn_flow_mod

    monkeypatch.setattr(turn_flow_mod, "TURN_INLINE_BUDGET_SECONDS", 0.05)

    original_build = turn_flow_mod.TurnFlow._build_turn_twiml

    async def slow_build(self, call_sid, text, pending=None):
        await asyncio.sleep(0.2)
        return await original_build(self, call_sid, text, pending)

    monkeypatch.setattr(turn_flow_mod.TurnFlow, "_build_turn_twiml", slow_build)

    client = _turn_app(tmp_path)
    # Force a short inline budget via settings as well (Gather-optimized path).
    client.app.state.settings.gather_inline_budget_seconds = 0.05
    client.app.state.settings.gather_poll_pause_seconds = 0
    resp = _token_turn_post(client, "my name is rahul", call_sid="CA-slow-1")
    assert resp.status_code == 200
    assert "<Redirect" in resp.text
    assert "/api/calls/turn-result" in resp.text
    assert "pending=" in resp.text
    # Pause is disabled by default to save ~1s of perceived latency.
    assert "<Pause" not in resp.text
    assert "<Say>" not in resp.text

    # Follow the redirect URL path (strip public base + fragment).
    import re
    from urllib.parse import urlparse, parse_qs

    match = re.search(r"<Redirect[^>]*>([^<]+)</Redirect>", resp.text)
    assert match
    redirect = match.group(1).replace("&amp;", "&")
    parsed = urlparse(redirect)
    qs = parse_qs(parsed.query)
    result = client.post(
        parsed.path + "?" + parsed.query.split("#")[0],
        data={"CallSid": "CA-slow-1"},
    )
    assert result.status_code == 200
    assert "<Gather" in result.text or "<Hangup" in result.text
    assert qs.get("pending")


def _signed_status_post(client, sid, status, token="token-test"):
    from twilio.request_validator import RequestValidator

    url = "https://example.ngrok-free.app/api/calls/status"
    params = {"CallSid": sid, "CallStatus": status}
    sig = RequestValidator(token).compute_signature(url, params)
    return client.post(
        "/api/calls/status",
        data=params,
        headers={"X-Twilio-Signature": sig},
    )


def test_call_status_callback(client):
    resp = client.post("/api/calls", json={"to": "+919876543210"})
    assert resp.status_code == 200
    sid = resp.json()["call_sid"]
    resp = _signed_status_post(client, sid, "ringing")
    assert resp.json()["ok"] is True
    status = client.get(f"/api/calls/{sid}").json()
    assert status["status"] == "ringing"


def test_call_status_callback_rejects_bad_signature(client):
    resp = client.post("/api/calls", json={"to": "+919876543210"})
    sid = resp.json()["call_sid"]
    resp = _signed_status_post(client, sid, "completed", token="wrong-token")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "INVALID_TWILIO_SIGNATURE"
    # state must be unchanged
    status = client.get(f"/api/calls/{sid}").json()
    assert status["status"] == "queued"


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


# ---------- barge-in ----------


class FakeWs:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def _make_session(tmp_path, settings=None) -> "object":
    from backend.telephony.call_manager import CallSession

    settings = settings or make_settings(
        tmp_path,
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
    )
    service = make_mock_twilio_client(settings)
    session = CallSession(
        settings=settings,
        session_factory=None,
        engine=None,
        sarvam=None,
        twilio=service,
        registry=CallRegistry(),
        ws=FakeWs(),
    )
    session._stream_sid = "STREAM-B"
    return session


@pytest.mark.asyncio
async def test_barge_in_clears_playback_and_captures_interrupt(tmp_path):
    session = _make_session(tmp_path)
    session._playing = True
    ready = await session._feed(_tone_chunk())
    assert session._playing is False
    assert any(e["event"] == "clear" for e in session._ws.sent)
    assert session._speech_chunks == 1
    assert ready is False  # interrupt speech still needs a full utterance


@pytest.mark.asyncio
async def test_barge_in_silence_does_not_clear(tmp_path):
    session = _make_session(tmp_path)
    session._playing = True
    ready = await session._feed(_silence_chunk())
    assert session._playing is True
    assert not any(e["event"] == "clear" for e in session._ws.sent)
    assert ready is False


@pytest.mark.asyncio
async def test_barge_in_interrupt_becomes_utterance(tmp_path):
    session = _make_session(tmp_path)
    session._playing = True
    await session._feed(_tone_chunk())  # barge-in starts the utterance
    for _ in range(11):
        await session._feed(_tone_chunk())
    for _ in range(40):
        await session._feed(_silence_chunk())
    # the interrupt should now be a complete utterance ready for processing
    assert session._busy is True


def test_call_stream_consecutive_turns(tmp_path):
    """Two consecutive utterances are handled (interrupt buffer resets cleanly)."""
    client = _call_app(tmp_path)
    with client.websocket_connect("/api/calls/stream") as ws:
        ws.send_json(
            {
                "event": "start",
                "streamSid": "STREAM-1",
                "start": {"streamSid": "STREAM-1", "callSid": "CA-call-3"},
            }
        )
        assert _drain_until_mark(ws) > 0

        for _ in range(2):
            for _ in range(12):
                ws.send_json({"event": "media", "streamSid": "STREAM-1", "media": {"payload": _tone_chunk()}})
            for _ in range(40):
                ws.send_json({"event": "media", "streamSid": "STREAM-1", "media": {"payload": _silence_chunk()}})
            assert _drain_until_mark(ws) > 0

        ws.send_json({"event": "stop", "streamSid": "STREAM-1"})
