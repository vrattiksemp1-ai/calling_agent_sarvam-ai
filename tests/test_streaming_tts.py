"""Streaming TTS tests (Sarvam /text-to-speech/stream + live /audio serving)."""

import base64
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.errors import TtsError
from backend.main import build_app
from tests.conftest import (
    _wav_bytes,
    lead_llm_handler,
    make_mock_llm_client,
    make_mock_sarvam_client,
    make_mock_twilio_client,
    make_settings,
)


def _streaming_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/text-to-speech/stream"):
        return httpx.Response(
            200, content=_wav_bytes(), headers={"content-type": "audio/wav"}
        )
    if request.url.path.endswith("/text-to-speech"):
        return httpx.Response(
            200, json={"audios": [base64.b64encode(_wav_bytes()).decode("ascii")]}
        )
    return httpx.Response(404, json={})


def _app(tmp_path, *, settings=None, handler=None):
    settings = settings or make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+917048211715",
    )
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, handler or _streaming_handler),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    return TestClient(app)


def _audio_url_from_twiml(twiml: str) -> str:
    match = re.search(r"/api/calls/audio/([0-9a-f]+)", twiml)
    assert match, twiml
    return f"/api/calls/audio/{match.group(1)}"


@pytest.mark.asyncio
async def test_stream_synthesize_yields_audio(tmp_path):
    settings = make_settings(tmp_path)
    client = make_mock_sarvam_client(settings, _streaming_handler)
    chunks = [c async for c in client.stream_synthesize("hello", "en")]
    assert chunks == [_wav_bytes()]


@pytest.mark.asyncio
async def test_stream_synthesize_sends_stream_endpoint(tmp_path):
    settings = make_settings(tmp_path)
    seen: dict[str, bool] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["stream"] = request.url.path.endswith("/text-to-speech/stream")
        return httpx.Response(
            200, content=_wav_bytes(), headers={"content-type": "audio/wav"}
        )

    client = make_mock_sarvam_client(settings, handler)
    async for _ in client.stream_synthesize("hello"):
        pass
    assert seen["stream"] is True


@pytest.mark.asyncio
async def test_stream_synthesize_sends_integer_sample_rate(tmp_path):
    """Sarvam's stream endpoint rejects a string sample rate (400 invalid
    request); it must be sent as an int even though /text-to-speech accepts
    the string form."""
    settings = make_settings(tmp_path)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content or b"{}")
        return httpx.Response(
            200, content=_wav_bytes(), headers={"content-type": "audio/wav"}
        )

    client = make_mock_sarvam_client(settings, handler)
    async for _ in client.stream_synthesize("hello"):
        pass
    assert captured["json"]["speech_sample_rate"] == 8000
    assert isinstance(captured["json"]["speech_sample_rate"], int)


@pytest.mark.asyncio
async def test_stream_synthesize_http_error_becomes_tts_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    settings = make_settings(tmp_path)
    client = make_mock_sarvam_client(settings, handler)
    with pytest.raises(TtsError):
        chunks = [c async for c in client.stream_synthesize("hello")]
        assert not chunks


def test_turn_reply_served_from_live_stream(tmp_path):
    client = _app(tmp_path)
    url = "https://x.ngrok-free.app/api/calls/turn"
    from twilio.request_validator import RequestValidator

    params = {"CallSid": "CA-stream-1", "SpeechResult": "my name is rahul sharma"}
    sig = RequestValidator("tok").compute_signature(url, params)
    resp = client.post(
        "/api/calls/turn", data=params, headers={"X-Twilio-Signature": sig}
    )
    assert resp.status_code == 200, resp.text
    audio_url = _audio_url_from_twiml(resp.text)
    audio = client.get(audio_url)
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content[:4] == b"RIFF"


def test_stream_failure_falls_back_to_buffered_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/text-to-speech/stream"):
            return httpx.Response(500, json={"error": "stream unavailable"})
        if request.url.path.endswith("/text-to-speech"):
            return httpx.Response(
                200, json={"audios": [base64.b64encode(_wav_bytes()).decode("ascii")]}
            )
        return httpx.Response(404, json={})

    client = _app(tmp_path, handler=handler)
    from twilio.request_validator import RequestValidator

    url = "https://x.ngrok-free.app/api/calls/turn"
    params = {"CallSid": "CA-fallback-1", "SpeechResult": "my name is rahul sharma"}
    sig = RequestValidator("tok").compute_signature(url, params)
    resp = client.post(
        "/api/calls/turn", data=params, headers={"X-Twilio-Signature": sig}
    )
    assert resp.status_code == 200, resp.text
    audio_url = _audio_url_from_twiml(resp.text)
    audio = client.get(audio_url)
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content[:4] == b"RIFF"


def test_greeting_uses_streaming_when_enabled(tmp_path):
    client = _app(tmp_path)
    resp = client.post("/api/calls/twiml")
    assert resp.status_code == 200
    audio_url = _audio_url_from_twiml(resp.text)
    audio = client.get(audio_url)
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"


def test_streaming_disabled_uses_buffered_file(tmp_path):
    settings = make_settings(
        tmp_path,
        sarvam_api_key="x",
        llm_api_key="x",
        twilio_account_sid="AC123",
        twilio_auth_token="tok",
        twilio_phone_number="+15005550006",
        public_base_url="https://x.ngrok-free.app",
        twilio_test_phone_number="+917048211715",
        sarvam_tts_streaming=False,
    )
    client = _app(tmp_path, settings=settings)
    resp = client.post("/api/calls/twiml")
    assert resp.status_code == 200
    audio_url = _audio_url_from_twiml(resp.text)
    audio = client.get(audio_url)
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content[:4] == b"RIFF"


def test_missing_audio_still_404(tmp_path):
    client = _app(tmp_path)
    resp = client.get("/api/calls/audio/deadbeefdeadbeef")
    assert resp.status_code == 404
