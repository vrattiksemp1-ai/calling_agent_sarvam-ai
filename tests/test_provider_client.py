"""Provider client tests with mocked HTTP responses."""

import base64
import json

import httpx
import pytest

from backend.errors import ProviderUnavailableError, SttError, TtsError
from tests.conftest import make_mock_sarvam_client, make_settings


def _write_wav(path, payload=b"RIFF" + b"\x00" * 100):
    path.write_bytes(payload)
    return str(path)


def _wav_bytes() -> bytes:
    return bytes.fromhex(
        "524946460400000057415645666d74201000000001000100401f0000803e0000020010006461746100000000"
    )


@pytest.mark.asyncio
async def test_transcribe_success(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/speech-to-text")
        assert request.method == "POST"
        assert request.headers.get("api-subscription-key") == "test-key"
        return httpx.Response(200, json={"transcript": "hello world"})

    settings = make_settings(tmp_path, sarvam_api_key="test-key")
    client = make_mock_sarvam_client(settings, handler)
    text, latency = await client.transcribe(_write_wav(tmp_path / "a.wav"), 500)
    assert text == "hello world"
    assert latency >= 0


@pytest.mark.asyncio
async def test_transcribe_empty_text_raises_stt_error(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"transcript": "   "})

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    with pytest.raises(SttError):
        await client.transcribe(_write_wav(tmp_path / "a.wav"), 500)


@pytest.mark.asyncio
async def test_synthesize_success(tmp_path):
    def handler(request):
        assert request.url.path.endswith("/text-to-speech")
        body = json.loads(request.content)
        assert body["target_language_code"] == "en-IN"
        assert body["model"] == "bulbul:v3"
        assert body["text"] == "hello"
        wav = _wav_bytes()
        return httpx.Response(
            200, json={"audios": [base64.b64encode(wav).decode("ascii")]}
        )

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    audio, mime, latency = await client.synthesize("hello", "en")
    assert audio == _wav_bytes()
    assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_synthesize_maps_detected_language_to_tts_code(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        wav = _wav_bytes()
        return httpx.Response(
            200, json={"audios": [base64.b64encode(wav).decode("ascii")]}
        )

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    await client.synthesize("namaste", "hi")
    assert captured["body"]["target_language_code"] == "hi-IN"


@pytest.mark.asyncio
async def test_synthesize_maps_gujarati_to_gu_in(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        wav = _wav_bytes()
        return httpx.Response(
            200, json={"audios": [base64.b64encode(wav).decode("ascii")]}
        )

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    await client.synthesize("kem cho", "gu")
    assert captured["body"]["target_language_code"] == "gu-IN"


@pytest.mark.asyncio
async def test_synthesize_sends_expressiveness_params(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        wav = _wav_bytes()
        return httpx.Response(
            200, json={"audios": [base64.b64encode(wav).decode("ascii")]}
        )

    settings = make_settings(tmp_path, sarvam_tts_temperature=0.9, sarvam_tts_pace=1.1)
    client = make_mock_sarvam_client(settings, handler)
    await client.synthesize("hello")
    assert captured["body"]["temperature"] == 0.9
    assert captured["body"]["pace"] == 1.1
    assert captured["body"]["speech_sample_rate"] == "8000"


@pytest.mark.asyncio
async def test_stream_synthesize_sends_expressiveness_params(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_wav_bytes(), headers={"content-type": "audio/wav"}
        )

    settings = make_settings(tmp_path, sarvam_tts_temperature=0.9, sarvam_tts_pace=1.1)
    client = make_mock_sarvam_client(settings, handler)
    chunks = [c async for c in client.stream_synthesize("hello")]
    assert chunks
    assert captured["body"]["temperature"] == 0.9
    assert captured["body"]["pace"] == 1.1


@pytest.mark.asyncio
async def test_synthesize_http_error_becomes_tts_error(tmp_path):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    with pytest.raises(TtsError):
        await client.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_missing_audio_becomes_tts_error(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"audios": []})

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    with pytest.raises(TtsError):
        await client.synthesize("hello")


@pytest.mark.asyncio
async def test_health_check_ok(tmp_path):
    def handler(request):
        return httpx.Response(200, json={})

    settings = make_settings(tmp_path, sarvam_api_key="test-key")
    client = make_mock_sarvam_client(settings, handler)
    status = await client.health_check()
    assert status.status == "ok"
    assert status.details["api_key_set"] is True


@pytest.mark.asyncio
async def test_health_check_degraded_without_key(tmp_path):
    client = make_mock_sarvam_client(make_settings(tmp_path), lambda req: httpx.Response(200, json={}))
    status = await client.health_check()
    assert status.status == "degraded"
    assert status.details["api_key_set"] is False


@pytest.mark.asyncio
async def test_health_check_error(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    settings = make_settings(tmp_path, sarvam_api_key="test-key")
    client = make_mock_sarvam_client(settings, handler)
    status = await client.health_check()
    assert status.status == "error"


@pytest.mark.asyncio
async def test_provider_unavailable_when_unreachable(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_mock_sarvam_client(make_settings(tmp_path), handler)
    with pytest.raises(ProviderUnavailableError):
        await client.transcribe(_write_wav(tmp_path / "a.wav"), 500)
