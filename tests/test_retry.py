"""Retry behaviour tests for the Sarvam and LLM clients."""

import json

import httpx
import pytest

from backend.providers.llm_client import LlmClient
from backend.providers.sarvam_client import SarvamClient
from tests.conftest import make_mock_llm_client, make_settings


def _silence_sleep(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("backend.providers.sarvam_client.asyncio.sleep", no_sleep)
    monkeypatch.setattr("backend.providers.llm_client.asyncio.sleep", no_sleep)


@pytest.mark.asyncio
async def test_sarvam_retries_transient_error(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) <= 2:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"transcript": "recovered"})

    client = SarvamClient(
        make_settings(tmp_path),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    path = tmp_path / "a.wav"
    path.write_bytes(b"data")
    text, _, _ = await client.transcribe(str(path), 500)
    assert text == "recovered"
    assert len(calls) == 3  # 1 + MAX_RETRIES(2)
    assert client.last_attempt_count == 3


@pytest.mark.asyncio
async def test_sarvam_gives_up_after_retries(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": "busy"})

    client = SarvamClient(
        make_settings(tmp_path),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    path = tmp_path / "a.wav"
    path.write_bytes(b"data")
    with pytest.raises(Exception) as exc:
        await client.transcribe(str(path), 500)
    assert "not reachable" in str(exc.value.message) or "returned an error" in str(exc.value.message)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_llm_retries_then_succeeds(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, json={"error": "rate limited"})
        reply = {
            "assistant_message": "ok",
            "detected_language": "en",
            "extracted_fields": {},
            "fields_to_clear": [],
            "next_state": "",
            "conversation_complete": False,
            "needs_confirmation": False,
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(reply)}}],
                "usage": {},
            },
        )

    client = make_mock_llm_client(make_settings(tmp_path), handler)
    text, latency, usage = await client.generate(
        [{"role": "user", "content": "hi"}]
    )
    assert "assistant_message" in text
    assert len(calls) == 3
    assert client.last_attempt_count == 3


@pytest.mark.asyncio
async def test_llm_gives_up_after_retries(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)

    def handler(request):
        return httpx.Response(500, json={"error": "internal"})

    client = make_mock_llm_client(make_settings(tmp_path), handler)
    with pytest.raises(Exception) as exc:
        await client.generate([{"role": "user", "content": "hi"}])
    assert "not reachable" in str(exc.value.message) or "returned an error" in str(exc.value.message)


@pytest.mark.asyncio
async def test_retry_respects_limits_bounded(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": "busy"})

    client = SarvamClient(
        make_settings(tmp_path, sarvam_api_key="test-key"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await client.health_check()
    # health_check does not retry by design (bounded, cheap status check)
    assert len(calls) == 1
