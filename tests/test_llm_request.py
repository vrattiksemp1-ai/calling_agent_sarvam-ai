"""LLM request body construction tests (reasoning_effort / max_tokens)."""

import json

import httpx
import pytest

from tests.conftest import make_mock_llm_client, make_settings


def _llm_response() -> httpx.Response:
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


def _capture(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _llm_response()

    return handler


@pytest.mark.asyncio
async def test_reasoning_off_sends_null(tmp_path):
    captured = {}
    settings = make_settings(tmp_path, llm_reasoning_effort="none")
    client = make_mock_llm_client(settings, _capture(captured))
    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["body"]["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_reasoning_level_is_sent(tmp_path):
    captured = {}
    settings = make_settings(tmp_path, llm_reasoning_effort="low")
    client = make_mock_llm_client(settings, _capture(captured))
    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["body"]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_reasoning_absent_when_unconfigured(tmp_path):
    captured = {}
    settings = make_settings(tmp_path)
    client = make_mock_llm_client(settings, _capture(captured))
    await client.generate([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in captured["body"]


@pytest.mark.asyncio
async def test_max_tokens_sent_when_configured(tmp_path):
    captured = {}
    settings = make_settings(tmp_path, llm_max_tokens=512)
    client = make_mock_llm_client(settings, _capture(captured))
    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["body"]["max_tokens"] == 512


@pytest.mark.asyncio
async def test_json_mode_still_requested(tmp_path):
    captured = {}
    settings = make_settings(tmp_path, llm_use_json_mode=True)
    client = make_mock_llm_client(settings, _capture(captured))
    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["body"]["response_format"] == {"type": "json_object"}
