"""Dynamic greeting generation tests (ConversationEngine.generate_greeting)."""

import httpx
import pytest

from backend.conversation import ConversationEngine
from backend.errors import LlmStructuredOutputError
from backend.metrics import TurnTimings
from tests.conftest import make_mock_llm_client, make_settings, structured_json


def _greeting_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Hi there! Lovely day. May I have your name?",
            extracted_fields={},
            next_state="collecting_identity",
        )

    return handler


@pytest.mark.asyncio
async def test_generate_greeting_returns_llm_text(tmp_path):
    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, _greeting_handler()))
    text, latency, usage = await engine.generate_greeting(TurnTimings(settings=settings))
    assert text == "Hi there! Lovely day. May I have your name?"
    assert latency >= 0
    assert isinstance(usage, dict)


@pytest.mark.asyncio
async def test_generate_greeting_allows_empty_text(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json("", extracted_fields={}, next_state="collecting_identity")

    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    text, _, _ = await engine.generate_greeting(TurnTimings(settings=settings))
    assert text == ""


@pytest.mark.asyncio
async def test_generate_greeting_raises_on_non_structured(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "definitely not json"}}
                ],
                "usage": {},
            },
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    with pytest.raises(LlmStructuredOutputError):
        await engine.generate_greeting(TurnTimings(settings=settings))