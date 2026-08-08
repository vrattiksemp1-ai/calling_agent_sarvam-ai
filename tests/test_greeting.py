"""Dynamic greeting generation tests (ConversationEngine.generate_greeting)."""

import json

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
    # LLM wording is returned as-is (no hardcoded prefix stitching).
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


@pytest.mark.asyncio
async def test_generate_greeting_injects_language_instruction(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "કેમ છો? તમારું નામ શું છે?",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    text, _, _ = await engine.generate_greeting(
        TurnTimings(settings=settings), language="gu"
    )
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "Gujarati" in system
    # No hardcoded AI prefix stitching — LLM text is kept as returned.
    assert text == "કેમ છો? તમારું નામ શું છે?"


@pytest.mark.asyncio
async def test_generate_greeting_no_language_no_extra_instruction(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "Hello! May I have your name?",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    await engine.generate_greeting(TurnTimings(settings=settings))
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "The caller speaks" not in system


@pytest.mark.asyncio
async def test_generate_greeting_injects_business_identity(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "હાય! હું Vrattiks તરફથી કૉલ કરું છું. તમને શું જોઈએ છે, કહી દો?",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(
        make_mock_llm_client(settings, handler),
        business_name="Vrattiks",
        business_description="a technology and software company",
    )
    text, _, _ = await engine.generate_greeting(TurnTimings(settings=settings))
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "Vrattiks" in system
    assert "a technology and software company" in system
    # Phone greetings use the compact prompt for lower TTFT.
    assert "CALL PROFILE" in system
    assert "Vrattiks" in text


@pytest.mark.asyncio
async def test_generate_greeting_default_persona_is_shivangi(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "Hello! Do you have a minute?",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    await engine.generate_greeting(TurnTimings(settings=settings))
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "Shivangi" in system
    assert "CALL PROFILE" in system
    assert "phone caller" in system.lower() or "BDE" in system