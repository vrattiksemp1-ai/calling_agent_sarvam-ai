"""Tests for structured LLM response parsing and the repair path."""

import json

import pytest

from backend.llm_parsing import (
    parse_structured_response,
    parse_with_repair,
    extract_json,
)


def _valid_raw(**overrides):
    data = {
        "assistant_message": "Hello!",
        "detected_language": "en",
        "extracted_fields": {"full_name": "Rahul"},
        "fields_to_clear": [],
        "next_state": "collecting_contact",
        "conversation_complete": False,
        "needs_confirmation": False,
    }
    data.update(overrides)
    return json.dumps(data)


def test_parses_plain_json():
    parsed = parse_structured_response(_valid_raw())
    assert parsed is not None
    assert parsed.assistant_message == "Hello!"
    assert parsed.extracted_fields == {"full_name": "Rahul"}
    assert parsed.next_state == "collecting_contact"


def test_parses_markdown_fenced_json():
    raw = "```json\n" + _valid_raw() + "\n```"
    parsed = parse_structured_response(raw)
    assert parsed is not None


def test_parses_json_embedded_in_text():
    raw = "Sure, here is the output: " + _valid_raw() + " \n hope that helps!"
    parsed = parse_structured_response(raw)
    assert parsed is not None


def test_rejects_garbage():
    assert parse_structured_response("not json at all") is None
    assert parse_structured_response("") is None
    assert parse_structured_response("[1,2,3]") is None


def test_rejects_missing_required_fields():
    assert parse_structured_response('{"assistant_message": 42}') is None
    assert parse_structured_response('{"assistant_message": "hi", "next_state": 9}') is None


def test_unknown_next_state_cleared():
    parsed = parse_structured_response(_valid_raw(next_state="quantum_leap"))
    assert parsed is not None
    assert parsed.next_state == ""


@pytest.mark.asyncio
async def test_repair_retry_once_then_success():
    calls = []

    async def producer(repair):
        calls.append(repair)
        if repair:
            return _valid_raw(assistant_message="retried ok")
        return "this is not json"

    parsed, error = await parse_with_repair(producer)
    assert parsed is not None
    assert parsed.assistant_message == "retried ok"
    assert error is None
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_repair_failure_returns_error():
    calls = []

    async def producer(repair):
        calls.append(repair)
        return "still not json"

    parsed, error = await parse_with_repair(producer)
    assert parsed is None
    assert "valid structured response" in (error or "")
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_empty_first_response_triggers_repair():
    async def producer(repair):
        return _valid_raw(assistant_message="second attempt") if repair else ""

    parsed, error = await parse_with_repair(producer)
    assert parsed is not None
    assert error is None


def test_extract_json_handles_braces():
    assert extract_json('text {"a": 1} tail') == {"a": 1}
    assert extract_json('{"a": 1} {"b": 2}') == {"a": 1}
    assert extract_json("no braces") == {}
