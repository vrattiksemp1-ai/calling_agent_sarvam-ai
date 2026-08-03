"""Structured JSON response parsing from the LLM.

The LLM must return exactly one JSON object. We parse leniently (strip code
fences), validate with Pydantic, and support a single repair retry before
falling back to a safe message.
"""

import json
import re

from pydantic import BaseModel, Field, field_validator

from backend.state_machine import is_valid_state
from backend.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMStructuredResponse(BaseModel):
    assistant_message: str = Field(..., max_length=1200)
    detected_language: str = Field(default="en", max_length=16)
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    fields_to_clear: list[str] = Field(default_factory=list)
    next_state: str = Field(default="", max_length=40)
    conversation_complete: bool = False
    needs_confirmation: bool = False

    @field_validator("assistant_message")
    @classmethod
    def message_not_empty(cls, value: str) -> str:
        return value.strip()


def extract_json(text: str) -> dict:
    cleaned = _FENCE_RE.sub("", text or "").strip()
    if not cleaned:
        return {}
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        candidate = _extract_first_object(cleaned)
        if candidate is None:
            return {}
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _extract_first_object(text: str) -> str | None:
    """Return the first balanced JSON object found in text, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_structured_response(text: str) -> LLMStructuredResponse | None:
    """Return a validated response, or None when the text is not usable."""
    data = extract_json(text)
    if not data:
        return None
    try:
        parsed = LLMStructuredResponse.model_validate(data)
        if not is_valid_state(parsed.next_state) and parsed.next_state:
            logger.warning("LLM suggested unknown state '%s'; keeping current.", parsed.next_state)
            parsed.next_state = ""
        return parsed
    except Exception as exc:
        logger.warning("Structured LLM response failed validation: %s", exc)
        return None


async def parse_with_repair(producer) -> tuple[LLMStructuredResponse | None, str | None]:
    """Try parsing the raw output; retry once with a repair instruction.

    producer is an async callable that accepts a bool `repair` flag and returns
    the raw LLM text. Returns (parsed | None, error_message | None).
    """
    raw = await producer(repair=False)
    parsed = parse_structured_response(raw)
    if parsed is not None:
        return parsed, None

    logger.warning("First structured response unusable; sending repair instruction.")
    raw2 = await producer(repair=True)
    parsed = parse_structured_response(raw2)
    if parsed is not None:
        return parsed, None

    return None, "The assistant did not return a valid structured response."


SAFE_FALLBACK_MESSAGE = (
    "Sorry, I missed that. Could you repeat that in a few words, please?"
)
