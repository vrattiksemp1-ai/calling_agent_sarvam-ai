"""Structured JSON response parsing from the LLM.

The LLM must return exactly one JSON object. We parse leniently (strip code
fences), validate with Pydantic, and support a single repair retry before
falling back to a safe message.
"""

import json
import re

from pydantic import BaseModel, Field, field_validator

from backend.models import LEAD_FIELDS
from backend.state_machine import is_valid_state
from backend.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_ASSISTANT_MESSAGE_RE = re.compile(
    r'"assistant_message"\s*:\s*"((?:\\.|[^"\\])*)"',
    re.DOTALL,
)
_TOP_LEVEL_KEYS_IN_FIELDS = {
    "assistant_message",
    "detected_language",
    "extracted_fields",
    "fields_to_clear",
    "next_state",
    "conversation_complete",
    "needs_confirmation",
}
_ALLOWED_LEAD_FIELDS = set(LEAD_FIELDS)


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
            candidate = _salvage_truncated_object(cleaned)
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


def _json_depths(fragment: str) -> tuple[int, int, bool]:
    """Return (object_depth, array_depth, in_string) at end of fragment."""
    obj_depth = 0
    arr_depth = 0
    in_string = False
    escape = False
    for ch in fragment:
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
            obj_depth += 1
        elif ch == "}":
            obj_depth = max(0, obj_depth - 1)
        elif ch == "[":
            arr_depth += 1
        elif ch == "]":
            arr_depth = max(0, arr_depth - 1)
    return obj_depth, arr_depth, in_string


def _strip_incomplete_json_tail(fragment: str) -> str:
    """Drop a trailing incomplete object member so the rest can parse."""
    text = fragment.rstrip().rstrip(",")

    # Key with colon but no value yet: ..., "team_size":
    m = re.search(r',\s*"[^"]*"\s*:\s*$', text)
    if m:
        return text[: m.start()].rstrip().rstrip(",")

    # Trailing quoted token: either a complete string value or a bare key.
    m = re.search(r'(,\s*)?("[^"]*")\s*$', text)
    if m:
        before_quote = text[: m.start(2)].rstrip()
        if before_quote.endswith(":"):
            # Complete "key": "value" — keep it.
            return text
        # Bare key (optionally preceded by a comma) — drop it.
        cut = m.start(1) if m.group(1) is not None else m.start(2)
        return text[:cut].rstrip().rstrip(",")

    # First member of an object incomplete after `{`
    m = re.search(r'\{\s*"[^"]*"\s*:?\s*$', text)
    if m:
        return text[: m.start() + 1]
    return text


def _salvage_truncated_object(text: str) -> str | None:
    """Close a truncated JSON object that still has assistant_message.

    Phone turns sometimes hit max_tokens mid-object (often inside
    extracted_fields). Close nested structures first so top-level keys are not
    accidentally nested, then finish the root object.
    """
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:].rstrip()
    if '"assistant_message"' not in fragment:
        return None

    obj_depth, arr_depth, in_string = _json_depths(fragment)
    if in_string:
        fragment += '"'
        obj_depth, arr_depth, in_string = _json_depths(fragment)

    fragment = _strip_incomplete_json_tail(fragment)
    obj_depth, arr_depth, _ = _json_depths(fragment)

    # Close nested arrays/objects until only the root object remains open.
    while arr_depth > 0:
        fragment += "]"
        arr_depth -= 1
    while obj_depth > 1:
        fragment += "}"
        obj_depth -= 1

    if obj_depth != 1:
        return None

    lower = fragment.lower()
    if '"conversation_complete"' not in lower:
        fragment += ', "conversation_complete": false'
    if '"needs_confirmation"' not in lower:
        fragment += ', "needs_confirmation": false'
    if '"extracted_fields"' not in fragment:
        fragment += ', "extracted_fields": {}'
    if '"fields_to_clear"' not in fragment:
        fragment += ', "fields_to_clear": []'
    if '"next_state"' not in fragment:
        fragment += ', "next_state": ""'
    if '"detected_language"' not in fragment:
        fragment += ', "detected_language": "en"'

    fragment += "}"
    return fragment


def _coerce_field_value(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _normalize_structured_payload(data: dict) -> dict:
    """Coerce common LLM mistakes so Pydantic validation can succeed."""
    out = dict(data)

    extracted = out.get("extracted_fields")
    if not isinstance(extracted, dict):
        extracted = {}
    else:
        extracted = dict(extracted)

    # Mis-nested top-level keys sometimes land inside extracted_fields after
    # truncation salvage.
    for key in list(extracted.keys()):
        if key not in _TOP_LEVEL_KEYS_IN_FIELDS:
            continue
        if key not in out or out.get(key) in (None, "", {}, []):
            out[key] = extracted.pop(key)
        else:
            extracted.pop(key, None)

    cleaned: dict[str, str] = {}
    for key, value in extracted.items():
        if key not in _ALLOWED_LEAD_FIELDS:
            continue
        coerced = _coerce_field_value(value)
        if coerced is None:
            continue
        cleaned[key] = coerced
    out["extracted_fields"] = cleaned

    fields_to_clear = out.get("fields_to_clear")
    if isinstance(fields_to_clear, list):
        out["fields_to_clear"] = [
            str(item) for item in fields_to_clear if str(item or "").strip()
        ]
    else:
        out["fields_to_clear"] = []

    next_state = out.get("next_state")
    if next_state is None:
        out["next_state"] = ""
    elif not isinstance(next_state, str):
        out["next_state"] = str(next_state)

    lang = out.get("detected_language")
    if not isinstance(lang, str) or not lang.strip():
        out["detected_language"] = "en"

    for bool_key in ("conversation_complete", "needs_confirmation"):
        value = out.get(bool_key, False)
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            out[bool_key] = value.strip().lower() in {"1", "true", "yes"}
        else:
            out[bool_key] = bool(value)

    return out


def _message_only_fallback(text: str) -> LLMStructuredResponse | None:
    """Last resort: keep the spoken line if JSON is otherwise unusable."""
    match = _ASSISTANT_MESSAGE_RE.search(text or "")
    if not match:
        return None
    try:
        message = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        message = match.group(1)
    message = (message or "").strip()
    if not message:
        return None
    logger.warning("Using assistant_message-only fallback from broken JSON")
    return LLMStructuredResponse(assistant_message=message)


def parse_structured_response(text: str) -> LLMStructuredResponse | None:
    """Return a validated response, or None when the text is not usable."""
    data = extract_json(text)
    if data:
        data = _normalize_structured_payload(data)
        try:
            parsed = LLMStructuredResponse.model_validate(data)
            if not is_valid_state(parsed.next_state) and parsed.next_state:
                logger.warning(
                    "LLM suggested unknown state '%s'; keeping current.",
                    parsed.next_state,
                )
                parsed.next_state = ""
            return parsed
        except Exception as exc:
            logger.warning("Structured LLM response failed validation: %s", exc)

    return _message_only_fallback(text)


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
