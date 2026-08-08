"""End-to-end pipeline tracing for voice turns.

Logs listen → transcript → LLM → TTS → play with timings and I/O previews.
Phone numbers / emails are still masked by the global PiiFilter.
API keys and auth headers are never logged.
"""

from __future__ import annotations

import json
import time
from typing import Any

from backend.utils.logging import get_logger

logger = get_logger("backend.pipeline")

_ENABLED = True
_MAX_CHARS = 2000

_SECRET_KEYS = {
    "authorization",
    "api-subscription-key",
    "api_key",
    "apikey",
    "auth_token",
    "password",
    "secret",
    "token",
    "x-api-key",
}


def configure_pipeline_trace(*, enabled: bool = True, max_chars: int = 2000) -> None:
    global _ENABLED, _MAX_CHARS
    _ENABLED = bool(enabled)
    _MAX_CHARS = max(200, int(max_chars or 2000))


def is_enabled() -> bool:
    return _ENABLED


def clip(value: Any, max_chars: int | None = None) -> Any:
    """Truncate strings / nested payloads for readable logs."""
    limit = _MAX_CHARS if max_chars is None else max_chars
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"...<truncated:{len(value) - limit} chars>"
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_s = str(key)
            if key_s.lower() in _SECRET_KEYS or "token" in key_s.lower() or "key" in key_s.lower():
                out[key_s] = "[REDACTED]"
            else:
                out[key_s] = clip(item, limit)
        return out
    if isinstance(value, (list, tuple)):
        items = [clip(v, limit) for v in value[:40]]
        if len(value) > 40:
            items.append(f"...<truncated:{len(value) - 40} items>")
        return items
    text = str(value)
    return clip(text, limit)


def summarize_messages(messages: list[dict] | None) -> dict[str, Any]:
    """Compact view of chat messages (roles + clipped content)."""
    rows = []
    total_chars = 0
    for msg in messages or []:
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        total_chars += len(content)
        rows.append(
            {
                "role": msg.get("role"),
                "chars": len(content),
                "content": clip(content, min(800, _MAX_CHARS)),
            }
        )
    return {"count": len(rows), "total_chars": total_chars, "messages": rows}


def trace(event: str, **fields: Any) -> None:
    """Emit one structured pipeline event."""
    if not _ENABLED:
        return
    payload = {"event": event, **{k: clip(v) for k, v in fields.items() if v is not None}}
    try:
        logger.info("pipeline=%s", json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 - logging must never break the call
        logger.info("pipeline event=%s fields=%s", event, list(fields.keys()))


class StageTimer:
    """Context helper for timed pipeline stages."""

    def __init__(self, event: str, **fields: Any):
        self.event = event
        self.fields = fields
        self._started = 0.0

    def __enter__(self) -> "StageTimer":
        self._started = time.monotonic()
        trace(f"{self.event}.start", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = int((time.monotonic() - self._started) * 1000)
        extra = dict(self.fields)
        extra["elapsed_ms"] = elapsed_ms
        if exc is not None:
            extra["error"] = f"{exc_type.__name__}: {exc}"
            trace(f"{self.event}.error", **extra)
        else:
            trace(f"{self.event}.end", **extra)
