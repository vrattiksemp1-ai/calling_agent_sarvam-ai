"""Per-conversation speech provider selection (Sarvam or Bhashini).

``SpeechRouter`` exposes the same interface as ``SarvamClient`` and forwards
each call to the provider selected for the current conversation or call. The
selection lives in a ContextVar, so concurrent requests/calls do not interfere:
asyncio tasks copy the context when they are created, which means work spawned
after ``activate()`` keeps using the chosen provider.

Code that needs a fixed client for a whole call (``CallSession``) can call
``get(name)`` instead and hold the concrete client.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from backend.config import SPEECH_PROVIDERS

_current_provider: ContextVar[str | None] = ContextVar("speech_provider", default=None)


class UnknownSpeechProviderError(ValueError):
    pass


def normalize_speech_provider(value: str | None, default: str = "sarvam") -> str:
    name = (value or "").strip().lower() or default
    if name not in SPEECH_PROVIDERS:
        raise UnknownSpeechProviderError(
            f"Unknown speech provider '{value}'. Use one of: {', '.join(SPEECH_PROVIDERS)}."
        )
    return name


def provider_name_of(client: Any, fallback: str = "sarvam") -> str:
    return getattr(client, "provider_name", None) or fallback


async def transcribe_with_hint(client: Any, audio_path, duration_ms: int, language_hint: str | None):
    """Call ``transcribe`` passing the language only to providers that need it."""
    if language_hint and getattr(client, "accepts_language_hint", False):
        return await client.transcribe(audio_path, duration_ms, language_hint=language_hint)
    return await client.transcribe(audio_path, duration_ms)


class SpeechRouter:
    def __init__(self, clients: dict[str, Any], default: str = "sarvam") -> None:
        if default not in clients:
            raise ValueError(f"Default speech provider '{default}' has no client")
        self._clients = dict(clients)
        self._default = default

    # ---------- selection ----------

    @property
    def default(self) -> str:
        return self._default

    @property
    def names(self) -> list[str]:
        return list(self._clients)

    @property
    def current_name(self) -> str:
        name = _current_provider.get()
        return name if name in self._clients else self._default

    @property
    def active(self) -> Any:
        return self._clients[self.current_name]

    def get(self, name: str | None) -> Any:
        """Concrete client for ``name`` (default provider when empty/unknown)."""
        return self._clients.get((name or "").strip().lower(), self._clients[self._default])

    def activate(self, name: str | None) -> None:
        """Select a provider for the rest of the current task (and its children)."""
        _current_provider.set(name if name in self._clients else self._default)

    @contextmanager
    def use(self, name: str | None) -> Iterator[Any]:
        token = _current_provider.set(name if name in self._clients else self._default)
        try:
            yield self.active
        finally:
            _current_provider.reset(token)

    # ---------- SarvamClient-compatible surface ----------

    @property
    def provider_name(self) -> str:
        return provider_name_of(self.active, self.current_name)

    @property
    def supports_realtime_stt(self) -> bool:
        return getattr(self.active, "supports_realtime_stt", True)

    @property
    def supports_realtime_tts(self) -> bool:
        return getattr(self.active, "supports_realtime_tts", True)

    @property
    def supports_streaming_tts(self) -> bool:
        return getattr(self.active, "supports_streaming_tts", True)

    @property
    def accepts_language_hint(self) -> bool:
        return getattr(self.active, "accepts_language_hint", False)

    async def transcribe(self, audio_path, duration_ms: int, language_hint: str | None = None):
        return await transcribe_with_hint(self.active, audio_path, duration_ms, language_hint)

    def __getattr__(self, item: str) -> Any:
        # synthesize, stream_synthesize, open_realtime_*, last_attempt_count,
        # api_key_set, health_check ... all forward to the active client.
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self.active, item)

    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
