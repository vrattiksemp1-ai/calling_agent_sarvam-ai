"""LLM client for the Sarvam Lead Agent.

Two providers are supported via LLM_PROVIDER:
  - "sarvam"            -> POST https://api.sarvam.ai/v1/chat/completions (model sarvam-105b)
  - "openai-compatible" -> any OpenAI-compatible chat endpoint (e.g. Ollama at LLM_BASE_URL)

Both send a single structured JSON object to the conversation engine.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from backend.config import Settings
from backend.errors import LlmError, ProviderUnavailableError
from backend.schemas import ProviderStatus
from backend.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 2.0)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LlmClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(settings.llm_timeout))
        self._headers = {}
        if settings.llm_provider == "sarvam":
            if settings.llm_api_key:
                self._headers["api-subscription-key"] = settings.llm_api_key
                self._headers["Authorization"] = f"Bearer {settings.llm_api_key}"
            elif settings.sarvam_api_key:
                self._headers["api-subscription-key"] = settings.sarvam_api_key
                self._headers["Authorization"] = f"Bearer {settings.sarvam_api_key}"
        elif settings.llm_api_key:
            self._headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    @property
    def model(self) -> str:
        return self._settings.llm_model

    def _chat_url(self) -> str:
        if self._settings.llm_provider == "sarvam":
            base = self._settings.sarvam_base_url.rstrip("/")
        else:
            base = self._settings.llm_base_url.rstrip("/")
        return f"{base}/v1/chat/completions"

    async def _chat(self, messages: list[dict]) -> httpx.Response:
        last_error: Exception | None = None
        attempts = 1 + MAX_RETRIES
        body: dict = {"model": self._settings.llm_model, "messages": messages, "temperature": 0.2}
        reasoning = (self._settings.llm_reasoning_effort or "").strip().lower()
        if reasoning in {"none", "off", "disabled", "false"}:
            body["reasoning_effort"] = None
        elif reasoning:
            body["reasoning_effort"] = reasoning
        if self._settings.llm_max_tokens:
            body["max_tokens"] = self._settings.llm_max_tokens
        if self._settings.llm_use_json_mode:
            body["response_format"] = {"type": "json_object"}
        url = self._chat_url()
        for attempt in range(attempts):
            try:
                resp = await self._http.post(url, json=body, headers=self._headers)
                if resp.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"LLM returned {resp.status_code}: {resp.text[:500]}",
                        request=resp.request,
                        response=resp,
                    )
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                break
        raise ProviderUnavailableError(
            "The LLM is not reachable. Check LLM_PROVIDER, LLM_BASE_URL and the API key.",
            details=str(last_error) if last_error else None,
        )

    async def generate(self, messages: list[dict]) -> tuple[str, int, dict]:
        started = time.monotonic()
        resp = await self._chat(messages)
        payload = resp.json()
        try:
            content = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                "The LLM response did not contain assistant content.",
                details=str(exc)[:300],
            )
        usage = payload.get("usage") or {}
        latency_ms = int((time.monotonic() - started) * 1000)
        return content, latency_ms, usage

    async def health_check(self) -> ProviderStatus:
        started = time.monotonic()
        try:
            resp = await self._http.get(
                f"{self._settings.llm_base_url.rstrip('/')}/v1/models",
                headers=self._headers,
                timeout=httpx.Timeout(15.0),
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            if resp.status_code < 500:
                return ProviderStatus(
                    provider="llm",
                    status="ok",
                    message=f"LLM endpoint reachable (model: {self._settings.llm_model}).",
                    latency_ms=latency_ms,
                    details={"provider": self._settings.llm_provider, "model": self._settings.llm_model},
                )
            return ProviderStatus(
                provider="llm",
                status="degraded",
                message=f"LLM responded with HTTP {resp.status_code}.",
                latency_ms=latency_ms,
                details={"provider": self._settings.llm_provider},
            )
        except httpx.HTTPError as exc:
            return ProviderStatus(
                provider="llm",
                status="error",
                message="LLM endpoint is not reachable. Check LLM_PROVIDER, LLM_BASE_URL and the API key.",
                latency_ms=int((time.monotonic() - started) * 1000),
                details={"provider": self._settings.llm_provider, "error": str(exc)[:300]},
            )

    async def aclose(self) -> None:
        if self._own_client:
            await self._http.aclose()
