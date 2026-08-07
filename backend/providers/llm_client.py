"""LLM client for the Sarvam Lead Agent.

Two providers are supported via LLM_PROVIDER:
  - "sarvam"            -> POST https://api.sarvam.ai/v1/chat/completions (model sarvam-105b)
  - "openai-compatible" -> any OpenAI-compatible chat endpoint (e.g. Ollama at LLM_BASE_URL)

Both send a single structured JSON object to the conversation engine.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field

import httpx

from backend.config import Settings
from backend.errors import LlmError, ProviderUnavailableError
from backend.schemas import ProviderStatus
from backend.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 2.0)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class LlmStreamEvent:
    """One normalized OpenAI-compatible streaming completion event."""

    type: str
    delta: str = ""
    first_token_latency_ms: int | None = None
    completion_latency_ms: int | None = None
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


class LlmClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(settings.llm_timeout))
        self._last_attempt_count: ContextVar[int] = ContextVar(
            f"llm_attempt_count_{id(self)}", default=1
        )
        self._last_first_token_latency_ms: ContextVar[int | None] = ContextVar(
            f"llm_first_token_latency_{id(self)}", default=None
        )
        self._last_stream_usage: ContextVar[dict] = ContextVar(
            f"llm_stream_usage_{id(self)}", default={}
        )
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

    @property
    def last_attempt_count(self) -> int:
        """Attempts made by the latest request in the current async task."""
        return self._last_attempt_count.get()

    @property
    def last_first_token_latency_ms(self) -> int | None:
        return self._last_first_token_latency_ms.get()

    @property
    def last_stream_usage(self) -> dict:
        return dict(self._last_stream_usage.get())

    def _chat_url(self) -> str:
        if self._settings.llm_provider == "sarvam":
            base = self._settings.sarvam_base_url.rstrip("/")
        else:
            base = self._settings.llm_base_url.rstrip("/")
        return f"{base}/v1/chat/completions"

    def _chat_body(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> dict:
        body: dict = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": getattr(self._settings, "llm_temperature", 0.55),
        }
        reasoning = (
            self._settings.llm_reasoning_effort
            if reasoning_effort is None
            else reasoning_effort
        )
        reasoning = (reasoning or "").strip().lower()
        if reasoning in {"none", "off", "disabled", "false"}:
            body["reasoning_effort"] = None
        elif reasoning:
            body["reasoning_effort"] = reasoning
        token_limit = (
            max_tokens
            if max_tokens is not None
            else self._settings.llm_max_tokens
        )
        if token_limit:
            body["max_tokens"] = token_limit
        if self._settings.llm_use_json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
            # Sarvam includes a final usage-only SSE event by default and its
            # documented API doesn't accept OpenAI's stream_options extension.
            if self._settings.llm_provider != "sarvam":
                body["stream_options"] = {"include_usage": True}
        return body

    async def _chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        reasoning_effort: str | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        retries = MAX_RETRIES if max_retries is None else max(0, max_retries)
        attempts = 1 + retries
        body = self._chat_body(
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        url = self._chat_url()
        for attempt in range(attempts):
            try:
                resp = await self._http.post(url, json=body, headers=self._headers)
                if resp.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                    logger.info(
                        "provider_retry provider=%s operation=llm attempt=%s status=%s",
                        self._settings.llm_provider,
                        attempt + 1,
                        resp.status_code,
                    )
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"LLM returned {resp.status_code}: {resp.text[:500]}",
                        request=resp.request,
                        response=resp,
                    )
                self._last_attempt_count.set(attempt + 1)
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    logger.info(
                        "provider_retry provider=%s operation=llm attempt=%s error=%s",
                        self._settings.llm_provider,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                break
        self._last_attempt_count.set(attempts)
        raise ProviderUnavailableError(
            "The LLM is not reachable. Check LLM_PROVIDER, LLM_BASE_URL and the API key.",
            details=str(last_error) if last_error else None,
        )

    @staticmethod
    async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
        """Parse SSE records, including records split across multiple data lines."""
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if field == "data":
                data_lines.append(value[1:] if value.startswith(" ") else value)
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _delta_text(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("delta") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        return ""

    async def stream_generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LlmStreamEvent]:
        """Yield normalized deltas from an OpenAI-compatible SSE response.

        Retries are allowed only before any model text has been emitted. Once a
        token is visible to a caller, silently replaying the request would
        duplicate speech, so subsequent failures are surfaced immediately.
        """
        started = time.monotonic()
        retries = MAX_RETRIES if max_retries is None else max(0, max_retries)
        attempts = 1 + retries
        body = self._chat_body(
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        first_token_ms: int | None = None
        usage: dict = {}
        finish_reason: str | None = None
        last_error: Exception | None = None
        self._last_first_token_latency_ms.set(None)
        self._last_stream_usage.set({})

        for attempt in range(attempts):
            emitted = False
            try:
                async with self._http.stream(
                    "POST", self._chat_url(), json=body, headers=self._headers
                ) as response:
                    if response.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                        await response.aread()
                        logger.info(
                            "provider_retry provider=%s operation=llm_stream attempt=%s status=%s",
                            self._settings.llm_provider,
                            attempt + 1,
                            response.status_code,
                        )
                        await asyncio.sleep(
                            BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                        )
                        continue
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")
                        raise httpx.HTTPStatusError(
                            f"LLM returned {response.status_code}: {detail[:500]}",
                            request=response.request,
                            response=response,
                        )

                    self._last_attempt_count.set(attempt + 1)
                    async for data in self._sse_data(response):
                        if data.strip() == "[DONE]":
                            break
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise LlmError(
                                "The LLM returned an invalid streaming event.",
                                details=str(exc)[:300],
                            ) from exc
                        if payload.get("error"):
                            raise LlmError(
                                "The LLM streaming request failed.",
                                details=str(payload["error"])[:500],
                            )
                        if payload.get("usage"):
                            usage = payload["usage"]
                        choices = payload.get("choices") or []
                        if choices and choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                        delta = self._delta_text(payload)
                        if not delta:
                            continue
                        if first_token_ms is None:
                            first_token_ms = int(
                                (time.monotonic() - started) * 1000
                            )
                            self._last_first_token_latency_ms.set(first_token_ms)
                        emitted = True
                        yield LlmStreamEvent(
                            type="delta",
                            delta=delta,
                            first_token_latency_ms=first_token_ms,
                        )

                completion_ms = int((time.monotonic() - started) * 1000)
                self._last_stream_usage.set(dict(usage))
                yield LlmStreamEvent(
                    type="done",
                    first_token_latency_ms=first_token_ms,
                    completion_latency_ms=completion_ms,
                    usage=dict(usage),
                    finish_reason=finish_reason,
                )
                return
            except asyncio.CancelledError:
                logger.info(
                    "llm_stream_cancelled provider=%s emitted=%s",
                    self._settings.llm_provider,
                    emitted,
                )
                raise
            except LlmError:
                raise
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = exc
                if not emitted and attempt < attempts - 1:
                    logger.info(
                        "provider_retry provider=%s operation=llm_stream attempt=%s error=%s",
                        self._settings.llm_provider,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(
                        BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                    )
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                break

        self._last_attempt_count.set(attempts)
        raise ProviderUnavailableError(
            "The LLM stream is not reachable. Check LLM_PROVIDER, LLM_BASE_URL and the API key.",
            details=str(last_error)[:500] if last_error else None,
        )

    async def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, int, dict]:
        started = time.monotonic()
        resp = await self._chat(
            messages,
            max_tokens=max_tokens,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
        )
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
