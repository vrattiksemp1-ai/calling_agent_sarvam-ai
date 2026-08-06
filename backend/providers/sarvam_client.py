"""Sarvam cloud provider client (STT + TTS).

Endpoints (verified against https://docs.sarvam.ai):
  POST https://api.sarvam.ai/speech-to-text      STT (model saaras:v3, multipart)
  POST https://api.sarvam.ai/text-to-speech      TTS (model bulbul:v3, base64 WAV)
  POST https://api.sarvam.ai/text-to-speech/stream  TTS streaming (WAV chunks)
  GET  https://api.sarvam.ai/                   health/reachability probe

Authentication: the `api-subscription-key` header is required for every call.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator

import httpx

from backend.config import Settings, tts_language_code_for
from backend.errors import ProviderUnavailableError, SttError, TtsError
from backend.schemas import ProviderStatus
from backend.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 2.0)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class SarvamClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.sarvam_request_timeout)
        )
        self._headers = {}
        if settings.sarvam_api_key:
            self._headers["api-subscription-key"] = settings.sarvam_api_key

    @property
    def api_key_set(self) -> bool:
        return bool(self._settings.sarvam_api_key)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json: dict | None = None,
        files: dict | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        attempts = 1 + MAX_RETRIES
        for attempt in range(attempts):
            try:
                resp = await self._http.request(
                    method, url, json=json, files=files, headers=self._headers
                )
                if resp.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                if resp.status_code >= 400:
                    detail = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"Sarvam returned {resp.status_code}: {detail}",
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
            "Sarvam is not reachable or returned an error. "
            "Check SARVAM_API_KEY and your internet connection.",
            details=str(last_error) if last_error else None,
        )

    async def transcribe(self, audio_path, duration_ms: int) -> tuple[str, int]:
        started = time.monotonic()
        try:
            with open(audio_path, "rb") as fh:
                files = {
                    "file": ("audio.wav", fh, "audio/wav"),
                    "model": (None, self._settings.sarvam_stt_model),
                    "mode": (None, self._settings.sarvam_stt_mode),
                    "language_code": (None, self._settings.sarvam_language_code),
                }
                resp = await self._request_with_retry(
                    "POST",
                    f"{self._settings.sarvam_base_url.rstrip('/')}/speech-to-text",
                    files=files,
                )
            payload = resp.json()
            text = payload.get("transcript") or ""
            if not text.strip():
                raise SttError(
                    "Sarvam returned an empty transcript. Try speaking clearly and closer to the microphone."
                )
            latency_ms = int((time.monotonic() - started) * 1000)
            return text.strip(), latency_ms
        except ProviderUnavailableError:
            raise
        except SttError:
            raise
        except Exception as exc:
            raise SttError(
                "Speech-to-text failed. The audio may be unintelligible or the Sarvam quota is exhausted.",
                details=str(exc)[:500],
            )

    async def synthesize(self, text: str, detected_language: str | None = None) -> tuple[bytes, str, int]:
        started = time.monotonic()
        target_language = tts_language_code_for(
            detected_language, self._settings.sarvam_tts_language_code
        )
        try:
            body = {
                "text": text,
                "target_language_code": target_language,
                "speaker": self._settings.sarvam_tts_speaker,
                "model": self._settings.sarvam_tts_model,
                "output_audio_codec": "wav",
                "speech_sample_rate": str(self._settings.sarvam_tts_speech_sample_rate),
                "temperature": self._settings.sarvam_tts_temperature,
                "pace": self._settings.sarvam_tts_pace,
            }
            resp = await self._request_with_retry(
                "POST",
                f"{self._settings.sarvam_base_url.rstrip('/')}/text-to-speech",
                json=body,
            )
            payload = resp.json()
            audios = payload.get("audios") or []
            if not audios or not audios[0]:
                raise TtsError("Sarvam returned no audio for the TTS request.")
            audio_bytes = base64.b64decode(audios[0])
            latency_ms = int((time.monotonic() - started) * 1000)
            return audio_bytes, "audio/wav", latency_ms
        except ProviderUnavailableError as exc:
            raise TtsError(
                "Text-to-speech failed. Check SARVAM_API_KEY and the Sarvam service status.",
                details=str(exc.details or exc.message)[:500],
            )
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(
                "Text-to-speech failed. The reply is shown as text instead.",
                details=str(exc)[:500],
            )

    async def stream_synthesize(
        self, text: str, detected_language: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio from the /text-to-speech/stream endpoint.

        Yields WAV bytes as soon as chunks arrive, so the first audio can be
        handed to the caller before synthesis finishes. The request is a single
        HTTP POST; the response body is the audio stream itself. Raises
        TtsError if the request fails before any audio arrives.
        """
        target_language = tts_language_code_for(
            detected_language, self._settings.sarvam_tts_language_code
        )
        body = {
            "text": text,
            "language_code": target_language,
            "speaker": self._settings.sarvam_tts_speaker,
            "model": self._settings.sarvam_tts_model,
            "output_audio_codec": "wav",
            "speech_sample_rate": self._settings.sarvam_tts_speech_sample_rate,
            "temperature": self._settings.sarvam_tts_temperature,
            "pace": self._settings.sarvam_tts_pace,
        }
        url = f"{self._settings.sarvam_base_url.rstrip('/')}/text-to-speech/stream"
        try:
            async with self._http.stream(
                "POST", url, json=body, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread())[:500]
                    raise TtsError(
                        "Text-to-speech failed. Check SARVAM_API_KEY and the Sarvam service status.",
                        details=detail.decode("utf-8", "replace")[:500],
                    )
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(
                "Text-to-speech failed. The reply is shown as text instead.",
                details=str(exc)[:500],
            )

    async def health_check(self) -> ProviderStatus:
        started = time.monotonic()
        if not self.api_key_set:
            return ProviderStatus(
                provider="sarvam",
                status="degraded",
                message="SARVAM_API_KEY is not configured. Set it in .env to use voice mode.",
                latency_ms=int((time.monotonic() - started) * 1000),
                details={"base_url": self._settings.sarvam_base_url, "api_key_set": False},
            )
        try:
            resp = await self._http.get(
                f"{self._settings.sarvam_base_url.rstrip('/')}/",
                headers=self._headers,
                timeout=httpx.Timeout(min(self._settings.sarvam_request_timeout, 15.0)),
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            if resp.status_code < 500:
                return ProviderStatus(
                    provider="sarvam",
                    status="ok",
                    message="Sarvam is reachable and an API key is configured.",
                    latency_ms=latency_ms,
                    details={"base_url": self._settings.sarvam_base_url, "api_key_set": True},
                )
            return ProviderStatus(
                provider="sarvam",
                status="degraded",
                message=f"Sarvam responded with HTTP {resp.status_code}.",
                latency_ms=latency_ms,
                details={"base_url": self._settings.sarvam_base_url, "api_key_set": True},
            )
        except httpx.HTTPError as exc:
            return ProviderStatus(
                provider="sarvam",
                status="error",
                message="Sarvam is not reachable. Check your internet connection and SARVAM_BASE_URL.",
                latency_ms=int((time.monotonic() - started) * 1000),
                details={"base_url": self._settings.sarvam_base_url, "api_key_set": True, "error": str(exc)[:300]},
            )

    async def aclose(self) -> None:
        if self._own_client:
            await self._http.aclose()
