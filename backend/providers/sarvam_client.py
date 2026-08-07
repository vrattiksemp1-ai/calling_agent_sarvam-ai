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
import inspect
import json
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

from backend.config import Settings, tts_language_code_for
from backend.errors import ProviderUnavailableError, SttError, TtsError
from backend.language_utils import map_stt_language_code
from backend.schemas import ProviderStatus
from backend.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 2.0)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


async def _websocket_connect(url: str, headers: dict):
    from websockets.asyncio.client import connect

    return await connect(
        url,
        additional_headers=headers,
        ping_interval=None,
        max_size=None,
        open_timeout=15,
    )


async def _open_websocket(factory, url: str, headers: dict):
    result = factory(url, headers)
    return await result if inspect.isawaitable(result) else result


@dataclass(frozen=True)
class SarvamSttEvent:
    type: str
    transcript: str = ""
    language_code: str | None = None
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SarvamTtsEvent:
    type: str
    audio: bytes = b""
    payload: dict = field(default_factory=dict)


class SarvamRealtimeSttSession:
    """One persistent Saaras v3 realtime transcription socket."""

    def __init__(self, settings: Settings, connect_factory) -> None:
        self._settings = settings
        self._connect_factory = connect_factory
        self._ws = None
        self._keepalive_task: asyncio.Task | None = None

    async def connect(self) -> "SarvamRealtimeSttSession":
        query = urlencode(
            {
                "model": self._settings.sarvam_realtime_stt_model,
                "language_code": "auto",
                "stream_type": "fast",
                "mode": "codemix",
                "endpointing": "vad",
                "encoding": "linear16",
                "sample_rate": 8000,
                "threshold": self._settings.sarvam_realtime_stt_vad_threshold,
                "silence_duration_ms": self._settings.sarvam_realtime_stt_silence_ms,
                "min_speech_duration_ms": self._settings.sarvam_realtime_stt_min_speech_ms,
            }
        )
        separator = "&" if "?" in self._settings.sarvam_realtime_stt_url else "?"
        url = f"{self._settings.sarvam_realtime_stt_url}{separator}{query}"
        headers = {"api-subscription-key": self._settings.sarvam_api_key}
        self._ws = await _open_websocket(self._connect_factory, url, headers)
        await self.update_config()
        self._keepalive_task = asyncio.create_task(
            self._keepalive(), name="sarvam-stt-keepalive"
        )
        return self

    async def _keepalive(self) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                await self.ping()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Sarvam realtime STT keepalive stopped", exc_info=True)

    async def update_config(
        self,
        *,
        threshold: float | None = None,
        silence_ms: int | None = None,
        min_speech_ms: int | None = None,
    ) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime STT session is not connected")
        await self._ws.send(
            json.dumps(
                {
                    "event": "config.update",
                    "threshold": (
                        self._settings.sarvam_realtime_stt_vad_threshold
                        if threshold is None
                        else threshold
                    ),
                    "silence_duration_ms": (
                        self._settings.sarvam_realtime_stt_silence_ms
                        if silence_ms is None
                        else silence_ms
                    ),
                    "min_speech_duration_ms": (
                        self._settings.sarvam_realtime_stt_min_speech_ms
                        if min_speech_ms is None
                        else min_speech_ms
                    ),
                }
            )
        )

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime STT session is not connected")
        await self._ws.send(
            json.dumps(
                {
                    "event": "audio_input",
                    "audio": base64.b64encode(pcm16).decode("ascii"),
                }
            )
        )

    async def ping(self) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime STT session is not connected")
        await self._ws.send(json.dumps({"event": "ping"}))

    @staticmethod
    def _normalize(payload: dict) -> SarvamSttEvent:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        raw_type = str(payload.get("type") or payload.get("event") or "")
        aliases = {
            "session_started": "session.begin",
            "speech_start": "vad.speech_start",
            "speech_started": "vad.speech_start",
            "speech_end": "vad.speech_end",
            "speech_stopped": "vad.speech_end",
            "partial": "transcript.partial",
            "final": "transcript.final",
        }
        event_type = aliases.get(raw_type, raw_type)
        if not event_type and data.get("transcript") is not None:
            event_type = (
                "transcript.final"
                if data.get("is_final") or data.get("final")
                else "transcript.partial"
            )
        return SarvamSttEvent(
            type=event_type or "unknown",
            transcript=str(data.get("transcript") or data.get("text") or ""),
            language_code=data.get("language_code") or data.get("language"),
            payload=payload,
        )

    async def receive(self) -> SarvamSttEvent:
        if self._ws is None:
            raise RuntimeError("Realtime STT session is not connected")
        message = await self._ws.recv()
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        return self._normalize(json.loads(message))

    def __aiter__(self) -> "SarvamRealtimeSttSession":
        return self

    async def __anext__(self) -> SarvamSttEvent:
        try:
            return await self.receive()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            if self._ws is None:
                raise StopAsyncIteration from exc
            raise

    async def close(self) -> None:
        keepalive, self._keepalive_task = self._keepalive_task, None
        if keepalive is not None:
            keepalive.cancel()
            await asyncio.gather(keepalive, return_exceptions=True)
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.send(json.dumps({"event": "end"}))
            except Exception:
                pass
            await ws.close()


class SarvamRealtimeTtsSession:
    """Bulbul v3 socket configured for one language and output codec."""

    def __init__(
        self,
        settings: Settings,
        connect_factory,
        language_code: str,
        codec: str,
    ) -> None:
        self._settings = settings
        self._connect_factory = connect_factory
        self.language_code = language_code
        self.codec = codec
        self._ws = None
        self._keepalive_task: asyncio.Task | None = None

    async def connect(self) -> "SarvamRealtimeTtsSession":
        headers = {"api-subscription-key": self._settings.sarvam_api_key}
        query = urlencode(
            {
                "model": self._settings.sarvam_tts_model,
                "send_completion_event": "true",
            }
        )
        separator = "&" if "?" in self._settings.sarvam_realtime_tts_url else "?"
        url = f"{self._settings.sarvam_realtime_tts_url}{separator}{query}"
        self._ws = await _open_websocket(
            self._connect_factory,
            url,
            headers,
        )
        await self._ws.send(
            json.dumps(
                {
                    "type": "config",
                    "data": {
                        "language_code": self.language_code,
                        "speaker": self._settings.sarvam_tts_speaker,
                        "model": self._settings.sarvam_tts_model,
                        "output_audio_codec": self.codec,
                        "speech_sample_rate": 8000,
                        "temperature": self._settings.sarvam_tts_temperature,
                        "pace": self._settings.sarvam_tts_pace,
                    },
                }
            )
        )
        self._keepalive_task = asyncio.create_task(
            self._keepalive(), name="sarvam-tts-keepalive"
        )
        return self

    async def _keepalive(self) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                await self.ping()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Sarvam realtime TTS keepalive stopped", exc_info=True)

    async def send_text(self, text: str) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime TTS session is not connected")
        if text:
            await self._ws.send(json.dumps({"type": "text", "data": {"text": text}}))

    async def flush(self) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime TTS session is not connected")
        await self._ws.send(json.dumps({"type": "flush"}))

    async def ping(self) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime TTS session is not connected")
        await self._ws.send(json.dumps({"type": "ping"}))

    async def receive(self) -> SarvamTtsEvent:
        if self._ws is None:
            raise RuntimeError("Realtime TTS session is not connected")
        message = await self._ws.recv()
        if isinstance(message, bytes):
            return SarvamTtsEvent(type="audio", audio=message)
        payload = json.loads(message)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        raw_type = str(payload.get("type") or payload.get("event") or "")
        if raw_type in {"audio", "audio_chunk", "data"} and (
            data.get("audio") or data.get("audio_data") or data.get("chunk")
        ):
            encoded = data.get("audio") or data.get("audio_data") or data.get("chunk")
            return SarvamTtsEvent(
                type="audio", audio=base64.b64decode(encoded), payload=payload
            )
        if raw_type in {"complete", "completed", "done", "flush.done"} or (
            raw_type == "event" and data.get("event_type") == "final"
        ):
            return SarvamTtsEvent(type="complete", payload=payload)
        if raw_type == "error" or payload.get("error"):
            return SarvamTtsEvent(type="error", payload=payload)
        return SarvamTtsEvent(type=raw_type or "unknown", payload=payload)

    async def close(self) -> None:
        keepalive, self._keepalive_task = self._keepalive_task, None
        if keepalive is not None:
            keepalive.cancel()
            await asyncio.gather(keepalive, return_exceptions=True)
        ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()


class SarvamClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        *,
        stt_connect_factory=None,
        tts_connect_factory=None,
    ):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.sarvam_request_timeout)
        )
        self._last_attempt_count: ContextVar[int] = ContextVar(
            f"sarvam_attempt_count_{id(self)}", default=1
        )
        self._headers = {}
        if settings.sarvam_api_key:
            self._headers["api-subscription-key"] = settings.sarvam_api_key
        self._stt_connect_factory = stt_connect_factory or _websocket_connect
        self._tts_connect_factory = tts_connect_factory or _websocket_connect

    @property
    def api_key_set(self) -> bool:
        return bool(self._settings.sarvam_api_key)

    @property
    def last_attempt_count(self) -> int:
        """Attempts made by the latest request in the current async task."""
        return self._last_attempt_count.get()

    async def open_realtime_stt(self) -> SarvamRealtimeSttSession:
        return await SarvamRealtimeSttSession(
            self._settings, self._stt_connect_factory
        ).connect()

    async def open_realtime_tts(
        self,
        detected_language: str | None = None,
        *,
        codec: str | None = None,
    ) -> SarvamRealtimeTtsSession:
        language = tts_language_code_for(
            detected_language, self._settings.sarvam_tts_language_code
        )
        output_codec = codec or self._settings.sarvam_realtime_tts_codec
        if output_codec not in {"linear16", "mulaw"}:
            raise ValueError("Realtime TTS codec must be linear16 or mulaw")
        return await SarvamRealtimeTtsSession(
            self._settings,
            self._tts_connect_factory,
            language,
            output_codec,
        ).connect()

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
                    logger.info(
                        "provider_retry provider=sarvam operation=%s attempt=%s status=%s",
                        url.rsplit("/", 1)[-1],
                        attempt + 1,
                        resp.status_code,
                    )
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                if resp.status_code >= 400:
                    detail = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"Sarvam returned {resp.status_code}: {detail}",
                        request=resp.request,
                        response=resp,
                    )
                self._last_attempt_count.set(attempt + 1)
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    logger.info(
                        "provider_retry provider=sarvam operation=%s attempt=%s error=%s",
                        url.rsplit("/", 1)[-1],
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
            "Sarvam is not reachable or returned an error. "
            "Check SARVAM_API_KEY and your internet connection.",
            details=str(last_error) if last_error else None,
        )

    async def transcribe(
        self, audio_path, duration_ms: int
    ) -> tuple[str, int, str | None]:
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
            detected_language = map_stt_language_code(payload.get("language_code"))
            return text.strip(), latency_ms, detected_language
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
