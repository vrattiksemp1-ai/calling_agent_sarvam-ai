"""Bhashini provider client (STT + TTS).

Verified against https://dibd-bhashini.gitbook.io/bhashini-apis (Nov 2026):

  1. Pipeline Config call (mandatory, cached here)
       POST https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline
       headers: userID, ulcaApiKey
       body:    {"pipelineTasks": [{"taskType": "asr"|"tts",
                                    "config": {"language": {"sourceLanguage": "gu"}}}],
                 "pipelineRequestConfig": {"pipelineId": "<id>"}}
       returns: pipelineResponseConfig[].config[].serviceId and
                pipelineInferenceAPIEndPoint {callbackUrl, inferenceApiKey{name,value}}

  2. Pipeline Compute call
       POST <callbackUrl>   header <inferenceApiKey.name>: <inferenceApiKey.value>
       ASR -> pipelineResponse[0].output[0].source
       TTS -> pipelineResponse[0].audio[0].audioContent (base64 WAV)

Bhashini documents no streaming TTS and its streaming ASR is a socket.io
WebSocket, so this client is request/response only. Phone calls fall back to
local VAD + this REST STT, and TTS is always buffered.

Bhashini states the APIs are for proof-of-concept use only; production use
needs a paid agreement with the Bhashini team.
"""

from __future__ import annotations

import array
import asyncio
import base64
import io
import struct
import sys
import time
import wave
from collections.abc import AsyncIterator
from contextvars import ContextVar

import httpx

from backend.config import DETECTED_TO_TTS_LANGUAGE, Settings
from backend.errors import ProviderUnavailableError, SttError, TtsError
from backend.pipeline_trace import trace
from backend.schemas import ProviderStatus
from backend.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 2.0)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
BHASHINI_LANGUAGES = {"en", "hi", "gu"}


def bhashini_language(value: str | None, default: str = "en") -> str:
    """Map conversation language tags (en, hi-IN, hinglish, gujlish...) to ISO-639."""
    key = (value or "").strip().lower()
    if not key:
        return default
    mapped = DETECTED_TO_TTS_LANGUAGE.get(key, key)
    primary = mapped.lower().split("-")[0]
    return primary if primary in BHASHINI_LANGUAGES else default


def _pcm16_to_wav(pcm16: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm16)
    return buf.getvalue()


def _resample_pcm16(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono 16-bit PCM (no numpy needed)."""
    if src_rate == dst_rate or not pcm16:
        return pcm16
    samples = array.array("h")
    samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    if src_rate > dst_rate:
        # Moving-average low-pass before decimating: without it, 48 kHz voice
        # energy above 4 kHz folds back into the phone band as harshness.
        window = max(1, round(src_rate / dst_rate))
        if window > 1:
            smoothed = array.array("h", bytes(2 * len(samples)))
            acc = 0
            for i, value in enumerate(samples):
                acc += value
                if i >= window:
                    acc -= samples[i - window]
                smoothed[i] = int(acc / min(i + 1, window))
            samples = smoothed
    n_src = len(samples)
    n_dst = max(1, int(n_src * dst_rate / src_rate))
    out = array.array("h", bytes(2 * n_dst))
    step = src_rate / dst_rate
    last = n_src - 1
    for i in range(n_dst):
        pos = i * step
        idx = int(pos)
        if idx >= last:
            out[i] = samples[last]
            continue
        frac = pos - idx
        out[i] = int(samples[idx] + (samples[idx + 1] - samples[idx]) * frac)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _riff_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Decode a WAV to mono 16-bit PCM.

    Bhashini voices differ: IISC and AI4Bharat return 32-bit float WAV
    (format 3, which the stdlib ``wave`` module rejects); IITM and Bodhan
    return 16-bit PCM at 8/22.05/48 kHz.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise TtsError("Bhashini returned audio that is not a WAV file.")
    fmt = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            payload = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None or len(fmt) < 16:
        raise TtsError("Bhashini returned a WAV without fmt/data chunks.")
    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]  # first 2 bytes of SubFormat GUID
    width = bits // 8
    frame = width * max(1, channels)
    payload = payload[: len(payload) - (len(payload) % frame)] if frame else b""

    if tag == WAVE_FORMAT_IEEE_FLOAT and width in (4, 8):
        floats = array.array("f" if width == 4 else "d")
        floats.frombytes(payload)
        if sys.byteorder != "little":
            floats.byteswap()
        values = [max(-32768, min(32767, int(v * 32767.0))) for v in floats[:: max(1, channels)]]
        samples = array.array("h", values)
    elif tag == WAVE_FORMAT_PCM and width == 2:
        samples = array.array("h")
        samples.frombytes(payload)
        if sys.byteorder != "little":
            samples.byteswap()
        if channels > 1:
            samples = array.array("h", samples[::channels])
    elif tag == WAVE_FORMAT_PCM and width in (1, 3, 4):
        values = []
        for off in range(0, len(payload), frame):
            raw = payload[off : off + width]
            if width == 1:  # unsigned 8-bit
                values.append((raw[0] - 128) << 8)
            else:  # signed little-endian; keep the top 16 bits
                values.append(int.from_bytes(raw, "little", signed=True) >> (8 * (width - 2)))
        samples = array.array("h", values)
    else:
        raise TtsError(f"Bhashini returned an unsupported WAV format (tag {tag}, {bits}-bit).")
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes(), rate


def normalize_tts_audio(audio: bytes, reported_rate: int | None, target_rate: int) -> bytes:
    """Return mono 16-bit WAV at ``target_rate`` whatever Bhashini sent back.

    The docs say audioContent is a WAV, but the returned rate may differ from
    the requested one (the sample response shows 22050 Hz). The phone path
    only downsamples by integer steps, so resample here to keep pitch right.
    """
    if audio[:4] == b"RIFF":
        pcm16, rate = _riff_to_pcm16(audio)
    else:
        # Raw PCM16 without a header: trust the rate Bhashini reported.
        pcm16 = audio
        rate = reported_rate or target_rate
    return _pcm16_to_wav(_resample_pcm16(pcm16, rate, target_rate), target_rate)


def _wav_sample_rate(path) -> int:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getframerate()
    except (wave.Error, EOFError, OSError):
        return 16000


class BhashiniClient:
    provider_name = "bhashini"
    # No documented streaming TTS; streaming ASR is socket.io (not wired here).
    supports_realtime_stt = False
    supports_realtime_tts = False
    supports_streaming_tts = False
    # Bhashini ASR requires the source language up front.
    accepts_language_hint = True

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.bhashini_request_timeout)
        )
        self._last_attempt_count: ContextVar[int] = ContextVar(
            f"bhashini_attempt_count_{id(self)}", default=1
        )
        # (task, language) -> (service_id, callback_url, auth_name, auth_value)
        self._service_cache: dict[tuple[str, str], tuple[str, str, str, str]] = {}
        self._config_lock = asyncio.Lock()

    # ---------- configuration ----------

    @property
    def default_language(self) -> str:
        return bhashini_language(
            self._settings.bhashini_default_language or self._settings.default_language,
            "en",
        )

    @property
    def api_key_set(self) -> bool:
        s = self._settings
        direct = bool(
            s.bhashini_inference_api_key
            and s.bhashini_asr_service_id
            and s.bhashini_tts_service_id
        )
        return direct or bool(s.bhashini_user_id and s.bhashini_ulca_api_key)

    @property
    def last_attempt_count(self) -> int:
        return self._last_attempt_count.get()

    def _missing_keys_message(self) -> str:
        return (
            "Bhashini keys are not configured. Set BHASHINI_USER_ID and "
            "BHASHINI_ULCA_API_KEY in .env (from the Bhashini 'My Profile' page)."
        )

    async def _service(self, task: str, language: str) -> tuple[str, str, str, str]:
        """Resolve (serviceId, endpoint, auth header name, auth value) for a task."""
        s = self._settings
        override = s.bhashini_asr_service_id if task == "asr" else s.bhashini_tts_service_id
        if override and s.bhashini_inference_api_key:
            return override, s.bhashini_inference_url, "Authorization", s.bhashini_inference_api_key
        if not (s.bhashini_user_id and s.bhashini_ulca_api_key):
            raise ProviderUnavailableError(self._missing_keys_message())

        key = (task, language)
        cached = self._service_cache.get(key)
        if cached is not None:
            return cached
        async with self._config_lock:
            cached = self._service_cache.get(key)
            if cached is not None:
                return cached
            body = {
                "pipelineTasks": [
                    {"taskType": task, "config": {"language": {"sourceLanguage": language}}}
                ],
                "pipelineRequestConfig": {"pipelineId": s.bhashini_pipeline_id},
            }
            headers = {"userID": s.bhashini_user_id, "ulcaApiKey": s.bhashini_ulca_api_key}
            trace("bhashini.config.request", task=task, language=language)
            resp = await self._request_with_retry("POST", s.bhashini_config_url, json=body, headers=headers)
            payload = resp.json()
            resolved = self._parse_config(payload, task, language, override)
            self._service_cache[key] = resolved
            trace("bhashini.config.response", task=task, language=language, service_id=resolved[0])
            return resolved

    async def inference_endpoint(self, language: str | None = None) -> tuple[str, str, str]:
        """(url, auth header name, auth value) for direct pipeline calls."""
        _, url, auth_name, auth_value = await self._service("asr", language or self.default_language)
        return url, auth_name, auth_value

    @staticmethod
    def _parse_config(
        payload: dict, task: str, language: str, override: str = ""
    ) -> tuple[str, str, str, str]:
        endpoint = payload.get("pipelineInferenceAPIEndPoint") or {}
        callback_url = endpoint.get("callbackUrl") or ""
        key = endpoint.get("inferenceApiKey") or {}
        auth_name = key.get("name") or "Authorization"
        auth_value = key.get("value") or ""
        service_id = override
        if not service_id:
            for block in payload.get("pipelineResponseConfig") or []:
                if block.get("taskType") != task:
                    continue
                for entry in block.get("config") or []:
                    lang = ((entry.get("language") or {}).get("sourceLanguage") or "").lower()
                    if lang == language and entry.get("serviceId"):
                        service_id = entry["serviceId"]
                        break
                if service_id:
                    break
        if not (service_id and callback_url and auth_value):
            raise ProviderUnavailableError(
                f"Bhashini has no {task.upper()} service for language '{language}' "
                "in this pipeline, or the config response was incomplete.",
                details={"task": task, "language": language},
            )
        return service_id, callback_url, auth_name, auth_value

    async def _request_with_retry(
        self, method: str, url: str, *, json: dict, headers: dict
    ) -> httpx.Response:
        last_error: Exception | None = None
        attempts = 1 + MAX_RETRIES
        for attempt in range(attempts):
            try:
                resp = await self._http.request(method, url, json=json, headers=headers)
                if resp.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                    logger.info(
                        "provider_retry provider=bhashini attempt=%s status=%s",
                        attempt + 1,
                        resp.status_code,
                    )
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"Bhashini returned {resp.status_code}: {resp.text[:500]}",
                        request=resp.request,
                        response=resp,
                    )
                self._last_attempt_count.set(attempt + 1)
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code in {401, 403}:
                    # Inference keys can rotate; force a fresh config call next time.
                    self._service_cache.clear()
                break
        self._last_attempt_count.set(attempts)
        raise ProviderUnavailableError(
            "Bhashini is not reachable or returned an error. "
            "Check the BHASHINI_* keys in .env and your internet connection.",
            details=str(last_error)[:500] if last_error else None,
        )

    async def _compute(self, task: str, language: str, task_config: dict, input_data: dict) -> dict:
        service_id, url, auth_name, auth_value = await self._service(task, language)
        body = {
            "pipelineTasks": [
                {
                    "taskType": task,
                    "config": {
                        "language": {"sourceLanguage": language},
                        "serviceId": service_id,
                        **task_config,
                    },
                }
            ],
            "inputData": input_data,
        }
        resp = await self._request_with_retry(
            "POST", url, json=body, headers={auth_name: auth_value}
        )
        results = (resp.json() or {}).get("pipelineResponse") or []
        if not results:
            raise ProviderUnavailableError("Bhashini returned an empty pipelineResponse.")
        return results[0]

    # ---------- STT ----------

    async def _asr(self, audio_b64: str, language: str, sampling_rate: int) -> str:
        result = await self._compute(
            "asr",
            language,
            {"audioFormat": "wav", "samplingRate": sampling_rate},
            {"audio": [{"audioContent": audio_b64}]},
        )
        outputs = result.get("output") or []
        return ((outputs[0] or {}).get("source") if outputs else "") or ""

    @staticmethod
    def _pick_language(predictions: list, min_confidence: float) -> str | None:
        """Map an audio-lang-detection result to en/hi/gu (None = keep hint)."""
        if not predictions:
            return None
        top = predictions[0] or {}
        code = (top.get("langCode") or "").lower()
        score = top.get("langScore")
        score = float(score) if isinstance(score, (int, float)) else 0.0
        if score >= min_confidence:
            return code if code in BHASHINI_LANGUAGES else None
        # Indic speech is detected near-certainly; English scores low (and is
        # sometimes mislabelled, e.g. as Marathi), so uncertain means English.
        return "en"

    async def detect_language(self, audio_b64: str, fallback_language: str) -> str | None:
        """Bhashini audio language detection; None when unavailable."""
        s = self._settings
        try:
            # Same inference endpoint + key as ASR (ALD is not a config-call task).
            _, url, auth_name, auth_value = await self._service("asr", fallback_language)
            body = {
                "pipelineTasks": [
                    {"taskType": "audio-lang-detection", "config": {"serviceId": s.bhashini_ald_service_id}}
                ],
                "inputData": {"audio": [{"audioContent": audio_b64}]},
            }
            resp = await self._http.post(
                url, json=body, headers={auth_name: auth_value}, timeout=httpx.Timeout(10.0)
            )
            if resp.status_code >= 400:
                logger.info("bhashini_ald_failed status=%s", resp.status_code)
                return None
            results = (resp.json() or {}).get("pipelineResponse") or []
            outputs = (results[0] or {}).get("output") or [] if results else []
            predictions = (outputs[0] or {}).get("langPrediction") or [] if outputs else []
            picked = self._pick_language(predictions, s.bhashini_ald_min_confidence)
            trace("stt.language_detected", provider="bhashini", predictions=predictions, picked=picked)
            return picked
        except Exception:  # noqa: BLE001 - detection is best-effort
            logger.info("bhashini_ald_error", exc_info=True)
            return None

    async def transcribe(
        self, audio_path, duration_ms: int, language_hint: str | None = None
    ) -> tuple[str, int, str | None]:
        started = time.monotonic()
        language = bhashini_language(language_hint, self.default_language)
        sampling_rate = _wav_sample_rate(audio_path)
        trace(
            "stt.request",
            provider="bhashini",
            language=language,
            sampling_rate=sampling_rate,
            audio_path=str(audio_path),
            audio_duration_ms=duration_ms,
        )
        try:
            with open(audio_path, "rb") as fh:
                audio_b64 = base64.b64encode(fh.read()).decode("ascii")
            if self._settings.bhashini_language_detection:
                # Speculative: transcribe in the expected language while the
                # language is detected; redo ASR only if the lead switched.
                asr_task = asyncio.create_task(self._asr(audio_b64, language, sampling_rate))
                detected = await self.detect_language(audio_b64, language)
                if detected and detected != language:
                    asr_task.cancel()
                    await asyncio.gather(asr_task, return_exceptions=True)
                    trace("stt.language_switch", provider="bhashini", expected=language, detected=detected)
                    language = detected
                    text = await self._asr(audio_b64, language, sampling_rate)
                else:
                    text = await asr_task
            else:
                text = await self._asr(audio_b64, language, sampling_rate)
            if not text.strip():
                raise SttError(
                    "Bhashini returned an empty transcript. Try speaking clearly and closer to the microphone."
                )
            latency_ms = int((time.monotonic() - started) * 1000)
            transcript = text.strip()
            trace(
                "stt.response",
                provider="bhashini",
                latency_ms=latency_ms,
                language=language,
                transcript=transcript,
                transcript_chars=len(transcript),
            )
            return transcript, latency_ms, language
        except (ProviderUnavailableError, SttError):
            raise
        except Exception as exc:
            raise SttError(
                "Speech-to-text failed on Bhashini. The audio may be unintelligible or the service is busy.",
                details=str(exc)[:500],
            )

    # ---------- TTS ----------

    async def synthesize(self, text: str, detected_language: str | None = None) -> tuple[bytes, str, int]:
        started = time.monotonic()
        language = bhashini_language(detected_language, self.default_language)
        target_rate = int(self._settings.bhashini_tts_sample_rate)
        trace(
            "tts.request",
            provider="bhashini",
            mode="buffered",
            language=language,
            gender=self._settings.bhashini_tts_gender,
            text=text,
            text_chars=len(text or ""),
        )
        try:
            result = await self._compute(
                "tts",
                language,
                {
                    "gender": self._settings.bhashini_tts_gender,
                    "speed": self._settings.bhashini_tts_speed,
                    "samplingRate": target_rate,
                },
                {"input": [{"source": text}]},
            )
            audios = result.get("audio") or []
            content = (audios[0] or {}).get("audioContent") if audios else None
            if not content:
                raise TtsError("Bhashini returned no audio for the TTS request.")
            reported_rate = ((result.get("config") or {}).get("samplingRate")) or None
            audio_bytes = normalize_tts_audio(base64.b64decode(content), reported_rate, target_rate)
            latency_ms = int((time.monotonic() - started) * 1000)
            trace(
                "tts.response",
                provider="bhashini",
                mode="buffered",
                latency_ms=latency_ms,
                audio_bytes=len(audio_bytes),
                language=language,
            )
            return audio_bytes, "audio/wav", latency_ms
        except ProviderUnavailableError as exc:
            raise TtsError(
                "Text-to-speech failed on Bhashini. Check the BHASHINI_* keys and the Bhashini service status.",
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
        """Bhashini has no streaming TTS: yield the whole clip as one chunk."""
        audio, _, _ = await self.synthesize(text, detected_language)
        yield audio

    async def open_realtime_stt(self):
        raise ProviderUnavailableError("Realtime STT is not available with Bhashini.")

    async def open_realtime_tts(self, detected_language: str | None = None, *, codec: str | None = None):
        raise ProviderUnavailableError("Realtime TTS is not available with Bhashini.")

    # ---------- health ----------

    async def health_check(self) -> ProviderStatus:
        started = time.monotonic()
        details = {"pipeline_id": self._settings.bhashini_pipeline_id, "api_key_set": self.api_key_set}
        if not self.api_key_set:
            return ProviderStatus(
                provider="bhashini",
                status="degraded",
                message=self._missing_keys_message(),
                latency_ms=0,
                details=details,
            )
        s = self._settings
        if s.bhashini_inference_api_key and s.bhashini_asr_service_id and s.bhashini_tts_service_id:
            # Direct mode makes no config call, so there is nothing cheap to
            # probe; real failures surface on the first ASR/TTS request.
            return ProviderStatus(
                provider="bhashini",
                status="ok",
                message=(
                    "Bhashini inference key and service IDs are configured "
                    f"(ASR {s.bhashini_asr_service_id}, TTS {s.bhashini_tts_service_id})."
                ),
                latency_ms=0,
                details={**details, "mode": "direct"},
            )
        try:
            service_id, _, _, _ = await self._service("tts", self.default_language)
            return ProviderStatus(
                provider="bhashini",
                status="ok",
                message="Bhashini keys work and a TTS service is available.",
                latency_ms=int((time.monotonic() - started) * 1000),
                details={**details, "tts_service_id": service_id},
            )
        except ProviderUnavailableError as exc:
            return ProviderStatus(
                provider="bhashini",
                status="error",
                message=exc.message,
                latency_ms=int((time.monotonic() - started) * 1000),
                details={**details, "error": str(exc.details)[:300] if exc.details else None},
            )

    async def aclose(self) -> None:
        if self._own_client:
            await self._http.aclose()
