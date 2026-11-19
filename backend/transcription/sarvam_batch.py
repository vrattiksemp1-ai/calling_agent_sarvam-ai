"""Sarvam Batch Speech-to-Text with speaker diarization.

Verified against https://docs.sarvam.ai (Nov 2026) and live-tested:
  POST {base}                     create job  {"job_parameters": {...}}
  POST {base}/upload-files        presigned upload URL per file
  PUT  <upload url>               x-ms-blob-type: BlockBlob
  POST {base}/{job_id}/start
  GET  {base}/{job_id}/status     job_state: Accepted|Pending|Running|Completed|Failed
  POST {base}/download-files      presigned URL for e.g. "0.json"
Output JSON: diarized_transcript.entries[] {transcript, start_time_seconds,
end_time_seconds, speaker_id}. Files up to 2 hours.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from backend.config import Settings
from backend.errors import ProviderUnavailableError, SttError
from backend.transcription.models import Segment, language_of_text

SARVAM_MAX_SECONDS = 2 * 3600
DONE_STATES = {"Completed", "PartiallyCompleted", "Failed"}

Progress = Callable[[str, float | None], None]


class SarvamBatchTranscriber:
    provider = "sarvam"

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self._s = settings
        self._own = http is None
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    @property
    def detail(self) -> str:
        return f"Sarvam ({self._s.sarvam_batch_model}, {self._s.sarvam_batch_mode}, batch diarization)"

    def _headers(self) -> dict:
        if not self._s.sarvam_api_key:
            raise ProviderUnavailableError("SARVAM_API_KEY is not configured.")
        return {"api-subscription-key": self._s.sarvam_api_key}

    async def _call(self, method: str, url: str, **kwargs) -> dict:
        resp = await self._http.request(method, url, headers=self._headers(), **kwargs)
        if resp.status_code >= 400:
            raise ProviderUnavailableError(
                f"Sarvam batch API returned HTTP {resp.status_code}.", details=resp.text[:500]
            )
        return resp.json()

    async def transcribe(
        self,
        audio_file: Path,
        duration: float,
        progress: Progress,
        num_speakers: int | None = None,
    ) -> list[Segment]:
        if duration > SARVAM_MAX_SECONDS:
            raise SttError(
                "Sarvam's batch API accepts recordings up to 2 hours. Split the file or use Bhashini.",
                retryable=False,
            )
        base = self._s.sarvam_batch_url.rstrip("/")
        params = {
            "model": self._s.sarvam_batch_model,
            "mode": self._s.sarvam_batch_mode,
            "language_code": "unknown",
            "with_diarization": True,
            "with_timestamps": True,
        }
        if num_speakers:
            params["num_speakers"] = int(num_speakers)
        progress("Creating Sarvam transcription job", 0.05)
        job = await self._call("POST", base, json={"job_parameters": params})
        job_id = job["job_id"]

        progress("Uploading audio to Sarvam", 0.1)
        urls = await self._call("POST", f"{base}/upload-files", json={"job_id": job_id, "files": [audio_file.name]})
        upload_url = urls["upload_urls"][audio_file.name]["file_url"]
        put = await self._http.put(
            upload_url,
            content=audio_file.read_bytes(),
            headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "audio/flac"},
        )
        if put.status_code >= 400:
            raise ProviderUnavailableError("Uploading the audio to Sarvam failed.", details=put.text[:300])

        await self._call("POST", f"{base}/{job_id}/start")
        started = time.monotonic()
        status: dict = {}
        while True:
            await asyncio.sleep(self._s.sarvam_batch_poll_seconds)
            status = await self._call("GET", f"{base}/{job_id}/status")
            state = status.get("job_state")
            elapsed = time.monotonic() - started
            # Rough progress: Sarvam gives no percentage, so creep towards 90%.
            progress(f"Sarvam is transcribing ({state}, {int(elapsed)} s)", min(0.9, 0.15 + elapsed / max(60.0, duration / 4)))
            if state in DONE_STATES:
                break
            if elapsed > self._s.sarvam_batch_timeout_seconds:
                raise SttError("Sarvam did not finish the transcription in time.", details={"job_id": job_id})

        details = status.get("job_details") or []
        names = [o.get("file_name") for d in details for o in (d.get("outputs") or []) if o.get("file_name")]
        if status.get("job_state") == "Failed" or not names:
            reason = status.get("error_message") or "; ".join(
                str(d.get("error_message")) for d in details if d.get("error_message")
            )
            raise SttError("Sarvam could not transcribe this file.", details=reason or {"job_id": job_id})

        progress("Downloading the transcript", 0.95)
        links = await self._call("POST", f"{base}/download-files", json={"job_id": job_id, "files": names[:1]})
        out = await self._http.get(links["download_urls"][names[0]]["file_url"])
        out.raise_for_status()
        return self.parse_output(out.json())

    @staticmethod
    def parse_output(data: dict) -> list[Segment]:
        entries = (data.get("diarized_transcript") or {}).get("entries") or []
        segments = [
            Segment(
                start=float(e.get("start_time_seconds") or 0.0),
                end=float(e.get("end_time_seconds") or 0.0),
                speaker=str(e.get("speaker_id", "0")),
                text=(e.get("transcript") or "").strip(),
            )
            for e in entries
        ]
        if not segments and (data.get("transcript") or "").strip():
            # Diarization produced nothing: keep the text under one speaker.
            segments = [Segment(0.0, 0.0, "0", data["transcript"].strip())]
        for seg in segments:
            # Sarvam reports one language per file; tag each line by script.
            seg.language = language_of_text(seg.text)
        return [s for s in segments if s.text]

    async def aclose(self) -> None:
        if self._own:
            await self._http.aclose()
