"""Background transcription jobs: convert -> transcribe -> write the .txt file."""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backend.config import PROJECT_ROOT, Settings
from backend.errors import AppError
from backend.transcription.audio_prep import to_flac, to_wav
from backend.transcription.models import (
    LANGUAGE_NAMES,
    Transcript,
    merge_turns,
    number_speakers,
    render_text,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionJob:
    id: str
    filename: str
    provider: str
    num_speakers: int | None = None
    status: str = "queued"  # queued | running | completed | failed
    message: str = "Waiting to start"
    progress: float = 0.0
    error: str | None = None
    duration: float | None = None
    speakers: int | None = None
    languages: list[str] = field(default_factory=list)
    text: str | None = None
    text_path: Path | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "filename": self.filename,
            "speech_provider": self.provider,
            "status": self.status,
            "message": self.message,
            "progress": round(self.progress, 3),
            "error": self.error,
            "duration_seconds": round(self.duration, 1) if self.duration else None,
            "speakers": self.speakers,
            "languages": self.languages,
            "transcript": self.text,
            "download_name": self.text_path.name if self.text_path else None,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.created_at, 1),
        }


def _safe_stem(filename: str) -> str:
    stem = Path(filename or "audio").stem
    return re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")[:60] or "audio"


class TranscriptionService:
    def __init__(self, settings: Settings, transcribers: dict, max_parallel_jobs: int = 2):
        self._s = settings
        self._transcribers = transcribers
        self._jobs: dict[str, TranscriptionJob] = {}
        self._tasks: set[asyncio.Task] = set()
        self._slots = asyncio.Semaphore(max_parallel_jobs)

    @property
    def output_dir(self) -> Path:
        path = Path(self._s.transcription_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def get(self, job_id: str) -> TranscriptionJob | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[TranscriptionJob]:
        return sorted(self._jobs.values(), key=lambda j: -j.created_at)[:limit]

    def submit(self, upload: Path, filename: str, provider: str, num_speakers: int | None) -> TranscriptionJob:
        job = TranscriptionJob(id=uuid.uuid4().hex[:16], filename=filename, provider=provider, num_speakers=num_speakers)
        self._jobs[job.id] = job
        task = asyncio.create_task(self._run(job, upload), name=f"transcribe-{job.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: TranscriptionJob, upload: Path) -> None:
        work = Path(tempfile.mkdtemp(prefix=f"stt_{job.id}_", dir=self._ensure_temp()))

        def progress(message: str, fraction: float | None) -> None:
            job.message = message
            if fraction is not None:
                job.progress = max(job.progress, min(0.99, fraction))

        try:
            async with self._slots:
                job.status = "running"
                progress("Converting audio", 0.02)
                wav = work / "audio.wav"
                job.duration = await to_wav(self._s.ffmpeg_path, upload, wav)
                transcriber = self._transcribers[job.provider]
                notes: list[str] = []
                if job.provider == "sarvam":
                    flac = work / "audio.flac"
                    await to_flac(self._s.ffmpeg_path, wav, flac)
                    segments = await transcriber.transcribe(flac, job.duration, progress, job.num_speakers)
                else:
                    segments, notes = await transcriber.transcribe(wav, job.duration, progress, job.num_speakers)
                transcript = Transcript(
                    filename=job.filename,
                    provider=job.provider,
                    provider_detail=transcriber.detail,
                    duration=job.duration,
                    segments=segments,
                    notes=notes,
                )
                text = render_text(transcript, datetime.now())
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / (
                    f"{_safe_stem(job.filename)}_{job.provider}_{datetime.now():%Y%m%d-%H%M%S}.txt"
                )
                path.write_text(text, encoding="utf-8")
                merged = merge_turns(segments)
                job.speakers = len(number_speakers(merged))
                job.languages = list(dict.fromkeys(LANGUAGE_NAMES.get(s.language or "", s.language or "?") for s in merged))
                job.text, job.text_path = text, path
                job.status, job.progress, job.message = "completed", 1.0, f"Saved {path.name}"
                logger.info("transcription_done job=%s provider=%s file=%s", job.id, job.provider, path)
        except AppError as exc:
            job.status, job.error = "failed", exc.message
            job.message = exc.message
            logger.warning("transcription_failed job=%s code=%s details=%s", job.id, exc.code, exc.details)
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", "Unexpected error while transcribing. See the server log."
            job.message = job.error
            logger.exception("transcription_crashed job=%s: %s", job.id, exc)
        finally:
            job.finished_at = time.time()
            shutil.rmtree(work, ignore_errors=True)
            upload.unlink(missing_ok=True)

    def _ensure_temp(self) -> Path:
        path = self._s.resolved_temp_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for transcriber in self._transcribers.values():
            close = getattr(transcriber, "aclose", None)
            if close is not None:
                await close()
