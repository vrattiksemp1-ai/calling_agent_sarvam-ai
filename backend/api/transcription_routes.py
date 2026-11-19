"""Transcription page API: upload a recording, poll progress, download .txt.

  POST /api/transcriptions                 multipart: file, speech_provider, num_speakers
  GET  /api/transcriptions                 recent jobs (this server run)
  GET  /api/transcriptions/{job_id}        status, progress and transcript text
  GET  /api/transcriptions/{job_id}/download   the saved .txt file
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from backend.config import Settings
from backend.errors import AppError
from backend.providers.speech_router import UnknownSpeechProviderError, normalize_speech_provider
from backend.transcription.jobs import TranscriptionService

router = APIRouter(prefix="/api/transcriptions", tags=["transcriptions"])

CHUNK = 1024 * 1024


def _service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


def _job_or_404(request: Request, job_id: str):
    job = _service(request).get(job_id)
    if job is None:
        raise AppError(
            code="TRANSCRIPTION_NOT_FOUND",
            message="That transcription does not exist (jobs are kept until the server restarts).",
            retryable=False,
            status_code=404,
        )
    return job


@router.post("")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    speech_provider: str = Form("sarvam"),
    num_speakers: int | None = Form(None),
) -> dict:
    settings: Settings = request.app.state.settings
    try:
        provider = normalize_speech_provider(speech_provider, settings.speech_provider)
    except UnknownSpeechProviderError as exc:
        raise AppError(code="INVALID_SPEECH_PROVIDER", message=str(exc), retryable=False, status_code=400)
    if num_speakers is not None and not 1 <= num_speakers <= 20:
        raise AppError(code="INVALID_SPEAKER_COUNT", message="Number of speakers must be 1-20.", retryable=False, status_code=400)

    limit = settings.transcription_max_mb * 1024 * 1024
    settings.resolved_temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix[:10]
    upload = settings.resolved_temp_dir / f"upload_{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with upload.open("wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise AppError(
                        code="FILE_TOO_LARGE",
                        message=f"The file is larger than {settings.transcription_max_mb} MB.",
                        retryable=False,
                        status_code=413,
                    )
                out.write(chunk)
    except BaseException:
        upload.unlink(missing_ok=True)
        raise
    if size == 0:
        upload.unlink(missing_ok=True)
        raise AppError(code="EMPTY_FILE", message="The uploaded file is empty.", retryable=False, status_code=400)
    job = _service(request).submit(upload, file.filename or "audio", provider, num_speakers)
    return job.to_dict()


@router.get("")
def list_transcriptions(request: Request) -> list[dict]:
    return [{k: v for k, v in job.to_dict().items() if k != "transcript"} for job in _service(request).recent()]


@router.get("/{job_id}")
def get_transcription(request: Request, job_id: str) -> dict:
    return _job_or_404(request, job_id).to_dict()


@router.get("/{job_id}/download")
def download_transcription(request: Request, job_id: str) -> FileResponse:
    job = _job_or_404(request, job_id)
    if job.status != "completed" or job.text_path is None or not job.text_path.is_file():
        raise AppError(code="TRANSCRIPT_NOT_READY", message="The transcript is not ready yet.", retryable=True, status_code=409)
    return FileResponse(job.text_path, media_type="text/plain; charset=utf-8", filename=job.text_path.name)
