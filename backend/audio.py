"""Audio upload validation and FFmpeg conversion.

Browser MediaRecorder produces WebM/Opus (Chrome/Firefox) or MP4/M4A (Safari).
The local STT API expects common formats; we normalise to 16 kHz mono WAV with
FFmpeg so the provider receives exactly the format it expects.

FFmpeg is required and must be installed separately (see SETUP.md).
"""

import asyncio
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend.config import Settings
from backend.errors import FfmpegMissingError, InvalidAudioError

ALLOWED_CONTENT_TYPES = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/ogg",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/x-flac",
    "audio/flac",
    "",
}

ALLOWED_EXTENSIONS = {".webm", ".ogg", ".opus", ".mp4", ".m4a", ".wav", ".mp3", ".flac"}


@dataclass
class PreparedAudio:
    wav_path: Path
    duration_ms: int
    original_name: str

    def cleanup(self) -> None:
        for path in (self.wav_path,):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def safe_extension(filename: str) -> str:
    return Path(filename or "audio.webm").suffix.lower()


def validate_upload(filename: str, content_type: str, size: int, max_audio_mb: int) -> None:
    ext = safe_extension(filename)
    if size <= 0:
        raise InvalidAudioError("The uploaded audio is empty. Please record again.")
    if size > max_audio_mb * 1024 * 1024:
        raise InvalidAudioError(
            f"Audio file is too large ({size // (1024 * 1024)} MB). "
            f"Maximum allowed is {max_audio_mb} MB."
        )
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidAudioError(
            f"Unsupported audio content type: {content_type or 'unknown'}. "
            "Record with a modern browser that supports MediaRecorder."
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio extension: {ext}. Expected webm, ogg, mp4, m4a, wav, mp3 or flac."
        )


def wav_duration_ms(wav_path: Path) -> int:
    """Estimate duration of a 16-bit PCM WAV from its size (no ffprobe needed)."""
    try:
        size = wav_path.stat().st_size
        return max(0, int((size - 44) / 2 / 16000 * 1000))
    except OSError:
        return 0


async def _run_ffmpeg(ffmpeg_bin: str, args: list[str], timeout: float = 120.0) -> None:
    if not shutil.which(ffmpeg_bin):
        raise FfmpegMissingError(details=f"ffmpeg binary '{ffmpeg_bin}' not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise InvalidAudioError("Audio conversion timed out.", details="ffmpeg conversion timeout")
    if proc.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise InvalidAudioError(
            "FFmpeg could not convert the uploaded audio.",
            details=tail,
        )


async def prepare_audio(
    data: bytes,
    filename: str,
    content_type: str,
    settings: Settings,
    ffmpeg_bin: str | None = None,
) -> PreparedAudio:
    """Validate an upload, convert it to 16 kHz mono WAV and return a handle.

    The returned PreparedAudio must be cleaned up by the caller.
    """
    validate_upload(filename, content_type, len(data), settings.max_audio_mb)

    temp_dir = settings.resolved_temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    session_token = uuid.uuid4().hex
    raw_name = f"raw_{session_token}{safe_extension(filename)}"
    wav_name = f"norm_{session_token}.wav"

    raw_path = temp_dir / raw_name
    wav_path = temp_dir / wav_name

    binary = ffmpeg_bin or settings.ffmpeg_path
    try:
        raw_path.write_bytes(data)
        # 16 kHz mono 16-bit PCM: the format recommended by local ASR engines.
        await _run_ffmpeg(
            binary,
            ["-y", "-i", str(raw_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
        )
        duration_ms = wav_duration_ms(wav_path)
    finally:
        try:
            if raw_path.exists():
                raw_path.unlink()
        except OSError:
            pass

    if settings.retain_audio:
        retain_dir = temp_dir / "retained"
        retain_dir.mkdir(exist_ok=True)
        shutil.copy2(wav_path, retain_dir / wav_name)

    return PreparedAudio(wav_path=wav_path, duration_ms=duration_ms, original_name=filename)
