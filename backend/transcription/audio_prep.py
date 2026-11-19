"""Convert any audio/video file with ffmpeg and slice it for providers."""

from __future__ import annotations

import wave
from pathlib import Path

from backend.audio import _run_ffmpeg
from backend.errors import InvalidAudioError

WORK_RATE = 16000


async def to_wav(ffmpeg_bin: str, source: Path, target: Path, rate: int = WORK_RATE) -> float:
    """Decode anything ffmpeg understands to mono 16-bit WAV; return seconds."""
    await _run_ffmpeg(
        ffmpeg_bin,
        ["-y", "-i", str(source), "-vn", "-ac", "1", "-ar", str(rate), "-sample_fmt", "s16", str(target)],
        timeout=1800.0,
    )
    duration = wav_seconds(target)
    if duration < 0.3:
        raise InvalidAudioError("The file contains no usable audio.")
    return duration


async def to_flac(ffmpeg_bin: str, source_wav: Path, target: Path) -> None:
    """Lossless and about half the size of WAV, for uploading to Sarvam."""
    await _run_ffmpeg(ffmpeg_bin, ["-y", "-i", str(source_wav), "-c:a", "flac", str(target)], timeout=1800.0)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def slice_wav(source: Path, start: float, end: float, target: Path) -> float:
    """Copy [start, end) seconds of a PCM WAV into a new WAV; return seconds."""
    with wave.open(str(source), "rb") as src:
        rate = src.getframerate()
        width = src.getsampwidth()
        channels = src.getnchannels()
        first = max(0, int(start * rate))
        last = min(src.getnframes(), int(end * rate))
        src.setpos(first)
        frames = src.readframes(max(0, last - first))
    with wave.open(str(target), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(width)
        dst.setframerate(rate)
        dst.writeframes(frames)
    return max(0, last - first) / float(rate)


def downsample_wav_8k(source: Path, target: Path, start: float, end: float) -> None:
    """Window at 8 kHz for diarization (half the upload of 16 kHz)."""
    import array

    with wave.open(str(source), "rb") as src:
        rate = src.getframerate()
        src.setpos(max(0, int(start * rate)))
        frames = src.readframes(max(0, int((end - start) * rate)))
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - len(frames) % 2])
    if rate == 2 * 8000:
        # Average pairs: a cheap low-pass before dropping every other sample.
        out = array.array("h", ((samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples) - 1, 2)))
        out_rate = 8000
    else:
        out, out_rate = samples, rate
    with wave.open(str(target), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(out_rate)
        dst.writeframes(out.tobytes())
