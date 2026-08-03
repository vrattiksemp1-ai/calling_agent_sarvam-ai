"""Audio validation and conversion tests."""

import pytest

from backend.audio import (
    prepare_audio,
    safe_extension,
    validate_upload,
    wav_duration_ms,
)
from backend.config import Settings
from backend.errors import FfmpegMissingError, InvalidAudioError
from tests.conftest import make_settings


def test_safe_extension():
    assert safe_extension("recording.webm") == ".webm"
    assert safe_extension("../../etc/passwd") == ""
    assert safe_extension("") == ".webm"


def test_empty_audio_rejected():
    with pytest.raises(InvalidAudioError):
        validate_upload("a.webm", "audio/webm", 0, 15)


def test_oversize_rejected():
    with pytest.raises(InvalidAudioError) as exc:
        validate_upload("a.webm", "audio/webm", 16 * 1024 * 1024, 15)
    assert "too large" in exc.value.message


def test_bad_content_type_rejected():
    with pytest.raises(InvalidAudioError):
        validate_upload("a.webm", "text/plain", 100, 15)


def test_bad_extension_rejected():
    with pytest.raises(InvalidAudioError):
        validate_upload("evil.exe", "audio/webm", 100, 15)


def test_valid_types_accepted():
    validate_upload("a.webm", "audio/webm", 100, 15)
    validate_upload("a.mp4", "audio/mp4", 100, 15)
    validate_upload("a.wav", "", 100, 15)


def test_wav_duration():
    path = tmp_path_write()
    assert wav_duration_ms(path) > 0


def tmp_path_write():
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    p = d / "test.wav"
    # 44-byte header + 1 second of 16-bit mono 16 kHz = 32000 bytes
    p.write_bytes(b"\x00" * 44 + b"\x00" * 32000)
    return p


@pytest.mark.asyncio
async def test_prepare_audio_missing_ffmpeg(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("backend.audio.shutil.which", lambda name: None)
    with pytest.raises(FfmpegMissingError):
        await prepare_audio(b"webm-data", "a.webm", "audio/webm", settings)


@pytest.mark.asyncio
async def test_prepare_audio_converts_and_cleans_raw(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    calls = []

    async def fake_ffmpeg(binary, args, timeout=120.0):
        calls.append(args)
        out_path = args[-1]
        from pathlib import Path

        Path(out_path).write_bytes(b"\x00" * 44 + b"\x00" * 32000)

    monkeypatch.setattr("backend.audio._run_ffmpeg", fake_ffmpeg)
    prepared = await prepare_audio(b"webm-data", "a.webm", "audio/webm", settings)
    assert prepared.duration_ms == 1000
    assert prepared.wav_path.exists()
    # Raw input file must not be left behind.
    leftovers = list(settings.resolved_temp_dir.glob("raw_*"))
    assert leftovers == []
    prepared.cleanup()
    assert not prepared.wav_path.exists()


@pytest.mark.asyncio
async def test_prepare_audio_ffmpeg_error(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)

    async def fake_ffmpeg(binary, args, timeout=120.0):
        raise InvalidAudioError("FFmpeg could not convert the uploaded audio.")

    monkeypatch.setattr("backend.audio._run_ffmpeg", fake_ffmpeg)
    with pytest.raises(InvalidAudioError):
        await prepare_audio(b"webm-data", "a.webm", "audio/webm", settings)


@pytest.mark.asyncio
async def test_retain_audio_flag(monkeypatch, tmp_path):
    settings = make_settings(tmp_path, retain_audio=True)

    async def fake_ffmpeg(binary, args, timeout=120.0):
        from pathlib import Path

        Path(args[-1]).write_bytes(b"\x00" * 44 + b"\x00" * 32000)

    monkeypatch.setattr("backend.audio._run_ffmpeg", fake_ffmpeg)
    prepared = await prepare_audio(b"webm-data", "a.webm", "audio/webm", settings)
    retained = list(settings.resolved_temp_dir.glob("retained/*.wav"))
    assert len(retained) == 1
    prepared.cleanup()
