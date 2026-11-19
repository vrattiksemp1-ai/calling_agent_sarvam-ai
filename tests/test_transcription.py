"""Transcription page: diarization stitching, formatting, providers, API."""

from __future__ import annotations

import io
import time
import wave

import httpx
import pytest

from backend.transcription.bhashini_diarized import (
    parse_diarization,
    plan_windows,
    stitch_windows,
    tidy_spans,
)
from backend.transcription.models import Segment, Transcript, merge_turns, render_text
from backend.transcription.sarvam_batch import SarvamBatchTranscriber
from tests.conftest import TestClientContext, make_settings


# ---------- Bhashini windowing ----------


def test_plan_windows_overlap_and_cover_the_file():
    assert plan_windows(100, 120, 30) == [(0.0, 100)]
    windows = plan_windows(300, 120, 30)
    assert windows == [(0.0, 120.0), (90.0, 210.0), (180.0, 300)]
    for (_, prev_end), (start, _) in zip(windows, windows[1:]):
        assert prev_end - start == 30.0


def test_parse_diarization_accepts_both_time_spellings():
    payload = {
        "pipelineResponse": [
            {
                "output": [
                    {
                        "speaker_labels": [
                            {"speaker1": [{"start_Time": 0.0, "duration": 2.0}]},  # as returned live
                            {"speaker2": [{"start_time": 3.0, "duration": 1.5}]},  # as documented
                        ]
                    }
                ]
            }
        ]
    }
    assert parse_diarization(payload, offset=90.0) == [("speaker1", 90.0, 92.0), ("speaker2", 93.0, 94.5)]


def test_stitch_matches_speakers_across_windows_by_overlap():
    windows = [(0.0, 120.0), (90.0, 210.0)]
    # Window 2 numbers the same people the other way round.
    w1 = [("speaker1", 0, 40), ("speaker2", 40, 95), ("speaker1", 95, 110), ("speaker2", 110, 120)]
    w2 = [("speaker2", 90, 110), ("speaker1", 110, 150), ("speaker3", 150, 210)]
    spans = stitch_windows(windows, [w1, w2])
    # Person A (w1 speaker1 / w2 speaker2) keeps one id, B likewise, C is new.
    a = next(g for g, s, e in spans if s == 0)
    # Window 2 owns time from 105 s; A's speech there (w2 "speaker2") keeps A's id.
    assert next(g for g, s, e in spans if s == 105) == a
    b = next(g for g, s, e in spans if s == 40)
    assert next(g for g, s, e in spans if s >= 110 and e <= 150) == b
    c = next(g for g, s, e in spans if s == 150)
    assert len({a, b, c}) == 3
    # No moment is transcribed twice: the overlap is split at its middle (105 s).
    for (_, _, e1), (_, s2, _) in zip(spans, spans[1:]):
        assert s2 >= e1 - 1e-6


def test_tidy_merges_and_cuts_long_turns():
    spans = tidy_spans([("g0", 0, 10), ("g0", 10.2, 70), ("g1", 70, 70.1)], max_len=30)
    assert [(g, round(s), round(e)) for g, s, e in spans] == [("g0", 0, 23), ("g0", 23, 47), ("g0", 47, 70)]


# ---------- formatting ----------


def test_render_text_numbers_speakers_and_tags_languages():
    t = Transcript(
        filename="call.mp3",
        provider="sarvam",
        provider_detail="Sarvam (test)",
        duration=95.0,
        segments=[
            Segment(0.0, 4.0, "7", "નમસ્તે, કેમ છો?", "gu"),
            Segment(4.5, 6.0, "3", "Hello, yes.", "en"),
            Segment(6.2, 8.0, "3", "Tell me more.", "en"),
            Segment(9.0, 11.0, "7", "अच्छा, ठीक है।", "hi"),
        ],
    )
    text = render_text(t)
    assert "Duration: 01:35   Speakers: 2   Languages: Gujarati, English, Hindi" in text
    assert "[00:00 - 00:04] Speaker 1 (gu): નમસ્તે, કેમ છો?" in text
    assert "[00:04 - 00:08] Speaker 2 (en): Hello, yes. Tell me more." in text  # same turn merged
    assert "[00:09 - 00:11] Speaker 1 (hi): अच्छा, ठीक है।" in text


def test_long_recordings_use_hours_in_timestamps():
    t = Transcript("m.mp3", "sarvam", "x", 3725.0, [Segment(3700.0, 3705.0, "0", "Bye.", "en")])
    assert "[01:01:40 - 01:01:45] Speaker 1 (en): Bye." in render_text(t)


def test_merge_keeps_language_changes_apart():
    merged = merge_turns([Segment(0, 2, "a", "Hello", "en"), Segment(2.1, 4, "a", "નમસ્તે", "gu")])
    assert len(merged) == 2


def test_sarvam_output_parsing_tags_language_per_line():
    data = {
        "diarized_transcript": {
            "entries": [
                {"transcript": "Yes, sure.", "start_time_seconds": 5.5, "end_time_seconds": 8.6, "speaker_id": "1"},
                {"transcript": "अच्छा यह तो दिलचस्प है", "start_time_seconds": 13.5, "end_time_seconds": 16.3, "speaker_id": "1"},
                {"transcript": "અમે business માટે", "start_time_seconds": 9.1, "end_time_seconds": 13.0, "speaker_id": "0"},
            ]
        },
        "language_code": "gu-IN",
    }
    segs = SarvamBatchTranscriber.parse_output(data)
    assert [(s.speaker, s.language) for s in segs] == [("1", "en"), ("1", "hi"), ("0", "gu")]


# ---------- Sarvam batch flow (mocked HTTP) ----------


@pytest.mark.asyncio
async def test_sarvam_batch_job_flow(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append((request.method, url))
        if request.method == "POST" and url.endswith("/job/v1"):
            import json

            params = json.loads(request.content)["job_parameters"]
            assert params["with_diarization"] is True and params["mode"] == "codemix"
            assert params["model"] == "saaras:v4" and params["num_speakers"] == 2
            return httpx.Response(202, json={"job_id": "J1", "job_state": "Accepted"})
        if url.endswith("/upload-files"):
            return httpx.Response(200, json={"upload_urls": {"a.flac": {"file_url": "https://blob/up"}}})
        if url == "https://blob/up":
            assert request.headers["x-ms-blob-type"] == "BlockBlob"
            return httpx.Response(201)
        if url.endswith("/J1/start"):
            return httpx.Response(200, json={"job_state": "Pending"})
        if url.endswith("/J1/status"):
            return httpx.Response(200, json={"job_state": "Completed", "job_details": [{"outputs": [{"file_name": "0.json"}]}]})
        if url.endswith("/download-files"):
            return httpx.Response(200, json={"download_urls": {"0.json": {"file_url": "https://blob/out"}}})
        if url == "https://blob/out":
            return httpx.Response(200, json={"diarized_transcript": {"entries": [
                {"transcript": "Hello", "start_time_seconds": 0, "end_time_seconds": 1, "speaker_id": "0"}]}})
        return httpx.Response(404)

    settings = make_settings(tmp_path, sarvam_api_key="k", sarvam_batch_poll_seconds=0.0)
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"fLaC")
    t = SarvamBatchTranscriber(settings, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    segs = await t.transcribe(audio, 60.0, lambda m, f: None, num_speakers=2)
    assert [(s.text, s.speaker, s.language) for s in segs] == [("Hello", "0", "en")]
    assert ("PUT", "https://blob/up") in calls
    await t.aclose()


@pytest.mark.asyncio
async def test_sarvam_rejects_over_two_hours(tmp_path):
    from backend.errors import SttError

    t = SarvamBatchTranscriber(make_settings(tmp_path, sarvam_api_key="k"))
    with pytest.raises(SttError):
        await t.transcribe(tmp_path / "x.flac", 2 * 3600 + 1, lambda m, f: None)
    await t.aclose()


# ---------- API ----------


class FakeTranscriber:
    detail = "Fake provider"

    def __init__(self, provider):
        self.provider = provider
        self.seen = None

    async def transcribe(self, path, duration, progress, num_speakers=None):
        self.seen = (path.suffix, round(duration, 1), num_speakers)
        progress("working", 0.5)
        segs = [Segment(0.0, 1.0, "x", "Hello there", "en"), Segment(1.2, 2.0, "y", "નમસ્તે", "gu")]
        return segs if self.provider == "sarvam" else (segs, ["note"])


def _wav_bytes(seconds=2.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(rate * seconds))
    return buf.getvalue()


@pytest.mark.parametrize("provider,suffix", [("sarvam", ".flac"), ("bhashini", ".wav")])
def test_transcription_api_end_to_end(app, tmp_path, provider, suffix):
    import shutil

    if not shutil.which(app.state.settings.ffmpeg_path):
        pytest.skip("ffmpeg not installed")
    fakes = {"sarvam": FakeTranscriber("sarvam"), "bhashini": FakeTranscriber("bhashini")}
    service = app.state.transcription_service
    service._transcribers = fakes
    app.state.settings.transcription_dir = str(tmp_path / "out")
    with TestClientContext(app) as c:
        resp = c.post(
            "/api/transcriptions",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
            data={"speech_provider": provider, "num_speakers": "2"},
        )
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        for _ in range(100):
            job = c.get(f"/api/transcriptions/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        assert job["speakers"] == 2 and job["languages"] == ["English", "Gujarati"]
        dl = c.get(f"/api/transcriptions/{job_id}/download")
        assert dl.status_code == 200
        assert "Speaker 1 (en): Hello there" in dl.text
        assert "Speaker 2 (gu): નમસ્તે" in dl.text
        assert len(c.get("/api/transcriptions").json()) == 1
    assert fakes[provider].seen == (suffix, 2.0, 2)
    assert (tmp_path / "out" / job["download_name"]).is_file()


def test_transcription_api_validation(app):
    with TestClientContext(app) as c:
        bad = c.post("/api/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")}, data={"speech_provider": "google"})
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_SPEECH_PROVIDER"
        empty = c.post("/api/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}, data={"speech_provider": "sarvam"})
        assert empty.json()["error"]["code"] == "EMPTY_FILE"
        assert c.get("/api/transcriptions/nope").status_code == 404
        assert c.get("/transcribe").status_code == 200


def test_split_label_in_one_window_maps_to_the_same_person():
    from backend.transcription.bhashini_diarized import absorb_minor_speakers

    windows = [(0.0, 120.0), (90.0, 210.0)]
    w1 = [("speaker1", 0, 100), ("speaker2", 100, 120)]
    # Window 2 splits the first person into two labels inside the overlap.
    w2 = [("s_a", 90, 95), ("s_b", 95, 100), ("s_c", 100, 210)]
    spans = stitch_windows(windows, [w1, w2])
    assert len({g for g, _, _ in spans}) == 2
    # A 1-second "speaker" between two turns of the same person is folded in.
    glitch = [("g0", 0, 50), ("g9", 50, 51), ("g0", 51, 100), ("g1", 100, 160)]
    assert {g for g, _, _ in absorb_minor_speakers(glitch)} == {"g0", "g1"}
