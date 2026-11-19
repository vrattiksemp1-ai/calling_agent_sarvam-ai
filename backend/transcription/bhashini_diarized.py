"""Bhashini speaker diarization + per-segment ASR for long recordings.

Live-tested Nov 2026 with the project's inference key:
  * ``bhashini/speaker-diarization`` works; ``bhashini/iisc/speaker-diarization``
    returned HTTP 500.
  * Diarization succeeds up to 2 minutes of audio and fails (HTTP 500,
    DHRUVA-101) from 3 minutes, so long files are diarized in overlapping
    windows. Speakers are matched across windows by who talks during the
    shared overlap; a person silent for the whole overlap gets a new number.
  * The response uses ``start_Time`` although the docs say ``start_time``.
  * ASR accepts at least 120 s per request; segments are cut to 30 s anyway
    so timestamps stay useful.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import httpx

from backend.config import Settings
from backend.errors import AppError, ProviderUnavailableError, SttError
from backend.providers.bhashini_client import BhashiniClient
from backend.transcription.audio_prep import downsample_wav_8k, slice_wav
from backend.transcription.models import Segment
from backend.utils.logging import get_logger

logger = get_logger(__name__)

Progress = Callable[[str, float | None], None]
Span = tuple[str, float, float]  # (speaker label, start, end) in file seconds


def plan_windows(duration: float, window: float, overlap: float) -> list[tuple[float, float]]:
    if duration <= window:
        return [(0.0, duration)]
    step = max(1.0, window - overlap)
    windows = []
    start = 0.0
    while True:
        end = min(start + window, duration)
        windows.append((start, end))
        if end >= duration:
            return windows
        start += step


def parse_diarization(payload: dict, offset: float) -> list[Span]:
    """Bhashini output -> spans in absolute seconds."""
    results = payload.get("pipelineResponse") or [payload]
    spans: list[Span] = []
    for result in results:
        for out in (result or {}).get("output") or []:
            for group in (out or {}).get("speaker_labels") or []:
                for label, items in (group or {}).items():
                    for item in items or []:
                        start = item.get("start_time", item.get("start_Time"))
                        duration = item.get("duration")
                        if start is None or duration is None:
                            continue
                        spans.append((str(label), offset + float(start), offset + float(start) + float(duration)))
    return sorted(spans, key=lambda s: s[1])


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def stitch_windows(
    windows: list[tuple[float, float]], per_window: list[list[Span]], min_match_seconds: float = 0.5
) -> list[Span]:
    """Give speakers file-wide ids and keep each moment from one window only.

    Window k's local labels are matched to the ids already assigned in window
    k-1 by how long they speak at the same time inside the shared overlap.
    Each window then "owns" the time up to the middle of each overlap.
    """
    result: list[Span] = []
    prev: list[Span] = []
    next_id = 0
    for k, ((w_start, w_end), spans) in enumerate(zip(windows, per_window)):
        mapping: dict[str, str] = {}
        if k > 0 and prev:
            ov_start, ov_end = w_start, windows[k - 1][1]
            scores: dict[tuple[str, str], float] = defaultdict(float)
            for label, s, e in spans:
                for gid, ps, pe in prev:
                    lo, hi = max(s, ps, ov_start), min(e, pe, ov_end)
                    if hi > lo:
                        scores[(label, gid)] += hi - lo
            # Many-to-one: within one window Bhashini sometimes splits a person
            # into two labels; both should map to that person's file-wide id.
            # (One-to-one matching turned each split into a "new speaker" -
            # 14 speakers instead of 3 on a 30-minute test.)
            for (label, gid), score in sorted(scores.items(), key=lambda kv: -kv[1]):
                if score >= min_match_seconds and label not in mapping:
                    mapping[label] = gid
        for label, _, _ in spans:
            if label not in mapping:
                mapping[label] = f"g{next_id}"
                next_id += 1
        mapped = [(mapping[label], s, e) for label, s, e in spans]
        own_start = w_start if k == 0 else (w_start + windows[k - 1][1]) / 2
        own_end = w_end if k == len(windows) - 1 else (windows[k + 1][0] + w_end) / 2
        for gid, s, e in mapped:
            s2, e2 = max(s, own_start), min(e, own_end)
            if e2 - s2 >= 0.2:
                result.append((gid, s2, e2))
        prev = mapped
    return sorted(result, key=lambda s: s[1])


def absorb_minor_speakers(spans: list[Span], min_total_seconds: float = 4.0) -> list[Span]:
    """Fold "speakers" with only a few seconds of speech in the whole file into
    the speaker talking around them. On long files these are almost always a
    diarization glitch rather than a real participant."""
    totals: dict[str, float] = defaultdict(float)
    for gid, s, e in spans:
        totals[gid] += e - s
    if len(totals) <= 1:
        return spans
    major = {gid for gid, total in totals.items() if total >= min_total_seconds}
    if not major:
        return spans
    ordered = sorted(spans, key=lambda x: x[1])
    out: list[Span] = []
    for i, (gid, s, e) in enumerate(ordered):
        if gid not in major:
            before = next((g for g, _, _ in reversed(ordered[:i]) if g in major), None)
            after = next((g for g, _, _ in ordered[i + 1 :] if g in major), None)
            gid = before or after or gid
        out.append((gid, s, e))
    return out


def tidy_spans(spans: list[Span], max_len: float, gap: float = 0.5, min_len: float = 0.3) -> list[Span]:
    """Merge a speaker's back-to-back spans, then cut long ones for ASR."""
    merged: list[list] = []
    for gid, s, e in sorted(spans, key=lambda x: x[1]):
        if merged and merged[-1][0] == gid and s - merged[-1][2] <= gap:
            merged[-1][2] = max(merged[-1][2], e)
        else:
            merged.append([gid, s, e])
    out: list[Span] = []
    for gid, s, e in merged:
        if e - s < min_len:
            continue
        pieces = max(1, int((e - s) // max_len) + (1 if (e - s) % max_len > 0.01 else 0))
        size = (e - s) / pieces
        for i in range(pieces):
            out.append((gid, s + i * size, s + (i + 1) * size))
    return out


class BhashiniDiarizedTranscriber:
    provider = "bhashini"

    def __init__(self, settings: Settings, client: BhashiniClient, http: httpx.AsyncClient | None = None):
        self._s = settings
        self._client = client
        self._own = http is None
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(180.0))

    @property
    def detail(self) -> str:
        return (
            f"Bhashini ({self._s.bhashini_diarization_service_id} + "
            f"{self._s.bhashini_asr_service_id or 'ASR from config call'})"
        )

    async def _diarize_window(self, wav_8k: Path, offset: float) -> list[Span]:
        url, auth_name, auth_value = await self._client.inference_endpoint()
        body = {
            "pipelineTasks": [
                {"taskType": "speaker-diarization", "config": {"serviceId": self._s.bhashini_diarization_service_id}}
            ],
            "inputData": {"audio": [{"audioContent": base64.b64encode(wav_8k.read_bytes()).decode("ascii")}]},
        }
        last = ""
        for attempt in range(3):
            try:
                resp = await self._http.post(url, json=body, headers={auth_name: auth_value})
                if resp.status_code < 400:
                    return parse_diarization(resp.json(), offset)
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2 * (attempt + 1))
        raise ProviderUnavailableError(
            "Bhashini speaker identification failed for part of the file.",
            details={"window_start_seconds": round(offset, 1), "error": last},
        )

    async def transcribe(
        self,
        wav_16k: Path,
        duration: float,
        progress: Progress,
        num_speakers: int | None = None,  # Bhashini diarization has no such option
    ) -> tuple[list[Segment], list[str]]:
        notes: list[str] = []
        windows = plan_windows(
            duration, self._s.bhashini_diarization_window_seconds, self._s.bhashini_diarization_overlap_seconds
        )
        per_window: list[list[Span]] = []
        self._s.resolved_temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._s.resolved_temp_dir) as tmp:
            tmp_dir = Path(tmp)
            for i, (start, end) in enumerate(windows):
                progress(f"Identifying speakers (part {i + 1} of {len(windows)})", 0.05 + 0.35 * i / len(windows))
                part = tmp_dir / f"win_{i}.wav"
                downsample_wav_8k(wav_16k, part, start, end)
                per_window.append(await self._diarize_window(part, start))
            stitched = stitch_windows(windows, per_window)
            if len(windows) > 1:
                stitched = absorb_minor_speakers(stitched)
            spans = tidy_spans(stitched, self._s.bhashini_segment_max_seconds)
            if len(windows) > 1:
                # Measured Nov 2026: a 3-person, 30-minute test file came out as
                # 11-14 speakers with ~73% of speech under the right label,
                # while the text itself was accurate. Say so in the file.
                notes.append(
                    "Bhashini identifies speakers only in 2-minute parts, so on longer recordings the "
                    "speaker numbers are unreliable (the same person can appear as several speakers). "
                    "The text is accurate. Use Sarvam when correct speaker labels matter."
                )
            if not spans:
                return [], notes

            sem = asyncio.Semaphore(max(1, self._s.bhashini_transcription_concurrency))
            done = 0
            failed = 0
            segments: list[Segment | None] = [None] * len(spans)

            async def run(index: int, span: Span) -> None:
                nonlocal done, failed
                gid, s, e = span
                async with sem:
                    piece = tmp_dir / f"seg_{index}.wav"
                    seconds = slice_wav(wav_16k, s, e, piece)
                    try:
                        text, _, lang = await self._client.transcribe(
                            piece, int(seconds * 1000), language_hint=self._s.default_language
                        )
                        segments[index] = Segment(s, e, gid, text, lang)
                    except SttError:
                        pass  # silence / noise: nothing recognisable
                    except AppError:
                        failed += 1
                        logger.warning("bhashini segment %s failed", index, exc_info=True)
                    finally:
                        piece.unlink(missing_ok=True)
                        done += 1
                        progress(f"Transcribing speech ({done} of {len(spans)} segments)", 0.4 + 0.55 * done / len(spans))

            await asyncio.gather(*(run(i, sp) for i, sp in enumerate(spans)))
        if failed:
            notes.append(f"{failed} of {len(spans)} speech segments could not be transcribed by Bhashini and are missing.")
            if failed == len(spans):
                raise SttError("Bhashini could not transcribe any part of this file.")
        return [s for s in segments if s is not None], notes

    async def aclose(self) -> None:
        if self._own:
            await self._http.aclose()
