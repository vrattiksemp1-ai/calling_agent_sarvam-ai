"""Transcript data model and the text-file format."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backend.language_utils import infer_script_language

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}


@dataclass
class Segment:
    start: float
    end: float
    speaker: str  # provider label before numbering, e.g. "0" or "g3"
    text: str = ""
    language: str | None = None


@dataclass
class Transcript:
    filename: str
    provider: str
    provider_detail: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def language_of_text(text: str) -> str:
    """Per-line language from the script: Gujarati, Devanagari, else English."""
    return infer_script_language(text) or "en"


def number_speakers(segments: list[Segment]) -> dict[str, int]:
    """Speaker 1, 2, ... in order of first appearance."""
    numbers: dict[str, int] = {}
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.speaker not in numbers:
            numbers[seg.speaker] = len(numbers) + 1
    return numbers


def merge_turns(segments: list[Segment], max_gap: float = 1.5, max_len: float = 60.0) -> list[Segment]:
    """Join consecutive pieces of one speaker's turn (same language, short gap)."""
    merged: list[Segment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.text.strip():
            continue
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.speaker == seg.speaker
            and prev.language == seg.language
            and seg.start - prev.end <= max_gap
            and seg.end - prev.start <= max_len
        ):
            prev.end = max(prev.end, seg.end)
            prev.text = f"{prev.text.rstrip()} {seg.text.strip()}"
        else:
            merged.append(Segment(seg.start, seg.end, seg.speaker, seg.text.strip(), seg.language))
    return merged


def fmt_time(seconds: float, long_form: bool) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if long_form else f"{m:02d}:{s:02d}"


def render_text(transcript: Transcript, created: datetime | None = None) -> str:
    segments = merge_turns(transcript.segments)
    numbers = number_speakers(segments)
    long_form = transcript.duration >= 3600
    languages = []
    for seg in segments:
        name = LANGUAGE_NAMES.get(seg.language or "", seg.language or "unknown")
        if name not in languages:
            languages.append(name)
    created = created or datetime.now()
    lines = [
        f"File: {transcript.filename}",
        f"Provider: {transcript.provider_detail}",
        f"Duration: {fmt_time(transcript.duration, long_form)}   "
        f"Speakers: {len(numbers)}   Languages: {', '.join(languages) or '-'}",
        f"Transcribed: {created.strftime('%Y-%m-%d %H:%M')}",
    ]
    lines += [f"Note: {note}" for note in transcript.notes]
    lines.append("")
    for seg in segments:
        lines.append(
            f"[{fmt_time(seg.start, long_form)} - {fmt_time(seg.end, long_form)}] "
            f"Speaker {numbers[seg.speaker]} ({seg.language or '?'}): {seg.text}"
        )
    if not segments:
        lines.append("(No speech was recognised in this file.)")
    return "\n".join(lines) + "\n"
