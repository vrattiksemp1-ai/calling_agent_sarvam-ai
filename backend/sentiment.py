"""Transcript-only delivery hints for empathetic voice responses.

This module intentionally does not inspect audio or infer consent, lead quality,
or user intent.  It is a tiny local classifier whose output is suitable only
for adjusting response empathy and cadence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptStyleSignal:
    label: str
    directive: str
    evidence_count: int
    source: str = "rolling_transcript"
    consequential: bool = False


_FRUSTRATED = {
    "annoyed",
    "frustrated",
    "stop asking",
    "not listening",
    "irritated",
    "गुस्सा",
    "परेशान",
    "बार बार",
    "समझ नहीं",
    "કંટાળો",
    "હેરાન",
    "વારંવાર",
    "સમજાતું નથી",
}
_RUSHED = {
    "in a hurry",
    "no time",
    "quickly",
    "make it quick",
    "जल्दी",
    "समय नहीं",
    "टाइम नहीं",
    "જલ્દી",
    "ટાઈમ નથી",
    "સમય નથી",
}
_POSITIVE = {
    "great",
    "perfect",
    "sounds good",
    "thank you",
    "बहुत अच्छा",
    "धन्यवाद",
    "સરસ",
    "બહુ સારું",
    "આભાર",
}


def rolling_transcript_style(
    messages: list[dict] | list[str],
    *,
    window: int = 4,
) -> TranscriptStyleSignal:
    """Return a deterministic, nonblocking style hint from recent user text."""
    texts: list[str] = []
    for item in messages:
        if isinstance(item, str):
            texts.append(item)
        elif item.get("role") == "user" and item.get("content"):
            texts.append(str(item["content"]))
    transcript = " ".join(texts[-max(1, window) :]).casefold()

    frustrated = sum(1 for term in _FRUSTRATED if term in transcript)
    rushed = sum(1 for term in _RUSHED if term in transcript)
    positive = sum(1 for term in _POSITIVE if term in transcript)
    if frustrated:
        return TranscriptStyleSignal(
            "frustrated",
            "Acknowledge the concern briefly, use a calm cadence, and ask one simple question.",
            frustrated,
        )
    if rushed:
        return TranscriptStyleSignal(
            "rushed",
            "Use a brisk cadence, omit optional pleasantries, and keep the next question short.",
            rushed,
        )
    if positive:
        return TranscriptStyleSignal(
            "positive",
            "Match the positive tone lightly while staying concise and professional.",
            positive,
        )
    return TranscriptStyleSignal(
        "neutral",
        "Keep a warm, steady, professional cadence.",
        0,
    )


def style_prompt_block(signal: TranscriptStyleSignal) -> str:
    """Serialize the signal with hard boundaries around its allowed use."""
    return (
        "\n\nTRANSCRIPT STYLE SIGNAL (delivery only): "
        f"{signal.label}. {signal.directive}\n"
        "- This is a lightweight rolling transcript signal, not acoustic emotion.\n"
        "- Use it ONLY for empathy, wording, and cadence.\n"
        "- Never use it to infer or alter consent, extracted fields, qualification, "
        "state transitions, task completion, or any consequential action.\n"
        "- The caller's explicit words remain authoritative."
    )
