"""Lightweight multilingual endpoint hints for realtime voice turns."""

from __future__ import annotations

import re


_TRAILING_COMPLETE = re.compile(r"[.!?।？！]\s*$")
_TRAILING_CONNECTOR = re.compile(
    r"(?:\b(?:and|but|because|so|then|or|also|that|with|for|to|"
    r"aur|lekin|kyunki|phir|ke|ki|ka|ane|pan|kem ke|etle|athva)\b|"
    r"(?:અને|પણ|કારણ કે|એટલે|અથવા|તો|और|लेकिन|क्योंकि|तो))\s*$",
    re.IGNORECASE,
)
_SHORT_COMPLETE = {
    "yes",
    "no",
    "okay",
    "ok",
    "sure",
    "thanks",
    "thank you",
    "હા",
    "ના",
    "ઠીક છે",
    "બરાબર",
    "હા બરાબર",
    "जी",
    "हाँ",
    "नहीं",
    "ठीक है",
}


class SemanticEndpointing:
    """Map partial transcript shape to a conservative VAD silence window.

    This doesn't replace Sarvam VAD. It adjusts only the server's silence
    duration: complete clauses and short confirmations finalize faster, while
    obvious conjunctions get extra room to prevent false endpoints.
    """

    def __init__(self, base_ms: int, fast_ms: int = 320, slow_ms: int = 550):
        self.base_ms = base_ms
        self.fast_ms = min(base_ms, fast_ms)
        self.slow_ms = max(base_ms, slow_ms)

    def recommend(self, partial: str) -> int:
        text = " ".join((partial or "").strip().split())
        if not text:
            return self.base_ms
        lowered = text.casefold()
        if _TRAILING_CONNECTOR.search(lowered):
            return self.slow_ms
        if _TRAILING_COMPLETE.search(text) or lowered in _SHORT_COMPLETE:
            return self.fast_ms
        return self.base_ms
