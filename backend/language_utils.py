"""Language mapping helpers.

The agent decides reply language from the conversation itself. The only
deterministic signal we use is the STT engine's own auto-detected
language_code - that is the provider's intelligence, not a hardcoded word list.
"""


def map_stt_language_code(language_code: str | None) -> str | None:
    """Map a Sarvam STT language_code (e.g. 'gu-IN') to gu/hi/en or None."""
    if not language_code:
        return None
    primary = (language_code or "").strip().lower().split("-")[0]
    return primary if primary in {"gu", "hi", "en"} else None
