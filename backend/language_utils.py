"""Language mapping helpers.

The agent mostly decides reply language from the conversation itself. Signals:

  * Sarvam STT's auto-detected ``language_code`` (provider signal)
  * Unicode script of the caller's transcript (Gujarati / Devanagari letters)
  * Explicit switch requests ("english mein baat karo", etc.)
  * Clear Latin-script English sentences

Script detection is NOT a general word-list language classifier.
"""

from __future__ import annotations

import re
import unicodedata


def map_stt_language_code(language_code: str | None) -> str | None:
    """Map a Sarvam STT language_code (e.g. 'gu-IN') to gu/hi/en or None."""
    if not language_code:
        return None
    primary = (language_code or "").strip().lower().split("-")[0]
    return primary if primary in {"gu", "hi", "en"} else None


def infer_script_language(text: str | None) -> str | None:
    """Infer gu/hi from Unicode script letters in ``text``.

    Returns:
      * ``"gu"`` when Gujarati-script letters dominate
      * ``"hi"`` when Devanagari letters dominate (Hindi/Marathi etc.)
      * ``None`` for Latin-only / empty / inconclusive text (LLM decides)
    """
    if not text or not text.strip():
        return None
    gu = 0
    hi = 0
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x0A80 <= code <= 0x0AFF:  # Gujarati
            gu += 1
        elif 0x0900 <= code <= 0x097F:  # Devanagari
            hi += 1
        else:
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                name = ""
            if "GUJARATI" in name:
                gu += 1
            elif "DEVANAGARI" in name:
                hi += 1
    if gu == 0 and hi == 0:
        return None
    if gu >= hi and gu > 0:
        return "gu"
    if hi > 0:
        return "hi"
    return None


def should_keep_prior_language(text: str | None, prior: str | None) -> bool:
    """True when a short Latin reply should keep the prior gu/hi language.

    Full English sentences are allowed to switch; one-word acknowledgements
    ("yes", "ok", "haan") after a Gujarati turn should not flip the call.
    """
    if (prior or "").strip().lower() not in {"gu", "hi"}:
        return False
    if not text or not text.strip():
        return True
    if infer_script_language(text):
        return False
    if detect_explicit_language_switch(text):
        return False
    words = [w for w in text.strip().split() if w]
    return len(words) <= 3


def detect_explicit_language_switch(text: str | None) -> str | None:
    """Return en/hi/gu when the user clearly asks to change language."""
    if not text or not text.strip():
        return None
    t = text.strip().lower()
    # Normalize common ASR spacing/punctuation noise.
    compact = re.sub(r"[^a-z0-9\u0900-\u097f\u0a80-\u0aff]+", " ", t)
    compact = re.sub(r"\s+", " ", compact).strip()

    en_patterns = (
        "english mein",
        "english me",
        "in english",
        "speak english",
        "speaking english",
        "talk in english",
        "talk english",
        "can we speak in english",
        "can we talk in english",
        "please speak english",
        "english please",
        "english bolo",
        "english ma",
        "englishmaa",
        "angrezi",
        "angreji",
        # Gujarati / Hindi script ASR of English switch requests
        "અંગ્રેજી",
        "અંગ્રેજીમાં",
        "ઇંગલિશ",
        "ઈંગલિશ",
        "ઇંગ્લિશ",
        "સ્પીકિંગ ઇંગ",
        "સ્પીકિંગ ઇંગલિશ",
        "સ્પીક ઇંગલિશ",
        "ઇંગલિશ માં",
        "ઇંગલિશમાં",
        "इंग्लिश",
        "अंग्रेजी",
        "स्पीकिंग इंग्लिश",
        "इंग्लिश में",
    )
    hi_patterns = (
        "hindi mein",
        "hindi me",
        "in hindi",
        "speak hindi",
        "speaking hindi",
        "talk in hindi",
        "can you speak in hindi",
        "can you speak hindi",
        "hindi bolo",
        "hindima",
        "हिंदी में",
        "हिन्दी में",
        "हिन्दी",
        "હિન્દી",
        "સ્પીક હિન્દી",
        "હિન્દીમાં",
    )
    gu_patterns = (
        "gujarati mein",
        "gujarati me",
        "gujarati ma",
        "in gujarati",
        "speak gujarati",
        "speaking gujarati",
        "talk in gujarati",
        "gujarati bolo",
        "gujarati maa",
        "ગુજરાતીમાં",
        "ગુજરાતી માં",
        "પીકિંગ ગુજરાતી",
        "स्पीक गुजराती",
        "गुजराती में",
    )
    for p in en_patterns:
        if p in compact:
            return "en"
    for p in hi_patterns:
        if p in compact:
            return "hi"
    for p in gu_patterns:
        if p in compact:
            return "gu"
    return None


def looks_like_latin_english_sentence(text: str | None) -> bool:
    """True for a clear Latin-script English sentence (language-switch signal).

    Roman Gujarati/Hindi ("kem cho maru naam...") must NOT match. We only treat
    Latin text as English when it carries common English cues.
    """
    if not text or not text.strip():
        return False
    if infer_script_language(text):
        return False
    words = [w.strip(".,!?;:\"'()").lower() for w in text.strip().split() if w]
    if len(words) < 3:
        return False
    cues = {
        "the",
        "are",
        "is",
        "am",
        "we",
        "using",
        "currently",
        "looking",
        "need",
        "want",
        "please",
        "because",
        "have",
        "this",
        "that",
        "with",
        "from",
        "about",
        "for",
        "my",
        "name",
        "hello",
        "hi",
        "automation",
        "spreadsheets",
        "can",
        "you",
        "what",
        "how",
        "building",
        "software",
        "company",
        "business",
    }
    hits = sum(1 for w in words if w in cues)
    # 3+ word English with one strong cue, or 4+ words with two cues.
    if len(words) >= 4 and hits >= 2:
        return True
    if len(words) >= 3 and hits >= 2:
        return True
    return False


def resolve_turn_language(
    user_text: str | None,
    *,
    prior_language: str | None = None,
    stt_language: str | None = None,
) -> str | None:
    """Pick a hard language pin for this turn, or None to let the LLM decide.

    Priority:
      1) explicit switch request
      2) provider STT language
      3) clear Latin English sentence
      4) Indic script of the transcript
      5) short acknowledgement keeps prior gu/hi
    """
    explicit = detect_explicit_language_switch(user_text)
    if explicit:
        return explicit
    if stt_language in {"en", "hi", "gu"}:
        return stt_language
    if looks_like_latin_english_sentence(user_text):
        return "en"
    script = infer_script_language(user_text)
    if script:
        return script
    if should_keep_prior_language(user_text, prior_language):
        return (prior_language or "").strip().lower() or None
    return None
