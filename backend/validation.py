"""Field validation for lead data.

Validators are intentionally permissive so legitimate international and
Indian phone/email formats are not rejected too aggressively.
"""

import re

_PHONE_CLEAN_RE = re.compile(r"[^\d+]")
_PHONE_OK_RE = re.compile(r"^\+?\d{7,15}$")
_EMAIL_OK_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def clean_phone(value: str) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    cleaned = _PHONE_CLEAN_RE.sub("", value)
    if not _PHONE_OK_RE.match(cleaned):
        return None
    return cleaned


def is_valid_email(value: str) -> bool:
    return bool((value or "").strip()) and bool(_EMAIL_OK_RE.match(value.strip()))


def is_valid_name(value: str) -> bool:
    value = (value or "").strip()
    if len(value) < 2:
        return False
    if not any(ch.isalpha() for ch in value):
        return False
    if value.isdigit():
        return False
    return True


def normalize_phone(value: str) -> str | None:
    """Return a normalized E.164-ish phone string, or None if invalid."""
    return clean_phone(value)


def has_valid_contact(phone: str | None, email: str | None) -> bool:
    return bool(clean_phone(phone or "")) or is_valid_email(email or "")


def is_decisive_yes(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    yes_tokens = {
        "yes", "y", "yeah", "yep", "sure", "ha", "han", "haan", "haha", "हाँ",
        "हां", "ho", "hmm", "ok", "okay", "theek", "ठीक",
    }
    return any(token in lowered for token in yes_tokens)


def is_decision_maker(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    if lowered in {"no", "nope", "not sure", "nhi", "nahi", "नहीं"}:
        return False
    yes_tokens = {"yes", "y", "yeah", "ha", "haan", "हाँ", "हां"}
    role_tokens = {"owner", "founder", "ceo", "director", "manager", "head", "vp"}
    if any(token in lowered for token in yes_tokens):
        return True
    return any(token in lowered for token in role_tokens)


def is_consent_value(value: str | None) -> bool:
    return (value or "").strip().lower() in {"yes", "no", "y", "n"}


def consent_bool(value: str | None) -> bool:
    return is_decisive_yes(value or "")
