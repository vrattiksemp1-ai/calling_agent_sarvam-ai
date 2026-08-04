"""Phone-number validation and normalization to E.164.

Kept dependency-free for the MVP: a number must be an optional leading '+'
followed by 7-15 digits (E.164 rules). No country guessing is attempted.
"""

from __future__ import annotations

from backend.errors import AppError

MIN_DIGITS = 7
MAX_DIGITS = 15


class InvalidPhoneNumberError(AppError):
    def __init__(self, message: str | None = None):
        super().__init__(
            code="INVALID_PHONE_NUMBER",
            message=message
            or "Phone number is not valid. Use E.164 format, e.g. +919876543210.",
            retryable=False,
            status_code=400,
        )


def normalize_e164(raw: str | None) -> str:
    """Normalize a phone number to canonical E.164 form or raise."""
    value = (raw or "").strip()
    if value.startswith("+"):
        digits = value[1:]
    else:
        digits = value
    if not digits.isdigit():
        raise InvalidPhoneNumberError()
    if not (MIN_DIGITS <= len(digits) <= MAX_DIGITS):
        raise InvalidPhoneNumberError()
    return "+" + digits


def is_valid_e164(raw: str | None) -> bool:
    try:
        normalize_e164(raw)
        return True
    except AppError:
        return False
