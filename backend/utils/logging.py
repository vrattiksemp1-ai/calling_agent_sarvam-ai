"""Structured logging setup with PII protection.

Phone numbers and email addresses found in log messages are masked.
"""

import logging
import re

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def mask_pii(message: str) -> str:
    masked = _PHONE_RE.sub("[PHONE_REDACTED]", message)
    masked = _EMAIL_RE.sub("[EMAIL_REDACTED]", masked)
    return masked


class PiiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_pii(record.msg)
        return True


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(PiiFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
