"""Minimal in-memory rate limiting (MVP grade).

Suitable for a local tool. Replace with a real store before any public
deployment - see the security notes in README.md.
"""

import time
from collections import defaultdict, deque

from backend.config import Settings


class RateLimiter:
    def __init__(self, enabled: bool = True, per_minute: int = 60):
        self._enabled = enabled
        self._per_minute = max(1, per_minute)
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if not self._enabled:
            return True, 0
        now = time.monotonic()
        window = self._requests[client_key]
        while window and window[0] <= now - 60.0:
            window.popleft()
        if len(window) >= self._per_minute:
            retry_after = int(60.0 - (now - window[0]))
            return False, max(1, retry_after)
        window.append(now)
        return True, 0
