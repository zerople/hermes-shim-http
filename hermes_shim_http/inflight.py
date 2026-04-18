from __future__ import annotations

import threading
import time


class InFlightRegistry:
    """Reject concurrent duplicate requests sharing an idempotency key.

    Hermes (and similar clients) retry on RemoteProtocolError. Without this
    guard, a retry fired while the original request is still producing output
    spawns a second claude CLI session for the same logical turn — wasting
    tokens and sometimes writing the same result twice. Reserving on arrival
    and releasing on exit lets the shim return 409 to the retry instead of
    double-spending.
    """

    def __init__(self, *, stale_after_seconds: float = 3600.0) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, float] = {}
        self._stale_after_seconds = stale_after_seconds

    def reserve(self, key: str) -> bool:
        if not key:
            return True
        now = time.monotonic()
        with self._lock:
            existing = self._active.get(key)
            if existing is not None and (now - existing) < self._stale_after_seconds:
                return False
            self._active[key] = now
            return True

    def release(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._active.pop(key, None)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._active)
