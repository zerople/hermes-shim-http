"""Single-instance child-spawn lock.

Ensures only one CLI child process runs at a time. When a second spawn is
requested while a child is still alive, the lock attempt fails immediately
(non-blocking). The HTTP layer maps this to 409 instead of spawning a
phantom sibling — preventing the race where two claude CLI children write
concurrently to the same disk SSOT (autonomous.json, .lck/, worktree files).
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Iterator


class ChildLockBusy(RuntimeError):
    """Raised when the single-child lock is already held by another spawn."""


@contextmanager
def acquire_single_child_lock(lock_path: str) -> Iterator[None]:
    if not lock_path:
        yield
        return

    parent_dir = os.path.dirname(lock_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ChildLockBusy(f"another child holds {lock_path}") from exc

        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}".encode())
        except OSError:
            pass

        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
