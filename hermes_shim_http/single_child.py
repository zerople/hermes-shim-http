"""Child-spawn lock helper.

The runner derives a request-scoped lock path and acquires it non-blockingly.
In the current shim design, only resumed requests for the same parent session
share a lock path; fresh sessions do not serialize globally. This prevents two
children from forking the same parent session concurrently across retries or
multi-process deployments.
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
