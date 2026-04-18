"""Tests for single-instance child-spawn lock.

The shim must ensure that only one CLI child process runs at a time. When a
second spawn is attempted while a child is still alive, the lock attempt must
fail immediately (non-blocking) so the caller can return 409 instead of
forking a phantom sibling that would race on the same disk SSOT.
"""

from __future__ import annotations

import multiprocessing
import os
import time

import pytest

from hermes_shim_http.single_child import ChildLockBusy, acquire_single_child_lock


def _hold_lock_then_exit(lock_path: str, seconds: float, started_event, released_event) -> None:
    with acquire_single_child_lock(lock_path):
        started_event.set()
        time.sleep(seconds)
    released_event.set()


def test_lock_acquires_when_free(tmp_path):
    lock_path = str(tmp_path / "child.lock")
    with acquire_single_child_lock(lock_path):
        pass  # acquired and released without error
    assert os.path.exists(lock_path)


def test_lock_is_reacquirable_after_release(tmp_path):
    lock_path = str(tmp_path / "child.lock")
    with acquire_single_child_lock(lock_path):
        pass
    with acquire_single_child_lock(lock_path):
        pass  # must not raise — prior release freed the flock


def test_lock_rejects_concurrent_acquire_from_another_process(tmp_path):
    lock_path = str(tmp_path / "child.lock")
    ctx = multiprocessing.get_context("fork")
    started = ctx.Event()
    released = ctx.Event()
    holder = ctx.Process(
        target=_hold_lock_then_exit,
        args=(lock_path, 1.5, started, released),
        daemon=True,
    )
    holder.start()
    try:
        assert started.wait(timeout=3.0), "holder never acquired the lock"

        with pytest.raises(ChildLockBusy):
            with acquire_single_child_lock(lock_path):
                pytest.fail("should not have acquired — holder still alive")

        assert released.wait(timeout=5.0), "holder never released the lock"
        holder.join(timeout=2.0)

        with acquire_single_child_lock(lock_path):
            pass  # now free
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=2.0)


def test_lock_no_op_when_path_empty(tmp_path):
    with acquire_single_child_lock(""):
        pass  # empty path → disabled, never raises


def test_lock_writes_owner_pid_for_diagnostics(tmp_path):
    lock_path = str(tmp_path / "child.lock")
    with acquire_single_child_lock(lock_path):
        with open(lock_path, "rb") as fh:
            contents = fh.read().decode().strip()
        assert contents == str(os.getpid())
