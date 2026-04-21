"""Tests for LiveChildPool — persistent multi-turn Claude child processes.

The pool holds one long-lived claude process per conversation session_key.
Prompts on the same session_key reuse the same child (fast path).
Idle children are evicted after idle_ttl; the pool is LRU-capped by size.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_shim_http.live_child_pool import LiveChildPool


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = REPO_ROOT / "tests" / "fake_cli.py"


def _collect_text(events) -> str:
    return "".join(event.text or "" for event in events if event.kind == "text")


@pytest.fixture
def pool_factory():
    created: list[LiveChildPool] = []

    def _factory(**overrides) -> LiveChildPool:
        kwargs = dict(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "claude-multiturn"],
            cwd=str(REPO_ROOT),
            size=4,
            idle_ttl=30.0,
        )
        kwargs.update(overrides)
        pool = LiveChildPool(**kwargs)
        created.append(pool)
        return pool

    try:
        yield _factory
    finally:
        for pool in created:
            pool.shutdown()


def test_same_session_key_reuses_same_child_process(pool_factory):
    pool = pool_factory()

    events1 = list(pool.stream("sess-a", "hello"))
    pid1 = pool.peek_pid("sess-a")

    events2 = list(pool.stream("sess-a", "world"))
    pid2 = pool.peek_pid("sess-a")

    assert pid1 is not None
    assert pid1 == pid2, "same session_key must reuse the same child process"
    assert "echo:hello" in _collect_text(events1)
    assert "echo:world" in _collect_text(events2)


def test_different_session_keys_get_distinct_children(pool_factory):
    pool = pool_factory()

    list(pool.stream("sess-a", "first"))
    list(pool.stream("sess-b", "first"))

    pid_a = pool.peek_pid("sess-a")
    pid_b = pool.peek_pid("sess-b")

    assert pid_a is not None and pid_b is not None
    assert pid_a != pid_b


def test_idle_child_is_evicted_after_idle_ttl(pool_factory):
    pool = pool_factory(idle_ttl=0.2)

    list(pool.stream("sess-a", "first"))
    pid1 = pool.peek_pid("sess-a")
    assert pid1 is not None

    time.sleep(0.4)
    pool.sweep()

    assert pool.peek_pid("sess-a") is None, "idle child should have been evicted"

    list(pool.stream("sess-a", "second"))
    pid2 = pool.peek_pid("sess-a")
    assert pid2 is not None
    assert pid2 != pid1, "a fresh child should be spawned after eviction"


def test_lru_eviction_when_pool_exceeds_size(pool_factory):
    pool = pool_factory(size=2, idle_ttl=300.0)

    list(pool.stream("sess-a", "one"))
    list(pool.stream("sess-b", "two"))
    list(pool.stream("sess-a", "three"))  # touch 'a' so 'b' is LRU
    list(pool.stream("sess-c", "four"))   # forces eviction of 'b'

    assert pool.peek_pid("sess-a") is not None
    assert pool.peek_pid("sess-b") is None, "LRU entry should be evicted"
    assert pool.peek_pid("sess-c") is not None


def test_shutdown_kills_all_live_children(pool_factory):
    pool = pool_factory()

    list(pool.stream("sess-a", "hello"))
    list(pool.stream("sess-b", "hello"))
    pid_a = pool.peek_pid("sess-a")
    pid_b = pool.peek_pid("sess-b")

    pool.shutdown()

    assert pool.peek_pid("sess-a") is None
    assert pool.peek_pid("sess-b") is None
    for pid in (pid_a, pid_b):
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                Path(f"/proc/{pid}").stat()
            except FileNotFoundError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"child pid {pid} did not exit after shutdown")
