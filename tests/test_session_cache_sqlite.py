from __future__ import annotations

import time

from hermes_shim_http.session_cache import SessionCache


def test_session_cache_blocks_concurrent_resume_of_same_parent(tmp_path):
    cache_path = tmp_path / "sessions.sqlite"
    cache = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=10)
    first = cache.plan_request(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    cache.record_success(first, assistant_messages=[{"role": "assistant", "content": "hi"}])

    resume_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]
    plan_a = cache.plan_request(
        messages=resume_messages,
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    plan_b = cache.plan_request(
        messages=resume_messages,
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )

    assert plan_a.resume_session_id == first.session_id
    # Second concurrent planner must NOT resume the same parent while first is in flight.
    assert plan_b.resume_session_id is None

    # Once released, subsequent plans are free to resume again.
    cache.release_plan(plan_a)
    plan_c = cache.plan_request(
        messages=resume_messages,
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    assert plan_c.resume_session_id == first.session_id


def test_session_cache_persists_across_instances(tmp_path):
    cache_path = tmp_path / "sessions.sqlite"
    cache_a = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=10)
    first = cache_a.plan_request(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    cache_a.record_success(first, assistant_messages=[{"role": "assistant", "content": "hi"}])

    cache_b = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=10)
    second = cache_b.plan_request(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "continue"},
        ],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )

    assert second.resume_session_id == first.session_id
    assert second.prefix_message_count == 2
    assert cache_b.stats()["hit_count"] == 1


def test_session_cache_expires_entries_after_ttl(tmp_path):
    cache_path = tmp_path / "sessions.sqlite"
    cache = SessionCache(path=str(cache_path), ttl_seconds=0.01, max_entries=10)
    first = cache.plan_request(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    cache.record_success(first, assistant_messages=[{"role": "assistant", "content": "hi"}])

    time.sleep(0.03)
    second = cache.plan_request(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "continue"},
        ],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )

    assert second.resume_session_id is None
    assert cache.stats()["cache_size"] == 0


def test_session_cache_distinguishes_tool_call_histories(tmp_path):
    cache_path = tmp_path / "sessions.sqlite"
    cache = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=10)

    first = cache.plan_request(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    cache.record_success(
        first,
        assistant_messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    }
                ],
            }
        ],
    )

    second = cache.plan_request(
        messages=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"OTHER.md"}'},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )

    assert second.resume_session_id is None


def test_session_cache_evicts_least_recently_used_entries(tmp_path):
    cache_path = tmp_path / "sessions.sqlite"
    cache = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=2)

    first = cache.plan_request(messages=[{"role": "user", "content": "one"}], model="claude-cli", tools=None, tool_choice=None)
    cache.record_success(first, assistant_messages=[{"role": "assistant", "content": "1"}])
    second = cache.plan_request(messages=[{"role": "user", "content": "two"}], model="claude-cli", tools=None, tool_choice=None)
    cache.record_success(second, assistant_messages=[{"role": "assistant", "content": "2"}])
    _ = cache.plan_request(
        messages=[{"role": "user", "content": "one"}, {"role": "assistant", "content": "1"}, {"role": "user", "content": "again"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    third = cache.plan_request(messages=[{"role": "user", "content": "three"}], model="claude-cli", tools=None, tool_choice=None)
    cache.record_success(third, assistant_messages=[{"role": "assistant", "content": "3"}])

    stats = cache.stats()
    assert stats["cache_size"] == 2

    fresh = SessionCache(path=str(cache_path), ttl_seconds=60.0, max_entries=2)
    resumed_two = fresh.plan_request(
        messages=[{"role": "user", "content": "two"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "next"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    resumed_three = fresh.plan_request(
        messages=[{"role": "user", "content": "three"}, {"role": "assistant", "content": "3"}, {"role": "user", "content": "next"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )
    resumed_one = fresh.plan_request(
        messages=[{"role": "user", "content": "one"}, {"role": "assistant", "content": "1"}, {"role": "user", "content": "next"}],
        model="claude-cli",
        tools=None,
        tool_choice=None,
    )

    assert resumed_two.resume_session_id is None
    assert resumed_three.resume_session_id == third.session_id
    assert resumed_one.resume_session_id == first.session_id
