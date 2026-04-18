from __future__ import annotations

import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from hermes_shim_http.inflight import InFlightRegistry
from hermes_shim_http.models import CliRunResult, ShimConfig
from hermes_shim_http.server import create_app


def _client(**overrides) -> TestClient:
    overrides.setdefault("http_heartbeat_interval", 0)
    config = ShimConfig(
        command="claude",
        args=["-p"],
        cwd="/tmp",
        timeout=30.0,
        models=["sonnet"],
        **overrides,
    )
    return TestClient(create_app(config))


def test_in_flight_registry_rejects_concurrent_duplicates():
    registry = InFlightRegistry()
    assert registry.reserve("abc")
    assert not registry.reserve("abc")
    registry.release("abc")
    assert registry.reserve("abc")


def test_in_flight_registry_ignores_empty_key():
    registry = InFlightRegistry()
    assert registry.reserve("")
    assert registry.reserve("")  # empty keys never collide


def test_in_flight_registry_expires_stale_reservations():
    registry = InFlightRegistry(stale_after_seconds=0.01)
    assert registry.reserve("stale")
    time.sleep(0.02)
    assert registry.reserve("stale")  # previous reservation is now stale


def test_chat_completions_rejects_duplicate_in_flight_request():
    client = _client()
    gate = threading.Event()

    def _slow_cli(*args, **kwargs):
        gate.wait(timeout=5.0)
        return CliRunResult(stdout="done", stderr="", exit_code=0, duration_ms=1)

    first_response: list = []

    def _first_request() -> None:
        with patch("hermes_shim_http.server.run_cli_prompt", side_effect=_slow_cli):
            resp = client.post(
                "/v1/chat/completions",
                headers={"Idempotency-Key": "dup-key"},
                json={"model": "sonnet", "messages": [{"role": "user", "content": "hello"}]},
            )
            first_response.append(resp)

    thread = threading.Thread(target=_first_request)
    thread.start()
    # Give the first request a chance to reserve the key before the duplicate fires.
    time.sleep(0.05)

    duplicate = client.post(
        "/v1/chat/completions",
        headers={"Idempotency-Key": "dup-key"},
        json={"model": "sonnet", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert duplicate.status_code == 409
    payload = duplicate.json()
    assert payload["error"]["type"] == "duplicate_request"
    assert payload["error"]["idempotency_key"] == "dup-key"
    assert duplicate.headers["Retry-After"] == "5"

    gate.set()
    thread.join(timeout=5.0)
    assert first_response and first_response[0].status_code == 200


def test_chat_completions_releases_idempotency_key_after_success():
    client = _client()
    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="done", stderr="", exit_code=0, duration_ms=1),
    ):
        first = client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "reused"},
            json={"model": "sonnet", "messages": [{"role": "user", "content": "hello"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "reused"},
            json={"model": "sonnet", "messages": [{"role": "user", "content": "hello again"}]},
        )

    assert first.status_code == 200
    assert second.status_code == 200


def test_chat_completions_accepts_x_request_id_as_idempotency_key():
    client = _client()
    gate = threading.Event()

    def _slow_cli(*args, **kwargs):
        gate.wait(timeout=5.0)
        return CliRunResult(stdout="done", stderr="", exit_code=0, duration_ms=1)

    first_response: list = []

    def _first_request() -> None:
        with patch("hermes_shim_http.server.run_cli_prompt", side_effect=_slow_cli):
            resp = client.post(
                "/v1/chat/completions",
                headers={"X-Request-Id": "req-1"},
                json={"model": "sonnet", "messages": [{"role": "user", "content": "hi"}]},
            )
            first_response.append(resp)

    thread = threading.Thread(target=_first_request)
    thread.start()
    time.sleep(0.05)

    duplicate = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "req-1"},
        json={"model": "sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert duplicate.status_code == 409

    gate.set()
    thread.join(timeout=5.0)
    assert first_response and first_response[0].status_code == 200
