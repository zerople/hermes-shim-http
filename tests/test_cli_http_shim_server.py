import json
import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from hermes_shim_http import __version__
from hermes_shim_http.models import CliRunResult, CliStreamEvent, ShimConfig
from hermes_shim_http.server import create_app


def _client():
    app = create_app(
        ShimConfig(
            command="claude",
            args=["-p"],
            cwd="/tmp",
            timeout=30.0,
            models=["sonnet"],
            http_heartbeat_interval=0,
        )
    )
    return TestClient(app)


def _client_with_config(**overrides):
    overrides.setdefault("http_heartbeat_interval", 0)
    overrides.setdefault("models", ["claude-cli"])
    config = ShimConfig(
        command="claude",
        args=["-p"],
        cwd="/tmp",
        timeout=30.0,
        **overrides,
    )
    return TestClient(create_app(config))


def test_models_endpoint_returns_configured_models():
    client = _client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "sonnet"


def test_model_detail_endpoint_returns_configured_model():
    client = _client()

    response = client.get("/v1/models/sonnet")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "sonnet"
    assert payload["object"] == "model"


def test_model_detail_endpoint_returns_404_for_unknown_model():
    client = _client()

    response = client.get("/v1/models/not-a-real-model")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_health_endpoint_returns_ok():
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_probe_endpoints_return_benign_compatibility_responses():
    client = _client()

    assert client.get("/api/v1/models").status_code == 200
    assert client.get("/api/tags").status_code == 200
    assert client.get("/v1/props").status_code == 200
    assert client.get("/props").status_code == 200
    assert client.get("/version").status_code == 200
    assert client.get("/v1/info").status_code == 200
    assert client.get("/info").status_code == 200

    assert client.get("/api/v1/models").json()["object"] == "list"
    assert client.get("/api/tags").json()["models"][0]["name"] == "sonnet"
    assert client.get("/v1/props").json()["api_mode"] == "chat_completions"
    assert client.get("/props").json()["provider_label"] == "cli-http-shim"
    assert client.get("/version").json()["version"] == __version__


def test_info_endpoint_reports_capabilities_and_context_window_for_claude():
    client = _client()

    payload = client.get("/v1/info").json()

    assert payload["server"] == "hermes-shim-http"
    assert payload["version"] == __version__
    assert payload["cli_profile"] == "claude"
    assert payload["model_id"] == "sonnet"
    assert payload["max_input_length"] == 200_000
    assert payload["max_total_tokens"] == 200_000
    assert payload["max_concurrent_requests"] is None
    assert payload["capabilities"]["streaming"] is True
    assert payload["capabilities"]["tools"] is True
    assert payload["capabilities"]["prompt_caching"] is True
    assert payload["capabilities"]["session_resume"] is True
    assert payload["api_modes"] == ["chat_completions", "responses"]
    assert payload["models"][0]["context_length"] == 200_000


def test_info_endpoint_reports_1m_context_for_opus():
    client = _client_with_config(models=["opus"])

    payload = client.get("/v1/info").json()

    assert payload["model_id"] == "opus"
    assert payload["max_input_length"] == 1_000_000
    assert payload["max_total_tokens"] == 1_000_000
    assert payload["models"][0]["context_length"] == 1_000_000


def test_models_endpoint_includes_context_length_metadata():
    client = _client()

    data = client.get("/v1/models").json()["data"]

    for entry in data:
        assert entry["context_length"] == 200_000
        assert entry["max_model_len"] == 200_000
        assert entry["max_completion_tokens"] == 64_000


def test_models_endpoint_reports_1m_context_for_opus():
    client = _client_with_config(models=["opus"])

    entry = client.get("/v1/models").json()["data"][0]

    assert entry["id"] == "opus"
    assert entry["context_length"] == 1_000_000
    assert entry["max_model_len"] == 1_000_000


def test_chat_completions_returns_plain_text():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=10),
    ) as mock_run:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Hello from Claude"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert mock_run.call_args.kwargs["session_id"]
    assert mock_run.call_args.kwargs["resume_session_id"] is None


def test_non_claude_chat_completions_keep_full_transcript_without_resume():
    client = TestClient(
        create_app(
            ShimConfig(
                command="codex",
                args=["exec"],
                cwd="/tmp",
                timeout=30.0,
                models=["codex"],
                http_heartbeat_interval=0,
            )
        )
    )

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="ok", stderr="", exit_code=0, duration_ms=10),
    ) as mock_run:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "codex",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "continue"},
                ],
            },
        )

    assert response.status_code == 200
    assert "<user>\nhello\n</user>\n\n<assistant>\nhi\n</assistant>\n\n<user>\ncontinue\n</user>" in mock_run.call_args.args[0]
    assert mock_run.call_args.kwargs["resume_session_id"] is None


def test_chat_completions_logs_request_summary(capsys):
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "stream": False,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    stdout = capsys.readouterr().out
    summary_logs = [line for line in stdout.splitlines() if "chat_completions_request" in line]
    assert summary_logs
    payload = json.loads(summary_logs[-1].split("[hermes-shim-http] ", 1)[1])
    assert payload["event"] == "chat_completions_request"
    assert payload["model"] == "sonnet"
    assert payload["stream"] is False
    assert payload["message_count"] == 1
    assert payload["tool_count"] == 1
    assert payload["tool_names"] == ["read_file"]
    assert payload["last_user_message_len"] == 5
    assert payload["request_json_bytes"] > 0


def test_chat_completions_returns_tool_calls():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(
            stdout='<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call>',
            stderr="",
            exit_code=0,
            duration_ms=10,
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Read the readme"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_chat_completions_reuses_prefix_matched_session_on_followup_request():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="ok", stderr="", exit_code=0, duration_ms=10),
    ) as mock_run:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "continue"},
                ],
            },
        )

    assert response.status_code == 200
    first_call = mock_run.call_args_list[0].kwargs
    second_call = mock_run.call_args_list[1].kwargs
    assert second_call["resume_session_id"] == first_call["session_id"]
    assert second_call["session_id"] != first_call["session_id"]


def test_chat_completions_rejects_tool_calls_not_advertised_in_request():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(
            stdout='<tool_call>{"id":"call_1","type":"function","function":{"name":"browser_navigate","arguments":"{\\"url\\":\\"https://example.com\\"}"}}</tool_call>',
            stderr="",
            exit_code=0,
            duration_ms=10,
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Do something"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == "Wrapped CLI emitted unsupported tool call(s): browser_navigate"
    assert "tool_calls" not in payload["choices"][0]["message"]


def test_chat_completions_streaming_returns_live_sse_for_plain_text():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(kind="text", text="Hello "),
                CliStreamEvent(kind="text", text="from "),
                CliStreamEvent(kind="text", text="Claude"),
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "chat.completion.chunk" in body
    assert "Hello" in body
    assert "Claude" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_chat_completions_streaming_returns_live_tool_call_chunks():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(kind="text", text="Using tool: read_file\n"),
                CliStreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    },
                )
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Read the readme"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert 'Using tool: read_file ```path=README.md```' in body
    assert '"tool_calls"' in body
    assert '"read_file"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert "data: [DONE]" in body


def test_tool_progress_preview_shows_primary_args_inline():
    from hermes_shim_http.server import _tool_progress_preview

    assert _tool_progress_preview("terminal", '{"command":"git status"}') == " ```command=git status```"
    assert _tool_progress_preview("read_file", {"path": "/etc/hosts", "offset": 1}) == " ```path=/etc/hosts```"
    assert _tool_progress_preview("patch", '{"path":"a.py","mode":"replace"}') == " ```path=a.py mode=replace```"
    assert _tool_progress_preview("search_files", '{"pattern":"TODO","path":"src"}') == " ```pattern=TODO path=src```"
    # Long values are truncated with an ellipsis so the progress line stays compact.
    long_cmd = "x" * 200
    preview = _tool_progress_preview("terminal", json.dumps({"command": long_cmd}))
    assert preview.startswith(" ```command=")
    assert preview.endswith("…```")
    assert len(preview) < 110
    # Newlines collapse to a single space.
    assert _tool_progress_preview("terminal", '{"command":"a\\nb"}') == " ```command=a b```"
    # Unknown tool with a single scalar arg surfaces it; empty payload stays silent.
    assert _tool_progress_preview("some_future_tool", '{"query":"hi"}') == " ```query=hi```"
    assert _tool_progress_preview("terminal", "") == ""
    assert _tool_progress_preview("terminal", "{not json") == ""
    assert _tool_progress_preview("", '{"command":"x"}') == ""


def test_chat_completions_streaming_progress_text_omits_preview_when_no_primary_field():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            # Empty args dict — preview should stay quiet so we don't print "Using tool: terminal \n".
                            "arguments": "{}",
                        },
                    },
                )
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "run it"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "description": "Run a command",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert "Using tool: terminal\\n" in body
    assert "Using tool: terminal \\n" not in body


def test_chat_completions_drops_native_claude_tool_without_hermes_equivalent():
    """Claude natives with no Hermes mapping (e.g. WebSearch) are dropped silently.

    Claude Code executed them internally via --dangerously-skip-permissions, so
    emitting "unsupported" would both surface a confusing error to the user and
    undo the internal tool result the model already reasoned against.
    """
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(
            stdout='<tool_call>{"id":"call_1","type":"function","function":{"name":"WebSearch","arguments":"{\\"query\\":\\"x\\"}"}}</tool_call>',
            stderr="",
            exit_code=0,
            duration_ms=10,
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Search"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert "unsupported" not in (payload["choices"][0]["message"].get("content") or "")
    assert "tool_calls" not in payload["choices"][0]["message"]


def test_chat_completions_streaming_drops_native_claude_tool_without_equivalent():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(kind="text", text="Let me check. "),
                CliStreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "WebSearch",
                            "arguments": '{"query":"x"}',
                        },
                    },
                ),
                CliStreamEvent(kind="text", text="Found it."),
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Search"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert "unsupported" not in body
    assert '"tool_calls"' not in body
    assert '"finish_reason": "stop"' in body


def test_chat_completions_streaming_downgrades_unsupported_tool_calls_to_text():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "browser_navigate",
                            "arguments": '{"url":"https://example.com"}',
                        },
                    },
                )
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Do something"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                    }
                ],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert 'unsupported tool call(s): browser_navigate' in body
    assert '"tool_calls"' not in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_responses_endpoint_returns_response_object_for_string_input():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Responses", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "sonnet",
                "input": "Say hello",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][0]["content"][0]["type"] == "output_text"
    assert payload["output"][0]["content"][0]["text"] == "Hello from Responses"


def test_responses_endpoint_returns_function_call_items():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(
            stdout='<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call>',
            stderr="",
            exit_code=0,
            duration_ms=10,
        ),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "sonnet",
                "input": "Read the readme",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["name"] == "read_file"
    assert payload["output"][0]["call_id"] == "call_1"


def test_responses_endpoint_rejects_unadvertised_tool_calls():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(
            stdout='<tool_call>{"id":"call_1","type":"function","function":{"name":"browser_navigate","arguments":"{\\"url\\":\\"https://example.com\\"}"}}</tool_call>',
            stderr="",
            exit_code=0,
            duration_ms=10,
        ),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "sonnet",
                "input": "Do something",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "message"
    assert "unsupported tool call(s): browser_navigate" in payload["output"][0]["content"][0]["text"]


def test_responses_endpoint_rejects_missing_input_with_openai_style_error():
    client = _client()

    response = client.post(
        "/v1/responses",
        json={
            "model": "claude-cli",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "input" in payload["error"]["message"].lower()


def test_responses_endpoint_rejects_invalid_tools_payload():
    client = _client()

    response = client.post(
        "/v1/responses",
        json={
            "model": "sonnet",
            "input": "Do something",
            "tools": [{"type": "function"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "tools[0]" in payload["error"]["message"]


def test_responses_endpoint_rejects_invalid_nested_function_payload():
    client = _client()

    response = client.post(
        "/v1/responses",
        json={
            "model": "sonnet",
            "input": "Do something",
            "tools": [{"type": "function", "function": {}}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "tools[0]" in payload["error"]["message"]


def test_responses_endpoint_streams_text_events():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(kind="text", text="Hello "),
                CliStreamEvent(kind="text", text="Responses"),
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "sonnet",
                "input": "Say hello",
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert "response.created" in body
    assert "response.output_text.delta" in body
    assert "Hello" in body
    assert body.count('response.output_item.done') == 1
    assert "response.completed" in body
    assert "data: [DONE]" in body


def test_responses_endpoint_streams_function_call_events():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter(
            [
                CliStreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    },
                )
            ]
        ),
    ):
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "sonnet",
                "input": "Read the readme",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
                "stream": True,
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert "response.output_item.added" in body
    assert '"type": "function_call"' in body
    assert '"name": "read_file"' in body
    assert "response.completed" in body
    assert "data: [DONE]" in body


def test_debug_stats_endpoint_returns_observability_fields():
    client = _client()

    response = client.get("/v1/debug/stats")

    assert response.status_code == 200
    payload = response.json()
    for key in [
        "cache_size",
        "hit_rate",
        "avg_latency_ms",
        "active_sessions",
        "uptime_s",
        "avg_context_tokens_used",
        "max_context_tokens_used",
    ]:
        assert key in payload


def test_debug_stats_updates_after_requests_and_debug_quota_is_unknown():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=25),
    ):
        client.post(
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "hello"}]},
        )

    stats = client.get("/v1/debug/stats")
    quota = client.get("/v1/debug/quota")

    assert stats.status_code == 200
    assert stats.json()["cache_size"] >= 1
    assert stats.json()["active_sessions"] >= 1
    assert stats.json()["avg_latency_ms"] >= 25
    assert quota.json() == {"status": "unknown"}


def test_chat_completions_include_context_and_response_token_metadata():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "Say hello with metadata"}]},
        )

    payload = response.json()
    assert payload["usage"]["context_tokens_used"] > 0
    assert payload["usage"]["context_tokens_limit"] >= payload["usage"]["context_tokens_used"]
    assert payload["usage"]["response_tokens"] > 0


def test_chat_completions_streaming_finishes_with_token_metadata():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter([CliStreamEvent(kind="text", text="Hello")]),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "Say hello"}], "stream": True},
        ) as response:
            body = response.read().decode()

    assert '"context_tokens_used":' in body
    assert '"response_tokens":' in body


def test_chat_completions_sets_compaction_header_when_threshold_triggers():
    client = _client_with_config(compaction="window", compaction_threshold=0.01)
    messages = [{"role": "user", "content": "very long message " * 80} for _ in range(6)]

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="compacted", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post("/v1/chat/completions", json={"model": "claude-cli", "messages": messages})

    assert response.headers["X-Context-Compacted"] == "true"


def test_slash_commands_are_handled_without_invoking_cli():
    client = _client_with_config(compaction="off", compaction_threshold=0.9)

    with patch("hermes_shim_http.server.run_cli_prompt") as mock_run:
        clear_response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "/clear"}]},
        )
        compact_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-cli",
                "messages": [
                    {"role": "user", "content": "older context " * 60},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "/compact"},
                ],
            },
        )
        stats_response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "/stats"}]},
        )

    assert mock_run.call_count == 0
    assert "cleared" in clear_response.json()["choices"][0]["message"]["content"].lower()
    assert "compaction" in compact_response.json()["choices"][0]["message"]["content"].lower()
    assert compact_response.headers["X-Context-Compacted"] == "true"
    assert client.get("/v1/debug/stats").json()["compactions"] >= 1
    assert "cache" in stats_response.json()["choices"][0]["message"]["content"].lower()


def test_compact_command_applies_pending_compaction_to_next_chat_request():
    client = _client_with_config(compaction="off", compaction_threshold=0.9)
    base_messages = [
        {"role": "user", "content": "older context " * 60},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "follow-up question"},
        {"role": "assistant", "content": "follow-up answer"},
    ]

    compact_response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-cli", "messages": [*base_messages, {"role": "user", "content": "/compact"}]},
    )
    assert compact_response.headers["X-Context-Compacted"] == "true"
    compaction_token = compact_response.headers["X-Compaction-Token"]

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="after compact", stderr="", exit_code=0, duration_ms=10),
    ) as mock_run:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Compaction-Token": compaction_token},
            json={"model": "claude-cli", "messages": [*base_messages, {"role": "user", "content": "continue"}]},
        )

    assert response.status_code == 200
    assert response.headers["X-Context-Compacted"] == "true"
    sent_prompt = mock_run.call_args.args[0]
    assert "Summary of earlier conversation" in sent_prompt


def test_model_slash_command_overrides_response_model_without_invoking_cli():
    client = _client()

    with patch("hermes_shim_http.server.run_cli_prompt") as mock_run:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "/model sonnet"}]},
        )

    assert mock_run.call_count == 0
    assert response.json()["model"] == "sonnet"


def test_responses_slash_compact_is_handled_locally_without_invoking_cli():
    client = _client()

    with patch("hermes_shim_http.server.run_cli_prompt") as mock_run:
        response = client.post(
            "/v1/responses",
            json={"model": "claude-cli", "input": "/compact"},
        )

    assert mock_run.call_count == 0
    assert response.headers["X-Context-Compacted"] == "true"
    assert "compaction" in response.json()["output"][0]["content"][0]["text"].lower()


def test_responses_compact_command_applies_pending_compaction_to_next_request():
    client = _client()
    base_messages = [
        {"role": "user", "content": "older context " * 60},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "follow-up question"},
        {"role": "assistant", "content": "follow-up answer"},
    ]

    compact_response = client.post(
        "/v1/responses",
        json={"model": "claude-cli", "input": [*base_messages, {"role": "user", "content": "/compact"}]},
    )
    assert compact_response.headers["X-Context-Compacted"] == "true"
    compaction_token = compact_response.headers["X-Compaction-Token"]

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="after compact", stderr="", exit_code=0, duration_ms=10),
    ) as mock_run:
        response = client.post(
            "/v1/responses",
            headers={"X-Compaction-Token": compaction_token},
            json={"model": "claude-cli", "input": [*base_messages, {"role": "user", "content": "continue"}]},
        )

    assert response.status_code == 200
    assert response.headers["X-Context-Compacted"] == "true"
    sent_prompt = mock_run.call_args.args[0]
    assert "Summary of earlier conversation" in sent_prompt


def test_streaming_keepalive_emits_ping_comments_during_idle_periods(monkeypatch):
    client = _client()

    def slow_events():
        yield CliStreamEvent(kind="text", text="hello")
        import time

        time.sleep(0.03)
        yield CliStreamEvent(kind="text", text=" world")

    monkeypatch.setattr("hermes_shim_http.server.KEEPALIVE_INTERVAL_SECONDS", 0.01, raising=False)
    with patch("hermes_shim_http.server.stream_cli_prompt", return_value=slow_events()):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "claude-cli", "messages": [{"role": "user", "content": "Say hello"}], "stream": True},
        ) as response:
            body = response.read().decode()

    assert ": ping\n\n" in body


def test_structured_logging_emits_request_ids_and_cache_events(caplog):
    client = _client_with_config(log_level="debug", log_format="json")

    with caplog.at_level(logging.DEBUG, logger="hermes_shim_http"):
        with patch(
            "hermes_shim_http.server.run_cli_prompt",
            return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=10),
        ):
            client.post(
                "/v1/chat/completions",
                json={"model": "claude-cli", "messages": [{"role": "user", "content": "hello"}]},
            )

    assert any('"event": "cache_miss"' in record.message for record in caplog.records)
    assert any('"event": "spawn"' in record.message for record in caplog.records)
    assert all('"request_id":' in record.message for record in caplog.records if '"event":' in record.message)


# ---------------------------------------------------------------------------
# Silence sentinel (`<silent/>`) — intentional empty-response signaling.
# An empty response without the sentinel is still treated as an error by the
# upstream client; only an explicit sentinel marks the turn as silent ACK.
# ---------------------------------------------------------------------------


def test_chat_completions_marks_silent_when_sentinel_is_only_output():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="<silent/>", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "sonnet", "messages": [{"role": "user", "content": "noop"}]},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Shim-Silent") == "true"
    payload = response.json()
    assert payload["choices"][0]["silent"] is True
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_keeps_content_when_sentinel_is_mixed_with_text():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="hi <silent/>", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "sonnet", "messages": [{"role": "user", "content": "noop"}]},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Shim-Silent") is None
    payload = response.json()
    assert "silent" not in payload["choices"][0]
    assert payload["choices"][0]["message"]["content"] == "hi"


def test_chat_completions_streaming_marks_silent_in_final_chunk():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter([CliStreamEvent(kind="text", text="<silent/>")]),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "sonnet", "messages": [{"role": "user", "content": "noop"}], "stream": True},
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert '"silent": true' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_responses_endpoint_marks_silent_when_sentinel_is_only_output():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="<silent/>", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "sonnet", "input": "noop"},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Shim-Silent") == "true"
    payload = response.json()
    assert payload["silent"] is True
    assert payload["status"] == "completed"


def test_responses_streaming_marks_silent_in_completed_event():
    client = _client()

    with patch(
        "hermes_shim_http.server.stream_cli_prompt",
        return_value=iter([CliStreamEvent(kind="text", text="<silent/>")]),
    ):
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "sonnet", "input": "noop", "stream": True},
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    assert '"silent": true' in body
    assert '"response.completed"' in body
    assert "data: [DONE]" in body


def test_chat_completions_silent_flag_emits_silent_log_event(caplog):
    client = _client_with_config(log_level="debug", log_format="json")

    with caplog.at_level(logging.DEBUG, logger="hermes_shim_http"):
        with patch(
            "hermes_shim_http.server.run_cli_prompt",
            return_value=CliRunResult(stdout="<silent/>", stderr="", exit_code=0, duration_ms=10),
        ):
            client.post(
                "/v1/chat/completions",
                json={"model": "claude-cli", "messages": [{"role": "user", "content": "noop"}]},
            )

    assert any('"event": "silent"' in record.message for record in caplog.records)


def test_silence_sentinel_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_SHIM_SILENT_SENTINEL", "<<<HUSH>>>")
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="<<<HUSH>>>", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "sonnet", "messages": [{"role": "user", "content": "noop"}]},
        )

    assert response.headers.get("X-Shim-Silent") == "true"
    assert response.json()["choices"][0]["silent"] is True


def test_empty_response_without_sentinel_is_not_marked_silent():
    """An empty CLI response without the sentinel is treated as a normal empty
    completion (not silent). Upstream clients can decide whether to retry."""
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "sonnet", "messages": [{"role": "user", "content": "noop"}]},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Shim-Silent") is None
    payload = response.json()
    assert "silent" not in payload["choices"][0]
    assert payload["choices"][0]["message"]["content"] == ""
