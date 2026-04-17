from unittest.mock import patch

from fastapi.testclient import TestClient

from hermes_shim_http.models import CliRunResult, CliStreamEvent, ShimConfig
from hermes_shim_http.server import create_app


def _client():
    app = create_app(
        ShimConfig(
            command="claude",
            args=["-p"],
            cwd="/tmp",
            timeout=30.0,
            models=["claude-cli"],
        )
    )
    return TestClient(app)


def test_models_endpoint_returns_configured_models():
    client = _client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "claude-cli"


def test_model_detail_endpoint_returns_configured_model():
    client = _client()

    response = client.get("/v1/models/claude-cli")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "claude-cli"
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

    assert client.get("/api/v1/models").json()["object"] == "list"
    assert client.get("/api/tags").json()["models"][0]["name"] == "claude-cli"
    assert client.get("/v1/props").json()["api_mode"] == "chat_completions"
    assert client.get("/props").json()["provider_label"] == "cli-http-shim"
    assert client.get("/version").json()["version"] == "0.1.1"


def test_chat_completions_returns_plain_text():
    client = _client()

    with patch(
        "hermes_shim_http.server.run_cli_prompt",
        return_value=CliRunResult(stdout="Hello from Claude", stderr="", exit_code=0, duration_ms=10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-cli",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Hello from Claude"
    assert payload["choices"][0]["finish_reason"] == "stop"


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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
    assert '"tool_calls"' in body
    assert '"read_file"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert "data: [DONE]" in body


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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
                "model": "claude-cli",
                "input": "Do something",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "message"
    assert "unsupported tool call(s): browser_navigate" in payload["output"][0]["content"][0]["text"]


def test_responses_endpoint_rejects_invalid_tools_payload():
    client = _client()

    response = client.post(
        "/v1/responses",
        json={
            "model": "claude-cli",
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
            "model": "claude-cli",
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
                "model": "claude-cli",
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
                "model": "claude-cli",
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
