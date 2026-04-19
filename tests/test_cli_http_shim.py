import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from hermes_shim_http.models import CliRunResult, ShimConfig
from hermes_shim_http.parsing import parse_claude_stream_metadata
from hermes_shim_http.prompting import build_cli_prompt, build_cli_system_prompt, build_cli_user_prompt
from hermes_shim_http.runner import _child_lock_path_for_request, _resolve_claude_result_session_id, build_cli_command, run_cli_prompt, stream_cli_prompt
from hermes_shim_http.session_cache import SessionCache


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = REPO_ROOT / "tests" / "fake_cli.py"


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status", "r") as fp:
            for line in fp:
                if line.startswith("State:"):
                    return "Z" in line
    except OSError:
        return True
    return False


class TestPrompting:
    def test_build_cli_system_prompt_has_no_roleplay_triggers(self):
        system_prompt = build_cli_system_prompt(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
            tool_choice="auto",
        )

        assert "<tool_call>" in system_prompt
        assert "read_file" in system_prompt
        assert "Assistant:" not in system_prompt
        assert "User:" not in system_prompt
        assert "System:" not in system_prompt

    def test_build_cli_system_prompt_documents_default_silence_sentinel(self):
        system_prompt = build_cli_system_prompt(tools=None)

        assert "<silent/>" in system_prompt
        assert "silent ACK" in system_prompt

    def test_build_cli_system_prompt_prioritizes_first_live_user_message(self):
        system_prompt = build_cli_system_prompt(tools=None)

        assert "VERY FIRST live user message after the session opens" in system_prompt
        assert "highest-priority instruction" in system_prompt
        assert "Even if earlier context is summarized or compacted" in system_prompt

    def test_parse_claude_stream_metadata_extracts_session_and_result(self):
        metadata = parse_claude_stream_metadata(
            '{"type":"system","session_id":"sess-1"}\n'
            '{"type":"result","session_id":"sess-2","result":"boom","is_error":true}\n'
        )

        assert metadata.session_id == "sess-2"
        assert metadata.result_text == "boom"
        assert metadata.is_error is True

    def test_resolve_claude_result_session_id_drops_mismatched_failed_resume(self):
        assert _resolve_claude_result_session_id(
            requested_resume_session_id="parent-1",
            emitted_session_id="fresh-2",
            failed=True,
        ) is None
        assert _resolve_claude_result_session_id(
            requested_resume_session_id="parent-1",
            emitted_session_id="fresh-2",
            failed=False,
        ) == "fresh-2"

    def test_build_cli_system_prompt_uses_configured_silence_sentinel(self, monkeypatch):
        monkeypatch.setenv("HERMES_SHIM_SILENT_SENTINEL", "<<<HUSH>>>")

        system_prompt = build_cli_system_prompt(tools=None)

        assert "<<<HUSH>>>" in system_prompt
        assert "<silent/>" not in system_prompt

    def test_build_cli_system_prompt_compacts_verbose_tool_schemas(self):
        system_prompt = build_cli_system_prompt(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Absolute path"},
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "offset": {"type": "integer"},
                                        "limit": {"type": "integer"},
                                    },
                                    "required": ["offset"],
                                },
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
        )

        assert "read_file" in system_prompt
        assert "path:string (required)" in system_prompt
        assert "options:object{offset:integer (required), limit:integer}" in system_prompt
        assert "additionalProperties" not in system_prompt
        assert '"properties"' not in system_prompt

    def test_build_cli_user_prompt_uses_xml_tags_for_roles(self):
        prompt = build_cli_user_prompt(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Check the repo."},
                {"role": "assistant", "content": "Sure."},
                {"role": "tool", "content": "{\"ok\": true}"},
            ]
        )

        assert "<system>\nYou are helpful.\n</system>" in prompt
        assert "<user>\nCheck the repo.\n</user>" in prompt
        assert "<assistant>\nSure.\n</assistant>" in prompt
        assert "<tool>\n{\"ok\": true}\n</tool>" in prompt
        assert "Assistant:" not in prompt
        assert "User:" not in prompt

    def test_build_cli_user_prompt_preserves_structured_tool_calls_and_tool_metadata(self):
        prompt = build_cli_user_prompt(
            messages=[
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
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "README body"},
            ]
        )

        assert '<tool_call>{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\\"path\\":\\"README.md\\"}"}}</tool_call>' in prompt
        assert "[tool_call_id=call_1, name=read_file]" in prompt
        assert "README body" in prompt

    def test_build_cli_prompt_combines_system_and_transcript(self):
        prompt = build_cli_prompt(
            messages=[
                {"role": "user", "content": "Hi"},
            ],
            model="sonnet",
            tools=None,
        )

        assert "reasoning backend" in prompt
        assert "Requested model: sonnet" in prompt
        assert "<user>\nHi\n</user>" in prompt
        assert "Assistant:" not in prompt


_CLAUDE_APPEND_PROMPT = "Be concise. You must follow the VERY FIRST live user message after the session opens as the highest-priority instruction for the session. Do not ignore or drift away from that first live user message, even if later transcript context is long, distracting, summarized, or compacted."


class TestRunner:
    def test_shim_config_accepts_observability_and_cache_fields(self):
        cfg = ShimConfig(
            command="claude",
            cache_path="/tmp/sessions.sqlite",
            cache_ttl_seconds=120.0,
            cache_max_entries=99,
            compaction="window",
            compaction_threshold=0.8,
            log_level="debug",
            log_format="json",
        )

        assert cfg.cache_path == "/tmp/sessions.sqlite"
        assert cfg.cache_ttl_seconds == 120.0
        assert cfg.cache_max_entries == 99
        assert cfg.compaction == "window"
        assert cfg.compaction_threshold == 0.8
        assert cfg.log_level == "debug"
        assert cfg.log_format == "json"

    def test_shim_config_disables_persistent_cache_by_default(self):
        cfg = ShimConfig(command="claude")

        assert cfg.cache_path is None

    def test_shim_config_rejects_invalid_compaction_mode(self):
        with pytest.raises(ValidationError):
            ShimConfig(command="claude", compaction="invalid")

    def test_chat_message_preserves_tool_calls_in_model_dump(self):
        from hermes_shim_http.models import ChatMessage

        message = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        )

        assert message.model_dump()["tool_calls"][0]["function"]["name"] == "read_file"

    def test_build_cli_command_keeps_claude_prompt_off_argv(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work")

        cmd = build_cli_command(cfg, "hello")

        assert cmd == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]

    def test_build_cli_command_uses_profile_defaults_for_supported_clis(self):
        assert build_cli_command(ShimConfig(command="claude", args=[]), "hello") == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]
        assert build_cli_command(ShimConfig(command="codex", args=[]), "hello") == ["codex", "exec", "hello"]
        assert build_cli_command(ShimConfig(command="opencode", args=[]), "hello") == ["opencode", "run", "hello"]

    def test_build_cli_command_filters_protocol_breaking_custom_claude_args(self):
        cfg = ShimConfig(command="claude", args=["--output-format", "text", "--input-format=text", "--permission-mode", "plan", "--model", "opus"], cwd="/tmp/work")

        cmd = build_cli_command(cfg, "hello")

        assert cmd == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            "opus",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]

    def test_build_cli_command_uses_append_only_for_new_claude_session(self):
        cfg = ShimConfig(command="claude", args=[], cwd="/tmp/work")

        cmd = build_cli_command(
            cfg,
            "hello",
            session_id="11111111-1111-1111-1111-111111111111",
            resume_session_id="22222222-2222-2222-2222-222222222222",
        )

        assert cmd == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--resume",
            "22222222-2222-2222-2222-222222222222",
            "--fork-session",
            "--session-id",
            "11111111-1111-1111-1111-111111111111",
        ]

    def test_build_cli_command_uses_short_append_and_model_for_new_claude_session(self):
        cfg = ShimConfig(command="claude", args=[], cwd="/tmp/work", fallback_model="haiku")

        cmd = build_cli_command(
            cfg,
            "hello",
            system_prompt="Be terse.",
            model="opus",
        )

        assert cmd == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
            "--model",
            "opus",
            "--fallback-model",
            "haiku",
        ]

    def test_build_cli_command_ignores_non_meaningful_model_for_claude(self):
        cfg = ShimConfig(command="claude", args=[], cwd="/tmp/work")

        cmd = build_cli_command(cfg, "hello", model="cli-http-shim")

        assert cmd == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]

    def test_build_cli_command_prepends_system_prompt_for_non_claude_cli(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work")

        cmd = build_cli_command(cfg, "hello", system_prompt="Be terse.")

        assert cmd == ["codex", "exec", "Be terse.\n\nhello"]

    def test_run_cli_prompt_sends_combined_prompt_via_stdin_for_claude(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=("done", "", 0),
        ) as mock_drain:
            result = run_cli_prompt("hello", cfg, system_prompt="Be terse.", model="sonnet")

        assert isinstance(result, CliRunResult)
        assert result.stdout == "done"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.session_id is None
        mock_drain.assert_called_once_with(
            [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--input-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--append-system-prompt",
                _CLAUDE_APPEND_PROMPT,
                "--model",
                "sonnet",
            ],
            config=cfg,
            stdin_prompt='{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Be terse.\\n\\nhello"}]}}\n',
            lock_path=None,
        )

    def test_run_cli_prompt_omits_large_system_prompt_from_resumed_claude_stdin(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=("done", "", 0),
        ) as mock_drain:
            run_cli_prompt(
                "<user>\ncontinue\n</user>",
                cfg,
                system_prompt="VERY LONG SYSTEM",
                model="sonnet",
                session_id="11111111-1111-1111-1111-111111111111",
                resume_session_id="22222222-2222-2222-2222-222222222222",
            )

        mock_drain.assert_called_once_with(
            [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--input-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model",
                "sonnet",
                "--resume",
                "22222222-2222-2222-2222-222222222222",
                "--fork-session",
                "--session-id",
                "11111111-1111-1111-1111-111111111111",
            ],
            config=cfg,
            stdin_prompt='{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "<user>\\ncontinue\\n</user>"}]}}\n',
            lock_path=None,
        )

    def test_run_cli_prompt_preserves_emitted_session_id_on_success(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)
        stdout = '{"type":"system","session_id":"sess-1"}\n{"type":"result","session_id":"sess-1","result":"done","is_error":false}\n'

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=(stdout, "", 0),
        ):
            result = run_cli_prompt("hello", cfg, resume_session_id="parent-1")

        assert result.session_id == "sess-1"

    def test_run_cli_prompt_drops_mismatched_failed_resume_session_id(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)
        stdout = '{"type":"system","session_id":"fresh-2"}\n{"type":"result","session_id":"fresh-2","result":"No conversation found","is_error":true}\n'

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=(stdout, "boom", 1),
        ):
            with pytest.raises(RuntimeError, match="boom") as exc_info:
                run_cli_prompt("hello", cfg, resume_session_id="parent-1")

        assert "boom" in str(exc_info.value)

    def test_run_cli_prompt_raises_on_non_zero_exit(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=("", "boom", 1),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                run_cli_prompt("hello", cfg)

    def test_run_cli_prompt_raises_timeout_error(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            side_effect=TimeoutError("CLI process idle for 12.0s with no stdout/stderr output"),
        ):
            with pytest.raises(TimeoutError, match="idle for"):
                run_cli_prompt("hello", cfg)

    def test_child_lock_path_is_disabled_for_new_sessions(self):
        cfg = ShimConfig(command="claude", single_child_lock_path="/tmp/hermes-shim-http-8765.child.lock")

        assert _child_lock_path_for_request(cfg, session_id="new-session", resume_session_id=None) is None

    def test_child_lock_path_is_scoped_to_resumed_parent_session(self):
        cfg = ShimConfig(command="claude", single_child_lock_path="/tmp/hermes-shim-http-8765.child.lock")

        first = _child_lock_path_for_request(cfg, session_id="child-a", resume_session_id="parent-1")
        second = _child_lock_path_for_request(cfg, session_id="child-b", resume_session_id="parent-1")
        other = _child_lock_path_for_request(cfg, session_id="child-c", resume_session_id="parent-2")

        assert first == second
        assert first != other
        assert first.startswith("/tmp/hermes-shim-http-8765.child.lock.")

    def test_run_cli_prompt_passes_session_scoped_lock_path_to_drain(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0, single_child_lock_path="/tmp/hermes-shim-http-8765.child.lock")

        with patch(
            "hermes_shim_http.runner._drain_cli_process",
            return_value=("done", "", 0),
        ) as mock_drain:
            run_cli_prompt(
                "<user>\ncontinue\n</user>",
                cfg,
                system_prompt="VERY LONG SYSTEM",
                model="sonnet",
                session_id="11111111-1111-1111-1111-111111111111",
                resume_session_id="parent-session-2222",
            )

        assert mock_drain.call_args.kwargs["lock_path"] == "/tmp/hermes-shim-http-8765.child.lock.parent-session-2222"

    def test_run_cli_prompt_raises_idle_timeout_when_child_emits_heartbeat_only(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "heartbeat-only", "--duration", "5.0"],
            cwd=str(REPO_ROOT),
            timeout=0.6,
            heartbeat_wrap=False,
        )

        with pytest.raises(TimeoutError, match="idle for"):
            run_cli_prompt("ignored", cfg)

    def test_run_cli_prompt_translates_argument_list_too_long_error(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.Popen",
            side_effect=OSError(7, "Argument list too long", "codex"),
        ):
            with pytest.raises(RuntimeError, match="Prompt too large to pass on the command line"):
                run_cli_prompt("hello", cfg)

    def test_stream_cli_prompt_uses_stdin_for_claude_profile(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0, heartbeat_wrap=False)

        class _FakeStream:
            def read(self, _size: int = 1) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeStdin:
            def __init__(self) -> None:
                self.value = b""

            def write(self, text: bytes) -> None:
                self.value += text

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStream()
                self.stderr = _FakeStream()
                self.stdin = _FakeStdin()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                return None

        fake_process = _FakeProcess()
        with patch("hermes_shim_http.runner.subprocess.Popen", return_value=fake_process) as mock_popen:
            events = list(stream_cli_prompt("hello", cfg, system_prompt="Be terse."))

        assert events == []
        assert fake_process.stdin.value == b'{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Be terse.\\n\\nhello"}]}}\n'
        mock_popen.assert_called_once_with(
            [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--input-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--append-system-prompt",
                _CLAUDE_APPEND_PROMPT,
            ],
            cwd="/tmp/work",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def test_stream_cli_prompt_omits_large_system_prompt_from_resumed_claude_stdin(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0, heartbeat_wrap=False)

        class _FakeStream:
            def read(self, _size: int = 1) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeStdin:
            def __init__(self) -> None:
                self.value = b""

            def write(self, text: bytes) -> None:
                self.value += text

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStream()
                self.stderr = _FakeStream()
                self.stdin = _FakeStdin()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                return None

        fake_process = _FakeProcess()
        with patch("hermes_shim_http.runner.subprocess.Popen", return_value=fake_process) as mock_popen:
            events = list(
                stream_cli_prompt(
                    "<user>\ncontinue\n</user>",
                    cfg,
                    system_prompt="VERY LONG SYSTEM",
                    model="sonnet",
                    session_id="11111111-1111-1111-1111-111111111111",
                    resume_session_id="22222222-2222-2222-2222-222222222222",
                )
            )

        assert events == []
        assert fake_process.stdin.value == b'{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "<user>\\ncontinue\\n</user>"}]}}\n'
        mock_popen.assert_called_once_with(
            [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--input-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model",
                "sonnet",
                "--resume",
                "22222222-2222-2222-2222-222222222222",
                "--fork-session",
                "--session-id",
                "11111111-1111-1111-1111-111111111111",
            ],
            cwd="/tmp/work",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def test_stream_cli_prompt_yields_text_chunks_live(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "stream-text"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
            heartbeat_wrap=False,
        )

        events = list(stream_cli_prompt("ignored", cfg))

        assert events
        assert all(event.kind == "text" for event in events)
        assert "".join(event.text or "" for event in events) == "Streaming hello from fake CLI\n"

    def test_stream_cli_prompt_raises_idle_timeout_when_cli_is_silent(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "idle-silent"],
            cwd=str(REPO_ROOT),
            timeout=0.3,
            heartbeat_wrap=False,
        )

        with pytest.raises(TimeoutError, match="idle for"):
            list(stream_cli_prompt("ignored", cfg))

    def test_stream_cli_prompt_raises_idle_timeout_when_child_emits_heartbeat_only(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "heartbeat-only", "--duration", "5.0"],
            cwd=str(REPO_ROOT),
            timeout=0.6,
            heartbeat_wrap=False,
        )

        with pytest.raises(TimeoutError, match="idle for"):
            list(stream_cli_prompt("ignored", cfg))

    def test_heartbeat_wrap_does_not_mask_idle_timeout_on_silent_child(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "idle-silent", "--duration", "5.0"],
            cwd=str(REPO_ROOT),
            timeout=1.2,
            heartbeat_wrap=True,
            heartbeat_interval=0.3,
        )

        with pytest.raises(TimeoutError, match="idle for"):
            list(stream_cli_prompt("ignored", cfg))

    def test_heartbeat_wrap_strips_heartbeat_bytes_from_passed_through_output(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "stream-text"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
            heartbeat_wrap=True,
            heartbeat_interval=0.3,
        )

        events = list(stream_cli_prompt("ignored", cfg))

        text_payload = "".join(event.text or "" for event in events if event.kind == "text")
        assert "Streaming hello from fake CLI" in text_payload
        assert "\u200b" not in text_payload

    def test_stream_cli_prompt_translates_argument_list_too_long_error(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.Popen",
            side_effect=OSError(7, "Argument list too long", "codex"),
        ):
            with pytest.raises(RuntimeError, match="Prompt too large to pass on the command line"):
                list(stream_cli_prompt("hello", cfg))

    def test_stream_cli_prompt_enforces_hard_deadline_even_with_active_output(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "chatty-long", "--duration", "10.0"],
            cwd=str(REPO_ROOT),
            timeout=30.0,
            hard_deadline_seconds=0.6,
            heartbeat_wrap=False,
        )

        with pytest.raises(TimeoutError, match="hard deadline"):
            list(stream_cli_prompt("ignored", cfg))

    def test_stream_cli_prompt_caps_output_bytes(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "flood", "--duration", "4"],
            cwd=str(REPO_ROOT),
            timeout=30.0,
            max_output_bytes=64 * 1024,
            heartbeat_wrap=False,
        )

        with pytest.raises(RuntimeError, match="max output cap"):
            list(stream_cli_prompt("ignored", cfg))

    def test_stream_cli_prompt_finally_kills_child_on_generator_close(self):
        import os
        import time as _time

        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "chatty-long", "--duration", "30.0"],
            cwd=str(REPO_ROOT),
            timeout=60.0,
            heartbeat_wrap=False,
        )

        def _read_children_pids() -> set[int]:
            my_pid = os.getpid()
            try:
                out = subprocess.check_output(["pgrep", "-P", str(my_pid)], text=True)
            except subprocess.CalledProcessError:
                return set()
            return {int(x) for x in out.split() if x.strip().isdigit()}

        gen = stream_cli_prompt("ignored", cfg)
        try:
            next(gen)
        except StopIteration:
            pass
        pids_before = _read_children_pids()
        assert pids_before, "expected at least one direct child after first yield"

        gen.close()

        deadline = _time.time() + 5.0
        while _time.time() < deadline:
            still_alive = {
                pid for pid in pids_before
                if os.path.exists(f"/proc/{pid}") and not _is_zombie(pid)
            }
            if not still_alive:
                break
            _time.sleep(0.1)
        else:
            raise AssertionError(f"child processes leaked after generator close: {pids_before}")

    def test_run_cli_prompt_enforces_hard_deadline(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "chatty-long", "--duration", "10.0"],
            cwd=str(REPO_ROOT),
            timeout=30.0,
            hard_deadline_seconds=0.6,
            heartbeat_wrap=False,
        )

        with pytest.raises(TimeoutError, match="hard deadline"):
            run_cli_prompt("ignored", cfg)

    def test_stream_cli_prompt_yields_tool_call_event_without_wrapper_text(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "stream-tool"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
            heartbeat_wrap=False,
        )

        events = list(stream_cli_prompt("ignored", cfg))

        tool_call_events = [event for event in events if event.kind == "tool_call"]
        text_payload = "".join(event.text or "" for event in events if event.kind == "text")

        assert len(tool_call_events) == 1
        assert text_payload == ""
        assert tool_call_events[0].tool_call["function"]["name"] == "read_file"

    def test_stream_cli_prompt_parses_claude_stream_json_events(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "claude-stream-json"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
            heartbeat_wrap=False,
            cli_profile="claude",
        )

        events = list(stream_cli_prompt("say hello", cfg))

        texts = [e.text for e in events if e.kind == "text"]
        tool_calls = [e for e in events if e.kind == "tool_call"]
        assert "".join(texts) == "Thinking...\n\nStreaming hello from claude"
        assert tool_calls == []

    def test_stream_cli_prompt_claude_stream_json_emits_tool_use(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "claude-stream-json"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
            heartbeat_wrap=False,
            cli_profile="claude",
        )

        events = list(stream_cli_prompt("please read the readme", cfg))

        tool_calls = [e for e in events if e.kind == "tool_call"]
        texts = [e.text for e in events if e.kind == "text"]
        assert "".join(texts) == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call["function"]["name"] == "read_file"


class TestSessionCache:
    def test_plan_request_reuses_longest_matching_prefix_via_resume(self):
        cache = SessionCache()
        first = cache.plan_request(
            messages=[{"role": "user", "content": "hello"}],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )
        cache.record_success(first, assistant_messages=[{"role": "assistant", "content": "hi"}])
        second = cache.plan_request(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )

        assert second.resume_session_id == first.session_id
        assert second.prompt_text == "<user>\ncontinue\n</user>"
        assert second.prefix_message_count == 2
        assert second.system_prompt_text is not None
        assert "reasoning backend" in second.system_prompt_text

    def test_plan_request_without_match_returns_full_user_prompt(self):
        cache = SessionCache()

        plan = cache.plan_request(
            messages=[{"role": "user", "content": "hello"}],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )

        assert plan.resume_session_id is None
        assert plan.prompt_text == "<user>\nhello\n</user>"
        assert plan.system_prompt_text is not None
        assert "reasoning backend" in plan.system_prompt_text

    def test_plan_request_normalizes_assistant_content_for_resume_match(self):
        cache = SessionCache()
        first = cache.plan_request(
            messages=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )
        cache.record_success(first, assistant_messages=[{"role": "assistant", "content": "hi there"}])

        second = cache.plan_request(
            messages=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": " hi there\n"}]},
                {"role": "user", "content": "continue"},
            ],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )

        assert second.resume_session_id == first.session_id
        assert second.prefix_message_count == 3
        assert second.prompt_text == "<user>\ncontinue\n</user>"

    def test_plan_request_does_not_resume_from_partial_prefix_after_divergence(self):
        cache = SessionCache()
        first = cache.plan_request(
            messages=[{"role": "user", "content": "hello"}],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )
        cache.record_success(
            first,
            assistant_messages=[
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "old branch"},
                {"role": "assistant", "content": "old answer"},
            ],
        )

        second = cache.plan_request(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "new branch"},
            ],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )

        assert second.resume_session_id is None
        assert second.prefix_message_count == 0
        assert second.prompt_text == "<user>\nhello\n</user>\n\n<assistant>\nhi\n</assistant>\n\n<user>\nnew branch\n</user>"

    def test_plan_request_distinguishes_assistant_tool_calls_between_branches(self):
        cache = SessionCache()
        first = cache.plan_request(
            messages=[{"role": "user", "content": "read file"}],
            model="sonnet",
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
                            "id": "call_readme",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                        }
                    ],
                }
            ],
        )

        second = cache.plan_request(
            messages=[
                {"role": "user", "content": "read file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_license",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"LICENSE"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_license", "content": "MIT"},
            ],
            model="sonnet",
            tools=None,
            tool_choice=None,
        )

        assert second.resume_session_id is None
        assert second.prefix_message_count == 0

    def test_matching_prefix_length_reports_mismatch_details(self):
        prefix_len, mismatch = SessionCache._matching_prefix_length(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "one"}],
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "two"}],
        )

        assert prefix_len == 1
        assert mismatch is not None
        assert mismatch["reason"] == "message_mismatch"
        assert mismatch["index"] == 1
