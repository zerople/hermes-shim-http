import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from hermes_shim_http.models import CliRunResult, ShimConfig
from hermes_shim_http.prompting import build_cli_prompt, build_cli_system_prompt, build_cli_user_prompt
from hermes_shim_http.runner import build_cli_command, run_cli_prompt, stream_cli_prompt
from hermes_shim_http.session_cache import SessionCache


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = REPO_ROOT / "tests" / "fake_cli.py"


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


_CLAUDE_APPEND_PROMPT = "Be concise and follow the user's instructions exactly."


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
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]

    def test_build_cli_command_uses_profile_defaults_for_supported_clis(self):
        assert build_cli_command(ShimConfig(command="claude", args=[]), "hello") == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--append-system-prompt",
            _CLAUDE_APPEND_PROMPT,
        ]
        assert build_cli_command(ShimConfig(command="codex", args=[]), "hello") == ["codex", "exec", "hello"]
        assert build_cli_command(ShimConfig(command="opencode", args=[]), "hello") == ["opencode", "run", "hello"]

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
        mock_drain.assert_called_once_with(
            [
                "claude",
                "-p",
                "--append-system-prompt",
                _CLAUDE_APPEND_PROMPT,
                "--model",
                "sonnet",
            ],
            config=cfg,
            stdin_prompt="Be terse.\n\nhello",
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
                "--model",
                "sonnet",
                "--resume",
                "22222222-2222-2222-2222-222222222222",
                "--fork-session",
                "--session-id",
                "11111111-1111-1111-1111-111111111111",
            ],
            config=cfg,
            stdin_prompt="<user>\ncontinue\n</user>",
        )

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

    def test_run_cli_prompt_translates_argument_list_too_long_error(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.Popen",
            side_effect=OSError(7, "Argument list too long", "codex"),
        ):
            with pytest.raises(RuntimeError, match="Prompt too large to pass on the command line"):
                run_cli_prompt("hello", cfg)

    def test_stream_cli_prompt_uses_stdin_for_claude_profile(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        class _FakeStream:
            def read(self, _size: int = 1) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeStdin:
            def __init__(self) -> None:
                self.value = ""

            def write(self, text: str) -> None:
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
        assert fake_process.stdin.value == "Be terse.\n\nhello"
        mock_popen.assert_called_once_with(
            [
                "claude",
                "-p",
                "--append-system-prompt",
                _CLAUDE_APPEND_PROMPT,
            ],
            cwd="/tmp/work",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

    def test_stream_cli_prompt_omits_large_system_prompt_from_resumed_claude_stdin(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        class _FakeStream:
            def read(self, _size: int = 1) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeStdin:
            def __init__(self) -> None:
                self.value = ""

            def write(self, text: str) -> None:
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
        assert fake_process.stdin.value == "<user>\ncontinue\n</user>"
        mock_popen.assert_called_once_with(
            [
                "claude",
                "-p",
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
            text=True,
            bufsize=0,
        )

    def test_stream_cli_prompt_yields_text_chunks_live(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "stream-text"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
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
        )

        with pytest.raises(TimeoutError, match="idle for"):
            list(stream_cli_prompt("ignored", cfg))

    def test_stream_cli_prompt_translates_argument_list_too_long_error(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.Popen",
            side_effect=OSError(7, "Argument list too long", "codex"),
        ):
            with pytest.raises(RuntimeError, match="Prompt too large to pass on the command line"):
                list(stream_cli_prompt("hello", cfg))

    def test_stream_cli_prompt_yields_tool_call_event_without_wrapper_text(self):
        cfg = ShimConfig(
            command="python3",
            args=[str(FAKE_CLI), "--mode", "stream-tool"],
            cwd=str(REPO_ROOT),
            timeout=12.0,
        )

        events = list(stream_cli_prompt("ignored", cfg))

        tool_call_events = [event for event in events if event.kind == "tool_call"]
        text_payload = "".join(event.text or "" for event in events if event.kind == "text")

        assert len(tool_call_events) == 1
        assert text_payload == ""
        assert tool_call_events[0].tool_call["function"]["name"] == "read_file"


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
