import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_shim_http.models import CliRunResult, ShimConfig
from hermes_shim_http.prompting import build_cli_prompt
from hermes_shim_http.runner import build_cli_command, run_cli_prompt, stream_cli_prompt
from hermes_shim_http.session_cache import SessionCache


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = REPO_ROOT / "tests" / "fake_cli.py"


class TestPrompting:
    def test_build_cli_prompt_includes_transcript_tools_and_native_tool_ban(self):
        prompt = build_cli_prompt(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Check the repo."},
                {"role": "tool", "content": "{\"ok\": true}"},
            ],
            model="claude-cli",
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

        assert "Hermes requested model hint: claude-cli" in prompt
        assert "Do NOT use any native CLI built-in tools" in prompt
        assert "<tool_call>{...}</tool_call>" in prompt
        assert "read_file" in prompt
        assert "System:\nYou are helpful." in prompt
        assert "User:\nCheck the repo." in prompt
        assert "Tool:\n{\"ok\": true}" in prompt


class TestRunner:
    def test_build_cli_command_keeps_claude_prompt_off_argv(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work")

        cmd = build_cli_command(cfg, "hello")

        assert cmd == ["claude", "-p"]

    def test_build_cli_command_uses_profile_defaults_for_supported_clis(self):
        assert build_cli_command(ShimConfig(command="claude", args=[]), "hello") == ["claude", "-p"]
        assert build_cli_command(ShimConfig(command="codex", args=[]), "hello") == ["codex", "exec", "hello"]
        assert build_cli_command(ShimConfig(command="opencode", args=[]), "hello") == ["opencode", "run", "hello"]

    def test_build_cli_command_includes_claude_session_resume_flags(self):
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
            "--resume",
            "22222222-2222-2222-2222-222222222222",
            "--fork-session",
            "--session-id",
            "11111111-1111-1111-1111-111111111111",
        ]

    def test_run_cli_prompt_returns_result(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)
        completed = subprocess.CompletedProcess(
            args=["claude", "-p"],
            returncode=0,
            stdout="done",
            stderr="",
        )

        with patch("hermes_shim_http.runner.subprocess.run", return_value=completed) as mock_run:
            result = run_cli_prompt("hello", cfg)

        assert isinstance(result, CliRunResult)
        assert result.stdout == "done"
        assert result.stderr == ""
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            ["claude", "-p"],
            cwd="/tmp/work",
            capture_output=True,
            text=True,
            input="hello",
            timeout=12.0,
            check=False,
        )

    def test_run_cli_prompt_raises_on_non_zero_exit(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)
        completed = subprocess.CompletedProcess(
            args=["claude", "-p"],
            returncode=1,
            stdout="",
            stderr="boom",
        )

        with patch("hermes_shim_http.runner.subprocess.run", return_value=completed):
            with pytest.raises(RuntimeError, match="boom"):
                run_cli_prompt("hello", cfg)

    def test_run_cli_prompt_raises_timeout_error(self):
        cfg = ShimConfig(command="claude", args=["-p"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=12.0),
        ):
            with pytest.raises(TimeoutError, match="Timed out"):
                run_cli_prompt("hello", cfg)

    def test_run_cli_prompt_translates_argument_list_too_long_error(self):
        cfg = ShimConfig(command="codex", args=["exec"], cwd="/tmp/work", timeout=12.0)

        with patch(
            "hermes_shim_http.runner.subprocess.run",
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

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStream()
                self.stderr = _FakeStream()
                self.stdin = None

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                return None

        with patch("hermes_shim_http.runner.subprocess.Popen", return_value=_FakeProcess()) as mock_popen:
            events = list(stream_cli_prompt("hello", cfg))

        assert events == []
        mock_popen.assert_called_once_with(
            ["claude", "-p"],
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
            model="claude-cli",
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
            model="claude-cli",
            tools=None,
            tool_choice=None,
        )

        assert second.resume_session_id == first.session_id
        assert second.prompt_text == "User:\ncontinue"
        assert second.prefix_message_count == 2
