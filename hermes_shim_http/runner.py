from __future__ import annotations

import codecs
import errno
import os
import queue
import subprocess
import threading
import time
from typing import Iterator, List

from .models import CliRunResult, CliStreamEvent, ShimConfig
from .parsing import IncrementalToolCallParser
from .telemetry import emit_event


_DEFAULT_CHUNK_SIZE = 4096
_CLAUDE_APPEND_SYSTEM_PROMPT = "Be concise and follow the user's instructions exactly."
_HEARTBEAT_CHAR = "\u200b"
_HEARTBEAT_WRAPPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "heartbeat-wrap.py",
)


def _command_basename(command: str) -> str:
    return os.path.basename((command or "").strip()).lower()


def _resolved_profile(config: ShimConfig) -> str:
    profile = config.cli_profile
    if profile == "auto":
        profile = {
            "claude": "claude",
            "codex": "codex",
            "opencode": "opencode",
        }.get(_command_basename(config.command), "generic")
    return profile


def _resolved_args(config: ShimConfig) -> list[str]:
    if config.args:
        return list(config.args)

    return {
        "claude": ["-p", "--dangerously-skip-permissions"],
        "codex": ["exec"],
        "opencode": ["run"],
        "generic": [],
    }[_resolved_profile(config)]


def _pipes_prompt_via_stdin(config: ShimConfig) -> bool:
    return _resolved_profile(config) == "claude"


def supports_cli_resume(config: ShimConfig) -> bool:
    return _resolved_profile(config) == "claude"


def resolved_cli_args(config: ShimConfig) -> list[str]:
    return _resolved_args(config)


def _is_meaningful_model(model: str | None) -> bool:
    if not model:
        return False
    value = model.strip().lower()
    if not value:
        return False
    return value not in {"cli-http-shim", "default", "auto"}


def _log_cli_dispatch(
    command: list[str],
    *,
    stdin_bytes: int,
    session_id: str | None,
    resume_session_id: str | None,
) -> None:
    flags = command[1:]
    emit_event(
        "cli_dispatch",
        argv=flags,
        append_system_prompt_used="--append-system-prompt" in flags,
        model_flag_used="--model" in flags,
        model_value=flags[flags.index("--model") + 1] if "--model" in flags else None,
        resume_used=resume_session_id is not None,
        session_id=session_id,
        resume_session_id=resume_session_id,
        stdin_bytes=stdin_bytes,
    )


def _combine_prompt_text(prompt_text: str, *, system_prompt: str | None = None) -> str:
    if system_prompt:
        return f"{system_prompt}\n\n{prompt_text}" if prompt_text else system_prompt
    return prompt_text


def _stdin_prompt_text(
    config: ShimConfig,
    prompt_text: str,
    *,
    system_prompt: str | None = None,
    resume_session_id: str | None = None,
) -> str:
    if _resolved_profile(config) == "claude" and resume_session_id:
        return prompt_text
    return _combine_prompt_text(prompt_text, system_prompt=system_prompt)


def build_cli_command(
    config: ShimConfig,
    prompt_text: str,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> List[str]:
    base = [config.command, *_resolved_args(config)]
    combined = _combine_prompt_text(prompt_text, system_prompt=system_prompt)

    if _resolved_profile(config) == "claude":
        command = list(base)
        if not resume_session_id:
            command.extend(["--append-system-prompt", _CLAUDE_APPEND_SYSTEM_PROMPT])
        if _is_meaningful_model(model):
            command.extend(["--model", str(model).strip()])
        if config.fallback_model:
            command.extend(["--fallback-model", config.fallback_model])
        if session_id:
            if resume_session_id:
                command.extend(["--resume", resume_session_id, "--fork-session"])
            command.extend(["--session-id", session_id])
        return command

    return [*base, combined]


def _heartbeat_prefix(config: ShimConfig) -> list[str]:
    if not config.heartbeat_wrap:
        return []
    return ["python3", _HEARTBEAT_WRAPPER, "-i", str(config.heartbeat_interval), "--"]


def _strip_heartbeat(text: str) -> str:
    return text.replace(_HEARTBEAT_CHAR, "") if _HEARTBEAT_CHAR in text else text


def _translate_spawn_oserror(exc: OSError, *, command: str) -> RuntimeError:
    if exc.errno == errno.E2BIG:
        return RuntimeError(
            f"Prompt too large to pass on the command line for '{command}'. "
            "Use a CLI/profile that accepts stdin input, reduce prompt size, or update the shim to stream via stdin/file input."
        )
    return RuntimeError(f"Failed to start CLI process '{command}': {exc}")


def run_cli_prompt(
    prompt_text: str,
    config: ShimConfig,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> CliRunResult:
    stdin_prompt = _stdin_prompt_text(
        config,
        prompt_text,
        system_prompt=system_prompt,
        resume_session_id=resume_session_id,
    )
    command = build_cli_command(
        config,
        prompt_text,
        session_id=session_id,
        resume_session_id=resume_session_id,
        system_prompt=system_prompt,
        model=model,
    )
    stdin_bytes = len(stdin_prompt.encode("utf-8")) if _pipes_prompt_via_stdin(config) else 0
    _log_cli_dispatch(command, stdin_bytes=stdin_bytes, session_id=session_id, resume_session_id=resume_session_id)
    started = time.time()
    stdout_text, stderr_text, exit_code = _drain_cli_process(
        command,
        config=config,
        stdin_prompt=stdin_prompt if _pipes_prompt_via_stdin(config) else None,
    )
    duration_ms = int((time.time() - started) * 1000)
    result = CliRunResult(
        stdout=stdout_text,
        stderr=stderr_text,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"CLI process failed: {detail}")
    return result


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=2.0)
    except Exception:
        pass


def _drain_cli_process(
    command: List[str],
    *,
    config: ShimConfig,
    stdin_prompt: str | None,
) -> tuple[str, str, int]:
    popen_kwargs = {
        "cwd": config.cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if stdin_prompt is not None:
        popen_kwargs["stdin"] = subprocess.PIPE
    spawn_command = _heartbeat_prefix(config) + command
    try:
        process = subprocess.Popen(spawn_command, **popen_kwargs)
    except OSError as exc:
        raise _translate_spawn_oserror(exc, command=config.command) from exc

    if stdin_prompt is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin_prompt.encode("utf-8"))
            process.stdin.flush()
        finally:
            process.stdin.close()
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise RuntimeError("CLI process did not expose stdout/stderr pipes")

    stdout_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stderr_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stdout_thread = threading.Thread(target=_pump_stream, args=(process.stdout, stdout_queue, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=_pump_stream, args=(process.stderr, stderr_queue, "stderr"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_done = False
    stderr_done = False
    started_at = time.time()
    last_activity = started_at
    total_bytes = 0
    max_bytes = config.max_output_bytes
    hard_deadline = config.hard_deadline_seconds

    try:
        while not stdout_done or not stderr_done:
            if not stdout_done:
                try:
                    item = stdout_queue.get(timeout=0.05)
                except queue.Empty:
                    item = None
                if item is None:
                    if not stdout_thread.is_alive() and stdout_queue.empty():
                        stdout_done = True
                else:
                    _, chunk = item
                    cleaned = _strip_heartbeat(chunk)
                    if cleaned:
                        last_activity = time.time()
                        if max_bytes and total_bytes < max_bytes:
                            remaining = max_bytes - total_bytes
                            chunk_bytes = len(cleaned.encode("utf-8"))
                            if chunk_bytes > remaining:
                                stdout_chunks.append(cleaned.encode("utf-8")[:remaining].decode("utf-8", errors="ignore"))
                                total_bytes = max_bytes
                            else:
                                stdout_chunks.append(cleaned)
                                total_bytes += chunk_bytes
                        elif not max_bytes:
                            stdout_chunks.append(cleaned)

            while True:
                try:
                    item = stderr_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    if not stderr_thread.is_alive() and stderr_queue.empty():
                        stderr_done = True
                    continue
                _, chunk = item
                cleaned = _strip_heartbeat(chunk)
                if cleaned:
                    last_activity = time.time()
                    stderr_chunks.append(cleaned)

            if not stderr_done and not stderr_thread.is_alive() and stderr_queue.empty():
                stderr_done = True
            if not stdout_done and not stdout_thread.is_alive() and stdout_queue.empty():
                stdout_done = True

            now = time.time()
            if now - last_activity > config.timeout:
                raise TimeoutError(
                    f"CLI process idle for {config.timeout:.1f}s with no real stdout/stderr output (heartbeat-only does not reset the timer)"
                )
            if hard_deadline and (now - started_at) > hard_deadline:
                raise TimeoutError(
                    f"CLI process exceeded hard deadline of {hard_deadline:.1f}s"
                )
            if max_bytes and total_bytes >= max_bytes:
                raise RuntimeError(
                    f"CLI process exceeded max output cap of {max_bytes} bytes"
                )

        stdout_thread.join(timeout=0.5)
        stderr_thread.join(timeout=0.5)
        try:
            exit_code = process.wait(timeout=max(1.0, min(config.timeout, 5.0)))
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            raise RuntimeError("CLI process did not exit after pipes closed")
    finally:
        _terminate_process(process)
        stdout_thread.join(timeout=0.5)
        stderr_thread.join(timeout=0.5)

    return "".join(stdout_chunks), "".join(stderr_chunks), int(exit_code)


def _pump_stream(
    stream,
    sink: queue.Queue[tuple[str, str] | None],
    stream_name: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> None:
    read = stream.read1 if hasattr(stream, "read1") else stream.read
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = read(chunk_size)
            if not chunk:
                tail = decoder.decode(b"", final=True)
                if tail:
                    sink.put((stream_name, tail))
                break
            if isinstance(chunk, bytes):
                text = decoder.decode(chunk)
                if not text:
                    continue
            else:
                text = chunk
            sink.put((stream_name, text))
    finally:
        try:
            stream.close()
        except Exception:
            pass
        sink.put(None)


def stream_cli_prompt(
    prompt_text: str,
    config: ShimConfig,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> Iterator[CliStreamEvent]:
    stdin_prompt = _stdin_prompt_text(
        config,
        prompt_text,
        system_prompt=system_prompt,
        resume_session_id=resume_session_id,
    )
    command = build_cli_command(
        config,
        prompt_text,
        session_id=session_id,
        resume_session_id=resume_session_id,
        system_prompt=system_prompt,
        model=model,
    )
    stdin_bytes = len(stdin_prompt.encode("utf-8")) if _pipes_prompt_via_stdin(config) else 0
    _log_cli_dispatch(command, stdin_bytes=stdin_bytes, session_id=session_id, resume_session_id=resume_session_id)
    started = time.time()
    last_activity = started
    popen_kwargs = {
        "cwd": config.cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if _pipes_prompt_via_stdin(config):
        popen_kwargs["stdin"] = subprocess.PIPE
    spawn_command = _heartbeat_prefix(config) + command
    try:
        process = subprocess.Popen(
            spawn_command,
            **popen_kwargs,
        )
    except OSError as exc:
        raise _translate_spawn_oserror(exc, command=config.command) from exc

    if _pipes_prompt_via_stdin(config) and process.stdin is not None:
        process.stdin.write(stdin_prompt.encode("utf-8"))
        process.stdin.flush()
        process.stdin.close()
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise RuntimeError("CLI process did not expose stdout/stderr pipes")

    stdout_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stderr_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stdout_done = False
    stderr_done = False
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    total_bytes = 0
    max_bytes = config.max_output_bytes
    hard_deadline = config.hard_deadline_seconds
    parser = IncrementalToolCallParser()

    stdout_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stdout, stdout_queue, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stderr, stderr_queue, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        while not stdout_done or not stderr_done:
            if not stdout_done:
                try:
                    item = stdout_queue.get(timeout=0.05)
                except queue.Empty:
                    item = None
                if item is None:
                    if not stdout_thread.is_alive() and stdout_queue.empty():
                        stdout_done = True
                else:
                    _, chunk = item
                    cleaned = _strip_heartbeat(chunk)
                    if cleaned:
                        last_activity = time.time()
                        if max_bytes:
                            chunk_bytes = len(cleaned.encode("utf-8"))
                            total_bytes += chunk_bytes
                        stdout_chunks.append(cleaned)
                        for event in parser.feed(cleaned):
                            yield event

            while True:
                try:
                    item = stderr_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    if not stderr_thread.is_alive() and stderr_queue.empty():
                        stderr_done = True
                    continue
                _, chunk = item
                cleaned = _strip_heartbeat(chunk)
                if cleaned:
                    last_activity = time.time()
                    stderr_chunks.append(cleaned)

            if not stderr_done and not stderr_thread.is_alive() and stderr_queue.empty():
                stderr_done = True

            if not stdout_done and not stdout_thread.is_alive() and stdout_queue.empty():
                stdout_done = True

            now = time.time()
            if now - last_activity > config.timeout:
                raise TimeoutError(
                    f"CLI process idle for {config.timeout:.1f}s with no real stdout/stderr output (heartbeat-only does not reset the timer)"
                )
            if hard_deadline and (now - started) > hard_deadline:
                raise TimeoutError(
                    f"CLI process exceeded hard deadline of {hard_deadline:.1f}s"
                )
            if max_bytes and total_bytes >= max_bytes:
                raise RuntimeError(
                    f"CLI process exceeded max output cap of {max_bytes} bytes"
                )

        stdout_thread.join(timeout=0.5)
        stderr_thread.join(timeout=0.5)
        for event in parser.finalize():
            yield event

        try:
            exit_code = process.wait(timeout=max(1.0, min(config.timeout, 5.0)))
        except subprocess.TimeoutExpired:
            raise RuntimeError("CLI process did not exit after pipes closed")
        if exit_code != 0:
            detail = "".join(stderr_chunks).strip() or "".join(stdout_chunks).strip() or f"exit code {exit_code}"
            raise RuntimeError(f"CLI process failed: {detail}")
    finally:
        _terminate_process(process)
        stdout_thread.join(timeout=0.5)
        stderr_thread.join(timeout=0.5)
