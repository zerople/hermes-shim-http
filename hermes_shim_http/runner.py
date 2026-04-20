from __future__ import annotations

import codecs
import errno
import json
import os
import queue
import re
import subprocess
import threading
import time
from typing import Iterator, List

from .models import CliRunResult, CliStreamEvent, ParsedShimOutput, ShimConfig
from .parsing import (
    ClaudeStreamJsonParser,
    IncrementalToolCallParser,
    parse_claude_stream_json,
    parse_claude_stream_metadata,
    parse_cli_output,
)
from .hermes_mcp import request_scoped_mcp_config
from .single_child import ChildLockBusy, acquire_single_child_lock
from .telemetry import emit_event

_CLAUDE_BLOCKED_ARG_MODES = {
    "-p": "standalone",
    "--print": "standalone",
    "--output-format": "value",
    "--input-format": "value",
    "--permission-mode": "value",
}


_DEFAULT_CHUNK_SIZE = 4096
_CLAUDE_APPEND_SYSTEM_PROMPT = "Be concise. You must follow the VERY FIRST live user message after the session opens as the highest-priority instruction for the session. Do not ignore or drift away from that first live user message, even if later transcript context is long, distracting, summarized, or compacted."
_HEARTBEAT_CHAR = "\u200b"
_HEARTBEAT_WRAPPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "heartbeat-wrap.py",
)
_RAW_LOG_ENV = "HERMES_SHIM_CLAUDE_RAW_LOG_DIR"


def _open_raw_log(label: str):
    log_dir = os.environ.get(_RAW_LOG_ENV, "").strip()
    if not log_dir:
        return None
    try:
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"claude-raw-{ts}-{os.getpid()}-{label}.log"
        path = os.path.join(log_dir, fname)
        handle = open(path, "a", encoding="utf-8")
        handle.write(f"# opened {time.strftime('%Y-%m-%dT%H:%M:%S%z')} label={label} pid={os.getpid()}\n")
        handle.flush()
        return handle
    except OSError:
        return None


def _raw_log_write(handle, *, started: float, stream: str, kind: str, payload: str) -> None:
    if handle is None:
        return
    try:
        elapsed = time.time() - started
        handle.write(f"[+{elapsed:8.3f}s] [{stream}] [{kind}] {payload!r}\n")
        handle.flush()
    except OSError:
        pass


def _raw_log_header(handle, *, spawn_command, session_id: str | None, resume_session_id: str | None) -> None:
    if handle is None:
        return
    try:
        handle.write(f"# spawn_command={list(spawn_command)!r}\n")
        handle.write(f"# session_id={session_id!r} resume_session_id={resume_session_id!r}\n")
        handle.flush()
    except OSError:
        pass


def _raw_log_close(handle, *, started: float, exit_code) -> None:
    if handle is None:
        return
    try:
        elapsed = time.time() - started
        handle.write(f"# closed elapsed={elapsed:.3f}s exit_code={exit_code}\n")
        handle.close()
    except OSError:
        pass


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


def _uses_native_claude_cli(config: ShimConfig) -> bool:
    return _resolved_profile(config) == "claude" and _command_basename(config.command) == "claude"


def _resolved_args(config: ShimConfig) -> list[str]:
    profile = _resolved_profile(config)
    if profile == "claude":
        if not _uses_native_claude_cli(config):
            return list(config.args)
        fixed = [
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if not config.args:
            return fixed
        return [*fixed, *_filter_claude_custom_args(config.args)]

    if config.args:
        return list(config.args)

    return {
        "codex": ["exec"],
        "opencode": ["run"],
        "generic": [],
    }[profile]


def _pipes_prompt_via_stdin(config: ShimConfig) -> bool:
    return _resolved_profile(config) == "claude"


def parse_cli_result(result: CliRunResult, config: ShimConfig) -> ParsedShimOutput:
    """Parse CLI stdout using the profile-appropriate parser."""
    if _resolved_profile(config) == "claude":
        return parse_claude_stream_json(result.stdout)
    return parse_cli_output(result.stdout)


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


def _filter_claude_custom_args(args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        flag = arg
        has_inline_value = False
        if "=" in arg:
            flag = arg.split("=", 1)[0]
            has_inline_value = True
        mode = _CLAUDE_BLOCKED_ARG_MODES.get(flag)
        if mode:
            if mode == "value" and not has_inline_value:
                skip_next = True
            continue
        filtered.append(arg)
    return filtered


def _build_claude_stdin_payload(prompt_text: str) -> str:
    payload = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _resolve_claude_result_session_id(*, requested_resume_session_id: str | None, emitted_session_id: str | None, failed: bool) -> str | None:
    if failed and requested_resume_session_id and emitted_session_id and emitted_session_id != requested_resume_session_id:
        return None
    return emitted_session_id


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
    if _resolved_profile(config) == "claude":
        payload_text = prompt_text if resume_session_id else _combine_prompt_text(prompt_text, system_prompt=system_prompt)
        return _build_claude_stdin_payload(payload_text)
    return _combine_prompt_text(prompt_text, system_prompt=system_prompt)
    return _combine_prompt_text(prompt_text, system_prompt=system_prompt)


def build_cli_command(
    config: ShimConfig,
    prompt_text: str,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    disable_builtin_tools: bool = True,
    mcp_config_path: str | None = None,
) -> List[str]:
    base = [config.command, *_resolved_args(config)]
    combined = _combine_prompt_text(prompt_text, system_prompt=system_prompt)

    if _resolved_profile(config) == "claude":
        command = list(base)
        if not _uses_native_claude_cli(config):
            return command
        if not resume_session_id:
            command.extend(["--append-system-prompt", _CLAUDE_APPEND_SYSTEM_PROMPT])
        if mcp_config_path:
            command.extend(["--mcp-config", mcp_config_path])
        if config.strict_mcp_config:
            command.append("--strict-mcp-config")
        if disable_builtin_tools:
            command.extend(["--tools", ""])
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


def _sanitize_lock_key(lock_key: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", (lock_key or "").strip())
    sanitized = sanitized.strip("._")
    return sanitized or "request"


def _child_lock_path_for_request(
    config: ShimConfig,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
) -> str | None:
    base_path = (config.single_child_lock_path or "").strip()
    if not base_path:
        return None
    if not resume_session_id:
        return None
    return f"{base_path}.{_sanitize_lock_key(resume_session_id)}"


def run_cli_prompt(
    prompt_text: str,
    config: ShimConfig,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    disable_builtin_tools: bool = True,
    advertised_tools: list[dict] | None = None,
) -> CliRunResult:
    with request_scoped_mcp_config(tools=advertised_tools if _resolved_profile(config) == "claude" else None) as mcp_config_path:
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
            disable_builtin_tools=disable_builtin_tools,
            mcp_config_path=mcp_config_path,
        )
        stdin_bytes = len(stdin_prompt.encode("utf-8")) if _pipes_prompt_via_stdin(config) else 0
        _log_cli_dispatch(command, stdin_bytes=stdin_bytes, session_id=session_id, resume_session_id=resume_session_id)
        started = time.time()
        stdout_text, stderr_text, exit_code = _drain_cli_process(
            command,
            config=config,
            stdin_prompt=stdin_prompt if _pipes_prompt_via_stdin(config) else None,
            lock_path=_child_lock_path_for_request(
                config,
                session_id=session_id,
                resume_session_id=resume_session_id,
            ),
        )
    duration_ms = int((time.time() - started) * 1000)
    result = CliRunResult(
        stdout=stdout_text,
        stderr=stderr_text,
        exit_code=exit_code,
        duration_ms=duration_ms,
        session_id=(
            _resolve_claude_result_session_id(
                requested_resume_session_id=resume_session_id,
                emitted_session_id=parse_claude_stream_metadata(stdout_text).session_id,
                failed=exit_code != 0 or parse_claude_stream_metadata(stdout_text).is_error,
            )
            if _resolved_profile(config) == "claude"
            else None
        ),
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
    lock_path: str | None = None,
) -> tuple[str, str, int]:
    with acquire_single_child_lock(lock_path or ""):
        return _drain_cli_process_inner(command, config=config, stdin_prompt=stdin_prompt)


def _drain_cli_process_inner(
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
    raw_log = _open_raw_log("drain")
    _raw_log_header(raw_log, spawn_command=spawn_command, session_id=None, resume_session_id=None)
    try:
        process = subprocess.Popen(spawn_command, **popen_kwargs)
    except OSError as exc:
        _raw_log_close(raw_log, started=time.time(), exit_code=None)
        raise _translate_spawn_oserror(exc, command=config.command) from exc

    if stdin_prompt is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin_prompt.encode("utf-8"))
            process.stdin.flush()
        finally:
            process.stdin.close()
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        _raw_log_close(raw_log, started=time.time(), exit_code=None)
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
    exit_code: int | None = None

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
                    _raw_log_write(raw_log, started=started_at, stream="stdout", kind="raw", payload=chunk)
                    cleaned = _strip_heartbeat(chunk)
                    if cleaned:
                        _raw_log_write(raw_log, started=started_at, stream="stdout", kind="real", payload=cleaned)
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
                _raw_log_write(raw_log, started=started_at, stream="stderr", kind="raw", payload=chunk)
                cleaned = _strip_heartbeat(chunk)
                if cleaned:
                    _raw_log_write(raw_log, started=started_at, stream="stderr", kind="real", payload=cleaned)
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
        _raw_log_close(raw_log, started=started_at, exit_code=exit_code)

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
    disable_builtin_tools: bool = True,
    advertised_tools: list[dict] | None = None,
) -> Iterator[CliStreamEvent]:
    lock_path = _child_lock_path_for_request(
        config,
        session_id=session_id,
        resume_session_id=resume_session_id,
    )
    with acquire_single_child_lock(lock_path or ""):
        yield from _stream_cli_prompt_inner(
            prompt_text,
            config,
            session_id=session_id,
            resume_session_id=resume_session_id,
            system_prompt=system_prompt,
            model=model,
            disable_builtin_tools=disable_builtin_tools,
            advertised_tools=advertised_tools,
        )


def _stream_cli_prompt_inner(
    prompt_text: str,
    config: ShimConfig,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    disable_builtin_tools: bool = True,
    advertised_tools: list[dict] | None = None,
) -> Iterator[CliStreamEvent]:
    with request_scoped_mcp_config(tools=advertised_tools if _resolved_profile(config) == "claude" else None) as mcp_config_path:
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
            disable_builtin_tools=disable_builtin_tools,
            mcp_config_path=mcp_config_path,
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
        raw_log = _open_raw_log("stream")
        _raw_log_header(raw_log, spawn_command=spawn_command, session_id=session_id, resume_session_id=resume_session_id)
        try:
            process = subprocess.Popen(
                spawn_command,
                **popen_kwargs,
            )
        except OSError as exc:
            _raw_log_close(raw_log, started=started, exit_code=None)
            raise _translate_spawn_oserror(exc, command=config.command) from exc

        if _pipes_prompt_via_stdin(config) and process.stdin is not None:
            process.stdin.write(stdin_prompt.encode("utf-8"))
            process.stdin.flush()
            process.stdin.close()
        if process.stdout is None or process.stderr is None:
            _terminate_process(process)
            _raw_log_close(raw_log, started=started, exit_code=None)
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
        parser = ClaudeStreamJsonParser(synthesize_progress=False) if _resolved_profile(config) == "claude" else IncrementalToolCallParser()
        exit_code: int | None = None

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
                        _raw_log_write(raw_log, started=started, stream="stdout", kind="raw", payload=chunk)
                        cleaned = _strip_heartbeat(chunk)
                        if cleaned:
                            _raw_log_write(raw_log, started=started, stream="stdout", kind="real", payload=cleaned)
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
                    _raw_log_write(raw_log, started=started, stream="stderr", kind="raw", payload=chunk)
                    cleaned = _strip_heartbeat(chunk)
                    if cleaned:
                        _raw_log_write(raw_log, started=started, stream="stderr", kind="real", payload=cleaned)
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
            _raw_log_close(raw_log, started=started, exit_code=exit_code)
