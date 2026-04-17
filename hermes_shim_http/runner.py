from __future__ import annotations

import errno
import os
import queue
import subprocess
import threading
import time
from typing import Iterator, List

from .models import CliRunResult, CliStreamEvent, ShimConfig
from .parsing import IncrementalToolCallParser


def _command_basename(command: str) -> str:
    return os.path.basename((command or "").strip()).lower()


def _resolved_args(config: ShimConfig) -> list[str]:
    if config.args:
        return list(config.args)

    profile = config.cli_profile
    if profile == "auto":
        profile = {
            "claude": "claude",
            "codex": "codex",
            "opencode": "opencode",
        }.get(_command_basename(config.command), "generic")

    return {
        "claude": ["-p"],
        "codex": ["exec"],
        "opencode": ["run"],
        "generic": [],
    }[profile]


def _pipes_prompt_via_stdin(config: ShimConfig) -> bool:
    profile = config.cli_profile
    if profile == "auto":
        profile = {
            "claude": "claude",
            "codex": "codex",
            "opencode": "opencode",
        }.get(_command_basename(config.command), "generic")
    return profile == "claude"


def resolved_cli_args(config: ShimConfig) -> list[str]:
    return _resolved_args(config)


def build_cli_command(config: ShimConfig, prompt_text: str) -> List[str]:
    command = [config.command, *_resolved_args(config)]
    if _pipes_prompt_via_stdin(config):
        return command
    return [*command, prompt_text]


def _translate_spawn_oserror(exc: OSError, *, command: str) -> RuntimeError:
    if exc.errno == errno.E2BIG:
        return RuntimeError(
            f"Prompt too large to pass on the command line for '{command}'. "
            "Use a CLI/profile that accepts stdin input, reduce prompt size, or update the shim to stream via stdin/file input."
        )
    return RuntimeError(f"Failed to start CLI process '{command}': {exc}")


def run_cli_prompt(prompt_text: str, config: ShimConfig) -> CliRunResult:
    command = build_cli_command(config, prompt_text)
    started = time.time()
    run_kwargs = {
        "cwd": config.cwd,
        "capture_output": True,
        "text": True,
        "timeout": config.timeout,
        "check": False,
    }
    if _pipes_prompt_via_stdin(config):
        run_kwargs["input"] = prompt_text
    try:
        completed = subprocess.run(
            command,
            **run_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Timed out waiting for CLI process after {config.timeout:.1f}s") from exc
    except OSError as exc:
        raise _translate_spawn_oserror(exc, command=config.command) from exc

    duration_ms = int((time.time() - started) * 1000)
    result = CliRunResult(
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        exit_code=int(completed.returncode),
        duration_ms=duration_ms,
    )
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"CLI process failed: {detail}")
    return result


def _pump_stream(stream, sink: queue.Queue[tuple[str, str] | None], stream_name: str, *, chunk_size: int = 1) -> None:
    try:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            sink.put((stream_name, chunk))
    finally:
        try:
            stream.close()
        except Exception:
            pass
        sink.put(None)


def stream_cli_prompt(prompt_text: str, config: ShimConfig) -> Iterator[CliStreamEvent]:
    command = build_cli_command(config, prompt_text)
    started = time.time()
    popen_kwargs = {
        "cwd": config.cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 0,
    }
    if _pipes_prompt_via_stdin(config):
        popen_kwargs["stdin"] = subprocess.PIPE
    try:
        process = subprocess.Popen(
            command,
            **popen_kwargs,
        )
    except OSError as exc:
        raise _translate_spawn_oserror(exc, command=config.command) from exc

    if _pipes_prompt_via_stdin(config) and process.stdin is not None:
        process.stdin.write(prompt_text)
        process.stdin.flush()
        process.stdin.close()
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("CLI process did not expose stdout/stderr pipes")

    stdout_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stderr_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stdout_done = False
    stderr_done = False
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    parser = IncrementalToolCallParser()

    stdout_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stdout, stdout_queue, "stdout"),
        kwargs={"chunk_size": 1},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stderr, stderr_queue, "stderr"),
        kwargs={"chunk_size": 1},
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
                    stdout_chunks.append(chunk)
                    for event in parser.feed(chunk):
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
                stderr_chunks.append(chunk)

            if not stderr_done and not stderr_thread.is_alive() and stderr_queue.empty():
                stderr_done = True

            if not stdout_done and not stdout_thread.is_alive() and stdout_queue.empty():
                stdout_done = True

            if time.time() - started > config.timeout:
                process.kill()
                raise TimeoutError(f"Timed out waiting for CLI process after {config.timeout:.1f}s")

        stdout_thread.join(timeout=0.2)
        stderr_thread.join(timeout=0.2)
        for event in parser.finalize():
            yield event

        exit_code = process.wait(timeout=max(1.0, min(config.timeout, 5.0)))
        if exit_code != 0:
            detail = "".join(stderr_chunks).strip() or "".join(stdout_chunks).strip() or f"exit code {exit_code}"
            raise RuntimeError(f"CLI process failed: {detail}")
    except Exception:
        if process.poll() is None:
            process.kill()
        raise
