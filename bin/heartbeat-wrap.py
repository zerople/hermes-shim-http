#!/usr/bin/env python3
"""heartbeat-wrap — keep the shim's idle timer alive during long child CLI runs.

The shim kills its child CLI after `timeout` seconds of stdout/stderr silence.
Legitimate long operations (extended reasoning, network-bound API calls, large
test runs) can exceed that window without producing output, which causes false
timeout kills and session churn.

This wrapper spawns the real CLI as a child, streams its stdin/stdout/stderr
through unchanged, and emits a single zero-width space (U+200B) byte to stderr
every `--interval` seconds while the child is still alive. The shim treats any
byte as activity and resets its idle timer, so the heartbeat prevents false
timeouts. If the child exits or crashes, the heartbeat stops immediately, so
the shim still detects truly hung processes via its normal idle timeout.

Usage: heartbeat-wrap [-i SECONDS] -- <cmd> [args...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from typing import IO

HEARTBEAT_BYTES = "\u200b".encode("utf-8")


def _pump(src: IO[bytes], dst: IO[bytes]) -> None:
    try:
        while True:
            chunk = src.read(4096)
            if not chunk:
                return
            dst.write(chunk)
            dst.flush()
    except (OSError, ValueError):
        return


def _heartbeat_loop(
    proc: subprocess.Popen,
    stop: threading.Event,
    interval: float,
    stderr_sink: IO[bytes],
    lock: threading.Lock,
) -> None:
    while not stop.is_set():
        if stop.wait(interval):
            return
        if proc.poll() is not None:
            return
        try:
            with lock:
                stderr_sink.write(HEARTBEAT_BYTES)
                stderr_sink.flush()
        except (OSError, ValueError):
            return


def _locked_pump(src: IO[bytes], dst: IO[bytes], lock: threading.Lock) -> None:
    try:
        while True:
            chunk = src.read(4096)
            if not chunk:
                return
            with lock:
                dst.write(chunk)
                dst.flush()
    except (OSError, ValueError):
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit periodic heartbeat bytes while a child CLI runs.",
        add_help=True,
    )
    parser.add_argument("-i", "--interval", type=float, default=60.0,
                        help="Seconds between heartbeat emissions (default: 60.0).")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Child command: `-- <cmd> [args...]`.")
    args = parser.parse_args(argv)

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing child command. use: heartbeat-wrap -- <cmd> [args...]")
    if args.interval <= 0:
        parser.error("--interval must be positive")

    stdout_sink = sys.stdout.buffer
    stderr_sink = sys.stderr.buffer
    stderr_lock = threading.Lock()

    proc = subprocess.Popen(
        cmd,
        stdin=sys.stdin.fileno() if sys.stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    stop = threading.Event()
    stdout_thread = threading.Thread(target=_pump, args=(proc.stdout, stdout_sink), daemon=True)
    stderr_thread = threading.Thread(
        target=_locked_pump,
        args=(proc.stderr, stderr_sink, stderr_lock),
        daemon=True,
    )
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(proc, stop, args.interval, stderr_sink, stderr_lock),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()
    heartbeat_thread.start()

    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            exit_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait()
    finally:
        stop.set()

    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    heartbeat_thread.join(timeout=2.0)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
