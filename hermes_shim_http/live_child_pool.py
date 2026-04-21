"""LiveChildPool — keep one long-lived claude child per conversation.

Claude Code CLI with `--input-format stream-json` accepts multiple user
messages on a single stdin stream. Each `{"type":"result"}` line marks
end-of-turn while the process stays alive waiting for the next prompt.

The pool maps session_key → one live child. Prompts on the same key reuse
that process (avoiding cold-start overhead). Idle children are evicted after
`idle_ttl`; the pool is LRU-capped by `size`.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator

from .models import CliStreamEvent
from .parsing import ClaudeStreamJsonParser


@dataclass(slots=True)
class LiveChildTurnResult:
    stdout: str
    stderr: str
    session_id: str | None
    is_error: bool


class _LiveChild:
    def __init__(self, *, spawn_command: list[str], cwd: str) -> None:
        self._spawn_command = list(spawn_command)
        self._cwd = cwd
        self._process = subprocess.Popen(
            self._spawn_command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._chunks: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
        self._stdout_reader = threading.Thread(target=self._pump, args=(self._process.stdout, "stdout"), daemon=True)
        self._stderr_reader = threading.Thread(target=self._pump, args=(self._process.stderr, "stderr"), daemon=True)
        self._stdout_reader.start()
        self._stderr_reader.start()
        self._turn_lock = threading.Lock()
        self.last_used = time.time()

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def _pump(self, stream, stream_name: str) -> None:
        if stream is None:
            self._chunks.put((stream_name, None))
            return
        try:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace")
                if text:
                    self._chunks.put((stream_name, text))
        finally:
            try:
                stream.close()
            except Exception:
                pass
            self._chunks.put((stream_name, None))

    @staticmethod
    def _build_payload(prompt: str) -> bytes:
        payload = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        }
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    def send_and_stream(
        self,
        prompt: str,
        *,
        read_timeout: float = 30.0,
        hard_deadline: float | None = None,
        max_output_bytes: int | None = None,
        on_complete=None,
    ) -> Iterator[CliStreamEvent]:
        if self._process.stdin is None:
            raise RuntimeError("LiveChild: stdin pipe unavailable")
        with self._turn_lock:
            self._process.stdin.write(self._build_payload(prompt))
            self._process.stdin.flush()

            parser = ClaudeStreamJsonParser(synthesize_progress=False)
            started = time.time()
            last_activity = started
            saw_result = False
            stdout_done = False
            stderr_done = False
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            total_bytes = 0

            try:
                while not saw_result:
                    now = time.time()
                    if now - last_activity > read_timeout:
                        raise TimeoutError(f"LiveChild: CLI process idle for {read_timeout:.1f}s with no real stdout/stderr output")
                    if hard_deadline and (now - started) > hard_deadline:
                        raise TimeoutError(f"LiveChild: CLI process exceeded hard deadline of {hard_deadline:.1f}s")
                    try:
                        stream_name, chunk = self._chunks.get(timeout=min(max(0.1, read_timeout), 1.0))
                    except queue.Empty:
                        continue
                    if chunk is None:
                        if stream_name == "stdout":
                            stdout_done = True
                        else:
                            stderr_done = True
                        if self._process.poll() is not None and (stdout_done or stderr_done):
                            raise RuntimeError("LiveChild: child exited mid-turn")
                        continue

                    cleaned = chunk.replace("\u200b", "")
                    if not cleaned:
                        continue
                    last_activity = time.time()

                    if stream_name == "stdout":
                        stdout_chunks.append(cleaned)
                        if max_output_bytes:
                            total_bytes += len(cleaned.encode("utf-8"))
                            if total_bytes >= max_output_bytes:
                                raise RuntimeError(f"LiveChild: CLI process exceeded max output cap of {max_output_bytes} bytes")
                        try:
                            decoded = json.loads(cleaned.strip())
                            if isinstance(decoded, dict) and decoded.get("type") == "result":
                                saw_result = True
                        except Exception:
                            pass
                        for event in parser.feed(cleaned):
                            yield event
                    else:
                        stderr_chunks.append(cleaned)

                for event in parser.finalize():
                    yield event
            finally:
                self.last_used = time.time()

            metadata = parser.result_metadata()
            if on_complete is not None:
                on_complete(
                    LiveChildTurnResult(
                        stdout="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                        session_id=metadata.session_id,
                        is_error=metadata.is_error,
                    )
                )

    def terminate(self) -> None:
        if self._process.poll() is not None:
            try:
                self._process.wait(timeout=0.2)
            except Exception:
                pass
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=2.0)
        except Exception:
            try:
                self._process.kill()
                self._process.wait(timeout=1.0)
            except Exception:
                pass


class LiveChildPool:
    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        size: int = 8,
        idle_ttl: float = 300.0,
    ) -> None:
        self._default_command = command
        self._default_args = list(args or [])
        self._default_cwd = cwd
        self._size = max(1, int(size))
        self._idle_ttl = float(idle_ttl)
        self._children: "OrderedDict[str, _LiveChild]" = OrderedDict()
        self._lock = threading.Lock()

    def stream(
        self,
        session_key: str,
        prompt: str,
        *,
        spawn_command: list[str] | None = None,
        cwd: str | None = None,
        read_timeout: float = 30.0,
        hard_deadline: float | None = None,
        max_output_bytes: int | None = None,
        on_complete=None,
    ) -> Iterator[CliStreamEvent]:
        child = self._acquire(session_key, spawn_command=spawn_command, cwd=cwd)
        try:
            yield from child.send_and_stream(
                prompt,
                read_timeout=read_timeout,
                hard_deadline=hard_deadline,
                max_output_bytes=max_output_bytes,
                on_complete=on_complete,
            )
        finally:
            with self._lock:
                if session_key in self._children:
                    self._children.move_to_end(session_key)

    def peek_pid(self, session_key: str) -> int | None:
        with self._lock:
            child = self._children.get(session_key)
            if child is None or not child.is_alive():
                return None
            return child.pid

    def sweep(self) -> None:
        now = time.time()
        with self._lock:
            stale = [key for key, child in self._children.items() if now - child.last_used > self._idle_ttl]
            victims = [self._children.pop(key) for key in stale]
        for victim in victims:
            victim.terminate()

    def shutdown(self) -> None:
        with self._lock:
            victims = list(self._children.values())
            self._children.clear()
        for victim in victims:
            victim.terminate()

    def _resolve_spawn(self, *, spawn_command: list[str] | None, cwd: str | None) -> tuple[list[str], str]:
        if spawn_command is None:
            if self._default_command is None or self._default_cwd is None:
                raise RuntimeError("LiveChildPool: spawn_command/cwd required when no constructor defaults were provided")
            return [self._default_command, *self._default_args], self._default_cwd
        resolved_cwd = cwd or self._default_cwd
        if resolved_cwd is None:
            raise RuntimeError("LiveChildPool: cwd required")
        return list(spawn_command), resolved_cwd

    def _acquire(self, session_key: str, *, spawn_command: list[str] | None, cwd: str | None) -> _LiveChild:
        evicted: list[_LiveChild] = []
        resolved_command, resolved_cwd = self._resolve_spawn(spawn_command=spawn_command, cwd=cwd)
        with self._lock:
            child = self._children.get(session_key)
            if child is not None and child.is_alive():
                self._children.move_to_end(session_key)
                return child
            if child is not None:
                self._children.pop(session_key)
                evicted.append(child)
            new_child = _LiveChild(spawn_command=resolved_command, cwd=resolved_cwd)
            self._children[session_key] = new_child
            while len(self._children) > self._size:
                _key, victim = self._children.popitem(last=False)
                evicted.append(victim)
        for victim in evicted:
            victim.terminate()
        return new_child
