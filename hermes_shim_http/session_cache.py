from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .prompting import build_cli_prompt, build_cli_resume_delta_prompt


@dataclass(slots=True)
class SessionPlan:
    session_id: str
    resume_session_id: str | None
    prompt_text: str
    prefix_message_count: int
    messages: list[dict[str, Any]]
    model: str
    tools: list[dict[str, Any]] | None
    tool_choice: Any = None


@dataclass(slots=True)
class _CacheEntry:
    cache_key: str
    session_id: str
    signature: str
    conversation_messages: list[dict[str, Any]]
    created_at: float
    last_used_at: float
    hit_count: int


class SessionCache:
    def __init__(
        self,
        *,
        path: str | None = None,
        max_entries: int = 256,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.path = str(Path(path).expanduser()) if path else None
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._memory_dsn: str | None = None
        self._keeper: sqlite3.Connection | None = None
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_dsn = f"file:hermes_shim_http_{id(self)}?mode=memory&cache=shared"
            self._keeper = sqlite3.connect(self._memory_dsn, uri=True, check_same_thread=False)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    cache_key TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_signature ON sessions(signature)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self.path:
            return sqlite3.connect(self.path, check_same_thread=False)
        if not self._memory_dsn:
            raise RuntimeError("In-memory session cache not initialized")
        return sqlite3.connect(self._memory_dsn, uri=True, check_same_thread=False)

    def plan_request(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any = None,
    ) -> SessionPlan:
        normalized_messages = [self._normalize_message(message) for message in messages]
        now = time.time()
        session_id = str(uuid.uuid4())
        signature = self._signature_prefix(model=model, tools=tools, tool_choice=tool_choice)

        with self._lock:
            self._prune_locked(now)
            best_match: _CacheEntry | None = None
            best_prefix_len = 0
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT cache_key, session_id, signature, messages_json, created_at, last_used, hit_count FROM sessions WHERE signature = ?",
                    (signature,),
                ).fetchall()
                for row in rows:
                    entry = _CacheEntry(
                        cache_key=row[0],
                        session_id=row[1],
                        signature=row[2],
                        conversation_messages=json.loads(row[3]),
                        created_at=float(row[4]),
                        last_used_at=float(row[5]),
                        hit_count=int(row[6]),
                    )
                    prefix_len = self._matching_prefix_length(entry.conversation_messages, normalized_messages)
                    if prefix_len > best_prefix_len:
                        best_prefix_len = prefix_len
                        best_match = entry

                if best_match is not None and best_prefix_len > 0:
                    conn.execute(
                        "UPDATE sessions SET last_used = ?, hit_count = hit_count + 1 WHERE cache_key = ?",
                        (now, best_match.cache_key),
                    )
                    conn.commit()
                    delta_messages = normalized_messages[best_prefix_len:]
                    prompt_text = build_cli_resume_delta_prompt(messages=delta_messages)
                    return SessionPlan(
                        session_id=session_id,
                        resume_session_id=best_match.session_id,
                        prompt_text=prompt_text,
                        prefix_message_count=best_prefix_len,
                        messages=normalized_messages,
                        model=model,
                        tools=tools,
                        tool_choice=tool_choice,
                    )

        return SessionPlan(
            session_id=session_id,
            resume_session_id=None,
            prompt_text=build_cli_prompt(messages=normalized_messages, model=model, tools=tools, tool_choice=tool_choice),
            prefix_message_count=0,
            messages=normalized_messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

    def record_success(self, plan: SessionPlan, *, assistant_messages: list[dict[str, Any]] | None = None) -> None:
        now = time.time()
        conversation_messages = [*plan.messages]
        if assistant_messages:
            conversation_messages.extend(self._normalize_message(message) for message in assistant_messages)
        key = self._conversation_key(
            model=plan.model,
            tools=plan.tools,
            tool_choice=plan.tool_choice,
            messages=conversation_messages,
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (cache_key, session_id, model, signature, messages_json, created_at, last_used, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT hit_count FROM sessions WHERE cache_key = ?), 0))
                    ON CONFLICT(cache_key) DO UPDATE SET
                        session_id=excluded.session_id,
                        model=excluded.model,
                        signature=excluded.signature,
                        messages_json=excluded.messages_json,
                        created_at=excluded.created_at,
                        last_used=excluded.last_used
                    """,
                    (
                        key,
                        plan.session_id,
                        plan.model,
                        self._signature_prefix(model=plan.model, tools=plan.tools, tool_choice=plan.tool_choice),
                        json.dumps(conversation_messages, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                        key,
                    ),
                )
                conn.commit()
            self._prune_locked(now)

    def clear(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM sessions")
                conn.commit()

    def stats(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            with self._connect() as conn:
                cache_size, active_sessions, hit_count = conn.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(hit_count), 0) FROM sessions"
                ).fetchone()
        return {
            "cache_size": int(cache_size or 0),
            "active_sessions": int(active_sessions or 0),
            "hit_count": int(hit_count or 0),
        }

    def _prune_locked(self, now: float) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE ? - last_used > ?", (now, self.ttl_seconds))
            overflow = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] - self.max_entries
            if overflow > 0:
                rows = conn.execute(
                    "SELECT cache_key FROM sessions ORDER BY last_used ASC, created_at ASC LIMIT ?",
                    (overflow,),
                ).fetchall()
                if rows:
                    conn.executemany("DELETE FROM sessions WHERE cache_key = ?", rows)
            conn.commit()

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "role": str(message.get("role") or "context").strip().lower(),
            "content": message.get("content") or "",
            **({"tool_call_id": message.get("tool_call_id")} if message.get("tool_call_id") else {}),
            **({"name": message.get("name")} if message.get("name") else {}),
        }
        if message.get("tool_calls"):
            normalized["tool_calls"] = message.get("tool_calls")
        return normalized

    @staticmethod
    def _signature_prefix(*, model: str, tools: list[dict[str, Any]] | None, tool_choice: Any) -> str:
        payload = {
            "model": model,
            "tools": tools or [],
            "tool_choice": tool_choice,
        }
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def _conversation_key(
        cls,
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        messages: list[dict[str, Any]],
    ) -> str:
        payload = {
            "signature": cls._signature_prefix(model=model, tools=tools, tool_choice=tool_choice),
            "messages": messages,
        }
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _matching_prefix_length(prefix_messages: list[dict[str, Any]], messages: list[dict[str, Any]]) -> int:
        if len(prefix_messages) > len(messages):
            return 0
        for index, prefix_message in enumerate(prefix_messages):
            if messages[index] != prefix_message:
                return 0
        return len(prefix_messages)
