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

from .prompting import build_cli_resume_delta_prompt, build_cli_system_prompt, build_cli_user_prompt
from .telemetry import emit_event, env_flag


@dataclass(slots=True)
class SessionPlan:
    session_id: str
    resume_session_id: str | None
    prompt_text: str
    system_prompt_text: str | None
    tool_call_nonce: str | None
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
        self._in_flight_parents: set[str] = set()
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
        tool_call_nonce: str | None = None,
    ) -> SessionPlan:
        normalized_messages = [self._normalize_message(message) for message in messages]
        now = time.time()
        session_id = str(uuid.uuid4())
        signature = self._signature_prefix(model=model, tools=tools, tool_choice=tool_choice)
        system_prompt_text = build_cli_system_prompt(tools=tools, tool_choice=tool_choice, model=model, tool_call_nonce=tool_call_nonce)

        with self._lock:
            self._prune_locked(now)
            best_match: _CacheEntry | None = None
            best_prefix_len = 0
            candidate_count = 0
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
                    candidate_count += 1
                    prefix_len, mismatch = self._matching_prefix_length(entry.conversation_messages, normalized_messages)
                    if env_flag("HERMES_SHIM_HTTP_DEBUG_SESSION_CACHE"):
                        emit_event(
                            "session_candidate",
                            candidate_session_id=entry.session_id,
                            model=model,
                            candidate_message_count=len(entry.conversation_messages),
                            incoming_message_count=len(normalized_messages),
                            prefix_len=prefix_len,
                            mismatch=mismatch,
                        )
                    if mismatch is None and prefix_len > best_prefix_len:
                        best_prefix_len = prefix_len
                        best_match = entry

                if best_match is not None and best_prefix_len > 0 and best_match.session_id not in self._in_flight_parents:
                    conn.execute(
                        "UPDATE sessions SET last_used = ?, hit_count = hit_count + 1 WHERE cache_key = ?",
                        (now, best_match.cache_key),
                    )
                    conn.commit()
                    self._in_flight_parents.add(best_match.session_id)
                    delta_messages = normalized_messages[best_prefix_len:]
                    prompt_text = build_cli_resume_delta_prompt(messages=delta_messages, tool_call_nonce=tool_call_nonce)
                    emit_event(
                        "session_plan",
                        mode="resume",
                        model=model,
                        candidate_count=candidate_count,
                        best_prefix_len=best_prefix_len,
                        incoming_message_count=len(normalized_messages),
                        matched_session_id=best_match.session_id,
                        new_session_id=session_id,
                    )
                    return SessionPlan(
                        session_id=session_id,
                        resume_session_id=best_match.session_id,
                        prompt_text=prompt_text,
                        system_prompt_text=system_prompt_text,
                        tool_call_nonce=tool_call_nonce,
                        prefix_message_count=best_prefix_len,
                        messages=normalized_messages,
                        model=model,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                if best_match is not None and best_match.session_id in self._in_flight_parents:
                    emit_event(
                        "session_plan_in_flight_skip",
                        model=model,
                        parent_session_id=best_match.session_id,
                        new_session_id=session_id,
                    )

        emit_event(
            "session_plan",
            mode="new",
            model=model,
            candidate_count=candidate_count,
            best_prefix_len=best_prefix_len,
            incoming_message_count=len(normalized_messages),
            matched_session_id=best_match.session_id if best_match else None,
            new_session_id=session_id,
        )
        return SessionPlan(
            session_id=session_id,
            resume_session_id=None,
            prompt_text=build_cli_user_prompt(messages=normalized_messages, tool_call_nonce=tool_call_nonce),
            system_prompt_text=system_prompt_text,
            tool_call_nonce=tool_call_nonce,
            prefix_message_count=0,
            messages=normalized_messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

    def release_plan(self, plan: SessionPlan) -> None:
        if not plan.resume_session_id:
            return
        with self._lock:
            self._in_flight_parents.discard(plan.resume_session_id)

    def record_success(
        self,
        plan: SessionPlan,
        *,
        assistant_messages: list[dict[str, Any]] | None = None,
        actual_session_id: str | None = None,
    ) -> None:
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
            if plan.resume_session_id:
                self._in_flight_parents.discard(plan.resume_session_id)
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
                        actual_session_id or plan.session_id,
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
            with self._connect() as conn:
                cache_entry_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
        emit_event(
            "session_recorded",
            session_id=actual_session_id or plan.session_id,
            resume_session_id=plan.resume_session_id,
            model=plan.model,
            request_message_count=len(plan.messages),
            stored_message_count=len(conversation_messages),
            assistant_message_count=len(assistant_messages or []),
            cache_entry_count=cache_entry_count,
        )

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

    @classmethod
    def _normalize_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "role": str(message.get("role") or "context").strip().lower(),
            "content": cls._normalize_content(message.get("content")),
        }
        if message.get("tool_calls"):
            normalized["tool_calls"] = cls._normalize_tool_calls(message.get("tool_calls"))
        if message.get("tool_call_id"):
            normalized["tool_call_id"] = message.get("tool_call_id")
        if message.get("name"):
            normalized["name"] = message.get("name")
        return normalized

    @classmethod
    def _normalize_tool_calls(cls, tool_calls: Any) -> list[dict[str, Any]]:
        normalized_calls: list[dict[str, Any]] = []
        if not isinstance(tool_calls, list):
            return normalized_calls
        for item in tool_calls:
            if not isinstance(item, dict):
                normalized_calls.append({"raw": cls._normalize_content(item)})
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else {}
            normalized_calls.append(
                {
                    "id": str(item.get("id") or "").strip(),
                    "type": str(item.get("type") or "function").strip() or "function",
                    "function": {
                        "name": str(function.get("name") or "").strip(),
                        "arguments": cls._normalize_content(function.get("arguments")),
                    },
                }
            )
        return normalized_calls

    @classmethod
    def _normalize_content(cls, content: Any) -> Any:
        if content is None:
            return ""
        if isinstance(content, str):
            return cls._collapse_whitespace(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                normalized = cls._normalize_content(item)
                if isinstance(normalized, str) and normalized:
                    parts.append(normalized)
                elif normalized not in {"", None}:
                    parts.append(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return cls._collapse_whitespace(content.get("text") or "")
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        return cls._collapse_whitespace(str(content))

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return "\n".join(line.strip() for line in str(text).replace("\r\n", "\n").split("\n")).strip()

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

    @classmethod
    def _matching_prefix_length(cls, prefix_messages: list[dict[str, Any]], messages: list[dict[str, Any]]) -> tuple[int, dict[str, Any] | None]:
        if len(prefix_messages) > len(messages):
            return 0, {
                "reason": "incoming_shorter_than_candidate",
                "candidate_message_count": len(prefix_messages),
                "incoming_message_count": len(messages),
            }
        for index, prefix_message in enumerate(prefix_messages):
            incoming_message = messages[index]
            if incoming_message != prefix_message:
                return index, {
                    "reason": "message_mismatch",
                    "index": index,
                    "candidate": cls._message_debug_repr(prefix_message),
                    "incoming": cls._message_debug_repr(incoming_message),
                }
        return len(prefix_messages), None

    @staticmethod
    def _message_debug_repr(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)
        return {
            "role": message.get("role"),
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
            "content_len": len(text),
            "content_preview": text[:160],
        }
