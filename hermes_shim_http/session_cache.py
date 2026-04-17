from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
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
    session_id: str
    signature: str
    conversation_messages: list[dict[str, Any]]
    created_at: float
    last_used_at: float


class SessionCache:
    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, _CacheEntry] = {}

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
        self._prune(now)
        session_id = str(uuid.uuid4())
        best_match: _CacheEntry | None = None
        best_prefix_len = 0
        signature = self._signature_prefix(model=model, tools=tools, tool_choice=tool_choice)

        for entry in self._entries.values():
            if entry.signature != signature:
                continue
            prefix_len = self._matching_prefix_length(entry.conversation_messages, normalized_messages)
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_match = entry

        if best_match is not None and best_prefix_len > 0:
            delta_messages = normalized_messages[best_prefix_len:]
            prompt_text = build_cli_resume_delta_prompt(messages=delta_messages)
            best_match.last_used_at = now
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
        self._entries[key] = _CacheEntry(
            session_id=plan.session_id,
            signature=self._signature_prefix(model=plan.model, tools=plan.tools, tool_choice=plan.tool_choice),
            conversation_messages=conversation_messages,
            created_at=now,
            last_used_at=now,
        )
        self._prune(now)

    def _prune(self, now: float) -> None:
        expired_keys = [
            key for key, entry in self._entries.items() if now - entry.last_used_at > self.ttl_seconds
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
        while len(self._entries) > self.max_entries:
            oldest_key = min(self._entries.items(), key=lambda item: item[1].last_used_at)[0]
            self._entries.pop(oldest_key, None)

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": str(message.get("role") or "context").strip().lower(),
            "content": message.get("content") or "",
            **({"tool_call_id": message.get("tool_call_id")} if message.get("tool_call_id") else {}),
            **({"name": message.get("name")} if message.get("name") else {}),
        }

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
