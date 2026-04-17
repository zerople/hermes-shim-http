from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .prompting import build_cli_resume_delta_prompt, build_cli_system_prompt, build_cli_user_prompt
from .telemetry import emit_event, env_flag


@dataclass(slots=True)
class SessionPlan:
    session_id: str
    resume_session_id: str | None
    prompt_text: str
    system_prompt_text: str | None
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
        candidate_count = 0
        signature = self._signature_prefix(model=model, tools=tools, tool_choice=tool_choice)
        system_prompt_text = build_cli_system_prompt(tools=tools, tool_choice=tool_choice)

        for entry in self._entries.values():
            if entry.signature != signature:
                continue
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

        if best_match is not None and best_prefix_len > 0:
            delta_messages = normalized_messages[best_prefix_len:]
            prompt_text = build_cli_resume_delta_prompt(messages=delta_messages)
            best_match.last_used_at = now
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
                prefix_message_count=best_prefix_len,
                messages=normalized_messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
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
            prompt_text=build_cli_user_prompt(messages=normalized_messages),
            system_prompt_text=system_prompt_text,
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
        emit_event(
            "session_recorded",
            session_id=plan.session_id,
            resume_session_id=plan.resume_session_id,
            model=plan.model,
            request_message_count=len(plan.messages),
            stored_message_count=len(conversation_messages),
            assistant_message_count=len(assistant_messages or []),
            cache_entry_count=len(self._entries),
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

    @classmethod
    def _normalize_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        tool_calls = message.get("tool_calls")
        normalized: dict[str, Any] = {
            "role": str(message.get("role") or "context").strip().lower(),
            "content": cls._normalize_content(message.get("content")),
        }
        if tool_calls:
            normalized["tool_calls"] = cls._normalize_tool_calls(tool_calls)
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
