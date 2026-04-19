from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

DEFAULT_CONTEXT_LIMIT = 8192


def context_limit_for_model(model: str | None, *, profile: str | None = None) -> int:
    name = (model or "").strip().lower()
    profile_name = (profile or "").strip().lower()
    if "opus" in name:
        return 1_000_000
    if profile_name == "claude" or "claude" in name or name in {"sonnet", "haiku"}:
        return 200_000
    if profile_name == "codex" or "gpt-" in name or "codex" in name:
        return 400_000
    if profile_name == "opencode":
        return 200_000
    return 128_000


@dataclass(frozen=True, slots=True)
class TokenUsageEstimate:
    context_tokens_used: int
    context_tokens_limit: int
    response_tokens: int


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return " ".join(str(value) for value in content.values())
    if isinstance(content, list):
        return " ".join(_flatten_content(item) for item in content)
    return str(content)


def estimate_text_tokens(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, ceil(len(normalized) / 4))


def estimate_context_tokens(messages: list[dict[str, Any]] | list[Any]) -> int:
    total = 0
    for raw in messages:
        message = raw if isinstance(raw, dict) else raw.model_dump()
        total += 4 + estimate_text_tokens(_flatten_content(message.get("content")))
        if message.get("name"):
            total += 1
        if message.get("tool_call_id"):
            total += 1
    return total


def estimate_token_usage(
    *,
    messages: list[dict[str, Any]] | list[Any],
    response_text: str,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> TokenUsageEstimate:
    return TokenUsageEstimate(
        context_tokens_used=estimate_context_tokens(messages),
        context_tokens_limit=context_limit,
        response_tokens=estimate_text_tokens(response_text),
    )
