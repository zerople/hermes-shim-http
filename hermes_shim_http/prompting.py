from __future__ import annotations

import json
from typing import Any, Iterable

from .models import ToolDefinition
from .token_usage import DEFAULT_CONTEXT_LIMIT, estimate_context_tokens


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _render_tools(tools: Iterable[ToolDefinition] | None) -> str:
    if not tools:
        return ""
    payload = []
    for tool in tools:
        payload.append(
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _render_transcript(messages: list[dict[str, Any]] | list[Any]) -> list[str]:
    transcript: list[str] = []
    for raw in messages:
        message = raw if isinstance(raw, dict) else raw.model_dump()
        role = str(message.get("role") or "context").strip().lower()
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(role, "Context")
        rendered = _render_content(message.get("content"))
        tool_calls = message.get("tool_calls")
        if rendered:
            transcript.append(f"{label}:\n{rendered}")
        elif tool_calls:
            transcript.append(f"{label} tool calls:\n{json.dumps(tool_calls, ensure_ascii=False)}")
    return transcript


def compact_messages(
    *,
    messages: list[dict[str, Any]] | list[Any],
    mode: str,
    threshold: float,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    force: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    normalized = [(message if isinstance(message, dict) else message.model_dump()) for message in messages]
    if mode == "off":
        return normalized, False

    used = estimate_context_tokens(normalized)
    if not force and used <= max(1, int(context_limit * threshold)):
        return normalized, False

    system_messages = [message for message in normalized if str(message.get("role", "")).lower() == "system"]
    conversation_messages = [message for message in normalized if str(message.get("role", "")).lower() != "system"]
    if len(conversation_messages) <= 2:
        return normalized, False

    if mode == "window":
        window = conversation_messages[-3:]
        compacted = [*system_messages, *window]
        return compacted, compacted != normalized

    summary_source = conversation_messages[:-3]
    tail = conversation_messages[-3:]
    summary_text = " | ".join(_render_content(message.get("content")) for message in summary_source if _render_content(message.get("content")))
    summary_text = summary_text[:280] or "Earlier conversation omitted."
    compacted = [
        *system_messages,
        {
            "role": "system",
            "content": f"Summary of earlier conversation: {summary_text}",
        },
        *tail,
    ]
    return compacted, compacted != normalized


def build_cli_resume_delta_prompt(*, messages: list[dict[str, Any]] | list[Any]) -> str:
    return "\n\n".join(_render_transcript(messages)).strip()


def build_cli_prompt(*, messages: list[dict[str, Any]] | list[Any], model: str, tools: list[dict[str, Any]] | list[ToolDefinition] | None, tool_choice: Any = None) -> str:
    sections = [
        "You are being used as a local HTTP shim backend for Hermes Agent.",
        f"Hermes requested model hint: {model}",
        "Do NOT use any native CLI built-in tools, editors, shell integrations, tmux workflows, or file operations.",
        "You must behave only as a reasoning engine that emits Hermes-compatible tool calls when action is required.",
        "If a tool is needed, emit ONLY <tool_call>{...}</tool_call> blocks with JSON in OpenAI function-call shape.",
        "Use exactly one JSON object per <tool_call> block containing id, type, and function{name,arguments}.",
        "If no tool is needed, answer normally in plain text.",
    ]

    if tools:
        normalized_tools: list[ToolDefinition] = []
        for item in tools:
            if isinstance(item, ToolDefinition):
                normalized_tools.append(item)
            else:
                normalized_tools.append(ToolDefinition.model_validate(item))
        sections.append("Available Hermes tools (OpenAI function schema):\n" + _render_tools(normalized_tools))

    if tool_choice is not None:
        sections.append("Tool choice hint: " + json.dumps(tool_choice, ensure_ascii=False))

    transcript = _render_transcript(messages)
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("When you decide to call a tool, use the literal wrapper <tool_call>{...}</tool_call>.")
    return "\n\n".join(section for section in sections if section.strip())
