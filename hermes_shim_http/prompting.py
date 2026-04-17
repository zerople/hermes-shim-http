from __future__ import annotations

import json
from typing import Any, Iterable

from .models import ToolDefinition


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
        if rendered:
            transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("When you decide to call a tool, use the literal wrapper <tool_call>{...}</tool_call>.")
    return "\n\n".join(section for section in sections if section.strip())
