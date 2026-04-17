from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SlashCommandResult:
    command: str
    message: dict[str, Any]
    model_override: str | None = None
    force_compaction: bool = False


def dispatch_slash_command(text: str, *, model: str, stats: dict[str, Any]) -> SlashCommandResult | None:
    command_text = (text or "").strip()
    if not command_text.startswith("/"):
        return None

    parts = command_text.split(None, 1)
    command = parts[0][1:].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "clear":
        return SlashCommandResult(
            command="clear",
            message={"role": "assistant", "content": "Session cache cleared."},
        )
    if command == "compact":
        return SlashCommandResult(
            command="compact",
            message={"role": "assistant", "content": "Context compaction requested for the current session."},
            force_compaction=True,
        )
    if command == "model":
        target_model = arg or model
        return SlashCommandResult(
            command="model",
            model_override=target_model,
            message={"role": "assistant", "content": f"Model switched to {target_model}."},
        )
    if command == "stats":
        return SlashCommandResult(
            command="stats",
            message={
                "role": "assistant",
                "content": (
                    f"Session stats: cache={stats.get('cache_size', 0)}, "
                    f"active_sessions={stats.get('active_sessions', 0)}, "
                    f"hit_rate={stats.get('hit_rate', 0):.2f}"
                ),
            },
        )
    return None
