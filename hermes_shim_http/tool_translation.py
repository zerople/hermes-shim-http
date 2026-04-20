"""Translate Claude Code native tool calls into Hermes-compatible equivalents.

Claude Code CLI emits tool calls for its built-in tools (Read, Edit, Write,
Glob, Grep, Bash, TodoWrite) using its own name/argument conventions. Hermes
only advertises its own tool suite (read_file, patch, write_file, search_files,
terminal, todo, ...). Without translation, every Claude-native call arrives as
"unsupported" and the model cannot actually read/edit files.

This module provides:

1. A narrow translation layer: map Claude-native tool names to Hermes
   equivalents and rewrite argument keys accordingly.
2. A registry of known Claude-native tool names (``CLAUDE_NATIVE_TOOL_NAMES``).
   When a native tool has no Hermes equivalent (WebSearch, WebFetch,
   NotebookEdit, ExitPlanMode, ...), Claude Code has already executed it
   internally via ``--dangerously-skip-permissions``. The caller can use
   ``is_claude_native_without_hermes_equivalent`` to drop the corresponding
   ``tool_use`` event silently instead of surfacing a noisy "unsupported"
   message to the user — Hermes tools are preferred whenever available, but
   Claude's built-ins remain usable for everything else at no extra token
   cost (no system-prompt text required).
"""

from __future__ import annotations

import json
from typing import Any, Callable

_Translator = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]


def _translate_hermes_mcp_prefixed(name: str, *, allowed_names: set[str] | None) -> str | None:
    prefix = "mcp__hermes__"
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    translated = name[len(prefix):].strip()
    if not translated:
        return None
    if allowed_names is not None and translated not in allowed_names:
        return None
    return translated


def _rename(mapping: dict[str, str], *, target: str, extra: dict[str, Any] | None = None) -> _Translator:
    def translate(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        out: dict[str, Any] = {}
        for src, dst in mapping.items():
            if src in args and args[src] is not None:
                out[dst] = args[src]
        if extra:
            out.update(extra)
        return target, out

    return translate


def _translate_edit(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    out: dict[str, Any] = {"mode": "replace"}
    if "file_path" in args:
        out["path"] = args["file_path"]
    for key in ("old_string", "new_string", "replace_all"):
        if key in args and args[key] is not None:
            out[key] = args[key]
    return "patch", out


def _translate_grep(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    out: dict[str, Any] = {"target": "content"}
    if "pattern" in args:
        out["pattern"] = args["pattern"]
    if "path" in args and args["path"] is not None:
        out["path"] = args["path"]
    if "glob" in args and args["glob"] is not None:
        out["file_glob"] = args["glob"]
    for src, dst in (("head_limit", "limit"), ("offset", "offset"), ("output_mode", "output_mode")):
        if src in args and args[src] is not None:
            out[dst] = args[src]
    ctx = args.get("context")
    if ctx is None:
        ctx = args.get("-C")
    if ctx is None:
        ctx = args.get("-A") or args.get("-B")
    if ctx is not None:
        out["context"] = ctx
    return "search_files", out


def _translate_glob(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    out: dict[str, Any] = {"target": "files"}
    if "pattern" in args:
        out["file_glob"] = args["pattern"]
    if "path" in args and args["path"] is not None:
        out["path"] = args["path"]
    return "search_files", out


def _translate_bash(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    out: dict[str, Any] = {}
    if "command" in args:
        out["command"] = args["command"]
    if "timeout" in args and args["timeout"] is not None:
        timeout_ms = args["timeout"]
        out["timeout"] = int(timeout_ms / 1000) if isinstance(timeout_ms, (int, float)) and timeout_ms > 1000 else timeout_ms
    if args.get("run_in_background"):
        out["background"] = True
    return "terminal", out


def _translate_todowrite(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    out: dict[str, Any] = {}
    todos = args.get("todos")
    if isinstance(todos, list):
        sanitized: list[dict[str, Any]] = []
        for idx, item in enumerate(todos):
            if not isinstance(item, dict):
                continue
            sanitized.append(
                {
                    "id": str(item.get("id") or f"t{idx + 1}"),
                    "content": str(item.get("content") or item.get("activeForm") or ""),
                    "status": str(item.get("status") or "pending"),
                }
            )
        out["todos"] = sanitized
    return "todo", out


_TRANSLATORS: dict[str, _Translator] = {
    "Read": _rename({"file_path": "path", "offset": "offset", "limit": "limit"}, target="read_file"),
    "Write": _rename({"file_path": "path", "content": "content"}, target="write_file"),
    "Edit": _translate_edit,
    "Glob": _translate_glob,
    "Grep": _translate_grep,
    "Bash": _translate_bash,
    "TodoWrite": _translate_todowrite,
}


# Known Claude Code native tools. When Claude Code runs with
# --dangerously-skip-permissions (as this shim launches it), it executes these
# tools internally and continues the turn on its own; the shim only needs to
# avoid surfacing them as "unsupported" to Hermes. Keep this list in sync with
# Claude Code's published tool set.
CLAUDE_NATIVE_TOOL_NAMES: frozenset[str] = frozenset({
    "Agent",
    "Bash",
    "BashOutput",
    "Edit",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillBash",
    "KillShell",
    "NotebookEdit",
    "Read",
    "ScheduleWakeup",
    "Skill",
    "SlashCommand",
    "Task",
    "TodoWrite",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Write",
})


def is_claude_native_without_hermes_equivalent(
    name: str, *, allowed_names: set[str] | None
) -> bool:
    """Return True if *name* is a Claude-native tool that cannot be forwarded.

    "Cannot be forwarded" means either:
      - no translator maps it to a Hermes tool name, or
      - a translator exists but the translated target is not in
        ``allowed_names`` (Hermes did not advertise it on this request).

    Callers use this to decide whether to drop a ``tool_use`` event silently
    (native-tool fallback: Claude executed it internally) vs. emit the
    "unsupported" error (genuinely unknown tool).
    """
    if not name or name not in CLAUDE_NATIVE_TOOL_NAMES:
        return False
    translator = _TRANSLATORS.get(name)
    if translator is None:
        return True
    if allowed_names is None:
        return False
    try:
        translated_name, _ = translator({})
    except Exception:
        return True
    return translated_name not in allowed_names


def translate_tool_call(tool_call: dict[str, Any], *, allowed_names: set[str] | None = None) -> dict[str, Any]:
    """Return a Hermes-compatible tool call for a Claude-native one.

    If the tool name is already in ``allowed_names``, the call is returned
    unchanged. Otherwise, if a translator exists for this Claude-native name
    and the translated target is acceptable (either the allowlist permits it,
    or no allowlist was supplied), the name and arguments are rewritten.
    Calls with no translator are returned unchanged so the caller's allowlist
    check can reject them as unsupported.
    """
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return tool_call
    name = str(fn.get("name") or "").strip()
    if not name:
        return tool_call
    hermes_mcp_name = _translate_hermes_mcp_prefixed(name, allowed_names=allowed_names)
    if hermes_mcp_name:
        return {
            **tool_call,
            "function": {
                **fn,
                "name": hermes_mcp_name,
            },
        }
    if allowed_names is not None and name in allowed_names:
        return tool_call
    translator = _TRANSLATORS.get(name)
    if translator is None:
        return tool_call
    raw_args = fn.get("arguments") or "{}"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except Exception:
            return tool_call
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return tool_call
    try:
        new_name, new_args = translator(args)
    except Exception:
        return tool_call
    if allowed_names is not None and new_name not in allowed_names:
        return tool_call
    return {
        **tool_call,
        "function": {
            "name": new_name,
            "arguments": json.dumps(new_args, ensure_ascii=False),
        },
    }
