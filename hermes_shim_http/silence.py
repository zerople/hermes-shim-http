from __future__ import annotations

import os
import re

DEFAULT_SILENT_SENTINEL = "<silent/>"
_SILENT_ENV_VAR = "HERMES_SHIM_SILENT_SENTINEL"


def silent_sentinel() -> str:
    raw = os.getenv(_SILENT_ENV_VAR, "").strip()
    return raw or DEFAULT_SILENT_SENTINEL


def _sentinel_pattern(sentinel: str) -> re.Pattern[str]:
    return re.compile(re.escape(sentinel))


def detect_and_strip(content: str, *, has_tool_calls: bool = False) -> tuple[str, bool]:
    """Strip the silence sentinel from content.

    Returns (cleaned_content, silent). silent is True only when the sentinel was
    present AND no other meaningful content / tool calls remain — that is, the
    model's whole response was the silence signal.
    """
    if not isinstance(content, str) or not content:
        return content or "", False
    sentinel = silent_sentinel()
    if sentinel not in content:
        return content, False
    cleaned = _sentinel_pattern(sentinel).sub("", content).strip()
    silent = not cleaned and not has_tool_calls
    return cleaned, silent
