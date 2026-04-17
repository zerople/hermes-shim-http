from __future__ import annotations

import json
import re
from typing import Any

from .models import CliStreamEvent, ParsedShimOutput

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _normalize_tool_call(raw_obj: dict[str, Any], index: int) -> dict[str, Any] | None:
    fn = raw_obj.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = fn.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    call_id = raw_obj.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call_{index}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


class IncrementalToolCallParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._tool_call_count = 0

    def feed(self, chunk: str) -> list[CliStreamEvent]:
        if not chunk:
            return []
        self._buffer += chunk
        return self._drain(final=False)

    def finalize(self) -> list[CliStreamEvent]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[CliStreamEvent]:
        events: list[CliStreamEvent] = []

        while True:
            match = _TOOL_CALL_BLOCK_RE.search(self._buffer)
            if not match:
                break

            prefix = self._buffer[: match.start()]
            if prefix:
                events.append(CliStreamEvent(kind="text", text=prefix))

            raw = match.group(1)
            try:
                obj = json.loads(raw)
            except Exception:
                events.append(CliStreamEvent(kind="text", text=match.group(0)))
            else:
                normalized = _normalize_tool_call(obj, self._tool_call_count + 1)
                if normalized is None:
                    events.append(CliStreamEvent(kind="text", text=match.group(0)))
                else:
                    self._tool_call_count += 1
                    events.append(CliStreamEvent(kind="tool_call", tool_call=normalized))

            self._buffer = self._buffer[match.end() :]

        if final:
            if self._buffer:
                events.append(CliStreamEvent(kind="text", text=self._buffer))
                self._buffer = ""
            return events

        safe_length = self._safe_prefix_length(self._buffer)
        if safe_length > 0:
            safe_text = self._buffer[:safe_length]
            if safe_text:
                events.append(CliStreamEvent(kind="text", text=safe_text))
            self._buffer = self._buffer[safe_length:]
        return events

    @staticmethod
    def _safe_prefix_length(buffer: str) -> int:
        if not buffer:
            return 0

        open_index = buffer.find(_TOOL_CALL_OPEN)
        close_index = buffer.find(_TOOL_CALL_CLOSE)
        candidate_indexes = [idx for idx in (open_index, close_index) if idx >= 0]
        if candidate_indexes:
            return min(candidate_indexes)

        for start in range(len(buffer)):
            suffix = buffer[start:]
            if _TOOL_CALL_OPEN.startswith(suffix) or _TOOL_CALL_CLOSE.startswith(suffix):
                return start

        return len(buffer)


def parse_cli_output(text: str) -> ParsedShimOutput:
    if not isinstance(text, str) or not text.strip():
        return ParsedShimOutput(content="", tool_calls=[])

    tool_calls: list[dict[str, Any]] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = match.group(1)
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        normalized = _normalize_tool_call(obj, len(tool_calls) + 1)
        if normalized is None:
            continue
        tool_calls.append(normalized)
        consumed_spans.append((match.start(), match.end()))

    if not consumed_spans:
        return ParsedShimOutput(content=text.strip(), tool_calls=[])

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ParsedShimOutput(content=cleaned, tool_calls=tool_calls)
