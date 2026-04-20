from __future__ import annotations

import json
import re
from typing import Any
from dataclasses import dataclass

from .models import CliStreamEvent, ParsedShimOutput
from .silence import detect_and_strip as _detect_silent

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

_CLAUDE_IGNORED_DELTA_TYPES = frozenset({"thinking_delta", "signature_delta"})


@dataclass(frozen=True, slots=True)
class ClaudeResultMetadata:
    session_id: str | None = None
    result_text: str = ""
    is_error: bool = False


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

        tail_window = max(len(_TOOL_CALL_OPEN), len(_TOOL_CALL_CLOSE)) - 1
        tail_start = max(0, len(buffer) - tail_window)
        for start in range(tail_start, len(buffer)):
            suffix = buffer[start:]
            if _TOOL_CALL_OPEN.startswith(suffix) or _TOOL_CALL_CLOSE.startswith(suffix):
                return start

        return len(buffer)


class ClaudeStreamJsonParser:
    """Parse Claude Code CLI --output-format=stream-json output.

    Claude emits newline-delimited JSON objects. With --include-partial-messages
    each assistant content block arrives as a sequence of deltas (text_delta,
    thinking_delta, input_json_delta) bracketed by content_block_start/stop.
    Aggregate `assistant` events may also arrive as snapshots.

    This parser emits CliStreamEvent(kind="text") for each text_delta and one
    CliStreamEvent(kind="tool_call") per completed tool_use block. In live
    streaming mode it can also synthesize lightweight progress text from
    thinking/tool_use block starts so upstream chat UIs keep moving even when
    Anthropic does not stream the body of a thinking block.
    """

    def __init__(self, *, synthesize_progress: bool = False) -> None:
        self._line_buffer = ""
        self._blocks: dict[int, dict[str, Any]] = {}
        self._tool_call_count = 0
        self._seen_assistant_msg_ids: set[str] = set()
        self._saw_stream_events = False
        self._saw_any_json = False
        self._tag_parser = IncrementalToolCallParser()
        self._session_id: str | None = None
        self._result_text = ""
        self._result_is_error = False
        self._synthesize_progress = synthesize_progress
        self._thinking_emitted_this_turn = False

    def saw_any_json(self) -> bool:
        return self._saw_any_json

    def result_metadata(self) -> ClaudeResultMetadata:
        return ClaudeResultMetadata(
            session_id=self._session_id,
            result_text=self._result_text,
            is_error=self._result_is_error,
        )

    def feed(self, chunk: str) -> list[CliStreamEvent]:
        if not chunk:
            return []
        self._line_buffer += chunk
        events: list[CliStreamEvent] = []
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            events.extend(self._handle_line(line))
        return events

    def finalize(self) -> list[CliStreamEvent]:
        events: list[CliStreamEvent] = []
        tail = self._line_buffer.strip()
        self._line_buffer = ""
        if tail:
            events.extend(self._handle_line(tail))
        events.extend(self._tag_parser.finalize())
        return events

    def _handle_line(self, line: str) -> list[CliStreamEvent]:
        try:
            event = json.loads(line)
        except Exception:
            return []
        if not isinstance(event, dict):
            return []
        self._saw_any_json = True
        event_type = event.get("type")
        if event_type == "stream_event":
            inner = event.get("event") or {}
            if isinstance(inner, dict):
                return self._handle_stream_event(inner)
            return []
        if event_type == "assistant":
            message = event.get("message") or {}
            if isinstance(message, dict):
                return self._handle_assistant_aggregate(message)
            return []
        if event_type == "system":
            session_id = event.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self._session_id = session_id.strip()
            return []
        if event_type == "result":
            session_id = event.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self._session_id = session_id.strip()
            result_text = event.get("result")
            if isinstance(result_text, str):
                self._result_text = result_text
            self._result_is_error = bool(event.get("is_error"))
            return []
        return []

    def _handle_stream_event(self, event: dict[str, Any]) -> list[CliStreamEvent]:
        self._saw_stream_events = True
        inner_type = event.get("type")
        if inner_type == "message_start":
            self._blocks = {}
            self._thinking_emitted_this_turn = False
            return []
        if inner_type == "content_block_start":
            index = event.get("index")
            block = event.get("content_block") or {}
            if isinstance(index, int) and isinstance(block, dict):
                self._blocks[index] = {
                    "type": block.get("type"),
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input_json": "",
                }
                return self._progress_events_for_block_start(block)
            return []
        if inner_type == "content_block_delta":
            index = event.get("index")
            delta = event.get("delta") or {}
            if not isinstance(index, int) or not isinstance(delta, dict):
                return []
            delta_type = delta.get("type")
            if delta_type in _CLAUDE_IGNORED_DELTA_TYPES:
                return []
            block = self._blocks.setdefault(
                index,
                {"type": None, "id": None, "name": None, "input_json": ""},
            )
            if delta_type == "text_delta":
                text = delta.get("text") or ""
                if text:
                    return list(self._tag_parser.feed(text))
                return []
            if delta_type == "input_json_delta":
                partial = delta.get("partial_json") or ""
                if partial:
                    block["input_json"] = (block.get("input_json") or "") + partial
                return []
            return []
        if inner_type == "content_block_stop":
            index = event.get("index")
            if not isinstance(index, int):
                return []
            block = self._blocks.get(index)
            if not block or block.get("type") != "tool_use":
                return []
            return self._emit_tool_use_from_block(block)
        return []

    def _handle_assistant_aggregate(self, message: dict[str, Any]) -> list[CliStreamEvent]:
        if self._saw_stream_events:
            return []
        msg_id = message.get("id")
        if isinstance(msg_id, str):
            if msg_id in self._seen_assistant_msg_ids:
                return []
            self._seen_assistant_msg_ids.add(msg_id)
        events: list[CliStreamEvent] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text") or ""
                if text:
                    events.extend(self._tag_parser.feed(text))
            elif block_type == "tool_use":
                normalized = self._normalize_tool_use(
                    tool_id=block.get("id"),
                    name=block.get("name"),
                    input_value=block.get("input"),
                )
                if normalized:
                    events.append(CliStreamEvent(kind="tool_call", tool_call=normalized))
        return events

    def _emit_tool_use_from_block(self, block: dict[str, Any]) -> list[CliStreamEvent]:
        raw = (block.get("input_json") or "").strip() or "{}"
        try:
            input_value = json.loads(raw)
        except Exception:
            input_value = {}
        normalized = self._normalize_tool_use(
            tool_id=block.get("id"),
            name=block.get("name"),
            input_value=input_value,
        )
        if normalized:
            return [CliStreamEvent(kind="tool_call", tool_call=normalized)]
        return []

    def _progress_events_for_block_start(self, block: dict[str, Any]) -> list[CliStreamEvent]:
        return []

    def _normalize_tool_use(
        self,
        *,
        tool_id: Any,
        name: Any,
        input_value: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(name, str) or not name.strip():
            return None
        self._tool_call_count += 1
        call_id = tool_id if (isinstance(tool_id, str) and tool_id.strip()) else f"call_{self._tool_call_count}"
        if input_value is None:
            arguments_str = "{}"
        elif isinstance(input_value, str):
            arguments_str = input_value
        else:
            arguments_str = json.dumps(input_value, ensure_ascii=False)
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name.strip(),
                "arguments": arguments_str,
            },
        }


def parse_claude_stream_metadata(text: str) -> ClaudeResultMetadata:
    if not isinstance(text, str) or not text.strip():
        return ClaudeResultMetadata()
    parser = ClaudeStreamJsonParser()
    parser.feed(text if text.endswith("\n") else text + "\n")
    parser.finalize()
    if not parser.saw_any_json():
        return ClaudeResultMetadata()
    return parser.result_metadata()


def parse_claude_stream_json(text: str) -> ParsedShimOutput:
    """Parse a complete Claude stream-json blob (one JSON object per line).

    Falls back to plain-text `<tool_call>`-tag parsing when the input contains
    no parseable stream-json lines. This preserves backward compatibility with
    the legacy text format and with mocks that supply raw strings.
    """
    if not isinstance(text, str) or not text.strip():
        return ParsedShimOutput(content="", tool_calls=[])
    parser = ClaudeStreamJsonParser()
    events = parser.feed(text if text.endswith("\n") else text + "\n")
    events.extend(parser.finalize())
    if not parser.saw_any_json():
        return parse_cli_output(text)
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for event in events:
        if event.kind == "text" and event.text:
            text_parts.append(event.text)
        elif event.kind == "tool_call" and event.tool_call:
            tool_calls.append(event.tool_call)
    content = "".join(text_parts).strip()
    cleaned, silent = _detect_silent(content, has_tool_calls=bool(tool_calls))
    return ParsedShimOutput(content="" if silent else cleaned, tool_calls=tool_calls, silent=silent)


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
        cleaned, silent = _detect_silent(text.strip(), has_tool_calls=False)
        return ParsedShimOutput(content="" if silent else cleaned, tool_calls=[], silent=silent)

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
    cleaned, silent = _detect_silent(cleaned, has_tool_calls=bool(tool_calls))
    return ParsedShimOutput(content="" if silent else cleaned, tool_calls=tool_calls, silent=silent)
