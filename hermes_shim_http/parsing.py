from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .models import CliStreamEvent, ParsedShimOutput
from .silence import detect_and_strip as _detect_silent
from .telemetry import emit_event

_JSON_DECODER = json.JSONDecoder()
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_MALFORMED_NOTICE = "⚠️ shim: dropped malformed tool_call (name={name}, reason={reason}) — see shim logs"
_MALFORMED_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_JSON_REPAIR_ENV = "HERMES_SHIM_JSON_REPAIR_ENABLED"
_RAW_LOG_ENV = "HERMES_SHIM_CLAUDE_RAW_LOG_DIR"
_DEFAULT_RAW_LOG_DIR = os.path.expanduser("~/.hermes/hermes-shim-http/raw-logs")
_RAW_LOG_MAX_FILES = 200
_RAW_LOG_MAX_BYTES = 500 * 1024 * 1024

_CLAUDE_IGNORED_DELTA_TYPES = frozenset({"thinking_delta", "signature_delta"})


try:
    from json_repair import loads as _json_repair_loads  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _json_repair_loads = None


def _json_repair_enabled() -> bool:
    return os.getenv(_JSON_REPAIR_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolved_raw_log_dir() -> str | None:
    env_value = os.environ.get(_RAW_LOG_ENV)
    if env_value is None:
        return _DEFAULT_RAW_LOG_DIR
    value = env_value.strip()
    if value == "":
        return None
    return os.path.expanduser(value)


def _rotate_raw_logs(log_dir: str) -> None:
    try:
        entries: list[tuple[float, int, str]] = []
        for name in os.listdir(log_dir):
            path = os.path.join(log_dir, name)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            entries.append((stat.st_mtime, stat.st_size, path))
        entries.sort(key=lambda item: item[0], reverse=True)

        keep: list[tuple[float, int, str]] = []
        total = 0
        for item in entries:
            if len(keep) >= _RAW_LOG_MAX_FILES:
                continue
            if total + item[1] > _RAW_LOG_MAX_BYTES:
                continue
            keep.append(item)
            total += item[1]

        keep_paths = {path for _, _, path in keep}
        for _, _, path in entries:
            if path in keep_paths:
                continue
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        return


def _dump_malformed_raw_block(raw_block: str) -> str | None:
    log_dir = _resolved_raw_log_dir()
    if not log_dir:
        return None
    try:
        os.makedirs(log_dir, exist_ok=True)
        _rotate_raw_logs(log_dir)
        ts = time.strftime("%Y%m%d-%H%M%S")
        unique = uuid.uuid4().hex[:8]
        filename = f"malformed-tool-call-{ts}-{os.getpid()}-{unique}.log"
        path = os.path.join(log_dir, filename)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(raw_block)
        return path
    except OSError:
        return None


def _best_effort_tool_name(raw_block: str) -> str:
    match = _MALFORMED_NAME_RE.search(raw_block)
    if not match:
        return "unknown"
    return match.group(1) or "unknown"


def _malformed_notice(name: str, reason: str) -> str:
    return _MALFORMED_NOTICE.format(name=name, reason=reason)


def _tool_call_inner_payload(raw_block: str) -> str:
    open_end = raw_block.find(">")
    close_start = raw_block.rfind(_TOOL_CALL_CLOSE)
    if open_end < 0 or close_start < 0 or close_start <= open_end:
        return raw_block.strip()
    return raw_block[open_end + 1 : close_start].strip()


def _should_suppress_malformed_telemetry(raw_block: str) -> bool:
    """Best-effort filter for obvious prompt-echo placeholders.

    We still reject execution, but avoid noisy malformed telemetry/raw-log churn
    for clearly non-protocol examples like `<tool_call ...>{...}</tool_call>`.
    """
    payload = _tool_call_inner_payload(raw_block)
    compact = "".join(payload.split())
    if compact in {"{...}", "{…}"}:
        return True
    if "..." in compact and all(key not in compact for key in ('"function"', '"name"', '"id"', '"type"')):
        return True
    return False


def _emit_malformed_event(*, raw_block: str, reason: str) -> str:
    if _should_suppress_malformed_telemetry(raw_block):
        return ""
    name = _best_effort_tool_name(raw_block)
    raw_file = _dump_malformed_raw_block(raw_block)
    emit_event(
        "tool_call_malformed",
        name=name,
        reason=reason,
        raw_size=len(raw_block),
        raw_sha256=sha256(raw_block.encode("utf-8", errors="ignore")).hexdigest(),
        full_raw_file=raw_file,
    )
    return _malformed_notice(name, reason)


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


def _tool_call_open_tag(expected_tool_call_nonce: str | None) -> str:
    if expected_tool_call_nonce:
        return f'<tool_call nonce="{expected_tool_call_nonce}">'
    return _TOOL_CALL_OPEN


def _try_extract_tool_call(buffer: str, open_idx: int, *, tool_call_open_tag: str = _TOOL_CALL_OPEN) -> tuple[int, dict[str, Any]] | None:
    json_start = open_idx + len(tool_call_open_tag)
    while json_start < len(buffer) and buffer[json_start].isspace():
        json_start += 1
    if json_start >= len(buffer) or buffer[json_start] != "{":
        return None
    try:
        obj, end_idx = _JSON_DECODER.raw_decode(buffer, json_start)
    except json.JSONDecodeError:
        return None
    close_scan = end_idx
    while close_scan < len(buffer) and buffer[close_scan].isspace():
        close_scan += 1
    if not buffer.startswith(_TOOL_CALL_CLOSE, close_scan):
        return None
    return close_scan + len(_TOOL_CALL_CLOSE), obj


def _find_tool_call_block_end(buffer: str, open_idx: int, *, tool_call_open_tag: str = _TOOL_CALL_OPEN) -> int | None:
    close_idx = buffer.find(_TOOL_CALL_CLOSE, open_idx + len(tool_call_open_tag))
    if close_idx < 0:
        return None
    return close_idx + len(_TOOL_CALL_CLOSE)


def _try_repair_tool_call(raw_block: str, *, tool_call_open_tag: str = _TOOL_CALL_OPEN) -> dict[str, Any] | None:
    if not _json_repair_enabled() or _json_repair_loads is None:
        return None
    json_part = raw_block[len(tool_call_open_tag) :]
    close_idx = json_part.rfind(_TOOL_CALL_CLOSE)
    if close_idx < 0:
        return None
    json_part = json_part[:close_idx].strip()
    if not json_part.startswith("{"):
        return None
    try:
        repaired = _json_repair_loads(json_part)
    except Exception:
        return None
    return repaired if isinstance(repaired, dict) else None


class IncrementalToolCallParser:
    def __init__(self, *, expected_tool_call_nonce: str | None = None) -> None:
        self._buffer = ""
        self._tool_call_count = 0
        self._tool_call_open_tag = _tool_call_open_tag(expected_tool_call_nonce)

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
            open_idx = self._buffer.find(self._tool_call_open_tag)
            if open_idx < 0:
                break
            extracted = _try_extract_tool_call(self._buffer, open_idx, tool_call_open_tag=self._tool_call_open_tag)

            prefix = self._buffer[:open_idx]
            if prefix:
                events.append(CliStreamEvent(kind="text", text=prefix))

            if extracted is not None:
                block_end, obj = extracted
                normalized = _normalize_tool_call(obj, self._tool_call_count + 1)
                if normalized is None:
                    notice = _emit_malformed_event(
                        raw_block=self._buffer[open_idx:block_end],
                        reason="normalize_rejected",
                    )
                    if notice:
                        events.append(CliStreamEvent(kind="text", text=notice))
                else:
                    self._tool_call_count += 1
                    events.append(CliStreamEvent(kind="tool_call", tool_call=normalized))
                self._buffer = self._buffer[block_end:]
                continue

            malformed_end = _find_tool_call_block_end(self._buffer, open_idx, tool_call_open_tag=self._tool_call_open_tag)
            if malformed_end is None:
                # Wait for more bytes unless we're finalizing.
                if final:
                    events.append(CliStreamEvent(kind="text", text=self._buffer[open_idx:]))
                    self._buffer = ""
                else:
                    self._buffer = self._buffer[open_idx:]
                break

            raw_block = self._buffer[open_idx:malformed_end]
            repaired = _try_repair_tool_call(raw_block, tool_call_open_tag=self._tool_call_open_tag)
            if repaired is not None:
                normalized = _normalize_tool_call(repaired, self._tool_call_count + 1)
                if normalized is not None:
                    self._tool_call_count += 1
                    events.append(CliStreamEvent(kind="tool_call", tool_call=normalized))
                    notice = _emit_malformed_event(raw_block=raw_block, reason="repaired_from_malformed")
                    if notice:
                        events.append(CliStreamEvent(kind="text", text=notice))
                else:
                    notice = _emit_malformed_event(raw_block=raw_block, reason="normalize_rejected")
                    if notice:
                        events.append(CliStreamEvent(kind="text", text=notice))
            else:
                notice = _emit_malformed_event(raw_block=raw_block, reason="json_decode_error")
                if notice:
                    events.append(CliStreamEvent(kind="text", text=notice))

            self._buffer = self._buffer[malformed_end:]

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

    def _safe_prefix_length(self, buffer: str) -> int:
        if not buffer:
            return 0

        open_index = buffer.find(self._tool_call_open_tag)
        close_index = buffer.find(_TOOL_CALL_CLOSE)
        candidate_indexes = [idx for idx in (open_index, close_index) if idx >= 0]
        if candidate_indexes:
            return min(candidate_indexes)

        tail_window = max(len(self._tool_call_open_tag), len(_TOOL_CALL_CLOSE)) - 1
        tail_start = max(0, len(buffer) - tail_window)
        for start in range(tail_start, len(buffer)):
            suffix = buffer[start:]
            if self._tool_call_open_tag.startswith(suffix) or _TOOL_CALL_CLOSE.startswith(suffix):
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

    def __init__(self, *, synthesize_progress: bool = False, expected_tool_call_nonce: str | None = None) -> None:
        self._line_buffer = ""
        self._blocks: dict[int, dict[str, Any]] = {}
        self._tool_call_count = 0
        self._seen_assistant_msg_ids: set[str] = set()
        self._saw_stream_events = False
        self._saw_any_json = False
        self._tag_parser = IncrementalToolCallParser(expected_tool_call_nonce=expected_tool_call_nonce)
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


def parse_claude_stream_json(text: str, *, expected_tool_call_nonce: str | None = None) -> ParsedShimOutput:
    """Parse a complete Claude stream-json blob (one JSON object per line).

    Falls back to plain-text `<tool_call>`-tag parsing when the input contains
    no parseable stream-json lines. This preserves backward compatibility with
    the legacy text format and with mocks that supply raw strings.
    """
    if not isinstance(text, str) or not text.strip():
        return ParsedShimOutput(content="", tool_calls=[])
    parser = ClaudeStreamJsonParser(expected_tool_call_nonce=expected_tool_call_nonce)
    events = parser.feed(text if text.endswith("\n") else text + "\n")
    events.extend(parser.finalize())
    if not parser.saw_any_json():
        return parse_cli_output(text, expected_tool_call_nonce=expected_tool_call_nonce)
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


def parse_cli_output(text: str, *, expected_tool_call_nonce: str | None = None) -> ParsedShimOutput:
    if not isinstance(text, str) or not text.strip():
        return ParsedShimOutput(content="", tool_calls=[])

    tool_calls: list[dict[str, Any]] = []
    parts: list[str] = []
    cursor = 0
    tool_call_open_tag = _tool_call_open_tag(expected_tool_call_nonce)

    while cursor < len(text):
        open_idx = text.find(tool_call_open_tag, cursor)
        if open_idx < 0:
            parts.append(text[cursor:])
            break

        if cursor < open_idx:
            parts.append(text[cursor:open_idx])

        extracted = _try_extract_tool_call(text, open_idx, tool_call_open_tag=tool_call_open_tag)
        if extracted is not None:
            block_end, obj = extracted
            normalized = _normalize_tool_call(obj, len(tool_calls) + 1)
            if normalized is None:
                notice = _emit_malformed_event(raw_block=text[open_idx:block_end], reason="normalize_rejected")
                if notice:
                    parts.append(notice)
            else:
                tool_calls.append(normalized)
            cursor = block_end
            continue

        malformed_end = _find_tool_call_block_end(text, open_idx, tool_call_open_tag=tool_call_open_tag)
        if malformed_end is None:
            parts.append(text[open_idx:])
            cursor = len(text)
            break

        raw_block = text[open_idx:malformed_end]
        repaired = _try_repair_tool_call(raw_block, tool_call_open_tag=tool_call_open_tag)
        if repaired is not None:
            normalized = _normalize_tool_call(repaired, len(tool_calls) + 1)
            if normalized is not None:
                tool_calls.append(normalized)
                notice = _emit_malformed_event(raw_block=raw_block, reason="repaired_from_malformed")
                if notice:
                    parts.append(notice)
            else:
                notice = _emit_malformed_event(raw_block=raw_block, reason="normalize_rejected")
                if notice:
                    parts.append(notice)
        else:
            notice = _emit_malformed_event(raw_block=raw_block, reason="json_decode_error")
            if notice:
                parts.append(notice)

        cursor = malformed_end

    cleaned = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    cleaned, silent = _detect_silent(cleaned, has_tool_calls=bool(tool_calls))
    return ParsedShimOutput(content="" if silent else cleaned, tool_calls=tool_calls, silent=silent)
