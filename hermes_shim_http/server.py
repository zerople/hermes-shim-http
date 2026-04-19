from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from hashlib import sha256
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .logging_utils import configure_logger, emit_log
from .inflight import InFlightRegistry
from .single_child import ChildLockBusy
from .models import ChatCompletionsRequest, CliStreamEvent, ParsedShimOutput, ShimConfig, ToolDefinition
from .parsing import IncrementalToolCallParser
from .prompting import build_cli_prompt, compact_messages
from .runner import parse_cli_result, resolved_cli_args, run_cli_prompt, stream_cli_prompt, supports_cli_resume
from .session_cache import SessionCache, SessionPlan
from .silence import detect_and_strip as _detect_silent, silent_sentinel
from .slash_commands import dispatch_slash_command
from .telemetry import emit_event
from .token_usage import TokenUsageEstimate, context_limit_for_model, estimate_token_usage

SILENT_HEADER = "X-Shim-Silent"

KEEPALIVE_INTERVAL_SECONDS = 10.0

_HTTP_HEARTBEAT_BYTE = b" "


def _streaming_json_with_heartbeat(
    run_fn: Callable[[], dict[str, Any]],
    *,
    interval: float,
    on_error: Callable[[BaseException], dict[str, Any]],
) -> Iterator[bytes]:
    """Run ``run_fn`` on a background thread, emit whitespace every ``interval``s
    while it runs, then yield the final JSON body.

    JSON (RFC 8259) allows arbitrary leading whitespace, so callers parsing the
    concatenated stream as JSON still get a valid document. The heartbeat bytes
    keep Hermes→shim HTTP connections alive through long CLI runs.

    If ``run_fn`` raises, ``on_error`` builds an OpenAI-style error body. The
    HTTP status is 200 (already committed before we knew the outcome); callers
    must inspect the ``error`` field.
    """
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            outcome["body"] = run_fn()
        except BaseException as exc:  # noqa: BLE001 — surface via JSON body
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()
    while not done.wait(interval):
        yield _HTTP_HEARTBEAT_BYTE

    if "error" in outcome:
        body = on_error(outcome["error"])
    else:
        body = outcome["body"]
    yield json.dumps(body).encode("utf-8")


def _is_silent_candidate(accumulated: str, sentinel: str) -> bool:
    """True while the streamed-so-far text could still collapse to a silent ACK.

    Streaming paths must not flush the literal sentinel as a content delta —
    once the client sees it, the turn can't be collapsed to an empty ACK.
    Buffer while the output is either (a) an incomplete prefix of the
    sentinel, or (b) the sentinel itself possibly padded with whitespace.
    """
    stripped = accumulated.strip()
    if not stripped:
        return True
    if sentinel.startswith(stripped):
        return True
    _, is_silent = _detect_silent(stripped, has_tool_calls=False)
    return is_silent


def _usage_dict(estimate: TokenUsageEstimate) -> dict[str, int]:
    prompt_tokens = estimate.context_tokens_used
    completion_tokens = estimate.response_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "context_tokens_used": estimate.context_tokens_used,
        "context_tokens_limit": estimate.context_tokens_limit,
        "response_tokens": estimate.response_tokens,
    }


def _chat_response(*, model: str, parsed: ParsedShimOutput, usage: TokenUsageEstimate) -> dict[str, Any]:
    finish_reason = "tool_calls" if parsed.tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    choice: dict[str, Any] = {"index": 0, "message": message, "finish_reason": finish_reason}
    if parsed.silent:
        choice["silent"] = True
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": _usage_dict(usage),
    }


def _silent_headers(parsed: ParsedShimOutput, headers: dict[str, str]) -> dict[str, str]:
    if parsed.silent:
        return {**headers, SILENT_HEADER: "true"}
    return headers


def _idempotency_key(http_request: Request) -> str:
    raw = (
        http_request.headers.get("Idempotency-Key")
        or http_request.headers.get("X-Idempotency-Key")
        or http_request.headers.get("X-Request-Id")
        or ""
    )
    return raw.strip()


def _release_on_exit(gen: Iterator[bytes], registry: InFlightRegistry, key: str) -> Iterator[bytes]:
    try:
        yield from gen
    finally:
        registry.release(key)


def _duplicate_request_response(key: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "message": "duplicate in-flight request for idempotency key",
                "type": "duplicate_request",
                "idempotency_key": key,
            }
        },
        headers={"X-Request-Id": request_id, "X-Idempotency-Key": key, "Retry-After": "5"},
    )


def _assistant_messages_from_parsed(parsed: ParsedShimOutput) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": parsed.content, **({"tool_calls": parsed.tool_calls} if parsed.tool_calls else {})}]


def _normalize_chat_tools(raw_tools: Any, *, strict: bool = False) -> list[dict[str, Any]] | None:
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list):
        if strict:
            raise HTTPException(status_code=400, detail="'tools' must be an array")
        return None
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            if strict:
                raise HTTPException(status_code=400, detail=f"tools[{idx}] must be an object")
            continue
        if isinstance(item.get("function"), dict):
            try:
                normalized.append(ToolDefinition.model_validate(item).model_dump())
            except ValidationError as exc:
                if strict:
                    raise HTTPException(status_code=400, detail=f"tools[{idx}] is not a valid function tool definition: {exc.errors()[0]['msg']}") from exc
            continue
        if item.get("type") == "function" and isinstance(item.get("name"), str):
            normalized.append(
                ToolDefinition.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "description": item.get("description", ""),
                            "parameters": item.get("parameters", {}),
                        },
                    }
                ).model_dump()
            )
            continue
        if strict:
            raise HTTPException(status_code=400, detail=f"tools[{idx}] is not a valid function tool definition")
    return normalized or None


def _allowed_tool_names_from_tools(raw_tools: Any, *, reject_if_missing: bool = False, strict: bool = False) -> set[str] | None:
    normalized = _normalize_chat_tools(raw_tools, strict=strict)
    if not normalized:
        return set() if reject_if_missing else None
    return {tool["function"]["name"] for tool in normalized if tool.get("function", {}).get("name")}


def _unsupported_tool_message(tool_names: list[str]) -> str:
    return f"Wrapped CLI emitted unsupported tool call(s): {', '.join(tool_names)}"


def _sanitize_parsed_output(parsed: ParsedShimOutput, allowed_tool_names: set[str] | None) -> ParsedShimOutput:
    if allowed_tool_names is None:
        return parsed

    allowed_calls: list[dict[str, Any]] = []
    unsupported_names: list[str] = []
    for tool_call in parsed.tool_calls:
        name = str(tool_call.get("function", {}).get("name") or "").strip()
        if name and name in allowed_tool_names:
            allowed_calls.append(tool_call)
        elif name:
            unsupported_names.append(name)

    if not unsupported_names:
        return ParsedShimOutput(content=parsed.content, tool_calls=allowed_calls, silent=parsed.silent)

    content = parsed.content.strip()
    if not content:
        content = _unsupported_tool_message(unsupported_names)
    return ParsedShimOutput(content=content, tool_calls=allowed_calls, silent=parsed.silent and not content)


def _sse_line(payload: dict[str, Any] | str) -> bytes:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n".encode("utf-8")


def _message_content_len(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    total += len(text)
                else:
                    total += len(json.dumps(item, ensure_ascii=False))
            else:
                total += len(str(item))
        return total
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return len(text)
        return len(json.dumps(content, ensure_ascii=False))
    return len(str(content))


def _last_user_message_len(messages: list[dict[str, Any]]) -> int:
    for message in reversed(messages):
        if str(message.get("role") or "").strip().lower() == "user":
            return _message_content_len(message.get("content"))
    return 0


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _safe_json_size_bytes(payload: Any) -> int | None:
    try:
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return None


def _log_chat_request_summary(*, request: ChatCompletionsRequest, request_messages: list[dict[str, Any]], request_tools: list[dict[str, Any]] | None, session_plan: SessionPlan) -> None:
    emit_event(
        "chat_completions_request",
        model=request.model,
        stream=bool(request.stream),
        message_count=len(request_messages),
        message_roles=[str(message.get("role") or "") for message in request_messages],
        last_user_message_len=_last_user_message_len(request_messages),
        tool_count=len(request_tools or []),
        tool_names=_tool_names(request_tools)[:20],
        tool_choice_present=request.tool_choice is not None,
        request_json_bytes=_safe_json_size_bytes(
            {
                "model": request.model,
                "stream": request.stream,
                "messages": request_messages,
                "tools": request_tools,
                "tool_choice": request.tool_choice,
            }
        ),
        system_prompt_len=len(session_plan.system_prompt_text or ""),
        cli_prompt_len=len(session_plan.prompt_text or ""),
        resume_used=session_plan.resume_session_id is not None,
        prefix_message_count=session_plan.prefix_message_count,
    )


def _log_responses_request_summary(*, body: dict[str, Any], messages: list[dict[str, Any]], normalized_tools: list[dict[str, Any]] | None, prompt: str) -> None:
    emit_event(
        "responses_request",
        model=str(body.get("model") or ""),
        stream=bool(body.get("stream")),
        message_count=len(messages),
        message_roles=[str(message.get("role") or "") for message in messages],
        last_user_message_len=_last_user_message_len(messages),
        tool_count=len(normalized_tools or []),
        tool_names=_tool_names(normalized_tools)[:20],
        tool_choice_present=body.get("tool_choice") is not None,
        request_json_bytes=_safe_json_size_bytes(body),
        cli_prompt_len=len(prompt or ""),
    )


def _stream_chunk_for_text(*, completion_id: str, created: int, model: str, text: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def _stream_chunk_for_tool_call(*, completion_id: str, created: int, model: str, tool_call: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def _flush_pending_chat_text(*, completion_id: str, created: int, model: str, pending_text: str) -> Iterator[bytes]:
    if not pending_text:
        return
    yield _sse_line(_stream_chunk_for_text(completion_id=completion_id, created=created, model=model, text=pending_text))


def _iter_events_with_keepalive(source: Iterator[CliStreamEvent], *, keepalive_interval: float = KEEPALIVE_INTERVAL_SECONDS) -> Iterator[CliStreamEvent | None]:
    sentinel = object()
    sink: queue.Queue[Any] = queue.Queue()

    def _worker() -> None:
        try:
            for item in source:
                sink.put(item)
            sink.put(sentinel)
        except BaseException as exc:  # pragma: no cover
            sink.put(exc)
            sink.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    done = False
    try:
        while not done:
            try:
                item = sink.get(timeout=keepalive_interval)
            except queue.Empty:
                yield None
                continue
            if item is sentinel:
                done = True
                continue
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not done:
            try:
                if hasattr(source, "close"):
                    source.close()
            except Exception:
                pass
            thread.join(timeout=5.0)


def _record_metrics(app: FastAPI, *, latency_ms: int, cache_hit: bool, usage: TokenUsageEstimate, compacted: bool) -> None:
    metrics = app.state.metrics
    metrics["request_count"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["cache_hits"] += 1 if cache_hit else 0
    metrics["cache_misses"] += 0 if cache_hit else 1
    metrics["token_context_total"] += usage.context_tokens_used
    metrics["token_context_max"] = max(metrics["token_context_max"], usage.context_tokens_used)
    metrics["token_response_total"] += usage.response_tokens
    metrics["compactions"] += 1 if compacted else 0


def _estimate_usage_for_stream(model: str, *, config: ShimConfig, messages: list[dict[str, Any]], response_text: str) -> TokenUsageEstimate:
    return estimate_token_usage(
        messages=messages,
        response_text=response_text,
        context_limit=context_limit_for_model(model, profile=config.cli_profile if config.cli_profile != "auto" else None),
    )


def _debug_stats_payload(app: FastAPI) -> dict[str, Any]:
    cache_stats = app.state.session_cache.stats()
    metrics = app.state.metrics
    request_count = metrics["request_count"]
    total_cache_decisions = metrics["cache_hits"] + metrics["cache_misses"]
    return {
        "cache_size": cache_stats["cache_size"],
        "hit_rate": (metrics["cache_hits"] / total_cache_decisions) if total_cache_decisions else 0.0,
        "avg_latency_ms": (metrics["total_latency_ms"] / request_count) if request_count else 0.0,
        "active_sessions": cache_stats["active_sessions"],
        "uptime_s": max(0.0, time.time() - app.state.started_at),
        "avg_context_tokens_used": (metrics["token_context_total"] / request_count) if request_count else 0.0,
        "max_context_tokens_used": metrics["token_context_max"],
        "avg_response_tokens": (metrics["token_response_total"] / request_count) if request_count else 0.0,
        "compactions": metrics["compactions"],
        "hit_count": cache_stats["hit_count"],
    }


def _matching_prefix_length(prefix_messages: list[dict[str, Any]], messages: list[dict[str, Any]]) -> int:
    if len(prefix_messages) > len(messages):
        return 0
    for index, prefix_message in enumerate(prefix_messages):
        if messages[index] != prefix_message:
            return 0
    return len(prefix_messages)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _messages_without_last_user_command(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = list(messages)
    for index in range(len(trimmed) - 1, -1, -1):
        message = trimmed[index]
        if str(message.get("role") or "").lower() == "user" and isinstance(message.get("content"), str):
            if message["content"].strip().startswith("/"):
                return trimmed[:index]
            break
    return trimmed


def _pending_compaction_key(*, model: str, base_messages: list[dict[str, Any]]) -> str:
    payload = {"model": model, "messages": base_messages}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _remember_pending_compaction(app: FastAPI, *, model: str, base_messages: list[dict[str, Any]], compacted_messages: list[dict[str, Any]]) -> str:
    pending = getattr(app.state, "pending_compactions", {})
    token = uuid.uuid4().hex
    pending[token] = {
        "key": _pending_compaction_key(model=model, base_messages=base_messages),
        "model": model,
        "base_messages": base_messages,
        "compacted_messages": compacted_messages,
        "created_at": time.time(),
    }
    if len(pending) > 128:
        oldest_key = min(pending.items(), key=lambda item: item[1].get("created_at", 0))[0]
        pending.pop(oldest_key, None)
    return token


def _apply_slash_compaction(*, messages: list[dict[str, Any]], config: ShimConfig, model: str) -> tuple[list[dict[str, Any]], bool]:
    base_messages = _messages_without_last_user_command(messages)
    effective_mode = config.compaction if config.compaction != "off" else "summarize"
    return compact_messages(
        messages=base_messages,
        mode=effective_mode,
        threshold=config.compaction_threshold,
        context_limit=context_limit_for_model(model, profile=config.cli_profile if config.cli_profile != "auto" else None),
        force=True,
    )


def _consume_pending_compaction(app: FastAPI, *, token: str | None, messages: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], bool]:
    if not token:
        return messages, False
    pending = getattr(app.state, "pending_compactions", {})
    stale_keys = [key for key, entry in pending.items() if time.time() - entry.get("created_at", 0) > 600]
    for key in stale_keys:
        pending.pop(key, None)
    entry = pending.pop(token, None)
    if not entry or entry.get("model") != model:
        return messages, False
    base_messages = entry.get("base_messages") or []
    prefix_len = _matching_prefix_length(base_messages, messages)
    if prefix_len != len(base_messages):
        return messages, False
    compacted_messages = entry.get("compacted_messages") or []
    return [*compacted_messages, *messages[prefix_len:]], True


def _maybe_apply_compaction(*, messages: list[dict[str, Any]], config: ShimConfig, model: str, force: bool = False) -> tuple[list[dict[str, Any]], bool]:
    return compact_messages(
        messages=messages,
        mode=config.compaction,
        threshold=config.compaction_threshold,
        context_limit=context_limit_for_model(model, profile=config.cli_profile if config.cli_profile != "auto" else None),
        force=force,
    )


def _ordered_cli_events_from_text(text: str, allowed_tool_names: set[str] | None) -> list[CliStreamEvent]:
    parser = IncrementalToolCallParser()
    events = [*parser.feed(text or ""), *parser.finalize()]
    sanitized: list[CliStreamEvent] = []
    for event in events:
        if event.kind == "tool_call" and event.tool_call:
            name = str(event.tool_call.get("function", {}).get("name") or "").strip()
            if allowed_tool_names is not None and name not in allowed_tool_names:
                sanitized.append(CliStreamEvent(kind="text", text=_unsupported_tool_message([name])))
            else:
                sanitized.append(event)
        else:
            sanitized.append(event)
    return sanitized


def _responses_output_items_from_events(events: list[CliStreamEvent]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for event in events:
        if event.kind == "tool_call" and event.tool_call:
            items.append(
                {
                    "type": "function_call",
                    "call_id": event.tool_call["id"],
                    "name": event.tool_call["function"]["name"],
                    "arguments": event.tool_call["function"]["arguments"],
                }
            )
        elif event.kind == "text" and event.text:
            items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": event.text}]})
    if not items:
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]})
    return items


def _parsed_output_from_events(events: list[CliStreamEvent]) -> ParsedShimOutput:
    tool_calls = [event.tool_call for event in events if event.kind == "tool_call" and event.tool_call]
    content = "".join(event.text or "" for event in events if event.kind == "text").strip()
    cleaned, silent = _detect_silent(content, has_tool_calls=bool(tool_calls))
    return ParsedShimOutput(
        content="" if silent else cleaned,
        tool_calls=tool_calls,
        silent=silent,
    )


def _responses_json_response(*, model: str, parsed: ParsedShimOutput, usage: TokenUsageEstimate, output_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"resp_{uuid.uuid4().hex[:28]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output_items or _responses_output_items_from_events([CliStreamEvent(kind="text", text=parsed.content)] if parsed.content and not parsed.tool_calls else []),
        "usage": {
            "input_tokens": usage.context_tokens_used,
            "output_tokens": usage.response_tokens,
            "total_tokens": usage.context_tokens_used + usage.response_tokens,
            "context_tokens_used": usage.context_tokens_used,
            "context_tokens_limit": usage.context_tokens_limit,
            "response_tokens": usage.response_tokens,
        },
    }
    if parsed.silent:
        payload["silent"] = True
    return payload


def _normalize_responses_input(raw_input: Any) -> list[dict[str, Any]]:
    if raw_input is None:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if not isinstance(raw_input, list):
        raise HTTPException(status_code=400, detail="'input' must be a string or array")

    messages: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_input):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"input[{idx}] must be an object or string")
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{idx}")
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(item.get("name") or "").strip(),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
            messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>"})
            continue
        if item_type == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": str(item.get("call_id") or "").strip() or None, "content": str(item.get("output") or "")})
            continue
        role = str(item.get("role") or "user").strip().lower()
        content = item.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype in {"text", "input_text", "output_text"} and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


def _responses_prompt_from_body(body: dict[str, Any], *, config: ShimConfig, force_compact: bool = False) -> tuple[str, set[str], list[dict[str, Any]] | None, list[dict[str, Any]], bool]:
    tools_present = "tools" in body
    normalized_tools = _normalize_chat_tools(body.get("tools"), strict=tools_present)
    messages = _normalize_responses_input(body.get("input"))
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages = [{"role": "system", "content": instructions.strip()}, *messages]
    compacted_messages, compacted = _maybe_apply_compaction(messages=messages, config=config, model=str(body.get("model") or "cli-http-shim"), force=force_compact)
    prompt = build_cli_prompt(messages=compacted_messages, model=str(body.get("model") or "cli-http-shim"), tools=normalized_tools, tool_choice=body.get("tool_choice"))
    allowed_tool_names = _allowed_tool_names_from_tools(body.get("tools"), reject_if_missing=True, strict=tools_present)
    return prompt, allowed_tool_names or set(), normalized_tools, compacted_messages, compacted


def _stream_live_chat_chunks(*, app: FastAPI, request_id: str, logger: logging.Logger, model: str, request_messages: list[dict[str, Any]], prompt: str, config: ShimConfig, allowed_tool_names: set[str] | None, session_plan: SessionPlan, session_cache: SessionCache, compacted: bool) -> Iterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    saw_tool_calls = False
    tool_index = 0
    pending_text = ""
    completed_tool_calls: list[dict[str, Any]] = []
    assistant_text_chunks: list[str] = []
    sentinel = silent_sentinel()
    silent_window_closed = False
    started = time.time()

    emit_log(logger, event="stream_start", request_id=request_id, model=model)
    yield _sse_line({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    source = stream_cli_prompt(
        prompt,
        config,
        session_id=session_plan.session_id,
        resume_session_id=session_plan.resume_session_id,
        system_prompt=session_plan.system_prompt_text,
        model=model,
    )
    recorded = False
    try:
        for event in _iter_events_with_keepalive(source, keepalive_interval=KEEPALIVE_INTERVAL_SECONDS):
            if event is None:
                yield b": ping\n\n"
                continue
            if event.kind == "tool_call" and event.tool_call:
                name = str(event.tool_call.get("function", {}).get("name") or "").strip()
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    fallback_text = _unsupported_tool_message([name])
                    pending_text += fallback_text
                    assistant_text_chunks.append(fallback_text)
                    continue
                yield from _flush_pending_chat_text(completion_id=completion_id, created=created, model=model, pending_text=pending_text)
                pending_text = ""
                saw_tool_calls = True
                completed_tool_calls.append(event.tool_call)
                yield _sse_line(_stream_chunk_for_tool_call(completion_id=completion_id, created=created, model=model, tool_call=event.tool_call, index=tool_index))
                tool_index += 1
            elif event.kind == "text" and event.text:
                pending_text += event.text
                assistant_text_chunks.append(event.text)
                if not silent_window_closed and not _is_silent_candidate("".join(assistant_text_chunks), sentinel):
                    silent_window_closed = True
                if silent_window_closed and (
                    "\n" in pending_text
                    or len(pending_text) >= 24
                    or (event.text[-1:].isspace() and len(pending_text) >= 8)
                ):
                    yield from _flush_pending_chat_text(completion_id=completion_id, created=created, model=model, pending_text=pending_text)
                    pending_text = ""

        accumulated_text = "".join(assistant_text_chunks)
        cleaned_text, silent = _detect_silent(accumulated_text.strip(), has_tool_calls=bool(completed_tool_calls))
        if not silent:
            yield from _flush_pending_chat_text(completion_id=completion_id, created=created, model=model, pending_text=pending_text)
        pending_text = ""
        parsed = ParsedShimOutput(content="" if silent else cleaned_text, tool_calls=completed_tool_calls, silent=silent)
        session_cache.record_success(session_plan, assistant_messages=_assistant_messages_from_parsed(parsed))
        recorded = True
        usage = _estimate_usage_for_stream(model, config=config, messages=request_messages, response_text=parsed.content)
        _record_metrics(app, latency_ms=int((time.time() - started) * 1000), cache_hit=session_plan.resume_session_id is not None, usage=usage, compacted=compacted)
        final_choice: dict[str, Any] = {"index": 0, "delta": {}, "finish_reason": "tool_calls" if saw_tool_calls else "stop"}
        if silent:
            final_choice["silent"] = True
            emit_log(logger, event="silent", request_id=request_id, model=model)
        yield _sse_line({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [final_choice], "usage": _usage_dict(usage)})
        yield _sse_line("[DONE]")
        emit_log(logger, event="stream_end", request_id=request_id, model=model)
    except Exception as exc:
        emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
        raise
    finally:
        try:
            source.close()
        except Exception:
            pass
        if not recorded:
            try:
                session_cache.release_plan(session_plan)
            except Exception:
                pass
            emit_log(logger, event="stream_aborted", request_id=request_id, model=model)


def _stream_live_responses_events(*, app: FastAPI, request_id: str, logger: logging.Logger, model: str, prompt: str, request_messages: list[dict[str, Any]], config: ShimConfig, allowed_tool_names: set[str] | None, compacted: bool) -> Iterator[bytes]:
    response_id = f"resp_{uuid.uuid4().hex[:28]}"
    created_at = int(time.time())
    pending_text = ""
    pending_text_item_id: str | None = None
    pending_text_output_index: int | None = None
    output_index = 0
    output_items: list[dict[str, Any]] = []
    sentinel = silent_sentinel()
    silent_window_closed = False
    buffered_text = ""
    accumulated_text_for_silence = ""
    started = time.time()

    response_stub = {"id": response_id, "object": "response", "created_at": created_at, "status": "in_progress", "model": model, "output": []}
    emit_log(logger, event="stream_start", request_id=request_id, model=model)
    yield _sse_line({"type": "response.created", "response": response_stub})

    def flush_text() -> Iterator[bytes]:
        nonlocal pending_text, pending_text_item_id, pending_text_output_index, output_index
        if pending_text_output_index is None or pending_text_item_id is None:
            pending_text = ""
            return
        message_item = {"type": "message", "id": pending_text_item_id, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": pending_text}]}
        output_items.append(message_item)
        yield _sse_line({"type": "response.output_item.done", "output_index": pending_text_output_index, "item": message_item})
        pending_text = ""
        pending_text_item_id = None
        pending_text_output_index = None
        output_index += 1

    def ensure_text_item_started() -> Iterator[bytes]:
        nonlocal pending_text_item_id, pending_text_output_index
        if pending_text_item_id is not None and pending_text_output_index is not None:
            return
        pending_text_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        pending_text_output_index = output_index
        yield _sse_line({"type": "response.output_item.added", "output_index": pending_text_output_index, "item": {"type": "message", "id": pending_text_item_id, "role": "assistant", "status": "in_progress", "content": []}})

    source = stream_cli_prompt(prompt, config, model=model)
    try:
        for event in _iter_events_with_keepalive(source, keepalive_interval=KEEPALIVE_INTERVAL_SECONDS):
            if event is None:
                yield b": ping\n\n"
                continue
            if event.kind == "tool_call" and event.tool_call:
                # Tool call definitively rules out silent ACK; drain any buffered text.
                if not silent_window_closed:
                    silent_window_closed = True
                    if buffered_text:
                        pending_text += buffered_text
                        yield from ensure_text_item_started()
                        yield _sse_line({"type": "response.output_text.delta", "delta": buffered_text, "output_index": pending_text_output_index, "content_index": 0})
                        buffered_text = ""
                name = str(event.tool_call.get("function", {}).get("name") or "").strip()
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    fallback_text = _unsupported_tool_message([name])
                    pending_text += fallback_text
                    accumulated_text_for_silence += fallback_text
                    yield from ensure_text_item_started()
                    yield _sse_line({"type": "response.output_text.delta", "delta": fallback_text, "output_index": pending_text_output_index, "content_index": 0})
                    continue
                yield from flush_text()
                item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}", "call_id": event.tool_call["id"], "name": event.tool_call["function"]["name"], "arguments": event.tool_call["function"]["arguments"]}
                output_items.append(item)
                yield _sse_line({"type": "response.output_item.added", "output_index": output_index, "item": item})
                yield _sse_line({"type": "response.output_item.done", "output_index": output_index, "item": item})
                output_index += 1
            elif event.kind == "text" and event.text:
                accumulated_text_for_silence += event.text
                if not silent_window_closed:
                    if _is_silent_candidate(accumulated_text_for_silence, sentinel):
                        buffered_text += event.text
                        continue
                    silent_window_closed = True
                    if buffered_text:
                        pending_text += buffered_text
                        yield from ensure_text_item_started()
                        yield _sse_line({"type": "response.output_text.delta", "delta": buffered_text, "output_index": pending_text_output_index, "content_index": 0})
                        buffered_text = ""
                pending_text += event.text
                yield from ensure_text_item_started()
                yield _sse_line({"type": "response.output_text.delta", "delta": event.text, "output_index": pending_text_output_index, "content_index": 0})

        has_tool_calls = any(item.get("type") == "function_call" for item in output_items)
        cleaned_text, silent = _detect_silent(accumulated_text_for_silence.strip(), has_tool_calls=has_tool_calls)
        if silent:
            # Silent ACK — never emit the buffered sentinel as a message item.
            buffered_text = ""
        else:
            if buffered_text:
                pending_text += buffered_text
                yield from ensure_text_item_started()
                yield _sse_line({"type": "response.output_text.delta", "delta": buffered_text, "output_index": pending_text_output_index, "content_index": 0})
                buffered_text = ""
            yield from flush_text()
        response_text = "".join(
            content["text"]
            for item in output_items
            if item.get("type") == "message"
            for content in item.get("content", [])
            if isinstance(content, dict) and isinstance(content.get("text"), str)
        )
        # Re-derive silent after possibly flushing — response_text is the canonical source.
        _, silent = _detect_silent(response_text.strip() or accumulated_text_for_silence.strip(), has_tool_calls=has_tool_calls)
        usage = _estimate_usage_for_stream(model, config=config, messages=request_messages, response_text="" if silent else cleaned_text)
        _record_metrics(app, latency_ms=int((time.time() - started) * 1000), cache_hit=False, usage=usage, compacted=compacted)
        completed_output_items = [item for item in output_items if item.get("type") != "message"] if silent else output_items
        completed_response: dict[str, Any] = {"id": response_id, "object": "response", "created_at": created_at, "status": "completed", "model": model, "output": completed_output_items, "usage": {"input_tokens": usage.context_tokens_used, "output_tokens": usage.response_tokens, "total_tokens": usage.context_tokens_used + usage.response_tokens, "context_tokens_used": usage.context_tokens_used, "context_tokens_limit": usage.context_tokens_limit, "response_tokens": usage.response_tokens}}
        if silent:
            completed_response["silent"] = True
            emit_log(logger, event="silent", request_id=request_id, model=model)
        yield _sse_line({"type": "response.completed", "response": completed_response})
        yield _sse_line("[DONE]")
        emit_log(logger, event="stream_end", request_id=request_id, model=model)
    except Exception as exc:
        emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
        raise
    finally:
        try:
            source.close()
        except Exception:
            pass


def _infer_cli_profile(command: str) -> str:
    import os as _os
    base = _os.path.basename((command or "").strip()).lower()
    return {"claude": "claude", "codex": "codex", "opencode": "opencode"}.get(base, "generic")


def create_app(config: ShimConfig | None = None) -> FastAPI:
    cfg = config or ShimConfig(command="claude")
    app = FastAPI(title="Hermes CLI HTTP Shim", version=__version__)
    app.state.shim_config = cfg
    app.state.session_cache = SessionCache(path=cfg.cache_path, ttl_seconds=cfg.cache_ttl_seconds, max_entries=cfg.cache_max_entries)
    app.state.started_at = time.time()
    app.state.metrics = {"request_count": 0, "cache_hits": 0, "cache_misses": 0, "total_latency_ms": 0.0, "token_context_total": 0, "token_context_max": 0, "token_response_total": 0, "compactions": 0}
    app.state.pending_compactions = {}
    app.state.in_flight = InFlightRegistry()
    logger = configure_logger(log_level=cfg.log_level, log_format=cfg.log_format, logger_name=f"hermes_shim_http.{id(app)}")

    @app.exception_handler(ChildLockBusy)
    async def _child_lock_busy_handler(request: Request, exc: ChildLockBusy) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": {"message": str(exc), "type": "child_lock_busy", "code": "child_lock_busy"}},
            headers={"Retry-After": "5"},
        )

    def _context_window_for(model: str) -> int:
        profile = cfg.cli_profile if cfg.cli_profile != "auto" else _infer_cli_profile(cfg.command)
        return context_limit_for_model(model, profile=profile)

    def _max_output_tokens_for(model: str) -> int:
        profile = cfg.cli_profile if cfg.cli_profile != "auto" else _infer_cli_profile(cfg.command)
        if profile == "claude":
            return 64_000
        if profile == "codex":
            return 32_000
        return 16_000

    def _estimate_usage_for(model: str, *, messages: list[dict[str, Any]], response_text: str) -> TokenUsageEstimate:
        return estimate_token_usage(
            messages=messages,
            response_text=response_text,
            context_limit=_context_window_for(model),
        )

    def _model_payload(model: str) -> dict[str, Any]:
        ctx = _context_window_for(model)
        return {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": cfg.provider_label,
            "context_length": ctx,
            "max_model_len": ctx,
            "max_completion_tokens": _max_output_tokens_for(model),
        }

    def _models_payload() -> dict[str, Any]:
        return {"object": "list", "data": [_model_payload(model) for model in cfg.models]}

    def _props_payload() -> dict[str, Any]:
        return {"provider_label": cfg.provider_label, "api_mode": "chat_completions", "models": list(cfg.models)}

    def _info_payload() -> dict[str, Any]:
        default_model = cfg.models[0] if cfg.models else ""
        profile = cfg.cli_profile if cfg.cli_profile != "auto" else _infer_cli_profile(cfg.command)
        ctx = _context_window_for(default_model) if default_model else 128_000
        max_out = _max_output_tokens_for(default_model) if default_model else 16_000
        return {
            "server": "hermes-shim-http",
            "version": __version__,
            "backend": "cli-http-shim",
            "cli_profile": profile,
            "cli_command": cfg.command,
            "provider_label": cfg.provider_label,
            "model_id": default_model,
            "model_dtype": "external-cli",
            "models": [_model_payload(m) for m in cfg.models],
            "max_input_length": ctx,
            "max_total_tokens": ctx,
            "max_concurrent_requests": None,
            "capabilities": {
                "streaming": True,
                "tools": True,
                "tool_choice": True,
                "vision": profile == "claude",
                "prompt_caching": profile == "claude",
                "session_resume": profile in {"claude", "codex", "opencode"},
                "reasoning": profile == "claude",
                "chat_completions": True,
                "responses_api": True,
            },
            "api_modes": ["chat_completions", "responses"],
            "compaction": cfg.compaction,
            "heartbeat": {
                "child_wrap": cfg.heartbeat_wrap,
                "child_interval_seconds": cfg.heartbeat_interval,
                "http_interval_seconds": cfg.http_heartbeat_interval,
            },
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return _models_payload()

    @app.get("/v1/models/{model_id:path}")
    def model_detail(model_id: str) -> dict[str, Any]:
        if model_id not in cfg.models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return _model_payload(model_id)

    @app.get("/api/v1/models")
    def compat_models() -> dict[str, Any]:
        return _models_payload()

    @app.get("/api/tags")
    def ollama_tags() -> dict[str, Any]:
        return {"models": [{"name": model, "model": model, "modified_at": None, "size": 0, "digest": "", "details": {}} for model in cfg.models]}

    @app.get("/v1/props")
    def props_v1() -> dict[str, Any]:
        return _props_payload()

    @app.get("/props")
    def props_root() -> dict[str, Any]:
        return _props_payload()

    @app.get("/v1/info")
    def info_v1() -> dict[str, Any]:
        return _info_payload()

    @app.get("/info")
    def info_root() -> dict[str, Any]:
        return _info_payload()

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"version": __version__}

    @app.get("/v1/debug/stats")
    def debug_stats() -> dict[str, Any]:
        return _debug_stats_payload(app)

    @app.get("/v1/debug/quota")
    def debug_quota() -> dict[str, str]:
        return {"status": "unknown"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionsRequest, http_request: Request) -> Any:
        idempotency_key = _idempotency_key(http_request)
        request_id = idempotency_key or uuid.uuid4().hex
        request_messages = [message.model_dump() for message in request.messages]
        last_user_text = _extract_last_user_text(request_messages)
        slash = dispatch_slash_command(last_user_text or "", model=request.model, stats=_debug_stats_payload(app)) if last_user_text else None
        if slash is not None:
            response_text = str(slash.message["content"])
            if slash.command == "clear":
                app.state.session_cache.clear()
                usage = _estimate_usage_for(slash.model_override or request.model, messages=request_messages, response_text=response_text)
                payload = _chat_response(model=slash.model_override or request.model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                return JSONResponse(content=payload, headers={"X-Request-Id": request_id})
            if slash.command == "compact":
                compacted_messages, _compacted = _apply_slash_compaction(messages=request_messages, config=cfg, model=request.model)
                compacted = True
                compaction_token = _remember_pending_compaction(app, model=request.model, base_messages=_messages_without_last_user_command(request_messages), compacted_messages=compacted_messages)
                usage = _estimate_usage_for(request.model, messages=compacted_messages or request_messages, response_text=response_text)
                _record_metrics(app, latency_ms=0, cache_hit=False, usage=usage, compacted=compacted)
                payload = _chat_response(model=request.model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                headers = {"X-Request-Id": request_id, "X-Compaction-Token": compaction_token, **({"X-Context-Compacted": "true"} if compacted else {})}
                return JSONResponse(content=payload, headers=headers)
            usage = _estimate_usage_for(slash.model_override or request.model, messages=request_messages, response_text=response_text)
            payload = _chat_response(model=slash.model_override or request.model, parsed=ParsedShimOutput(content=response_text), usage=usage)
            return JSONResponse(content=payload, headers={"X-Request-Id": request_id})

        if idempotency_key and not app.state.in_flight.reserve(idempotency_key):
            emit_log(logger, event="duplicate_request", request_id=request_id, model=request.model, idempotency_key=idempotency_key)
            return _duplicate_request_response(idempotency_key, request_id)

        compaction_token = http_request.headers.get("X-Compaction-Token")
        effective_messages, pending_compacted = _consume_pending_compaction(app, token=compaction_token, messages=request_messages, model=request.model)
        compacted_messages, compacted = _maybe_apply_compaction(messages=effective_messages, config=cfg, model=request.model, force=False)
        compacted = compacted or pending_compacted
        request_tools = [tool.model_dump() for tool in request.tools] if request.tools else None
        session_cache: SessionCache = app.state.session_cache
        if supports_cli_resume(cfg):
            session_plan = session_cache.plan_request(messages=compacted_messages, model=request.model, tools=request_tools, tool_choice=request.tool_choice)
        else:
            session_plan = SessionPlan(
                session_id=str(uuid.uuid4()),
                resume_session_id=None,
                prompt_text=build_cli_prompt(messages=compacted_messages, model=request.model, tools=request_tools, tool_choice=request.tool_choice),
                system_prompt_text=None,
                prefix_message_count=0,
                messages=compacted_messages,
                model=request.model,
                tools=request_tools,
                tool_choice=request.tool_choice,
            )
        _log_chat_request_summary(request=request, request_messages=compacted_messages, request_tools=request_tools, session_plan=session_plan)
        allowed_tool_names = _allowed_tool_names_from_tools(request_tools)
        emit_log(logger, event="cache_hit" if session_plan.resume_session_id else "cache_miss", request_id=request_id, model=request.model)
        headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
        if request.stream:
            emit_log(logger, event="spawn", request_id=request_id, model=request.model, stream=True)
            inner_stream = _stream_live_chat_chunks(app=app, request_id=request_id, logger=logger, model=request.model, request_messages=compacted_messages, prompt=session_plan.prompt_text, config=cfg, allowed_tool_names=allowed_tool_names, session_plan=session_plan, session_cache=session_cache, compacted=compacted)
            return StreamingResponse(
                _release_on_exit(inner_stream, app.state.in_flight, idempotency_key),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers},
            )
        started = time.time()
        emit_log(logger, event="spawn", request_id=request_id, model=request.model, stream=False)

        def _run_chat() -> dict[str, Any]:
            result = run_cli_prompt(
                session_plan.prompt_text,
                cfg,
                session_id=session_plan.session_id,
                resume_session_id=session_plan.resume_session_id,
                system_prompt=session_plan.system_prompt_text,
                model=request.model,
            )
            parsed = _sanitize_parsed_output(parse_cli_result(result, cfg), allowed_tool_names)
            session_cache.record_success(
                session_plan,
                assistant_messages=_assistant_messages_from_parsed(parsed),
                actual_session_id=result.session_id,
            )
            usage = _estimate_usage_for(request.model, messages=compacted_messages, response_text=parsed.content)
            _record_metrics(app, latency_ms=max(result.duration_ms, int((time.time() - started) * 1000)), cache_hit=session_plan.resume_session_id is not None, usage=usage, compacted=compacted)
            if parsed.silent:
                emit_log(logger, event="silent", request_id=request_id, model=request.model)
            emit_log(logger, event="stream_end", request_id=request_id, model=request.model)
            nonlocal headers
            headers = _silent_headers(parsed, headers)
            return _chat_response(model=request.model, parsed=parsed, usage=usage)

        def _on_chat_error(exc: BaseException) -> dict[str, Any]:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=request.model)
            return {"error": {"message": str(exc), "type": "shim_error"}}

        if cfg.http_heartbeat_interval > 0:
            heartbeat_stream = _streaming_json_with_heartbeat(_run_chat, interval=cfg.http_heartbeat_interval, on_error=_on_chat_error)
            return StreamingResponse(
                _release_on_exit(heartbeat_stream, app.state.in_flight, idempotency_key),
                media_type="application/json",
                headers=headers,
            )
        try:
            return JSONResponse(content=_run_chat(), headers=headers)
        except Exception as exc:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=request.model)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            app.state.in_flight.release(idempotency_key)

    @app.post("/v1/responses")
    def responses(body: dict[str, Any], http_request: Request) -> Any:
        idempotency_key = _idempotency_key(http_request)
        request_id = idempotency_key or uuid.uuid4().hex
        model = str(body.get("model") or (cfg.models[0] if cfg.models else "cli-http-shim"))
        try:
            raw_messages = _normalize_responses_input(body.get("input"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})
        last_user_text = _extract_last_user_text(raw_messages)
        slash = dispatch_slash_command(last_user_text or "", model=model, stats=_debug_stats_payload(app)) if last_user_text else None
        if slash is not None:
            response_text = str(slash.message["content"])
            if slash.command == "clear":
                app.state.session_cache.clear()
                usage = _estimate_usage_for(slash.model_override or model, messages=raw_messages, response_text=response_text)
                payload = _responses_json_response(model=slash.model_override or model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                return JSONResponse(content=payload, headers={"X-Request-Id": request_id})
            if slash.command == "compact":
                compacted_messages, _compacted = _apply_slash_compaction(messages=raw_messages, config=cfg, model=model)
                compacted = True
                compaction_token = _remember_pending_compaction(app, model=model, base_messages=_messages_without_last_user_command(raw_messages), compacted_messages=compacted_messages)
                usage = _estimate_usage_for(model, messages=compacted_messages or raw_messages, response_text=response_text)
                _record_metrics(app, latency_ms=0, cache_hit=False, usage=usage, compacted=compacted)
                payload = _responses_json_response(model=model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                headers = {"X-Request-Id": request_id, "X-Compaction-Token": compaction_token, **({"X-Context-Compacted": "true"} if compacted else {})}
                return JSONResponse(content=payload, headers=headers)
            usage = _estimate_usage_for(slash.model_override or model, messages=raw_messages, response_text=response_text)
            payload = _responses_json_response(model=slash.model_override or model, parsed=ParsedShimOutput(content=response_text), usage=usage)
            return JSONResponse(content=payload, headers={"X-Request-Id": request_id})

        if idempotency_key and not app.state.in_flight.reserve(idempotency_key):
            emit_log(logger, event="duplicate_request", request_id=request_id, model=model, idempotency_key=idempotency_key)
            return _duplicate_request_response(idempotency_key, request_id)

        compaction_token = http_request.headers.get("X-Compaction-Token")
        effective_messages, pending_compacted = _consume_pending_compaction(app, token=compaction_token, messages=raw_messages, model=model)
        body_for_prompt = {**body, "input": effective_messages}
        try:
            prompt, allowed_tool_names, normalized_tools, compacted_messages, compacted = _responses_prompt_from_body(body_for_prompt, config=cfg, force_compact=False)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})
        compacted = compacted or pending_compacted
        _log_responses_request_summary(body=body_for_prompt, messages=compacted_messages, normalized_tools=normalized_tools, prompt=prompt)
        headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
        if body.get("stream"):
            emit_log(logger, event="spawn", request_id=request_id, model=model, stream=True)
            inner_stream = _stream_live_responses_events(app=app, request_id=request_id, logger=logger, model=model, prompt=prompt, request_messages=compacted_messages, config=cfg, allowed_tool_names=allowed_tool_names, compacted=compacted)
            return StreamingResponse(
                _release_on_exit(inner_stream, app.state.in_flight, idempotency_key),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers},
            )
        started = time.time()
        emit_log(logger, event="spawn", request_id=request_id, model=model, stream=False)

        def _run_responses() -> dict[str, Any]:
            result = run_cli_prompt(prompt, cfg, model=model)
            ordered_events = _ordered_cli_events_from_text(result.stdout, allowed_tool_names)
            parsed = _parsed_output_from_events(ordered_events)
            usage = _estimate_usage_for(model, messages=compacted_messages, response_text=parsed.content)
            _record_metrics(app, latency_ms=max(result.duration_ms, int((time.time() - started) * 1000)), cache_hit=False, usage=usage, compacted=compacted)
            if parsed.silent:
                emit_log(logger, event="silent", request_id=request_id, model=model)
            emit_log(logger, event="stream_end", request_id=request_id, model=model)
            nonlocal headers
            headers = _silent_headers(parsed, headers)
            return _responses_json_response(model=model, parsed=parsed, usage=usage, output_items=_responses_output_items_from_events(ordered_events))

        def _on_responses_error(exc: BaseException) -> dict[str, Any]:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
            return {"error": {"message": str(exc), "type": "shim_error"}}

        if cfg.http_heartbeat_interval > 0:
            heartbeat_stream = _streaming_json_with_heartbeat(_run_responses, interval=cfg.http_heartbeat_interval, on_error=_on_responses_error)
            return StreamingResponse(
                _release_on_exit(heartbeat_stream, app.state.in_flight, idempotency_key),
                media_type="application/json",
                headers=headers,
            )
        try:
            return JSONResponse(content=_run_responses(), headers=headers)
        except Exception as exc:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            app.state.in_flight.release(idempotency_key)

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose a local CLI as an OpenAI-compatible HTTP shim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--command", default="claude")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--fallback-model", dest="fallback_model", default=None)
    parser.add_argument("--profile", choices=["auto", "claude", "codex", "opencode", "generic"], default="auto")
    parser.add_argument("--cache-path")
    parser.add_argument("--cache-ttl-seconds", type=float, default=3600.0)
    parser.add_argument("--cache-max-entries", type=int, default=256)
    parser.add_argument("--compaction", choices=["off", "summarize", "window"], default="off")
    parser.add_argument("--compaction-threshold", type=float, default=0.9)
    parser.add_argument("--log-level", choices=["info", "debug"], default="info")
    parser.add_argument("--log-format", choices=["text", "json"], default="text")
    parser.add_argument("--heartbeat-wrap", dest="heartbeat_wrap", action=argparse.BooleanOptionalAction, default=True,
                        help="Wrap child CLI with heartbeat-wrap.py so stdout silence during long runs doesn't trip the idle timeout (default: on).")
    parser.add_argument("--heartbeat-interval", dest="heartbeat_interval", type=float, default=60.0,
                        help="Seconds between child-CLI heartbeat emissions when --heartbeat-wrap is on (default: 60.0).")
    parser.add_argument("--single-child-lock", dest="single_child_lock", action=argparse.BooleanOptionalAction, default=True,
        help="Enforce resumed-parent child-spawn locks (default: enabled) so retries cannot fork the same upstream session concurrently.")
    parser.add_argument("--single-child-lock-path", dest="single_child_lock_path", default=None,
        help="Base path for resumed-session lock files (default: /tmp/hermes-shim-http-<port>.child.lock).")
    parser.add_argument("--http-heartbeat-interval", dest="http_heartbeat_interval", type=float, default=30.0,
                        help="Seconds between whitespace heartbeat bytes on non-streaming HTTP responses. 0 disables (default: 30.0).")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _startup_config_payload(*, host: str, port: int, config: ShimConfig) -> dict[str, Any]:
    return {
        "host": host,
        "port": port,
        "command": config.command,
        "cwd": config.cwd,
        "timeout": config.timeout,
        "models": list(config.models),
        "fallback_model": config.fallback_model,
        "provider_label": config.provider_label,
        "cli_profile": config.cli_profile,
        "provided_args": list(config.args),
        "effective_args": resolved_cli_args(config),
        "cache_path": config.cache_path,
        "cache_ttl_seconds": config.cache_ttl_seconds,
        "cache_max_entries": config.cache_max_entries,
        "compaction": config.compaction,
        "compaction_threshold": config.compaction_threshold,
        "log_level": config.log_level,
        "log_format": config.log_format,
        "heartbeat_wrap": config.heartbeat_wrap,
        "heartbeat_interval": config.heartbeat_interval,
        "http_heartbeat_interval": config.http_heartbeat_interval,
        "single_child_lock_path": config.single_child_lock_path,
    }


def main() -> None:
    parser = _build_arg_parser()
    ns = parser.parse_args()
    args = list(ns.args or [])
    if args and args[0] == "--":
        args = args[1:]
    config = ShimConfig(
        command=ns.command,
        args=args,
        cwd=ns.cwd,
        timeout=ns.timeout,
        models=ns.models or ["claude-cli"],
        cli_profile=ns.profile,
        fallback_model=ns.fallback_model,
        cache_path=ns.cache_path or ShimConfig(command=ns.command).cache_path,
        cache_ttl_seconds=ns.cache_ttl_seconds,
        cache_max_entries=ns.cache_max_entries,
        compaction=ns.compaction,
        compaction_threshold=ns.compaction_threshold,
        log_level=ns.log_level,
        log_format=ns.log_format,
        heartbeat_wrap=ns.heartbeat_wrap,
        heartbeat_interval=ns.heartbeat_interval,
        http_heartbeat_interval=ns.http_heartbeat_interval,
        single_child_lock_path=(
            (ns.single_child_lock_path or f"/tmp/hermes-shim-http-{ns.port}.child.lock")
            if ns.single_child_lock
            else None
        ),
    )
    import uvicorn

    print("[hermes-shim-http] effective startup config:")
    print(json.dumps(_startup_config_payload(host=ns.host, port=ns.port, config=config), indent=2, ensure_ascii=False))
    uvicorn.run(create_app(config), host=ns.host, port=ns.port)


if __name__ == "__main__":
    main()
