from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .logging_utils import configure_logger, emit_log
from .models import ChatCompletionsRequest, CliStreamEvent, ParsedShimOutput, ShimConfig, ToolDefinition
from .parsing import IncrementalToolCallParser, parse_cli_output
from .prompting import build_cli_prompt, compact_messages
from .runner import resolved_cli_args, run_cli_prompt, stream_cli_prompt
from .session_cache import SessionCache, SessionPlan
from .slash_commands import dispatch_slash_command
from .token_usage import DEFAULT_CONTEXT_LIMIT, TokenUsageEstimate, estimate_token_usage

KEEPALIVE_INTERVAL_SECONDS = 10.0


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
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.content,
    }
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_dict(usage),
    }


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
        return ParsedShimOutput(content=parsed.content, tool_calls=allowed_calls)

    content = parsed.content.strip()
    if not content:
        content = _unsupported_tool_message(unsupported_names)
    return ParsedShimOutput(content=content, tool_calls=allowed_calls)


def _sse_line(payload: dict[str, Any] | str) -> bytes:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n".encode("utf-8")


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
        except BaseException as exc:  # pragma: no cover - defensive
            sink.put(exc)
            sink.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    done = False
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


def _apply_slash_compaction(*, messages: list[dict[str, Any]], config: ShimConfig) -> tuple[list[dict[str, Any]], bool]:
    base_messages = _messages_without_last_user_command(messages)
    effective_mode = config.compaction if config.compaction != "off" else "summarize"
    compacted_messages, compacted = compact_messages(
        messages=base_messages,
        mode=effective_mode,
        threshold=config.compaction_threshold,
        context_limit=DEFAULT_CONTEXT_LIMIT,
        force=True,
    )
    return compacted_messages, compacted


def _maybe_apply_compaction(*, messages: list[dict[str, Any]], config: ShimConfig, force: bool = False) -> tuple[list[dict[str, Any]], bool]:
    return compact_messages(
        messages=messages,
        mode=config.compaction,
        threshold=config.compaction_threshold,
        context_limit=DEFAULT_CONTEXT_LIMIT,
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
    return ParsedShimOutput(
        content="".join(event.text or "" for event in events if event.kind == "text").strip(),
        tool_calls=[event.tool_call for event in events if event.kind == "tool_call" and event.tool_call],
    )


def _responses_json_response(*, model: str, parsed: ParsedShimOutput, usage: TokenUsageEstimate, output_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
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
            messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps({'id': call_id, 'type': 'function', 'function': {'name': str(item.get('name') or '').strip(), 'arguments': str(item.get('arguments') or '{}')}} , ensure_ascii=False)}</tool_call>"})
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
    compacted_messages, compacted = _maybe_apply_compaction(messages=messages, config=config, force=force_compact)
    prompt = build_cli_prompt(messages=compacted_messages, model=str(body.get("model") or "cli-http-shim"), tools=normalized_tools, tool_choice=body.get("tool_choice"))
    allowed_tool_names = _allowed_tool_names_from_tools(body.get("tools"), reject_if_missing=True, strict=tools_present)
    return prompt, allowed_tool_names or set(), normalized_tools, compacted_messages, compacted


def _stream_live_chat_chunks(
    *,
    app: FastAPI,
    request_id: str,
    logger: logging.Logger,
    model: str,
    request_messages: list[dict[str, Any]],
    prompt: str,
    config: ShimConfig,
    allowed_tool_names: set[str] | None,
    session_plan: SessionPlan,
    session_cache: SessionCache,
    compacted: bool,
) -> Iterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    saw_tool_calls = False
    tool_index = 0
    pending_text = ""
    completed_tool_calls: list[dict[str, Any]] = []
    assistant_text_chunks: list[str] = []
    started = time.time()

    emit_log(logger, event="stream_start", request_id=request_id, model=model)
    yield _sse_line({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    try:
        source = stream_cli_prompt(prompt, config, session_id=session_plan.session_id, resume_session_id=session_plan.resume_session_id)
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
                if "\n" in pending_text or len(pending_text) >= 24 or (event.text[-1:].isspace() and len(pending_text) >= 8):
                    yield from _flush_pending_chat_text(completion_id=completion_id, created=created, model=model, pending_text=pending_text)
                    pending_text = ""

        yield from _flush_pending_chat_text(completion_id=completion_id, created=created, model=model, pending_text=pending_text)
        parsed = ParsedShimOutput(content="".join(assistant_text_chunks), tool_calls=completed_tool_calls)
        session_cache.record_success(session_plan, assistant_messages=_assistant_messages_from_parsed(parsed))
        usage = estimate_token_usage(messages=request_messages, response_text=parsed.content)
        _record_metrics(app, latency_ms=int((time.time() - started) * 1000), cache_hit=session_plan.resume_session_id is not None, usage=usage, compacted=compacted)
        yield _sse_line({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if saw_tool_calls else "stop"}], "usage": _usage_dict(usage)})
        yield _sse_line("[DONE]")
        emit_log(logger, event="stream_end", request_id=request_id, model=model)
    except Exception as exc:
        emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
        raise


def _stream_live_responses_events(
    *,
    app: FastAPI,
    request_id: str,
    logger: logging.Logger,
    model: str,
    prompt: str,
    request_messages: list[dict[str, Any]],
    config: ShimConfig,
    allowed_tool_names: set[str] | None,
    compacted: bool,
) -> Iterator[bytes]:
    response_id = f"resp_{uuid.uuid4().hex[:28]}"
    created_at = int(time.time())
    pending_text = ""
    pending_text_item_id: str | None = None
    pending_text_output_index: int | None = None
    output_index = 0
    output_items: list[dict[str, Any]] = []
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

    try:
        for event in _iter_events_with_keepalive(stream_cli_prompt(prompt, config), keepalive_interval=KEEPALIVE_INTERVAL_SECONDS):
            if event is None:
                yield b": ping\n\n"
                continue
            if event.kind == "tool_call" and event.tool_call:
                name = str(event.tool_call.get("function", {}).get("name") or "").strip()
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    pending_text += _unsupported_tool_message([name])
                    yield from ensure_text_item_started()
                    yield _sse_line({"type": "response.output_text.delta", "delta": _unsupported_tool_message([name]), "output_index": pending_text_output_index, "content_index": 0})
                    continue
                yield from flush_text()
                item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}", "call_id": event.tool_call["id"], "name": event.tool_call["function"]["name"], "arguments": event.tool_call["function"]["arguments"]}
                output_items.append(item)
                yield _sse_line({"type": "response.output_item.added", "output_index": output_index, "item": item})
                yield _sse_line({"type": "response.output_item.done", "output_index": output_index, "item": item})
                output_index += 1
            elif event.kind == "text" and event.text:
                pending_text += event.text
                yield from ensure_text_item_started()
                yield _sse_line({"type": "response.output_text.delta", "delta": event.text, "output_index": pending_text_output_index, "content_index": 0})

        yield from flush_text()
        response_text = "".join(
            content["text"]
            for item in output_items
            if item.get("type") == "message"
            for content in item.get("content", [])
            if isinstance(content, dict) and isinstance(content.get("text"), str)
        )
        usage = estimate_token_usage(messages=request_messages, response_text=response_text)
        _record_metrics(app, latency_ms=int((time.time() - started) * 1000), cache_hit=False, usage=usage, compacted=compacted)
        completed_response = {"id": response_id, "object": "response", "created_at": created_at, "status": "completed", "model": model, "output": output_items, "usage": {"input_tokens": usage.context_tokens_used, "output_tokens": usage.response_tokens, "total_tokens": usage.context_tokens_used + usage.response_tokens, "context_tokens_used": usage.context_tokens_used, "context_tokens_limit": usage.context_tokens_limit, "response_tokens": usage.response_tokens}}
        yield _sse_line({"type": "response.completed", "response": completed_response})
        yield _sse_line("[DONE]")
        emit_log(logger, event="stream_end", request_id=request_id, model=model)
    except Exception as exc:
        emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
        raise


def create_app(config: ShimConfig | None = None) -> FastAPI:
    cfg = config or ShimConfig(command="claude")
    app = FastAPI(title="Hermes CLI HTTP Shim", version=__version__)
    app.state.shim_config = cfg
    app.state.session_cache = SessionCache(path=cfg.cache_path, ttl_seconds=cfg.cache_ttl_seconds, max_entries=cfg.cache_max_entries)
    app.state.started_at = time.time()
    app.state.metrics = {"request_count": 0, "cache_hits": 0, "cache_misses": 0, "total_latency_ms": 0.0, "token_context_total": 0, "token_context_max": 0, "token_response_total": 0, "compactions": 0}
    logger = configure_logger(log_level=cfg.log_level, log_format=cfg.log_format, logger_name=f"hermes_shim_http.{id(app)}")

    def _model_payload(model: str) -> dict[str, Any]:
        return {"id": model, "object": "model", "created": 0, "owned_by": cfg.provider_label}

    def _models_payload() -> dict[str, Any]:
        return {"object": "list", "data": [_model_payload(model) for model in cfg.models]}

    def _props_payload() -> dict[str, Any]:
        return {"provider_label": cfg.provider_label, "api_mode": "chat_completions", "models": list(cfg.models)}

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
    def chat_completions(request: ChatCompletionsRequest) -> Any:
        request_id = uuid.uuid4().hex
        request_messages = [message.model_dump() for message in request.messages]
        last_user_text = _extract_last_user_text(request_messages)
        slash = dispatch_slash_command(last_user_text or "", model=request.model, stats=_debug_stats_payload(app)) if last_user_text else None
        if slash is not None:
            if slash.command == "clear":
                app.state.session_cache.clear()
                usage = estimate_token_usage(messages=request_messages, response_text=str(slash.message["content"]))
                payload = _chat_response(model=slash.model_override or request.model, parsed=ParsedShimOutput(content=str(slash.message["content"])), usage=usage)
                return JSONResponse(content=payload, headers={"X-Request-Id": request_id})
            if slash.command == "compact":
                compacted_messages, _compacted = _apply_slash_compaction(messages=request_messages, config=cfg)
                compacted = True
                compacted_note = str(slash.message["content"])
                usage = estimate_token_usage(messages=compacted_messages or request_messages, response_text=compacted_note)
                _record_metrics(app, latency_ms=0, cache_hit=False, usage=usage, compacted=compacted)
                payload = _chat_response(model=request.model, parsed=ParsedShimOutput(content=compacted_note), usage=usage)
                headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
                return JSONResponse(content=payload, headers=headers)
            payload = _chat_response(model=slash.model_override or request.model, parsed=ParsedShimOutput(content=str(slash.message["content"])), usage=estimate_token_usage(messages=request_messages, response_text=str(slash.message["content"])))
            return JSONResponse(content=payload, headers={"X-Request-Id": request_id})

        compacted_messages, compacted = _maybe_apply_compaction(messages=request_messages, config=cfg, force=False)
        request_tools = [tool.model_dump() for tool in request.tools] if request.tools else None
        session_cache: SessionCache = app.state.session_cache
        session_plan = session_cache.plan_request(messages=compacted_messages, model=request.model, tools=request_tools, tool_choice=request.tool_choice)
        allowed_tool_names = _allowed_tool_names_from_tools(request_tools)
        emit_log(logger, event="cache_hit" if session_plan.resume_session_id else "cache_miss", request_id=request_id, model=request.model)
        headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
        if request.stream:
            emit_log(logger, event="spawn", request_id=request_id, model=request.model, stream=True)
            return StreamingResponse(
                _stream_live_chat_chunks(app=app, request_id=request_id, logger=logger, model=request.model, request_messages=compacted_messages, prompt=session_plan.prompt_text, config=cfg, allowed_tool_names=allowed_tool_names, session_plan=session_plan, session_cache=session_cache, compacted=compacted),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers},
            )
        try:
            started = time.time()
            emit_log(logger, event="spawn", request_id=request_id, model=request.model, stream=False)
            result = run_cli_prompt(session_plan.prompt_text, cfg, session_id=session_plan.session_id, resume_session_id=session_plan.resume_session_id)
            parsed = _sanitize_parsed_output(parse_cli_output(result.stdout), allowed_tool_names)
            session_cache.record_success(session_plan, assistant_messages=_assistant_messages_from_parsed(parsed))
            usage = estimate_token_usage(messages=compacted_messages, response_text=parsed.content)
            _record_metrics(app, latency_ms=max(result.duration_ms, int((time.time() - started) * 1000)), cache_hit=session_plan.resume_session_id is not None, usage=usage, compacted=compacted)
            emit_log(logger, event="stream_end", request_id=request_id, model=request.model)
            return JSONResponse(content=_chat_response(model=request.model, parsed=parsed, usage=usage), headers=headers)
        except Exception as exc:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=request.model)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/responses")
    def responses(body: dict[str, Any]) -> Any:
        request_id = uuid.uuid4().hex
        model = str(body.get("model") or (cfg.models[0] if cfg.models else "cli-http-shim"))
        try:
            raw_messages = _normalize_responses_input(body.get("input"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})
        last_user_text = _extract_last_user_text(raw_messages)
        slash = dispatch_slash_command(last_user_text or "", model=model, stats=_debug_stats_payload(app)) if last_user_text else None
        if slash is not None:
            if slash.command == "clear":
                app.state.session_cache.clear()
                response_text = str(slash.message["content"])
                usage = estimate_token_usage(messages=raw_messages, response_text=response_text)
                payload = _responses_json_response(model=slash.model_override or model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                return JSONResponse(content=payload, headers={"X-Request-Id": request_id})
            if slash.command == "compact":
                compacted_messages, _compacted = _apply_slash_compaction(messages=raw_messages, config=cfg)
                compacted = True
                response_text = str(slash.message["content"])
                usage = estimate_token_usage(messages=compacted_messages or raw_messages, response_text=response_text)
                _record_metrics(app, latency_ms=0, cache_hit=False, usage=usage, compacted=compacted)
                payload = _responses_json_response(model=model, parsed=ParsedShimOutput(content=response_text), usage=usage)
                headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
                return JSONResponse(content=payload, headers=headers)
            response_text = str(slash.message["content"])
            usage = estimate_token_usage(messages=raw_messages, response_text=response_text)
            payload = _responses_json_response(model=slash.model_override or model, parsed=ParsedShimOutput(content=response_text), usage=usage)
            return JSONResponse(content=payload, headers={"X-Request-Id": request_id})
        try:
            prompt, allowed_tool_names, _normalized_tools, compacted_messages, compacted = _responses_prompt_from_body(body, config=cfg, force_compact=False)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})

        headers = {"X-Request-Id": request_id, **({"X-Context-Compacted": "true"} if compacted else {})}
        if body.get("stream"):
            emit_log(logger, event="spawn", request_id=request_id, model=model, stream=True)
            return StreamingResponse(
                _stream_live_responses_events(app=app, request_id=request_id, logger=logger, model=model, prompt=prompt, request_messages=compacted_messages, config=cfg, allowed_tool_names=allowed_tool_names, compacted=compacted),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers},
            )
        try:
            started = time.time()
            emit_log(logger, event="spawn", request_id=request_id, model=model, stream=False)
            result = run_cli_prompt(prompt, cfg)
            ordered_events = _ordered_cli_events_from_text(result.stdout, allowed_tool_names)
            parsed = _parsed_output_from_events(ordered_events)
            usage = estimate_token_usage(messages=compacted_messages, response_text=parsed.content)
            _record_metrics(app, latency_ms=max(result.duration_ms, int((time.time() - started) * 1000)), cache_hit=False, usage=usage, compacted=compacted)
            emit_log(logger, event="stream_end", request_id=request_id, model=model)
            return JSONResponse(content=_responses_json_response(model=model, parsed=parsed, usage=usage, output_items=_responses_output_items_from_events(ordered_events)), headers=headers)
        except Exception as exc:
            emit_log(logger, event="error", request_id=request_id, error=str(exc), model=model)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose a local CLI as an OpenAI-compatible HTTP shim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--command", default="claude")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--profile", choices=["auto", "claude", "codex", "opencode", "generic"], default="auto")
    parser.add_argument("--cache-path")
    parser.add_argument("--cache-ttl-seconds", type=float, default=3600.0)
    parser.add_argument("--cache-max-entries", type=int, default=256)
    parser.add_argument("--compaction", choices=["off", "summarize", "window"], default="off")
    parser.add_argument("--compaction-threshold", type=float, default=0.9)
    parser.add_argument("--log-level", choices=["info", "debug"], default="info")
    parser.add_argument("--log-format", choices=["text", "json"], default="text")
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
        cache_path=ns.cache_path or ShimConfig(command=ns.command).cache_path,
        cache_ttl_seconds=ns.cache_ttl_seconds,
        cache_max_entries=ns.cache_max_entries,
        compaction=ns.compaction,
        compaction_threshold=ns.compaction_threshold,
        log_level=ns.log_level,
        log_format=ns.log_format,
    )
    import uvicorn

    print("[hermes-shim-http] effective startup config:")
    print(json.dumps(_startup_config_payload(host=ns.host, port=ns.port, config=config), indent=2, ensure_ascii=False))
    uvicorn.run(create_app(config), host=ns.host, port=ns.port)


if __name__ == "__main__":
    main()
