from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .models import ChatCompletionsRequest, CliStreamEvent, ParsedShimOutput, ShimConfig, ToolDefinition
from .parsing import IncrementalToolCallParser, parse_cli_output
from .prompting import build_cli_prompt
from .runner import resolved_cli_args, run_cli_prompt, stream_cli_prompt
from .session_cache import SessionCache, SessionPlan


def _chat_response(*, model: str, parsed: ParsedShimOutput) -> dict[str, Any]:
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
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _assistant_messages_from_parsed(parsed: ParsedShimOutput) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": parsed.content}]


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
    yield _sse_line(
        _stream_chunk_for_text(
            completion_id=completion_id,
            created=created,
            model=model,
            text=pending_text,
        )
    )


def _stream_live_chat_chunks(
    *,
    model: str,
    prompt: str,
    config: ShimConfig,
    allowed_tool_names: set[str] | None,
    session_plan: SessionPlan,
    session_cache: SessionCache,
) -> Iterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    saw_tool_calls = False
    tool_index = 0
    pending_text = ""
    completed_tool_calls: list[dict[str, Any]] = []
    assistant_text_chunks: list[str] = []

    yield _sse_line(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    for event in stream_cli_prompt(
        prompt,
        config,
        session_id=session_plan.session_id,
        resume_session_id=session_plan.resume_session_id,
    ):
        if event.kind == "tool_call" and event.tool_call:
            name = str(event.tool_call.get("function", {}).get("name") or "").strip()
            if allowed_tool_names is not None and name not in allowed_tool_names:
                pending_text += _unsupported_tool_message([name])
                continue
            yield from _flush_pending_chat_text(
                completion_id=completion_id,
                created=created,
                model=model,
                pending_text=pending_text,
            )
            pending_text = ""
            saw_tool_calls = True
            completed_tool_calls.append(event.tool_call)
            yield _sse_line(
                _stream_chunk_for_tool_call(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    tool_call=event.tool_call,
                    index=tool_index,
                )
            )
            tool_index += 1
        elif event.kind == "text" and event.text:
            pending_text += event.text
            assistant_text_chunks.append(event.text)
            if "\n" in pending_text or len(pending_text) >= 24 or (event.text[-1:].isspace() and len(pending_text) >= 8):
                yield from _flush_pending_chat_text(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    pending_text=pending_text,
                )
                pending_text = ""

    yield from _flush_pending_chat_text(
        completion_id=completion_id,
        created=created,
        model=model,
        pending_text=pending_text,
    )

    session_cache.record_success(
        session_plan,
        assistant_messages=[
            {
                "role": "assistant",
                "content": "".join(assistant_text_chunks),
                **({"tool_calls": completed_tool_calls} if completed_tool_calls else {}),
            }
        ],
    )

    yield _sse_line(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if saw_tool_calls else "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )
    yield _sse_line("[DONE]")


def _normalize_responses_input(raw_input: Any) -> list[dict[str, Any]]:
    if raw_input is None:
        raise HTTPException(status_code=400, detail="Missing 'input' field")

    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]

    if not isinstance(raw_input, list):
        raise HTTPException(status_code=400, detail="'input' must be a string or array")

    messages: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(raw_input):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"input[{idx}] must be an object or string")

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{idx}")
            pending_tool_calls[call_id] = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(item.get("name") or "").strip(),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
            messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps(pending_tool_calls[call_id], ensure_ascii=False)}</tool_call>"})
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or "").strip() or None,
                    "content": str(item.get("output") or ""),
                }
            )
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


def _responses_prompt_from_body(body: dict[str, Any]) -> tuple[str, set[str], list[dict[str, Any]] | None]:
    tools_present = "tools" in body
    normalized_tools = _normalize_chat_tools(body.get("tools"), strict=tools_present)
    messages = _normalize_responses_input(body.get("input"))
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages = [{"role": "system", "content": instructions.strip()}, *messages]
    prompt = build_cli_prompt(
        messages=messages,
        model=str(body.get("model") or "cli-http-shim"),
        tools=normalized_tools,
        tool_choice=body.get("tool_choice"),
    )
    allowed_tool_names = _allowed_tool_names_from_tools(body.get("tools"), reject_if_missing=True, strict=tools_present)
    return prompt, allowed_tool_names or set(), normalized_tools


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
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": event.text}],
                }
            )
    if not items:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            }
        )
    return items


def _parsed_output_from_events(events: list[CliStreamEvent]) -> ParsedShimOutput:
    return ParsedShimOutput(
        content="".join(event.text or "" for event in events if event.kind == "text").strip(),
        tool_calls=[event.tool_call for event in events if event.kind == "tool_call" and event.tool_call],
    )


def _responses_json_response(*, model: str, parsed: ParsedShimOutput, output_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex[:28]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output_items or _responses_output_items_from_events([CliStreamEvent(kind="text", text=parsed.content)] if parsed.content and not parsed.tool_calls else []),
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_live_responses_events(*, model: str, prompt: str, config: ShimConfig, allowed_tool_names: set[str] | None) -> Iterator[bytes]:
    response_id = f"resp_{uuid.uuid4().hex[:28]}"
    created_at = int(time.time())
    pending_text = ""
    pending_text_item_id: str | None = None
    pending_text_output_index: int | None = None
    output_index = 0
    output_items: list[dict[str, Any]] = []

    response_stub = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model,
        "output": [],
    }
    yield _sse_line({"type": "response.created", "response": response_stub})

    def flush_text() -> Iterator[bytes]:
        nonlocal pending_text, pending_text_item_id, pending_text_output_index, output_index
        if pending_text_output_index is None or pending_text_item_id is None:
            pending_text = ""
            return
        message_item = {
            "type": "message",
            "id": pending_text_item_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": pending_text}],
        }
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
        yield _sse_line(
            {
                "type": "response.output_item.added",
                "output_index": pending_text_output_index,
                "item": {
                    "type": "message",
                    "id": pending_text_item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )

    for event in stream_cli_prompt(prompt, config):
        if event.kind == "tool_call" and event.tool_call:
            name = str(event.tool_call.get("function", {}).get("name") or "").strip()
            if allowed_tool_names is not None and name not in allowed_tool_names:
                pending_text += _unsupported_tool_message([name])
                yield from ensure_text_item_started()
                yield _sse_line({"type": "response.output_text.delta", "delta": _unsupported_tool_message([name]), "output_index": pending_text_output_index, "content_index": 0})
                continue
            yield from flush_text()
            item = {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": event.tool_call["id"],
                "name": event.tool_call["function"]["name"],
                "arguments": event.tool_call["function"]["arguments"],
            }
            output_items.append(item)
            yield _sse_line({"type": "response.output_item.added", "output_index": output_index, "item": item})
            yield _sse_line({"type": "response.output_item.done", "output_index": output_index, "item": item})
            output_index += 1
        elif event.kind == "text" and event.text:
            pending_text += event.text
            yield from ensure_text_item_started()
            yield _sse_line({"type": "response.output_text.delta", "delta": event.text, "output_index": pending_text_output_index, "content_index": 0})

    yield from flush_text()

    completed_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    yield _sse_line({"type": "response.completed", "response": completed_response})
    yield _sse_line("[DONE]")


def create_app(config: ShimConfig | None = None) -> FastAPI:
    cfg = config or ShimConfig(command="claude")
    app = FastAPI(title="Hermes CLI HTTP Shim", version=__version__)
    app.state.shim_config = cfg
    app.state.session_cache = SessionCache()

    def _model_payload(model: str) -> dict[str, Any]:
        return {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": cfg.provider_label,
        }

    def _models_payload() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [_model_payload(model) for model in cfg.models],
        }

    def _props_payload() -> dict[str, Any]:
        return {
            "provider_label": cfg.provider_label,
            "api_mode": "chat_completions",
            "models": list(cfg.models),
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
        return {
            "models": [
                {
                    "name": model,
                    "model": model,
                    "modified_at": None,
                    "size": 0,
                    "digest": "",
                    "details": {},
                }
                for model in cfg.models
            ]
        }

    @app.get("/v1/props")
    def props_v1() -> dict[str, Any]:
        return _props_payload()

    @app.get("/props")
    def props_root() -> dict[str, Any]:
        return _props_payload()

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"version": __version__}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionsRequest) -> Any:
        request_messages = [message.model_dump() for message in request.messages]
        request_tools = [tool.model_dump() for tool in request.tools] if request.tools else None
        session_cache: SessionCache = app.state.session_cache
        session_plan = session_cache.plan_request(
            messages=request_messages,
            model=request.model,
            tools=request_tools,
            tool_choice=request.tool_choice,
        )
        allowed_tool_names = _allowed_tool_names_from_tools(request_tools)
        if request.stream:
            return StreamingResponse(
                _stream_live_chat_chunks(
                    model=request.model,
                    prompt=session_plan.prompt_text,
                    config=cfg,
                    allowed_tool_names=allowed_tool_names,
                    session_plan=session_plan,
                    session_cache=session_cache,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        result = run_cli_prompt(
            session_plan.prompt_text,
            cfg,
            session_id=session_plan.session_id,
            resume_session_id=session_plan.resume_session_id,
        )
        parsed = _sanitize_parsed_output(parse_cli_output(result.stdout), allowed_tool_names)
        session_cache.record_success(session_plan, assistant_messages=_assistant_messages_from_parsed(parsed))
        return _chat_response(model=request.model, parsed=parsed)

    @app.post("/v1/responses")
    def responses(body: dict[str, Any]) -> Any:
        try:
            prompt, allowed_tool_names, _normalized_tools = _responses_prompt_from_body(body)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})

        model = str(body.get("model") or (cfg.models[0] if cfg.models else "cli-http-shim"))
        if body.get("stream"):
            return StreamingResponse(
                _stream_live_responses_events(model=model, prompt=prompt, config=cfg, allowed_tool_names=allowed_tool_names),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        result = run_cli_prompt(prompt, cfg)
        ordered_events = _ordered_cli_events_from_text(result.stdout, allowed_tool_names)
        parsed = _parsed_output_from_events(ordered_events)
        return _responses_json_response(model=model, parsed=parsed, output_items=_responses_output_items_from_events(ordered_events))

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
    )
    import uvicorn

    print("[hermes-shim-http] effective startup config:")
    print(json.dumps(_startup_config_payload(host=ns.host, port=ns.port, config=config), indent=2, ensure_ascii=False))
    uvicorn.run(create_app(config), host=ns.host, port=ns.port)


if __name__ == "__main__":
    main()
