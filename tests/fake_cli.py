#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time


def _emit_claude_event(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _emit_claude_stream_json(prompt: str) -> None:
    _emit_claude_event({"type": "system", "subtype": "init", "session_id": "fake-session"})
    _emit_claude_event({"type": "stream_event", "event": {"type": "message_start", "message": {"id": "msg_fake"}}})

    if "read the readme" in prompt.lower():
        _emit_claude_event({
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0,
                      "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file"}},
        })
        for partial in ['{"pa', 'th":"R', 'EADME.md"}']:
            _emit_claude_event({
                "type": "stream_event",
                "event": {"type": "content_block_delta", "index": 0,
                          "delta": {"type": "input_json_delta", "partial_json": partial}},
            })
        _emit_claude_event({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
    else:
        _emit_claude_event({
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        })
        _emit_claude_event({
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 0,
                      "delta": {"type": "thinking_delta", "thinking": "pondering"}},
        })
        _emit_claude_event({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
        _emit_claude_event({
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        })
        for piece in ["Streaming ", "hello ", "from ", "claude"]:
            _emit_claude_event({
                "type": "stream_event",
                "event": {"type": "content_block_delta", "index": 1,
                          "delta": {"type": "text_delta", "text": piece}},
            })
        _emit_claude_event({"type": "stream_event", "event": {"type": "content_block_stop", "index": 1}})

    _emit_claude_event({"type": "result", "subtype": "success"})


def _emit_multiturn_response(prompt: str) -> None:
    reply = f"echo:{prompt.strip()}"
    _emit_claude_event({
        "type": "stream_event",
        "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
    })
    _emit_claude_event({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "text_delta", "text": reply}},
    })
    _emit_claude_event({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
    _emit_claude_event({"type": "result", "subtype": "success"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="legacy")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Sleep duration for idle-silent / delayed-output modes.")
    # Accept Claude CLI flags so fake_cli can be invoked with the real claude
    # profile argv without choking the argparse step.
    parser.add_argument("-p", action="store_true")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    parser.add_argument("--output-format", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--include-partial-messages", action="store_true")
    parser.add_argument("--append-system-prompt", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fork-session", action="store_true")
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args()
    prompt = args.prompt or ""
    # Claude profile pipes the prompt via stdin; read it opportunistically.
    # Multiturn mode reads stdin line-by-line in its own loop.
    if not prompt and not sys.stdin.isatty() and args.mode != "claude-multiturn":
        try:
            stdin_text = sys.stdin.read()
        except Exception:
            stdin_text = ""
        if stdin_text:
            prompt = stdin_text

    if args.mode == "stream-text":
        for chunk in ["Streaming ", "hello ", "from ", "fake CLI"]:
            print(chunk, end="", flush=True)
            time.sleep(0.01)
        print()
        return 0

    if args.mode == "stream-tool":
        payload = (
            '<tool_call>{"id":"call_1","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            '</tool_call>'
        )
        for chunk in [payload[:20], payload[20:55], payload[55:]]:
            print(chunk, end="", flush=True)
            time.sleep(0.01)
        return 0

    if args.mode == "stream-slow":
        print("slow", end="", flush=True)
        time.sleep(0.05)
        print(" stream", end="", flush=True)
        return 0

    if args.mode == "idle-silent":
        time.sleep(args.duration)
        return 0

    if args.mode == "heartbeat-only":
        deadline = time.time() + args.duration
        while time.time() < deadline:
            sys.stderr.buffer.write("\u200b".encode("utf-8"))
            sys.stderr.buffer.flush()
            time.sleep(0.1)
        return 0

    if args.mode == "delayed-output":
        time.sleep(args.duration)
        print("delayed done")
        return 0

    if args.mode == "chatty-long":
        deadline = time.time() + args.duration
        i = 0
        while time.time() < deadline:
            print(f"tick {i}", flush=True)
            time.sleep(0.1)
            i += 1
        return 0

    if args.mode == "claude-stream-json":
        _emit_claude_stream_json(prompt)
        return 0

    if args.mode == "claude-multiturn":
        session_id = args.session_id or "fake-session"
        _emit_claude_event({"type": "system", "subtype": "init", "session_id": session_id})
        if prompt:
            _emit_multiturn_response(prompt)
        while True:
            line = sys.stdin.readline()
            if not line:
                return 0
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("type") != "user":
                continue
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                turn_prompt = "".join(texts)
            else:
                turn_prompt = str(content or "")
            _emit_multiturn_response(turn_prompt)

    if args.mode == "flood":
        # Emit args.duration megabytes as fast as possible.
        megs = int(args.duration)
        payload = ("x" * 1024 + "\n") * 1024  # ~1 MB
        for _ in range(megs):
            sys.stdout.write(payload)
            sys.stdout.flush()
        return 0

    if "read_file" in prompt.lower() and "read the readme" in prompt.lower():
        print('<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call>')
    elif "say hello" in prompt.lower():
        print("Hello from fake CLI")
    else:
        print("Generic response from fake CLI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
