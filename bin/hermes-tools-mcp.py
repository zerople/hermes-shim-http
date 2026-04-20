#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hermes-tools-mcp", "version": "0.1.0"}


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        try:
            key, value = line.decode("utf-8").split(":", 1)
        except ValueError:
            continue
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or "0")
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _load_tools(path: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        tools.append(
            {
                "name": name,
                "description": str(function.get("description") or "").strip(),
                "inputSchema": function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}},
            }
        )
    return tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-file", required=True)
    args = parser.parse_args()
    tools = _load_tools(args.tools_file)

    while True:
        message = _read_message()
        if message is None:
            return
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            if request_id is not None:
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": {"tools": {}},
                            "serverInfo": SERVER_INFO,
                        },
                    }
                )
            continue

        if method == "notifications/initialized":
            continue

        if method == "ping":
            if request_id is not None:
                _write_message({"jsonrpc": "2.0", "id": request_id, "result": {}})
            continue

        if method == "tools/list":
            if request_id is not None:
                _write_message({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}})
            continue

        if method == "tools/call":
            name = str(params.get("name") or "").strip()
            if request_id is not None:
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Delegated to upstream shim: {name}",
                                }
                            ],
                            "isError": False,
                        },
                    }
                )
            continue

        if request_id is not None:
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
