#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="legacy")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Sleep duration for idle-silent / delayed-output modes.")
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args()
    prompt = args.prompt or ""

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

    if args.mode == "delayed-output":
        time.sleep(args.duration)
        print("delayed done")
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
