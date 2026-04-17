from __future__ import annotations

import json
import os
from typing import Any


_DEBUG_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}


def env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _DEBUG_TRUE_VALUES


def emit_event(event: str, /, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(f"[hermes-shim-http] {json.dumps(payload, ensure_ascii=False)}", flush=True)
