from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mcp_server_script() -> Path:
    return _repo_root() / "bin" / "hermes-tools-mcp.py"


@contextmanager
def request_scoped_mcp_config(*, tools: list[dict[str, Any]] | None) -> Iterator[str | None]:
    if not tools:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="hermes-shim-mcp-") as tmpdir:
        tmp = Path(tmpdir)
        tools_path = tmp / "tools.json"
        config_path = tmp / "mcp.json"
        tools_path.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")
        config = {
            "mcpServers": {
                "hermes": {
                    "command": sys.executable,
                    "args": [str(_mcp_server_script()), "--tools-file", str(tools_path)],
                    "env": {
                        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
                    },
                }
            }
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        yield str(config_path)
