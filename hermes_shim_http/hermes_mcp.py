from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mcp_server_script() -> Path:
    return _repo_root() / "bin" / "hermes-tools-mcp.py"


_cache_lock = threading.Lock()
_config_cache: dict[str, str] = {}
_cache_dirs: list[str] = []
_atexit_registered = False


def _cleanup_cache_dirs() -> None:
    for path in _cache_dirs:
        shutil.rmtree(path, ignore_errors=True)
    _cache_dirs.clear()


def _tools_fingerprint(tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(tools, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _build_config_for_tools(tools: list[dict[str, Any]]) -> str:
    global _atexit_registered
    tmpdir = tempfile.mkdtemp(prefix="hermes-shim-mcp-")
    tools_path = Path(tmpdir) / "tools.json"
    config_path = Path(tmpdir) / "mcp.json"
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
    _cache_dirs.append(tmpdir)
    if not _atexit_registered:
        atexit.register(_cleanup_cache_dirs)
        _atexit_registered = True
    return str(config_path)


@contextmanager
def request_scoped_mcp_config(*, tools: list[dict[str, Any]] | None) -> Iterator[str | None]:
    if not tools:
        yield None
        return
    fingerprint = _tools_fingerprint(tools)
    with _cache_lock:
        cached = _config_cache.get(fingerprint)
        if cached and os.path.isfile(cached):
            yield cached
            return
        path = _build_config_for_tools(tools)
        _config_cache[fingerprint] = path
    yield path


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cleanup_cache_dirs()
        _config_cache.clear()
