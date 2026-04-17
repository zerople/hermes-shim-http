from __future__ import annotations

import json
import logging
import uuid
from typing import Any

LOGGER_NAME = "hermes_shim_http"


def format_log_event(*, event: str, request_id: str, log_format: str, extra: dict[str, Any] | None = None) -> str:
    payload = {"event": event, "request_id": request_id, **(extra or {})}
    if log_format == "json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    pieces = [f"event={event}", f"request_id={request_id}"]
    for key, value in (extra or {}).items():
        pieces.append(f"{key}={value}")
    return " ".join(pieces)


def configure_logger(*, log_level: str, log_format: str, logger_name: str | None = None) -> logging.Logger:
    name = logger_name or f"{LOGGER_NAME}.{uuid.uuid4().hex}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if log_level == "debug" else logging.INFO)
    logger.propagate = True
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger._hermes_log_format = log_format  # type: ignore[attr-defined]
    return logger


def emit_log(logger: logging.Logger, *, event: str, request_id: str, level: int = logging.INFO, **extra: Any) -> None:
    log_format = getattr(logger, "_hermes_log_format", "text")
    logger.log(level, format_log_event(event=event, request_id=request_id, log_format=log_format, extra=extra))
