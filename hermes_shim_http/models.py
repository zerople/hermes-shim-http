from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False


class ParsedShimOutput(BaseModel):
    content: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class CliRunResult(BaseModel):
    stdout: str
    stderr: str = ""
    exit_code: int
    duration_ms: int


class CliStreamEvent(BaseModel):
    kind: Literal["text", "tool_call"]
    text: str | None = None
    tool_call: Dict[str, Any] | None = None


class ShimConfig(BaseModel):
    command: str
    args: List[str] = Field(default_factory=list)
    cwd: str = "."
    timeout: float = 300.0
    models: List[str] = Field(default_factory=lambda: ["sonnet", "opus", "haiku"])
    provider_label: str = "cli-http-shim"
    cli_profile: Literal["auto", "claude", "codex", "opencode", "generic"] = "auto"
    fallback_model: Optional[str] = None
    cache_path: Optional[str] = None
    cache_ttl_seconds: float = 3600.0
    cache_max_entries: int = 256
    compaction: Literal["off", "summarize", "window"] = "off"
    compaction_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    log_level: Literal["info", "debug"] = "info"
    log_format: Literal["text", "json"] = "text"
