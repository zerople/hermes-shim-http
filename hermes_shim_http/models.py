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
    timeout: float = 120.0
    models: List[str] = Field(default_factory=lambda: ["claude-cli"])
    provider_label: str = "cli-http-shim"
    cli_profile: Literal["auto", "claude", "codex", "opencode", "generic"] = "auto"
