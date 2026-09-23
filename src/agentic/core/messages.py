from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "other"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str = "{}"  # raw JSON from the model; validated when the tool runs


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Self:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Self:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] | None = None) -> Self:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Self:
        return cls(role="tool", tool_call_id=tool_call_id, content=content)


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class LLMResponse(BaseModel):
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)


class ToolCallRecord(BaseModel):
    call_id: str
    name: str
    arguments: str
    result: str
    ok: bool
    ms: float


class AgentResult(BaseModel):
    agent: str
    text: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list[ToolCallRecord])
    usage: Usage = Field(default_factory=Usage)
    ms: float = 0.0
