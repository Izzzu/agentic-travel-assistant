import itertools
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_json

from agentic.core.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

_call_ids = itertools.count(1)


def reply(content: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        model="scripted",
        content=content,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def call_tool(
    name: str, arguments: dict[str, Any] | str | None = None, *, call_id: str | None = None
) -> LLMResponse:
    """A response that calls one tool; a string `arguments` is sent as-is (e.g. malformed JSON)."""
    raw = arguments if isinstance(arguments, str) else to_json(arguments or {}).decode()
    return LLMResponse(
        model="scripted",
        tool_calls=[ToolCall(id=call_id or f"call_{next(_call_ids)}", name=name, arguments=raw)],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class ScriptedLLM:
    """Returns predefined responses in order and records every request, for offline tests."""

    def __init__(self, responses: Iterable[LLMResponse], *, model: str = "scripted") -> None:
        self._responses = deque(responses)
        self._model = model
        self.requests: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        self.tools_seen.append(list(tools))
        if not self._responses:
            raise RuntimeError("ScriptedLLM has no responses left")
        return self._responses.popleft()
