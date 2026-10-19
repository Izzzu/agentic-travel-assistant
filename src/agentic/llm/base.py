from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from agentic.core.messages import LLMResponse, Message, ToolSpec


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Return the next assistant message; with `response_format`, `content` is JSON for it."""
        ...
