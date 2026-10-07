from collections.abc import Sequence
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from pydantic_core import to_json

from agentic.core.tools import Tool as AgenticTool


class ToolBridge(BaseTool[BaseModel, str]):  # pyright: ignore[reportInvalidTypeArguments]
    """An agentic tool seen by AutoGen.

    AutoGen's `FunctionTool` rejects our pydantic `Field` descriptions, so the bridge reuses the
    tool's own args model.
    """

    def __init__(self, tool: AgenticTool[..., Any]) -> None:
        super().__init__(tool.args_model, str, tool.name, tool.description)
        self._tool = tool

    async def run(self, args: BaseModel, cancellation_token: CancellationToken) -> str:
        value = self._tool.fn(**dict(args))
        return value if isinstance(value, str) else to_json(value).decode()


def bridge(tools: Sequence[AgenticTool[..., Any]]) -> list[BaseTool[Any, Any]]:
    return [ToolBridge(t) for t in tools]
