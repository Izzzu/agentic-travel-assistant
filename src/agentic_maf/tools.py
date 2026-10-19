from collections.abc import Callable, Sequence
from typing import Any

from agent_framework import FunctionTool

from agentic.core.tools import Tool


def _validated(tool: Tool[..., Any]) -> Callable[..., Any]:
    """Our args model parses the raw JSON arguments (dates, bounds) before the tool runs."""

    def call(**arguments: Any) -> Any:
        return tool.fn(**dict(tool.args_model.model_validate(arguments)))

    return call


def bridge(tools: Sequence[Tool[..., Any]]) -> list[FunctionTool]:
    """Our tools as Agent Framework tools, advertising the schema of each tool's args model."""
    return [
        FunctionTool(
            name=t.name,
            description=t.description,
            func=_validated(t),
            input_model=t.args_model.model_json_schema(),
        )
        for t in tools
    ]
