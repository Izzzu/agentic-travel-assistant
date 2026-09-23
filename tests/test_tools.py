import json
from typing import Annotated

from pydantic import Field

from agentic.core.messages import ToolCall
from agentic.core.session import SessionContext
from agentic.core.tools import execute_tool_call, tool
from agentic.io.user_io import ScriptedIO
from agentic.logs.bus import EventBus


@tool
def search_hotels(
    city: Annotated[str, Field(description="City to search")],
    guests: int,
    max_price: float | None = None,
) -> list[str]:
    """Search hotels.

    Returns hotel names.
    """
    return [f"{city}-{guests}-{max_price}"]


@tool
async def whoami(question: str, ctx: SessionContext) -> str:
    """Async tool that uses the injected session context."""
    return f"{ctx.session_id}:{question}"


def _ctx() -> SessionContext:
    return SessionContext(pattern="test", bus=EventBus(), io=ScriptedIO())


def test_schema_from_type_hints_and_docstring() -> None:
    spec = search_hotels.spec
    assert spec.name == "search_hotels"
    assert spec.description == "Search hotels.\n\nReturns hotel names."
    params = spec.parameters
    assert params["required"] == ["city", "guests"]
    assert params["properties"]["city"]["description"] == "City to search"
    assert params["properties"]["guests"]["type"] == "integer"
    assert params["additionalProperties"] is False


def test_context_parameter_is_hidden_from_the_model() -> None:
    assert list(whoami.spec.parameters["properties"]) == ["question"]


def test_tool_stays_directly_callable() -> None:
    assert search_hotels("Lisbon", 3) == ["Lisbon-3-None"]


async def test_async_tool_receives_context() -> None:
    ctx = _ctx()
    call = ToolCall(id="c1", name="whoami", arguments='{"question": "hi"}')
    record = await execute_tool_call(call, {"whoami": whoami}, ctx, agent="a")
    assert record.ok
    assert record.result == f"{ctx.session_id}:hi"


async def test_bad_calls_become_error_results() -> None:
    tools = {"search_hotels": search_hotels}
    cases = {
        "missing": ToolCall(id="1", name="search_hotels", arguments='{"city": "Lisbon"}'),
        "malformed": ToolCall(id="2", name="search_hotels", arguments="{not json"),
        "unknown": ToolCall(id="3", name="book_hotel", arguments="{}"),
    }
    for label, call in cases.items():
        record = await execute_tool_call(call, tools, _ctx(), agent="a")
        assert not record.ok, label
        assert "error" in json.loads(record.result), label
