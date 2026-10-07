import json

import pytest

pytest.importorskip("autogen_agentchat")

from autogen_core import CancellationToken

from agentic.tools.hotel import search_hotels
from agentic_autogen.tools import ToolBridge, bridge

STAY = {"city": "Lisbon", "checkin": "2026-10-09", "checkout": "2026-10-11", "guests": 3}


def test_bridge_exposes_the_schema_of_the_agentic_tool() -> None:
    (tool,) = bridge([search_hotels])

    assert tool.name == "search_hotels"
    assert tool.description == search_hotels.description
    properties = tool.schema.get("parameters", {}).get("properties", {})
    assert set(properties) == {"city", "checkin", "checkout", "guests", "max_price"}
    assert properties["city"]["description"] == "City, e.g. Lisbon"


async def test_bridge_runs_the_tool_and_returns_json() -> None:
    tool = ToolBridge(search_hotels)

    result = await tool.run_json(STAY, CancellationToken())

    assert json.loads(tool.return_value_as_string(result))["currency"] == "CHF"


async def test_tool_failures_propagate_for_autogen_to_report() -> None:
    tool = ToolBridge(search_hotels)

    with pytest.raises(ValueError, match="no mock hotels"):
        await tool.run_json({**STAY, "city": "Paris"}, CancellationToken())
