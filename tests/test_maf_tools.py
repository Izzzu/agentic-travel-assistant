import json

import pytest

pytest.importorskip("agent_framework")

from agentic.tools.hotel import search_hotels
from agentic_maf.tools import bridge

STAY = {"city": "Lisbon", "checkin": "2026-10-09", "checkout": "2026-10-11", "guests": 3}


def test_bridge_exposes_the_schema_of_the_agentic_tool() -> None:
    (tool,) = bridge([search_hotels])

    assert tool.name == "search_hotels"
    assert tool.description == search_hotels.description
    properties = tool.parameters()["properties"]
    assert set(properties) == {"city", "checkin", "checkout", "guests", "max_price"}
    assert properties["city"]["description"] == "City name"


async def test_bridge_parses_dates_and_returns_json() -> None:
    (tool,) = bridge([search_hotels])

    result = await tool.invoke(arguments=STAY)

    assert json.loads(result[0].text or "")["currency"] == "CHF"


async def test_tool_failures_propagate_for_agent_framework_to_report() -> None:
    (tool,) = bridge([search_hotels])

    with pytest.raises(ValueError, match="available cities: Lisbon"):
        await tool.invoke(arguments={**STAY, "city": "Paris"})
