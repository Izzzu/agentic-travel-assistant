import pytest

pytest.importorskip("agent_framework")

from typing import Any

from maf_fakes import Call, scripted_client

from agentic_maf.agents import SPECIALISTS, create_specialist

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


def tool_names(request: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in request["tools"]}


async def test_agent_runs_its_tools_and_keeps_going_until_it_has_an_answer() -> None:
    client, sdk = scripted_client([Call("check_availability", STAY), "Casa Alfama is full."])
    agent = create_specialist("hotel", client)

    result = await agent.run("Is Casa Alfama free 9-11 October?")

    assert result.text == "Casa Alfama is full."
    assert "sold_out" in str(sdk.requests[1]["messages"])


async def test_instructions_and_description_describe_the_specialist() -> None:
    client, sdk = scripted_client(["ok"])
    hotel = create_specialist("hotel", client, rules="## Extra\nHand back.")

    await hotel.run("Hello")

    instructions = str(sdk.requests[0]["messages"][0]["content"])
    assert hotel.description == "accommodation and availability"
    assert "You are the Hotel specialist" in instructions
    assert "Always confirm your pick with check_availability" in instructions
    assert instructions.endswith("## Extra\nHand back.")


async def test_every_specialist_has_its_own_tools() -> None:
    expected = {
        "flight": {"search_flights", "get_flight_details"},
        "hotel": {"search_hotels", "check_availability"},
        "activities": {"search_activities", "get_opening_hours"},
        "budget": {"calculate_total", "check_budget", "suggest_savings"},
    }

    assert set(SPECIALISTS) == set(expected)
    for name, tools in expected.items():
        client, sdk = scripted_client(["ok"])
        await create_specialist(name, client).run("Hello")
        assert tool_names(sdk.requests[0]) == tools
