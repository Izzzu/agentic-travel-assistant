import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from agentic_autogen.agents import SPECIALISTS, create_specialist
from agentic_autogen.model import MODEL_INFO

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


def replay(replies: list[str | CreateResult]) -> ReplayChatCompletionClient:
    return ReplayChatCompletionClient(replies, model_info=MODEL_INFO)


def call(name: str, arguments: str) -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id="c1", name=name, arguments=arguments)],
        usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
        cached=False,
    )


async def test_agent_runs_its_tools_and_keeps_going_until_it_has_an_answer() -> None:
    model = replay([call("check_availability", STAY), "Casa Alfama is full."])
    agent = create_specialist("hotel", model)

    result = await agent.run(task="Is Casa Alfama free 9-11 October?")

    assert result.messages[-1].to_text() == "Casa Alfama is full."
    (executed,) = [m for m in result.messages if isinstance(m, ToolCallExecutionEvent)]
    assert "sold_out" in executed.content[0].content


def test_instructions_and_description_describe_the_specialist() -> None:
    hotel = create_specialist("hotel", replay([]), rules="## Extra\nHand back.")

    instructions = hotel._system_messages[0].content  # pyright: ignore[reportPrivateUsage]
    assert hotel.description == "accommodation and availability"
    assert "You are the Hotel specialist" in instructions
    assert "Always confirm your pick with check_availability" in instructions
    assert instructions.endswith("## Extra\nHand back.")


def test_every_specialist_has_its_own_tools() -> None:
    expected = {
        "flight": {"search_flights", "get_flight_details"},
        "hotel": {"search_hotels", "check_availability"},
        "activities": {"search_activities", "get_opening_hours"},
        "budget": {"calculate_total", "check_budget", "suggest_savings"},
    }

    assert set(SPECIALISTS) == set(expected)
    for name, tools in expected.items():
        agent = create_specialist(name, replay([]))
        assert {t.name for t in agent._tools} == tools  # pyright: ignore[reportPrivateUsage]
