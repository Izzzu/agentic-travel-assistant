import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    TextMessage,
)
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from agentic_autogen.model import MODEL_INFO
from agentic_autogen.patterns.handoff import GRAPH, Handoff

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


def call(name: str, arguments: str = "{}") -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=f"call-{name}", name=name, arguments=arguments)],
        usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
        cached=False,
    )


def team(replies: list[str | CreateResult], **options: int) -> Handoff:
    model = ReplayChatCompletionClient(replies, model_info=MODEL_INFO)
    return Handoff.create(model, **options)


def speakers(messages: list[BaseAgentEvent | BaseChatMessage]) -> list[str]:
    """The agents that spoke, in order, without repeats."""
    names = [m.source for m in messages if m.source != "user"]
    return [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]


async def test_control_moves_through_the_team_and_the_consultant_reply_ends_the_turn() -> None:
    handoff = team(
        [
            call("transfer_to_hotel"),
            call("check_availability", STAY),
            call("transfer_to_consultant"),
            "Casa Alfama is full; Baixa Boutique is free.",
        ]
    )

    result = await handoff.turn("A hotel for 9-11 October")

    assert speakers(list(result.messages)) == ["consultant", "hotel", "consultant"]
    last = result.messages[-1]
    assert isinstance(last, TextMessage) and last.source == "consultant"
    assert "sold_out" in str(result.messages)
    assert handoff.holder == "consultant"
    assert handoff.prompt == "[consultant] you>"


async def test_a_specialist_message_without_a_handoff_does_not_end_the_turn() -> None:
    handoff = team(
        [
            call("transfer_to_budget"),
            "Over budget by CHF 200; the traveller must decide.",
            call("transfer_to_consultant"),
            "Go over budget or change area?",
            "Cheaper area it is.",
        ]
    )
    first = await handoff.turn("Check the budget")
    assert speakers(list(first.messages)) == ["consultant", "budget", "consultant"]
    assert handoff.holder == "consultant"

    result = await handoff.turn("Change area")

    assert result.messages[0].source == "user"
    assert result.messages[1].source == "consultant"
    assert result.messages[-1].to_text() == "Cheaper area it is."


async def test_the_message_limit_stops_the_run_and_the_target_keeps_control() -> None:
    handoff = team([call("transfer_to_flight")], max_messages=2)

    result = await handoff.turn("Plan a trip")

    last = result.messages[-1]
    assert isinstance(last, HandoffMessage) and last.target == "flight"
    assert handoff.holder == "flight"
    assert handoff.prompt == "[flight] you>"


async def test_reset_returns_to_the_consultant() -> None:
    handoff = team([call("transfer_to_flight")], max_messages=2)
    await handoff.turn("Plan a trip")

    await handoff.reset()

    assert handoff.holder is None
    assert handoff.prompt == "[consultant] you>"


def test_nobody_can_hand_off_to_the_user_and_the_consultant_speaks_first() -> None:
    participants = team([]).team._participants  # pyright: ignore[reportPrivateUsage]

    def tools(name: str) -> set[str]:
        agent = next(p for p in participants if p.name == name)
        assert isinstance(agent, AssistantAgent)
        return set(agent._handoffs)  # pyright: ignore[reportPrivateUsage]

    assert tools("consultant") == {f"transfer_to_{n}" for n in GRAPH["consultant"]}
    assert all("transfer_to_user" not in tools(n) for n in GRAPH)
    assert participants[0].name == "consultant"


def test_the_consultant_has_no_tools_and_no_ask_user_instruction() -> None:
    consultant = team([]).team._participants[0]  # pyright: ignore[reportPrivateUsage]

    assert isinstance(consultant, AssistantAgent)
    assert consultant._tools == []  # pyright: ignore[reportPrivateUsage]
    assert "ask_user" not in consultant._system_messages[0].content  # pyright: ignore[reportPrivateUsage]
