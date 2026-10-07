import json

import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from agentic_autogen.model import MODEL_INFO
from agentic_autogen.patterns.magentic import Magentic

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


def call(name: str, arguments: str) -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=f"call-{name}", name=name, arguments=arguments)],
        usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
        cached=False,
    )


def ledger(*, done: bool = False, speaker: str = "hotel", instruction: str = "Check it.") -> str:
    def answer(value: object) -> dict[str, object]:
        return {"reason": "test", "answer": value}

    return json.dumps(
        {
            "is_request_satisfied": answer(done),
            "is_progress_being_made": answer(True),
            "is_in_loop": answer(False),
            "instruction_or_question": answer(instruction),
            "next_speaker": answer(speaker),
        }
    )


def team(
    replies: list[str | CreateResult], **options: int
) -> tuple[Magentic, ReplayChatCompletionClient]:
    model = ReplayChatCompletionClient(replies, model_info=MODEL_INFO)
    return Magentic.create(model, **options), model


def one_hotel_step(final: str) -> list[str | CreateResult]:
    return [
        "Facts: the traveller wants a hotel.",
        "Plan: ask hotel, then answer.",
        ledger(speaker="hotel", instruction="Check Casa Alfama for 9-11 October."),
        call("check_availability", STAY),
        "Casa Alfama is sold out.",
        ledger(done=True),
        final,
    ]


def test_the_team_is_the_four_specialists_and_the_traveller() -> None:
    magentic, _ = team([])
    participants = magentic.team._participants  # pyright: ignore[reportPrivateUsage]

    assert [p.name for p in participants] == [
        "flight",
        "hotel",
        "activities",
        "budget",
        "traveller",
    ]
    assert all(isinstance(p, AssistantAgent) for p in participants[:4])
    assert isinstance(participants[4], UserProxyAgent)


async def test_the_orchestrator_delegates_and_writes_the_final_answer() -> None:
    magentic, _ = team(one_hotel_step("Casa Alfama is full; try Baixa."))

    result = await magentic.turn("Is Casa Alfama free 9-11 October?")

    sources = [m.source for m in result.messages]
    assert "hotel" in sources
    assert "sold_out" in str(result.messages)
    last = result.messages[-1]
    assert last.source == "MagenticOneOrchestrator"
    assert last.to_text() == "Casa Alfama is full; try Baixa."
    assert magentic.history == [
        ("Is Casa Alfama free 9-11 October?", "Casa Alfama is full; try Baixa.")
    ]


async def test_a_follow_up_carries_the_earlier_conversation_in_the_task() -> None:
    magentic, _ = team([*one_hotel_step("Casa Alfama is full."), *one_hotel_step("Baixa is free.")])
    await magentic.turn("Is Casa Alfama free 9-11 October?")

    result = await magentic.turn("What about Baixa?")

    task = result.messages[0].to_text()
    assert "Traveller: Is Casa Alfama free 9-11 October?" in task
    assert "Team: Casa Alfama is full." in task
    assert "The traveller's new message: What about Baixa?" in task


async def test_reset_forgets_the_conversation() -> None:
    magentic, _ = team(one_hotel_step("Casa Alfama is full."))
    await magentic.turn("Is Casa Alfama free?")

    await magentic.reset()

    assert magentic.history == []
    task = magentic.task("Hello")
    assert task.startswith("Hello\n\nNotes for the manager:")
    assert "Today is" in task


async def test_the_turn_limit_ends_the_run_with_a_final_answer() -> None:
    magentic, _ = team(
        [
            "Facts.",
            "Plan.",
            ledger(speaker="hotel"),
            "Still checking.",
            ledger(speaker="hotel"),
            "Still checking again.",
            "Stopped at the limit.",
        ],
        max_turns=1,
    )

    result = await magentic.turn("Plan my trip")

    assert result.stop_reason == "Max rounds reached."
    assert result.messages[-1].to_text() == "Stopped at the limit."
