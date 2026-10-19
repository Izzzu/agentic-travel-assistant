import json

import pytest

pytest.importorskip("agent_framework")

from maf_fakes import Call, Reply, ScriptedOpenAI, scripted_client

from agentic_maf.patterns.magentic import Magentic

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


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


def one_hotel_step(final: str) -> list[Reply]:
    return [
        "Facts: the traveller wants a hotel.",
        "Plan: ask hotel, then answer.",
        ledger(speaker="hotel", instruction="Check Casa Alfama for 9-11 October."),
        Call("check_availability", STAY),
        "Casa Alfama is sold out.",
        ledger(done=True),
        final,
    ]


class Traveller:
    """Stands in for the person at the terminal."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.questions: list[str] = []

    async def __call__(self, question: str) -> str:
        self.questions.append(question)
        return self.answers.pop(0)


def team(
    replies: list[Reply], traveller: Traveller | None = None, *, max_rounds: int = 15
) -> tuple[Magentic, ScriptedOpenAI]:
    client, sdk = scripted_client(replies)
    return Magentic.create(client, max_rounds=max_rounds, ask=traveller or Traveller()), sdk


async def test_the_manager_delegates_and_writes_the_final_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    magentic, sdk = team(one_hotel_step("Casa Alfama is full; try Baixa."))

    await magentic.turn("Is Casa Alfama free 9-11 October?")

    out = capsys.readouterr().out
    assert "=== Task ledger ===" in out
    assert "next: hotel · Check Casa Alfama for 9-11 October." in out
    assert "--- hotel ---" in out
    assert "sold_out" in out
    assert "satisfied=True" in out
    assert "Casa Alfama is full; try Baixa." in out
    assert magentic.history == [
        ("Is Casa Alfama free 9-11 October?", "Casa Alfama is full; try Baixa.")
    ]
    assert sdk.unused == 0


async def test_the_team_is_the_four_specialists_and_the_traveller() -> None:
    magentic, sdk = team(one_hotel_step("Done."))

    await magentic.turn("Plan my trip")

    plan_request = str(sdk.requests[1]["messages"])
    for name in ("flight", "hotel", "activities", "budget", "traveller"):
        assert f"- {name}:" in plan_request


async def test_a_follow_up_carries_the_earlier_conversation_in_the_task() -> None:
    magentic, sdk = team(
        [*one_hotel_step("Casa Alfama is full."), *one_hotel_step("Baixa is free.")]
    )
    await magentic.turn("Is Casa Alfama free 9-11 October?")
    first_requests = len(sdk.requests)

    await magentic.turn("What about Baixa?")

    task = str(sdk.requests[first_requests]["messages"])
    assert "Traveller: Is Casa Alfama free 9-11 October?" in task
    assert "Team: Casa Alfama is full." in task
    assert "The traveller's new message: What about Baixa?" in task


async def test_reset_forgets_the_conversation() -> None:
    magentic, _ = team(one_hotel_step("Casa Alfama is full."))
    await magentic.turn("Is Casa Alfama free?")

    await magentic.reset()

    assert magentic.history == []
    assert magentic.task("Hello") == "Hello"


async def test_the_traveller_answers_a_question_the_manager_asks() -> None:
    traveller = Traveller("Change area")
    magentic, sdk = team(
        [
            "Facts.",
            "Plan.",
            ledger(speaker="traveller", instruction="Go over budget or change area?"),
            ledger(done=True),
            "Cheaper area it is.",
        ],
        traveller,
    )

    await magentic.turn("Plan my trip")

    assert traveller.questions == ["Go over budget or change area?"]
    assert "Change area" in str(sdk.requests[-1]["messages"])
    assert magentic.history == [("Plan my trip", "Cheaper area it is.")]


async def test_the_round_limit_ends_the_run_with_a_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    magentic, _ = team(
        ["Facts.", "Plan.", ledger(speaker="hotel"), "Still checking."],
        max_rounds=1,
    )

    await magentic.turn("Plan my trip")

    assert "maximum round count" in capsys.readouterr().out
    assert "maximum round count" in magentic.history[0][1]
