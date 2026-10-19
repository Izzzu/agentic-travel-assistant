import pytest

pytest.importorskip("agent_framework")

from typing import Any

from maf_fakes import Call, Reply, ScriptedOpenAI, scripted_client

from agentic_maf.patterns.handoff import ENTRY, GRAPH, Handoff

STAY = '{"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}'


def team(replies: list[Reply]) -> tuple[Handoff, ScriptedOpenAI]:
    client, sdk = scripted_client(replies)
    return Handoff.create(client), sdk


def tool_names(request: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in request["tools"]}


async def test_control_moves_through_the_team_until_someone_replies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff, sdk = team(
        [
            Call("handoff_to_hotel"),
            Call("check_availability", STAY),
            Call("handoff_to_consultant"),
            "Casa Alfama is full; Baixa Boutique is free.",
        ]
    )

    await handoff.turn("A hotel for 9-11 October")

    out = capsys.readouterr().out
    assert ["hotel", "consultant"] == [  # a handoff-only turn prints no header
        line.strip("- ") for line in out.splitlines() if line.startswith("--- ")
    ]
    assert "⇢ handoff: consultant → hotel" in out
    assert "⚙ check_availability(" in out
    assert "sold_out" in out
    assert "Baixa Boutique is free." in out
    assert handoff.holder == ENTRY
    assert handoff.prompt == "[consultant] you>"
    assert sdk.unused == 0


async def test_a_specialist_reply_without_a_handoff_keeps_control_for_the_next_message() -> None:
    handoff, sdk = team(
        [
            Call("handoff_to_budget"),
            "Over budget by CHF 200; the traveller must decide.",
            "Cheaper area it is.",
        ]
    )
    await handoff.turn("Check the budget")
    assert handoff.holder == "budget"
    assert handoff.prompt == "[budget] you>"

    await handoff.turn("Change area")

    assert "Change area" in str(sdk.requests[-1]["messages"])
    assert handoff.holder == "budget"
    assert sdk.unused == 0


async def test_reset_returns_to_the_consultant() -> None:
    handoff, _ = team([Call("handoff_to_flight"), "Which dates?"])
    await handoff.turn("Plan a trip")
    assert handoff.holder == "flight"

    await handoff.reset()

    assert handoff.pending is None
    assert handoff.prompt == "[consultant] you>"


async def test_nobody_can_hand_off_to_the_user_and_the_graph_limits_the_targets() -> None:
    handoff, sdk = team([Call("handoff_to_hotel"), Call("handoff_to_consultant"), "Done."])

    await handoff.turn("Plan a trip")

    consultant, hotel = sdk.requests[0], sdk.requests[1]
    assert tool_names(consultant) == {f"handoff_to_{n}" for n in GRAPH[ENTRY]}
    assert {n for n in tool_names(hotel) if n.startswith("handoff_to_")} == {
        f"handoff_to_{n}" for n in GRAPH["hotel"]
    }
    assert all("handoff_to_user" not in tool_names(r) for r in sdk.requests)


async def test_the_consultant_has_no_tools_besides_handoffs() -> None:
    handoff, sdk = team(["Hello!"])

    await handoff.turn("Hi")

    assert all(n.startswith("handoff_to_") for n in tool_names(sdk.requests[0]))
    assert "ask_user" not in str(sdk.requests[0]["messages"][0]["content"])
