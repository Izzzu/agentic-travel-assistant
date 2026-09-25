from pathlib import Path

from rich.console import Console

from agentic.cli import chat
from agentic.core.messages import LLMResponse
from agentic.core.session import SessionContext
from agentic.io.user_io import ScriptedIO
from agentic.llm.scripted import ScriptedLLM, call_tool, reply
from agentic.logs.bus import EventBus
from agentic.logs.console import ConsoleRenderer
from agentic.logs.events import Event, EventType
from agentic.logs.recorder import SessionRecorder
from agentic.patterns.group_chat import GroupChat, ModeratorDecision, Speaker

E = EventType
STAY = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}


def pick(speaker: Speaker, reason: str) -> LLMResponse:
    decision = ModeratorDecision(
        next_speaker=speaker, reason=reason, finished=False, final_answer=None
    )
    return reply(decision.model_dump_json())


def finish(answer: str, reason: str = "Plan is complete.") -> LLMResponse:
    decision = ModeratorDecision(
        next_speaker=None, reason=reason, finished=True, final_answer=answer
    )
    return reply(decision.model_dump_json())


async def _run(tmp_path: Path, pattern: GroupChat, inputs: list[str]) -> tuple[Path, str]:
    console = Console(record=True, width=200, force_terminal=False)
    recorder = SessionRecorder(tmp_path)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console))
    bus.subscribe(recorder)
    ctx = SessionContext(pattern=pattern.name, bus=bus, io=ScriptedIO(inputs=inputs))
    folder = recorder.folder(ctx.session_id)
    await chat(pattern, ctx, prompt="you>", console=console, folder=folder)
    return folder, console.export_text()


def _events(folder: Path, type: EventType) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(Event.model_validate_json, lines) if e.type is type]


async def test_moderator_picks_speakers_who_see_the_shared_chat(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            pick("hotel", "Find a hotel for 3."),
            call_tool("check_availability", STAY),
            reply("Hotel: Casa Alfama sold out; Baixa Boutique, CHF 700."),
            pick("budget", "Check the total with Baixa Boutique."),
            reply("Budget: CHF 1,700, CHF 200 over."),
            pick("activities", "Replace the paid tour with something free."),
            reply("Activities: free beach day at Carcavelos."),
            finish("Plan within budget."),
        ]
    )

    folder, output = await _run(tmp_path, GroupChat.create(llm), ["Plan Lisbon 9-11 Oct"])

    decisions = _events(folder, E.SPEAKER_SELECTED)
    assert [e.payload["next_speaker"] for e in decisions] == ["hotel", "budget", "activities", None]
    assert [e.payload["round"] for e in decisions] == [1, 2, 3, 4]
    assert decisions[-1].payload["finished"] and not decisions[-1].payload["forced"]
    assert [e.agent for e in _events(folder, E.AGENT_END)] == [
        "consultant",
        "hotel",
        "consultant",
        "budget",
        "consultant",
        "activities",
        "consultant",
    ]
    assert [f is ModeratorDecision for f in llm.formats_seen] == [
        True, False, False, True, False, True, False, True
    ]  # fmt: skip
    budget_brief = llm.requests[4][-1].content or ""
    assert budget_brief.startswith("Plan Lisbon 9-11 Oct")
    assert "**🧳 consultant → 🏨 hotel:** Find a hotel for 3." in budget_brief
    assert "Baixa Boutique, CHF 700" in budget_brief
    assert "## Your turn, 💰 budget\nThe moderator picked you: Check the total" in budget_brief
    last_moderator_brief = llm.requests[7][-1].content or ""
    assert "Carcavelos" in last_moderator_brief and "## Round 4/10" in last_moderator_brief
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Plan within budget."
    assert "🎤 round 2/10 · next: 💰 budget  Check the total" in output
    assert "🏁 round 4/10 · finished  Plan is complete." in output
    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert "🎤 **Round 1/10 · Next: 🏨 hotel**: Find a hotel for 3." in transcript
    assert "🏁 **Round 4/10 · Finished**: Plan is complete." in transcript


async def test_moderator_is_forced_to_summarize_at_the_round_limit(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            pick("flight", "Find flights."),
            reply("Flight: TP941."),
            pick("budget", "Price it."),
            reply("Budget: over."),
            finish("Over-budget plan with warning.", "Round limit reached."),
        ]
    )

    folder, output = await _run(tmp_path, GroupChat.create(llm, max_rounds=2), ["Plan Lisbon"])

    decisions = _events(folder, E.SPEAKER_SELECTED)
    assert [e.payload["next_speaker"] for e in decisions] == ["flight", "budget", None]
    assert decisions[-1].payload["forced"] and decisions[-1].payload["round"] == 2
    assert "## Round limit reached" in (llm.requests[-1][-1].content or "")
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Over-budget plan with warning."
    assert "forced to finish" in output


async def test_moderator_can_answer_without_specialists(tmp_path: Path) -> None:
    llm = ScriptedLLM([finish("Hi! Where would you like to go?", "Greeting.")])

    folder, _ = await _run(tmp_path, GroupChat.create(llm), ["hello"])

    assert [e.agent for e in _events(folder, E.AGENT_END)] == ["consultant"]
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"].startswith("Hi!")


async def test_follow_up_turn_sees_the_conversation(tmp_path: Path) -> None:
    llm = ScriptedLLM([finish("first answer"), finish("second answer")])

    await _run(tmp_path, GroupChat.create(llm), ["first", "and then?"])

    assert [(m.role, m.content) for m in llm.requests[1][1:3]] == [
        ("user", "first"),
        ("assistant", "first answer"),
    ]


async def test_invalid_decisions_fail_the_turn(tmp_path: Path) -> None:
    undecided = ModeratorDecision(next_speaker=None, reason="?", finished=False, final_answer=None)
    llm = ScriptedLLM([reply("not json"), reply(undecided.model_dump_json())])

    folder, output = await _run(tmp_path, GroupChat.create(llm), ["Plan", "Plan again"])

    errors = [e.payload["exception"] for e in _events(folder, E.ERROR)]
    assert len(errors) == 2 and "invalid decision" in errors[0]
    assert "no next speaker" in errors[1]
    assert output.count("Turn failed") == 2
    assert not _events(folder, E.FINAL_ANSWER)


def test_only_the_moderator_can_ask_the_user() -> None:
    pattern = GroupChat.create(ScriptedLLM([]))
    assert set(pattern.participants) == {"flight", "hotel", "activities", "budget"}
    assert all("ask_user" not in {t.name for t in a.tools} for a in pattern.participants.values())
    assert "ask_user" in {t.name for t in pattern.moderator.tools}


def test_decision_schema_is_strict_compatible() -> None:
    schema = ModeratorDecision.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
