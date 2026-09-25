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
from agentic.patterns.handoff import HANDOFF_GRAPH, PREFIX, Handoff

E = EventType
STAY = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}


def hand(to: str, reason: str) -> LLMResponse:
    return call_tool(f"{PREFIX}{to}", {"reason": reason})


async def _run(
    tmp_path: Path, pattern: Handoff, inputs: list[str], answers: list[str] | None = None
) -> tuple[Path, str, ScriptedIO]:
    console = Console(record=True, width=200, force_terminal=False)
    recorder = SessionRecorder(tmp_path)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console))
    bus.subscribe(recorder)
    io = ScriptedIO(answers or [], inputs=inputs)
    ctx = SessionContext(pattern=pattern.name, bus=bus, io=io)
    folder = recorder.folder(ctx.session_id)
    await chat(pattern, ctx, prompt=pattern.prompt, console=console, folder=folder)
    return folder, console.export_text(), io


def _events(folder: Path, type: EventType) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(Event.model_validate_json, lines) if e.type is type]


async def test_surprise_travels_through_the_team_back_to_the_consultant(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            hand("flight", "Find flights ZRH-LIS 9-11 Oct for 3."),
            hand("hotel", "Flights TP941/TP944 chosen; find a hotel."),
            call_tool("check_availability", STAY),
            hand("budget", "Casa Alfama is sold out; check Baixa Boutique."),
            hand("consultant", "CHF 200 over budget; the traveller must decide."),
            call_tool("ask_user", {"question": "Go over budget or change area?"}),
            hand("hotel", "Find a cheaper area."),
            reply("Graça Guesthouse is free and fits the budget."),
            reply("Booked Graça."),
        ]
    )

    folder, output, io = await _run(
        tmp_path,
        Handoff.create(llm),
        ["Plan Lisbon 9-11 Oct for 3", "Book it"],
        answers=["Change area"],
    )

    handoffs = [
        (e.payload["from_agent"], e.payload["to_agent"]) for e in _events(folder, E.HANDOFF)
    ]
    assert handoffs == [
        ("consultant", "flight"),
        ("flight", "hotel"),
        ("hotel", "budget"),
        ("budget", "consultant"),
        ("consultant", "hotel"),
    ]
    assert io.asked == [("consultant", "Go over budget or change area?")]
    assert [e.payload["text"] for e in _events(folder, E.FINAL_ANSWER)] == [
        "Graça Guesthouse is free and fits the budget.",
        "Booked Graça.",
    ]
    assert io.prompts == ["[🧳 consultant] you>", "[🏨 hotel] you>", "[🏨 hotel] you>"]
    budget_request = llm.requests[4]
    assert budget_request[0].content and "Handoffs" in budget_request[0].content
    assert any(m.role == "tool" and "sold_out" in (m.content or "") for m in budget_request)
    assert (
        budget_request[-1].role == "tool" and budget_request[-1].content == "💰 budget takes over."
    )
    second_turn = llm.requests[-1]
    assert second_turn[-1].content == "Book it"
    assert second_turn[-2].content == "Graça Guesthouse is free and fits the budget."
    assert "↪ 🏨 hotel → 💰 budget  Casa Alfama is sold out" in output
    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert "↪️ **💰 budget → 🧳 consultant**: CHF 200 over budget" in transcript


async def test_handoffs_only_follow_graph_edges(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            hand("budget", "Check the total."),
            hand("hotel", "Find something cheaper."),
            reply("I can only hand back to the consultant."),
        ]
    )

    folder, _, _ = await _run(tmp_path, Handoff.create(llm), ["Is this in budget?"])

    assert [e.payload["to_agent"] for e in _events(folder, E.HANDOFF)] == ["budget"]
    [result] = [e for e in _events(folder, E.TOOL_RESULT) if e.agent == "budget"]
    assert not result.payload["ok"] and "unknown tool" in result.payload["result"]
    for name, targets in HANDOFF_GRAPH.items():
        tools = {t.name for t in Handoff.create(llm).agents[name].tools if t.ends_run}
        assert tools == {f"{PREFIX}{t}" for t in targets}


async def test_handoff_limit_forces_a_reply(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            hand("hotel", "Find a hotel."),
            hand("consultant", "Back to you."),
            reply("Here is what we have so far."),
        ]
    )

    folder, _, _ = await _run(tmp_path, Handoff.create(llm, max_handoffs=2), ["Plan Lisbon"])

    assert len(_events(folder, E.HANDOFF)) == 2
    assert "## Handoff limit reached" in (llm.requests[-1][-1].content or "")
    assert not any(t.name.startswith(PREFIX) for t in llm.tools_seen[-1])
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Here is what we have so far."


async def test_reset_returns_to_the_consultant(tmp_path: Path) -> None:
    llm = ScriptedLLM([hand("flight", "Find flights."), reply("TP941."), reply("Hi again!")])

    _, _, io = await _run(tmp_path, Handoff.create(llm), ["Flights?", "/reset", "hello"])

    assert io.prompts == [
        "[🧳 consultant] you>",
        "[✈️ flight] you>",
        "[🧳 consultant] you>",
        "[🧳 consultant] you>",
    ]
    assert llm.requests[-1][0].content and "front door" in llm.requests[-1][0].content


def test_only_the_consultant_can_ask_the_user() -> None:
    pattern = Handoff.create(ScriptedLLM([]))
    owners = {n for n, a in pattern.agents.items() if "ask_user" in {t.name for t in a.tools}}
    assert owners == {"consultant"}
