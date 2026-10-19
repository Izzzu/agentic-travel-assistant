import json
from pathlib import Path

from rich.console import Console

from agentic.agents.registry import AgentName
from agentic.cli import chat
from agentic.core.session import SessionContext
from agentic.io.user_io import ScriptedIO
from agentic.llm.scripted import ScriptedLLM, call_tool, reply
from agentic.logs.bus import EventBus
from agentic.logs.console import ConsoleRenderer
from agentic.logs.events import Event, EventType
from agentic.logs.recorder import SessionRecorder
from agentic.patterns.sequential import Sequential

E = EventType
STAY = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}


async def _run(
    tmp_path: Path, llm: ScriptedLLM, inputs: list[str], pattern: Sequential | None = None
) -> tuple[Path, str]:
    console = Console(record=True, width=200, force_terminal=False)
    recorder = SessionRecorder(tmp_path)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console))
    bus.subscribe(recorder)
    pattern = pattern or Sequential.create(llm)
    ctx = SessionContext(pattern=pattern.name, bus=bus, io=ScriptedIO(inputs=inputs))
    folder = recorder.folder(ctx.session_id)
    await chat(pattern, ctx, prompt="you>", console=console, folder=folder)
    return folder, console.export_text()


def _events(folder: Path, type: EventType) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(Event.model_validate_json, lines) if e.type is type]


async def test_runs_five_steps_in_order_and_passes_outputs_along(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            reply("Flight: TP941, CHF 600."),
            call_tool("check_availability", STAY),
            reply("Hotel: Casa Alfama sold out; Baixa Boutique, CHF 700."),
            reply("Activities: food tour, CHF 400."),
            reply("Budget: CHF 1,700, CHF 200 over."),
            reply("Plan (over budget by CHF 200)."),
        ]
    )

    folder, output = await _run(tmp_path, llm, ["Plan Lisbon 9-11 Oct"])

    steps = _events(folder, E.STEP)
    assert [(e.agent, e.payload["step"], e.payload["total"]) for e in steps] == [
        ("flight", 1, 5),
        ("hotel", 2, 5),
        ("activities", 3, 5),
        ("budget", 4, 5),
        ("consultant", 5, 5),
    ]
    assert [e.agent for e in _events(folder, E.AGENT_END)] == [e.agent for e in steps]
    assert llm.requests[0][-1].content == "Plan Lisbon 9-11 Oct"
    writer = llm.requests[5]
    assert writer[0].content and "✈️ flight → 🏨 hotel → 🎭 activities → 💰 budget → you" in (
        writer[0].content
    )
    brief = writer[-1].content or ""
    assert brief.startswith("Plan Lisbon 9-11 Oct")
    for text in ("TP941", "Baixa Boutique", "food tour", "CHF 200 over"):
        assert text in brief
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Plan (over budget by CHF 200)."
    assert "step 2/5 · 🏨 hotel" in output
    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert transcript.index("## Step 1/5 · ✈️ flight") < transcript.index(
        "## Step 5/5 · 🧳 consultant"
    )
    assert (
        json.loads((folder / "summary.json").read_text(encoding="utf-8"))["agents"]["hotel"][
            "tool_calls"
        ]
        == 1
    )


async def test_follow_up_turn_sees_the_conversation(tmp_path: Path) -> None:
    llm = ScriptedLLM([reply(f"out {i}") for i in range(4)])
    pattern = Sequential.create(llm, order=[AgentName.BUDGET])

    folder, _ = await _run(tmp_path, llm, ["first", "cheaper please"], pattern)

    assert [e.agent for e in _events(folder, E.STEP)] == ["budget", "consultant"] * 2
    assert [(m.role, m.content) for m in llm.requests[2][1:]] == [
        ("user", "first"),
        ("assistant", "out 1"),
        ("user", "cheaper please"),
    ]
