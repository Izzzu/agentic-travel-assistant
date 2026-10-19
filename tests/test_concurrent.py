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
from agentic.patterns.concurrent import Concurrent

E = EventType
STAY = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}
BLOCKED = LLMResponse(model="scripted", finish_reason="content_filter")


async def _run(tmp_path: Path, llm: ScriptedLLM, inputs: list[str]) -> tuple[Path, str]:
    console = Console(record=True, width=200, force_terminal=False)
    recorder = SessionRecorder(tmp_path)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console))
    bus.subscribe(recorder)
    pattern = Concurrent.create(llm)
    ctx = SessionContext(pattern=pattern.name, bus=bus, io=ScriptedIO(inputs=inputs))
    folder = recorder.folder(ctx.session_id)
    await chat(pattern, ctx, prompt="you>", console=console, folder=folder)
    return folder, console.export_text()


def _events(folder: Path, type: EventType | None = None) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(Event.model_validate_json, lines) if type is None or e.type is type]


async def test_specialists_run_in_parallel_and_the_aggregator_merges(tmp_path: Path) -> None:
    # One shared script: the four first calls are answered in start order, then hotel's second.
    llm = ScriptedLLM(
        [
            reply("Flight: TP941 lands 22:00."),
            call_tool("check_availability", STAY),
            reply("Activities: food tour on day 1.\n\nOpen questions:\n- Beach on day 1 or 2?"),
            reply("Budget: nothing to price yet."),
            reply("Hotel: Casa Alfama sold out; Baixa Boutique."),
            reply("Merged plan with conflicts."),
        ]
    )

    folder, _ = await _run(tmp_path, llm, ["Plan Lisbon 9-11 Oct"])

    kinds = [(e.type, e.agent) for e in _events(folder) if e.type in (E.AGENT_START, E.AGENT_END)]
    first_end = next(i for i, (t, _) in enumerate(kinds) if t is E.AGENT_END)
    assert {a for _, a in kinds[:first_end]} == {"flight", "hotel", "activities", "budget"}
    assert kinds[-1] == (E.AGENT_END, "consultant")
    for request in llm.requests[:4]:
        assert [(m.role, m.content) for m in request[1:]] == [("user", "Plan Lisbon 9-11 Oct")]
    brief = llm.requests[5][-1].content or ""
    for text in ("TP941", "Baixa Boutique", "food tour", "nothing to price"):
        assert text in brief
    assert "- activities: Beach on day 1 or 2?" in brief
    assert "Failed" not in brief
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Merged plan with conflicts."
    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert "- 🏨 hotel · 🔧 `check_availability`" in transcript


async def test_a_failed_specialist_does_not_stop_the_others(tmp_path: Path) -> None:
    llm = ScriptedLLM([reply("F"), BLOCKED, reply("A"), reply("B"), reply("Plan without hotel.")])

    folder, _ = await _run(tmp_path, llm, ["Plan Lisbon"])

    assert [e.agent for e in _events(folder, E.ERROR)] == ["hotel"]
    brief = llm.requests[4][-1].content or ""
    assert "## Failed specialists (no output)\n- 🏨 hotel: AgentError" in brief
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Plan without hotel."


async def test_turn_fails_when_every_specialist_fails(tmp_path: Path) -> None:
    llm = ScriptedLLM([BLOCKED] * 4)

    folder, output = await _run(tmp_path, llm, ["Plan Lisbon"])

    assert len(_events(folder, E.ERROR)) == 4
    assert not _events(folder, E.FINAL_ANSWER)
    assert "Turn failed" in output


def test_only_the_aggregator_can_ask_the_user() -> None:
    pattern = Concurrent.create(ScriptedLLM([]))
    assert all("ask_user" not in {t.name for t in a.tools} for a in pattern.specialists)
    assert "ask_user" in {t.name for t in pattern.aggregator.tools}
