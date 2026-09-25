from collections.abc import Sequence
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
from agentic.patterns.magentic import Magentic, ProgressLedger, Speaker, TaskLedger

E = EventType
STAY = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}
FACTS = ("3 travellers, 9-11 Oct, CHF 1,500",)


def ledger(
    plan: list[str], facts: Sequence[str] = FACTS, assumptions: Sequence[str] = ()
) -> LLMResponse:
    entry = TaskLedger(facts=list(facts), assumptions=list(assumptions), plan=plan)
    return reply(entry.model_dump_json())


def progress(
    speaker: Speaker = "flight",
    instruction: str = "",
    *,
    reason: str = "Working.",
    done: bool = False,
    moving: bool = True,
    loop: bool = False,
    replan: bool = False,
) -> LLMResponse:
    entry = ProgressLedger(
        reason=reason,
        is_request_satisfied=done,
        is_in_loop=loop,
        is_progress_being_made=moving,
        needs_replan=replan,
        next_speaker=speaker,
        instruction=instruction,
    )
    return reply(entry.model_dump_json())


async def _run(
    tmp_path: Path, pattern: Magentic, inputs: list[str], answers: list[str] | None = None
) -> tuple[Path, str]:
    console = Console(record=True, width=200, force_terminal=False)
    recorder = SessionRecorder(tmp_path)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console))
    bus.subscribe(recorder)
    ctx = SessionContext(pattern=pattern.name, bus=bus, io=ScriptedIO(answers or [], inputs=inputs))
    folder = recorder.folder(ctx.session_id)
    await chat(pattern, ctx, prompt="you>", console=console, folder=folder)
    return folder, console.export_text()


def _events(folder: Path, type: EventType) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(Event.model_validate_json, lines) if e.type is type]


async def test_surprise_triggers_a_visible_replan(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            ledger(
                ["hotel: check Casa Alfama", "budget: check the total"],
                assumptions=["Casa Alfama is available"],
            ),
            progress("hotel", "Check Casa Alfama for 9-11 Oct."),
            call_tool("check_availability", STAY),
            reply("Casa Alfama sold out; Baixa Boutique CHF 700."),
            progress(reason="Casa Alfama is sold out.", replan=True),
            ledger(
                ["budget: check Baixa Boutique with a free beach day"],
                facts=[*FACTS, "Casa Alfama sold out 9-10 Oct"],
            ),
            progress("budget", "Check Baixa Boutique with a free beach day."),
            reply("Budget: CHF 1,480, within budget."),
            progress(reason="Budget confirmed.", done=True),
            call_tool("ask_user", {"question": "Baixa or Graça?"}),
            reply("Plan within budget."),
        ]
    )

    folder, output = await _run(tmp_path, Magentic.create(llm), ["Plan Lisbon 9-11 Oct"], ["Baixa"])

    updates = _events(folder, E.LEDGER_UPDATE)
    assert [(e.payload["ledger"], e.payload.get("action")) for e in updates] == [
        ("task", None),
        ("progress", "delegate"),
        ("progress", "replan"),
        ("task", None),
        ("progress", "delegate"),
        ("progress", "finish"),
    ]
    first, replanned = updates[0].payload, updates[3].payload
    assert first["version"] == 1 and first["added"] == {"facts": [], "assumptions": [], "plan": []}
    assert replanned["version"] == 2
    assert replanned["reason"] == "The plan no longer works: Casa Alfama is sold out."
    assert replanned["added"]["facts"] == ["Casa Alfama sold out 9-10 Oct"]
    assert replanned["removed"]["assumptions"] == ["Casa Alfama is available"]
    assert [e.agent for e in _events(folder, E.AGENT_END)] == [
        "consultant",
        "consultant",
        "hotel",
        "consultant",
        "consultant",
        "consultant",
        "budget",
        "consultant",
        "consultant",
    ]
    assert [f is not None for f in llm.formats_seen] == [
        True, True, False, False, True, True, True, False, True, False, False
    ]  # fmt: skip
    budget_brief = llm.requests[7][-1].content or ""
    assert budget_brief.startswith("Plan Lisbon 9-11 Oct")
    assert "## Task ledger v2" in budget_brief and "- Casa Alfama sold out 9-10 Oct" in budget_brief
    assert "**Step 1 · 🧳 consultant → 🏨 hotel:** Check Casa Alfama" in budget_brief
    assert "## Your task, 💰 budget\nThe manager assigned you: Check Baixa" in budget_brief
    assert "## Final answer" in (llm.requests[9][-1].content or "")
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Plan within budget."
    assert _events(folder, E.USER_REPLY)[0].payload["answer"] == "Baixa"
    assert "📒 task ledger v2 · re-plan" in output
    assert "+ Casa Alfama sold out 9-10 Oct" in output and "− Casa Alfama is available" in output
    assert "→ 🏨 hotel  Check Casa Alfama for 9-11 Oct." in output
    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert "### 📒 Task ledger v2 · re-plan" in transcript
    assert "- ➕ Casa Alfama sold out 9-10 Oct" in transcript
    assert "- ~~Casa Alfama is available~~" in transcript
    assert "→ **🏨 hotel**: Check Casa Alfama for 9-11 Oct." in transcript
    assert "📊 **Step 3/15 · progress ledger** · satisfied ✗" in transcript


async def test_stalling_triggers_a_replan(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            ledger(["flight: find flights"]),
            progress("flight", "Find flights."),
            reply("TP941."),
            progress("flight", "Find flights again.", moving=False),
            reply("TP941."),
            progress("flight", "Find flights again.", moving=False, loop=True, reason="Looping."),
            ledger(["budget: check the total"]),
            progress(done=True),
            reply("Plan."),
        ]
    )

    folder, _ = await _run(tmp_path, Magentic.create(llm), ["Plan Lisbon"])

    progress_updates = [e.payload for e in _events(folder, E.LEDGER_UPDATE)][1:4]
    assert [(p["stalls"], p["action"]) for p in progress_updates] == [
        (0, "delegate"),
        (1, "delegate"),
        (2, "replan"),
    ]
    replanned = _events(folder, E.LEDGER_UPDATE)[4].payload
    assert replanned["reason"] == "No progress for 2 steps: Looping."
    assert replanned["added"]["plan"] == ["budget: check the total"]


async def test_step_limit_forces_the_final_answer(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            ledger(["flight: find flights", "budget: check"]),
            progress("flight", "Find flights."),
            reply("TP941."),
            reply("Partial plan with warning."),
        ]
    )

    folder, output = await _run(tmp_path, Magentic.create(llm, max_steps=1), ["Plan Lisbon"])

    assert "## Step limit reached" in (llm.requests[-1][-1].content or "")
    assert _events(folder, E.FINAL_ANSWER)[0].payload["text"] == "Partial plan with warning."
    assert "📊 progress ledger · step 1/1" in output


async def test_empty_plan_replies_without_specialists(tmp_path: Path) -> None:
    llm = ScriptedLLM([ledger([]), reply("Hi!"), ledger([]), reply("Still here.")])

    folder, _ = await _run(tmp_path, Magentic.create(llm), ["hello", "you there?"])

    assert {e.agent for e in _events(folder, E.AGENT_END)} == {"consultant"}
    assert "## Reply" in (llm.requests[1][-1].content or "")
    assert [(m.role, m.content) for m in llm.requests[2][1:3]] == [
        ("user", "hello"),
        ("assistant", "Hi!"),
    ]


async def test_invalid_ledger_fails_the_turn(tmp_path: Path) -> None:
    llm = ScriptedLLM([reply("not json")])

    folder, output = await _run(tmp_path, Magentic.create(llm), ["Plan"])

    errors = [e.payload["exception"] for e in _events(folder, E.ERROR)]
    assert len(errors) == 1 and "invalid TaskLedger" in errors[0]
    assert "Turn failed" in output
    assert not _events(folder, E.FINAL_ANSWER)


def test_only_the_manager_can_ask_the_user() -> None:
    pattern = Magentic.create(ScriptedLLM([]))
    assert set(pattern.specialists) == {"flight", "hotel", "activities", "budget"}
    assert all("ask_user" not in {t.name for t in a.tools} for a in pattern.specialists.values())
    assert "ask_user" in {t.name for t in pattern.manager.tools}


def test_ledger_schemas_are_strict_compatible() -> None:
    for model in (TaskLedger, ProgressLedger):
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
