from rich.console import Console

from agentic.core.session import SessionContext
from agentic.io.user_io import ScriptedIO
from agentic.logs.bus import EventBus
from agentic.logs.console import ConsoleRenderer
from agentic.logs.events import EventType as E


def _render(*, verbose: bool) -> str:
    console = Console(record=True, width=120, force_terminal=False)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console, verbose=verbose))
    ctx = SessionContext(pattern="demo", bus=bus, io=ScriptedIO())
    ctx.start()
    ctx.emit(E.AGENT_START, agent="hotel", run_id="r", emoji="🏨", input="find")
    ctx.emit(E.TOOL_CALL, agent="hotel", run_id="r", call_id="c", tool="search", args="{}")
    ctx.emit(
        E.TOOL_RESULT, agent="hotel", run_id="r", call_id="c", result="[red]x[/red]", ok=True, ms=1
    )
    ctx.emit(E.AGENT_END, agent="hotel", run_id="r", output="ok", tokens=15, ms=1200)
    ctx.end()
    return console.export_text()


def test_default_view_shows_steps_and_summary() -> None:
    text = _render(verbose=False)
    assert "🏨 hotel" in text
    assert "🔧 search" in text
    assert "Session summary" in text
    assert "[red]x[/red]" not in text


def test_verbose_shows_tool_results_without_interpreting_markup() -> None:
    assert "[red]x[/red]" in _render(verbose=True)
