import json
from itertools import cycle
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic.logs.events import Event, EventType
from agentic.logs.stats import SessionStats

PALETTE = ("cyan", "green", "magenta", "yellow", "blue", "bright_red", "bright_cyan")
PREVIEW = 200


def _clip(text: str, limit: int = PREVIEW) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _seconds(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f}s"


class ConsoleRenderer:
    """Live view of a session; all model and user text is rendered as plain text, not markup."""

    def __init__(self, console: Console | None = None, *, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self._colors = cycle(PALETTE)
        self._styles: dict[str, tuple[str, str]] = {}
        self._stats = SessionStats()

    def _style(self, agent: str | None, emoji: str = "") -> tuple[str, str]:
        if agent is None:
            return "white", ""
        if agent not in self._styles:
            self._styles[agent] = (next(self._colors), emoji)
        return self._styles[agent]

    def _label(self, agent: str | None) -> Text:
        color, emoji = self._style(agent)
        return Text(f"{emoji} {agent}".strip(), style=f"bold {color}")

    def _line(self, agent: str | None, *parts: str | tuple[str, str], indent: int = 0) -> None:
        self.console.print(Text.assemble(" " * indent, self._label(agent), " ", *parts))

    def __call__(self, event: Event) -> None:
        self._stats.update(event)
        p: dict[str, Any] = event.payload
        match event.type:
            case EventType.SESSION_START:
                self.console.rule(Text(f"{event.pattern} · {event.session_id}", style="bold"))
            case EventType.AGENT_START:
                self._style(event.agent, str(p.get("emoji", "")))
                self._line(event.agent, ("▸ working…", "dim"))
            case EventType.LLM_CALL if self.verbose:
                stats = (
                    f"🧠 {p.get('model')} · {p.get('prompt_tokens', 0)}→"
                    f"{p.get('completion_tokens', 0)} tok · {p.get('ms', 0)} ms"
                )
                self._line(event.agent, (stats, "dim"), indent=2)
            case EventType.TOOL_CALL:
                args = f" {_clip(str(p.get('args', '')))}" if self.verbose else ""
                self._line(event.agent, f"🔧 {p.get('tool')}", (args, "dim"), indent=2)
            case EventType.TOOL_RESULT if self.verbose or not p.get("ok", True):
                style = "dim" if p.get("ok", True) else "red"
                result = _clip(str(p.get("result", "")))
                self._line(event.agent, (f"↳ {result} ({p.get('ms', 0)} ms)", style), indent=4)
            case EventType.AGENT_END:
                self._line(
                    event.agent,
                    (
                        f"✓ done · {_seconds(float(p.get('ms', 0)))} · {p.get('tokens', 0)} tok",
                        "dim",
                    ),
                )
                if self.verbose and p.get("output"):
                    self.console.print(Text(_clip(str(p["output"]), 600), style="dim"))
            case EventType.HANDOFF:
                self.console.print(
                    Text.assemble(
                        ("↪ ", "bold"),
                        self._label(str(p.get("from_agent"))),
                        " → ",
                        self._label(str(p.get("to_agent"))),
                        (f"  {p.get('reason', '')}", "italic"),
                    )
                )
            case EventType.SPEAKER_SELECTED:
                self.console.print(
                    Text.assemble(
                        "🎤 next: ",
                        self._label(str(p.get("next_speaker"))),
                        (f"  {p.get('reason', '')}", "italic"),
                    )
                )
            case EventType.LEDGER_UPDATE:
                self.console.print(
                    Panel(Text(json.dumps(p, indent=2)), title="📒 ledger", border_style="blue")
                )
            case EventType.SURPRISE:
                self.console.print(
                    Panel(
                        Text(str(p.get("description", "")), style="bold"),
                        title="⚡ surprise",
                        border_style="bold yellow",
                    )
                )
            case EventType.ERROR:
                self._line(event.agent, (f"✗ {p.get('exception', '')}", "bold red"))
            case EventType.FINAL_ANSWER:
                self.console.print(
                    Panel(Markdown(str(p.get("text", ""))), title="✅ answer", border_style="green")
                )
            case EventType.SESSION_END:
                self.console.print(self.summary_table())
            case _:
                pass

    def summary_table(self) -> Table:
        stats = self._stats
        total = stats.total
        table = Table(
            title=f"Session summary · wall clock {_seconds(stats.wall_ms)}"
            f" · agent time {_seconds(total.ms)}",
            title_justify="left",
        )
        for column in ("agent", "runs", "time", "LLM calls", "tool calls", "tokens"):
            table.add_column(column, justify="left" if column == "agent" else "right")
        for name, s in stats.agents.items():
            table.add_row(
                self._label(name),
                str(s.runs),
                _seconds(s.ms),
                str(s.llm_calls),
                str(s.tool_calls),
                str(s.total_tokens),
            )
        table.add_row(
            Text("total", style="bold"),
            str(total.runs),
            _seconds(total.ms),
            str(total.llm_calls),
            str(total.tool_calls),
            str(total.total_tokens),
            end_section=True,
        )
        return table
