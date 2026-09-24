import asyncio
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.text import Text

from agentic.agents.registry import AgentName, get_agent
from agentic.config import Settings
from agentic.core.session import SessionContext, new_session_id
from agentic.io.user_io import ConsoleIO
from agentic.llm.azure_openai import AzureOpenAIClient
from agentic.llm.base import LLMClient
from agentic.logs.bus import EventBus
from agentic.logs.console import ConsoleRenderer
from agentic.logs.events import EventType
from agentic.logs.recorder import SessionRecorder
from agentic.patterns.base import Pattern
from agentic.patterns.sequential import Sequential
from agentic.patterns.single import SingleAgent

COMMANDS = "/exit quit · /reset clear the conversation · /session show the session folder"


class PatternName(StrEnum):
    SEQUENTIAL = "sequential"


PATTERNS: dict[PatternName, Callable[[LLMClient], Pattern]] = {
    PatternName.SEQUENTIAL: Sequential.create,
}

app = typer.Typer(help="Agentic travel assistant: one trip, five orchestration patterns.")


async def chat(
    pattern: Pattern,
    ctx: SessionContext,
    *,
    prompt: str,
    console: Console,
    folder: Path,
    **info: Any,
) -> None:
    """The chat loop shared by every mode: read input, run a turn, repeat."""
    ctx.start(**info)
    console.print(Text(f"Commands: {COMMANDS}", style="dim"))
    try:
        while True:
            try:
                text = (await ctx.io.read(prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.startswith("/"):
                command = text.lower()
                if command in ("/exit", "/quit"):
                    break
                if command == "/reset":
                    ctx.history.clear()
                    console.print(Text("Conversation cleared.", style="dim"))
                elif command == "/session":
                    console.print(Text(str(folder.resolve())))
                else:
                    console.print(Text(f"Unknown command {text}. {COMMANDS}", style="yellow"))
                continue
            ctx.emit(EventType.USER_MESSAGE, text=text)
            try:
                answer = await pattern.handle_turn(text, ctx)
            except Exception as exc:  # noqa: BLE001 - a failed turn must not end the demo
                console.print(
                    Text(f"Turn failed ({type(exc).__name__}); try again or /reset.", style="red")
                )
                continue
            ctx.emit(EventType.FINAL_ANSWER, text=answer)
    finally:
        ctx.end()


async def _run(
    pattern: Pattern,
    llm: AzureOpenAIClient,
    *,
    label: str,
    prompt: str,
    verbose: bool,
    sessions_dir: Path,
    **info: Any,
) -> None:
    console = Console()
    recorder = SessionRecorder(sessions_dir)
    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console, verbose=verbose))
    bus.subscribe(recorder)
    ctx = SessionContext(
        pattern=pattern.name,
        bus=bus,
        io=ConsoleIO(console),
        session_id=new_session_id(label),
    )
    try:
        await chat(
            pattern,
            ctx,
            prompt=prompt,
            console=console,
            folder=recorder.folder(ctx.session_id),
            **info,
            model=llm.model,
            verbose=verbose,
        )
    finally:
        await llm.aclose()


@app.command()
def main(
    agent: Annotated[
        AgentName | None, typer.Option("--agent", "-a", help="Chat with a single agent.")
    ] = None,
    pattern: Annotated[
        PatternName | None,
        typer.Option(
            "--pattern", "-p", help="Chat with the team through an orchestration pattern."
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show tool arguments, results and LLM stats.")
    ] = False,
    sessions_dir: Annotated[Path, typer.Option(help="Where sessions are saved.")] = Path(
        "sessions"
    ),
) -> None:
    """Start an interactive travel-planning chat."""
    if (agent is None) == (pattern is None):
        raise typer.BadParameter("choose either --agent or --pattern", param_hint="--agent")
    try:
        llm = AzureOpenAIClient(Settings())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    if agent is not None:
        single = get_agent(agent, llm)
        mode: Pattern = SingleAgent(single)
        label, prompt, info = (
            f"{mode.name}-{single.name}",
            f"[{single.emoji} {single.name}] you>",
            {"agent": single.name},
        )
    else:
        assert pattern is not None
        mode = PATTERNS[pattern](llm)
        label, prompt, info = mode.name, f"[{mode.name}] you>", {}
    asyncio.run(
        _run(
            mode,
            llm,
            label=label,
            prompt=prompt,
            verbose=verbose,
            sessions_dir=sessions_dir,
            **info,
        )
    )
