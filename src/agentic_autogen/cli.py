import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Protocol

import typer
from autogen_agentchat.base import TaskResult

from agentic_autogen.config import AutogenSettings
from agentic_autogen.model import model_client
from agentic_autogen.patterns.handoff import Handoff
from agentic_autogen.patterns.magentic import MAX_STALLS, MAX_TURNS, Magentic

COMMANDS = "/exit quit · /reset clear the conversation"


class PatternName(StrEnum):
    HANDOFF = "handoff"
    MAGENTIC = "magentic"


class Chat(Protocol):
    @property
    def prompt(self) -> str: ...

    async def turn(self, text: str) -> TaskResult: ...

    async def reset(self) -> None: ...


app = typer.Typer(help="Agentic travel assistant built with AutoGen: Handoff and Magentic.")


async def chat(pattern: Chat, read: Callable[[str], str] = input) -> None:
    """Read a message, run a turn, repeat; AutoGen prints the conversation itself."""
    print(f"Commands: {COMMANDS}")
    while True:
        try:
            text = read(f"\n{pattern.prompt} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.startswith("/"):
            command = text.lower()
            if command in ("/exit", "/quit"):
                break
            if command == "/reset":
                await pattern.reset()
                print("Conversation cleared.")
            else:
                print(f"Unknown command {text}. {COMMANDS}")
            continue
        try:
            await pattern.turn(text)
        except Exception as exc:  # noqa: BLE001 - a failed turn must not end the demo
            print(f"Turn failed ({type(exc).__name__}); try again or /reset.")


async def run(name: PatternName, *, max_turns: int, max_stalls: int) -> None:
    async with model_client(
        AutogenSettings(),
        parallel_tool_calls=False if name is PatternName.HANDOFF else None,  # one handoff per turn
    ) as model:
        match name:
            case PatternName.HANDOFF:
                pattern: Chat = Handoff.create(model)
            case PatternName.MAGENTIC:
                pattern = Magentic.create(model, max_turns=max_turns, max_stalls=max_stalls)
        await chat(pattern)


@app.command()
def main(
    pattern: Annotated[
        PatternName,
        typer.Option("--pattern", "-p", help="Orchestration pattern to chat with."),
    ],
    max_turns: Annotated[
        int, typer.Option(min=1, help="Magentic: team turns per message.")
    ] = MAX_TURNS,
    max_stalls: Annotated[
        int, typer.Option(min=1, help="Magentic: stalled steps before a re-plan.")
    ] = MAX_STALLS,
) -> None:
    """Start an interactive travel-planning chat."""
    try:
        asyncio.run(run(pattern, max_turns=max_turns, max_stalls=max_stalls))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
