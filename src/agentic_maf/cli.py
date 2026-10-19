import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Protocol

import typer

from agentic.config import Settings
from agentic_maf.model import chat_client
from agentic_maf.patterns.handoff import Handoff
from agentic_maf.patterns.magentic import MAX_ROUNDS, MAX_STALLS, Magentic

COMMANDS = "/exit quit · /reset clear the conversation"


class PatternName(StrEnum):
    HANDOFF = "handoff"
    MAGENTIC = "magentic"


class Chat(Protocol):
    @property
    def prompt(self) -> str: ...

    async def turn(self, text: str) -> None: ...

    async def reset(self) -> None: ...


app = typer.Typer(
    help="Agentic travel assistant built with Microsoft Agent Framework: Handoff and Magentic."
)


async def chat(pattern: Chat, read: Callable[[str], str] = input) -> None:
    """Read a message, run a turn, repeat; the pattern prints the conversation itself."""
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


async def run(name: PatternName, *, max_rounds: int, max_stalls: int) -> None:
    async with chat_client(Settings()) as client:
        match name:
            case PatternName.HANDOFF:
                pattern: Chat = Handoff.create(client)
            case PatternName.MAGENTIC:
                pattern = Magentic.create(client, max_rounds=max_rounds, max_stalls=max_stalls)
        await chat(pattern)


@app.command()
def main(
    pattern: Annotated[
        PatternName,
        typer.Option("--pattern", "-p", help="Orchestration pattern to chat with."),
    ],
    max_rounds: Annotated[
        int, typer.Option(min=1, help="Magentic: coordination rounds per message.")
    ] = MAX_ROUNDS,
    max_stalls: Annotated[
        int, typer.Option(min=1, help="Magentic: stalled rounds before a re-plan.")
    ] = MAX_STALLS,
) -> None:
    """Start an interactive travel-planning chat."""
    try:
        asyncio.run(run(pattern, max_rounds=max_rounds, max_stalls=max_stalls))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
