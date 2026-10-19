import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from rich.console import Console
from rich.markup import escape


class UserIO(Protocol):
    async def ask(self, agent: str, question: str) -> str: ...

    async def read(self, prompt: str) -> str:
        """Next chat message; raises EOFError when the user is done."""
        ...


class ConsoleIO:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._lock = asyncio.Lock()  # one prompt at a time, even under concurrency

    async def ask(self, agent: str, question: str) -> str:
        async with self._lock:
            prompt = f"[bold]{escape(agent)} asks:[/] {escape(question)}\n[bold]you>[/] "
            return await asyncio.to_thread(self._console.input, prompt)

    async def read(self, prompt: str) -> str:
        async with self._lock:
            # Blocking on purpose: nothing runs between turns, and Ctrl+C exits at once.
            return self._console.input(f"\n[bold]{escape(prompt)}[/] ")


class ScriptedIO:
    """Answers questions and sends chat messages from fixed lists, for tests."""

    def __init__(self, answers: Iterable[str] = (), *, inputs: Iterable[str] = ()) -> None:
        self._answers = deque(answers)
        self._inputs = deque(inputs)
        self.asked: list[tuple[str, str]] = []
        self.prompts: list[str] = []

    async def ask(self, agent: str, question: str) -> str:
        self.asked.append((agent, question))
        if not self._answers:
            raise RuntimeError(f"ScriptedIO has no answer for {agent}: {question!r}")
        return self._answers.popleft()

    async def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._inputs:
            raise EOFError
        return self._inputs.popleft()
