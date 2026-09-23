import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from rich.console import Console
from rich.markup import escape


class UserIO(Protocol):
    async def ask(self, agent: str, question: str) -> str: ...


class ConsoleIO:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._lock = asyncio.Lock()  # one prompt at a time, even under concurrency

    async def ask(self, agent: str, question: str) -> str:
        async with self._lock:
            prompt = f"[bold]{escape(agent)} asks:[/] {escape(question)}\n[bold]you>[/] "
            return await asyncio.to_thread(self._console.input, prompt)


class ScriptedIO:
    """Answers questions from a fixed list, for tests."""

    def __init__(self, answers: Iterable[str] = ()) -> None:
        self._answers = deque(answers)
        self.asked: list[tuple[str, str]] = []

    async def ask(self, agent: str, question: str) -> str:
        self.asked.append((agent, question))
        if not self._answers:
            raise RuntimeError(f"ScriptedIO has no answer for {agent}: {question!r}")
        return self._answers.popleft()
