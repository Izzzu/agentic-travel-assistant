import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self, overload

from agent_framework import (
    Agent,
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    BaseAgent,
    Content,
    Message,
    ResponseStream,
    SupportsChatGetResponse,
    Workflow,
    normalize_messages,
)
from agent_framework.orchestrations import MagenticBuilder

from agentic_maf.agents import SPECIALISTS, create_specialist, trip_context
from agentic_maf.console import Console

TRAVELLER = "traveller"
MAX_ROUNDS = 15
MAX_STALLS = 2  # stalled rounds tolerated before a re-plan

TRAVELLER_ABOUT = (
    "The traveller. Only for a question that must be answered before the work can go on, "
    "e.g. go over budget or change area; batch related questions into one. Never use them to "
    "present results or to confirm a finished plan."
)

MANAGER = """\
You coordinate a travel team of Flight, Hotel, Activities and Budget specialists and the \
traveller.
{context}

- The request is satisfied as soon as flights, a hotel, a day-by-day program and Budget's check \
of the chosen options are in the conversation. Say so then: the final reply to the traveller is \
written for you afterwards, so it is not a step for the team.
- Pick the traveller as next speaker only when you need an answer to a question. Do not ask for \
details you can assume; state your assumptions instead.
- Never do arithmetic yourself; totals and budget checks come from Budget."""

# Agent Framework fills {task}; the text must not contain other braces.
FINAL = """\
We are working on the following task:
{task}

The work is done; the messages above hold it.

Write the final reply to the traveller. A plan has flights, accommodation, a day-by-day program, \
a cost table in CHF and a clear statement of whether it fits the budget. Use Budget's totals and \
never do arithmetic yourself. If the plan is over budget or something is unresolved, say so \
clearly at the top and ask the traveller what they want to do.
"""

Ask = Callable[[str], Awaitable[str]]


async def ask_terminal(question: str) -> str:
    return await asyncio.to_thread(input, f"\n{question}\n[{TRAVELLER}] you> ")


class Traveller(BaseAgent):
    """A participant that puts the manager's question to the person at the terminal."""

    def __init__(self, ask: Ask) -> None:
        super().__init__(name=TRAVELLER, description=TRAVELLER_ABOUT)
        self._ask = ask

    async def _answer(self, messages: Sequence[Message]) -> Message:
        answer = await self._ask(messages[-1].text if messages else "")
        return Message("user", [Content.from_text(answer)], author_name=TRAVELLER)

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = False,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        incoming = normalize_messages(messages)

        async def respond() -> AgentResponse[Any]:
            return AgentResponse(messages=[await self._answer(incoming)])

        async def updates() -> AsyncIterable[AgentResponseUpdate]:
            reply = await self._answer(incoming)
            yield AgentResponseUpdate(
                contents=reply.contents, role=reply.role, author_name=TRAVELLER
            )

        if stream:
            return ResponseStream(updates(), finalizer=AgentResponse.from_updates)
        return respond()


@dataclass
class Magentic:
    """A Magentic workflow. Each message builds a new one, so the task carries the history."""

    client: SupportsChatGetResponse[Any]
    max_rounds: int = MAX_ROUNDS
    max_stalls: int = MAX_STALLS
    ask: Ask = ask_terminal
    console: Console = field(default_factory=Console)
    history: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    @classmethod
    def create(
        cls,
        client: SupportsChatGetResponse[Any],
        *,
        max_rounds: int = MAX_ROUNDS,
        max_stalls: int = MAX_STALLS,
        ask: Ask = ask_terminal,
        console: Console | None = None,
    ) -> Self:
        return cls(client, max_rounds, max_stalls, ask, console or Console())

    @property
    def prompt(self) -> str:
        return "[magentic] you>"

    def build(self) -> Workflow:
        specialists = [create_specialist(name, self.client) for name in SPECIALISTS]
        manager = Agent(
            client=self.client,
            instructions=MANAGER.format(context=trip_context()),
            name="manager",
            description="Plans, picks who acts next and writes the final reply",
        )
        return MagenticBuilder(
            participants=[*specialists, Traveller(self.ask)],
            manager_agent=manager,
            intermediate_output_from=specialists,
            final_answer_prompt=FINAL,
            max_round_count=self.max_rounds,
            max_stall_count=self.max_stalls,
        ).build()

    def task(self, text: str) -> str:
        if not self.history:
            return text
        earlier = "\n\n".join(
            f"Traveller: {asked}\nTeam: {answer}" for asked, answer in self.history
        )
        return f"Conversation so far:\n\n{earlier}\n\nThe traveller's new message: {text}"

    async def turn(self, text: str) -> None:
        """Run one user message, printing the workflow's events as they happen."""
        stream = self.build().run(self.task(text), stream=True)
        answer = ""
        async for event in stream:
            if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
                answer += event.data.text
            self.console.show(event)
        await stream.get_final_response()
        self.console.finish()
        self.history.append((text, answer))

    async def reset(self) -> None:
        self.history.clear()
