from dataclasses import dataclass, field
from typing import Self

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import ChatAgent, TaskResult
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient

from agentic_autogen.agents import SPECIALISTS, create_specialist, trip_context

TRAVELLER = "traveller"
MAX_TURNS = 15
MAX_STALLS = 2  # stalled steps tolerated before a re-plan

TRAVELLER_ABOUT = (
    "The traveller. Only for a question that must be answered before the work can go on, "
    "e.g. go over budget or change area; batch related questions into one. Never use them to "
    "present results or to confirm a finished plan."
)

NOTES = """\
Notes for the manager:
- {context}
- The request is satisfied as soon as flights, a hotel, a day-by-day program and Budget's check \
of the chosen options are in the conversation. Say so then: the final reply to the traveller is \
written for you afterwards, so it is not a step for the team.
- Pick the traveller as next speaker only when you need an answer to a question. Do not ask for \
details you can assume; state your assumptions instead."""

# AutoGen fills {task}; the text must not contain other braces.
FINAL = """\
We are working on the following task:
{task}

The work is done; the messages above hold it.

Write the final reply to the traveller. A plan has flights, accommodation, a day-by-day program, \
a cost table in CHF and a clear statement of whether it fits the budget. Use Budget's totals and \
never do arithmetic yourself. If the plan is over budget or something is unresolved, say so \
clearly at the top and ask the traveller what they want to do.
"""


@dataclass
class Magentic:
    """A MagenticOne team. Every turn starts a new orchestration, so the task carries the history."""

    team: MagenticOneGroupChat
    history: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    @property
    def prompt(self) -> str:
        return "[magentic] you>"

    @classmethod
    def create(
        cls,
        model: ChatCompletionClient,
        *,
        max_turns: int = MAX_TURNS,
        max_stalls: int = MAX_STALLS,
    ) -> Self:
        participants: list[ChatAgent] = [create_specialist(name, model) for name in SPECIALISTS]
        participants.append(UserProxyAgent(TRAVELLER, description=TRAVELLER_ABOUT))
        team = MagenticOneGroupChat(
            participants,
            model_client=model,
            max_turns=max_turns,
            max_stalls=max_stalls,
            final_answer_prompt=FINAL,
        )
        return cls(team)

    def task(self, text: str) -> str:
        notes = NOTES.format(context=trip_context())
        if not self.history:
            return f"{text}\n\n{notes}"
        earlier = "\n\n".join(
            f"Traveller: {asked}\nTeam: {answer}" for asked, answer in self.history
        )
        return (
            f"Conversation so far:\n\n{earlier}\n\nThe traveller's new message: {text}\n\n{notes}"
        )

    async def turn(self, text: str) -> TaskResult:
        """Run one user message, printing AutoGen's default console output."""
        result = await Console(self.team.run_stream(task=self.task(text)))
        self.history.append((text, result.messages[-1].to_text()))
        return result

    async def reset(self) -> None:
        await self.team.reset()
        self.history.clear()
