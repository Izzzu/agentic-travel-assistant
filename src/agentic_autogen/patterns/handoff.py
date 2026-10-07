from dataclasses import dataclass
from typing import Self

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import ChatAgent, TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMessageTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient

from agentic_autogen.agents import SPECIALISTS, create_specialist, trip_context

USER = "user"
ENTRY = "consultant"
MAX_MESSAGES = 40  # counts every chat message, not just handoffs

# Who each agent may hand control to.
GRAPH: dict[str, tuple[str, ...]] = {
    ENTRY: ("flight", "hotel", "activities", "budget"),
    "flight": (ENTRY, "hotel", "budget"),
    "hotel": (ENTRY, "flight", "budget"),
    "activities": (ENTRY, "budget"),
    "budget": (ENTRY,),
}

CONSULTANT = """\
You are the Travel Consultant, the traveller's only point of contact in a travel team with \
Flight, Hotel, Activities and Budget specialists. Control moves through the team by handoffs; \
you can hand off to {targets}.
{context}

## Rules
- Make sure you know the destination, dates, number of travellers, origin, budget and interests. \
Assume what you can and state your assumptions.
- You have no search or pricing tools. Never invent flights, hotels, activities, prices or \
availability; they come from the specialists.
- New trip request: once the brief is clear, hand off to the first specialist needed (usually \
flight). They pass the work along and hand back to you.
- When control comes back to you, read the whole conversation: resolve the open points, hand off \
again if something is missing (e.g. the program, or Budget's check), or ask the traveller when \
only they can decide (e.g. go over budget or change area?).
- A final plan has flights, accommodation, a day-by-day program, a cost table in CHF and a clear \
statement of whether it fits the budget. Present it only when flights, hotel, program and \
Budget's check are in the conversation. Never do arithmetic yourself.
- A plain message ends your turn and the traveller answers: send one to ask a question, present \
the plan or just reply. To hand off instead, call the handoff tool; put any note in the same \
response as the call, because a message without a call ends your turn."""

SPECIALIST = """\
## Handoffs
Control moves through the team by handoffs; you can hand off to {targets}. Whoever takes over \
sees the whole conversation, including your tool results.
- Do your part with your tools first, write your recommendation, then hand off to whoever should \
continue in the same turn. Hand off at once when the request is outside your domain.
- Hand back to an agent that already worked only for a concrete conflict they can fix (e.g. a \
late arrival vs. check-in), never just to re-check their work.
- You cannot talk to the traveller. When your work needs a decision only they can make (e.g. \
going over budget), say so in your message and hand off to the consultant.
- Always end by handing off: a message without a handoff keeps control with you."""


@dataclass
class Handoff:
    """A Swarm team; the agent that last held control keeps it across user turns."""

    team: Swarm
    holder: str | None = None

    @classmethod
    def create(cls, model: ChatCompletionClient, *, max_messages: int = MAX_MESSAGES) -> Self:
        agents: list[ChatAgent] = [
            AssistantAgent(
                ENTRY,
                model_client=model,
                handoffs=list(GRAPH[ENTRY]),
                description="the traveller's main contact: decisions only the traveller can make, "
                "and the final plan",
                system_message=CONSULTANT.format(
                    targets=", ".join(GRAPH[ENTRY]), context=trip_context()
                ),
            ),
            *(
                create_specialist(
                    name,
                    model,
                    rules=SPECIALIST.format(targets=", ".join(GRAPH[name])),
                    handoffs=GRAPH[name],
                )
                for name in SPECIALISTS
            ),
        ]
        # Swarm keeps the same speaker after a plain message, so the consultant's reply must end the turn.
        termination = TextMessageTermination(source=ENTRY) | MaxMessageTermination(max_messages)
        return cls(Swarm(agents, termination_condition=termination))

    @property
    def prompt(self) -> str:
        return f"[{self.holder or ENTRY}] you>"

    async def turn(self, text: str) -> TaskResult:
        """Run one user message, printing AutoGen's default console output."""
        task = (
            text
            if self.holder is None
            else HandoffMessage(source=USER, target=self.holder, content=text)
        )
        result = await Console(self.team.run_stream(task=task))
        last = result.messages[-1]
        if isinstance(last, HandoffMessage):
            self.holder = last.target  # stopped by the message limit mid-handoff
        else:
            self.holder = last.source
        return result

    async def reset(self) -> None:
        await self.team.reset()
        self.holder = None
