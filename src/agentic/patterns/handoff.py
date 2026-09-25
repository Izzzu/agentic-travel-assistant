import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Self

from pydantic import Field

from agentic.agents import consultant
from agentic.agents.registry import AgentName, get_agent
from agentic.core.agent import Agent
from agentic.core.messages import AgentResult, Message
from agentic.core.session import SessionContext
from agentic.core.tools import Tool, tool
from agentic.llm.base import LLMClient
from agentic.logs.events import EventType

A = AgentName
HANDOFF_GRAPH: dict[AgentName, tuple[AgentName, ...]] = {
    A.CONSULTANT: (A.FLIGHT, A.HOTEL, A.ACTIVITIES, A.BUDGET),
    A.FLIGHT: (A.CONSULTANT, A.HOTEL, A.BUDGET),
    A.HOTEL: (A.CONSULTANT, A.FLIGHT, A.BUDGET),
    A.ACTIVITIES: (A.CONSULTANT, A.BUDGET),
    A.BUDGET: (A.CONSULTANT,),
}
ENTRY = A.CONSULTANT
MAX_HANDOFFS = 8
PREFIX = "handoff_to_"

ABOUT: dict[AgentName, str] = {
    A.CONSULTANT: "the traveller's main contact: decisions only the traveller can make, and the "
    "final plan",
    A.FLIGHT: "flights between Zurich and Lisbon",
    A.HOTEL: "accommodation and availability",
    A.ACTIVITIES: "the day-by-day food and beach program",
    A.BUDGET: "totals, budget checks and savings",
}

ROUTER = """\
You are the front door. Control moves through the team with handoff tools: you can hand off to \
{targets}, and each specialist hands work on to another specialist or back to you.
- New trip request: once the brief is clear, hand off to the first specialist needed (usually \
flight). Say in `reason` what they should do; they pass the work along.
- When control comes back to you, read the whole conversation: resolve the open points, hand \
off again if something is missing (e.g. the program, or Budget's check), or ask the traveller \
when only they can decide (e.g. go over budget or change area?).
- Present the final plan only when flights, hotel, program and Budget's check are in the \
conversation. Never do arithmetic yourself.
- Small talk or questions you can answer from the conversation: just reply."""

SPECIALIST = """\
## Handoffs
Control moves through the team with handoff tools; you can hand off to {targets}. Whoever \
takes over sees the whole conversation, including your tool results.
- Do your part first, then hand off to whoever should continue, with a short `reason` saying \
what they should do. Hand off at once when the request is outside your domain.
- Hand back to an agent that already worked only for a concrete conflict they can fix (e.g. a \
late arrival vs. check-in), never just to re-check their work.
- When nobody else needs to act, reply to the traveller directly: your reply ends the turn and \
their next message comes back to you.
- You cannot ask the traveller mid-task. When the work needs a decision only they can make \
(e.g. going over budget), hand off to the consultant with that question as the reason instead \
of listing `Open questions:`."""

LIMIT = """\
## Handoff limit reached
No more handoffs this turn. Reply to the traveller now with what the conversation has so far, \
and say clearly what is still missing."""


def _names(agents: Sequence[Agent]) -> str:
    return ", ".join(f"{a.emoji} {a.name}" for a in agents)


def handoff_tool(target: Agent, about: str) -> Tool[[str], str]:
    def handoff(
        reason: Annotated[str, Field(description="What they should do next, in one sentence")],
    ) -> str:
        return f"{target.emoji} {target.name} takes over."

    return replace(
        tool(handoff),
        name=f"{PREFIX}{target.name}",
        description=f"Hand control to {target.emoji} {target.name} ({about}). They see the "
        "whole conversation.",
        ends_run=True,
    )


@dataclass
class Handoff:
    """Agents pass control along a handoff graph; the active agent keeps it across user turns."""

    agents: dict[str, Agent]
    entry: str = ENTRY
    max_handoffs: int = MAX_HANDOFFS
    name: str = "handoff"
    active: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.active = self.entry

    @classmethod
    def create(
        cls,
        llm: LLMClient,
        graph: Mapping[AgentName, Sequence[AgentName]] = HANDOFF_GRAPH,
        *,
        max_handoffs: int = MAX_HANDOFFS,
    ) -> Self:
        if max_handoffs < 0:
            raise ValueError("max_handoffs must not be negative")
        if ENTRY not in graph or any(
            t not in graph or t == s for s, ts in graph.items() for t in ts
        ):
            raise ValueError("graph needs the consultant and an entry for every other target")
        base = {name: get_agent(name, llm) for name in graph}
        agents: dict[str, Agent] = {}
        for name, targets in graph.items():
            others = [base[t] for t in targets]
            tools = [handoff_tool(base[t], ABOUT[t]) for t in targets]
            if name == ENTRY:
                agent = consultant.create(llm, role=ROUTER.format(targets=_names(others)))
                agents[name] = replace(agent, tools=[*agent.tools, *tools])
            else:
                agent = base[name]
                rules = SPECIALIST.format(targets=_names(others))
                agents[name] = replace(
                    agent,
                    instructions=f"{agent.instructions}\n\n{rules}",
                    tools=[*agent.tools, *tools],
                )
        return cls(agents, max_handoffs=max_handoffs)

    def active_agent(self, ctx: SessionContext) -> Agent:
        """A new or reset conversation starts at the entry agent."""
        return self.agents[self.active if ctx.history else self.entry]

    def prompt(self, ctx: SessionContext) -> str:
        agent = self.active_agent(ctx)
        return f"[{agent.emoji} {agent.name}] you>"

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        agent = self.active_agent(ctx)
        conversation = [*ctx.history, Message.user(user_message)]
        handoffs = 0
        while True:
            if handoffs < self.max_handoffs:
                result = await agent.run(conversation, ctx)
            else:
                stuck = replace(agent, tools=[t for t in agent.tools if not t.ends_run])
                result = await stuck.run([*conversation, Message.user(LIMIT)], ctx)
            conversation += result.messages
            handoff = self._handoff(result)
            if handoff is None:
                break
            target, reason = handoff
            handoffs += 1
            ctx.emit(
                EventType.HANDOFF,
                agent=agent.name,
                from_agent=agent.name,
                to_agent=target.name,
                emoji=target.emoji,
                reason=reason,
                handoff=handoffs,
                max_handoffs=self.max_handoffs,
            )
            agent = target
        self.active = agent.name
        ctx.history = conversation
        return result.text

    def _handoff(self, result: AgentResult) -> tuple[Agent, str] | None:
        record = result.stopped_by
        if record is None or not record.name.startswith(PREFIX):
            return None
        args: dict[str, Any] = json.loads(record.arguments)
        return self.agents[record.name.removeprefix(PREFIX)], str(args.get("reason", ""))
