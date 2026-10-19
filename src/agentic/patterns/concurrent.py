import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from agentic.agents import consultant
from agentic.agents.registry import AgentName, get_agent
from agentic.core.agent import Agent
from agentic.core.messages import AgentResult, Message
from agentic.core.session import SessionContext
from agentic.llm.base import LLMClient

TEAM: tuple[AgentName, ...] = (
    AgentName.FLIGHT,
    AgentName.HOTEL,
    AgentName.ACTIVITIES,
    AgentName.BUDGET,
)

AGGREGATOR = """\
You are the aggregator. {team} worked on the traveller's message in parallel; none of them saw \
the others' output, and you cannot send work back to them.
- Merge their outputs into one plan, using only what they found.
- Never do arithmetic yourself or recompute totals. If Budget did not price the flight, hotel and \
activities that were actually recommended, say that the total is unverified.
- They did not coordinate, so check that the pieces fit: did Budget price the flight and hotel \
that were actually chosen, does the arrival time suit the day-1 program, is the hotel near the \
activities? Point out every conflict you cannot resolve instead of hiding it.
- If a specialist failed, say what is missing from the plan.
- Collect all open questions; if any need the traveller, ask them together in one ask_user call."""


def _brief(
    request: str,
    results: Sequence[tuple[Agent, AgentResult]],
    failed: Sequence[tuple[Agent, Exception]],
) -> str:
    parts = [request, "## Specialist outputs (worked in parallel, independently)"]
    parts += [f"### {a.emoji} {a.name}\n{r.text}" for a, r in results]
    if failed:
        lines = "\n".join(f"- {a.emoji} {a.name}: {type(e).__name__}: {e}" for a, e in failed)
        parts.append(f"## Failed specialists (no output)\n{lines}")
    questions = [f"- {a.name}: {q}" for a, r in results for q in r.open_questions]
    if questions:
        parts.append("## Open questions from specialists\n" + "\n".join(questions))
    return "\n\n".join(parts)


@dataclass
class Concurrent:
    """Specialists run at the same time on the same input; the aggregator merges their outputs."""

    specialists: list[Agent]
    aggregator: Agent
    name: str = "concurrent"

    @classmethod
    def create(cls, llm: LLMClient, team: Sequence[AgentName] = TEAM) -> Self:
        specialists = [get_agent(name, llm) for name in team]
        names = ", ".join(f"{a.emoji} {a.name}" for a in specialists)
        return cls(specialists, consultant.create(llm, role=AGGREGATOR.format(team=names)))

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        messages = [*ctx.history, Message.user(user_message)]
        outcomes = await asyncio.gather(
            *(agent.run(messages, ctx) for agent in self.specialists), return_exceptions=True
        )
        results: list[tuple[Agent, AgentResult]] = []
        failed: list[tuple[Agent, Exception]] = []
        for agent, outcome in zip(self.specialists, outcomes, strict=True):
            if isinstance(outcome, AgentResult):
                results.append((agent, outcome))
            elif isinstance(outcome, Exception):
                failed.append((agent, outcome))
            else:
                raise outcome
        if not results:
            raise failed[0][1]
        brief = Message.user(_brief(user_message, results, failed))
        answer = (await self.aggregator.run([*ctx.history, brief], ctx)).text
        ctx.history += [Message.user(user_message), Message.assistant(answer)]
        return answer
