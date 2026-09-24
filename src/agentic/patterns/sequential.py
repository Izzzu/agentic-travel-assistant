from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from agentic.agents import consultant
from agentic.agents.registry import AgentName, get_agent
from agentic.core.agent import Agent
from agentic.core.messages import Message
from agentic.core.session import SessionContext
from agentic.llm.base import LLMClient
from agentic.logs.events import EventType

# Edit this list to demo a different order; the Travel Consultant always writes last.
ORDER: tuple[AgentName, ...] = (
    AgentName.FLIGHT,
    AgentName.HOTEL,
    AgentName.ACTIVITIES,
    AgentName.BUDGET,
)

WRITER = """\
You are the writer at the end of a fixed pipeline: {pipeline} → you. Each specialist worked once, \
in that order, and saw only the outputs before theirs. You cannot send work back to them.
- Write the final plan from their outputs only.
- If Budget reports that the plan is over budget, still present it: put a clear warning with the \
overrun in CHF at the top and list the savings Budget suggested.
- Resolve the specialists' open questions as your rules say."""


def _brief(request: str, outputs: Sequence[tuple[Agent, str]]) -> str:
    if not outputs:
        return request
    sections = "\n\n".join(f"### {a.emoji} {a.name}\n{text}" for a, text in outputs)
    return (
        f"{request}\n\n## Outputs from earlier steps\n"
        f"Build on these choices instead of redoing them.\n\n{sections}"
    )


@dataclass
class Sequential:
    """Fixed pipeline: each agent gets the request plus every earlier agent's output."""

    steps: list[Agent]
    name: str = "sequential"

    @classmethod
    def create(cls, llm: LLMClient, order: Sequence[AgentName] = ORDER) -> Self:
        specialists = [get_agent(name, llm) for name in order]
        pipeline = " → ".join(f"{a.emoji} {a.name}" for a in specialists)
        writer = consultant.create(llm, role=WRITER.format(pipeline=pipeline))
        return cls([*specialists, writer])

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        outputs: list[tuple[Agent, str]] = []
        for step, agent in enumerate(self.steps, 1):
            ctx.emit(
                EventType.STEP,
                agent=agent.name,
                emoji=agent.emoji,
                step=step,
                total=len(self.steps),
            )
            brief = Message.user(_brief(user_message, outputs))
            result = await agent.run([*ctx.history, brief], ctx)
            outputs.append((agent, result.text))
        answer = outputs[-1][1]
        ctx.history += [Message.user(user_message), Message.assistant(answer)]
        return answer
