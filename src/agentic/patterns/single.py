from dataclasses import dataclass

from agentic.core.agent import Agent
from agentic.core.messages import Message
from agentic.core.session import SessionContext


@dataclass
class SingleAgent:
    """Chat directly with one agent, keeping the conversation across turns."""

    agent: Agent
    name: str = "single"

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        user = Message.user(user_message)
        result = await self.agent.run([*ctx.history, user], ctx)
        ctx.history += [user, Message.assistant(result.text)]
        return result.text
