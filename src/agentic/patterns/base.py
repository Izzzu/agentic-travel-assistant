from typing import Protocol

from agentic.core.session import SessionContext


class Pattern(Protocol):
    name: str

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        """Answer one user message; conversation state lives in `ctx.history`."""
        ...
