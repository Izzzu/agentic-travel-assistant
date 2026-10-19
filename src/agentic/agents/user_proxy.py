import time

from agentic.core.session import SessionContext
from agentic.logs.events import EventType


class UserProxy:
    """Stands in for the traveller inside a pattern: relays a question and returns the answer.

    It has no LLM and no tools, so it never costs tokens and never answers on its own.
    """

    name = "user_proxy"
    emoji = "🧑"

    async def ask(self, question: str, ctx: SessionContext) -> str:
        started = time.perf_counter()
        ctx.emit(EventType.AGENT_START, agent=self.name, emoji=self.emoji, input=question)
        ctx.emit(EventType.ASK_USER, agent=self.name, question=question)
        answer = await ctx.io.ask(self.name, question)
        ctx.emit(EventType.USER_REPLY, agent=self.name, question=question, answer=answer)
        ctx.emit(
            EventType.AGENT_END,
            agent=self.name,
            output=answer,
            open_questions=[],
            tool_calls=0,
            tokens=0,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return answer
