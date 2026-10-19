from typing import Annotated

from pydantic import Field

from agentic.core.tools import ToolContext, tool
from agentic.logs.events import EventType


@tool
async def ask_user(
    question: Annotated[str, Field(description="One clear question; batch related ones together")],
    ctx: ToolContext,
) -> str:
    """Ask the traveller a question and wait for the answer.

    Only for input a task cannot continue without; normal replies end the turn instead.
    """
    ctx.emit(EventType.ASK_USER, question=question)
    answer = await ctx.session.io.ask(ctx.agent, question)
    ctx.emit(EventType.USER_REPLY, question=question, answer=answer)
    return answer
