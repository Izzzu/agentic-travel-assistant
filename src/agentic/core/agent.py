import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic.core.messages import AgentResult, Message, ToolCallRecord, Usage
from agentic.core.session import SessionContext
from agentic.core.tools import Tool, execute_tool_call
from agentic.llm.base import LLMClient
from agentic.logs.events import EventType


class AgentError(RuntimeError):
    pass


def _summarize(messages: Sequence[Message], limit: int = 200) -> str:
    text = next((m.content for m in reversed(messages) if m.content), "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Agent:
    name: str
    emoji: str
    instructions: str
    llm: LLMClient
    tools: list[Tool[..., Any]] = field(default_factory=list[Tool[..., Any]])
    max_iterations: int = 8

    async def run(self, messages: Sequence[Message], ctx: SessionContext) -> AgentResult:
        run_id = secrets.token_hex(4)
        ctx.emit(
            EventType.AGENT_START,
            agent=self.name,
            run_id=run_id,
            emoji=self.emoji,
            input=_summarize(messages),
        )
        started = time.perf_counter()
        records: list[ToolCallRecord] = []
        usage = Usage()
        try:
            conversation = [Message.system(self.instructions), *messages]
            text = ""
            for _ in range(self.max_iterations):
                t0 = time.perf_counter()
                response = await self.llm.complete(conversation, [t.spec for t in self.tools])
                usage += response.usage
                ctx.usage += response.usage
                ctx.emit(
                    EventType.LLM_CALL,
                    agent=self.name,
                    run_id=run_id,
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    finish_reason=response.finish_reason,
                    ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                if response.finish_reason == "content_filter":
                    raise AgentError(f"{self.name}: response blocked by the content filter")
                if not response.tool_calls:
                    text = response.content or ""
                    break
                conversation.append(Message.assistant(response.content, response.tool_calls))
                tools = {t.name: t for t in self.tools}
                for call in response.tool_calls:
                    record = await execute_tool_call(
                        call, tools, ctx, agent=self.name, run_id=run_id
                    )
                    records.append(record)
                    conversation.append(Message.tool(call.id, record.result))
            else:
                raise AgentError(f"{self.name}: no final answer after {self.max_iterations} steps")
        except Exception as exc:
            ctx.emit(
                EventType.ERROR,
                agent=self.name,
                run_id=run_id,
                exception=f"{type(exc).__name__}: {exc}",
            )
            raise
        ms = round((time.perf_counter() - started) * 1000, 1)
        ctx.emit(
            EventType.AGENT_END,
            agent=self.name,
            run_id=run_id,
            output=text,
            tool_calls=len(records),
            tokens=usage.total_tokens,
            ms=ms,
        )
        return AgentResult(agent=self.name, text=text, tool_calls=records, usage=usage, ms=ms)
