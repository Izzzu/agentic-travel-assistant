import re
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

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


_QUESTIONS_HEADING = re.compile(r"^[#*_\s]*open questions[*_\s]*(?::(?P<rest>.*))?$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(?P<text>.+)$")
_NONE = {"", "none", "n/a", "-"}


def parse_open_questions(text: str) -> list[str]:
    """Items listed under an `Open questions:` heading, as specialists are told to write them."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _QUESTIONS_HEADING.match(line)), None)
    if start is None:
        return []
    heading = _QUESTIONS_HEADING.match(lines[start])
    questions = [(heading["rest"] if heading and heading["rest"] else "").strip(" *_")]
    for line in lines[start + 1 :]:
        if not line.strip():
            if len(questions) > 1:
                break
            continue
        bullet = _BULLET.match(line)
        if bullet is None:
            break
        questions.append(bullet["text"].strip())
    return [q for q in questions if q.lower().rstrip(".") not in _NONE]


@dataclass
class Agent:
    name: str
    emoji: str
    instructions: str
    llm: LLMClient
    tools: list[Tool[..., Any]] = field(default_factory=list[Tool[..., Any]])
    max_iterations: int = 8

    async def run(
        self,
        messages: Sequence[Message],
        ctx: SessionContext,
        *,
        response_format: type[BaseModel] | None = None,
    ) -> AgentResult:
        """With `response_format`, the final text is JSON for that model."""
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
        stopped_by: ToolCallRecord | None = None
        usage = Usage()
        try:
            conversation = [Message.system(self.instructions), *messages]
            text = ""
            for _ in range(self.max_iterations):
                t0 = time.perf_counter()
                response = await self.llm.complete(
                    conversation, [t.spec for t in self.tools], response_format
                )
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
                    conversation.append(Message.assistant(text))
                    break
                conversation.append(Message.assistant(response.content, response.tool_calls))
                tools = {t.name: t for t in self.tools}
                for call in response.tool_calls:
                    record = await execute_tool_call(
                        call, tools, ctx, agent=self.name, run_id=run_id
                    )
                    records.append(record)
                    conversation.append(Message.tool(call.id, record.result))
                    if stopped_by is None and record.ok and tools[call.name].ends_run:
                        stopped_by = record
                if stopped_by is not None:
                    text = response.content or ""
                    break
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
        open_questions = parse_open_questions(text)
        ctx.emit(
            EventType.AGENT_END,
            agent=self.name,
            run_id=run_id,
            output=text,
            open_questions=open_questions,
            tool_calls=len(records),
            tokens=usage.total_tokens,
            ms=ms,
        )
        return AgentResult(
            agent=self.name,
            text=text,
            open_questions=open_questions,
            tool_calls=records,
            stopped_by=stopped_by,
            messages=conversation[len(messages) + 1 :],
            usage=usage,
            ms=ms,
        )
