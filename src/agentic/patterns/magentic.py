from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn, Self, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from agentic.agents import consultant
from agentic.agents.registry import get_agent
from agentic.core.agent import Agent, AgentError
from agentic.core.messages import Message
from agentic.core.session import SessionContext
from agentic.llm.base import LLMClient
from agentic.logs.events import EventType

Speaker = Literal["flight", "hotel", "activities", "budget"]
Action = Literal["delegate", "replan", "finish"]
MAX_STEPS = 15
MAX_STALLS = 1  # stalled steps tolerated before a re-plan


class TaskLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict structured output needs a closed schema

    facts: list[str]  # given or verified
    assumptions: list[str]  # still to verify
    plan: list[str]  # steps, each starting with the agent that does it


class ProgressLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    is_request_satisfied: bool
    is_in_loop: bool
    is_progress_being_made: bool
    needs_replan: bool  # a result invalidated the plan, e.g. a sold-out hotel
    next_speaker: Speaker
    instruction: str  # what that agent should do now


MANAGER = """\
You manage {team}. They work one at a time, only on the step you assign; you never search or \
price anything yourself.
- You keep a task ledger (facts, assumptions, plan) and, after every step, a progress ledger that \
says whether the request is done and who acts next with which instruction. When asked for a \
ledger, answer with its JSON only.
- Instructions are shown to everyone: make each one concrete, with the dates, ids and numbers \
the specialist needs.
- When a result contradicts the plan or its assumptions (e.g. an option is sold out, or the plan \
is over budget), update the facts and re-plan instead of carrying on.
- Never do arithmetic yourself; totals and budget checks come from Budget.
- At the end you write the final plan. If only the traveller can decide something, use ask_user."""

PLAN = """\
## Task ledger
Before anyone works, write the task ledger for the latest message:
- facts: what the request or the conversation states or has verified
- assumptions: what the plan relies on that nobody has verified yet (e.g. availability, prices, \
fitting the budget)
- plan: short steps, each starting with the specialist who does it (e.g. `hotel: ...`); end with \
Budget checking the chosen options against the budget
Leave the plan empty when the message needs no specialists (a greeting, a question you can \
answer from the conversation)."""

PROGRESS = """\
## Step {step}/{max_steps} · progress ledger
Compare the work so far with the task ledger and answer with the progress ledger JSON only.
- reason: one or two sentences on where the work stands; shown to everyone
- is_request_satisfied: flights, hotel, program and Budget's check of the chosen options are all \
in the work, and nothing a specialist can fix is left
- is_in_loop: agents repeat the same requests or answers
- is_progress_being_made: the last step added something new
- needs_replan: a result contradicts the plan or its assumptions, so the remaining steps no \
longer lead to a valid plan
- next_speaker, instruction: who acts next and exactly what they should do"""

REPLAN = """\
## Re-plan
{why}
Write the updated task ledger: move verified results into facts, drop or correct the assumptions \
the work disproved, and write a new plan for the remaining work that gets around the problem. \
Keep facts that still hold and do not repeat finished steps."""

TASK = """\
## Your task, {speaker}
The manager assigned you: {instruction}
Do only this. Build on the work above instead of repeating it, and use your tools for every new \
fact or price."""

FINAL = """\
## Final answer
Write the final plan for the traveller from the work above. Use Budget's totals; if the plan is \
still over budget, put a clear warning with Budget's overrun at the top. If the traveller must \
first choose between real options (e.g. go over budget or change area), ask with ask_user and \
write the plan for their choice."""

LIMIT = """\
## Step limit reached
No more steps. Write the final plan for the traveller from the work so far, with a clear warning \
about anything still unresolved or over budget."""

DIRECT = "## Reply\nNo specialists are needed. Reply to the traveller's latest message."


@dataclass
class Work:
    step: int
    agent: Agent
    instruction: str
    text: str


def _bullets(items: Sequence[str], *, numbered: bool = False) -> str:
    if not items:
        return "- none"
    return "\n".join(f"{i}. {x}" if numbered else f"- {x}" for i, x in enumerate(items, 1))


def _sections(ledger: TaskLedger) -> dict[str, list[str]]:
    return {"facts": ledger.facts, "assumptions": ledger.assumptions, "plan": ledger.plan}


def _brief(
    request: str,
    ledger: TaskLedger,
    version: int,
    work: Sequence[Work],
    manager: Agent,
    instruction: str,
) -> str:
    log = "\n\n".join(
        f"**Step {w.step} · {manager.emoji} {manager.name} → {w.agent.emoji} {w.agent.name}:** "
        f"{w.instruction}\n\n**{w.agent.emoji} {w.agent.name}:**\n{w.text}"
        for w in work
    )
    return (
        f"{request}\n\n## Task ledger v{version}\n\n### Facts\n{_bullets(ledger.facts)}\n\n"
        f"### Assumptions\n{_bullets(ledger.assumptions)}\n\n"
        f"### Plan\n{_bullets(ledger.plan, numbered=True)}\n\n"
        f"## Work so far\n\n{log or '_Nothing yet._'}\n\n{instruction}"
    )


@dataclass
class Magentic:
    """A manager keeps a task and a progress ledger, delegates step by step and re-plans."""

    specialists: dict[str, Agent]
    manager: Agent
    max_steps: int = MAX_STEPS
    name: str = "magentic"

    @classmethod
    def create(cls, llm: LLMClient, *, max_steps: int = MAX_STEPS) -> Self:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        specialists = [get_agent(name, llm) for name in get_args(Speaker)]
        team = ", ".join(f"{a.emoji} {a.name}" for a in specialists)
        manager = consultant.create(llm, role=MANAGER.format(team=team))
        return cls({a.name: a for a in specialists}, manager, max_steps)

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        work: list[Work] = []
        ledger = await self._ask(TaskLedger, Message.user(f"{user_message}\n\n{PLAN}"), ctx)
        version = 1
        self._announce_task(ledger, None, version, "", ctx)

        def brief(instruction: str) -> Message:
            text = _brief(user_message, ledger, version, work, self.manager, instruction)
            return Message.user(text)

        final = FINAL if ledger.plan else DIRECT
        if ledger.plan:
            stalls = 0
            for step in range(1, self.max_steps + 1):
                instruction = PROGRESS.format(step=step, max_steps=self.max_steps)
                progress = await self._ask(ProgressLedger, brief(instruction), ctx)
                stalled = progress.is_in_loop or not progress.is_progress_being_made
                stalls = stalls + 1 if stalled else max(0, stalls - 1)
                action: Action = "delegate"
                if progress.is_request_satisfied:
                    action = "finish"
                elif progress.needs_replan or stalls > MAX_STALLS:
                    action = "replan"
                self._announce_progress(progress, step, stalls, action, ctx)
                if action == "finish":
                    break
                if action == "replan":
                    why = (
                        f"The plan no longer works: {progress.reason}"
                        if progress.needs_replan
                        else f"No progress for {stalls} steps: {progress.reason}"
                    )
                    updated = await self._ask(TaskLedger, brief(REPLAN.format(why=why)), ctx)
                    version += 1
                    self._announce_task(updated, ledger, version, why, ctx)
                    ledger, stalls = updated, 0
                    continue
                speaker = self.specialists[progress.next_speaker]
                task = TASK.format(
                    speaker=f"{speaker.emoji} {speaker.name}", instruction=progress.instruction
                )
                result = await speaker.run([*ctx.history, brief(task)], ctx)
                work.append(Work(step, speaker, progress.instruction, result.text))
            else:
                final = LIMIT
        result = await self.manager.run([*ctx.history, brief(final)], ctx)
        answer = result.text
        ctx.history += [Message.user(user_message), Message.assistant(answer)]
        return answer

    async def _ask[M: BaseModel](self, model: type[M], brief: Message, ctx: SessionContext) -> M:
        result = await self.manager.run([*ctx.history, brief], ctx, response_format=model)
        try:
            return model.model_validate_json(result.text)
        except ValidationError as exc:
            self._fail(f"returned an invalid {model.__name__}: {exc.errors()[0]['msg']}", ctx)

    def _announce_task(
        self,
        ledger: TaskLedger,
        previous: TaskLedger | None,
        version: int,
        reason: str,
        ctx: SessionContext,
    ) -> None:
        now = _sections(ledger)
        before = _sections(previous) if previous else now
        ctx.emit(
            EventType.LEDGER_UPDATE,
            agent=self.manager.name,
            ledger="task",
            version=version,
            reason=reason,
            facts=ledger.facts,
            assumptions=ledger.assumptions,
            plan=ledger.plan,
            added={k: [x for x in v if x not in before[k]] for k, v in now.items()},
            removed={k: [x for x in before[k] if x not in v] for k, v in now.items()},
        )

    def _announce_progress(
        self,
        progress: ProgressLedger,
        step: int,
        stalls: int,
        action: Action,
        ctx: SessionContext,
    ) -> None:
        speaker = self.specialists[progress.next_speaker] if action == "delegate" else None
        ctx.emit(
            EventType.LEDGER_UPDATE,
            agent=self.manager.name,
            ledger="progress",
            step=step,
            max_steps=self.max_steps,
            stalls=stalls,
            action=action,
            emoji=speaker.emoji if speaker else "",
            **progress.model_dump(),
        )

    def _fail(self, problem: str, ctx: SessionContext) -> NoReturn:
        message = f"manager {problem}"
        ctx.emit(EventType.ERROR, agent=self.manager.name, exception=f"AgentError: {message}")
        raise AgentError(message)
