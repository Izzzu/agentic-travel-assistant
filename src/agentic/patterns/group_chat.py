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
MAX_ROUNDS = 10


class ModeratorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict structured output needs a closed schema

    next_speaker: Speaker | None
    reason: str
    finished: bool
    final_answer: str | None


MODERATOR = """\
You moderate a group chat with {team}. Everyone sees the same chat; a specialist speaks only \
when you pick them, one at a time, and can build on or challenge earlier messages.
- After every message, pick the next speaker or finish. Answer with the decision JSON only.
- Pick whoever moves the plan forward most: fill a missing piece, fix a conflict, or react to \
another specialist. Budget should check the prices of the options actually chosen; when the plan \
is over budget, ask the others for cheaper options (another area, free activities) and let \
Budget check again.
- `reason` is shown to everyone: write it as a short instruction to the next speaker.
- Finish when the plan is complete and consistent and Budget confirmed it fits the budget, or \
when more rounds would not help. Then set finished to true, next_speaker to null, and write the \
final plan in final_answer from the chat only. Never do arithmetic yourself; if the plan is \
still over budget, put a clear warning with Budget's overrun at the top.
- A message that needs no specialists (a greeting, a question about the trip so far): finish at \
once and reply in final_answer.
- If only the traveller can decide something, use ask_user before you decide."""

TURN = """\
## Your turn, {speaker}
The moderator picked you: {reason}
Reply to the chat above for your domain. Build on or challenge earlier messages instead of \
repeating them, and use your tools for every new fact or price."""

DECIDE = "## Round {round}/{max_rounds}\nPick the next speaker, or finish with the final plan."

FORCE = """\
## Round limit reached
The discussion is over. Set finished to true and write the final plan in final_answer from the \
chat so far, with a clear warning about anything still unresolved or over budget."""


@dataclass
class Post:
    speaker: Agent
    reason: str
    text: str


def _brief(request: str, posts: Sequence[Post], moderator: Agent, instruction: str) -> str:
    lines = [
        f"**{moderator.emoji} {moderator.name} → {p.speaker.emoji} {p.speaker.name}:** "
        f"{p.reason}\n\n**{p.speaker.emoji} {p.speaker.name}:**\n{p.text}"
        for p in posts
    ]
    chat = "\n\n".join(lines) if lines else "_No messages yet._"
    return f"{request}\n\n## Group chat\n\n{chat}\n\n{instruction}"


@dataclass
class GroupChat:
    """Specialists discuss in a shared chat; the moderator picks each speaker and ends the chat."""

    participants: dict[str, Agent]
    moderator: Agent
    max_rounds: int = MAX_ROUNDS
    name: str = "group_chat"

    @classmethod
    def create(cls, llm: LLMClient, *, max_rounds: int = MAX_ROUNDS) -> Self:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        specialists = [get_agent(name, llm) for name in get_args(Speaker)]
        team = ", ".join(f"{a.emoji} {a.name}" for a in specialists)
        moderator = consultant.create(llm, role=MODERATOR.format(team=team))
        return cls({a.name: a for a in specialists}, moderator, max_rounds)

    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str:
        posts: list[Post] = []
        for round_no in range(1, self.max_rounds + 1):
            instruction = DECIDE.format(round=round_no, max_rounds=self.max_rounds)
            decision = await self._decide(user_message, posts, instruction, ctx)
            self._announce(decision, round_no, ctx)
            if decision.finished:
                break
            if decision.next_speaker is None:
                self._fail("chose no next speaker without finishing", ctx)
            speaker = self.participants[decision.next_speaker]
            turn = TURN.format(speaker=f"{speaker.emoji} {speaker.name}", reason=decision.reason)
            brief = Message.user(_brief(user_message, posts, self.moderator, turn))
            result = await speaker.run([*ctx.history, brief], ctx)
            posts.append(Post(speaker, decision.reason, result.text))
        else:
            decision = await self._decide(user_message, posts, FORCE, ctx)
            self._announce(decision, self.max_rounds, ctx, forced=True)
        if not decision.final_answer:
            self._fail("finished without a final answer", ctx)
        answer = decision.final_answer
        ctx.history += [Message.user(user_message), Message.assistant(answer)]
        return answer

    async def _decide(
        self, request: str, posts: Sequence[Post], instruction: str, ctx: SessionContext
    ) -> ModeratorDecision:
        brief = Message.user(_brief(request, posts, self.moderator, instruction))
        result = await self.moderator.run(
            [*ctx.history, brief], ctx, response_format=ModeratorDecision
        )
        try:
            return ModeratorDecision.model_validate_json(result.text)
        except ValidationError as exc:
            self._fail(f"returned an invalid decision: {exc.errors()[0]['msg']}", ctx)

    def _announce(
        self,
        decision: ModeratorDecision,
        round_no: int,
        ctx: SessionContext,
        *,
        forced: bool = False,
    ) -> None:
        finished = decision.finished or forced
        speaker = None if finished else decision.next_speaker
        ctx.emit(
            EventType.SPEAKER_SELECTED,
            agent=self.moderator.name,
            next_speaker=speaker,
            emoji=self.participants[speaker].emoji if speaker else "",
            reason=decision.reason,
            finished=finished,
            forced=forced,
            round=round_no,
            max_rounds=self.max_rounds,
        )

    def _fail(self, problem: str, ctx: SessionContext) -> NoReturn:
        message = f"moderator {problem}"
        ctx.emit(EventType.ERROR, agent=self.moderator.name, exception=f"AgentError: {message}")
        raise AgentError(message)
