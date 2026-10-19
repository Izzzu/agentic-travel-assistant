from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from agentic.logs.events import Event, EventType


@dataclass
class AgentStats:
    runs: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class SessionStats:
    """Aggregates events into the numbers shown in summary.json and the console table."""

    session_id: str = ""
    pattern: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    completed: bool = False
    agents: dict[str, AgentStats] = field(default_factory=dict[str, AgentStats])

    def update(self, event: Event) -> None:
        if self.started_at is None:
            self.session_id, self.pattern, self.started_at = (
                event.session_id,
                event.pattern,
                event.timestamp,
            )
        self.ended_at = event.timestamp
        if event.type is EventType.SESSION_END:
            self.completed = True
        if event.agent is None:
            return
        stats = self.agents.setdefault(event.agent, AgentStats())
        match event.type:
            case EventType.AGENT_END:
                stats.runs += 1
                stats.ms += float(event.payload.get("ms", 0.0))
            case EventType.LLM_CALL:
                stats.llm_calls += 1
                stats.prompt_tokens += int(event.payload.get("prompt_tokens", 0))
                stats.completion_tokens += int(event.payload.get("completion_tokens", 0))
            case EventType.TOOL_CALL:
                stats.tool_calls += 1
            case _:
                pass

    @property
    def wall_ms(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000

    @property
    def total(self) -> AgentStats:
        total = AgentStats()
        for s in self.agents.values():
            total.runs += s.runs
            total.llm_calls += s.llm_calls
            total.tool_calls += s.tool_calls
            total.prompt_tokens += s.prompt_tokens
            total.completion_tokens += s.completion_tokens
            total.ms += s.ms
        return total

    def to_dict(self) -> dict[str, Any]:
        total = self.total
        return {
            "session_id": self.session_id,
            "pattern": self.pattern,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "completed": self.completed,
            "wall_ms": round(self.wall_ms, 1),
            "agent_ms_sum": round(total.ms, 1),
            "llm_calls": total.llm_calls,
            "tool_calls": total.tool_calls,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "total_tokens": total.total_tokens,
            "agents": {
                name: {**asdict(s), "total_tokens": s.total_tokens}
                for name, s in self.agents.items()
            },
        }
