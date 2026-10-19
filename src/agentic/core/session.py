import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentic.core.messages import Message, Usage
from agentic.io.user_io import UserIO
from agentic.logs.bus import EventBus
from agentic.logs.events import Event, EventType


def new_session_id(pattern: str) -> str:
    return f"{datetime.now(UTC).astimezone():%Y-%m-%d_%H%M}_{pattern}_{secrets.token_hex(2)}"


@dataclass
class SessionContext:
    pattern: str
    bus: EventBus
    io: UserIO
    session_id: str = ""
    history: list[Message] = field(default_factory=list[Message])
    usage: Usage = field(default_factory=Usage)
    _seq: int = field(default=0, init=False)
    _started: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = new_session_id(self.pattern)

    def emit(
        self,
        type: EventType,
        *,
        agent: str | None = None,
        run_id: str | None = None,
        **payload: Any,
    ) -> Event:
        self._seq += 1
        event = Event(
            seq=self._seq,
            session_id=self.session_id,
            timestamp=datetime.now(UTC),
            pattern=self.pattern,
            agent=agent,
            run_id=run_id,
            type=type,
            payload=payload,
        )
        self.bus.publish(event)
        return event

    def start(self, **info: Any) -> None:
        self._started = time.perf_counter()
        self.emit(EventType.SESSION_START, **info)

    def end(self) -> None:
        self.emit(
            EventType.SESSION_END,
            duration_ms=round((time.perf_counter() - self._started) * 1000, 1),
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
        )
