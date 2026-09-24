from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_MESSAGE = "user_message"
    FINAL_ANSWER = "final_answer"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    STEP = "step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASK_USER = "ask_user"
    USER_REPLY = "user_reply"
    HANDOFF = "handoff"
    SPEAKER_SELECTED = "speaker_selected"
    LEDGER_UPDATE = "ledger_update"
    SURPRISE = "surprise"
    ERROR = "error"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int  # monotonic per session; orders interleaved concurrent events
    session_id: str
    timestamp: datetime
    pattern: str
    agent: str | None = None
    run_id: str | None = None  # one id per Agent.run, correlates its events
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict[str, Any])
