from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    SESSION_START = "session_start"  # all modes (chat loop)
    SESSION_END = "session_end"  # all modes (chat loop)
    USER_MESSAGE = "user_message"  # all modes (chat loop)
    FINAL_ANSWER = "final_answer"  # all modes (chat loop)
    AGENT_START = "agent_start"  # all modes (Agent.run)
    AGENT_END = "agent_end"  # all modes (Agent.run)
    STEP = "step"  # sequential
    LLM_CALL = "llm_call"  # all modes (Agent.run)
    TOOL_CALL = "tool_call"  # all modes (Agent.run)
    TOOL_RESULT = "tool_result"  # all modes (Agent.run)
    ASK_USER = "ask_user"  # all modes, Travel Consultant only
    USER_REPLY = "user_reply"  # all modes, Travel Consultant only
    HANDOFF = "handoff"  # handoff
    SPEAKER_SELECTED = "speaker_selected"  # group chat
    LEDGER_UPDATE = "ledger_update"  # magentic
    ERROR = "error"  # all modes (Agent.run); group chat for invalid moderator decisions


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
