import json
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from agentic.core.agent import Agent, AgentError
from agentic.core.messages import Message
from agentic.core.session import SessionContext
from agentic.core.tools import tool
from agentic.io.user_io import ScriptedIO
from agentic.llm.scripted import ScriptedLLM, call_tool, reply
from agentic.logs.bus import EventBus
from agentic.logs.events import Event, EventType
from agentic.logs.recorder import SessionRecorder

E = EventType


@tool
def get_weather(city: Annotated[str, Field(description="City name")]) -> dict[str, object]:
    """Current weather for a city."""
    return {"city": city, "temp_c": 21}


def _session(tmp_path: Path) -> tuple[SessionContext, SessionRecorder]:
    bus = EventBus()
    recorder = SessionRecorder(tmp_path)
    bus.subscribe(recorder)
    return SessionContext(pattern="single", bus=bus, io=ScriptedIO()), recorder


def _events(folder: Path) -> list[Event]:
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [Event.model_validate_json(line) for line in lines]


async def test_agent_run_with_tool_call_is_recorded(tmp_path: Path) -> None:
    ctx, recorder = _session(tmp_path)
    llm = ScriptedLLM(
        [
            call_tool("get_weather", {"city": "Lisbon"}, call_id="call_w"),
            reply("It is 21°C in Lisbon."),
        ]
    )
    agent = Agent(
        name="weather",
        emoji="🌤️",
        instructions="Answer weather questions.",
        llm=llm,
        tools=[get_weather],
    )

    ctx.start(model=llm.model)
    ctx.emit(E.USER_MESSAGE, text="Weather in Lisbon?")
    result = await agent.run([Message.user("Weather in Lisbon?")], ctx)
    ctx.emit(E.FINAL_ANSWER, text=result.text)
    ctx.end()

    assert result.text == "It is 21°C in Lisbon."
    assert [r.name for r in result.tool_calls] == ["get_weather"]
    assert result.usage.total_tokens == 30

    # events.jsonl: every step, in order, correlated by run_id and call_id
    folder = recorder.folder(ctx.session_id)
    events = _events(folder)
    assert [e.type for e in events] == [
        E.SESSION_START,
        E.USER_MESSAGE,
        E.AGENT_START,
        E.LLM_CALL,
        E.TOOL_CALL,
        E.TOOL_RESULT,
        E.LLM_CALL,
        E.AGENT_END,
        E.FINAL_ANSWER,
        E.SESSION_END,
    ]
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert len({e.run_id for e in events[2:8]}) == 1
    tool_call, tool_result = events[4], events[5]
    assert tool_call.payload["call_id"] == tool_result.payload["call_id"] == "call_w"
    assert tool_result.payload["ok"] is True
    assert json.loads(tool_result.payload["result"]) == {"city": "Lisbon", "temp_c": 21}
    assert events[-1].payload["total_tokens"] == 30

    # the second LLM request carries the assistant tool call and its matching result
    follow_up = llm.requests[1]
    assert follow_up[0].role == "system"
    assert follow_up[-2].tool_calls[0].id == "call_w"
    assert (follow_up[-1].role, follow_up[-1].tool_call_id) == ("tool", "call_w")
    assert llm.tools_seen[0][0].name == "get_weather"

    transcript = (folder / "transcript.md").read_text(encoding="utf-8")
    assert f"# Session {ctx.session_id}" in transcript
    assert "### 🌤️ weather" in transcript
    assert "`get_weather`" in transcript
    assert "## ✅ Final answer\n\nIt is 21°C in Lisbon." in transcript

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["pattern"] == "single"
    assert (summary["llm_calls"], summary["tool_calls"], summary["total_tokens"]) == (2, 1, 30)
    assert summary["agents"]["weather"]["runs"] == 1


async def test_tool_loop_limit_raises_and_keeps_partial_session(tmp_path: Path) -> None:
    ctx, recorder = _session(tmp_path)
    llm = ScriptedLLM([call_tool("get_weather", {"city": "Lisbon"}) for _ in range(2)])
    agent = Agent(
        name="weather", emoji="", instructions="", llm=llm, tools=[get_weather], max_iterations=2
    )

    ctx.start()
    with pytest.raises(AgentError, match="no final answer"):
        await agent.run([Message.user("Weather?")], ctx)

    folder = recorder.folder(ctx.session_id)
    assert _events(folder)[-1].type is E.ERROR
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is False
    assert summary["tool_calls"] == 2


async def test_failed_tool_call_is_returned_to_the_model(tmp_path: Path) -> None:
    ctx, _ = _session(tmp_path)
    llm = ScriptedLLM([call_tool("get_weather", {"town": "Lisbon"}), reply("Sorry.")])
    agent = Agent(name="weather", emoji="", instructions="", llm=llm, tools=[get_weather])

    result = await agent.run([Message.user("Weather?")], ctx)

    assert result.tool_calls[0].ok is False
    assert "invalid arguments" in (llm.requests[1][-1].content or "")
