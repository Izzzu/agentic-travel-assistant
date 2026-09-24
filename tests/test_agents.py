import datetime as dt

import pytest

from agentic.agents import consultant
from agentic.agents.registry import AgentName, get_agent
from agentic.llm.scripted import ScriptedLLM

TOOLS = {
    "consultant": {"ask_user"},
    "flight": {"search_flights", "get_flight_details"},
    "hotel": {"search_hotels", "check_availability"},
    "activities": {"search_activities", "get_opening_hours"},
    "budget": {"calculate_total", "check_budget", "suggest_savings"},
}


@pytest.mark.parametrize("name", list(AgentName))
def test_each_agent_has_its_own_tools(name: AgentName) -> None:
    agent = get_agent(name, ScriptedLLM([]))
    assert agent.name == name
    assert agent.emoji
    assert {t.name for t in agent.tools} == TOOLS[name]
    assert dt.datetime.now(dt.UTC).astimezone().date().isoformat() in agent.instructions


def test_only_the_consultant_can_ask_the_user() -> None:
    owners = {
        name
        for name in AgentName
        if "ask_user" in {t.name for t in get_agent(name, ScriptedLLM([])).tools}
    }
    assert owners == {"consultant"}


@pytest.mark.parametrize("name", ["flight", "hotel", "activities", "budget"])
def test_specialists_report_open_questions_instead_of_asking(name: str) -> None:
    assert "Open questions:" in get_agent(name, ScriptedLLM([])).instructions


def test_consultant_role_is_set_by_the_pattern() -> None:
    agent = consultant.create(ScriptedLLM([]), role="You are the writer.")
    assert agent.instructions.endswith("## Your role in this session\nYou are the writer.")


def test_unknown_agent() -> None:
    with pytest.raises(LookupError, match="choose one of: consultant, flight"):
        get_agent("pilot", ScriptedLLM([]))
