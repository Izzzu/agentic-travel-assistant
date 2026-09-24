from collections.abc import Callable
from enum import StrEnum

from agentic.agents import activities, budget, consultant, flight, hotel
from agentic.core.agent import Agent
from agentic.llm.base import LLMClient


class AgentName(StrEnum):
    CONSULTANT = "consultant"
    FLIGHT = "flight"
    HOTEL = "hotel"
    ACTIVITIES = "activities"
    BUDGET = "budget"


FACTORIES: dict[AgentName, Callable[[LLMClient], Agent]] = {
    AgentName.CONSULTANT: consultant.create,
    AgentName.FLIGHT: flight.create,
    AgentName.HOTEL: hotel.create,
    AgentName.ACTIVITIES: activities.create,
    AgentName.BUDGET: budget.create,
}


def get_agent(name: str, llm: LLMClient) -> Agent:
    try:
        key = AgentName(name.strip().lower())
    except ValueError:
        choices = ", ".join(AgentName)
        raise LookupError(f"unknown agent {name!r}; choose one of: {choices}") from None
    return FACTORIES[key](llm)
