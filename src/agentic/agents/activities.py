from agentic.agents.common import specialist
from agentic.core.agent import Agent
from agentic.llm.base import LLMClient
from agentic.tools.activities import get_opening_hours, search_activities

DOMAIN = """\
A day-by-day program of activities, restaurants and beaches that matches the group's interests.
- Only plan an activity on a day it is open: search with a date or use get_opening_hours.
- Mix paid and free or low-cost options, and keep each day's areas close together.
- Give the price per person and the total for the group."""


def create(llm: LLMClient) -> Agent:
    return specialist(
        name="activities",
        emoji="🎭",
        title="Activities",
        domain=DOMAIN,
        tools=[search_activities, get_opening_hours],
        llm=llm,
    )
