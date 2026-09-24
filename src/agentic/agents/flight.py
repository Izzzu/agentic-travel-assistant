from agentic.agents.common import specialist
from agentic.core.agent import Agent
from agentic.llm.base import LLMClient
from agentic.tools.flight import get_flight_details, search_flights

DOMAIN = """\
Round-trip flights between Zurich (ZRH) and Lisbon (LIS). Assume the group flies from Zurich \
when no origin is given.
- Compare price, departure and arrival times, and stops.
- Flag arrivals after 20:00 and connections: they cost the group most of day one.
- Use get_flight_details to confirm baggage and fare rules of the option you recommend."""


def create(llm: LLMClient) -> Agent:
    return specialist(
        name="flight",
        emoji="✈️",
        title="Flight",
        domain=DOMAIN,
        tools=[search_flights, get_flight_details],
        llm=llm,
    )
