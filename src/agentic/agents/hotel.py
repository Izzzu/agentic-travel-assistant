from agentic.agents.common import specialist
from agentic.core.agent import Agent
from agentic.llm.base import LLMClient
from agentic.tools.hotel import check_availability, search_hotels

DOMAIN = """\
Accommodation for the whole group, in one room or apartment unless told otherwise.
- Always confirm your pick with check_availability before recommending it. If it is sold out, \
say so clearly and move on to the next best option.
- Mention the area and distance to the center, and the total for all nights."""


def create(llm: LLMClient) -> Agent:
    return specialist(
        name="hotel",
        emoji="🏨",
        title="Hotel",
        domain=DOMAIN,
        tools=[search_hotels, check_availability],
        llm=llm,
    )
