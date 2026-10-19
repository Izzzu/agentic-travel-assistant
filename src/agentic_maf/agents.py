import datetime as dt
from dataclasses import dataclass
from typing import Any

from agent_framework import Agent, SupportsChatGetResponse

from agentic.core.tools import Tool
from agentic.tools.activities import get_opening_hours, search_activities
from agentic.tools.budget import calculate_total, check_budget, suggest_savings
from agentic.tools.flight import get_flight_details, search_flights
from agentic.tools.hotel import check_availability, search_hotels
from agentic_maf.tools import bridge


def trip_context() -> str:
    today = dt.datetime.now(dt.UTC).astimezone().date()
    return (
        f"Today is {today:%A, %d %B %Y} ({today.isoformat()}). Dates given without a year mean"
        " their next occurrence. Tools take dates as YYYY-MM-DD. All prices are in CHF."
    )


INSTRUCTIONS = """\
You are the {title} specialist in a travel team.
{context}

## Your domain
{domain}

## Rules
- Stay in your domain; leave other topics to the other specialists.
- Get every fact, price, time and id from your tools. Never invent them.
- Make reasonable assumptions instead of stopping, and state them.
- You cannot talk to the traveller. If only they can decide something, say so at the end of \
your reply.

## Output
Concise Markdown: your recommendation first, then key numbers in CHF (per person and total) and \
the ids of the options you used, then your assumptions and a short fallback option."""


@dataclass(frozen=True)
class Specialist:
    about: str  # one line for whoever decides who acts next
    domain: str
    tools: tuple[Tool[..., Any], ...]


SPECIALISTS: dict[str, Specialist] = {
    "flight": Specialist(
        about="flights and fares",
        domain="""\
Round-trip flights for the group. Use the origin and destination the traveller gave.
- Compare price, departure and arrival times, and stops.
- Flag arrivals after 20:00 and connections: they cost the group most of day one.
- Use get_flight_details to confirm baggage and fare rules of the option you recommend.""",
        tools=(search_flights, get_flight_details),
    ),
    "hotel": Specialist(
        about="accommodation and availability",
        domain="""\
Accommodation for the whole group, in one room or apartment unless told otherwise.
- Always confirm your pick with check_availability before recommending it. If it is sold out, \
say so clearly and move on to the next best option.
- Mention the area and distance to the center, and the total for all nights.""",
        tools=(search_hotels, check_availability),
    ),
    "activities": Specialist(
        about="the day-by-day food and beach program",
        domain="""\
A day-by-day program of activities, restaurants and beaches that matches the group's interests.
- Only plan an activity on a day it is open: search with a date or use get_opening_hours.
- Mix paid and free or low-cost options, and keep each day's areas close together.
- Give the price per person and the total for the group.""",
        tools=(search_activities, get_opening_hours),
    ),
    "budget": Specialist(
        about="totals, budget checks and savings",
        domain="""\
Trip costs against the budget.
- Never do arithmetic yourself: use calculate_total, then check_budget.
- Give every cost item the ref_id of its flight, hotel or activity.
- When the plan is over budget, say by how much and use suggest_savings to show what closes \
the gap.""",
        tools=(calculate_total, check_budget, suggest_savings),
    ),
}


def create_specialist(
    name: str, client: SupportsChatGetResponse[Any], *, rules: str = ""
) -> Agent[Any]:
    """An Agent Framework agent for one specialist; `rules` is appended to its instructions."""
    specialist = SPECIALISTS[name]
    instructions = INSTRUCTIONS.format(
        title=name.capitalize(), context=trip_context(), domain=specialist.domain
    )
    return Agent(
        client,
        f"{instructions}\n\n{rules}" if rules else instructions,
        name=name,
        description=specialist.about,
        tools=bridge(specialist.tools),
        require_per_service_call_history_persistence=True,  # handoff workflows require it
    )
