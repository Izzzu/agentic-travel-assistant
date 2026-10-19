import datetime as dt
from typing import Any

from agentic.core.agent import Agent
from agentic.core.tools import Tool
from agentic.llm.base import LLMClient


def trip_context() -> str:
    today = dt.datetime.now(dt.UTC).astimezone().date()
    return (
        f"Today is {today:%A, %d %B %Y} ({today.isoformat()}). Dates given without a year mean"
        " their next occurrence. Tools take dates as YYYY-MM-DD. All prices are in CHF."
    )


SPECIALIST_RULES = """\
## Rules
- Stay in your domain; leave other topics to the other specialists.
- Get every fact, price, time and id from your tools. Never invent them.
- Make reasonable assumptions instead of stopping, and state them.
- Do not ask the traveller questions mid-task. List anything only they can decide under a final \
`Open questions:` heading, one `- ` bullet each. Leave the heading out when there are none.

## Output
Concise Markdown: your recommendation first, then key numbers in CHF (per person and total) and \
the ids of the options you used, then your assumptions and a short fallback option."""


def specialist(
    *,
    name: str,
    emoji: str,
    title: str,
    domain: str,
    tools: list[Tool[..., Any]],
    llm: LLMClient,
) -> Agent:
    instructions = (
        f"You are the {title} specialist in a travel team led by a Travel Consultant.\n"
        f"{trip_context()}\n\n## Your domain\n{domain}\n\n{SPECIALIST_RULES}"
    )
    return Agent(name=name, emoji=emoji, instructions=instructions, llm=llm, tools=tools)
