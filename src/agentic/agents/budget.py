from agentic.agents.common import specialist
from agentic.core.agent import Agent
from agentic.llm.base import LLMClient
from agentic.tools.budget import calculate_total, check_budget, suggest_savings

DOMAIN = """\
Trip costs against the budget.
- Never do arithmetic yourself: use calculate_total, then check_budget.
- Give every cost item the ref_id of its flight, hotel or activity.
- When the plan is over budget, say by how much and use suggest_savings to show what closes \
the gap."""


def create(llm: LLMClient) -> Agent:
    return specialist(
        name="budget",
        emoji="💰",
        title="Budget",
        domain=DOMAIN,
        tools=[calculate_total, check_budget, suggest_savings],
        llm=llm,
    )
