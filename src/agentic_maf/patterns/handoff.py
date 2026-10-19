from dataclasses import dataclass, field
from typing import Any, Self

from agent_framework import Agent, SupportsChatGetResponse, Workflow, WorkflowEvent
from agent_framework.orchestrations import HandoffAgentUserRequest, HandoffBuilder

from agentic_maf.agents import SPECIALISTS, create_specialist, trip_context
from agentic_maf.console import Console

ENTRY = "consultant"

# Who each agent may hand control to.
GRAPH: dict[str, tuple[str, ...]] = {
    ENTRY: ("flight", "hotel", "activities", "budget"),
    "flight": (ENTRY, "hotel", "budget"),
    "hotel": (ENTRY, "flight", "budget"),
    "activities": (ENTRY, "budget"),
    "budget": (ENTRY,),
}

CONSULTANT = """\
You are the Travel Consultant, the traveller's only point of contact in a travel team with \
Flight, Hotel, Activities and Budget specialists. Control moves through the team by handoffs; \
you can hand off to {targets}.
{context}

## Rules
- Make sure you know the destination, dates, number of travellers, origin, budget and interests. \
Assume what you can and state your assumptions.
- You have no search or pricing tools. Never invent flights, hotels, activities, prices or \
availability; they come from the specialists.
- New trip request: once the brief is clear, hand off to the first specialist needed (usually \
flight). They pass the work along and hand back to you.
- When control comes back to you, read the whole conversation: resolve the open points, hand off \
again if something is missing (e.g. the program, or Budget's check), or ask the traveller when \
only they can decide (e.g. go over budget or change area?).
- A final plan has flights, accommodation, a day-by-day program, a cost table in CHF and a clear \
statement of whether it fits the budget. Present it only when flights, hotel, program and \
Budget's check are in the conversation. Never do arithmetic yourself.
- A plain message, one without a handoff call, ends your turn and the traveller answers: send one \
to ask a question, present the plan or just reply."""

SPECIALIST = """\
## Handoffs
Control moves through the team by handoffs; you can hand off to {targets}. Whoever takes over \
sees the whole conversation, but not your tool results, so put what they need in your message.
- Do your part with your tools first, write your recommendation, then hand off to whoever should \
continue in the same turn. Hand off at once when the request is outside your domain.
- Hand back to an agent that already worked only for a concrete conflict they can fix (e.g. a \
late arrival vs. check-in), never just to re-check their work.
- You cannot talk to the traveller. When your work needs a decision only they can make (e.g. \
going over budget), say so in your message and hand off to the consultant.
- Always end by handing off: a message without a handoff call gives the turn to the traveller."""


def build_workflow(client: SupportsChatGetResponse[Any]) -> Workflow:
    consultant = Agent(
        client,
        CONSULTANT.format(targets=", ".join(GRAPH[ENTRY]), context=trip_context()),
        name=ENTRY,
        description="the traveller's main contact: decisions only the traveller can make, "
        "and the final plan",
        require_per_service_call_history_persistence=True,
    )
    agents = {
        ENTRY: consultant,
        **{
            name: create_specialist(
                name, client, rules=SPECIALIST.format(targets=", ".join(GRAPH[name]))
            )
            for name in SPECIALISTS
        },
    }
    builder = HandoffBuilder(name="travel_handoff", participants=list(agents.values()))
    builder.with_start_agent(consultant)
    for source, targets in GRAPH.items():
        builder.add_handoff(agents[source], [agents[t] for t in targets])
    return builder.build()


@dataclass
class Handoff:
    """A handoff workflow; the agent that last asked the traveller keeps control across turns."""

    client: SupportsChatGetResponse[Any]
    workflow: Workflow
    console: Console = field(default_factory=Console)
    pending: WorkflowEvent[Any] | None = None  # the traveller's open request

    @classmethod
    def create(
        cls, client: SupportsChatGetResponse[Any], *, console: Console | None = None
    ) -> Self:
        return cls(client, build_workflow(client), console or Console())

    @property
    def holder(self) -> str:
        return self.pending.source_executor_id if self.pending else ENTRY

    @property
    def prompt(self) -> str:
        return f"[{self.holder}] you>"

    async def turn(self, text: str) -> None:
        """Run one user message, printing the workflow's events as they happen."""
        request, self.pending = self.pending, None
        if request is None:
            stream = self.workflow.run(text, stream=True)
        else:
            reply = {request.request_id: HandoffAgentUserRequest.create_response(text)}
            stream = self.workflow.run(responses=reply, stream=True)
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, HandoffAgentUserRequest):
                self.pending = event
            else:
                self.console.show(event)
        await stream.get_final_response()
        self.console.finish()

    async def reset(self) -> None:
        self.workflow = build_workflow(self.client)
        self.pending = None
