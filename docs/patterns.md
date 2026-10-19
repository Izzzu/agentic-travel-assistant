# Multi-Agent Orchestration Patterns

This project gives an overview and comparison of orchestration patterns for multi-agent systems.

It's tempting to reach for the most powerful pattern every time, but each one comes with its own trade-offs in coordination, efficiency, and flexibility.

This repo implements a fictional multi-agent travel assistant to show these patterns in action. To run the agent follow the instructions in the [README](https://github.com/Izzzu/agentic-travel-assistant#readme).

Libraries such as Microsoft Agent Framework (the successor to AutoGen and Semantic Kernel) already help with implementing multi-agent orchestration. This repo uses a custom implementation instead, for learning purposes. The Handoff and Magentic patterns are also built with Agent Framework for comparison, see [Agent Framework version](https://github.com/Izzzu/agentic-travel-assistant#agent-framework-version).

## Demo purpose

This repository demonstrates each pattern with minimal implementation complexity. The agents are LLMs with different prompts, but they run in the same process and environment. To learn how separately deployed agents can interact, see the [A2A protocol](https://a2a-protocol.org/latest/).

Below you'll find a detailed explanation and example of each pattern.

## Sequential

In the sequential pattern, agents take turns performing their tasks one after another. Each agent waits for the previous one to finish before it starts, and the execution order is fixed.

This pattern suits tasks that require a strict order of execution, where each step depends on the one before it.

![Conceptual diagram](./assets/sequential.png)

Examples:
- A development loop (e.g., writing code, testing, and reviewing in a fixed sequence)

## Concurrent

A similar workflow to sequential, but agents perform their tasks simultaneously instead of waiting for one another. This pattern works well when tasks are independent, since running them in parallel can significantly improve efficiency.

Examples:
- Composing multiple independent tasks at once (e.g., fetching data from different sources in parallel)

![Conceptual diagram](./assets/concurrent.png)

## Group chat

Multiple agents interact in a shared group chat, coordinating their actions and sharing information to reach a common goal.

![Conceptual diagram](./assets/group_chat.png)

### How it works in the current implementation

A moderator agent (the Travel Consultant) coordinates the conversation and decides who speaks next. Every speaker sees the whole conversation history and answers based on all the information available.

Other ways to implement group chat:

- **Round-robin.** Everyone speaks in turn, and the moderator only decides when to stop. It's simple and predictable, but produces turns that add nothing.
- **Bidding.** Each round, every agent is asked whether it has something to add, and one of the agents that says yes speaks. This is closest to "raising a hand," but costs about four times as many LLM calls. It suits tasks like collaborative problem solving or brainstorming, e.g., "Show me how my day looks" against calendar notes and similar sources.
- **Asynchronous channel.** Agents post messages to a shared channel without requiring immediate responses. The channel is a shared message log, each agent runs as its own task, and each agent decides for itself when to speak.

## Handoff

Agents pass control directly to allowed targets in `HANDOFF_GRAPH`. Each agent sees the full conversation and can hand off with a reason or reply to end the turn.

![Conceptual diagram](./assets/handoff.png)

1. The Consultant receives the request.
2. The active agent uses its specialist tools.
3. It hands control to an allowed agent with a reason.
4. The next agent sees the full conversation and continues.
5. Handoffs repeat until an agent replies directly.
6. That agent remains active for the next user message.
7. After eight handoffs, the current agent must reply with partial results.

### Similarities and Differences with Group Chat

This handoff implementation can be considered a subtype of group chat: agents share the full conversation history but manage turns differently. In group chat, the Consultant decides who speaks next. In handoff, each active agent can transfer control directly to another allowed agent. Handoff is therefore a form of group chat with decentralized, graph-constrained turn management.

## Magentic

### Conceptual diagram

![Conceptual diagram](./assets/magentic.png)

How a turn works:

1. The Travel Consultant, acting as manager, writes the task ledger (facts, assumptions, plan). If the plan is empty, as for a greeting, it just replies and no specialist runs.
2. Before each step, it writes a progress ledger: is the request done, is it stuck in a loop, is progress being made, does the traveller need to provide essential details or choose between options, and who acts next with what instruction. If user input is required, it picks the `user_proxy` agent (no LLM, no tools) as the next speaker, with one batched question as the instruction. The answer lands in the work history, and the Consultant re-plans from it before delegating.
3. It re-plans when a result breaks the plan, or when the stall counter goes above `MAX_STALLS = 2`. A step counts as stalled when there's a loop or no progress. Productive steps leave the counter unchanged; a re-plan resets it. The new ledger updates the facts and assumptions and writes a new plan.
4. At the end, it writes the final plan. If `--max-steps` runs out first, it writes the plan anyway, with a warning.
5. Specialists receive the request, the current ledger, all work done so far, and their own instruction.

### Implementation Flow diagram

```mermaid
flowchart TD
	request[Traveller request] --> task[Consultant creates task ledger<br/>facts, assumptions, plan]
	task --> hasPlan{Plan has specialist steps?}
	hasPlan -- No --> direct[Consultant replies directly]
	hasPlan -- Yes --> progress[Consultant creates progress ledger]
	progress --> satisfied{Request satisfied?}
	satisfied -- Yes --> final[Consultant writes final plan<br/>and may ask the traveller]
	satisfied -- No --> replan{Plan invalidated or<br/>stalls greater than 2?}
	replan -- Yes --> update[Consultant updates task ledger<br/>and resets stall counter]
	replan -- No --> select[Select next specialist<br/>with a concrete instruction]
	select --> specialist[Flight, Hotel, Activities, or Budget<br/>runs tools and returns one result]
	specialist --> record[Append result to work history]
	update --> steps{Steps remaining?}
	record --> steps
	steps -- Yes --> progress
	steps -- No --> limit[Consultant writes final plan<br/>with unresolved-item warning]
	direct --> answer[Answer to traveller]
	final --> answer
	limit --> answer
```

## Consultant

In each pattern, the Travel Consultant plays a different role. For a clean separation of responsibilities, only the Travel Consultant has a tool to interact with the user (in Magentic, a separate `user_proxy` agent does it instead). The table below shows the role it plays in each pattern.

| Pattern | Consultant role | How it acts |
|---|---|---|
| Sequential | Writer | Speaks once at the end and can't send work back |
| Concurrent | Aggregator | Merges outputs from specialists who never saw each other |
| Group chat | Moderator | Chooses every speaker and can call on the same specialist again to fix a problem |
| Handoff | Coordinator | Delegates tasks to specialists and manages the handoff between them |
| Magentic | Orchestrator | Coordinates multiple agents in a magnetic-like manner, attracting and repelling tasks to optimize workflow |

## Results - happy path

Here is a comparison of the orchestration patterns, run once each on the same Lisbon weekend planning request:

> "Plan a 3-day Lisbon weekend, 9–11 October, for 3 friends. Budget CHF 1,500 total. They love food and the beach."

This is not a formal evaluation, just a stats comparison from a single run for illustration. A reliable evaluation would need multiple runs to account for variability in the agents' responses.

This is a happy path - all subtasks can be handled without issues.

### Time and cost

| Pattern | Turn time | LLM calls | Total tokens | Cost vs sequential |
|---|--:|--:|--:|--:|
| Sequential | 152 s | 13 | 38,890 | 1.00 |
| Concurrent | 222 s (45 s parallel + 177 s aggregator) | 12 | 23,847 | 0.61 |
| Group chat | 120 s | 19 | 52,829 | 1.36 |
| Handoff | 73 s | 13 | 41,406 | 1.06 |
| Magentic | 148 s | 20 | 68,452 | 1.76 |

### How well it solved the problem

| Pattern | Hit the sold-out hotel | Complete plan | Total vs CHF 1,500 | Verdict |
|---|---|---|---|---|
| Sequential | Yes | Yes | ✅ CHF 1,455 (CHF 45 left) | Good, very tight buffer |
| Concurrent | No | No: Budget priced nothing | ❓ Not checked | Poor: Activities treated CHF 1,500 as its own budget, and the Sunday program clashed with the 11:45 flight |
| Group chat | Yes | Yes, plus a priced cheaper fallback | ✅ CHF 1,350 (CHF 150 left) | Best overall |
| Handoff | Yes | No: Activities never ran | ⚠️ CHF 1,020, flights and hotel only | Fastest but incomplete: Budget replied to the traveller and control never went back to the Consultant |
| Magentic | No | Yes | ✅ CHF 1,350 (CHF 150 left) | Good, but the most expensive (no re-plan) |

## Results - edge case

Query that needs a replan, hotels are not available for the requested dates.

> Plan a 3-day Lisbon weekend, 6–8 November, for 3 friends. Budget CHF 1,500 total. They love food and the beach.

All Lisbon hotels are sold out on 6–7 November, so no pattern can finish the plan as requested. The patterns that can ask the traveller get to change the dates; the others can only report the gap. In the Handoff and Magentic runs, the traveller picked 30 Oct–1 Nov when asked.

### Time and cost

| Pattern | Turn time | LLM calls | Total tokens | Cost vs sequential |
|---|--:|--:|--:|--:|
| Sequential | 209 s | 16 | 52,931 | 1.00 |
| Concurrent | 114 s (53 s parallel + 61 s aggregator) | 17 | 33,672 | 0.64 |
| Group chat | 310 s | 38 | 168,372 | 3.18 |
| Handoff | 200 s (incl. 58 s waiting for the traveller) | 25 | 142,242 | 2.69 |
| Magentic | 250 s (incl. 42 s waiting for the traveller) | 36 | 152,548 | 2.88 |

### How well it solved the problem

| Pattern | Handled the sold-out hotels | Complete plan | Total vs CHF 1,500 | Verdict |
|---|---|---|---|---|
| Sequential | Reported them, but could not act on it | Itinerary yes, hotel no: priced the sold-out Cais do Sodré Loft as "if it becomes available" | ⚠️ CHF 1,467 only with the sold-out loft, CHF 1,077 confirmed | Honest but not bookable, and the writer cannot send work back |
| Concurrent | No: the hotel agent hit its 8-step limit and failed | No: no hotel, and the budget agent never priced the actual combination | ❓ Unverified (CHF 630 flights + CHF 672 activities, no lodging) | Poor: cheapest, but flagged the gap only after merging |
| Group chat | Searched wide (38 hotel tool calls across 3 runs, including nearby towns) and found nothing | No: no hotel, no dates change | ⚠️ CHF 936 without lodging, CHF 564 left | Thorough and honest, but 3.2× the cost for a plan that is still not bookable |
| Handoff | Yes: the hotel agent handed back to the Consultant, who asked the traveller and re-ran with the new dates | No: Activities never ran, and Budget replied to the traveller | ✅ CHF 867, flights and hotel only | Recovered but incomplete: same gap as in the happy path |
| Magentic | Yes: two re-plans (ask for new dates, then new dates) | Yes | ✅ CHF 1,272 (CHF 228 left) | Best: only pattern that recovered and delivered a full, bookable plan |


## Magentic vs Handoff

The Magentic and Handoff patterns are particularly useful when a task is somewhat ambiguous, or when the request specifies the desired output but not the exact steps to get there.

The choice between them should be backed by evaluation across multiple runs, to account for variability in the agents' responses and the specific requirements of the task at hand.

Handoff requires agents to know about each other in order to delegate tasks and manage the handoff process, which couples the agents. Many systems instead use a central orchestrator to handle coordination. This reduces the need for direct agent-to-agent knowledge and makes it easier to plug in new agents or change the orchestration logic without modifying the agents themselves. 

## Combining orchestration patterns

If the orchestration task is very broad or complex, it's useful to combine more than one pattern and choose between them depending on the task profile. 
