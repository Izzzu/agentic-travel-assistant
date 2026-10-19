---
title: Agent Framework version of Handoff and Magentic
description: How the Handoff and Magentic patterns look when built with Microsoft Agent Framework instead of the hand-written implementation, and what changes
author: agentic-travel-assistant
ms.date: 2026-10-12
---

The hand-written patterns in `src/agentic/patterns/` show how orchestration works. The `agentic_maf` package rebuilds two of them, Handoff and Magentic, with [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/) so you can compare the framework version with the hand-written one. It has its own agent prompts and handoff graph, adapted from the hand-written ones, and reuses only the mocked tools, mock data and settings. It runs the same trip request.

Agent Framework is the successor to both AutoGen and Semantic Kernel. An earlier version of this demo used AutoGen AgentChat; the notes below mention what changed in the move.

To install and run it, see [Agent Framework version](../README.md#agent-framework-version) in the README.

## Logging

The Agent Framework version prints a compact view of the workflow's event stream (`src/agentic_maf/console.py`), because the framework has no default console. It shows each agent's text as it streams, tool calls and results, handoffs and, for Magentic, both ledgers. There are no session folders, no event bus and no summary table.

## Layout

| File                                      | Purpose                                                          |
|-------------------------------------------|------------------------------------------------------------------|
| `src/agentic_maf/cli.py`                  | `app-maf` entry point and the chat loop                          |
| `src/agentic_maf/model.py`                | Azure OpenAI chat client on the v1 endpoint (Entra ID or API key) |
| `src/agentic_maf/tools.py`                | Bridge from our tools to Agent Framework tools                   |
| `src/agentic_maf/agents.py`               | The four specialists: prompts, tools and descriptions as agents  |
| `src/agentic_maf/console.py`              | Prints workflow events as they stream in                         |
| `src/agentic_maf/patterns/handoff.py`     | `HandoffBuilder` workflow and the turn loop                      |
| `src/agentic_maf/patterns/magentic.py`    | `MagenticBuilder` workflow, the traveller participant and the turn loop |

## Handoff with HandoffBuilder

| Hand-written                                   | Agent Framework                                                                 |
|------------------------------------------------|---------------------------------------------------------------------------------|
| `HANDOFF_GRAPH`                                | `GRAPH` in `handoff.py` (a copy), applied with `add_handoff(source, targets)`   |
| `handoff_to_<name>` tool with a `reason`       | Generated `handoff_to_<name>` tool, described by the target's description; no reason |
| Consultant is the entry agent                  | `with_start_agent(consultant)`                                                  |
| The active agent keeps control across messages | The agent that replied without a handoff receives the traveller's next message  |
| `ask_user` tool on the consultant              | None; a reply without a handoff call hands the turn to the traveller            |
| Eight handoffs, then a forced reply            | The workflow runner's iteration limit (100 supersteps), which raises an error   |

Differences to expect:

* Any agent can end a turn by replying without a handoff, so a specialist can ask the traveller directly. The specialist prompts tell them to always hand off, which keeps the traveller talking to the consultant in practice.
* Handoff tool calls and tool results are not shared between agents. Only user and agent messages are broadcast, so a specialist puts what the next agent needs in its message.
* Every agent must be built with `require_per_service_call_history_persistence=True`; the builder rejects agents without it.
* The builder disables parallel tool calls on each agent, so one response cannot trigger two handoffs.
* Agent Framework wires every agent to every other agent for context sharing. The `GRAPH` only limits who can take over.
* After a reset the workflow is built again.

## Magentic with MagenticBuilder

| Hand-written                                  | Agent Framework                                                              |
|-----------------------------------------------|------------------------------------------------------------------------------|
| Consultant as manager with two ledgers        | `StandardMagenticManager` over a manager agent, with a task and a progress ledger |
| `MAX_STEPS`                                   | `max_round_count`                                                            |
| `MAX_STALLS` and re-plan                      | `max_stall_count` (a re-plan above the limit; a productive round lowers the count, and the re-plan clears the history) |
| `ask_user` for missing details                | A `Traveller` participant that reads from the terminal                       |
| Consultant writes the final plan              | The manager writes it from `final_answer_prompt`                             |
| Conversation kept between messages            | Rebuilt into the task text on every message                                  |

Differences to expect:

* A Magentic workflow handles one task and then stops, so each message builds a new workflow and the earlier conversation travels inside the task text.
* The ledger prompts belong to Agent Framework. Both ledgers are visible in the event stream, which the console prints (AutoGen showed only the task ledger).
* The progress ledger is free-form JSON that the framework parses, retrying three times. The hand-written version asks for a strict schema.
* There is no step-limit reply. When the round limit is reached, the workflow ends with a short notice instead of a final plan written by the manager.
* The manager decides who acts next without knowing our rules, so its instructions say when the request counts as satisfied and when to pick the traveller.
* Stall detection resets the participants and re-plans, and the re-plan clears the chat history; the hand-written version keeps its work log. There is no separate `needs_replan` signal, so a sold-out hotel only triggers a re-plan after a stalled round.

## Moving from AutoGen

| AutoGen                                       | Agent Framework                                      |
|-----------------------------------------------|------------------------------------------------------|
| `AssistantAgent`, `FunctionTool` or `BaseTool` | `Agent`, `FunctionTool`                              |
| `Swarm` with `transfer_to_<name>`             | `HandoffBuilder` with `handoff_to_<name>`            |
| `MagenticOneGroupChat`                        | `MagenticBuilder`                                    |
| `UserProxyAgent`                              | A custom `BaseAgent` participant                     |
| `Console(team.run_stream(...))`               | Iterate `workflow.run(..., stream=True)` events      |
| `ReplayChatCompletionClient` in tests         | A scripted OpenAI SDK client (`tests/maf_fakes.py`)  |

Two surprises came up:

* `FunctionTool` validates arguments against the JSON schema after pydantic has already parsed them, so a `date` argument fails with "expected string, got date". The bridge passes the schema as a mapping and parses the arguments with our own args model.
* Handoff only supports `Agent` participants and needs the history persistence flag above.
