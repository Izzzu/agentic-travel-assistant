---
title: AutoGen version of Handoff and Magentic
description: How the Handoff and Magentic patterns look when built with AutoGen AgentChat instead of the hand-written implementation, and what changes
author: agentic-travel-assistant
ms.date: 2026-10-06
---

The hand-written patterns in `src/agentic/patterns/` show how orchestration works. The `agentic_autogen` package rebuilds two of them, Handoff and Magentic, with [AutoGen AgentChat](https://microsoft.github.io/autogen/stable/) so you can compare the framework version with the hand-written one. It has its own agent prompts and handoff graph, copies of the hand-written ones adapted to AutoGen, and reuses only the mocked tools and mock data. It runs the same trip request.

AutoGen AgentChat is the stable 0.7.x line. Microsoft Agent Framework is its successor, so check it before starting a new project.

To install and run it, see [AutoGen version](../README.md#autogen-version) in the README.

## Logging

The AutoGen version prints what AutoGen prints by default (`autogen_agentchat.ui.Console` over `run_stream`). There are no session folders, no event bus and no summary table. Tool calls, handoffs and agent messages appear in the console as AutoGen formats them.

## Layout

| File                                      | Purpose                                                              |
|-------------------------------------------|----------------------------------------------------------------------|
| `src/agentic_autogen/cli.py`              | `app-autogen` entry point and the chat loop                          |
| `src/agentic_autogen/config.py`           | Settings: model name and API version on top of the main settings     |
| `src/agentic_autogen/model.py`            | Azure OpenAI client for AutoGen (Entra ID or API key)                |
| `src/agentic_autogen/tools.py`            | Bridge from our tools to AutoGen tools                               |
| `src/agentic_autogen/agents.py`           | The four specialists: prompts, tools and descriptions as AutoGen agents |
| `src/agentic_autogen/patterns/handoff.py` | `Swarm` team and the turn loop                                       |
| `src/agentic_autogen/patterns/magentic.py`| `MagenticOneGroupChat` team and the turn loop                        |

## Handoff with Swarm

| Hand-written                                   | AutoGen                                                                                  |
|------------------------------------------------|------------------------------------------------------------------------------------------|
| `HANDOFF_GRAPH`                                | `GRAPH` in `handoff.py` (a copy), applied with `AssistantAgent(handoffs=[...])`           |
| `handoff_to_<name>` tool with a `reason`       | Generated `transfer_to_<name>` tool; no reason                                           |
| Consultant is the entry agent                  | First participant of `Swarm`                                                             |
| The active agent keeps control across messages | The last holder receives `HandoffMessage(source="user", target=<holder>)` on the next turn |
| `ask_user` tool on the consultant              | None; the consultant's plain message ends the turn (`TextMessageTermination`)            |
| Eight handoffs, then a forced reply            | `MaxMessageTermination(40)`, which counts messages, not handoffs                         |

Differences to expect:

* Specialists always hand back to the consultant. Only the consultant's plain message ends a turn, so the traveller only talks to the consultant.
* When the consultant writes a note and calls a handoff in the same response, AutoGen prints the note as a `ThoughtEvent`.
* Ending the turn on a handoff to a `user` target did not work reliably: the consultant wrote the plan without a handoff and Swarm kept calling it, repeating the plan until the message limit.
* The model client sets `parallel_tool_calls=False` so one response cannot trigger two handoffs.

## Magentic with MagenticOneGroupChat

| Hand-written                                  | AutoGen                                                                  |
|-----------------------------------------------|--------------------------------------------------------------------------|
| Consultant as manager with two ledgers        | Built-in orchestrator with a task ledger and a progress ledger           |
| `MAX_STEPS`                                   | `max_turns`                                                              |
| `MAX_STALLS` and re-plan                      | `max_stalls`                                                             |
| `ask_user` for missing details                | A `UserProxyAgent` named `traveller` that reads from the terminal        |
| Consultant writes the final plan              | The orchestrator writes it from `final_answer_prompt`                    |
| Conversation kept between messages            | Rebuilt into the task text on every message                              |

Differences to expect:

* Each message starts a new orchestration, so the earlier conversation travels inside the task text.
* The prompts behind the ledgers belong to AutoGen. The default output shows the task ledger as an orchestrator message and does not show the progress ledger.
* Orchestrator calls report no token usage, so token totals leave them out.
* The orchestrator decides who acts next without knowing our rules. A short notes block in the task gives it the date, says when the request counts as satisfied and says to pick the traveller only for a real question. Without it, the orchestrator asked the traveller to present a finished plan and to confirm the year.
* The orchestrator can plan more steps than a small request needs (a hotel question also ran flight, activities and budget).
