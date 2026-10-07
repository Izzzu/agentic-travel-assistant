---
title: AutoGen counterpart for Handoff and Magentic
description: Plan for a separate package that rebuilds the Handoff and Magentic patterns with AutoGen AgentChat on top of the existing agents and tools
ms.date: 2026-10-05
---

## Goal

Show how the two patterns that are hand-written in `src/agentic/patterns/` look when built with AutoGen:

- Handoff with `Swarm`
- Magentic with `MagenticOneGroupChat`

Same trip prompt, same five agents, same mocked tools, same surprise (sold-out hotel). Logging is whatever AutoGen prints by default (`Console` over `run_stream`); there are no session files and no custom event plumbing. The audience can compare the 200-line hand-written version with the framework version side by side.

Out of scope: Sequential, Concurrent and Group chat; changes to the hand-written patterns.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Location | New top-level package `src/agentic_autogen/` | Keeps AutoGen imports out of `agentic`; the core stays framework-free |
| Dependencies | Optional extra `autogen` (`autogen-agentchat`, `autogen-ext[openai,azure]`) | `uv sync` stays light; AutoGen is only installed on request |
| CLI | Second entry point `app-autogen --pattern handoff\|magentic` | `agentic.cli` stays untouched |
| Reuse | Import only the mocked tools and mock data from `agentic`; agent prompts, the handoff graph and the limits are copied into `agentic_autogen` | The AutoGen version reads like a normal AutoGen example, with no stub or adapter to make our agent classes fit |
| Logging | AutoGen's default only: `autogen_agentchat.ui.Console` | No `EventBus`, recorder or custom adapter to maintain |
| Asking the traveller | AutoGen mechanisms, not our `ask_user`: in Swarm the consultant's plain reply ends the turn; Magentic gets a `UserProxyAgent` | Keeps tools free of session state |
| Packaging | `module-name = ["agentic", "agentic_autogen"]` in `[tool.uv.build-backend]` | `uv_build` supports several modules; confirmed in phase 0 |
| Testing | `ReplayChatCompletionClient` from `autogen_ext.models.replay` | Offline tests, equivalent of `ScriptedLLM` |

Note: AutoGen AgentChat is the stable 0.7.x line. Microsoft Agent Framework is its successor, so the docs page should say so and link to it.

## Layout

```
src/agentic_autogen/
├── __init__.py
├── cli.py            # app-autogen entry point and a small chat loop around Console
├── config.py         # AutogenSettings(Settings): model name, api version
├── model.py          # AzureOpenAIChatCompletionClient from settings (key or Entra ID)
├── tools.py          # agentic Tool -> autogen BaseTool bridge
├── agents.py         # the four specialists as AssistantAgents (prompts, tools)
└── patterns/
    ├── handoff.py    # Swarm team and the turn loop
    └── magentic.py   # MagenticOneGroupChat team and the turn loop
tests/
├── test_autogen_tools.py
├── test_autogen_agents.py
├── test_autogen_handoff.py
└── test_autogen_magentic.py
```

The pattern modules do not use the `Pattern` protocol, `SessionContext` or `chat()` from `agentic`; each exposes a small async function that runs one user turn.

## Concept mapping

### Handoff

| Hand-written (`handoff.py`) | AutoGen |
|---|---|
| `HANDOFF_GRAPH` | `GRAPH` in `handoff.py` (a copy), applied with `AssistantAgent(handoffs=[...])` |
| `handoff_to_<name>` tool with `reason` | Generated `transfer_to_<name>` tool, `Handoff(target, message=...)` for custom text |
| Consultant as entry point | First participant of `Swarm` |
| Active agent kept across user turns | Last `HandoffMessage` target; next turn sends `HandoffMessage(source="user", target=<that agent>)` |
| Consultant asks the traveller | No `ask_user`. `TextMessageTermination(source="consultant")` ends the turn on the consultant's plain message, so a question and the final plan are both just a reply |
| `MAX_HANDOFFS` | `MaxMessageTermination`, which counts messages, not handoffs |
| Handoff limit note | Not available; the termination condition stops the run |

Known differences to document:

- Specialists always hand back to the consultant instead of replying to the traveller directly; only the consultant's plain message ends a turn. This keeps "only the consultant talks to the user".
- No model-written `reason` on a handoff.
- Set `parallel_tool_calls=False` on the model client so one turn cannot produce two handoffs.
- A consultant message that is not followed by a handoff ends the turn, so the prompt tells it to put any note in the same response as the handoff call (AutoGen then shows it as a `ThoughtEvent`).

### Magentic

| Hand-written (`magentic.py`) | AutoGen |
|---|---|
| Manager with `TaskLedger` and `ProgressLedger` | Built-in `MagenticOneOrchestrator` inside `MagenticOneGroupChat` |
| `MAX_STEPS` | `max_turns` |
| `MAX_STALLS` / re-plan | `max_stalls` |
| Consultant as manager | Orchestrator is the manager; the consultant is not a participant, and a `UserProxyAgent` (`traveller`) takes the questions |
| Final plan written by the manager | `final_answer_prompt` (present the plan, never do arithmetic, use Budget's numbers) |
| `LEDGER_UPDATE` events | Whatever `Console` prints: the task ledger message. The progress ledger is not shown |

Known differences to document:

- Ledger prompts and structure are AutoGen's, not ours, and the progress ledger is not visible in the default output.
- Participants are Flight, Hotel, Activities, Budget and the `traveller` proxy; the orchestrator is not a named agent from our registry.

## Phases

### Phase 0: Spike and dependencies

1. Add the `autogen` extra and `module-name` list to `pyproject.toml`; confirm `uv sync --extra autogen` and `uv build` include both packages.
2. Check current package versions and the Azure client signature (`AzureOpenAIChatCompletionClient` needs `azure_deployment`, `model`, `api_version`, and `api_key` or `azure_ad_token_provider`). If the v1 endpoint used by the main package is needed, fall back to `OpenAIChatCompletionClient(base_url=...)` with `model_info`.
3. Run one minimal `Swarm` and one `MagenticOneGroupChat` against the real deployment with a single mock tool, and save the raw message streams. These are the fixtures for phases 1 to 3.
4. Answer two open questions with the spike: does `MagenticOneGroupChat` keep its thread when `run_stream` is called again, and which messages carry the ledgers.

Done when: both minimal examples run and the open questions are answered in the plan.

Status: done on 2026-10-05. A throwaway script ran a minimal `Swarm` and `MagenticOneGroupChat` live and saved the raw message streams; the script and streams were deleted afterwards.

#### Findings

| Topic | Result | Consequence |
|---|---|---|
| Packaging | `uv sync --extra autogen` resolves `autogen-agentchat`/`-core`/`-ext` 0.7.5 next to `openai` 3.16.1; the built wheel contains both packages | Pyproject changes are final |
| Azure client | `AzureOpenAIChatCompletionClient` works with Entra ID (`azure_ad_token_provider`), `api_version="2024-10-21"` and an explicit `model_info`; no need for the v1 endpoint fallback | Add `azure_openai_model` and `api_version` to `AutogenSettings` |
| Model name | Passing the deployment name as `model` warns about a resolved-model mismatch | Let `AutogenSettings.azure_openai_model` override it and set it to the resolved name |
| Credential | `DefaultAzureCredential` left open prints "Unclosed client session" | `model.py` returns the client and closes the credential too |
| Tools | `FunctionTool` rejects our `Annotated[..., Field(description=...)]` parameters | Bridge through a `BaseTool` subclass that reuses `tool.args_model`, `tool.name`, `tool.description` and calls `tool.fn` (works in the spike) |
| Swarm | Works as planned: consultant, hotel (sold out, next option), consultant, `user`. Turn 2 with `HandoffMessage(source="user", target="consultant")` kept the earlier context | Phase 2 design confirmed |
| Swarm output | Specialists end with `ToolCallSummaryMessage` (raw tool JSON), not prose; the consultant writes the answer | Fine for the demo; `Console` shows it as is |
| Magentic thread | Each `run_stream(task=...)` starts a new orchestration and a new task ledger. The second run had no memory of the first ("I don't have the prior criteria") and hit the turn limit | Pass the earlier conversation (user messages and final answers) inside the task text |
| Magentic ledgers | The task ledger (facts, plan) is a `TextMessage` from `MagenticOneOrchestrator` starting with "We are working to address"; a re-plan emits it again. The progress ledger is **not** in the stream: it is only logged at DEBUG on the `autogen_agentchat` logger | Not parsed (default logging only); the demo shows the task ledger as `Console` prints it |
| Magentic usage | Orchestrator messages carry no `models_usage` | `Console` totals therefore leave out the orchestrator; note it in the docs |
| Magentic stop | `stop_reason` is the orchestrator's free-text reason, or "Max rounds reached." | Treat it as display text only |

### Phase 1: Shared bridge

1. `config.py`, `model.py`: settings and model client (Entra ID via `azure_ad_token_provider`, same as the main package).
2. `tools.py`: a `BaseTool` subclass that wraps each `Tool` (reusing `args_model`, `name` and `description`, calling `tool.fn`). Tools that need our session (`ask_user`) are left out; AutoGen's own mechanisms replace them.
3. `agents.py`: one `AssistantAgent` per specialist, with its own copy of the prompt (domain, rules, output), its tools bridged and a description for the orchestrator or router. The first version took instructions from the hand-written agents through a stub `LLMClient`; that workaround and the dependency on the hand-written agent classes were removed, so only the tools are shared.

No event adapter: AutoGen's default logging (`Console`) is the only output.

Done when: unit tests cover tool wrapping and agent construction, and a live run of one bridged agent renders in `Console`.

Status: done on 2026-10-05. `tests/test_autogen_tools.py` and `test_autogen_agents.py` pass; a throwaway script ran the hotel agent live through `Console` (sold-out detection, then Baixa Boutique). An earlier version with an event adapter and `ask_user` bridge was removed to keep the code free of logging plumbing. `uv.lock` also gained `size` fields from the local uv 0.6.9; versions are unchanged.

### Phase 2: Handoff with Swarm

1. `patterns/handoff.py`: build one `AssistantAgent` per graph node with its `handoffs` (`GRAPH`, a copy of the hand-written graph). The consultant prompt is written for AutoGen and has no `ask_user` text. Rules appended to the instructions: specialists hand off when done and never talk to the traveller; the consultant ends its turn with a plain message and puts any note in the same response as a handoff call.
2. Turn loop: first turn runs `Console(team.run_stream(task=message))`; later turns send `HandoffMessage(source="user", target=<holder>)`. Stop on `TextMessageTermination(source="consultant") | MaxMessageTermination(40)`.
3. The prompt shows the agent that holds control: the consultant after a reply, or the handoff target when the message limit stopped the run.
4. `/reset` calls `team.reset()` and returns to the consultant.

Done when: with the demo prompt the run goes consultant, flight, hotel (sold out, picks another area), budget, consultant; `/reset` works.

Status: done on 2026-10-05. `tests/test_autogen_handoff.py` passes (offline, replay client). Live run through a throwaway script on the demo prompt (origin Zurich) plus one follow-up: consultant, flight, hotel, activities, budget, consultant; Casa Alfama reported sold out, plan CHF 1,347 of 1,500; the follow-up resumed at the consultant and ended with its reply.

Design change found in the live run: with a handoff to `user` as the turn end, the consultant wrote the final plan as plain text with no handoff and Swarm kept calling it, repeating the plan 25 times until the message limit. Ending the turn on the consultant's text message removed that failure. Specialists and the consultant alternate between a `ThoughtEvent` (text plus handoff call in one response) and a `TextMessage`, both visible in `Console`.

### Phase 3: Magentic with MagenticOneGroupChat

1. `patterns/magentic.py`: participants Flight, Hotel, Activities, Budget and a `UserProxyAgent` named `traveller` that reads from the terminal; `max_turns` and `max_stalls` default to the hand-written values (15 and 2); `final_answer_prompt` describes the final plan. The consultant is not a participant: it had no tools and the orchestrator writes the final reply.
2. Turn loop: build the task from the earlier conversation (user messages and final answers), the new message and a short notes block (today's date, when the request counts as satisfied, when to pick the traveller), then run `Console(team.run_stream(task=...))`. Each turn starts a new orchestration (phase 0 finding).

Done when: with the demo prompt the orchestrator plans, delegates, handles the sold-out hotel (through re-plan or a Budget round), and the final plan stays within CHF 1,500.

Status: done on 2026-10-06. `tests/test_autogen_magentic.py` passes (offline, scripted ledgers): team composition, delegation and final answer, history in the follow-up task, reset, turn limit. Live run through a throwaway script (demo prompt with Zurich, then "swap the hotel for the cheapest available one"): flight, hotel, activities, budget, final plan; the follow-up reused the history, skipped the sold-out Casa Alfama, chose Belém Riverside and re-ran Budget (total CHF 1,380 of 1,500).

Findings from the live runs:

- The orchestrator used the `traveller` proxy to "present the plan" and blocked on terminal input, twice. Two changes fixed it: a stricter proxy description (only for questions that must be answered) and the notes block in the task (the request is satisfied once flights, hotel, program and Budget's check exist; pick the traveller only for a real question). Without the date in the notes it also asked the traveller to confirm the year.
- With `max_turns=N` the orchestrator still runs one more progress ledger and one more specialist step before it writes the final answer (the limit is checked at the start of the next step).
- `Console` shows the task ledger as an orchestrator message; the progress ledger is not printed.

### Phase 4: CLI, docs, quality

1. `cli.py` with `--pattern handoff|magentic`, `--max-turns`, `--max-stalls`; a plain input loop with `/exit` and `/reset`; entry point `app-autogen` in `[project.scripts]`.
2. README: short "AutoGen version" section with the install command (`uv sync --extra autogen`) and run commands.
3. `docs/autogen.md`: the two mapping tables above, the known differences, and a note on Microsoft Agent Framework.
4. Optionally run both versions on the demo and control prompts and compare the outcome by hand against `sessions/comparison/comparison.md`.
5. Gates: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (strict, the extra must be installed), `uv run pytest -q`.

Done when: all gates pass with and without the extra (autogen tests skip via `pytest.importorskip("autogen_agentchat")` when it is missing).

Status: done on 2026-10-06, except step 4 (the side-by-side comparison run, left optional). `app-autogen` is in `[project.scripts]`; `src/agentic_autogen/cli.py` has the chat loop (`/reset`, `/exit`, unknown commands, a failed turn does not end the chat) and `--pattern`, `--max-turns`, `--max-stalls`. `tests/test_autogen_cli.py` covers the loop, the options and missing Azure settings. README has an \"AutoGen version\" section, `docs/autogen.md` has the mapping tables and differences, and `docs/patterns.md` links to it. `.env.example` lists the optional `AZURE_OPENAI_MODEL` and `AZURE_OPENAI_API_VERSION`. Both patterns ran through the real CLI with piped input. Gates: with the extra 110 tests pass; without it 85 pass and 5 files skip. Pyright needs the extra installed.

## Tests

- Replay client scripted with replies; assert the order of agents in the returned messages and the final text.
- Handoff: next turn continues from the last agent; `/reset` returns to the consultant; the consultant's reply ends the turn and a specialist's message does not.
- Magentic: turn limit stops the run with a final answer; the history and notes are in the task; reset clears the history.

## Risks

| Risk | Mitigation |
|---|---|
| Swarm stays on one agent when it replies without a handoff | Seen live (25 repeated plans). The consultant's reply ends the turn; specialists are told to always hand off; `MaxMessageTermination` is the backstop |
| Parallel tool calls produce two handoffs | `parallel_tool_calls=False` |
| Magentic does not keep context across user turns | Confirmed in phase 0; the task carries the earlier conversation |
| Pyright strict complains about AutoGen typing | Narrow `# pyright: ignore` at the boundary (`ToolBridge`), nowhere else |
| AutoGen API drift | Pin minimum versions in the extra; the offline tests fail first when message shapes change |
