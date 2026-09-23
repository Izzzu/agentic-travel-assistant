# agentic-travel-assistant — Implementation Plan

A Python CLI demo showing **one problem solved by five orchestration patterns** (Sequential, Concurrent, Group chat, Handoff, Magentic) using the **same five agents** and the **same mocked tools**.

- **Repo / project:** `agentic-travel-assistant`
- **Package:** `agentic`
- **CLI command:** `app`
- **Package manager:** `uv`

---

## 1. Demo scenario

**Problem (identical for every pattern):**

> *"Plan a 3-day Lisbon weekend in October for 3 friends. Budget CHF 1,500 total. They love food and the beach."*

**Scripted surprise (identical for every pattern):** the best-value hotel (listed as the cheapest by search) returns **sold out** on `check_availability`. Every remaining naive option pushes the trip ~CHF 200 over budget, but a valid plan exists (cheaper area + free beach day instead of a paid tour).

The surprise is what makes the patterns behave differently — without it, all five produce similar plans.

### Agents

| Agent | Responsibility | Tools (mocked) |
|---|---|---|
| **Travel Consultant** | Understands the request, coordinates, presents the final plan | `ask_user` |
| ✈️ **Flight** | Finds flights ZRH ⇄ LIS | `search_flights`, `get_flight_details` |
| 🏨 **Hotel** | Finds accommodation for 3 | `search_hotels`, `check_availability` ⚡ |
| 🎭 **Activities** | Builds a food & beach program | `search_activities`, `get_opening_hours` |
| 💰 **Budget** | Totals costs, flags overruns, suggests savings | `calculate_total`, `check_budget`, `suggest_savings` |

**Only the Travel Consultant owns `ask_user`.** Specialists never interrupt the user; when they lack information they return open questions to the Consultant, which decides whether to ask the user, assume, or ask another agent.

### The Travel Consultant's role per pattern

| Pattern | Flow | Consultant role |
|---|---|---|
| Sequential | ✈️ → 🏨 → 🎭 → 💰 → Consultant | **Writer** — summarizes the result |
| Concurrent | ✈️ 🏨 🎭 💰 in parallel → Consultant | **Aggregator** — merges outputs |
| Group chat | ✈️ 🏨 🎭 💰 discuss | **Moderator** — picks next speaker, ends discussion |
| Handoff | Consultant ⇄ specialists ⇄ user | **Front door** — routes, receives control back |
| Magentic | Consultant plans & delegates | **Manager** — keeps ledgers, re-plans |

---

## 2. Requirements

1. **Good, demo-ready logging.** Every session is saved to disk.
2. **CLI first.** The pattern is picked with a flag; an interactive chat with the user then opens.
3. **Modular build**, reviewable piece by piece:
   1. Mock agents — talk to a single agent via `--agent <name>`
   2. Sequential
   3. Concurrent
   4. Group chat
   5. Handoff
   6. Magentic
4. **`ask_user` tool available in every pattern**, owned by the Travel Consultant only.
5. **`uv`** as the package manager.

### Assumptions

- "Mock agents" = real LLM agents with **mocked tools and data** (flights, hotels, prices are fake; reasoning is real).
- LLM: **Azure OpenAI** behind a small `LLMClient` protocol, so other providers can be added.
- Orchestration is **hand-written**, not taken from a framework: the patterns stay transparent for the audience and for review, and logging and `ask_user` are fully under our control. Microsoft Agent Framework builders can replace pattern modules later if desired.
- Everything is **async from day one** (required by Concurrent).

---

## 3. Project layout

```
agentic-travel-assistant/
├── pyproject.toml
├── .env.example
├── PLAN.md
├── src/agentic/
│   ├── cli.py                 # typer entry point
│   ├── config.py              # settings (pydantic-settings, .env)
│   ├── llm/
│   │   ├── base.py            # LLMClient protocol
│   │   ├── azure_openai.py    # real client
│   │   └── scripted.py        # fake client for tests
│   ├── core/
│   │   ├── agent.py           # Agent: instructions + tools + tool-call loop
│   │   ├── tools.py           # @tool decorator → JSON schema, registry
│   │   ├── messages.py        # message / result models
│   │   └── session.py         # SessionContext (id, history, io, event bus)
│   ├── io/
│   │   └── user_io.py         # UserIO protocol + ConsoleIO (with lock)
│   ├── logs/
│   │   ├── events.py          # event types (pydantic)
│   │   ├── bus.py             # event bus (publish/subscribe)
│   │   ├── recorder.py        # writes the session folder
│   │   └── console.py         # rich live rendering
│   ├── tools/
│   │   ├── shared.py          # ask_user
│   │   ├── flight.py
│   │   ├── hotel.py
│   │   ├── activities.py
│   │   └── budget.py
│   ├── mock_data/
│   │   ├── flights.json
│   │   ├── hotels.json
│   │   └── activities.json
│   ├── agents/
│   │   ├── registry.py        # name → factory
│   │   ├── consultant.py
│   │   ├── flight.py
│   │   ├── hotel.py
│   │   ├── activities.py
│   │   └── budget.py
│   └── patterns/
│       ├── base.py            # Pattern protocol
│       ├── single.py          # --agent mode
│       ├── sequential.py
│       ├── concurrent.py
│       ├── group_chat.py
│       ├── handoff.py
│       └── magentic.py
├── tests/
└── sessions/                  # gitignored — one folder per session
```

Each phase adds files without rewriting earlier ones.

---

## Phase 0 — Project setup

```bash
uv init --package agentic-travel-assistant
cd agentic-travel-assistant
mv src/agentic_travel_assistant src/agentic
uv add openai typer rich pydantic pydantic-settings python-dotenv
uv add --dev pytest pytest-asyncio ruff pyright
```

`pyproject.toml`:

```toml
[project]
name = "agentic-travel-assistant"

[project.scripts]
app = "agentic.cli:app"

[tool.uv.build-backend]
module-name = "agentic"

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "strict"
venvPath = "."
venv = ".venv"
```

`.env.example`:

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
LOG_LEVEL=INFO
```

Add `sessions/` and `.env` to `.gitignore`.

**Done when:** `uv run app --help` works; `uv run ruff check`, `uv run pyright` and `uv run pytest` run clean.

---

## Phase 1 — Core, logging, sessions

Built first, because every later piece emits events.

### Event model (`logs/events.py`)

Every event carries: `session_id`, `timestamp`, `pattern`, `agent` (if any), `type`, `payload`.

| Event | Payload |
|---|---|
| `session_start` / `session_end` | pattern, CLI args, model, duration, total tokens |
| `user_message` / `final_answer` | text |
| `agent_start` / `agent_end` | agent, input summary, output, tokens, ms |
| `llm_call` | agent, model, prompt tokens, completion tokens, ms |
| `tool_call` / `tool_result` | agent, tool, args, result, ms |
| `ask_user` / `user_reply` | agent, question, answer |
| `handoff` | from, to, reason |
| `speaker_selected` | next speaker, reason (group chat) |
| `ledger_update` | task ledger / progress ledger (Magentic) |
| `surprise` | description, e.g. "hotel sold out" |
| `error` | agent, exception |

### Event bus (`logs/bus.py`)

Agents and patterns only publish events. The **console renderer** and the **session recorder** are subscribers — neither is called directly by agent code.

### Session recorder (`logs/recorder.py`)

Each run creates a folder:

```
sessions/2026-09-23_1124_sequential_a1b2/
├── events.jsonl      # every event, machine-readable
├── transcript.md     # human-readable conversation + agent steps
└── summary.json      # pattern, duration, tokens, per-agent stats
```

Files are written incrementally (append per event), so a crashed session is still saved.

### Console renderer (`logs/console.py`)

Using `rich`:

- One color + emoji per agent; indented tool calls; highlighted `surprise` events.
- End-of-session summary table: time, tokens, LLM calls, tool calls per agent.
- Verbosity:
  - default — agent steps, handoffs, final answer (clean demo view)
  - `--verbose` / `-v` — tool arguments and results, LLM call stats

### Agent (`core/agent.py`)

```python
class Agent:
    name: str
    emoji: str
    instructions: str
    tools: list[Tool]

    async def run(self, messages: list[Message], ctx: SessionContext) -> AgentResult:
        # loop: call LLM → execute tool calls → feed results back → until final text
        # emits agent_start, llm_call, tool_call / tool_result, agent_end
```

- Max tool-call iterations per run (guard against loops).
- `AgentResult` contains final text, tool calls made, and token usage.

### Tools (`core/tools.py`)

- `@tool` decorator builds the JSON schema from type hints and the docstring.
- Supports sync and async functions.
- The wrapper logs `tool_call` / `tool_result` automatically — tools themselves contain no logging.

### LLM clients (`llm/`)

- `LLMClient` protocol: `async def complete(messages, tools, response_format=None) -> LLMResponse`.
- `AzureOpenAIClient` — real implementation.
- `ScriptedLLM` — returns predefined responses and tool calls, for offline deterministic tests.

**Done when:** a test using `ScriptedLLM` runs an agent through a tool call and produces a correct `events.jsonl`, `transcript.md` and `summary.json`.

---

## Phase 2 — Mock tools and data

### `ask_user` (`tools/shared.py`)

- Given **only to the Travel Consultant**, in every pattern.
- Calls `ctx.io.ask(agent_name, question)`; logs `ask_user` and `user_reply`.
- Rule: `ask_user` **interrupts a task** to get input. Normal conversation with the user happens through the chat loop (an agent's reply ends the turn, the user answers) and needs no tool.
- `ConsoleIO` still holds an **`asyncio.Lock`** as a safety net for concurrent calls.
- `UserIO` is a protocol, so tests use a `ScriptedIO`.

### Open questions from specialists

Specialists return an **`Open questions:`** section in their output when they lack information (e.g. "Is a shared room OK?"). The Consultant reads it and decides: ask the user, make an assumption, or ask another agent. Each open question is logged in the `agent_end` payload.

### Domain tools

| Agent | Tools |
|---|---|
| ✈️ Flight | `search_flights(origin, dest, depart, return_date, pax)`, `get_flight_details(flight_id)` |
| 🏨 Hotel | `search_hotels(city, checkin, checkout, guests, max_price=None)`, `check_availability(hotel_id)` ⚡ |
| 🎭 Activities | `search_activities(city, interests, date=None)`, `get_opening_hours(activity_id)` |
| 💰 Budget | `calculate_total(items)`, `check_budget(total, limit)`, `suggest_savings(items, target)` |

- All tools are deterministic and read from `mock_data/*.json`.
- **Budget math happens in Python, never in the LLM.**

### The surprise

- `check_availability` returns `sold_out` for the best-value hotel (e.g. "Casa Alfama"), which `search_hotels` lists as the cheapest, and emits a `surprise` event.
- Controlled via `--surprise / --no-surprise` (default: on).

### Mock data design

- Flights: several ZRH ⇄ LIS options with varying times and prices, including one late arrival (22:00) to create timing conflicts.
- Hotels: 4–6 options across Alfama, Baixa, Belém, Cais do Sodré; the cheapest one is the sold-out one.
- Activities: food tours, restaurants, beach day (Cascais / Costa da Caparica), paid and free options with opening hours.
- Tuned so that without the cheap hotel the naive plan is **~CHF 200 over budget**, while a valid plan still exists.

**Done when:** unit tests for every tool pass, including the surprise and the budget math.

---

## Phase 3 — Mock agents + single-agent chat

### Agents (`agents/*.py`)

Each file defines instructions + tool list.

- **Travel Consultant** — the only user-facing agent. Understands and clarifies the request via `ask_user`, resolves specialists' open questions, writes the final plan. No domain tools. Instructions are extended per pattern with a role section (writer, aggregator, moderator, router, manager).
- **Flight, Hotel, Activities, Budget** — narrow instructions: stay in their own domain, return concise structured output, make reasonable assumptions, and list anything unresolved under `Open questions:` instead of asking the user.

`agents/registry.py`: `get_agent("hotel") -> Agent`.

### Pattern interface (`patterns/base.py`)

```python
class Pattern(Protocol):
    name: str
    async def handle_turn(self, user_message: str, ctx: SessionContext) -> str: ...
```

The CLI chat loop is the same for every mode: read input → `pattern.handle_turn()` → render answer → repeat.

### Single-agent mode (`patterns/single.py`)

```bash
uv run app --agent hotel
```

Opens an interactive chat directly with one agent, keeping conversation history across turns. Specialists talk to the user through their normal replies (the chat loop is the channel); only `--agent consultant` has `ask_user`.

In-chat commands:

- `/exit` — quit
- `/reset` — clear history
- `/session` — print the session folder path

**Done when:** you can chat with each of the five agents; they use their tools and `ask_user`; each session is saved and readable in `transcript.md`.

---

## Phase 4 — Sequential

```bash
uv run app --pattern sequential
```

- Fixed order: ✈️ Flight → 🏨 Hotel → 🎭 Activities → 💰 Budget → Travel Consultant (writer).
- Each agent receives the original request **plus all previous agents' outputs**.
- Order is defined in one list, so a different order can be demoed.

**Expected with surprise:** Hotel picks the next option; Budget flags the overrun but can't send the plan back; the Consultant presents an over-budget plan with a warning.

**Done when:** a full run produces a plan, and the log clearly shows the five steps in order.

---

## Phase 5 — Concurrent

```bash
uv run app --pattern concurrent
```

- Flight, Hotel, Activities and Budget run via `asyncio.gather`; none sees the others' output.
- Travel Consultant (aggregator) merges the results.
- **Logging:** interleaved lines stay readable via agent colors and prefixes. The summary shows **wall-clock time vs. sum of agent times** — the key demo number.
- Specialists can't ask the user; their open questions are collected and resolved by the aggregator (which may call `ask_user` once, with all questions batched).
- **Error isolation:** if one agent fails, the others still return and the aggregator is told what's missing.

**Expected:** fastest run; plan is incoherent (e.g. Budget priced a hotel nobody chose, hotel far from activities, late flight vs. day-1 activity).

**Done when:** the run is visibly faster than Sequential, and the incoherence is visible in the transcript.

---

## Phase 6 — Group chat

```bash
uv run app --pattern group_chat --max-rounds 10
```

- **Shared transcript** visible to all participants.
- Travel Consultant is the **moderator**. After each message it returns structured output:

```python
class ModeratorDecision(BaseModel):
    next_speaker: Literal["flight", "hotel", "activities", "budget"] | None
    reason: str
    finished: bool
    final_answer: str | None
```

- Each decision emits a `speaker_selected` event with the reason, so the audience sees *why* each speaker was chosen.
- **Termination:** moderator sets `finished`, or `--max-rounds` is reached (then the moderator is forced to summarize).

**Expected:** Budget objects to the hotel price, Activities offers a free beach day, Hotel finds a cheaper area; the plan ends within budget after several rounds.

**Done when:** that negotiation is readable in the log and the final plan is within budget.

---

## Phase 7 — Handoff

```bash
uv run app --pattern handoff
```

- Each agent gets `handoff_to_<name>(reason)` tools, generated from a **handoff graph**:

```python
HANDOFF_GRAPH = {
    "consultant": ["flight", "hotel", "activities", "budget"],
    "flight":     ["consultant", "hotel"],
    "hotel":      ["consultant", "budget"],
    "activities": ["consultant", "budget"],
    "budget":     ["consultant"],
}
```

- Entry agent: Travel Consultant.
- **The active agent persists across user turns** — if you're talking to Hotel, your next message goes to Hotel.
- The full conversation history travels with each handoff; each emits a `handoff` event (from → to, reason).
- Max handoffs per turn, to prevent ping-pong loops.
- The active specialist talks to the user through its normal replies (its reply ends the turn). It cannot interrupt a task with `ask_user`; if it needs a user decision mid-task (e.g. "go over budget?"), it hands back to the Consultant.
- The prompt shows the active agent: `[🏨 hotel] you>`.

**Expected:** Consultant → Flight → Hotel; Hotel hits the surprise → Budget → back to Consultant, which asks the user: *"Go over budget or change area?"*

**Done when:** a conversation passes through at least three agents and back to the Consultant, and the surprise leads to a question for the user.

---

## Phase 8 — Magentic

```bash
uv run app --pattern magentic --max-steps 15
```

Travel Consultant is the **manager** (following the Magentic-One design).

1. **Task ledger** — created once, updated on re-plan:

```python
class TaskLedger(BaseModel):
    facts: list[str]         # given / verified
    assumptions: list[str]   # to verify
    plan: list[str]          # steps with assigned agent
```

2. **Progress ledger** — after every step:

```python
class ProgressLedger(BaseModel):
    is_request_satisfied: bool
    is_in_loop: bool
    is_progress_being_made: bool
    next_speaker: str
    instruction: str         # what that agent should do now
```

3. **Outer loop:** if progress stalls (stall counter exceeds a threshold) or a surprise invalidates the plan, the manager updates the facts and **re-plans**.
4. **Finish:** the manager writes the final answer; may call `ask_user` to let the user choose between options.

Every ledger version is logged as a `ledger_update` event and rendered in the console as a panel showing what changed — the centerpiece of the demo.

**Expected:** the surprise triggers a visible re-plan (ask Budget for remaining budget, ask Activities to cut costs, re-check flight times vs. day 1); the final plan is within budget and coherent; highest token cost.

**Done when:** the re-plan is visible and the final plan is within budget and consistent.

---

## 4. CLI summary

```bash
uv run app --agent <consultant|flight|hotel|activities|budget>
uv run app --pattern <sequential|concurrent|group_chat|handoff|magentic>
```

Common options:

| Option | Purpose |
|---|---|
| `--verbose` / `-v` | tool args/results, LLM stats |
| `--surprise` / `--no-surprise` | toggle the sold-out hotel (default: on) |
| `--max-rounds N` | group chat round limit |
| `--max-steps N` | Magentic step limit |
| `--sessions-dir PATH` | where sessions are saved (default `sessions/`) |

`--agent` and `--pattern` are mutually exclusive.

In-chat commands: `/exit`, `/reset`, `/session`.

---

## 5. Testing strategy

- **Pyright gate** — every phase is done only when `uv run pyright` passes in strict mode, in addition to its own criteria.
- **`ScriptedLLM` + `ScriptedIO`** — pattern logic tested offline, deterministically, for free.
- **Per-pattern event-order tests:**
  - Sequential emits agents in the configured order.
  - Concurrent starts all four agents before any finishes.
  - Group chat respects `max_rounds` and moderator decisions.
  - Handoff only follows edges in the handoff graph.
  - Magentic re-plans after a `surprise` event.
- **Tool tests** — mock data, surprise, budget math.
- **Ownership test** — only the Consultant has `ask_user`, in every pattern.
- **Real LLM runs** — manual smoke test at the end of each phase.

---

## 6. Later (after all five patterns work)

- `uv run app replay <session>` — re-render a saved session in the console at demo speed (safety net for a live demo).
- Consistency checker (plain code) scoring each plan: arrival time vs. day-1 activities, hotel distance to activities, total vs. budget.
- Scoreboard comparing the five patterns: time, tokens, within budget, coherent, surprise handled, user turns.
- Second, simpler problem where Concurrent wins (independent lookups) — "the shape of the problem picks the pattern".
- Web UI with a five-lane "race" view and maps.
- User Proxy participant — the human as a selectable participant in Group chat and Magentic (see `user-proxy-idea.md`).