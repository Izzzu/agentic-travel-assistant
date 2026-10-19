# agentic-travel-assistant

Demo project demonstrating multi-agent orchestration patterns

## Get Started

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Configure Azure OpenAI credentials:

   ```bash
   cp .env.example .env
   ```

   Then fill in `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_DEPLOYMENT`.

3. Chat with a single agent:

   ```bash
   uv run app --agent hotel
   ```

4. Or run a full orchestration pattern:

   ```bash
   uv run app --pattern sequential
   ```

   Available patterns: `sequential`, `concurrent`, `group_chat`, `handoff`, `magentic`.

Each session is saved under `sessions/`. See [docs/patterns.md](docs/patterns.md) for a comparison of the patterns.

## Interacting with the agent

Both `--agent` and `--pattern` modes open an interactive chat: type a message and press enter to send it. The prompt shows who you're talking to, e.g. `[hotel] you>`; in the `handoff` pattern it changes as control moves to a different specialist.

In-chat commands:

- `/exit` (or `/quit`) — end the chat
- `/reset` — clear the conversation history and start over
- `/session` — print the path to the current session folder

## Agent Framework version

The Handoff and Magentic patterns are also built with [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/), the successor to AutoGen and Semantic Kernel, in a separate package (`src/agentic_maf/`) that has its own agent prompts and reuses the mocked tools:

```bash
uv sync --extra agent-framework
uv run app-maf --pattern handoff
uv run app-maf --pattern magentic
```

The chat accepts `/reset` and `/exit`. Magentic also takes `--max-rounds` and `--max-stalls`.

It uses the same `.env` as the main app, prints the workflow's events as they stream in and does not write session files. See [docs/agent-framework.md](docs/agent-framework.md) for how the two implementations differ.

## Mock data

The project includes mock data for testing purposes. You can find it under the `mock_data/` directory. This data is used to simulate responses from various agents.