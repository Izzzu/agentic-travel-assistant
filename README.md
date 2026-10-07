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

Each session is saved under `sessions/`. See the [patterns guide](https://izzzu.github.io/agentic-travel-assistant/patterns) ([source](docs/patterns.md)) for a comparison of the patterns.

## Interacting with the agent

Both `--agent` and `--pattern` modes open an interactive chat: type a message and press enter to send it. The prompt shows who you're talking to, e.g. `[hotel] you>`; in the `handoff` pattern it changes as control moves to a different specialist.

In-chat commands:

- `/exit` (or `/quit`) — end the chat
- `/reset` — clear the conversation history and start over
- `/session` — print the path to the current session folder

## AutoGen version

The Handoff and Magentic patterns are also built with [AutoGen](https://microsoft.github.io/autogen/stable/), in a separate package (`src/agentic_autogen/`) that has its own agent prompts and reuses the mocked tools:

```bash
uv sync --extra autogen
uv run app-autogen --pattern handoff
uv run app-autogen --pattern magentic
```

The chat accepts `/reset` and `/exit`. Magentic also takes `--max-turns` and `--max-stalls`.

It uses the same `.env` as the main app. Optionally set `AZURE_OPENAI_MODEL` to the base model name of your deployment (for example `gpt-5.5-2026-04-24`) to silence AutoGen's "resolved model mismatch" warning; `AZURE_OPENAI_API_VERSION` defaults to `2024-10-21`.

It prints AutoGen's default console output and does not write session files. See [docs/autogen.md](docs/autogen.md) for how the two implementations differ.

## Mock data

The project includes mock data for testing purposes. You can find it under the `mock_data/` directory. This data is used to simulate responses from various agents.