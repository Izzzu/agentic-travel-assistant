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

Each session is saved under `sessions/`. See [.context/article.md](.context/article.md) for a comparison of the patterns.

## Interacting with the agent

Both `--agent` and `--pattern` modes open an interactive chat: type a message and press enter to send it. The prompt shows who you're talking to, e.g. `[hotel] you>`; in the `handoff` pattern it changes as control moves to a different specialist.

In-chat commands:

- `/exit` (or `/quit`) — end the chat
- `/reset` — clear the conversation history and start over
- `/session` — print the path to the current session folder
