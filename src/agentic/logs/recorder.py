import json
from pathlib import Path
from typing import Any

from agentic.logs.events import Event, EventType
from agentic.logs.stats import SessionStats

RESULT_PREVIEW = 500


def _clip(text: str, limit: int = RESULT_PREVIEW) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class SessionRecorder:
    """Writes each session to `<root>/<session_id>/`, appending after every event."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._stats: dict[str, SessionStats] = {}
        self._emoji: dict[str, str] = {}

    def folder(self, session_id: str) -> Path:
        return self.root / session_id

    def __call__(self, event: Event) -> None:
        folder = self.folder(event.session_id)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        if event.type in (EventType.STEP, EventType.AGENT_START) and event.agent:
            self._emoji[event.agent] = str(event.payload.get("emoji", ""))
        if text := self._transcript(event):
            with (folder / "transcript.md").open("a", encoding="utf-8") as f:
                f.write(text)
        stats = self._stats.setdefault(event.session_id, SessionStats())
        stats.update(event)
        tmp = folder / "summary.json.tmp"
        tmp.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(folder / "summary.json")

    def _label(self, agent: str | None) -> str:
        if agent is None:
            return ""
        emoji = self._emoji.get(agent, "")
        return f"{emoji} {agent}".strip()

    def _transcript(self, event: Event) -> str | None:
        p: dict[str, Any] = event.payload
        who = self._label(event.agent)
        match event.type:
            case EventType.SESSION_START:
                details = "".join(f"- {k}: `{v}`\n" for k, v in p.items())
                return (
                    f"# Session {event.session_id}\n\n- pattern: `{event.pattern}`\n"
                    f"- started: {event.timestamp.isoformat()}\n{details}"
                )
            case EventType.USER_MESSAGE:
                return f"\n## 🧑 User\n\n{p.get('text', '')}\n"
            case EventType.STEP:
                return f"\n## Step {p.get('step')}/{p.get('total')} · {who}\n"
            case EventType.AGENT_START:
                return f"\n### {who}\n\n_Input:_ {p.get('input', '')}\n\n"
            case EventType.LLM_CALL:
                return (
                    f"- {who} · 🧠 LLM · {p.get('prompt_tokens', 0)} + {p.get('completion_tokens', 0)}"
                    f" tokens · {p.get('ms', 0)} ms\n"
                )
            case EventType.TOOL_CALL:
                return f"- {who} · 🔧 `{p.get('tool')}` `{p.get('args', '')}`\n"
            case EventType.TOOL_RESULT:
                mark = "" if p.get("ok", True) else "❌ "
                result = _clip(str(p.get("result", "")))
                return f"  - ↳ {who} · {mark}`{result}` ({p.get('ms', 0)} ms)\n"
            case EventType.ASK_USER:
                return f"\n**❓ {who} asks:** {p.get('question', '')}\n\n"
            case EventType.USER_REPLY:
                return f"**🧑 User:** {p.get('answer', '')}\n\n"
            case EventType.AGENT_END:
                return (
                    f"\n**{who}:**\n\n{p.get('output', '')}\n\n"
                    f"_{p.get('ms', 0)} ms · {p.get('tokens', 0)} tokens_\n"
                )
            case EventType.HANDOFF:
                return (
                    f"\n↪️ **{p.get('from_agent')} → {p.get('to_agent')}**: {p.get('reason', '')}\n"
                )
            case EventType.SPEAKER_SELECTED:
                return f"\n🎤 **Next: {p.get('next_speaker')}**: {p.get('reason', '')}\n"
            case EventType.LEDGER_UPDATE:
                return f"\n📒 **Ledger update**\n\n```json\n{json.dumps(p, indent=2)}\n```\n"
            case EventType.SURPRISE:
                return f"\n> ⚡ **Surprise:** {p.get('description', '')}\n"
            case EventType.ERROR:
                return f"\n> ❌ **Error ({who}):** {p.get('exception', '')}\n"
            case EventType.FINAL_ANSWER:
                return f"\n## ✅ Final answer\n\n{p.get('text', '')}\n"
            case EventType.SESSION_END:
                return (
                    f"\n---\n\n_Session ended · {p.get('duration_ms', 0)} ms ·"
                    f" {p.get('total_tokens', 0)} tokens_\n"
                )
