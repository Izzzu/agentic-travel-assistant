import json
from pathlib import Path
from typing import Any

from agentic.logs.events import LEDGER_SECTIONS, PROGRESS_FLAGS, Event, EventType
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
        if event.type is EventType.SPEAKER_SELECTED and (
            speaker := event.payload.get("next_speaker")
        ):
            self._emoji.setdefault(str(speaker), str(event.payload.get("emoji", "")))
        if event.type is EventType.HANDOFF and (target := event.payload.get("to_agent")):
            self._emoji.setdefault(str(target), str(event.payload.get("emoji", "")))
        if event.type is EventType.LEDGER_UPDATE and event.payload.get("action") == "delegate":
            speaker = str(event.payload.get("next_speaker"))
            self._emoji.setdefault(speaker, str(event.payload.get("emoji", "")))
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

    def _task_ledger(self, p: dict[str, Any]) -> str:
        version = int(p.get("version", 1))
        added: dict[str, list[str]] = p.get("added") or {}
        removed: dict[str, list[str]] = p.get("removed") or {}
        parts = [f"\n### 📒 Task ledger v{version}{' · re-plan' if version > 1 else ''}\n"]
        if p.get("reason"):
            parts.append(f"\n_{p['reason']}_\n")
        for key, title in LEDGER_SECTIONS:
            items: list[str] = p.get(key) or []
            rows = [f"- ➕ {x}" if x in added.get(key, []) else f"- {x}" for x in items]
            rows += [f"- ~~{x}~~" for x in removed.get(key, [])]
            parts.append(f"\n**{title}**\n\n{'\n'.join(rows) or '- _none_'}\n")
        return "".join(parts)

    def _progress_ledger(self, p: dict[str, Any]) -> str:
        flags = " · ".join(f"{label} {'✓' if p.get(key) else '✗'}" for label, key in PROGRESS_FLAGS)
        match p.get("action"):
            case "finish":
                action = "🏁 **Request satisfied**"
            case "replan":
                action = "↻ **Re-plan**"
            case _:
                speaker = self._label(str(p.get("next_speaker")))
                action = f"→ **{speaker}**: {p.get('instruction', '')}"
        return (
            f"\n📊 **Step {p.get('step')}/{p.get('max_steps')} · progress ledger** · {flags}"
            f" · stalls {p.get('stalls', 0)}\n\n> {p.get('reason', '')}\n\n{action}\n"
        )

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
                source = self._label(str(p.get("from_agent")))
                target = self._label(str(p.get("to_agent")))
                return f"\n↪️ **{source} → {target}**: {p.get('reason', '')}\n"
            case EventType.SPEAKER_SELECTED:
                progress = f"Round {p['round']}/{p['max_rounds']} · " if "round" in p else ""
                if p.get("finished"):
                    done = "Forced to finish" if p.get("forced") else "Finished"
                    return f"\n🏁 **{progress}{done}**: {p.get('reason', '')}\n"
                speaker = self._label(str(p.get("next_speaker")))
                return f"\n🎤 **{progress}Next: {speaker}**: {p.get('reason', '')}\n"
            case EventType.LEDGER_UPDATE if p.get("ledger") == "task":
                return self._task_ledger(p)
            case EventType.LEDGER_UPDATE:
                return self._progress_ledger(p)
            case EventType.ERROR:
                return f"\n> ❌ **Error ({who}):** {p.get('exception', '')}\n"
            case EventType.FINAL_ANSWER:
                return f"\n## ✅ Final answer\n\n{p.get('text', '')}\n"
            case EventType.SESSION_END:
                return (
                    f"\n---\n\n_Session ended · {p.get('duration_ms', 0)} ms ·"
                    f" {p.get('total_tokens', 0)} tokens_\n"
                )
