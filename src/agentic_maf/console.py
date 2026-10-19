import sys
from typing import Any

from agent_framework import AgentResponseUpdate, WorkflowEvent
from agent_framework.orchestrations import (
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
    MagenticProgressLedger,
)

RESULT_WIDTH = 160
HANDOFF_PREFIX = "handoff_to_"  # the handoff arrow already shows these calls


def _short(value: object) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= RESULT_WIDTH else f"{text[: RESULT_WIDTH - 1]}…"


class Console:
    """Prints a workflow's events as they stream in; Agent Framework has no default console."""

    def __init__(self) -> None:
        self._speaker: str | None = None
        self._announced = False  # whether the current speaker's header is printed
        self._calls: dict[str, list[str]] = {}  # call id -> [name, argument text so far]
        self._handoff_calls: set[str] = set()

    def show(self, event: WorkflowEvent[Any]) -> None:
        match event.type:
            case "output" | "intermediate" if isinstance(event.data, AgentResponseUpdate):
                self._update(event.executor_id or "?", event.data)
            case "handoff_sent":
                self._flush()
                self._section(f"⇢ handoff: {event.data.source} → {event.data.target}")  # pyright: ignore[reportUnknownMemberType]
            case "magentic_orchestrator" if isinstance(event.data, MagenticOrchestratorEvent):
                self._orchestrator(event.data)
            case _:
                pass

    def finish(self) -> None:
        self._flush()
        self._speaker = None
        self._write("\n")

    def _update(self, speaker: str, update: AgentResponseUpdate) -> None:
        if speaker != self._speaker:
            self._flush()
            self._speaker, self._announced = speaker, False
        for content in update.contents:
            match content.type:
                case "text" if content.text:
                    self._flush()
                    self._say(content.text)
                case "function_call" if content.call_id:
                    if (content.name or "").startswith(HANDOFF_PREFIX):
                        self._handoff_calls.add(content.call_id)
                    if content.call_id in self._handoff_calls:
                        continue
                    call = self._calls.setdefault(content.call_id, ["", ""])
                    call[0] = content.name or call[0]
                    arguments = content.arguments
                    call[1] += arguments if isinstance(arguments, str) else ""
                case "function_result" if content.call_id not in self._handoff_calls:
                    self._flush()
                    self._say(f"  ↳ {_short(content.result)}\n")
                case _:
                    pass

    def _orchestrator(self, event: MagenticOrchestratorEvent) -> None:
        self._flush()
        content = event.content
        match event.event_type:
            case MagenticOrchestratorEventType.PLAN_CREATED:
                self._section(f"=== Task ledger ===\n{getattr(content, 'text', content)}")
            case MagenticOrchestratorEventType.REPLANNED:
                self._section(f"=== Task ledger (re-plan) ===\n{getattr(content, 'text', content)}")
            case MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
                if isinstance(content, MagenticProgressLedger):
                    self._progress(content)

    def _progress(self, ledger: MagenticProgressLedger) -> None:
        flags = (
            f"satisfied={ledger.is_request_satisfied.answer} "
            f"in_loop={ledger.is_in_loop.answer} "
            f"progressing={ledger.is_progress_being_made.answer}"
        )
        text = f"=== Progress ledger: {flags} ==="
        if not ledger.is_request_satisfied.answer:
            nxt, instruction = ledger.next_speaker.answer, ledger.instruction_or_question.answer
            text += f"\nnext: {nxt} · {_short(instruction)}"
        self._section(text)

    def _flush(self) -> None:
        for name, arguments in self._calls.values():
            self._say(f"  ⚙ {name}({_short(arguments)})\n")
        self._calls.clear()

    def _say(self, text: str) -> None:
        """Write for the current speaker, printing their header first."""
        if not self._announced:
            self._write(f"\n--- {self._speaker} ---\n")
            self._announced = True
        self._write(text)

    def _section(self, text: str) -> None:
        self._speaker = None
        self._write(f"\n{text}\n")

    @staticmethod
    def _write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
