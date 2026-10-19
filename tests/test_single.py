import json
from pathlib import Path

from rich.console import Console

from agentic.agents.registry import get_agent
from agentic.cli import chat
from agentic.core.messages import LLMResponse
from agentic.core.session import SessionContext
from agentic.io.user_io import ScriptedIO
from agentic.llm.scripted import ScriptedLLM, call_tool, reply
from agentic.logs.bus import EventBus
from agentic.logs.console import ConsoleRenderer
from agentic.logs.events import Event, EventType
from agentic.logs.recorder import SessionRecorder
from agentic.patterns.single import SingleAgent

E = EventType


class Chat:
    def __init__(self, tmp_path: Path, agent: str, llm: ScriptedLLM, io: ScriptedIO) -> None:
        self.console = Console(record=True, width=200, force_terminal=False)
        self.recorder = SessionRecorder(tmp_path)
        bus = EventBus()
        bus.subscribe(ConsoleRenderer(self.console))
        bus.subscribe(self.recorder)
        self.io = io
        self.llm = llm
        self.pattern = SingleAgent(get_agent(agent, llm))
        self.ctx = SessionContext(pattern=self.pattern.name, bus=bus, io=io)
        self.folder = self.recorder.folder(self.ctx.session_id)

    async def run(self) -> None:
        await chat(self.pattern, self.ctx, prompt="you>", console=self.console, folder=self.folder)

    def events(self, type: EventType | None = None) -> list[Event]:
        lines = (self.folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
        events = [Event.model_validate_json(line) for line in lines]
        return [e for e in events if type is None or e.type is type]

    def roles(self, request: int) -> list[tuple[str, str | None]]:
        return [(m.role, m.content) for m in self.llm.requests[request][1:]]


async def test_hotel_chat_uses_tools_and_keeps_history(tmp_path: Path) -> None:
    stay = {"hotel_id": "casa-alfama", "checkin": "2026-10-09", "checkout": "2026-10-11"}
    llm = ScriptedLLM(
        [
            call_tool("check_availability", stay),
            reply("Casa Alfama is sold out; Baixa Boutique is free."),
            reply("Yes, it sleeps 3."),
        ]
    )
    io = ScriptedIO(inputs=["Is Casa Alfama free 9-11 Oct?", "/session", "", "Room for 3?"])
    c = Chat(tmp_path, "hotel", llm, io)

    await c.run()

    assert c.roles(2) == [
        ("user", "Is Casa Alfama free 9-11 Oct?"),
        ("assistant", "Casa Alfama is sold out; Baixa Boutique is free."),
        ("user", "Room for 3?"),
    ]
    result = c.events(E.TOOL_RESULT)[0].payload
    assert json.loads(result["result"])["status"] == "sold_out"
    assert [e.payload["text"] for e in c.events(E.FINAL_ANSWER)] == [
        "Casa Alfama is sold out; Baixa Boutique is free.",
        "Yes, it sleeps 3.",
    ]
    assert c.events()[-1].type is E.SESSION_END
    assert str(c.folder.resolve()) in c.console.export_text().replace("\n", "")
    transcript = (c.folder / "transcript.md").read_text(encoding="utf-8")
    assert "## 🧑 User\n\nRoom for 3?" in transcript
    assert "`check_availability`" in transcript
    assert json.loads((c.folder / "summary.json").read_text(encoding="utf-8"))["completed"]


async def test_consultant_asks_the_user_mid_task(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [call_tool("ask_user", {"question": "Where do you fly from?"}), reply("Brief: ZRH-LIS.")]
    )
    io = ScriptedIO(["Zurich"], inputs=["Plan a Lisbon weekend"])
    c = Chat(tmp_path, "consultant", llm, io)

    await c.run()

    assert io.asked == [("consultant", "Where do you fly from?")]
    assert c.events(E.USER_REPLY)[0].payload["answer"] == "Zurich"
    assert c.events(E.FINAL_ANSWER)[0].payload["text"] == "Brief: ZRH-LIS."


async def test_reset_clears_the_conversation(tmp_path: Path) -> None:
    llm = ScriptedLLM([reply("Hi!"), reply("Hello again.")])
    c = Chat(tmp_path, "flight", llm, ScriptedIO(inputs=["hi", "/reset", "again"]))

    await c.run()

    assert c.roles(1) == [("user", "again")]


async def test_failed_turn_keeps_the_chat_going(tmp_path: Path) -> None:
    blocked = LLMResponse(model="scripted", finish_reason="content_filter")
    llm = ScriptedLLM([blocked, reply("Fine now.")])
    io = ScriptedIO(inputs=["bad", "/nope", "good", "/exit", "never read"])
    c = Chat(tmp_path, "budget", llm, io)

    await c.run()

    assert c.roles(1) == [("user", "good")]
    assert [e.payload["text"] for e in c.events(E.FINAL_ANSWER)] == ["Fine now."]
    assert len(c.events(E.ERROR)) == 1
    output = c.console.export_text()
    assert "Turn failed" in output
    assert "Unknown command /nope" in output
    assert io.prompts == ["you>"] * 4
