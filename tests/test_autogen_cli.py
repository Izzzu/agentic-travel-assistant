import pytest

pytest.importorskip("autogen_agentchat")

from collections.abc import Callable

from autogen_agentchat.base import TaskResult
from typer.testing import CliRunner

from agentic_autogen.cli import app, chat


class FakePattern:
    def __init__(self, fail_on: str = "") -> None:
        self.turns: list[str] = []
        self.resets = 0
        self.fail_on = fail_on

    @property
    def prompt(self) -> str:
        return "[fake] you>"

    async def turn(self, text: str) -> TaskResult:
        if text == self.fail_on:
            raise RuntimeError("boom")
        self.turns.append(text)
        return TaskResult(messages=[])

    async def reset(self) -> None:
        self.resets += 1


def reader(*lines: str) -> Callable[[str], str]:
    remaining = iter(lines)

    def read(prompt: str) -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read


async def test_messages_become_turns_until_the_input_ends() -> None:
    pattern = FakePattern()

    await chat(pattern, reader("plan a trip", "", "  and a hotel  "))

    assert pattern.turns == ["plan a trip", "and a hotel"]


async def test_reset_clears_and_exit_stops() -> None:
    pattern = FakePattern()

    await chat(pattern, reader("hello", "/reset", "/exit", "never read"))

    assert pattern.turns == ["hello"]
    assert pattern.resets == 1


async def test_unknown_commands_are_not_sent_to_the_team(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pattern = FakePattern()

    await chat(pattern, reader("/help"))

    assert pattern.turns == []
    assert "Unknown command /help" in capsys.readouterr().out


async def test_a_failed_turn_does_not_end_the_chat(capsys: pytest.CaptureFixture[str]) -> None:
    pattern = FakePattern(fail_on="bad")

    await chat(pattern, reader("bad", "good"))

    assert pattern.turns == ["good"]
    assert "Turn failed (RuntimeError)" in capsys.readouterr().out


def test_help_lists_the_patterns() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--pattern" in result.output
    assert "--max-turns" in result.output


def test_a_pattern_is_required_and_must_exist() -> None:
    assert CliRunner().invoke(app, []).exit_code == 2
    assert CliRunner().invoke(app, ["--pattern", "sequential"]).exit_code == 2


def test_missing_azure_settings_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "")

    result = CliRunner().invoke(app, ["--pattern", "handoff"])

    assert result.exit_code == 1
    assert "AZURE_OPENAI_ENDPOINT" in result.output
