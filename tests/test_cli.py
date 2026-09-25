from typer.testing import CliRunner

from agentic.cli import app


def test_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--agent" in result.output
    assert "--pattern" in result.output
    assert "--max-rounds" in result.output


def test_max_rounds_must_be_positive() -> None:
    result = CliRunner().invoke(app, ["--pattern", "group_chat", "--max-rounds", "0"])
    assert result.exit_code == 2


def test_agent_or_pattern_is_required() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2


def test_agent_and_pattern_are_exclusive() -> None:
    result = CliRunner().invoke(app, ["--agent", "hotel", "--pattern", "sequential"])
    assert result.exit_code == 2


def test_unknown_pattern_is_rejected() -> None:
    result = CliRunner().invoke(app, ["--pattern", "relay"])
    assert result.exit_code == 2


def test_unknown_agent_is_rejected() -> None:
    result = CliRunner().invoke(app, ["--agent", "pilot"])
    assert result.exit_code == 2
