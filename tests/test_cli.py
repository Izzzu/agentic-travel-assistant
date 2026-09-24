from typer.testing import CliRunner

from agentic.cli import app


def test_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--agent" in result.output


def test_agent_is_required() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2


def test_unknown_agent_is_rejected() -> None:
    result = CliRunner().invoke(app, ["--agent", "pilot"])
    assert result.exit_code == 2
