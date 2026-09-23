import typer

app = typer.Typer(help="Agentic travel assistant: one trip, five orchestration patterns.")


@app.command()
def main() -> None:
    """Start the travel assistant."""
    typer.echo("No agents or patterns are available yet.")
