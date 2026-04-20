"""CLI entry point for agent-terminal."""

from __future__ import annotations

from pathlib import Path

import click

from .app import AgentTerminalApp


@click.command()
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--event-log",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to events.jsonl. Defaults to <workspace>/logs/events.jsonl.",
)
@click.option(
    "--command-log",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to commands.jsonl (outbound chat). Defaults to <workspace>/logs/commands.jsonl.",
)
def main(workspace: Path, event_log: Path | None, command_log: Path | None) -> None:
    """Launch the TUI viewer against a research-builder WORKSPACE directory."""
    workspace = workspace.resolve()
    if event_log is None:
        event_log = workspace / "logs" / "events.jsonl"
    if command_log is None:
        command_log = workspace / "logs" / "commands.jsonl"
    AgentTerminalApp(
        workspace=workspace,
        event_log=event_log,
        command_log=command_log,
    ).run()


if __name__ == "__main__":
    main()
