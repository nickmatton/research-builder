"""CLI entry: ``rb-viewer <workspace>`` or ``python -m research_builder.viewer <workspace>``."""

from __future__ import annotations

from pathlib import Path

import click

from .live_viewer import run_viewer


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
def main(workspace: Path, event_log: Path | None) -> None:
    """Live terminal viewer for a research-builder workspace.

    Tails the events.jsonl file and renders agent activity as a scrolling
    Claude-Code-style transcript. Run alongside `bin/research-builder` —
    open the viewer in one terminal, the harness in another, both pointed at
    the same workspace.
    """
    workspace = workspace.resolve()
    run_viewer(workspace, event_log)


if __name__ == "__main__":
    main()
