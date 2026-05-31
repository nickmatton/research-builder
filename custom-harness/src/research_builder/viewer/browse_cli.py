"""``rb-browse`` console-script entry point.

Launches the read-only browse-mode TUI against a workspace produced by the
research-builder harness. Live mode (with intervention commands enabled) is
launched in-process from the InlineViewer's ``t`` hotkey, not via this CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Lazy-import the heavy TUI module so the CLI starts instantly when the
    # user just wants `--help`.
    import click

    @click.command(help="Browse a research-builder workspace (post-hoc, read-only).")
    @click.argument("workspace", type=click.Path(exists=True, path_type=Path))
    @click.option("--live", is_flag=True, help="Treat as a live workspace (enable interventions).")
    def cmd(workspace: Path, live: bool) -> None:
        from .browse import run_browse
        sys.exit(run_browse(workspace, mode="live" if live else "post_hoc"))

    cmd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
