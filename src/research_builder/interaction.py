"""Human-in-the-loop interaction helpers for the CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Callable

import click

from .models.results import SubAgentResult


class UserAction(Enum):
    CONTINUE = "continue"
    EDIT_SPEC = "edit_spec"
    SKIP = "skip"
    ABORT = "abort"


def prompt_after_spec(spec_md_path: Path) -> UserAction:
    """Prompt the user after spec creation."""
    click.echo(f"\nSpec written to: {spec_md_path}")
    click.echo("Review the spec before proceeding.\n")

    while True:
        choice = click.prompt(
            "  [c]ontinue / [e]dit spec / [a]bort",
            type=str,
            default="c",
            show_default=False,
        ).strip().lower()

        if choice in ("c", "continue"):
            return UserAction.CONTINUE
        elif choice in ("e", "edit"):
            open_in_editor(spec_md_path)
            click.echo("Spec updated. Reviewing again...\n")
            # After editing, prompt again so user can re-review or continue
            continue
        elif choice in ("a", "abort"):
            return UserAction.ABORT
        else:
            click.echo(f"  Unknown choice: '{choice}'")


def prompt_after_phase(phase_id: str, result: SubAgentResult) -> UserAction:
    """Prompt the user after a phase completes."""
    # Show result summary
    status_icon = "+" if result.status.value == "success" else "x"
    click.echo(f"\n  [{status_icon}] Phase '{phase_id}': {result.status.value}")
    click.echo(f"      {result.summary[:200]}")

    if result.test_report.tests_run > 0:
        tr = result.test_report
        click.echo(f"      Tests: {tr.tests_passed}/{tr.tests_run} passed")

    if result.outputs:
        click.echo(f"      Outputs: {', '.join(o.name for o in result.outputs)}")

    if result.is_spec_issue:
        click.echo("      Note: Sub-agent flagged this as a spec issue.")

    click.echo()

    while True:
        choice = click.prompt(
            "  [c]ontinue / [e]dit spec / [s]kip next phase / [a]bort",
            type=str,
            default="c",
            show_default=False,
        ).strip().lower()

        if choice in ("c", "continue"):
            return UserAction.CONTINUE
        elif choice in ("e", "edit"):
            return UserAction.EDIT_SPEC
        elif choice in ("s", "skip"):
            return UserAction.SKIP
        elif choice in ("a", "abort"):
            return UserAction.ABORT
        else:
            click.echo(f"  Unknown choice: '{choice}'")


def prompt_skip_which_phase(runnable: list[str]) -> str | None:
    """Ask which phase to skip when user selects 'skip'."""
    if not runnable:
        click.echo("  No runnable phases to skip.")
        return None

    if len(runnable) == 1:
        phase_id = runnable[0]
        confirm = click.confirm(f"  Skip phase '{phase_id}'?", default=True)
        return phase_id if confirm else None

    click.echo("  Which phase to skip?")
    for i, pid in enumerate(runnable, 1):
        click.echo(f"    {i}. {pid}")

    choice = click.prompt("  Enter number", type=int, default=1)
    if 1 <= choice <= len(runnable):
        return runnable[choice - 1]
    return None


def open_in_editor(file_path: Path) -> None:
    """Open a file in the user's preferred editor."""
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
    try:
        subprocess.run([editor, str(file_path)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        click.echo(f"  Could not open editor '{editor}': {e}", err=True)
        click.echo(f"  Edit manually: {file_path}", err=True)
