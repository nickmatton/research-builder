"""Human-in-the-loop interaction helpers for the CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Callable

import click

from . import ui
from .models.results import SubAgentResult


class UserAction(Enum):
    CONTINUE = "continue"
    EDIT_SPEC = "edit_spec"
    SKIP = "skip"
    ABORT = "abort"
    ASK = "ask"


@contextmanager
def _viewer_paused():
    """Pause the inline event viewer's Live footer (if any) while prompting.

    The InlineViewer's rich.live.Live region competes with click.prompt for
    the bottom of the terminal — pausing it for the duration of a prompt
    gives the user a clean input line and prevents flicker.
    """
    try:
        from .viewer.inline import get_active_viewer
        v = get_active_viewer()
    except Exception:
        v = None
    if v is not None:
        v.pause()
    try:
        yield
    finally:
        if v is not None:
            v.resume()


def prompt_after_spec(spec_md_path: Path) -> UserAction:
    """Prompt the user after spec creation."""
    ui.path_line("\nSpec written to", spec_md_path)
    ui.info("Review the plan before I start. You can ask me anything about the paper or the spec.")

    while True:
        with _viewer_paused():
            ui.prompt_bar("Spec Review")
            choice = click.prompt(
                "  [c]ontinue / [e]dit spec / [a]sk question / [q]uit",
                type=str,
                default="c",
                show_default=False,
            ).strip().lower()
            ui.prompt_end()

        if choice in ("c", "continue", ""):
            return UserAction.CONTINUE
        elif choice in ("e", "edit"):
            open_in_editor(spec_md_path)
            ui.success("Spec updated. Reviewing again...")
            continue
        elif choice in ("a", "ask"):
            return UserAction.ASK
        elif choice in ("q", "quit", "abort"):
            return UserAction.ABORT
        else:
            ui.warning(f"Unknown choice: '{choice}'")


def prompt_after_phase(phase_id: str, result: SubAgentResult) -> UserAction:
    """Prompt the user after a phase completes."""
    # Show result summary
    is_success = result.status.value == "success"
    click.echo()
    ui.phase_status(phase_id, result.status.value, is_success)
    ui.info(f"    {result.summary[:500]}")

    if result.test_report.tests_run > 0:
        tr = result.test_report
        ui.info(f"    Tests: {tr.tests_passed}/{tr.tests_run} passed")
        # Show individual failed tests
        for t in tr.test_details:
            if t.status.value != "passed":
                msg = f": {t.message}" if t.message else ""
                click.secho(f"      FAIL {t.test_name}{msg}", fg="red")

    if result.outputs:
        ui.info(f"    Outputs: {', '.join(o.name for o in result.outputs)}")

    if result.is_spec_issue:
        ui.warning("Sub-agent flagged this as a spec issue.")

    click.echo()

    while True:
        with _viewer_paused():
            ui.prompt_bar(f"Phase: {phase_id}")
            choice = click.prompt(
                "  [c]ontinue / [e]dit spec / [s]kip next phase / [a]sk question / [q]uit",
                type=str,
                default="c",
                show_default=False,
            ).strip().lower()
            ui.prompt_end()

        if choice in ("c", "continue", ""):
            return UserAction.CONTINUE
        elif choice in ("e", "edit"):
            return UserAction.EDIT_SPEC
        elif choice in ("s", "skip"):
            return UserAction.SKIP
        elif choice in ("a", "ask"):
            return UserAction.ASK
        elif choice in ("q", "quit", "abort"):
            return UserAction.ABORT
        else:
            ui.warning(f"Unknown choice: '{choice}'")


def prompt_long_running_phase(
    phase_id: str,
    estimated_minutes: int,
    threshold_minutes: int,
) -> UserAction:
    """Prompt the operator before dispatching a phase the refiner thinks will run long.

    Fires only in interactive mode and only when the refiner's estimate
    exceeds ``threshold_minutes``. Returns CONTINUE/SKIP/ABORT — EDIT_SPEC
    and ASK are intentionally omitted; the place to edit the spec is the
    spec-review checkpoint, not here.
    """
    ui.warning(
        f"Phase '{phase_id}' is estimated to take ~{estimated_minutes} min "
        f"(threshold: {threshold_minutes} min)."
    )
    while True:
        with _viewer_paused():
            ui.prompt_bar(f"Long phase: {phase_id}")
            choice = click.prompt(
                "  [c]ontinue / [s]kip phase / [q]uit",
                type=str,
                default="c",
                show_default=False,
            ).strip().lower()
            ui.prompt_end()

        if choice in ("c", "continue", ""):
            return UserAction.CONTINUE
        elif choice in ("s", "skip"):
            return UserAction.SKIP
        elif choice in ("q", "quit", "abort"):
            return UserAction.ABORT
        else:
            ui.warning(f"Unknown choice: '{choice}'")


async def run_chat_loop(spec_path: Path, paper_path: Path, model: str) -> None:
    """Mini Q&A loop: user asks questions, the assistant answers using the paper + spec.

    Returns when the user enters an empty line (or types 'done'/'back').
    Edits made to the spec inside chat are persisted to disk before returning.
    """
    from .chat import chat_query

    ui.info("Ask me anything about the paper or the spec. Press Enter on an empty line to return.")
    while True:
        with _viewer_paused():
            ui.prompt_bar("Ask")
            try:
                question = click.prompt(
                    "  > ",
                    type=str,
                    default="",
                    show_default=False,
                ).strip()
            except (EOFError, click.Abort):
                ui.prompt_end()
                return
            ui.prompt_end()

        if not question or question.lower() in ("done", "back", "exit", "quit"):
            return

        with _viewer_paused():
            try:
                answer = await chat_query(
                    conversation=question,
                    spec_path=spec_path,
                    paper_path=paper_path,
                    model=model,
                )
            except Exception as e:
                ui.failure(f"Chat failed: {e}")
                continue
            click.echo()
            click.secho("  assistant:", fg="cyan", bold=True)
            for line in answer.splitlines() or [answer]:
                click.echo(f"    {line}")
            click.echo()


def prompt_skip_which_phase(runnable: list[str]) -> str | None:
    """Ask which phase to skip when user selects 'skip'."""
    if not runnable:
        ui.warning("No runnable phases to skip.")
        return None

    if len(runnable) == 1:
        phase_id = runnable[0]
        confirm = click.confirm(f"  Skip phase '{phase_id}'?", default=True)
        return phase_id if confirm else None

    ui.prompt_bar("Skip Phase")
    ui.info("Which phase to skip?")
    for i, pid in enumerate(runnable, 1):
        ui.info(f"  {i}. {pid}")

    choice = click.prompt("  Enter number", type=int, default=1)
    ui.prompt_end()
    if 1 <= choice <= len(runnable):
        return runnable[choice - 1]
    return None


def open_in_editor(file_path: Path) -> None:
    """Open a file in the user's preferred editor."""
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
    try:
        subprocess.run([editor, str(file_path)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        ui.failure(f"Could not open editor '{editor}': {e}")
        ui.info(f"Edit manually: {file_path}")
