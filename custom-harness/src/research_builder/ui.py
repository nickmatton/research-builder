"""Thin UI helpers for styled CLI output. Uses click builtins only."""

from __future__ import annotations

import shutil
import sys

import click

# ANSI 256-color blue gradient (dark → bright → dark)
_BLUE_GRADIENT = [17, 18, 19, 20, 21, 27, 33, 39, 45, 75, 111, 147]

_BANNER_LINES = [
    r"  ____                               _           ____        _ _     _           ",
    r" |  _ \ ___  ___  ___  __ _ _ __ ___| |__       | __ ) _   _(_) | __| | ___ _ __ ",
    r" | |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \ _____|  _ \| | | | | |/ _` |/ _ \ '__|",
    r" |  _ <  __/\__ \  __/ (_| | | | (__| | | |_____| |_) | |_| | | | (_| |  __/ |   ",
    r" |_| \_\___||___/\___|\__,_|_|  \___|_| |_|     |____/ \__,_|_|_|\__,_|\___|_|   ",
]


def banner() -> None:
    """Print the startup banner in a blue gradient."""
    click.echo()
    n = len(_BANNER_LINES)
    for i, line in enumerate(_BANNER_LINES):
        # Pick a color from the gradient, peaking in the middle
        idx = int((i / max(n - 1, 1)) * (len(_BLUE_GRADIENT) - 1))
        color = _BLUE_GRADIENT[idx]
        click.echo(click.style(line, fg=color))
    click.echo()


def header(text: str) -> None:
    """Section header with banner."""
    click.secho(f"\n{'=' * 60}", fg="cyan", bold=True)
    click.secho(f"  {text}", fg="cyan", bold=True)
    click.secho(f"{'=' * 60}", fg="cyan", bold=True)


def step(text: str) -> None:
    """A step within a section."""
    click.secho(f"  -> {text}", fg="white")


def success(text: str) -> None:
    click.secho(f"  [ok] {text}", fg="green")


def failure(text: str) -> None:
    click.secho(f"  [!!] {text}", fg="red")


def warning(text: str) -> None:
    click.secho(f"  [!!] {text}", fg="yellow")


def info(text: str) -> None:
    click.echo(f"  {text}")


def path_line(label: str, path: object) -> None:
    """Print a labeled path."""
    click.echo(f"  {label}: {click.style(str(path), fg='blue', underline=True)}")


def phase_status(phase_id: str, title: str, is_success: bool) -> None:
    """Display a phase result line with colored icon."""
    icon = click.style("[ok]", fg="green") if is_success else click.style("[!!]", fg="red")
    click.echo(f"  {icon} Phase '{phase_id}': {title}")


def divider() -> None:
    click.secho("-" * 60, dim=True)


def prompt_bar(label: str = "") -> None:
    """Draw a visible separator before an input prompt, like Claude Code's input line."""
    width = shutil.get_terminal_size().columns
    clear_status_line()
    click.echo()
    if label:
        # ─── label ──────────────
        pad = width - len(label) - 6
        bar = f"{'─' * 3} {click.style(label, fg='cyan', bold=True)} {'─' * max(pad, 3)}"
    else:
        bar = "─" * width
    click.secho(bar, fg="bright_black")


def prompt_end() -> None:
    """Draw a closing bar after the prompt section."""
    width = shutil.get_terminal_size().columns
    click.secho("─" * width, fg="bright_black")
    click.echo()


def status_line(phase_id: str, kind: str, detail: str) -> None:
    """Overwrite the current terminal line with the latest agent activity."""
    if kind not in ("tool", "thinking"):
        return

    # Truncate detail to fit terminal width — single pass, no recursion.
    # The previous implementation recursed when ``plain_len > width`` but
    # could fail to converge when ``width`` was tiny (subprocess COLUMNS=0
    # or a phase_id longer than the budget), blowing the Python stack.
    width = shutil.get_terminal_size().columns
    if width <= 0:
        # No usable terminal; skip the redraw entirely.
        return

    prefix_plain = f"  [{phase_id}] " + ("-> " if kind == "tool" else "")
    budget = max(0, width - len(prefix_plain) - 3)  # 3 = "..." tail safety
    if budget > 0 and len(detail) > budget:
        detail = detail[: budget].rstrip() + "..."

    tag = click.style(f"[{phase_id}]", fg="magenta")
    if kind == "tool":
        icon = click.style("->", fg="cyan", bold=True)
        text = f"  {tag} {icon} {detail}"
    else:
        text = f"  {tag} {click.style(detail, dim=True)}"

    sys.stderr.write(f"\r\033[K{text}")
    sys.stderr.flush()


def clear_status_line() -> None:
    """Clear the overwriting status line."""
    sys.stderr.write("\r\033[K")
    sys.stderr.flush()


def activity_done(phase_id: str, detail: str) -> None:
    """Print a permanent phase-complete line (clears status line first)."""
    clear_status_line()
    tag = click.style(f"[{phase_id}]", fg="magenta")
    icon = click.style("[ok]", fg="green")
    click.echo(f"  {tag} {icon} Phase complete ({detail})")
