"""Interactive console: key listener active during agent execution."""

from __future__ import annotations

import asyncio
import logging
import os
import select
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

import click

from . import ui
from .config import Config

logger = logging.getLogger(__name__)

# ANSI escape helpers
_UP1 = "\033[1A"     # cursor up 1 line
_CLEAR = "\r\033[K"  # carriage return + clear to end of line

_HINTS = "  [v]iew spec  [e]dit spec  [s]tatus  [a]sk"



class InteractiveConsole:
    """Listens for keypresses while agents work, lets user view/edit spec and chat.

    Manages a display with keyboard hints always pinned at the bottom.
    Activity lines print above the hints.
    """

    def __init__(
        self,
        spec_path: Path,
        spec_manager: object,  # SpecManager — avoid circular import
        config: Config,
    ) -> None:
        self.spec_path = spec_path
        self.spec_manager = spec_manager
        self.config = config
        self._running = False
        self._paused = False
        self._display_active = False
        self._old_settings: list | None = None
        self._is_tty = sys.stdin.isatty()
        self._chat_history: list[dict[str, str]] = []

    def stop(self) -> None:
        self._running = False
        self._clear_display()

    # -- Display management ------------------------------------------------

    def _init_display(self) -> None:
        """Print the hints line."""
        click.echo(click.style(_HINTS, dim=True))
        self._display_active = True

    def print_activity(self, phase_id: str, kind: str, detail: str) -> None:
        """Print a permanent activity line, keeping hints anchored at bottom."""
        if self._paused:
            return

        if kind == "done":
            self._clear_hints()
            ui.activity_done(phase_id, detail)
            return

        tag = click.style(f"[{phase_id}]", fg="magenta")
        if kind == "tool":
            icon = click.style("->", fg="cyan", bold=True)
            text = f"  {tag} {icon} {detail}"
        else:
            text = f"  {tag} {click.style(detail, dim=True)}"

        self._clear_hints()
        click.echo(text)
        click.echo(click.style(_HINTS, dim=True))

    def _clear_hints(self) -> None:
        """Erase the hints line so new output can go above it."""
        if not self._display_active:
            return
        sys.stderr.write(f"{_UP1}{_CLEAR}")
        sys.stderr.flush()

    def _clear_display(self) -> None:
        """Erase the hints line entirely."""
        self._clear_hints()
        self._display_active = False

    # -- Lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        """Listen for keypresses until stopped. No-op if stdin is not a TTY."""
        if not self._is_tty:
            return

        self._running = True
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        self._enter_cbreak()
        self._init_display()

        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(0.2)
                    continue

                char = await asyncio.get_event_loop().run_in_executor(
                    None, self._read_char,
                )
                if char and self._running and not self._paused:
                    await self._handle_key(char)
        except asyncio.CancelledError:
            pass
        finally:
            self._clear_display()
            self._exit_cbreak()

    # -- Terminal mode management ------------------------------------------

    def _enter_cbreak(self) -> None:
        if self._is_tty and self._old_settings is None:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def _exit_cbreak(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    # -- Input -------------------------------------------------------------

    def _read_char(self) -> str | None:
        """Blocking read with 0.5s timeout (runs in thread executor)."""
        if not self._running:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if ready:
                return sys.stdin.read(1)
        except (IOError, OSError, ValueError):
            pass
        return None

    # -- Key handlers ------------------------------------------------------

    async def _handle_key(self, char: str) -> None:
        if char == "v":
            await self._view_spec()
        elif char == "e":
            await self._edit_spec()
        elif char == "s":
            self._show_status()
        elif char == "a":
            await self._chat()

    async def _open_external(self, cmd: list[str]) -> None:
        """Open an external program, restoring terminal around it."""
        self._paused = True
        self._clear_display()
        self._exit_cbreak()
        await asyncio.sleep(0.1)
        try:
            await asyncio.to_thread(subprocess.run, cmd)
        except FileNotFoundError:
            ui.failure(f"Could not launch: {cmd[0]}")
        self._enter_cbreak()
        self._init_display()
        self._paused = False

    async def _view_spec(self) -> None:
        pager = os.environ.get("PAGER", "less")
        await self._open_external([pager, str(self.spec_path)])

    async def _edit_spec(self) -> None:
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
        await self._open_external([editor, str(self.spec_path)])
        ui.success("Spec updated. Changes will apply to the next phase.")

    def _show_status(self) -> None:
        self._clear_display()
        click.echo()
        for phase in self.spec_manager.state.phases:
            is_ok = phase.status.value == "completed"
            ui.phase_status(
                phase.phase_id,
                f"{phase.title} \u2014 {phase.status.value}",
                is_ok,
            )
        click.echo()
        self._init_display()

    # -- Chat --------------------------------------------------------------

    async def _chat(self) -> None:
        """Enter interactive chat mode — ask about the paper or edit the spec."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML

        self._paused = True
        self._clear_display()
        self._exit_cbreak()
        await asyncio.sleep(0.1)

        ui.prompt_bar("Chat")
        ui.info("Ask about the paper or edit the spec. Type /exit to return.")
        ui.info("The sub-agent continues working in the background.\n")

        session = PromptSession()

        try:
            while True:
                try:
                    user_input = await session.prompt_async(
                        HTML("<b><cyan>  chat&gt; </cyan></b>"),
                    )
                except (EOFError, KeyboardInterrupt):
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input in ("/exit", "/done", "/q", "/quit"):
                    break

                self._chat_history.append({"role": "user", "content": user_input})

                # Build conversation context
                conversation = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in self._chat_history
                )

                ui.info("Thinking...")
                try:
                    response = await self._chat_query(conversation)
                except Exception as e:
                    logger.error("Chat query failed: %s", e)
                    response = f"Error: {e}"

                self._chat_history.append({"role": "assistant", "content": response})

                # Display response
                click.echo()
                click.secho(response, fg="white")
                click.echo()

        finally:
            ui.prompt_end()
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            self._enter_cbreak()
            self._init_display()
            self._paused = False

    async def _chat_query(self, conversation: str) -> str:
        """Send a chat message to the Claude agent and return the response."""
        from .chat import chat_query

        def on_tool(name: str) -> None:
            click.echo(click.style(f"    -> {name}", fg="cyan", dim=True))

        return await chat_query(
            conversation=conversation,
            spec_path=self.spec_path,
            model=self.config.model,
            paper_path=self.config.paper_path,
            on_tool=on_tool,
        )
