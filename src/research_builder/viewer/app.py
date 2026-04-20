"""Textual app: panes, layout, navigation, and the event-stream pump."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, TabbedContent, TabPane

from .models import ChatMessage
from .sources.command_sink import CommandSink
from .sources.event_stream import tail_jsonl
from .sources.state_loader import bootstrap_from_workspace
from .store import AgentTree
from .widgets import (
    ChatPane,
    DagPane,
    DagSelected,
    FileSelected,
    FilesPane,
    HeaderBar,
    OutputPane,
    ProcessesPane,
)


class AgentTerminalApp(App):
    """Bloomberg-terminal-style viewer for hierarchical agent workflows."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #col-left {
        width: 38;
        layout: vertical;
    }

    #col-center {
        width: 40;
        layout: vertical;
    }

    #col-right {
        width: 2fr;
        min-width: 60;
        layout: vertical;
    }

    FilesPane {
        height: 1fr;
    }

    DagPane {
        height: 1fr;
    }

    ChatPane {
        height: 1fr;
    }

    ProcessesPane {
        height: 1fr;
    }

    OutputPane {
        height: 1fr;
    }

    #right-tabs {
        height: 1fr;
    }

    #chat-input {
        height: 3;
        border: round $primary;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("escape", "go_up", "up"),
        Binding("backspace", "go_up", "up"),
        Binding("p", "switch_tab", "processes"),
        Binding("c", "toggle_copy_mode", "copy mode"),
        Binding("f", "focus_next", "focus next", show=False),
        Binding("F", "focus_previous", "focus prev", show=False),
    ]

    def __init__(
        self,
        workspace: Path,
        event_log: Path,
        command_log: Path,
        pipeline_proc: subprocess.Popen | None = None,
    ) -> None:
        super().__init__()
        self.workspace = Path(workspace)
        self.event_log = Path(event_log)
        self.command_log = Path(command_log)
        self.pipeline_proc = pipeline_proc
        self.agent_tree = AgentTree()
        self.command_sink = CommandSink(self.command_log)
        self._stop_tail = asyncio.Event()
        self._tail_task: asyncio.Task | None = None
        self._output_task: asyncio.Task | None = None
        self._copy_mode = False

    # ----- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        self.header_bar = HeaderBar()
        self.dag_pane = DagPane(id="dag")
        self.chat_pane = ChatPane()
        self.files_pane = FilesPane()
        self.processes_pane = ProcessesPane()
        self.output_pane = OutputPane()

        yield self.header_bar
        with Container(id="body"):
            with Vertical(id="col-left"):
                yield self.files_pane
            with Vertical(id="col-center"):
                yield self.dag_pane
            with Vertical(id="col-right"):
                with TabbedContent(id="right-tabs"):
                    with TabPane("Chat", id="tab-chat"):
                        yield self.chat_pane
                    with TabPane("Output", id="tab-output"):
                        yield self.output_pane
                    with TabPane("Processes", id="tab-proc"):
                        yield self.processes_pane
                yield Input(placeholder="chat → focused agent (Enter to send)", id="chat-input")

    # ----- lifecycle --------------------------------------------------------

    async def on_mount(self) -> None:
        bootstrap_from_workspace(self.workspace, self.agent_tree)
        self.refresh_panes()
        self._tail_task = asyncio.create_task(self._pump_events())

        # Start tailing pipeline.out for the output pane
        pipeline_out = self.workspace / "logs" / "pipeline.out"
        self.output_pane.set_path(str(pipeline_out))
        self._output_task = asyncio.create_task(self._pump_output())

    async def on_unmount(self) -> None:
        self._stop_tail.set()
        if self._tail_task is not None:
            self._tail_task.cancel()
        if self._output_task is not None:
            self._output_task.cancel()

    async def _pump_events(self) -> None:
        # NOTE: this coroutine runs on Textual's main event loop (started via
        # asyncio.create_task in on_mount). We must NOT use call_from_thread
        # here — in current Textual it raises when invoked from the main
        # thread, which would silently kill this task on the first event and
        # leave the UI frozen on the bootstrap snapshot.
        try:
            async for event in tail_jsonl(
                self.event_log,
                from_start=True,
                poll_interval=0.1,
                stop_event=self._stop_tail,
            ):
                try:
                    self.agent_tree.apply_event(event)
                    self.refresh_panes()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "viewer: failed to apply/render event %r", event.get("type")
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            import logging
            logging.getLogger(__name__).exception("viewer: event pump crashed")

    async def _pump_output(self) -> None:
        """Poll pipeline.out and surface new lines in the Output tab.

        Also monitors the pipeline subprocess (if we have a handle) and
        surfaces a crash notice + auto-switches to the Output tab so the
        error is immediately visible.
        """
        try:
            while not self._stop_tail.is_set():
                self.output_pane.poll()
                # Check if the pipeline subprocess has died
                if self.pipeline_proc is not None:
                    rc = self.pipeline_proc.poll()
                    if rc is not None and rc != 0:
                        # One final poll to grab any remaining output
                        self.output_pane.poll()
                        # Surface the crash in the chat pane
                        node = self.agent_tree.current()
                        if node is not None:
                            node.chat.append(
                                ChatMessage(
                                    role="system",
                                    text=f"Pipeline subprocess crashed (exit code {rc}). Check the Output tab for details.",
                                )
                            )
                        # Auto-switch to the Output tab
                        try:
                            tabs = self.query_one("#right-tabs", TabbedContent)
                            tabs.active = "tab-output"
                        except Exception:
                            pass
                        self.refresh_panes()
                        # Mark orchestrator as failed
                        orch = self.agent_tree.nodes.get("orchestrator")
                        if orch is not None and orch.status == "pending":
                            orch.status = "failed"
                            self.refresh_panes()
                        # Keep polling the file but stop checking the proc
                        self.pipeline_proc = None
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        except Exception:
            import logging
            logging.getLogger(__name__).exception("viewer: output pump crashed")

    # ----- rendering --------------------------------------------------------

    def refresh_panes(self) -> None:
        node = self.agent_tree.current()
        self.header_bar.render_for(self.agent_tree)
        self.chat_pane.render_for(node, self.agent_tree)
        self.files_pane.render_for(node, self.agent_tree)
        self.dag_pane.render_for(self.agent_tree)
        self.processes_pane.render_for(node, self.agent_tree)

    # ----- navigation -------------------------------------------------------

    def on_dag_selected(self, message: DagSelected) -> None:
        self.agent_tree.set_current(message.agent_id)
        self.refresh_panes()

    def on_file_selected(self, message: FileSelected) -> None:
        node = self.agent_tree.current()

        def notify(text: str) -> None:
            if node is not None:
                node.chat.append(ChatMessage(role="system", text=text))

        try:
            full = (self.workspace / message.path).resolve()
            workspace_root = self.workspace.resolve()
            if not full.is_relative_to(workspace_root):
                notify(f"(open refused: {message.path} outside workspace)")
                self.refresh_panes()
                return
            if not full.exists():
                notify(f"(open failed: {full} does not exist)")
                self.refresh_panes()
                return

            editor_env = os.environ.get("EDITOR", "").strip()
            # Try $EDITOR first, then known fallbacks; skip any entry not on PATH.
            candidates: list[list[str]] = []
            if editor_env:
                candidates.append(shlex.split(editor_env))
            for fallback in ("vim", "vi", "nano"):
                candidates.append([fallback])

            last_err: str | None = None
            for cmd in candidates:
                if not cmd:
                    continue
                if shutil.which(cmd[0]) is None:
                    last_err = f"{cmd[0]!r} not on PATH"
                    continue
                try:
                    with self.suspend():
                        result = subprocess.run([*cmd, str(full)])
                    if result.returncode != 0:
                        notify(
                            f"(editor {cmd[0]!r} exited with code {result.returncode})"
                        )
                    last_err = None
                    break
                except FileNotFoundError as e:
                    last_err = str(e)
                    continue
            else:
                notify(f"(open failed: no usable editor; last error: {last_err})")
        except Exception as e:
            notify(f"(open failed: {e!r})")
        self.refresh_panes()

    # ----- copy mode --------------------------------------------------------

    def action_toggle_copy_mode(self) -> None:
        """Toggle mouse capture so the user can select & copy text in the terminal.

        Textual captures mouse events for click/scroll, which blocks drag-to-
        select in most terminals. Toggling this off temporarily hands mouse
        control back to the terminal for native text selection.
        """
        driver = getattr(self, "_driver", None)
        if driver is None:
            return
        try:
            if self._copy_mode:
                # Re-enable mouse capture for normal TUI interaction.
                if hasattr(driver, "_enable_mouse_support"):
                    driver._enable_mouse_support()
                self._copy_mode = False
                node = self.agent_tree.current()
                if node is not None:
                    node.chat.append(
                        ChatMessage(role="system", text="(copy mode off — mouse re-enabled)")
                    )
            else:
                if hasattr(driver, "_disable_mouse_support"):
                    driver._disable_mouse_support()
                self._copy_mode = True
                node = self.agent_tree.current()
                if node is not None:
                    node.chat.append(
                        ChatMessage(
                            role="system",
                            text="(copy mode on — drag to select, press 'c' to exit)",
                        )
                    )
        except Exception as e:
            node = self.agent_tree.current()
            if node is not None:
                node.chat.append(
                    ChatMessage(role="system", text=f"(copy mode toggle failed: {e!r})")
                )
        self.refresh_panes()

    def action_go_up(self) -> None:
        self.agent_tree.go_up()
        self.refresh_panes()

    def action_switch_tab(self) -> None:
        tabs = self.query_one("#right-tabs", TabbedContent)
        active = tabs.active
        cycle = {"tab-chat": "tab-output", "tab-output": "tab-proc", "tab-proc": "tab-chat"}
        tabs.active = cycle.get(active, "tab-chat")

    # ----- chat input -------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        text = event.value.strip()
        if not text:
            return
        node = self.agent_tree.current()
        if node is None:
            return
        # Append locally for instant feedback so the user sees their own
        # message immediately, without waiting for the round-trip echo.
        node.chat.append(ChatMessage(role="user", text=text))
        try:
            self.command_sink.send_chat(node.id, text)
        except Exception as e:
            node.chat.append(ChatMessage(role="system", text=f"(send failed: {e})"))
        event.input.value = ""
        self.refresh_panes()
