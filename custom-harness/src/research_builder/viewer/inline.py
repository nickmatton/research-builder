"""Inline viewer: pretty-prints harness events into the SAME terminal.

Subscribes to the in-process event emitter (no subprocess, no separate
terminal). Each event becomes a styled rich line printed in order. A small
status footer at the bottom updates live with current phase, elapsed time,
event count, and cumulative cost.

Usage:

    from research_builder.events import get_emitter
    from research_builder.viewer.inline import InlineViewer

    emitter = get_emitter()
    with InlineViewer(emitter, workspace_label="my-paper") as viewer:
        await run_pipeline(config)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


# Same style palette as live_viewer.py for consistency.
STYLES = {
    "agent_started":   Style(color="bright_white", bold=True),
    "agent_thinking":  Style(color="grey50", italic=True),
    "agent_tool":      Style(color="cyan"),
    "agent_message":   Style(color="white"),
    "agent_completed": Style(color="green", bold=True),
    "agent_completed_failed": Style(color="red", bold=True),
    "file_created":    Style(color="bright_blue"),
    "file_planned":    Style(color="blue", dim=True),
    "compute_provisioned": Style(color="magenta", bold=True),
    "compute_terminated":  Style(color="magenta", dim=True),
    "run_completed":   Style(color="green", bold=True),
    "run_failed":      Style(color="red", bold=True),
    "spec_amended":    Style(color="yellow"),
}


class InlineViewer:
    """Subscribe to an EventEmitter and render styled output to the same terminal.

    Uses rich.live.Live with a small footer region — events scroll above,
    status pins to the bottom.
    """

    def __init__(self, emitter, workspace_label: str = "") -> None:
        self.emitter = emitter
        self.workspace_label = workspace_label
        self.console = Console()
        self.started_at = time.time()
        self.event_count = 0
        self.current_phase: str | None = None
        self.run_status = "starting"
        self.total_cost_usd = 0.0
        self._live: Live | None = None

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> "InlineViewer":
        if self.emitter is not None:
            self.emitter.subscribe(self._on_event)
        self._live = Live(
            self._render_footer(),
            console=self.console,
            refresh_per_second=4,
            transient=False,  # keep footer visible on exit
            redirect_stdout=False,  # let the rest of the harness print normally
            redirect_stderr=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            try:
                # Final footer update so the user sees the terminal state.
                self._live.update(self._render_footer())
            except Exception:
                pass
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        if self.emitter is not None:
            self.emitter.unsubscribe(self._on_event)

    # ---- event handling ----------------------------------------------------

    def _on_event(self, ev: dict[str, Any]) -> None:
        self.event_count += 1
        kind = ev.get("type", "")
        if kind == "agent_started":
            agent_id = ev.get("agent_id", "")
            if agent_id.startswith("phase:"):
                self.current_phase = agent_id[len("phase:") :]
            elif agent_id == "orchestrator":
                self.run_status = "running"
        elif kind == "run_completed":
            self.run_status = "completed"
        elif kind == "run_failed":
            self.run_status = "failed"

        text = self._format_event(ev)
        if text is not None and self._live is not None:
            # console.print() while Live is active prints ABOVE the live
            # region (the live region stays pinned at the bottom).
            self.console.print(text)
            # Footer needs a refresh to pick up status/cost/event count.
            self._live.update(self._render_footer())

    def _format_event(self, ev: dict[str, Any]) -> Text | None:
        kind = ev.get("type", "")
        ts_str = ev.get("ts", "")
        time_part = ts_str.split("T")[-1][:8] if "T" in ts_str else ts_str[:8]
        agent_id = ev.get("agent_id", "")
        agent_label = self._short_agent(agent_id)

        if kind == "agent_started":
            kind_field = ev.get("kind") or ""
            title = ev.get("title") or agent_id
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append("▶ ", style="bright_cyan")
            line.append(f"{title}", style=STYLES.get(kind, Style()))
            if kind_field:
                line.append(f"  [{kind_field}]", style="grey50")
            return line

        if kind == "agent_thinking":
            text = (ev.get("text") or "").strip()
            if not text:
                return None
            display = text[:160] + ("…" if len(text) > 160 else "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append("✻ ", style="grey50")
            line.append(display, style=STYLES["agent_thinking"])
            return line

        if kind == "agent_tool":
            summary = (ev.get("summary") or "").strip()
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append("→ ", style="bright_cyan")
            line.append(summary, style=STYLES["agent_tool"])
            return line

        if kind == "agent_message":
            role = ev.get("role", "")
            text = (ev.get("text") or "").strip()
            if not text:
                return None
            display = text[:200] + ("…" if len(text) > 200 else "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            role_mark = {"user": "user>", "assistant": "asst>", "system": "sys >"}.get(role, f"{role}>")
            role_style = {"user": "bright_white", "assistant": "cyan", "system": "grey50"}.get(role, "white")
            line.append(f"{role_mark} ", style=role_style)
            line.append(display, style=STYLES["agent_message"])
            return line

        if kind == "agent_completed":
            status = ev.get("status", "")
            summary = ev.get("summary", "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            mark = "✓" if status == "completed" else "✗"
            mark_style = STYLES["agent_completed"] if status == "completed" else STYLES["agent_completed_failed"]
            line.append(f"{mark} ", style=mark_style)
            line.append(f"{agent_label}  ", style="grey50")
            line.append(f"({status}) {summary}", style=mark_style)
            return line

        if kind in ("file_created", "file_planned"):
            path = ev.get("path", "")
            file_status = ev.get("status", "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            mark = "📄" if kind == "file_created" else "📋"
            line.append(f"{mark} ", style=STYLES.get(kind, Style()))
            line.append(path, style=STYLES.get(kind, Style()))
            if file_status:
                line.append(f"  [{file_status}]", style="grey50")
            return line

        if kind in ("compute_provisioned", "compute_terminated"):
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            mark = "🖥 " if kind == "compute_provisioned" else "🛑 "
            line.append(mark, style=STYLES.get(kind, Style()))
            inst_type = ev.get("instance_type", "")
            ip = ev.get("public_ip", "")
            inst_id = ev.get("instance_id", "")
            if kind == "compute_provisioned":
                line.append(f"GPU provisioned: {inst_type} ({ip})", style=STYLES.get(kind, Style()))
            else:
                line.append(f"GPU terminated: {inst_id}", style=STYLES.get(kind, Style()))
            return line

        if kind in ("run_completed", "run_failed"):
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            mark = "🏁 " if kind == "run_completed" else "💥 "
            line.append(mark, style=STYLES.get(kind, Style()))
            line.append(kind.upper(), style=STYLES.get(kind, Style()))
            return line

        if kind == "spec_amended":
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append("📝 spec amended", style=STYLES["spec_amended"])
            rationale = ev.get("rationale", "")
            if rationale:
                line.append(f": {rationale[:120]}", style=STYLES["spec_amended"])
            return line

        return None

    def _short_agent(self, agent_id: str) -> str:
        if not agent_id:
            return ""
        if agent_id == "orchestrator":
            return "orch"
        if agent_id.startswith("phase:"):
            return agent_id[len("phase:") :]
        return agent_id[:16]

    # ---- footer rendering --------------------------------------------------

    def _render_footer(self) -> Panel:
        elapsed = int(time.time() - self.started_at)
        mins, secs = divmod(elapsed, 60)
        t = Table.grid(expand=True)
        t.add_column(justify="left", ratio=1)
        t.add_column(justify="right")

        status_color = {
            "starting": "grey50",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }.get(self.run_status, "grey50")
        left = Text()
        left.append("● ", style=status_color)
        left.append(self.run_status, style=status_color)
        if self.workspace_label:
            left.append(f"  ·  {self.workspace_label}", style="cyan")
        if self.current_phase:
            left.append(f"  ·  ", style="grey50")
            left.append(f"phase: {self.current_phase}", style="yellow")

        right = Text()
        right.append(f"{mins:d}m{secs:02d}s  ·  {self.event_count} events", style="grey50")
        if self.total_cost_usd:
            right.append(f"  ·  ", style="grey50")
            right.append(f"${self.total_cost_usd:.2f}", style="green")
        t.add_row(left, right)
        return Panel(t, border_style="grey30", padding=(0, 1))


@contextmanager
def inline_viewer_for(workspace_label: str = ""):
    """Convenience wrapper — looks up the global emitter and wraps a with-block.

    Used like:
        with inline_viewer_for("my-paper"):
            await run_pipeline(config)
    """
    from ..events import get_emitter
    emitter = get_emitter()
    if emitter is None:
        # No emitter wired up — yield a no-op context.
        yield None
        return
    viewer = InlineViewer(emitter, workspace_label=workspace_label)
    with viewer:
        yield viewer
