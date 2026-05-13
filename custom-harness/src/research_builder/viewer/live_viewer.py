"""Minimal rich-based TUI: tails events.jsonl and renders a scrolling transcript.

Mimics the Claude Code TUI's most useful features:
- Header: paper + current phase + cost
- Scrolling transcript of agent activity, styled by event type
  - agent_started (orchestrator / subagent) → header line
  - agent_thinking → dim italic
  - agent_tool → cyan tool indicator
  - agent_message → role-tinted prose
  - agent_completed → ✓/✗ status line
- Footer: status + key counters (events seen, cost so far)

No tabs, no panes, no widgets. One file, ~200 LoC of rich.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


# Tunable: how many recent transcript lines to keep in the scrolling pane.
# Stays small so the terminal redraws are fast.
TRANSCRIPT_BUFFER = 200


# Style per event kind. Picked to read at-a-glance, matching Claude Code's palette
# loosely (cyan for tools, dim for thinking, bold for new agents).
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


class LiveViewer:
    """Tail an events.jsonl file and render a live transcript with rich."""

    def __init__(self, event_log: Path, workspace: Path | None = None) -> None:
        self.event_log = event_log
        self.workspace = workspace or event_log.parent.parent
        self.console = Console()
        self.transcript: deque[Text] = deque(maxlen=TRANSCRIPT_BUFFER)
        self.event_count = 0
        self.current_phase: str | None = None
        self.run_status: str = "starting"
        self.total_cost_usd: float = 0.0
        self.started_at = datetime.now()

    # ---- rendering ----------------------------------------------------------

    def _header(self) -> Panel:
        now = datetime.now()
        elapsed = (now - self.started_at).total_seconds()
        mins, secs = divmod(int(elapsed), 60)
        t = Table.grid(expand=True)
        t.add_column(justify="left", ratio=1)
        t.add_column(justify="right")
        left = Text()
        left.append("research-builder", style="bold bright_white")
        left.append("  ·  ")
        left.append(str(self.workspace.name), style="cyan")
        if self.current_phase:
            left.append("  ·  ")
            left.append(f"phase: {self.current_phase}", style="yellow")
        right = Text()
        right.append(f"{mins:d}m{secs:02d}s", style="grey50")
        if self.total_cost_usd:
            right.append("  ·  ", style="grey50")
            right.append(f"${self.total_cost_usd:.2f}", style="green")
        t.add_row(left, right)
        return Panel(t, border_style="grey30", padding=(0, 1))

    def _transcript_panel(self) -> Panel:
        if not self.transcript:
            body: Group | Text = Text("(waiting for events…)", style="grey50 italic")
        else:
            body = Group(*self.transcript)
        return Panel(body, border_style="grey30", padding=(0, 1), title="activity", title_align="left")

    def _footer(self) -> Panel:
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
        right = Text(f"{self.event_count} events  ·  tail {self.event_log.name}", style="grey50")
        t.add_row(left, right)
        return Panel(t, border_style="grey30", padding=(0, 1))

    def _render(self) -> Group:
        return Group(self._header(), self._transcript_panel(), self._footer())

    # ---- event processing ---------------------------------------------------

    def _format_event(self, ev: dict[str, Any]) -> Text | None:
        """Map one event → one styled line of transcript output (or None to skip)."""
        kind = ev.get("type", "")
        ts_str = ev.get("ts", "")
        time_part = ts_str.split("T")[-1][:8] if "T" in ts_str else ts_str[:8]
        agent_id = ev.get("agent_id", "")
        agent_label = self._short_agent(agent_id)

        # Header line for new agent
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

        # Thinking — dim italic
        if kind == "agent_thinking":
            text = (ev.get("text") or "").strip()
            if not text:
                return None
            # Truncate long thinking blocks for transcript
            display = text[:160] + ("…" if len(text) > 160 else "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append("✻ ", style="grey50")
            line.append(display, style=STYLES["agent_thinking"])
            return line

        # Tool use — cyan summary
        if kind == "agent_tool":
            summary = (ev.get("summary") or "").strip()
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append("→ ", style="bright_cyan")
            line.append(summary, style=STYLES["agent_tool"])
            return line

        # Agent message (user / assistant / system in chat pane)
        if kind == "agent_message":
            role = ev.get("role", "")
            text = (ev.get("text") or "").strip()
            if not text:
                return None
            display = text[:200] + ("…" if len(text) > 200 else "")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            role_mark = {
                "user": "user>",
                "assistant": "asst>",
                "system": "sys >",
            }.get(role, f"{role}>")
            role_style = {
                "user": "bright_white",
                "assistant": "cyan",
                "system": "grey50",
            }.get(role, "white")
            line.append(f"{role_mark} ", style=role_style)
            line.append(display, style=STYLES["agent_message"])
            return line

        # Agent completed
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

        # File lifecycle
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

        # Compute provisioning (GPU)
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

        # Run completion (terminal)
        if kind in ("run_completed", "run_failed"):
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            mark = "🏁 " if kind == "run_completed" else "💥 "
            line.append(mark, style=STYLES.get(kind, Style()))
            line.append(kind.upper(), style=STYLES.get(kind, Style()))
            return line

        # Spec amendment
        if kind == "spec_amended":
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append("📝 spec amended", style=STYLES["spec_amended"])
            rationale = ev.get("rationale", "")
            if rationale:
                line.append(f": {rationale[:120]}", style=STYLES["spec_amended"])
            return line

        # Unknown — skip silently rather than clutter
        return None

    def _short_agent(self, agent_id: str) -> str:
        """orchestrator / phase:section_5_data → readable label."""
        if not agent_id:
            return ""
        if agent_id == "orchestrator":
            return "orch"
        if agent_id.startswith("phase:"):
            return agent_id[len("phase:") :]
        return agent_id[:16]

    def _process(self, ev: dict[str, Any]) -> None:
        self.event_count += 1
        kind = ev.get("type", "")

        # Track current phase
        if kind == "agent_started":
            agent_id = ev.get("agent_id", "")
            if agent_id.startswith("phase:"):
                self.current_phase = agent_id[len("phase:") :]
        if kind in ("run_completed",):
            self.run_status = "completed"
        elif kind in ("run_failed",):
            self.run_status = "failed"
        elif kind == "agent_started" and ev.get("agent_id") == "orchestrator":
            self.run_status = "running"

        line = self._format_event(ev)
        if line is not None:
            self.transcript.append(line)

    # ---- main loop ----------------------------------------------------------

    async def run(self) -> None:
        """Tail event_log forever, redrawing on every new event."""
        # Open the file (touch if missing)
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        self.event_log.touch(exist_ok=True)

        with Live(self._render(), console=self.console, refresh_per_second=8, screen=False) as live:
            offset = 0
            while True:
                try:
                    with self.event_log.open("r", encoding="utf-8") as f:
                        f.seek(offset)
                        new = f.read()
                        offset = f.tell()
                    if new:
                        for line in new.splitlines():
                            if not line.strip():
                                continue
                            try:
                                ev = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            self._process(ev)
                        live.update(self._render())
                    if self.run_status in ("completed", "failed"):
                        # Stay on screen for the final state, then exit on user signal
                        live.update(self._render())
                        # Wait for ctrl-c; transcript still readable
                        await asyncio.sleep(0.5)
                        continue
                    await asyncio.sleep(0.2)
                except KeyboardInterrupt:
                    break


def run_viewer(workspace: Path, event_log: Path | None = None) -> None:
    """Blocking helper for the CLI entry point."""
    if event_log is None:
        event_log = workspace / "logs" / "events.jsonl"
    viewer = LiveViewer(event_log=event_log, workspace=workspace)
    try:
        asyncio.run(viewer.run())
    except KeyboardInterrupt:
        pass


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Live terminal viewer for a research-builder run."
    )
    p.add_argument("workspace", type=Path, help="Project root (the dir with logs/events.jsonl)")
    p.add_argument("--event-log", type=Path, default=None)
    args = p.parse_args()
    if not args.workspace.is_dir():
        print(f"error: {args.workspace} is not a directory", file=sys.stderr)
        sys.exit(2)
    run_viewer(args.workspace, args.event_log)


if __name__ == "__main__":
    _cli()
