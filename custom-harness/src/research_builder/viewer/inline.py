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
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, RenderableType
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
    "step_started":    Style(color="cyan", dim=True),
    "step_completed":  Style(color="cyan"),
    "skeleton_started":      Style(color="bright_yellow", dim=True),
    "skeleton_completed":    Style(color="bright_yellow", bold=True),
    "section_spec_started":  Style(color="cyan", dim=True),
    "section_spec_completed": Style(color="cyan", bold=True),
    "section_spec_critiqued": Style(color="green"),
    "claims_extracted":      Style(color="bright_blue", bold=True),
    "artifact_created":      Style(color="bright_blue"),
    "gate_reached":          Style(color="bright_yellow", bold=True),
    "gate_resolved":         Style(color="green"),
}

# Hue per retry attempt — bright on first try, hot as retries pile up. Reused
# inside _format_event when an agent_started for a sub-agent reveals what
# retry the phase is on. The phase label takes this hue.
RETRY_HUES = {0: "bright_white", 1: "yellow", 2: "orange3", 3: "red"}

# Single-letter glyphs per agent step role. Used in header + step_started lines.
ROLE_GLYPHS = {
    "refiner": "📝",
    "researcher": "🔬",
    "builder": "🔨",
    "verifier": "✅",
}


@dataclass
class PhaseSummary:
    """Live per-phase rollup that drives the footer/header.

    Updated by ``_on_event`` from dag_updated, agent_started, step_started,
    step_completed, agent_completed.
    """

    phase_id: str
    title: str = ""
    status: str = "pending"
    retry: int = 0
    current_agent: str | None = None  # refiner | researcher | builder | verifier
    duration_s: float = 0.0
    cost_usd: float = 0.0
    started_at: float | None = None  # wall-clock for the current step
    last_step_started_at: float | None = None


_active_viewer: "InlineViewer | None" = None


def get_active_viewer() -> "InlineViewer | None":
    """Return the InlineViewer currently bound to the process, if any."""
    return _active_viewer


class InlineViewer:
    """Subscribe to an EventEmitter and render styled output to the same terminal.

    In `--auto` mode this uses rich.live.Live with a pinned multi-row footer
    (current-phase header line + status/cost line + per-phase progress line).
    In interactive (assistant) mode the footer is skipped so checkpoint
    prompts have clean stdin/stdout.
    """

    def __init__(self, emitter, workspace_label: str = "", interactive: bool = False) -> None:
        self.emitter = emitter
        self.workspace_label = workspace_label
        self.interactive = interactive
        self.console = Console()
        self.started_at = time.time()
        self.event_count = 0
        self.current_phase: str | None = None
        self.run_status = "starting"
        self.total_cost_usd = 0.0
        self._live: Live | None = None
        # Per-phase rollup keyed by phase_id (no "phase:" prefix). Populated
        # from dag_updated; mutated by per-phase events.
        self.phases: dict[str, PhaseSummary] = {}
        # Total/done counters derived from dag_updated.
        self.total_phases: int = 0

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> "InlineViewer":
        global _active_viewer
        if self.emitter is not None:
            self.emitter.subscribe(self._on_event)
        if not self.interactive:
            self._live = Live(
                self._render_footer(),
                console=self.console,
                refresh_per_second=4,
                transient=False,  # keep footer visible on exit
                redirect_stdout=False,  # let the rest of the harness print normally
                redirect_stderr=False,
            )
            self._live.__enter__()
        _active_viewer = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _active_viewer
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
        if _active_viewer is self:
            _active_viewer = None

    # ---- pause/resume around interactive prompts ---------------------------

    def pause(self) -> None:
        """Temporarily stop the live footer so a prompt can write cleanly."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass

    def resume(self) -> None:
        """Re-start the live footer after a paused prompt."""
        if self._live is not None:
            try:
                self._live.start(refresh=True)
            except Exception:
                pass

    # ---- event handling ----------------------------------------------------

    def _on_event(self, ev: dict[str, Any]) -> None:
        self.event_count += 1
        kind = ev.get("type", "")
        self._update_state(ev)

        text = self._format_event(ev)
        if text is None:
            # Some events (step_started/completed for very short steps,
            # dag_updated) drive footer state without a transcript line.
            if self._live is not None:
                self._live.update(self._render_footer())
            return

        if self._live is not None:
            # console.print() while Live is active prints ABOVE the live
            # region (the live region stays pinned at the bottom).
            self.console.print(text)
            self._live.update(self._render_footer())
        else:
            # Interactive mode: no pinned footer; just print events inline.
            self.console.print(text)

    def _update_state(self, ev: dict[str, Any]) -> None:
        """Mutate per-phase state from the event so the footer reflects truth."""
        kind = ev.get("type", "")
        agent_id = ev.get("agent_id", "")
        phase_id = agent_id[len("phase:"):] if agent_id.startswith("phase:") else None

        if kind == "agent_started":
            if phase_id:
                self.current_phase = phase_id
                ps = self._phase(phase_id, title=ev.get("title") or phase_id)
                ps.status = "in_progress"
                ps.started_at = time.time()
                # retry_num on agent_started comes from the orchestrator's
                # subagent dispatch (not always present).
                if "retry_num" in ev:
                    ps.retry = int(ev["retry_num"])
            elif agent_id == "orchestrator":
                self.run_status = "running"

        elif kind == "step_started":
            if phase_id:
                ps = self._phase(phase_id)
                ps.current_agent = ev.get("role")
                if "retry_num" in ev:
                    ps.retry = int(ev["retry_num"])
                ps.last_step_started_at = time.time()

        elif kind == "step_completed":
            if phase_id:
                ps = self._phase(phase_id)
                dur = ev.get("duration_s") or 0.0
                ps.duration_s += float(dur)
                cost = ev.get("cost_usd")
                if cost is not None:
                    ps.cost_usd += float(cost)
                    self.total_cost_usd += float(cost)
                ps.last_step_started_at = None

        elif kind == "agent_completed":
            if phase_id:
                ps = self._phase(phase_id)
                status = ev.get("status", "")
                ps.status = "completed" if status == "completed" else ("failed" if status == "failed" else status or ps.status)
                ps.current_agent = None
                ps.last_step_started_at = None

        elif kind == "dag_updated":
            for node in ev.get("nodes", []):
                pid = node.get("phase_id") or ""
                if not pid:
                    continue
                ps = self._phase(pid, title=node.get("title", pid))
                ps.status = node.get("status", ps.status)
            self.total_phases = len(ev.get("nodes", []))

        elif kind == "run_completed":
            self.run_status = "completed"
        elif kind == "run_failed":
            self.run_status = "failed"

    def _phase(self, phase_id: str, *, title: str = "") -> PhaseSummary:
        ps = self.phases.get(phase_id)
        if ps is None:
            ps = PhaseSummary(phase_id=phase_id, title=title or phase_id)
            self.phases[phase_id] = ps
        elif title and not ps.title:
            ps.title = title
        return ps

    def _format_event(self, ev: dict[str, Any]) -> RenderableType | None:
        kind = ev.get("type", "")
        ts_str = ev.get("ts", "")
        time_part = ts_str.split("T")[-1][:8] if "T" in ts_str else ts_str[:8]
        agent_id = ev.get("agent_id", "")
        agent_label = self._short_agent(agent_id)

        if kind == "gate_reached":
            # Auto-mode gates fire-and-forget — no human action needed,
            # so no banner.
            if ev.get("auto"):
                return None
            prompt_text = (ev.get("prompt") or "").strip() or "(no prompt)"
            gate_id = ev.get("gate_id") or "gate"
            body = Text()
            body.append(prompt_text, style=Style(color="bright_white", bold=True))
            body.append("\n\n")
            body.append("↳ ", style="bright_yellow")
            body.append("Type a reply at the ", style="grey70")
            body.append(">", style=Style(color="#6366f1", bold=True))
            body.append(" prompt below.", style="grey70")
            return Panel(
                body,
                title=f"⏸  Awaiting your reply  ·  {gate_id}",
                title_align="left",
                border_style=STYLES["gate_reached"],
                padding=(0, 1),
            )

        if kind == "gate_resolved":
            if ev.get("auto"):
                return None
            gate_id = ev.get("gate_id") or "gate"
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append("▶ ", style=STYLES["gate_resolved"])
            line.append(f"resolved · {gate_id}", style=STYLES["gate_resolved"])
            return line

        if kind == "agent_started":
            kind_field = ev.get("kind") or ""
            title = ev.get("title") or agent_id
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append("▶ ", style="bright_cyan")
            # Retry-attempt color coding for sub-agent launches.
            label_style = STYLES.get(kind, Style())
            if kind_field == "subagent" and agent_id.startswith("phase:"):
                ps = self.phases.get(agent_id[len("phase:"):])
                if ps is not None:
                    hue = RETRY_HUES.get(ps.retry, "red")
                    label_style = Style(color=hue, bold=True)
            line.append(f"{title}", style=label_style)
            if kind_field:
                line.append(f"  [{kind_field}]", style="grey50")
            if agent_id.startswith("phase:"):
                ps = self.phases.get(agent_id[len("phase:"):])
                if ps is not None and ps.retry:
                    line.append(f"  · retry {ps.retry}", style="grey50")
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
            # Let rich wrap long messages naturally — 200 chars was cutting
            # off the orchestrator's intro and any meaningful narration. The
            # 4000-char backstop matches the emitter cap upstream.
            display = text[:4000] + ("…" if len(text) > 4000 else "")
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

        if kind == "step_started":
            role = ev.get("role", "")
            retry_num = ev.get("retry_num", 0)
            glyph = ROLE_GLYPHS.get(role, "•")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append(f"{glyph} ", style=STYLES["step_started"])
            line.append(f"{role}", style=STYLES["step_started"])
            if retry_num:
                line.append(f"  · retry {retry_num}", style="grey50")
            return line

        if kind == "step_completed":
            role = ev.get("role", "")
            dur = ev.get("duration_s") or 0.0
            cost = ev.get("cost_usd")
            glyph = ROLE_GLYPHS.get(role, "•")
            line = Text()
            line.append(f"{time_part}  ", style="grey50")
            line.append(f"{agent_label}  ", style="grey50")
            line.append(f"{glyph} ", style=STYLES["step_completed"])
            line.append(f"{role} done", style=STYLES["step_completed"])
            line.append(f"  · {dur:.1f}s", style="grey50")
            if cost is not None:
                line.append(f"  · ${float(cost):.3f}", style="green")
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
        """Three-row pinned footer.

        Row 1 — current-phase header (phase title · retry · current agent)
        Row 2 — run status · workspace · elapsed · events · total cost
        Row 3 — progress `[done/total] phase · retry N · agent Ms $X.XX`
        """
        completed = sum(1 for p in self.phases.values() if p.status == "completed")
        total = self.total_phases or len(self.phases) or 1

        elapsed = int(time.time() - self.started_at)
        mins, secs = divmod(elapsed, 60)

        t = Table.grid(expand=True, padding=(0, 0))
        t.add_column(justify="left", ratio=1)
        t.add_column(justify="right")

        # Row 1 — header
        header_left = Text()
        if self.current_phase:
            ps = self.phases.get(self.current_phase)
            phase_hue = RETRY_HUES.get(ps.retry if ps else 0, "red") if ps else "bright_white"
            header_left.append("▸ ", style="bright_cyan")
            header_left.append(f"{self.current_phase}", style=Style(color=phase_hue, bold=True))
            if ps and ps.retry:
                header_left.append(f"  · retry {ps.retry}", style="grey50")
            if ps and ps.current_agent:
                glyph = ROLE_GLYPHS.get(ps.current_agent, "•")
                header_left.append(f"  · {glyph} {ps.current_agent}", style="cyan")
                if ps.last_step_started_at:
                    step_dur = int(time.time() - ps.last_step_started_at)
                    header_left.append(f"  ({step_dur}s)", style="grey50")
        else:
            header_left.append("▸ ", style="grey50")
            header_left.append("awaiting phase", style="grey50")
        header_right = Text(f"[{completed}/{total}]", style="grey50")
        t.add_row(header_left, header_right)

        # Row 2 — status / workspace / elapsed / events / cost
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
        right = Text()
        right.append(f"{mins:d}m{secs:02d}s  ·  {self.event_count} events", style="grey50")
        if self.total_cost_usd:
            right.append(f"  ·  ", style="grey50")
            right.append(f"${self.total_cost_usd:.2f}", style="green")
        t.add_row(left, right)

        # Row 3 — phase progress detail
        progress_left = Text()
        if self.current_phase:
            ps = self.phases.get(self.current_phase)
            if ps:
                progress_left.append(f"phase {self.current_phase}", style="grey50")
                if ps.duration_s:
                    progress_left.append(f"  · ", style="grey50")
                    progress_left.append(f"{ps.duration_s:.0f}s", style="grey50")
                if ps.cost_usd:
                    progress_left.append(f"  · ", style="grey50")
                    progress_left.append(f"${ps.cost_usd:.2f}", style="green")
        else:
            progress_left.append(" ", style="grey50")
        # Mini per-phase status icons (compact roll-up of all phases).
        progress_right = Text()
        for pid, ps in list(self.phases.items())[:12]:
            icon = {"completed": "✓", "in_progress": "●", "failed": "✗"}.get(ps.status, "·")
            color = {"completed": "green", "in_progress": "yellow", "failed": "red"}.get(ps.status, "grey50")
            progress_right.append(icon, style=color)
        t.add_row(progress_left, progress_right)

        return Panel(t, border_style="grey30", padding=(0, 1))


@contextmanager
def inline_viewer_for(workspace_label: str = "", interactive: bool = False):
    """Convenience wrapper — looks up the global emitter and wraps a with-block.

    Used like:
        with inline_viewer_for("my-paper", interactive=True):
            await run_pipeline(config)
    """
    from ..events import get_emitter
    emitter = get_emitter()
    if emitter is None:
        # No emitter wired up — yield a no-op context.
        yield None
        return
    viewer = InlineViewer(emitter, workspace_label=workspace_label, interactive=interactive)
    with viewer:
        yield viewer
