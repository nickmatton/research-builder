"""Browse-mode TUI: navigate phases, attempts, and artifacts of a run.

Works in two modes:

- **post_hoc**: launched via ``rb-browse <workspace>``. Read-only. Loads
  ``canonical_spec/state.json`` and per-phase ``attempts/*/manifest.json``;
  tails ``logs/events.jsonl`` for visual freshness but issues no commands.
- **live** (Stage 3 future hook): launched from InlineViewer's ``t`` hotkey
  while a run is active. Same UI plus four intervention keybindings that
  append commands to ``logs/commands.jsonl`` (drained by the running pipeline).

Built with prompt_toolkit (NOT Textual) to stay lean — the Textual app was
removed in commit 9a99f54 for exactly this reason. Keyboard nav only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.styles import Style

logger = logging.getLogger(__name__)


STATUS_ICON = {
    "completed": "✓",
    "in_progress": "●",
    "failed": "✗",
    "pending": "·",
}
STATUS_CLASS = {
    "completed": "class:status-ok",
    "in_progress": "class:status-run",
    "failed": "class:status-fail",
    "pending": "class:status-pending",
}
ROLE_GLYPH = {
    "refiner": "📝",
    "researcher": "🔬",
    "builder": "🔨",
    "verifier": "✅",
}


@dataclass
class PhaseRow:
    phase_id: str
    title: str
    status: str
    attempts: int = 0


@dataclass
class AttemptStep:
    role: str
    status: str
    duration_s: float = 0.0
    cost_usd: float | None = None
    record_path: str = ""


@dataclass
class BrowseState:
    workspace: Path
    mode: str  # live | post_hoc
    phases: list[PhaseRow] = field(default_factory=list)
    selected: int = 0
    detail: dict[int, list[list[AttemptStep]]] = field(default_factory=dict)
    last_message: str = ""

    @property
    def commands_path(self) -> Path:
        return self.workspace / "logs" / "commands.jsonl"

    @property
    def events_path(self) -> Path:
        return self.workspace / "logs" / "events.jsonl"

    @property
    def state_json(self) -> Path:
        return self.workspace / "canonical_spec" / "state.json"

    def current_phase(self) -> PhaseRow | None:
        if not self.phases or not (0 <= self.selected < len(self.phases)):
            return None
        return self.phases[self.selected]


# ─── Disk loaders ────────────────────────────────────────────────────────


def _load_phases(state_json: Path) -> list[PhaseRow]:
    if not state_json.exists():
        return []
    try:
        data = json.loads(state_json.read_text() or "{}") or {}
    except Exception:
        logger.exception("browse: failed to read state.json at %s", state_json)
        return []
    out: list[PhaseRow] = []
    for p in data.get("phases", []) or []:
        out.append(PhaseRow(
            phase_id=p.get("phase_id", ""),
            title=p.get("title") or p.get("phase_id", ""),
            status=p.get("status", "pending"),
        ))
    return out


def _load_attempts(workspace: Path, phase_id: str) -> list[list[AttemptStep]]:
    """Return one list of AttemptStep per attempt directory, ordered by retry_num."""
    base = workspace / "phases" / phase_id / "attempts"
    if not base.exists():
        return []
    result: list[list[AttemptStep]] = []
    try:
        retries = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else 999)
    except Exception:
        retries = []
    for d in retries:
        steps: list[AttemptStep] = []
        manifest = d / "manifest.json"
        if manifest.exists():
            try:
                entries = json.loads(manifest.read_text())
            except Exception:
                entries = []
            for e in entries or []:
                steps.append(AttemptStep(
                    role=e.get("role", ""),
                    status=e.get("status", "ok"),
                    duration_s=float(e.get("duration_s") or 0.0),
                    cost_usd=e.get("cost_usd"),
                    record_path=e.get("record_path", ""),
                ))
        result.append(steps)
    return result


# ─── Renderers ───────────────────────────────────────────────────────────


def _render_phase_list(state: BrowseState) -> FormattedText:
    lines: list[tuple[str, str]] = [("class:pane-title", " Phases (j/k)\n")]
    if not state.phases:
        lines.append(("class:dim", " (no phases)\n"))
        return FormattedText(lines)
    for i, p in enumerate(state.phases):
        icon = STATUS_ICON.get(p.status, "·")
        style = STATUS_CLASS.get(p.status, "")
        sel_prefix = "▸ " if i == state.selected else "  "
        label = f"{sel_prefix}{icon} {p.phase_id}"
        if i == state.selected:
            lines.append(("class:selected", label + "\n"))
        else:
            lines.append((style, label + "\n"))
    return FormattedText(lines)


def _render_detail(state: BrowseState) -> FormattedText:
    cur = state.current_phase()
    if cur is None:
        return FormattedText([("class:dim", " (select a phase)\n")])
    lines: list[tuple[str, str]] = []
    lines.append(("class:pane-title", f" Phase: {cur.phase_id}\n"))
    lines.append(("", f" Title:  {cur.title}\n"))
    lines.append((STATUS_CLASS.get(cur.status, ""), f" Status: {cur.status}\n"))
    lines.append(("", "\n"))

    attempts = state.detail.get(state.selected) or _load_attempts(state.workspace, cur.phase_id)
    state.detail[state.selected] = attempts

    if not attempts:
        lines.append(("class:dim", " (no attempts recorded yet)\n"))
    else:
        lines.append(("class:pane-title", " Attempts:\n"))
        for i, steps in enumerate(attempts):
            lines.append(("class:retry-header", f"   retry {i}:\n"))
            if not steps:
                lines.append(("class:dim", "     (no steps)\n"))
                continue
            for s in steps:
                glyph = ROLE_GLYPH.get(s.role, "•")
                status_style = "class:status-ok" if s.status == "ok" else "class:status-fail"
                cost_str = f" ${s.cost_usd:.3f}" if s.cost_usd else ""
                lines.append((status_style, f"     {glyph} {s.role:<10} {s.duration_s:>6.1f}s{cost_str}\n"))

    lines.append(("", "\n"))
    lines.append(("class:dim", " enter: open _result.json · e: edit refined_spec · r: force retry · n: inject note · b: jump back\n"))
    return FormattedText(lines)


def _render_status_bar(state: BrowseState) -> FormattedText:
    msg = state.last_message or ""
    mode_label = "LIVE" if state.mode == "live" else "POST-HOC"
    return FormattedText([
        ("class:title", f" research-builder browse · {mode_label} · {state.workspace.name}  "),
        ("class:dim", f"  {msg}"),
    ])


def _render_keybar() -> FormattedText:
    return FormattedText([
        ("class:dim", " j/k nav · enter open · e edit · r retry · n note · b back · / search · q quit"),
    ])


# ─── Application ─────────────────────────────────────────────────────────


def build_app(state: BrowseState) -> Application:
    """Construct the prompt_toolkit Application."""

    phase_list = Window(
        content=FormattedTextControl(lambda: _render_phase_list(state), focusable=True),
        width=D(preferred=32, max=40),
        style="class:pane",
    )
    detail = Window(
        content=FormattedTextControl(lambda: _render_detail(state)),
        style="class:pane",
    )
    title = Window(content=FormattedTextControl(lambda: _render_status_bar(state)), height=1, style="class:bar")
    keybar = Window(content=FormattedTextControl(_render_keybar), height=1, style="class:bar")

    root = HSplit([
        title,
        VSplit([phase_list, Window(width=1, char="│", style="class:dim"), detail]),
        keybar,
    ])

    kb = KeyBindings()

    @kb.add("q")
    @kb.add("c-c")
    def _(event):
        event.app.exit()

    @kb.add("j")
    @kb.add("down")
    def _(event):
        if state.phases:
            state.selected = min(state.selected + 1, len(state.phases) - 1)

    @kb.add("k")
    @kb.add("up")
    def _(event):
        if state.phases:
            state.selected = max(state.selected - 1, 0)

    @kb.add("g")
    def _(event):
        state.selected = 0

    @kb.add("G")
    def _(event):
        if state.phases:
            state.selected = len(state.phases) - 1

    @kb.add("r")
    def _(event):
        _intervention_prompt(event, state, kind="force_retry")

    @kb.add("n")
    def _(event):
        _intervention_prompt(event, state, kind="inject_note")

    @kb.add("e")
    def _(event):
        _intervention_prompt(event, state, kind="edit_refined_spec")

    @kb.add("b")
    def _(event):
        _intervention_prompt(event, state, kind="jump_back")

    @kb.add("enter")
    def _(event):
        _open_result_in_pager(event, state)

    @kb.add("R")
    def _(event):
        # Force reload from disk.
        state.phases = _load_phases(state.state_json)
        state.detail.clear()
        state.last_message = "reloaded"

    style = Style.from_dict({
        "pane-title": "bold bg:#222222",
        "selected": "reverse bold",
        "status-ok": "ansigreen",
        "status-run": "ansiyellow",
        "status-fail": "ansired",
        "status-pending": "ansiblack bold",
        "title": "bold reverse",
        "bar": "bg:#1c1c1c",
        "dim": "#888888",
        "retry-header": "bold #88ccff",
        "pane": "",
    })

    app: Application = Application(
        layout=Layout(root, focused_element=phase_list),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
        style=style,
    )

    # Background tick: refresh phase list from disk every 2s (live mode).
    async def _ticker() -> None:
        while True:
            try:
                state.phases = _load_phases(state.state_json)
                # Invalidate detail cache for the visible phase so retries
                # show up as soon as their manifest.json appears.
                if state.current_phase():
                    state.detail.pop(state.selected, None)
            except Exception:
                logger.exception("browse: ticker reload failed")
            app.invalidate()
            await asyncio.sleep(2.0)

    if state.mode == "live":
        async def _kickoff_ticker():
            asyncio.create_task(_ticker())
        app.pre_run_callables.append(_kickoff_ticker)

    return app


# ─── Intervention prompt + pager helpers ─────────────────────────────────


def _intervention_prompt(event, state: BrowseState, *, kind: str) -> None:
    """Suspend the app, prompt for input, then append a command."""
    cur = state.current_phase()
    if cur is None:
        state.last_message = "no phase selected"
        return
    if state.mode != "live":
        state.last_message = "post-hoc mode: interventions disabled"
        return

    from prompt_toolkit.shortcuts import input_dialog

    if kind == "force_retry":
        async def _coro():
            ans = await input_dialog(title="Force retry", text=f"Rationale for retrying '{cur.phase_id}':").run_async()
            if ans is None:
                return
            from ..commands import force_retry as send
            send(state.commands_path, phase_id=cur.phase_id, rationale=ans)
            state.last_message = f"force_retry queued for {cur.phase_id}"
        event.app.create_background_task(_coro())
    elif kind == "inject_note":
        async def _coro():
            text = await input_dialog(title="Inject note", text=f"Note text for {cur.phase_id}:").run_async()
            if not text:
                return
            from ..commands import inject_note as send
            send(state.commands_path, scope="phase", phase_id=cur.phase_id, text=text, rationale="operator note")
            state.last_message = f"note injected for {cur.phase_id}"
        event.app.create_background_task(_coro())
    elif kind == "edit_refined_spec":
        # Opening $EDITOR full-screen is cleaner than a dialog for prose.
        async def _coro():
            refined = state.workspace / "phases" / cur.phase_id / "context" / "refined_spec.md"
            refined.parent.mkdir(parents=True, exist_ok=True)
            if not refined.exists():
                refined.write_text("")
            ret = await event.app.run_system_command(f"${{EDITOR:-vi}} {refined}")
            new_content = refined.read_text() if refined.exists() else ""
            from ..commands import edit_refined_spec as send
            send(state.commands_path, phase_id=cur.phase_id, content=new_content, before_agent="builder", mode="replace", rationale="operator edit")
            state.last_message = f"refined_spec edit queued for {cur.phase_id}"
        event.app.create_background_task(_coro())
    elif kind == "jump_back":
        async def _coro():
            text = await input_dialog(title="Jump back", text=f"Jump back to phase id (will invalidate {cur.phase_id} and cascade):", default=cur.phase_id).run_async()
            if not text:
                return
            from ..commands import jump_back as send
            send(state.commands_path, to_phase_id=text.strip(), rationale="operator jump_back")
            state.last_message = f"jump_back queued to {text}"
        event.app.create_background_task(_coro())


def _open_result_in_pager(event, state: BrowseState) -> None:
    cur = state.current_phase()
    if cur is None:
        return
    candidate = state.workspace / "phases" / cur.phase_id / "outputs" / "_result.json"
    if not candidate.exists():
        # Fall back to manifest of the latest attempt if no _result.json yet.
        attempts = state.workspace / "phases" / cur.phase_id / "attempts"
        if attempts.exists():
            dirs = sorted([d for d in attempts.iterdir() if d.is_dir()], key=lambda p: p.name)
            if dirs:
                candidate = dirs[-1] / "manifest.json"
    if not candidate.exists():
        state.last_message = "no result/manifest yet"
        return

    async def _coro():
        await event.app.run_system_command(f"${{PAGER:-less -R}} {candidate}")

    event.app.create_background_task(_coro())


# ─── Entry points ────────────────────────────────────────────────────────


def run_browse(workspace: Path, mode: str = "post_hoc") -> int:
    """Synchronous entry point used by the rb-browse CLI."""
    workspace = Path(workspace).resolve()
    if not workspace.exists():
        print(f"workspace not found: {workspace}")
        return 1
    state = BrowseState(workspace=workspace, mode=mode)
    state.phases = _load_phases(state.state_json)
    app = build_app(state)
    app.run()
    return 0
