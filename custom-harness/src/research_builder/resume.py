"""Detect partially-completed runs and let the user resume or start fresh.

A "run" lives under ``project_root`` (the ``-o`` directory). State is split
across:

  - ``canonical_spec/state.json``    — phase statuses
  - ``canonical_spec/spec.md``       — generated spec
  - ``canonical_spec/dag.json``      — plan DAG
  - ``canonical_spec/file_plan.json``— planned files
  - ``phases/<phase_id>/``            — phase artifacts
  - ``logs/events.jsonl``            — viewer event stream
  - ``logs/commands.jsonl``          — viewer command stream

A run is "in progress" if ``state.json`` exists with at least one phase that
isn't ``pending``. Resume = leave everything in place; ExecutionLoop already
picks runnable phases out of the existing dependency graph and skips
completed ones. Fresh = archive the in-progress state to
``.archive/<timestamp>/`` so the user can recover it later, and truncate the
JSONL streams so the viewer doesn't replay stale events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import json


@dataclass
class ExistingRun:
    spec_dir: Path
    state_path: Path
    total_phases: int
    completed: int
    in_progress: int
    failed: int
    # Stale workspace = canonical_spec/ has content but state.json is missing
    # or unparseable. The run isn't resumable; the user can only wipe or
    # archive. Detected so we don't silently proceed and overwrite leftovers.
    stale: bool = False
    stale_artifacts: list[str] = field(default_factory=list)

    @property
    def is_in_progress(self) -> bool:
        return (self.completed + self.in_progress + self.failed) > 0 or self.stale

    def summary(self) -> str:
        if self.stale:
            return (
                f"stale workspace (no state.json; leftovers: "
                f"{', '.join(self.stale_artifacts)})"
            )
        parts = [f"{self.completed}/{self.total_phases} completed"]
        if self.in_progress:
            parts.append(f"{self.in_progress} in progress")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


# Filenames inside canonical_spec/ that indicate a prior run reached at
# least spec-authoring. Listed explicitly so detect() distinguishes harness
# artifacts from user-placed files in canonical_spec/.
_STALE_SPEC_FILES = (
    "spec.md",
    "state.json",
    "state.yaml",
    "state.yaml.broken",
    "claims.json",
    "claims.yaml",
    "revision_log.json",
    "revision_log.yaml",
    "dag.json",
    "file_plan.json",
)

# Sibling directories that only exist if a prior run got past initialization.
# (paper/ is user-supplied input, not a stale artifact — excluded.)
_STALE_SIBLING_DIRS = (
    "phases",
    "report",
    "traces",
    "context",
)


def _stale_workspace(spec_dir: Path) -> ExistingRun | None:
    """Detect leftovers from a prior aborted run.

    Looks inside canonical_spec/ AND at sibling directories under
    project_root, so a workspace where the prior run created phases/
    or logs/ but never wrote state.json is still flagged.
    """
    project_root = spec_dir.parent
    found: list[str] = []
    if spec_dir.exists():
        found.extend(name for name in _STALE_SPEC_FILES if (spec_dir / name).exists())
    for name in _STALE_SIBLING_DIRS:
        d = project_root / name
        if d.exists() and any(d.iterdir()):
            found.append(name + "/")
    # logs/ counts only when populated by a prior run beyond the bare
    # commands.jsonl/events.jsonl streams the listener creates on startup.
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        if (logs_dir / "run.log").exists() or any((logs_dir / sub).exists() for sub in ("postmortems", "spec_amendments")):
            found.append("logs/")
    if not found:
        return None
    return ExistingRun(
        spec_dir=spec_dir,
        state_path=spec_dir / "state.json",
        total_phases=0,
        completed=0,
        in_progress=0,
        failed=0,
        stale=True,
        stale_artifacts=found,
    )


def detect(project_root: Path) -> ExistingRun | None:
    """Return an :class:`ExistingRun` if a partial or stale run is present.

    Two cases:
      - Resumable: state.json exists, parses, and has at least one phase
        with a non-pending status. ExecutionLoop can pick up from here.
      - Stale: canonical_spec/ has content but no parseable state.json
        (e.g. a prior run crashed during spec creation, or files were
        renamed for forensics). Not resumable; the operator must wipe
        or archive to avoid overwriting leftovers silently.
    """
    spec_dir = project_root / "canonical_spec"
    state_path = spec_dir / "state.json"
    if not state_path.exists():
        return _stale_workspace(spec_dir)
    try:
        data = json.loads(state_path.read_text() or "{}")
    except Exception:
        return _stale_workspace(spec_dir)
    phases = data.get("phases") or []
    if not phases:
        return _stale_workspace(spec_dir)

    completed = sum(1 for p in phases if p.get("status") == "completed")
    in_progress = sum(1 for p in phases if p.get("status") == "in_progress")
    failed = sum(1 for p in phases if p.get("status") == "failed")

    run = ExistingRun(
        spec_dir=spec_dir,
        state_path=state_path,
        total_phases=len(phases),
        completed=completed,
        in_progress=in_progress,
        failed=failed,
    )
    return run if run.is_in_progress else _stale_workspace(spec_dir)


# Harness-owned paths under project_root that wipe/archive may safely touch.
# Crucially excludes ``paper/`` (user input!), ``.claude/`` (skills/commands),
# ``pyproject.toml``, ``src/``, ``scripts/``, and any nested dirs we don't
# recognize — operators sometimes nest a prior workspace at <root>/<paper>/
# and we must not blow that away when wiping the parent.
_HARNESS_DIRS = (
    "canonical_spec",
    "phases",
    "report",
    "traces",
    "context",
    "logs",
    "notes",
)
_HARNESS_FILES = (
    "CLAUDE.md",
)


def archive_and_clear(project_root: Path) -> Path:
    """Move existing run state aside so a fresh run can start cleanly.

    Returns the archive directory the old state was moved to. Only touches
    harness-managed paths (see ``_HARNESS_DIRS`` / ``_HARNESS_FILES``);
    leaves ``paper/``, ``.claude/``, and any user-placed files alone.
    """
    import shutil

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = project_root / ".archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name in _HARNESS_DIRS + _HARNESS_FILES:
        src = project_root / name
        if not src.exists():
            continue
        # ``logs/`` is special — preserve the empty dir but move its
        # contents so the listener can immediately re-attach to a fresh
        # commands.jsonl without racing the archive step.
        if name == "logs" and src.is_dir():
            for child in list(src.iterdir()):
                shutil.move(str(child), str(archive_dir / child.name))
        else:
            shutil.move(str(src), str(archive_dir / name))

    return archive_dir


def wipe(project_root: Path) -> None:
    """Delete harness-managed paths under project_root.

    PRESERVES ``paper/`` (user input), ``.claude/``, ``pyproject.toml``,
    ``src/``, ``scripts/``, and any nested dir we don't manage (e.g. an
    old workspace at ``<root>/<paper-slug>/``). Operators who want a
    truly empty directory can ``rm -rf`` it themselves.
    """
    import shutil

    if not project_root.exists():
        return
    for name in _HARNESS_DIRS + _HARNESS_FILES:
        target = project_root / name
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
