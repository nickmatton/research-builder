"""Detect partially-completed runs and let the user resume or start fresh.

A "run" lives under ``project_root`` (the ``-o`` directory). State is split
across:

  - ``canonical_spec/state.yaml``    — phase statuses
  - ``canonical_spec/spec.md``       — generated spec
  - ``canonical_spec/dag.json``      — plan DAG
  - ``canonical_spec/file_plan.json``— planned files
  - ``phases/<phase_id>/``            — phase artifacts
  - ``logs/events.jsonl``            — viewer event stream
  - ``logs/commands.jsonl``          — viewer command stream

A run is "in progress" if ``state.yaml`` exists with at least one phase that
isn't ``pending``. Resume = leave everything in place; ExecutionLoop already
picks runnable phases out of the existing dependency graph and skips
completed ones. Fresh = archive the in-progress state to
``.archive/<timestamp>/`` so the user can recover it later, and truncate the
JSONL streams so the viewer doesn't replay stale events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ExistingRun:
    spec_dir: Path
    state_path: Path
    total_phases: int
    completed: int
    in_progress: int
    failed: int

    @property
    def is_in_progress(self) -> bool:
        return (self.completed + self.in_progress + self.failed) > 0

    def summary(self) -> str:
        parts = [f"{self.completed}/{self.total_phases} completed"]
        if self.in_progress:
            parts.append(f"{self.in_progress} in progress")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


def detect(project_root: Path) -> ExistingRun | None:
    """Return an :class:`ExistingRun` if a partial run is present, else None."""
    spec_dir = project_root / "canonical_spec"
    state_path = spec_dir / "state.yaml"
    if not state_path.exists():
        return None
    try:
        data = yaml.safe_load(state_path.read_text()) or {}
    except Exception:
        return None
    phases = data.get("phases") or []
    if not phases:
        return None

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
    return run if run.is_in_progress else None


def archive_and_clear(project_root: Path) -> Path:
    """Move existing run state aside so a fresh run can start cleanly.

    Returns the archive directory the old state was moved to.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = project_root / ".archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name in ("canonical_spec", "phases", "report"):
        src = project_root / name
        if src.exists():
            src.rename(archive_dir / name)

    # Truncate (not delete) the viewer streams so a freshly-attached viewer
    # doesn't replay events from the archived run.
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        for stream in ("events.jsonl", "commands.jsonl"):
            p = logs_dir / stream
            if p.exists():
                # Move the old stream alongside the archived state for forensics.
                p.rename(archive_dir / stream)

    return archive_dir


def wipe(project_root: Path) -> None:
    """Delete the entire run directory and all its contents."""
    import shutil

    if project_root.exists():
        shutil.rmtree(project_root)
