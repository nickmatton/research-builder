"""Workspace directory management (spec_v4 §10)."""

from __future__ import annotations

from pathlib import Path

from ..config import Config


class WorkspaceManager:
    """Manages the file system layout for a reproduction run.

    Layout:
        project_root/
        ├── paper/
        ├── canonical_spec/
        │   ├── spec.md
        │   ├── state.yaml
        │   └── revision_log.yaml
        ├── phases/
        │   └── <phase_id>/
        │       └── <try_num>/
        │           ├── src/
        │           └── outputs/
        └── report/
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = Path(config.project_root)

    def initialize(self) -> None:
        """Create top-level directories for a new run."""
        self.config.paper_dir.mkdir(parents=True, exist_ok=True)
        self.config.spec_dir.mkdir(parents=True, exist_ok=True)
        self.config.phases_dir.mkdir(parents=True, exist_ok=True)
        self.config.report_dir.mkdir(parents=True, exist_ok=True)

    def phase_dir(self, phase_id: str) -> Path:
        return self.config.phases_dir / phase_id

    def attempt_dir(self, phase_id: str, try_num: int) -> Path:
        return self.phase_dir(phase_id) / str(try_num)

    def src_dir(self, phase_id: str, try_num: int) -> Path:
        return self.attempt_dir(phase_id, try_num) / "src"

    def outputs_dir(self, phase_id: str, try_num: int) -> Path:
        return self.attempt_dir(phase_id, try_num) / "outputs"

    def create_attempt(self, phase_id: str, try_num: int) -> Path:
        """Create directories for a new sub-agent attempt. Returns the attempt dir."""
        attempt = self.attempt_dir(phase_id, try_num)
        self.src_dir(phase_id, try_num).mkdir(parents=True, exist_ok=True)
        self.outputs_dir(phase_id, try_num).mkdir(parents=True, exist_ok=True)
        return attempt

    def next_try_num(self, phase_id: str) -> int:
        """Return the next attempt number for a phase (1-indexed)."""
        phase = self.phase_dir(phase_id)
        if not phase.exists():
            return 1
        existing = [int(d.name) for d in phase.iterdir() if d.is_dir() and d.name.isdigit()]
        return max(existing, default=0) + 1

    @property
    def spec_md_path(self) -> Path:
        return self.config.spec_dir / "spec.md"

    @property
    def state_path(self) -> Path:
        return self.config.spec_dir / "state.yaml"

    @property
    def revision_log_path(self) -> Path:
        return self.config.spec_dir / "revision_log.yaml"

    @property
    def report_path(self) -> Path:
        return self.config.report_dir / "reproduction_report.md"
