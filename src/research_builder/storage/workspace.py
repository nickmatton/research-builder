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
        │       ├── src/
        │       └── outputs/
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
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.config.logs_dir / "postmortems").mkdir(parents=True, exist_ok=True)
        (self.config.logs_dir / "spec_amendments").mkdir(parents=True, exist_ok=True)
        self.config.context_dir.mkdir(parents=True, exist_ok=True)

    def postmortem_path(self, phase_id: str, retry_num: int = 0) -> Path:
        d = self.config.logs_dir / "postmortems" / phase_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"retry_{retry_num}.md"

    def amendment_path(self, phase_id: str, amendment_num: int) -> Path:
        d = self.config.logs_dir / "spec_amendments" / phase_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"amendment_{amendment_num}.md"

    def phase_dir(self, phase_id: str) -> Path:
        return self.config.phases_dir / phase_id

    def src_dir(self, phase_id: str) -> Path:
        return self.phase_dir(phase_id) / "src"

    def outputs_dir(self, phase_id: str) -> Path:
        return self.phase_dir(phase_id) / "outputs"

    def create_phase_dir(self, phase_id: str) -> Path:
        """Create directories for a phase. Returns the phase dir."""
        phase = self.phase_dir(phase_id)
        self.src_dir(phase_id).mkdir(parents=True, exist_ok=True)
        self.outputs_dir(phase_id).mkdir(parents=True, exist_ok=True)
        return phase

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
