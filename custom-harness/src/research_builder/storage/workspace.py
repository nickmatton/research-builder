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
        │   ├── state.json
        │   └── revision_log.json
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
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self.config.phases_dir.mkdir(parents=True, exist_ok=True)
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.config.logs_dir / "postmortems").mkdir(parents=True, exist_ok=True)
        (self.config.logs_dir / "spec_amendments").mkdir(parents=True, exist_ok=True)
        self.config.context_dir.mkdir(parents=True, exist_ok=True)

        # Symlink the source PDF into <workspace>/paper/paper.pdf so the
        # workspace is self-contained for downstream consumers (the web app's
        # PDF viewer, scripts that probe ./paper/, etc.). A symlink keeps
        # this cheap — no double-disk-usage even for huge supplements.
        # If symlink isn't supported (rare; some Windows configs) fall
        # back to a copy.
        src = Path(self.config.paper_path)
        if src.exists():
            link = self.config.paper_dir / "paper.pdf"
            if not link.exists():
                try:
                    link.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(src, link)

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

    def attempt_dir(self, phase_id: str, retry_num: int) -> Path:
        """Per-attempt directory for an agent-step trail (Stage 1)."""
        d = self.phase_dir(phase_id) / "attempts" / str(retry_num)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def step_record_path(self, phase_id: str, retry_num: int, role: str) -> Path:
        return self.attempt_dir(phase_id, retry_num) / f"{role}.json"

    def attempt_manifest_path(self, phase_id: str, retry_num: int) -> Path:
        return self.attempt_dir(phase_id, retry_num) / "manifest.json"

    @property
    def spec_md_path(self) -> Path:
        return self.config.spec_dir / "spec.md"

    @property
    def state_path(self) -> Path:
        return self.config.spec_dir / "state.json"

    @property
    def revision_log_path(self) -> Path:
        return self.config.spec_dir / "revision_log.json"

    @property
    def report_path(self) -> Path:
        return self.config.report_dir / "reproduction_report.md"

    @property
    def sections_dir(self) -> Path:
        return self.config.spec_dir / "sections"

    def section_spec_md_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.md"

    def section_spec_json_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.json"

    def section_critique_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.critique.json"
