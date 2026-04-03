"""Storage for the canonical spec: spec.md + state.yaml + revision_log.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..models.spec import Revision, SpecState


class SpecStore:
    """Reads and writes the three canonical spec files.

    - spec.md:           Rich markdown document (LLM-authored)
    - state.yaml:        Machine-readable phase state, dependency graph, metadata
    - revision_log.yaml: Append-only event log
    """

    def __init__(self, spec_dir: Path) -> None:
        self.spec_dir = spec_dir
        self.spec_md_path = spec_dir / "spec.md"
        self.state_path = spec_dir / "state.yaml"
        self.revision_log_path = spec_dir / "revision_log.yaml"

    # --- spec.md (markdown) ---

    def save_spec_md(self, content: str) -> None:
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.spec_md_path.write_text(content)

    def load_spec_md(self) -> str:
        if not self.spec_md_path.exists():
            return ""
        return self.spec_md_path.read_text()

    # --- state.yaml (machine state) ---

    def save_state(self, state: SpecState) -> None:
        data = state.model_dump(mode="json")
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def load_state(self) -> SpecState:
        data = yaml.safe_load(self.state_path.read_text())
        return SpecState.model_validate(data)

    # --- revision_log.yaml ---

    def save_revision_log(self, revisions: list[Revision]) -> None:
        data = [r.model_dump(mode="json") for r in revisions]
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.revision_log_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def load_revision_log(self) -> list[Revision]:
        if not self.revision_log_path.exists():
            return []
        data = yaml.safe_load(self.revision_log_path.read_text()) or []
        return [Revision.model_validate(r) for r in data]

    def append_revision(self, revision: Revision) -> None:
        existing: list[dict] = []
        if self.revision_log_path.exists():
            existing = yaml.safe_load(self.revision_log_path.read_text()) or []
        existing.append(revision.model_dump(mode="json"))
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.revision_log_path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=False))
