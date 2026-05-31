"""Storage for the canonical spec: spec.md + state.json + revision_log.json."""

from __future__ import annotations

import json
from pathlib import Path

from ..models.claims import ClaimsLedger
from ..models.spec import (
    PlanDocument,
    Revision,
    SectionCritique,
    SectionSpec,
    SpecState,
)


class SpecStore:
    """Reads and writes the canonical spec files.

    - spec.md:           Rich markdown document (LLM-authored)
    - state.json:        Machine-readable phase state, dependency graph, metadata
    - revision_log.json: Append-only event log
    - claims.json:       Numerical claims ledger
    """

    def __init__(self, spec_dir: Path) -> None:
        self.spec_dir = spec_dir
        self.spec_md_path = spec_dir / "spec.md"
        self.state_path = spec_dir / "state.json"
        self.revision_log_path = spec_dir / "revision_log.json"
        self.dag_path = spec_dir / "dag.json"
        self.file_plan_path = spec_dir / "file_plan.json"
        self.claims_path = spec_dir / "claims.json"
        self.sections_dir = spec_dir / "sections"

    # --- spec.md (markdown) ---

    def save_spec_md(self, content: str) -> None:
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.spec_md_path.write_text(content)

    def load_spec_md(self) -> str:
        if not self.spec_md_path.exists():
            return ""
        return self.spec_md_path.read_text()

    # --- state.json (machine state) ---

    def save_state(self, state: SpecState) -> None:
        data = state.model_dump(mode="json")
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))

    def load_state(self) -> SpecState:
        data = json.loads(self.state_path.read_text())
        return SpecState.model_validate(data)

    # --- revision_log.json ---

    def save_revision_log(self, revisions: list[Revision]) -> None:
        data = [r.model_dump(mode="json") for r in revisions]
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.revision_log_path.write_text(json.dumps(data, indent=2))

    def load_revision_log(self) -> list[Revision]:
        if not self.revision_log_path.exists():
            return []
        data = json.loads(self.revision_log_path.read_text() or "[]")
        return [Revision.model_validate(r) for r in data]

    # --- dag.json + file_plan.json (derived plan artifacts) ---

    def save_plan(self, plan: PlanDocument) -> None:
        """Write dag.json and file_plan.json as derived projections of the plan."""
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        dag_data = {"nodes": [n.model_dump(mode="json") for n in plan.nodes]}
        file_plan_data = {"files": [f.model_dump(mode="json") for f in plan.files]}
        self.dag_path.write_text(json.dumps(dag_data, indent=2))
        self.file_plan_path.write_text(json.dumps(file_plan_data, indent=2))

    def load_plan(self) -> PlanDocument | None:
        if not self.dag_path.exists() or not self.file_plan_path.exists():
            return None
        dag_data = json.loads(self.dag_path.read_text())
        file_plan_data = json.loads(self.file_plan_path.read_text())
        return PlanDocument.model_validate({
            "nodes": dag_data.get("nodes", []),
            "files": file_plan_data.get("files", []),
        })

    # --- claims.json ---

    def save_claims(self, ledger: ClaimsLedger) -> None:
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump(mode="json") for c in ledger.claims]
        self.claims_path.write_text(json.dumps(data, indent=2))

    def load_claims(self) -> ClaimsLedger:
        if not self.claims_path.exists():
            return ClaimsLedger()
        data = json.loads(self.claims_path.read_text() or "[]")
        from ..models.claims import Claim
        return ClaimsLedger(claims=[Claim.model_validate(c) for c in data])

    def append_revision(self, revision: Revision) -> None:
        existing: list[dict] = []
        if self.revision_log_path.exists():
            existing = json.loads(self.revision_log_path.read_text() or "[]")
        existing.append(revision.model_dump(mode="json"))
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.revision_log_path.write_text(json.dumps(existing, indent=2))

    # --- per-section specs (canonical_spec/sections/<phase_id>.{md,json}) ---

    def _section_md_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.md"

    def _section_json_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.json"

    def _section_critique_path(self, phase_id: str) -> Path:
        return self.sections_dir / f"{phase_id}.critique.json"

    def save_section_spec(self, spec: SectionSpec) -> None:
        """Persist a section spec as a markdown body + structured sidecar JSON.

        The .md is the human-editable surface; the .json carries the structured
        fields (acceptance criteria + citations) the validator and critic need.
        """
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self._section_md_path(spec.phase_id).write_text(spec.spec_markdown)
        sidecar = spec.model_dump(mode="json", exclude={"spec_markdown"})
        self._section_json_path(spec.phase_id).write_text(json.dumps(sidecar, indent=2))

    def load_section_spec(self, phase_id: str) -> SectionSpec | None:
        json_path = self._section_json_path(phase_id)
        md_path = self._section_md_path(phase_id)
        if not json_path.exists() or not md_path.exists():
            return None
        sidecar = json.loads(json_path.read_text())
        sidecar["spec_markdown"] = md_path.read_text()
        return SectionSpec.model_validate(sidecar)

    def list_section_spec_ids(self) -> list[str]:
        if not self.sections_dir.exists():
            return []
        return sorted(
            p.stem for p in self.sections_dir.glob("*.json")
            if not p.stem.endswith(".critique")
        )

    def save_section_critique(self, critique: SectionCritique) -> None:
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self._section_critique_path(critique.phase_id).write_text(
            json.dumps(critique.model_dump(mode="json"), indent=2)
        )

    def load_section_critique(self, phase_id: str) -> SectionCritique | None:
        path = self._section_critique_path(phase_id)
        if not path.exists():
            return None
        return SectionCritique.model_validate(json.loads(path.read_text()))
