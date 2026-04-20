"""Storage for the canonical spec: spec.md + state.yaml + revision_log.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

import json

from ..models.claims import ClaimsLedger
from ..models.spec import PlanDocument, Revision, SpecState


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
        self.dag_path = spec_dir / "dag.json"
        self.file_plan_path = spec_dir / "file_plan.json"
        self.claims_path = spec_dir / "claims.yaml"

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

    # --- claims.yaml ---

    def save_claims(self, ledger: ClaimsLedger) -> None:
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump(mode="json") for c in ledger.claims]
        self.claims_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def load_claims(self) -> ClaimsLedger:
        if not self.claims_path.exists():
            return ClaimsLedger()
        data = yaml.safe_load(self.claims_path.read_text()) or []
        from ..models.claims import Claim
        return ClaimsLedger(claims=[Claim.model_validate(c) for c in data])

    def append_revision(self, revision: Revision) -> None:
        existing: list[dict] = []
        if self.revision_log_path.exists():
            existing = yaml.safe_load(self.revision_log_path.read_text()) or []
        existing.append(revision.model_dump(mode="json"))
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.revision_log_path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=False))
