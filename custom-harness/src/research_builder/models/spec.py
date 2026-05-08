"""Data models for the canonical spec state (spec_v4 §2.1–2.4).

The canonical spec lives in two files:
  - spec.md:  Rich, LLM-authored markdown document with full descriptions,
              acceptance criteria, spec_details, etc.
  - state.yaml: Lightweight machine-readable state — phase statuses,
                dependency graph, artifact paths, metadata.

These models represent state.yaml. The markdown document is read/written
as plain text by the orchestrator LLM.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PhaseStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class EventType(str, Enum):
    spec_created = "spec_created"
    phase_started = "phase_started"
    phase_completed = "phase_completed"
    phase_failed = "phase_failed"
    retry_launched = "retry_launched"
    spec_amended = "spec_amended"
    ambiguity_resolved = "ambiguity_resolved"
    phase_invalidated = "phase_invalidated"
    run_completed = "run_completed"
    run_failed = "run_failed"


class Artifact(BaseModel):
    """Inter-phase data reference (§7)."""
    name: str
    file_path: str


class Revision(BaseModel):
    """Entry in the append-only revision log (§2.4)."""
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType
    phase_id: str | None = None
    rationale: str


class PhaseState(BaseModel):
    """Machine-readable state for a single phase."""
    phase_id: str
    title: str
    status: PhaseStatus = PhaseStatus.pending
    inputs: list[Artifact] = Field(default_factory=list)
    outputs: list[Artifact] = Field(default_factory=list)
    max_debug_attempts: int = 10


class FileRole(str, Enum):
    input = "input"
    output = "output"
    intermediate = "intermediate"


class FileStatus(str, Enum):
    planned = "planned"
    in_progress = "in_progress"
    written = "written"
    verified = "verified"


class PlannedFile(BaseModel):
    """A file the spec plans to produce/consume, tracked through its lifecycle."""
    file_id: str
    rel_path: str
    owning_phase: str
    role: FileRole
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    status: FileStatus = FileStatus.planned


class DagNode(BaseModel):
    """A planning-view node: phase plus its sub-steps and owned files."""
    phase_id: str
    title: str
    description: str = ""
    sub_steps: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: PhaseStatus = PhaseStatus.pending


class PlanDocument(BaseModel):
    """Explicit DAG + file plan emitted by the orchestrator at spec time."""
    nodes: list[DagNode] = Field(default_factory=list)
    files: list[PlannedFile] = Field(default_factory=list)

    def get_node(self, phase_id: str) -> DagNode | None:
        for n in self.nodes:
            if n.phase_id == phase_id:
                return n
        return None

    def files_for_phase(self, phase_id: str) -> list[PlannedFile]:
        return [f for f in self.files if f.owning_phase == phase_id]

    def get_file(self, file_id: str) -> PlannedFile | None:
        for f in self.files:
            if f.file_id == file_id:
                return f
        return None


class SpecMetadata(BaseModel):
    paper_id: str
    paper_title: str
    paper_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)


class SpecState(BaseModel):
    """The machine-readable state file (state.yaml).

    Tracks metadata, phase statuses, dependency graph, and artifact paths.
    The rich spec content lives in spec.md.
    """
    metadata: SpecMetadata
    phases: list[PhaseState] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    plan: PlanDocument | None = None

    def get_phase(self, phase_id: str) -> PhaseState | None:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None

    def set_phase_status(self, phase_id: str, status: PhaseStatus) -> None:
        phase = self.get_phase(phase_id)
        if phase is None:
            raise ValueError(f"Unknown phase: {phase_id}")
        phase.status = status
        self.metadata.last_modified = datetime.now()
