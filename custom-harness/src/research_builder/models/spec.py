"""Data models for the canonical spec state (spec_v4 §2.1–2.4).

The canonical spec lives in two files:
  - spec.md:  Rich, LLM-authored markdown document with full descriptions,
              acceptance criteria, spec_details, etc.
  - state.json: Lightweight machine-readable state — phase statuses,
                dependency graph, artifact paths, metadata.

These models represent state.json. The markdown document is read/written
as plain text by the orchestrator LLM.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class PhaseStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class PhaseKind(str, Enum):
    """Classification of a phase's role.

    ``build``      — produces reusable code/infrastructure (model architectures,
                     algorithms, data loaders, training scaffolding).
    ``experiment`` — runs a specific experiment from the paper to validate a
                     numerical claim (typically the §5+ benchmark sections).
    """
    build = "build"
    experiment = "experiment"


class EventType(str, Enum):
    spec_created = "spec_created"
    phase_started = "phase_started"
    phase_completed = "phase_completed"
    phase_failed = "phase_failed"
    retry_launched = "retry_launched"
    spec_amended = "spec_amended"
    ambiguity_resolved = "ambiguity_resolved"
    phase_invalidated = "phase_invalidated"
    user_intervened = "user_intervened"
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
    # Default to ``build`` so state.json files written before this field
    # existed still parse. The planner is now expected to set it explicitly.
    kind: PhaseKind = PhaseKind.build
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

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v: object) -> object:
        # The orchestrator LLM frequently invents category-descriptive role
        # labels ("core_module", "training_script", "task_spec"). The role
        # field is a label, not load-bearing logic — keep the file under a
        # default role rather than dropping it.
        if isinstance(v, FileRole):
            return v
        if isinstance(v, str):
            normalized = v.strip().lower()
            try:
                return FileRole(normalized)
            except ValueError:
                if normalized:
                    logger.debug("coercing unknown FileRole %r -> output", v)
                return FileRole.output
        return FileRole.output


class DagNode(BaseModel):
    """A planning-view node: phase plus its sub-steps and owned files."""
    phase_id: str
    title: str
    description: str = ""
    kind: PhaseKind = PhaseKind.build
    sub_steps: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: PhaseStatus = PhaseStatus.pending

    @field_validator("sub_steps", mode="before")
    @classmethod
    def _coerce_sub_steps(cls, v: object) -> object:
        # The LLM occasionally emits sub_steps items as {title: description}
        # dicts instead of plain strings. Flatten to "title: description" so
        # the whole node isn't dropped during validation.
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append("; ".join(f"{k}: {val}" for k, val in item.items()))
            else:
                out.append(str(item))
        return out


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
    """The machine-readable state file (state.json).

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


class Citation(BaseModel):
    """A pointer to the source paper. Page is required so claims are always grounded."""
    page: int
    section: str | None = None
    quote: str | None = None


class AcceptanceCriterion(BaseModel):
    """One acceptance criterion in a section spec, with a mandatory source citation."""
    text: str
    source: Citation


class CritiqueVerdict(str, Enum):
    verified = "verified"
    questionable = "questionable"
    missing_citations = "missing_citations"


class SectionCritique(BaseModel):
    """Critic agent's verdict on a section spec, persisted alongside the spec."""
    phase_id: str
    verdict: CritiqueVerdict
    reasons: list[str] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=datetime.now)


class SectionSpec(BaseModel):
    """Per-section spec authored upfront, in parallel with other sections.

    Stored at ``canonical_spec/sections/<phase_id>.md`` plus a sidecar JSON
    holding the structured fields below. The markdown body is the human-editable
    surface; the structured fields are the machine-checkable ones.
    """
    phase_id: str
    title: str
    goal: str
    spec_markdown: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    def validate_citations(self) -> None:
        """Raise ValueError if any acceptance criterion lacks a source citation.

        Source is enforced at the type level (AcceptanceCriterion.source is
        required), so this method exists for explicit pre-execution validation
        and to give callers a single failure mode to handle.
        """
        missing = [c.text for c in self.acceptance_criteria if c.source is None]
        if missing:
            raise ValueError(
                f"SectionSpec {self.phase_id} has uncited acceptance criteria: {missing}"
            )
