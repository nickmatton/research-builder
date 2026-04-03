"""Data models for retry context, sub-specs, and run state (spec_v4 §4.4–4.5)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .results import SubAgentResult
from .spec import Artifact, PhaseState


class SubSpec(BaseModel):
    """The payload handed to a sub-agent (§4.5).

    Combines machine-readable state (phase state, artifact paths) with
    the relevant markdown sections sliced from spec.md.
    """
    phase: PhaseState
    spec_markdown: str = ""           # relevant sections from spec.md
    adjacent_phases: list[AdjacentPhaseSummary] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    paper_path: str = ""


class AdjacentPhaseSummary(BaseModel):
    """Read-only summary of an adjacent phase — enough for interface contracts."""
    phase_id: str
    title: str
    inputs: list[Artifact] = Field(default_factory=list)
    outputs: list[Artifact] = Field(default_factory=list)


# Rebuild SubSpec now that AdjacentPhaseSummary is defined
SubSpec.model_rebuild()


class RetryContext(BaseModel):
    """Context passed to sub-agent on retry (§4.4)."""
    prior_results: list[SubAgentResult] = Field(default_factory=list)
    orchestrator_feedback: str | None = None


class RunStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"


class RunState(BaseModel):
    """Tracks overall run progress."""
    status: RunStatus = RunStatus.running
    orchestrator_retries: dict[str, int] = Field(default_factory=dict)
    phase_results: dict[str, list[SubAgentResult]] = Field(default_factory=dict)
