"""Spec management: creation, amendment, impact evaluation, sub-spec extraction (spec_v4 §4.1, §4.5, §4.6)."""

from __future__ import annotations

import logging
from datetime import datetime

from ..models.context import AdjacentPhaseSummary, SubSpec
from ..models.spec import EventType, PhaseStatus, Revision, SpecState
from ..storage.spec_store import SpecStore
from .dependency import DependencyGraph

logger = logging.getLogger(__name__)


class SpecManager:
    """Owns the canonical spec state. Handles amendments and sub-spec extraction."""

    MAX_AMENDMENTS_PER_PHASE = 5

    def __init__(self, store: SpecStore, state: SpecState) -> None:
        self.store = store
        self.state = state
        self.dep_graph = DependencyGraph.from_spec_state(state)
        # Per-phase amendment count for the spec refinement loop. Capped at
        # MAX_AMENDMENTS_PER_PHASE — at the cap we escalate to the human
        # checkpoint instead of looping forever.
        self._amendment_counts: dict[str, int] = {}

    def amendment_count(self, phase_id: str) -> int:
        return self._amendment_counts.get(phase_id, 0)

    def can_amend(self, phase_id: str) -> bool:
        return self.amendment_count(phase_id) < self.MAX_AMENDMENTS_PER_PHASE

    def record_amendment(self, phase_id: str) -> int:
        """Bump the per-phase amendment count and return the new count."""
        n = self._amendment_counts.get(phase_id, 0) + 1
        self._amendment_counts[phase_id] = n
        return n

    def save(self) -> None:
        """Persist current state to disk."""
        self.state.metadata.last_modified = datetime.now()
        self.store.save_state(self.state)

    def log_event(self, event_type: EventType, phase_id: str | None = None, rationale: str = "") -> None:
        """Append a revision to the log and persist."""
        rev = Revision(event_type=event_type, phase_id=phase_id, rationale=rationale)
        self.store.append_revision(rev)

    def set_phase_status(self, phase_id: str, status: PhaseStatus, rationale: str = "") -> None:
        """Update a phase's status, log the event, and persist."""
        self.state.set_phase_status(phase_id, status)

        event_map = {
            PhaseStatus.in_progress: EventType.phase_started,
            PhaseStatus.completed: EventType.phase_completed,
            PhaseStatus.failed: EventType.phase_failed,
        }
        if status in event_map:
            self.log_event(event_map[status], phase_id=phase_id, rationale=rationale)
        self.save()

    def get_runnable_phases(self) -> list[str]:
        """Return phase IDs that are ready to execute."""
        return self.dep_graph.get_runnable(self.state)

    def invalidate_phase(self, phase_id: str, rationale: str = "") -> list[str]:
        """Set a completed phase back to pending and cascade to downstream phases.

        Returns list of all invalidated phase IDs.
        """
        invalidated: list[str] = []
        phase = self.state.get_phase(phase_id)
        if phase is None:
            return invalidated

        if phase.status == PhaseStatus.completed:
            phase.status = PhaseStatus.pending
            self.log_event(EventType.phase_invalidated, phase_id=phase_id, rationale=rationale)
            invalidated.append(phase_id)

        # Cascade to downstream
        for downstream_id in self.dep_graph.get_downstream(phase_id):
            ds_phase = self.state.get_phase(downstream_id)
            if ds_phase and ds_phase.status == PhaseStatus.completed:
                ds_phase.status = PhaseStatus.pending
                self.log_event(
                    EventType.phase_invalidated,
                    phase_id=downstream_id,
                    rationale=f"Upstream phase '{phase_id}' was invalidated: {rationale}",
                )
                invalidated.append(downstream_id)

        if invalidated:
            self.save()
        return invalidated

    def amend_spec_md(self, new_content: str, rationale: str) -> None:
        """Replace the spec markdown and log the amendment."""
        self.store.save_spec_md(new_content)
        self.log_event(EventType.spec_amended, rationale=rationale)

    def extract_sub_spec(self, phase_id: str, paper_path: str = "") -> SubSpec:
        """Slice the spec into a SubSpec for a sub-agent (§4.5)."""
        phase = self.state.get_phase(phase_id)
        if phase is None:
            raise ValueError(f"Unknown phase: {phase_id}")

        # Build adjacent phase summaries
        adjacent: list[AdjacentPhaseSummary] = []
        # Upstream phases (this phase depends on)
        for dep_id in self.dep_graph.get_dependencies(phase_id):
            dep = self.state.get_phase(dep_id)
            if dep:
                adjacent.append(AdjacentPhaseSummary(
                    phase_id=dep.phase_id,
                    title=dep.title,
                    inputs=dep.inputs,
                    outputs=dep.outputs,
                ))
        # Downstream phases (depend on this phase)
        for downstream_id in self.dep_graph.get_downstream(phase_id):
            ds = self.state.get_phase(downstream_id)
            if ds and ds.phase_id not in {a.phase_id for a in adjacent}:
                adjacent.append(AdjacentPhaseSummary(
                    phase_id=ds.phase_id,
                    title=ds.title,
                    inputs=ds.inputs,
                    outputs=ds.outputs,
                ))

        # Load relevant spec markdown
        full_md = self.store.load_spec_md()
        phase_md = _extract_phase_markdown(full_md, phase.title, phase_id)

        return SubSpec(
            phase=phase,
            spec_markdown=phase_md,
            adjacent_phases=adjacent,
            paper_path=paper_path,
        )


def _extract_phase_markdown(full_md: str, phase_title: str, phase_id: str) -> str:
    """Extract the section of spec.md relevant to a phase.

    Looks for a heading containing the phase title or phase_id and returns
    everything until the next same-level heading.
    """
    lines = full_md.split("\n")
    result_lines: list[str] = []
    capturing = False
    capture_level = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped.lstrip("#").strip().lower()

            if not capturing:
                # Start capturing if heading matches phase
                if phase_title.lower() in heading_text or phase_id.lower() in heading_text:
                    capturing = True
                    capture_level = level
                    result_lines.append(line)
            else:
                # Stop if we hit a same-level or higher heading
                if level <= capture_level:
                    break
                result_lines.append(line)
        elif capturing:
            result_lines.append(line)

    return "\n".join(result_lines).strip() if result_lines else full_md
