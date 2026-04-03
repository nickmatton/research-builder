"""Tests for dependency graph computation and traversal."""

import pytest

from research_builder.models.spec import Artifact, PhaseState, PhaseStatus, SpecMetadata, SpecState
from research_builder.orchestrator.dependency import DependencyGraph


def _make_state(statuses: dict[str, PhaseStatus] | None = None) -> SpecState:
    """Standard 5-phase spec state."""
    s = statuses or {}
    return SpecState(
        metadata=SpecMetadata(paper_id="test", paper_title="Test"),
        phases=[
            PhaseState(phase_id="data", title="Data", status=s.get("data", PhaseStatus.pending)),
            PhaseState(phase_id="architecture", title="Architecture", status=s.get("architecture", PhaseStatus.pending)),
            PhaseState(phase_id="training", title="Training", status=s.get("training", PhaseStatus.pending)),
            PhaseState(phase_id="eval", title="Eval", status=s.get("eval", PhaseStatus.pending)),
            PhaseState(phase_id="results", title="Results", status=s.get("results", PhaseStatus.pending)),
        ],
        dependency_graph={
            "data": [],
            "architecture": [],
            "training": ["data", "architecture"],
            "eval": ["training", "data"],
            "results": ["eval", "training"],
        },
    )


class TestDependencies:
    def test_no_deps(self):
        g = DependencyGraph({"data": [], "arch": []})
        assert g.get_dependencies("data") == []
        assert g.get_dependencies("arch") == []

    def test_has_deps(self):
        g = DependencyGraph.from_spec_state(_make_state())
        assert g.get_dependencies("training") == ["data", "architecture"]

    def test_unknown_phase(self):
        g = DependencyGraph({"data": []})
        assert g.get_dependencies("nonexistent") == []


class TestDownstream:
    def test_direct_downstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        ds = g.get_downstream("data")
        assert "training" in ds
        assert "eval" in ds

    def test_transitive_downstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        ds = g.get_downstream("data")
        assert "results" in ds  # data -> training -> results

    def test_leaf_has_no_downstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        assert g.get_downstream("results") == set()

    def test_architecture_downstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        ds = g.get_downstream("architecture")
        assert "training" in ds
        assert "eval" in ds  # arch -> training -> eval
        assert "results" in ds


class TestUpstream:
    def test_root_has_no_upstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        assert g.get_upstream("data") == set()

    def test_direct_upstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        us = g.get_upstream("training")
        assert "data" in us
        assert "architecture" in us

    def test_transitive_upstream(self):
        g = DependencyGraph.from_spec_state(_make_state())
        us = g.get_upstream("results")
        assert "eval" in us
        assert "training" in us
        assert "data" in us
        assert "architecture" in us


class TestRunnable:
    def test_initial_state(self):
        state = _make_state()
        g = DependencyGraph.from_spec_state(state)
        runnable = g.get_runnable(state)
        assert set(runnable) == {"data", "architecture"}

    def test_after_data_complete(self):
        state = _make_state({"data": PhaseStatus.completed})
        g = DependencyGraph.from_spec_state(state)
        runnable = g.get_runnable(state)
        assert "architecture" in runnable
        assert "training" not in runnable  # still needs architecture

    def test_after_both_roots_complete(self):
        state = _make_state({
            "data": PhaseStatus.completed,
            "architecture": PhaseStatus.completed,
        })
        g = DependencyGraph.from_spec_state(state)
        runnable = g.get_runnable(state)
        assert "training" in runnable

    def test_in_progress_not_runnable(self):
        state = _make_state({"data": PhaseStatus.in_progress})
        g = DependencyGraph.from_spec_state(state)
        runnable = g.get_runnable(state)
        assert "data" not in runnable
        assert "architecture" in runnable

    def test_failed_blocks_downstream(self):
        state = _make_state({
            "data": PhaseStatus.completed,
            "architecture": PhaseStatus.failed,
        })
        g = DependencyGraph.from_spec_state(state)
        runnable = g.get_runnable(state)
        assert "training" not in runnable

    def test_all_complete(self):
        state = _make_state({p: PhaseStatus.completed for p in
                            ["data", "architecture", "training", "eval", "results"]})
        g = DependencyGraph.from_spec_state(state)
        assert g.get_runnable(state) == []


class TestValidation:
    def test_valid_graph(self):
        state = _make_state()
        g = DependencyGraph.from_spec_state(state)
        phase_ids = {p.phase_id for p in state.phases}
        errors = g.validate(phase_ids)
        assert errors == []

    def test_unknown_dependency(self):
        g = DependencyGraph({"training": ["data", "nonexistent"]})
        errors = g.validate({"data", "training"})
        assert any("nonexistent" in e for e in errors)

    def test_phase_not_in_list(self):
        g = DependencyGraph({"orphan": []})
        errors = g.validate({"data"})
        assert any("orphan" in e for e in errors)

    def test_cycle_detected(self):
        g = DependencyGraph({"a": ["b"], "b": ["a"]})
        errors = g.validate({"a", "b"})
        assert any("Cycle" in e for e in errors)

    def test_self_cycle(self):
        g = DependencyGraph({"a": ["a"]})
        errors = g.validate({"a"})
        assert any("Cycle" in e for e in errors)
