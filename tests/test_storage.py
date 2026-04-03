"""Tests for storage layer (workspace + spec store)."""

from pathlib import Path

import pytest
import yaml

from research_builder.config import Config
from research_builder.models.spec import (
    Artifact,
    EventType,
    PhaseState,
    PhaseStatus,
    Revision,
    SpecMetadata,
    SpecState,
)
from research_builder.storage.spec_store import SpecStore
from research_builder.storage.workspace import WorkspaceManager


@pytest.fixture
def config(tmp_path):
    return Config(project_root=tmp_path)


@pytest.fixture
def workspace(config):
    ws = WorkspaceManager(config)
    ws.initialize()
    return ws


@pytest.fixture
def spec_store(config):
    return SpecStore(config.spec_dir)


def _make_state(**overrides):
    defaults = dict(
        metadata=SpecMetadata(paper_id="test:001", paper_title="Test Paper"),
        phases=[
            PhaseState(
                phase_id="data",
                title="Data Phase",
                outputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
            ),
            PhaseState(
                phase_id="arch",
                title="Architecture Phase",
                outputs=[Artifact(name="model", file_path="phases/arch/1/outputs/model.py")],
            ),
            PhaseState(
                phase_id="training",
                title="Training Phase",
                inputs=[
                    Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt"),
                    Artifact(name="model", file_path="phases/arch/1/outputs/model.py"),
                ],
            ),
        ],
        dependency_graph={"data": [], "arch": [], "training": ["data", "arch"]},
    )
    defaults.update(overrides)
    return SpecState(**defaults)


SAMPLE_SPEC_MD = """\
# Canonical Spec: Test Paper

## Global Context

This paper introduces a small CNN for CIFAR-10 classification.

## Phase: Data

Download and preprocess CIFAR-10. Split into train/val/test.

### Acceptance Criteria
- Train split has 45,000 samples
- Val split has 5,000 samples
- Test split has 10,000 samples

## Phase: Architecture

Implement a 3-layer CNN as described in Section 3.

### Acceptance Criteria
- Parameter count ~50K
- Output shape: (batch, 10)
"""


# --- WorkspaceManager ---


class TestWorkspaceManager:
    def test_initialize_creates_dirs(self, workspace, config):
        assert config.paper_dir.exists()
        assert config.spec_dir.exists()
        assert config.phases_dir.exists()
        assert config.report_dir.exists()

    def test_create_attempt(self, workspace):
        attempt = workspace.create_attempt("data", 1)
        assert attempt.exists()
        assert (attempt / "src").exists()
        assert (attempt / "outputs").exists()

    def test_next_try_num_empty(self, workspace):
        assert workspace.next_try_num("data") == 1

    def test_next_try_num_increments(self, workspace):
        workspace.create_attempt("data", 1)
        assert workspace.next_try_num("data") == 2
        workspace.create_attempt("data", 2)
        assert workspace.next_try_num("data") == 3

    def test_next_try_num_handles_gaps(self, workspace):
        workspace.create_attempt("data", 1)
        workspace.create_attempt("data", 5)
        assert workspace.next_try_num("data") == 6

    def test_phase_dir(self, workspace, config):
        assert workspace.phase_dir("data") == config.phases_dir / "data"

    def test_paths(self, workspace, config):
        assert workspace.spec_md_path == config.spec_dir / "spec.md"
        assert workspace.state_path == config.spec_dir / "state.yaml"
        assert workspace.revision_log_path == config.spec_dir / "revision_log.yaml"
        assert workspace.report_path == config.report_dir / "reproduction_report.md"


# --- SpecStore: spec.md ---


class TestSpecStoreMd:
    def test_save_and_load_md(self, spec_store):
        spec_store.save_spec_md(SAMPLE_SPEC_MD)
        loaded = spec_store.load_spec_md()
        assert "# Canonical Spec: Test Paper" in loaded
        assert "CIFAR-10" in loaded

    def test_load_md_missing(self, spec_store):
        assert spec_store.load_spec_md() == ""


# --- SpecStore: state.yaml ---


class TestSpecStoreState:
    def test_save_and_load_state(self, spec_store):
        state = _make_state()
        spec_store.save_state(state)
        loaded = spec_store.load_state()
        assert loaded.metadata.paper_id == "test:001"
        assert len(loaded.phases) == 3
        assert loaded.dependency_graph["training"] == ["data", "arch"]

    def test_phase_status_preserved(self, spec_store):
        state = _make_state()
        state.phases[0].status = PhaseStatus.completed
        spec_store.save_state(state)
        loaded = spec_store.load_state()
        assert loaded.phases[0].status == PhaseStatus.completed

    def test_artifact_paths_preserved(self, spec_store):
        state = _make_state()
        spec_store.save_state(state)
        loaded = spec_store.load_state()
        assert loaded.phases[0].outputs[0].file_path == "phases/data/1/outputs/loader.pt"


# --- SpecStore: revision_log.yaml ---


class TestSpecStoreRevisionLog:
    def test_save_and_load_revision_log(self, spec_store):
        revisions = [
            Revision(event_type=EventType.spec_created, rationale="initial draft"),
            Revision(event_type=EventType.phase_started, phase_id="data", rationale="starting data"),
        ]
        spec_store.save_revision_log(revisions)
        loaded = spec_store.load_revision_log()
        assert len(loaded) == 2
        assert loaded[0].event_type == EventType.spec_created
        assert loaded[1].phase_id == "data"

    def test_append_revision(self, spec_store):
        r1 = Revision(event_type=EventType.spec_created, rationale="created")
        r2 = Revision(event_type=EventType.phase_started, phase_id="data", rationale="starting")
        spec_store.append_revision(r1)
        spec_store.append_revision(r2)
        loaded = spec_store.load_revision_log()
        assert len(loaded) == 2
        assert loaded[1].phase_id == "data"

    def test_load_revision_log_missing(self, spec_store):
        assert spec_store.load_revision_log() == []


# --- SpecStore: combined workflow ---


class TestSpecStoreWorkflow:
    def test_full_save_load_cycle(self, spec_store):
        """Save spec.md + state + revisions, then load them all back."""
        state = _make_state()
        spec_store.save_spec_md(SAMPLE_SPEC_MD)
        spec_store.save_state(state)
        spec_store.append_revision(
            Revision(event_type=EventType.spec_created, rationale="initial")
        )

        md = spec_store.load_spec_md()
        loaded_state = spec_store.load_state()
        revisions = spec_store.load_revision_log()

        assert "CIFAR-10" in md
        assert loaded_state.metadata.paper_title == "Test Paper"
        assert len(revisions) == 1

    def test_state_and_md_are_independent(self, spec_store):
        """Updating state.yaml should not affect spec.md and vice versa."""
        spec_store.save_spec_md(SAMPLE_SPEC_MD)
        state = _make_state()
        spec_store.save_state(state)

        # Update state
        state.set_phase_status("data", PhaseStatus.completed)
        spec_store.save_state(state)

        # spec.md should be unchanged
        md = spec_store.load_spec_md()
        assert "CIFAR-10" in md

        # Update md
        spec_store.save_spec_md("# Updated spec")
        loaded_state = spec_store.load_state()
        assert loaded_state.get_phase("data").status == PhaseStatus.completed


# --- Config ---


class TestConfig:
    def test_default_paths(self):
        cfg = Config()
        assert cfg.paper_dir == Path("paper")
        assert cfg.spec_dir == Path("canonical_spec")
        assert cfg.phases_dir == Path("phases")
        assert cfg.report_dir == Path("report")

    def test_custom_root(self, tmp_path):
        cfg = Config(project_root=tmp_path)
        assert cfg.paper_dir == tmp_path / "paper"
        assert cfg.phases_dir == tmp_path / "phases"

    def test_defaults(self):
        cfg = Config()
        assert cfg.model == "claude-opus-4-6"
        assert cfg.max_retries == 3
        assert cfg.max_debug_attempts == 10
