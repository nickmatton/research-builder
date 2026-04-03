"""Tests for SpecManager and OrchestratorAgent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_builder.config import Config
from research_builder.llm.client import LLMClient
from research_builder.models.results import ResultStatus, SubAgentResult, TestReport
from research_builder.models.spec import (
    Artifact,
    EventType,
    PhaseState,
    PhaseStatus,
    SpecMetadata,
    SpecState,
)
from research_builder.orchestrator.agent import OrchestratorAgent, _extract_json
from research_builder.orchestrator.spec_manager import SpecManager, _extract_phase_markdown
from research_builder.storage.spec_store import SpecStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_SPEC_MD = """\
# Canonical Spec: Test Paper

## Global Context

A test paper about widgets.

## Phase: Data

Download widget dataset from example.com.

### Acceptance Criteria
- 10,000 training samples
- 2,000 test samples

## Phase: Architecture

Build a 3-layer MLP.

### Acceptance Criteria
- Parameter count ~50K
- Output shape (batch, 10)

## Phase: Training

Train for 100 epochs with AdamW.

## Phase: Eval

Evaluate accuracy on test set.

## Phase: Results

Produce reproduction report.
"""


def _make_state() -> SpecState:
    return SpecState(
        metadata=SpecMetadata(paper_id="test", paper_title="Test Paper"),
        phases=[
            PhaseState(
                phase_id="data", title="Data",
                outputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
            ),
            PhaseState(
                phase_id="architecture", title="Architecture",
                outputs=[Artifact(name="model", file_path="phases/architecture/1/outputs/model.py")],
            ),
            PhaseState(
                phase_id="training", title="Training",
                inputs=[
                    Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt"),
                    Artifact(name="model", file_path="phases/architecture/1/outputs/model.py"),
                ],
                outputs=[Artifact(name="checkpoint", file_path="phases/training/1/outputs/checkpoint.pt")],
            ),
            PhaseState(
                phase_id="eval", title="Eval",
                inputs=[Artifact(name="checkpoint", file_path="phases/training/1/outputs/checkpoint.pt")],
                outputs=[Artifact(name="metrics", file_path="phases/eval/1/outputs/metrics.json")],
            ),
            PhaseState(
                phase_id="results", title="Results",
                inputs=[Artifact(name="metrics", file_path="phases/eval/1/outputs/metrics.json")],
            ),
        ],
        dependency_graph={
            "data": [],
            "architecture": [],
            "training": ["data", "architecture"],
            "eval": ["training"],
            "results": ["eval"],
        },
    )


@pytest.fixture
def store(tmp_path):
    return SpecStore(tmp_path / "canonical_spec")


@pytest.fixture
def spec_manager(store):
    state = _make_state()
    store.save_spec_md(SAMPLE_SPEC_MD)
    store.save_state(state)
    return SpecManager(store, state)


# ---------------------------------------------------------------------------
# SpecManager tests
# ---------------------------------------------------------------------------


class TestSpecManagerStatus:
    def test_set_phase_status(self, spec_manager):
        spec_manager.set_phase_status("data", PhaseStatus.in_progress, "starting data phase")
        assert spec_manager.state.get_phase("data").status == PhaseStatus.in_progress

        # Check revision was logged
        revisions = spec_manager.store.load_revision_log()
        assert any(r.event_type == EventType.phase_started for r in revisions)

    def test_set_completed(self, spec_manager):
        spec_manager.set_phase_status("data", PhaseStatus.completed, "all tests passed")
        revisions = spec_manager.store.load_revision_log()
        assert any(r.event_type == EventType.phase_completed for r in revisions)

    def test_set_failed(self, spec_manager):
        spec_manager.set_phase_status("data", PhaseStatus.failed, "could not download")
        revisions = spec_manager.store.load_revision_log()
        assert any(r.event_type == EventType.phase_failed for r in revisions)


class TestSpecManagerRunnable:
    def test_initial_runnable(self, spec_manager):
        runnable = spec_manager.get_runnable_phases()
        assert set(runnable) == {"data", "architecture"}

    def test_after_roots_complete(self, spec_manager):
        spec_manager.state.set_phase_status("data", PhaseStatus.completed)
        spec_manager.state.set_phase_status("architecture", PhaseStatus.completed)
        runnable = spec_manager.get_runnable_phases()
        assert "training" in runnable


class TestSpecManagerInvalidation:
    def test_invalidate_completed_phase(self, spec_manager):
        spec_manager.state.set_phase_status("data", PhaseStatus.completed)
        spec_manager.state.set_phase_status("architecture", PhaseStatus.completed)
        spec_manager.state.set_phase_status("training", PhaseStatus.completed)

        invalidated = spec_manager.invalidate_phase("data", "spec changed")
        assert "data" in invalidated
        assert "training" in invalidated  # downstream

    def test_cascade_invalidation(self, spec_manager):
        # Complete all phases
        for pid in ["data", "architecture", "training", "eval", "results"]:
            spec_manager.state.set_phase_status(pid, PhaseStatus.completed)

        invalidated = spec_manager.invalidate_phase("data", "data format changed")
        assert "data" in invalidated
        assert "training" in invalidated
        assert "eval" in invalidated
        assert "results" in invalidated
        # architecture should NOT be invalidated (not downstream of data)
        assert "architecture" not in invalidated

    def test_invalidate_pending_noop(self, spec_manager):
        invalidated = spec_manager.invalidate_phase("data", "no-op")
        assert invalidated == []

    def test_invalidation_logs_events(self, spec_manager):
        spec_manager.state.set_phase_status("data", PhaseStatus.completed)
        spec_manager.state.set_phase_status("training", PhaseStatus.completed)
        spec_manager.invalidate_phase("data", "spec changed")

        revisions = spec_manager.store.load_revision_log()
        invalidation_events = [r for r in revisions if r.event_type == EventType.phase_invalidated]
        assert len(invalidation_events) >= 2  # data + training


class TestSpecManagerSubSpec:
    def test_extract_sub_spec(self, spec_manager):
        sub_spec = spec_manager.extract_sub_spec("training", paper_path="/paper/paper.pdf")
        assert sub_spec.phase.phase_id == "training"
        assert sub_spec.paper_path == "/paper/paper.pdf"
        assert len(sub_spec.phase.inputs) == 2
        assert len(sub_spec.adjacent_phases) > 0

    def test_sub_spec_has_upstream_phases(self, spec_manager):
        sub_spec = spec_manager.extract_sub_spec("training")
        upstream_ids = {a.phase_id for a in sub_spec.adjacent_phases}
        assert "data" in upstream_ids
        assert "architecture" in upstream_ids

    def test_sub_spec_has_downstream_phases(self, spec_manager):
        sub_spec = spec_manager.extract_sub_spec("training")
        adj_ids = {a.phase_id for a in sub_spec.adjacent_phases}
        assert "eval" in adj_ids  # training -> eval

    def test_sub_spec_has_markdown(self, spec_manager):
        sub_spec = spec_manager.extract_sub_spec("data")
        assert "widget dataset" in sub_spec.spec_markdown

    def test_sub_spec_unknown_phase(self, spec_manager):
        with pytest.raises(ValueError, match="Unknown phase"):
            spec_manager.extract_sub_spec("nonexistent")


class TestSpecManagerAmend:
    def test_amend_spec_md(self, spec_manager):
        spec_manager.amend_spec_md("# Updated Spec\n\nNew content.", "hyperparams corrected")
        loaded = spec_manager.store.load_spec_md()
        assert "Updated Spec" in loaded

        revisions = spec_manager.store.load_revision_log()
        assert any(r.event_type == EventType.spec_amended for r in revisions)


# ---------------------------------------------------------------------------
# Markdown extraction tests
# ---------------------------------------------------------------------------


class TestExtractPhaseMarkdown:
    def test_extracts_data_section(self):
        md = _extract_phase_markdown(SAMPLE_SPEC_MD, "Data", "data")
        assert "widget dataset" in md
        assert "10,000" in md
        assert "3-layer MLP" not in md  # should not include architecture

    def test_extracts_architecture_section(self):
        md = _extract_phase_markdown(SAMPLE_SPEC_MD, "Architecture", "architecture")
        assert "3-layer MLP" in md
        assert "50K" in md

    def test_fallback_to_full(self):
        md = _extract_phase_markdown(SAMPLE_SPEC_MD, "Nonexistent", "nonexistent")
        assert md == SAMPLE_SPEC_MD  # falls back to full content

    def test_matches_by_phase_id(self):
        custom_md = "# Spec\n\n## data\n\nSome data info.\n\n## architecture\n\nModel info."
        md = _extract_phase_markdown(custom_md, "Data Phase", "data")
        assert "Some data info" in md
        assert "Model info" not in md


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_mixed_with_text(self):
        text = 'Here is my result: {"accept": true, "feedback": null}'
        result = _extract_json(text)
        assert result["accept"] is True

    def test_no_json(self):
        result = _extract_json("just plain text")
        assert result == {}


# ---------------------------------------------------------------------------
# OrchestratorAgent tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestOrchestratorAcceptanceReview:
    @pytest.mark.asyncio
    async def test_accept(self, tmp_path, spec_manager):
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)
        agent = OrchestratorAgent(config, llm_client)

        result = SubAgentResult(
            status=ResultStatus.success,
            phase_id="data",
            outputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
            summary="Downloaded and processed widget dataset",
            test_report=TestReport(tests_run=5, tests_passed=5, tests_failed=0),
        )

        # Mock LLM to return acceptance
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"accept": true, "feedback": null}')]

        async def mock_create(**kwargs):
            return mock_response

        with patch.object(llm_client, "create_message", side_effect=mock_create):
            accepted, feedback = await agent.acceptance_review("data", result, spec_manager)

        assert accepted is True
        assert feedback is None

    @pytest.mark.asyncio
    async def test_reject(self, tmp_path, spec_manager):
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)
        agent = OrchestratorAgent(config, llm_client)

        result = SubAgentResult(
            status=ResultStatus.success,
            phase_id="data",
            outputs=[],  # No outputs!
            summary="Done",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            type="text",
            text='{"accept": false, "feedback": "No output artifacts produced"}',
        )]

        async def mock_create(**kwargs):
            return mock_response

        with patch.object(llm_client, "create_message", side_effect=mock_create):
            accepted, feedback = await agent.acceptance_review("data", result, spec_manager)

        assert accepted is False
        assert "No output" in feedback
