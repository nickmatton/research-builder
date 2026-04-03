"""End-to-end integration test: full pipeline with mocked Agent SDK."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from research_builder.config import Config
from research_builder.main import run_pipeline
from research_builder.models.results import ResultStatus, SubAgentResult, TestReport
from research_builder.models.spec import Artifact
from research_builder.orchestrator.agent import OrchestratorAgent
from research_builder.sub_agent.agent import SubAgent

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_paper.pdf"


def _spec_creation_response() -> str:
    """JSON response for spec creation."""
    return json.dumps({
        "spec_md": (
            "# Canonical Spec: Widget Performance Study\n\n"
            "## Global Context\n\nThis paper studies widget performance.\n\n"
            "## Phase: Data\n\nDownload widget dataset.\n\n"
            "## Phase: Architecture\n\nBuild a 3-layer MLP.\n\n"
            "## Phase: Training\n\nTrain with AdamW.\n"
        ),
        "state": {
            "metadata": {"paper_id": "test:widget", "paper_title": "Widget Performance Study"},
            "phases": [
                {"phase_id": "data", "title": "Data", "inputs": [],
                 "outputs": [{"name": "loader", "file_path": "phases/data/1/outputs/loader.pt"}]},
                {"phase_id": "architecture", "title": "Architecture", "inputs": [],
                 "outputs": [{"name": "model", "file_path": "phases/architecture/1/outputs/model.py"}]},
                {"phase_id": "training", "title": "Training",
                 "inputs": [
                     {"name": "loader", "file_path": "phases/data/1/outputs/loader.pt"},
                     {"name": "model", "file_path": "phases/architecture/1/outputs/model.py"},
                 ],
                 "outputs": [{"name": "checkpoint", "file_path": "phases/training/1/outputs/ckpt.pt"}]},
            ],
            "dependency_graph": {"data": [], "architecture": [], "training": ["data", "architecture"]},
        },
    })


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """Full pipeline: create spec -> run 3 phases -> complete."""
        config = Config(
            paper_path=FIXTURE_PDF,
            project_root=tmp_path,
            max_retries=2,
            max_debug_attempts=5,
        )

        dispatched_phases: list[str] = []

        async def mock_sub_agent_run(self):
            phase_id = self.sub_spec.phase.phase_id
            dispatched_phases.append(phase_id)
            return SubAgentResult(
                status=ResultStatus.success,
                phase_id=phase_id,
                summary=f"{phase_id} completed",
                outputs=[Artifact(name="out", file_path=f"outputs/{phase_id}_output")],
                test_report=TestReport(tests_run=3, tests_passed=3, tests_failed=0),
            )

        # Mock orchestrator's _query method
        async def mock_orch_query(self_agent, system, prompt):
            if "research paper analyst" in system:
                return _spec_creation_response()
            else:
                return '{"accept": true, "feedback": null}'

        with patch.object(SubAgent, "run", mock_sub_agent_run), \
             patch.object(OrchestratorAgent, "_query", mock_orch_query):
            success = await run_pipeline(config)

        assert success is True
        assert "data" in dispatched_phases
        assert "architecture" in dispatched_phases
        assert "training" in dispatched_phases

        # Training ran after data and architecture
        assert dispatched_phases.index("data") < dispatched_phases.index("training")
        assert dispatched_phases.index("architecture") < dispatched_phases.index("training")

        # Files created
        assert (tmp_path / "canonical_spec" / "spec.md").exists()
        assert (tmp_path / "canonical_spec" / "state.yaml").exists()
        assert (tmp_path / "canonical_spec" / "revision_log.yaml").exists()

        # All phases completed
        from research_builder.storage.spec_store import SpecStore
        store = SpecStore(tmp_path / "canonical_spec")
        state = store.load_state()
        for phase in state.phases:
            assert phase.status.value == "completed"

    @pytest.mark.asyncio
    async def test_pipeline_with_retry(self, tmp_path):
        """Data phase fails once, then succeeds on retry."""
        config = Config(paper_path=FIXTURE_PDF, project_root=tmp_path, max_retries=2)

        attempt_count: dict[str, int] = {}

        async def mock_sub_agent_run(self):
            phase_id = self.sub_spec.phase.phase_id
            attempt_count[phase_id] = attempt_count.get(phase_id, 0) + 1
            if phase_id == "data" and attempt_count[phase_id] == 1:
                return SubAgentResult(
                    status=ResultStatus.failure, phase_id=phase_id,
                    summary="Download timeout", attempts_used=3,
                )
            return SubAgentResult(
                status=ResultStatus.success, phase_id=phase_id,
                summary=f"{phase_id} done",
                outputs=[Artifact(name="out", file_path=f"outputs/{phase_id}_out")],
                test_report=TestReport(tests_run=1, tests_passed=1, tests_failed=0),
            )

        async def mock_orch_query(self_agent, system, prompt):
            if "research paper analyst" in system:
                return _spec_creation_response()
            return '{"accept": true, "feedback": null}'

        with patch.object(SubAgent, "run", mock_sub_agent_run), \
             patch.object(OrchestratorAgent, "_query", mock_orch_query):
            success = await run_pipeline(config)

        assert success is True
        assert attempt_count["data"] == 2
        assert (tmp_path / "phases" / "data" / "1").exists()
        assert (tmp_path / "phases" / "data" / "2").exists()

    @pytest.mark.asyncio
    async def test_pipeline_paper_not_found(self, tmp_path):
        config = Config(paper_path=Path("/nonexistent/paper.pdf"), project_root=tmp_path)
        success = await run_pipeline(config)
        assert success is False
