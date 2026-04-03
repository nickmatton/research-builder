"""End-to-end integration test: full pipeline with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_builder.config import Config
from research_builder.llm.client import LLMClient
from research_builder.main import run_pipeline
from research_builder.models.results import ResultStatus, SubAgentResult, TestReport
from research_builder.models.spec import Artifact
from research_builder.sub_agent.agent import SubAgent

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_paper.pdf"


def _mock_llm_response(text: str):
    """Create a mock LLM Message response with text content."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def _spec_creation_response() -> str:
    """JSON response for spec creation."""
    return json.dumps({
        "spec_md": (
            "# Canonical Spec: Widget Performance Study\n\n"
            "## Global Context\n\n"
            "This paper studies widget performance.\n\n"
            "## Phase: Data\n\n"
            "Download widget dataset. 10K training, 2K test samples.\n\n"
            "### Acceptance Criteria\n- 10,000 training samples\n- 2,000 test samples\n\n"
            "## Phase: Architecture\n\n"
            "Build a 3-layer MLP with ~50K parameters.\n\n"
            "### Acceptance Criteria\n- Parameter count ~50K\n- Output shape (batch, 10)\n"
        ),
        "state": {
            "metadata": {
                "paper_id": "test:widget",
                "paper_title": "Widget Performance Study",
            },
            "phases": [
                {
                    "phase_id": "data",
                    "title": "Data",
                    "inputs": [],
                    "outputs": [{"name": "loader", "file_path": "phases/data/1/outputs/loader.pt"}],
                },
                {
                    "phase_id": "architecture",
                    "title": "Architecture",
                    "inputs": [],
                    "outputs": [{"name": "model", "file_path": "phases/architecture/1/outputs/model.py"}],
                },
                {
                    "phase_id": "training",
                    "title": "Training",
                    "inputs": [
                        {"name": "loader", "file_path": "phases/data/1/outputs/loader.pt"},
                        {"name": "model", "file_path": "phases/architecture/1/outputs/model.py"},
                    ],
                    "outputs": [{"name": "checkpoint", "file_path": "phases/training/1/outputs/ckpt.pt"}],
                },
            ],
            "dependency_graph": {
                "data": [],
                "architecture": [],
                "training": ["data", "architecture"],
            },
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

        # Track which phases got dispatched
        dispatched_phases: list[str] = []

        # Mock sub-agent: always succeeds
        original_run = SubAgent.run

        async def mock_sub_agent_run(self):
            phase_id = self.sub_spec.phase.phase_id
            dispatched_phases.append(phase_id)
            return SubAgentResult(
                status=ResultStatus.success,
                phase_id=phase_id,
                summary=f"{phase_id} completed successfully",
                outputs=[Artifact(name="out", file_path=f"outputs/{phase_id}_output")],
                test_report=TestReport(tests_run=3, tests_passed=3, tests_failed=0),
                attempts_used=1,
            )

        # Mock LLM responses
        call_count = {"n": 0}

        async def mock_create_message(self_client, **kwargs):
            call_count["n"] += 1
            system = kwargs.get("system", "")

            if "research paper analyst" in system:
                # Spec creation call
                return _mock_llm_response(_spec_creation_response())
            elif "acceptance review" in system.lower() or "reviewing" in system.lower():
                # Acceptance review call
                return _mock_llm_response('{"accept": true, "feedback": null}')
            else:
                return _mock_llm_response('{"accept": true}')

        with patch.object(SubAgent, "run", mock_sub_agent_run), \
             patch.object(LLMClient, "create_message", mock_create_message):
            success = await run_pipeline(config)

        # Verify success
        assert success is True

        # Verify all 3 phases were dispatched
        assert "data" in dispatched_phases
        assert "architecture" in dispatched_phases
        assert "training" in dispatched_phases

        # Verify training ran after data and architecture
        training_idx = dispatched_phases.index("training")
        assert dispatched_phases.index("data") < training_idx
        assert dispatched_phases.index("architecture") < training_idx

        # Verify files were created
        assert (tmp_path / "canonical_spec" / "spec.md").exists()
        assert (tmp_path / "canonical_spec" / "state.yaml").exists()
        assert (tmp_path / "canonical_spec" / "revision_log.yaml").exists()

        # Verify spec.md content
        spec_md = (tmp_path / "canonical_spec" / "spec.md").read_text()
        assert "Widget Performance Study" in spec_md

        # Verify state has all phases completed
        from research_builder.storage.spec_store import SpecStore
        store = SpecStore(tmp_path / "canonical_spec")
        state = store.load_state()
        for phase in state.phases:
            assert phase.status.value == "completed", f"Phase {phase.phase_id} not completed"

        # Verify revision log has events
        revisions = store.load_revision_log()
        event_types = [r.event_type.value for r in revisions]
        assert "spec_created" in event_types
        assert "phase_completed" in event_types

    @pytest.mark.asyncio
    async def test_pipeline_with_retry(self, tmp_path):
        """Pipeline where data phase fails once, then succeeds on retry."""
        config = Config(
            paper_path=FIXTURE_PDF,
            project_root=tmp_path,
            max_retries=2,
        )

        attempt_count: dict[str, int] = {}

        async def mock_sub_agent_run(self):
            phase_id = self.sub_spec.phase.phase_id
            attempt_count[phase_id] = attempt_count.get(phase_id, 0) + 1

            if phase_id == "data" and attempt_count[phase_id] == 1:
                # First data attempt fails
                return SubAgentResult(
                    status=ResultStatus.failure,
                    phase_id=phase_id,
                    summary="Download timeout",
                    attempts_used=3,
                )
            else:
                return SubAgentResult(
                    status=ResultStatus.success,
                    phase_id=phase_id,
                    summary=f"{phase_id} done",
                    outputs=[Artifact(name="out", file_path=f"outputs/{phase_id}_out")],
                    test_report=TestReport(tests_run=1, tests_passed=1, tests_failed=0),
                    attempts_used=1,
                )

        async def mock_create_message(self_client, **kwargs):
            system = kwargs.get("system", "")
            if "research paper analyst" in system:
                return _mock_llm_response(_spec_creation_response())
            else:
                return _mock_llm_response('{"accept": true, "feedback": null}')

        with patch.object(SubAgent, "run", mock_sub_agent_run), \
             patch.object(LLMClient, "create_message", mock_create_message):
            success = await run_pipeline(config)

        assert success is True
        assert attempt_count["data"] == 2  # failed once, then succeeded

        # Verify both attempt directories exist
        assert (tmp_path / "phases" / "data" / "1").exists()
        assert (tmp_path / "phases" / "data" / "2").exists()

    @pytest.mark.asyncio
    async def test_pipeline_paper_not_found(self, tmp_path):
        """Pipeline fails gracefully when paper doesn't exist."""
        config = Config(
            paper_path=Path("/nonexistent/paper.pdf"),
            project_root=tmp_path,
        )
        success = await run_pipeline(config)
        assert success is False
