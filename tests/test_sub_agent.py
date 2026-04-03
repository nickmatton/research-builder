"""Tests for sub-agent prompt construction, result parsing, and agent behavior."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from research_builder.config import Config
from research_builder.llm.client import LLMClient, ToolExit
from research_builder.models.context import AdjacentPhaseSummary, RetryContext, SubSpec
from research_builder.models.results import ResultStatus, SubAgentResult, TestReport, TestResult, TestStatus
from research_builder.models.spec import Artifact, PhaseState
from research_builder.sub_agent.agent import SubAgent, _parse_tool_exit
from research_builder.sub_agent.prompts import (
    BASE_SYSTEM_PROMPT,
    PHASE_GUIDANCE,
    build_system_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sub_spec(**overrides) -> SubSpec:
    defaults = dict(
        phase=PhaseState(
            phase_id="architecture",
            title="Architecture Phase",
            max_debug_attempts=10,
            inputs=[],
            outputs=[Artifact(name="model_code", file_path="outputs/model.py")],
        ),
        spec_markdown="## Architecture\n\nImplement a 3-layer CNN with 50K params.",
        adjacent_phases=[
            AdjacentPhaseSummary(
                phase_id="training",
                title="Training",
                inputs=[Artifact(name="model_code", file_path="outputs/model.py")],
            ),
        ],
        open_questions=["What activation function is used after each conv layer?"],
        paper_path="/paper/paper.pdf",
    )
    defaults.update(overrides)
    return SubSpec(**defaults)


def _make_retry_context() -> RetryContext:
    return RetryContext(
        prior_results=[
            SubAgentResult(
                status=ResultStatus.failure,
                phase_id="architecture",
                summary="Parameter count was 75K instead of 50K",
                test_report=TestReport(
                    tests_run=3,
                    tests_passed=2,
                    tests_failed=1,
                    test_details=[
                        TestResult(test_name="test_shapes", status=TestStatus.passed, description="Output shapes"),
                        TestResult(test_name="test_forward", status=TestStatus.passed, description="Forward pass"),
                        TestResult(
                            test_name="test_param_count",
                            status=TestStatus.failed,
                            description="Param count",
                            message="Expected ~50K, got 75,312",
                        ),
                    ],
                ),
                diagnostics={"param_count": 75312},
            ),
        ],
        orchestrator_feedback="Reduce channel widths to hit the 50K target. Check Table 1 in the paper for exact layer sizes.",
    )


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_contains_base_prompt(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "research paper reproduction agent" in prompt
        assert "report_result" in prompt

    def test_contains_phase_guidance(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "Phase: Architecture" in prompt
        assert "Parameter count matches spec" in prompt

    def test_all_standard_phases_have_guidance(self):
        for phase_id in ["data", "architecture", "training", "eval", "results"]:
            assert phase_id in PHASE_GUIDANCE

    def test_contains_debug_budget(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "10" in prompt
        assert "debug attempts" in prompt.lower()

    def test_contains_spec_markdown(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "3-layer CNN" in prompt
        assert "50K params" in prompt

    def test_contains_outputs(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "model_code" in prompt
        assert "outputs/model.py" in prompt

    def test_contains_adjacent_phases(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "Training" in prompt
        assert "training" in prompt

    def test_contains_open_questions(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "activation function" in prompt

    def test_contains_paper_path(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "/paper/paper.pdf" in prompt
        assert "read_paper_section" in prompt

    def test_no_retry_context_when_none(self):
        prompt = build_system_prompt(_make_sub_spec())
        assert "Retry Context" not in prompt

    def test_contains_retry_context(self):
        prompt = build_system_prompt(_make_sub_spec(), _make_retry_context())
        assert "Retry Context" in prompt
        assert "retry" in prompt.lower()
        assert "75,312" in prompt or "75312" in prompt
        assert "Reduce channel widths" in prompt

    def test_retry_shows_prior_test_results(self):
        prompt = build_system_prompt(_make_sub_spec(), _make_retry_context())
        assert "test_param_count" in prompt
        assert "FAIL" in prompt
        assert "test_shapes" in prompt
        assert "PASS" in prompt

    def test_unknown_phase_gets_fallback(self):
        sub_spec = _make_sub_spec(
            phase=PhaseState(phase_id="custom_phase", title="Custom Phase"),
        )
        prompt = build_system_prompt(sub_spec)
        assert "Custom Phase" in prompt
        assert "No specific guidance" in prompt


# ---------------------------------------------------------------------------
# Result parsing tests
# ---------------------------------------------------------------------------


class TestParseToolExit:
    def test_success_result(self):
        raw = {
            "status": "success",
            "summary": "Implemented 3-layer CNN",
            "outputs": [
                {"name": "model_code", "file_path": "outputs/model.py"},
                {"name": "test_code", "file_path": "src/test_model.py"},
            ],
            "test_report": {
                "tests_run": 3,
                "tests_passed": 3,
                "tests_failed": 0,
                "test_details": [
                    {"test_name": "test_shapes", "status": "passed", "description": "Output shapes match"},
                    {"test_name": "test_params", "status": "passed", "description": "Param count ~50K"},
                    {"test_name": "test_grads", "status": "passed", "description": "Gradients flow"},
                ],
            },
            "attempts_used": 2,
        }
        result = _parse_tool_exit("architecture", raw)
        assert result.status == ResultStatus.success
        assert result.phase_id == "architecture"
        assert len(result.outputs) == 2
        assert result.test_report.tests_passed == 3
        assert result.attempts_used == 2
        assert result.is_spec_issue is False

    def test_failure_with_spec_issue(self):
        raw = {
            "status": "failure",
            "summary": "Dataset URL returns 404",
            "is_spec_issue": True,
            "diagnostics": {"url": "https://example.com/data.tar.gz", "http_status": 404},
        }
        result = _parse_tool_exit("data", raw)
        assert result.status == ResultStatus.failure
        assert result.is_spec_issue is True
        assert result.diagnostics["http_status"] == 404

    def test_minimal_result(self):
        raw = {"status": "success", "summary": "done"}
        result = _parse_tool_exit("eval", raw)
        assert result.status == ResultStatus.success
        assert result.outputs == []
        assert result.test_report.tests_run == 0

    def test_defaults_for_missing_fields(self):
        raw = {"summary": "something"}
        result = _parse_tool_exit("data", raw)
        assert result.status == ResultStatus.failure  # missing status defaults to failure
        assert result.attempts_used == 1

    def test_test_report_auto_counts(self):
        """If tests_run/passed/failed are omitted, they're inferred from details."""
        raw = {
            "status": "failure",
            "summary": "test failed",
            "test_report": {
                "test_details": [
                    {"test_name": "a", "status": "passed"},
                    {"test_name": "b", "status": "failed", "message": "wrong output"},
                ],
            },
        }
        result = _parse_tool_exit("arch", raw)
        assert result.test_report.tests_run == 2
        assert result.test_report.tests_passed == 1
        assert result.test_report.tests_failed == 1


# ---------------------------------------------------------------------------
# SubAgent integration tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestSubAgentRun:
    @pytest.mark.asyncio
    async def test_successful_run(self, tmp_path):
        """SubAgent returns success when report_result tool exits cleanly."""
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)

        sub_spec = _make_sub_spec(paper_path="")
        work_dir = tmp_path / "phases" / "arch" / "1"
        work_dir.mkdir(parents=True)
        (work_dir / "src").mkdir()
        (work_dir / "outputs").mkdir()

        agent = SubAgent(config, llm_client, sub_spec, work_dir)

        # Mock run_tool_loop to raise ToolExit (simulating the model calling report_result)
        async def mock_tool_loop(**kwargs):
            raise ToolExit(result={
                "status": "success",
                "summary": "Built the CNN",
                "outputs": [{"name": "model", "file_path": "outputs/model.py"}],
                "test_report": {"tests_run": 2, "tests_passed": 2, "tests_failed": 0},
                "attempts_used": 1,
            })

        with patch.object(llm_client, "run_tool_loop", side_effect=mock_tool_loop):
            result = await agent.run()

        assert result.status == ResultStatus.success
        assert result.phase_id == "architecture"
        assert len(result.outputs) == 1
        assert result.summary == "Built the CNN"

    @pytest.mark.asyncio
    async def test_failure_run(self, tmp_path):
        """SubAgent returns failure when report_result is called with failure."""
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)

        sub_spec = _make_sub_spec(paper_path="")
        work_dir = tmp_path / "phases" / "arch" / "1"
        work_dir.mkdir(parents=True)
        (work_dir / "src").mkdir()
        (work_dir / "outputs").mkdir()

        agent = SubAgent(config, llm_client, sub_spec, work_dir)

        async def mock_tool_loop(**kwargs):
            raise ToolExit(result={
                "status": "failure",
                "summary": "Could not match param count",
                "is_spec_issue": True,
                "diagnostics": {"expected": 50000, "actual": 75000},
            })

        with patch.object(llm_client, "run_tool_loop", side_effect=mock_tool_loop):
            result = await agent.run()

        assert result.status == ResultStatus.failure
        assert result.is_spec_issue is True

    @pytest.mark.asyncio
    async def test_crash_recovery(self, tmp_path):
        """SubAgent returns failure with diagnostics if an unexpected error occurs."""
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)

        sub_spec = _make_sub_spec(paper_path="")
        work_dir = tmp_path / "phases" / "arch" / "1"
        work_dir.mkdir(parents=True)
        (work_dir / "src").mkdir()
        (work_dir / "outputs").mkdir()

        agent = SubAgent(config, llm_client, sub_spec, work_dir)

        async def mock_tool_loop(**kwargs):
            raise RuntimeError("API connection failed")

        with patch.object(llm_client, "run_tool_loop", side_effect=mock_tool_loop):
            result = await agent.run()

        assert result.status == ResultStatus.failure
        assert "crashed" in result.summary
        assert result.diagnostics["type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_report_result_called(self, tmp_path):
        """SubAgent returns failure if model stops without calling report_result."""
        config = Config(project_root=tmp_path)
        llm_client = LLMClient(config)

        sub_spec = _make_sub_spec(paper_path="")
        work_dir = tmp_path / "phases" / "arch" / "1"
        work_dir.mkdir(parents=True)
        (work_dir / "src").mkdir()
        (work_dir / "outputs").mkdir()

        agent = SubAgent(config, llm_client, sub_spec, work_dir)

        # Mock: model just stops (end_turn) without calling report_result
        mock_response = AsyncMock()
        mock_response.stop_reason = "end_turn"

        async def mock_tool_loop(**kwargs):
            return (mock_response, [])

        with patch.object(llm_client, "run_tool_loop", side_effect=mock_tool_loop):
            result = await agent.run()

        assert result.status == ResultStatus.failure
        assert "without reporting result" in result.summary
