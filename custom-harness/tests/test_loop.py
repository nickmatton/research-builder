"""Tests for the orchestrator execution loop."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_builder.config import Config
from research_builder.models.results import ResultStatus, SubAgentResult, TestReport
from research_builder.models.spec import (
    Artifact,
    PhaseState,
    PhaseStatus,
    SpecMetadata,
    SpecState,
)
from research_builder.orchestrator.agent import OrchestratorAgent
from research_builder.orchestrator.loop import ExecutionLoop
from research_builder.orchestrator.spec_manager import SpecManager
from research_builder.storage.spec_store import SpecStore
from research_builder.storage.workspace import WorkspaceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SAMPLE_SPEC_MD = "# Spec\n\n## Data\nGet data.\n\n## Architecture\nBuild model."


def _make_two_phase_state() -> SpecState:
    """Simple 2-phase setup: data -> training (sequential)."""
    return SpecState(
        metadata=SpecMetadata(paper_id="test", paper_title="Test"),
        phases=[
            PhaseState(
                phase_id="data", title="Data",
                outputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
            ),
            PhaseState(
                phase_id="training", title="Training",
                inputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
                outputs=[Artifact(name="checkpoint", file_path="phases/training/1/outputs/ckpt.pt")],
            ),
        ],
        dependency_graph={"data": [], "training": ["data"]},
    )


def _make_parallel_state() -> SpecState:
    """3-phase setup: data + arch (parallel) -> training."""
    return SpecState(
        metadata=SpecMetadata(paper_id="test", paper_title="Test"),
        phases=[
            PhaseState(
                phase_id="data", title="Data",
                outputs=[Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt")],
            ),
            PhaseState(
                phase_id="arch", title="Architecture",
                outputs=[Artifact(name="model", file_path="phases/arch/1/outputs/model.py")],
            ),
            PhaseState(
                phase_id="training", title="Training",
                inputs=[
                    Artifact(name="loader", file_path="phases/data/1/outputs/loader.pt"),
                    Artifact(name="model", file_path="phases/arch/1/outputs/model.py"),
                ],
            ),
        ],
        dependency_graph={"data": [], "arch": [], "training": ["data", "arch"]},
    )


def _success_result(phase_id: str, **kwargs) -> SubAgentResult:
    defaults = dict(
        status=ResultStatus.success,
        phase_id=phase_id,
        outputs=[Artifact(name="out", file_path=f"phases/{phase_id}/1/outputs/out")],
        summary=f"{phase_id} done",
        test_report=TestReport(tests_run=1, tests_passed=1, tests_failed=0),
    )
    defaults.update(kwargs)
    return SubAgentResult(**defaults)


def _failure_result(phase_id: str, **kwargs) -> SubAgentResult:
    defaults = dict(
        status=ResultStatus.failure,
        phase_id=phase_id,
        summary=f"{phase_id} failed",
    )
    defaults.update(kwargs)
    return SubAgentResult(**defaults)


@pytest.fixture
def setup(tmp_path):
    """Returns (config, store, workspace) for building an execution loop."""
    config = Config(project_root=tmp_path, max_retries=2)
    store = SpecStore(config.spec_dir)
    workspace = WorkspaceManager(config)
    workspace.initialize()
    return config, store, workspace


def _build_loop(setup, state, sub_agent_results, acceptance_results=None, verify_results=None):
    """Build an ExecutionLoop with mocked sub-agent, acceptance review, and section verifier.

    sub_agent_results: list of SubAgentResult, popped in order for each sub-agent dispatch.
    acceptance_results: list of (accepted, feedback), popped in order. Defaults to all accepted.
    verify_results: list of (accepted, feedback), popped in order. Defaults to all accepted.
        The section verifier now gates retries (acceptance_review is no longer called by
        the loop), so set this when you want to drive a rejection-triggered retry.
    """
    config, store, workspace = setup

    store.save_spec_md(SAMPLE_SPEC_MD)
    store.save_state(state)
    spec_manager = SpecManager(store, state)
    orch_agent = OrchestratorAgent(config)

    loop = ExecutionLoop(config, spec_manager, workspace, orch_agent)

    # Mock sub-agent runs
    results_iter = iter(sub_agent_results)

    async def mock_sub_agent_run(self):
        return next(results_iter)

    # Mock acceptance reviews
    if acceptance_results is None:
        acceptance_results = [(True, None)] * len(sub_agent_results)
    accept_iter = iter(acceptance_results)

    async def mock_acceptance(phase_id, result, spec_mgr):
        return next(accept_iter)

    # Mock per-section 4-agent chain (refiner/researcher/verifier) so tests
    # don't try to make real API calls. Each returns a minimal payload that
    # mirrors the live shape.
    async def mock_refine(phase_id, spec_mgr, paper_path):
        return {"refined_spec_md": "", "summary": "", "research_questions": []}

    async def mock_research(phase_id, questions, spec_mgr, paper_path):
        return {"research_notes_md": "", "sources": [], "summary": ""}

    if verify_results is None:
        verify_results = [(True, None)] * len(sub_agent_results)
    verify_iter = iter(verify_results)

    async def mock_verify(phase_id, builder_result, spec_mgr, work_dir=None):
        accepted, feedback = next(verify_iter)
        return accepted, feedback, {"deterministic_checks": [], "llm_judge": None}

    return loop, mock_sub_agent_run, mock_acceptance, mock_refine, mock_research, mock_verify


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_two_phases_succeed(self, setup):
        state = _make_two_phase_state()
        results = [_success_result("data"), _success_result("training")]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(setup, state, results)

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is True
        assert loop.spec_manager.state.get_phase("data").status == PhaseStatus.completed
        assert loop.spec_manager.state.get_phase("training").status == PhaseStatus.completed

    @pytest.mark.asyncio
    async def test_three_phases_sequential(self, setup):
        """data + arch (both runnable initially) -> training. MVP runs sequentially."""
        state = _make_parallel_state()
        results = [
            _success_result("data"),
            _success_result("arch"),
            _success_result("training"),
        ]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(setup, state, results)

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is True
        for pid in ["data", "arch", "training"]:
            assert loop.spec_manager.state.get_phase(pid).status == PhaseStatus.completed


class TestRetries:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self, setup):
        """Phase fails once, retries, then succeeds."""
        state = _make_two_phase_state()
        results = [
            _failure_result("data"),        # first attempt fails
            _success_result("data"),        # retry succeeds
            _success_result("training"),
        ]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(
            setup, state, results,
            acceptance_results=[
                (True, None),  # skip (failure doesn't get acceptance)
                (True, None),  # data retry accepted
                (True, None),  # training accepted
            ],
        )

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is True
        assert loop.failure_handler.retries_used("data") == 1

    @pytest.mark.asyncio
    async def test_exhaust_retries(self, setup):
        """Phase fails max_retries times -> run fails."""
        state = _make_two_phase_state()
        # config has max_retries=2, so 2 failures exhaust it
        results = [
            _failure_result("data"),
            _failure_result("data"),
        ]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(setup, state, results)

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is False
        assert loop.spec_manager.state.get_phase("data").status == PhaseStatus.failed


class TestSpecIssue:
    @pytest.mark.asyncio
    async def test_spec_issue_does_not_count(self, setup):
        """Spec-issue returns don't count against retry budget."""
        state = _make_two_phase_state()
        results = [
            _failure_result("data", is_spec_issue=True),   # spec issue — no penalty
            _failure_result("data", is_spec_issue=True),   # another spec issue
            _success_result("data"),                        # finally succeeds
            _success_result("training"),
        ]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(
            setup, state, results,
            acceptance_results=[(True, None)] * 4,
        )

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is True
        assert loop.failure_handler.retries_used("data") == 0  # spec issues not counted


class TestAcceptanceRejection:
    @pytest.mark.asyncio
    async def test_rejection_triggers_retry(self, setup):
        """Section-verifier rejection sends phase back for retry."""
        state = _make_two_phase_state()
        results = [
            _success_result("data"),        # sub-agent says success...
            _success_result("data"),        # retry succeeds
            _success_result("training"),
        ]
        loop, mock_run, mock_accept, mock_refine, mock_research, mock_verify = _build_loop(
            setup, state, results,
            verify_results=[
                (False, "Missing validation"),  # ...but verifier rejects
                (True, None),                   # retry accepted
                (True, None),                   # training accepted
            ],
        )

        from research_builder.sub_agent.agent import SubAgent
        with patch.object(SubAgent, "run", mock_run), \
             patch.object(loop.orchestrator_agent, "acceptance_review", mock_accept), \
             patch.object(loop.orchestrator_agent, "refine_section", mock_refine), \
             patch.object(loop.orchestrator_agent, "research_for_section", mock_research), \
             patch.object(loop.orchestrator_agent, "verify_section", mock_verify):
            success = await loop.run()

        assert success is True
        # The rejection counts as a failure in the retry handler
        assert loop.failure_handler.retries_used("data") == 1
