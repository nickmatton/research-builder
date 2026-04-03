"""Orchestrator execution loop (spec_v4 §4.2).

Runs phases in dependency order, dispatches sub-agents, handles results.
MVP: sequential execution. Parallel dispatch is a Phase 7 enhancement.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from ..models.context import RetryContext
from ..models.results import ResultStatus, SubAgentResult
from ..models.spec import EventType, PhaseStatus
from ..storage.spec_store import SpecStore
from ..storage.workspace import WorkspaceManager
from .agent import OrchestratorAgent
from .failure import FailureHandler
from .spec_manager import SpecManager
from ..sub_agent.agent import SubAgent

logger = logging.getLogger(__name__)


class ExecutionLoop:
    """Main orchestrator loop: dispatch phases, review results, handle failures."""

    def __init__(
        self,
        config: Config,
        spec_manager: SpecManager,
        workspace: WorkspaceManager,
        orchestrator_agent: OrchestratorAgent,
    ) -> None:
        self.config = config
        self.spec_manager = spec_manager
        self.workspace = workspace
        self.orchestrator_agent = orchestrator_agent
        self.failure_handler = FailureHandler(max_retries=config.max_retries)

    async def run(self) -> bool:
        """Execute all phases. Returns True if the run completed successfully."""
        logger.info("Starting execution loop")

        while True:
            # Check if all phases are done
            all_completed = all(
                p.status == PhaseStatus.completed
                for p in self.spec_manager.state.phases
            )
            if all_completed:
                self.spec_manager.log_event(EventType.run_completed, rationale="All phases completed")
                logger.info("All phases completed successfully")
                return True

            # Check for any failed phases (non-retryable)
            any_hard_failed = any(
                p.status == PhaseStatus.failed
                for p in self.spec_manager.state.phases
            )
            if any_hard_failed:
                failed_ids = [p.phase_id for p in self.spec_manager.state.phases if p.status == PhaseStatus.failed]
                self.spec_manager.log_event(
                    EventType.run_failed,
                    rationale=f"Phases exhausted all retries: {failed_ids}",
                )
                logger.error("Run failed — phases exhausted retries: %s", failed_ids)
                return False

            # Find runnable phases
            runnable = self.spec_manager.get_runnable_phases()
            if not runnable:
                # Check if anything is in progress (shouldn't happen in sequential mode)
                in_progress = [p for p in self.spec_manager.state.phases if p.status == PhaseStatus.in_progress]
                if in_progress:
                    logger.warning("No runnable phases but %d in progress — possible deadlock", len(in_progress))
                else:
                    logger.error("No runnable phases and nothing in progress — deadlock")
                    self.spec_manager.log_event(EventType.run_failed, rationale="Deadlock: no runnable phases")
                return False

            # MVP: execute sequentially
            for phase_id in runnable:
                await self._execute_phase(phase_id)
                # After each phase, re-evaluate (status may have changed)
                break  # Re-enter the while loop to recompute runnable

    async def _execute_phase(self, phase_id: str) -> None:
        """Dispatch a sub-agent for a phase and handle the result."""
        phase = self.spec_manager.state.get_phase(phase_id)
        if phase is None:
            return

        # Set up attempt directory
        try_num = self.workspace.next_try_num(phase_id)
        work_dir = self.workspace.create_attempt(phase_id, try_num)
        phase.current_try = try_num

        # Mark as in progress
        self.spec_manager.set_phase_status(phase_id, PhaseStatus.in_progress, f"Starting attempt {try_num}")

        # Build sub-spec and retry context
        paper_path = str(self.config.paper_path)
        sub_spec = self.spec_manager.extract_sub_spec(phase_id, paper_path=paper_path)

        retry_context = None
        prior_results = self.failure_handler.get_prior_results(phase_id)
        if prior_results:
            retry_context = RetryContext(prior_results=prior_results)

        # Run sub-agent
        logger.info("Dispatching sub-agent for phase=%s attempt=%d", phase_id, try_num)
        sub_agent = SubAgent(
            config=self.config,
            sub_spec=sub_spec,
            work_dir=work_dir,
            retry_context=retry_context,
        )
        result = await sub_agent.run()
        logger.info("Sub-agent returned: phase=%s status=%s", phase_id, result.status.value)

        # Record result
        self.failure_handler.record_result(phase_id, result)

        # Handle result
        await self._handle_result(phase_id, result)

    async def _handle_result(self, phase_id: str, result: SubAgentResult) -> None:
        """Process a sub-agent result: accept, retry, or fail."""
        if result.status == ResultStatus.success:
            # Run acceptance review
            try:
                accepted, feedback = await self.orchestrator_agent.acceptance_review(
                    phase_id, result, self.spec_manager,
                )
            except Exception as e:
                logger.warning("Acceptance review failed for phase=%s: %s. Auto-accepting.", phase_id, e)
                accepted, feedback = True, None
            if accepted:
                self.spec_manager.set_phase_status(
                    phase_id, PhaseStatus.completed, f"Accepted: {result.summary[:100]}",
                )
                # Update artifact paths to point to successful attempt
                phase = self.spec_manager.state.get_phase(phase_id)
                if phase:
                    phase.outputs = result.outputs
                    self.spec_manager.save()
                return
            else:
                # Rejection counts as a retry
                logger.info("Acceptance rejected for phase=%s: %s", phase_id, feedback)
                result = SubAgentResult(
                    status=ResultStatus.failure,
                    phase_id=phase_id,
                    summary=f"Acceptance rejected: {feedback}",
                    test_report=result.test_report,
                    outputs=result.outputs,
                )
                self.failure_handler.record_result(phase_id, result)

        # Handle failure
        if result.is_spec_issue:
            # Spec issues don't count against retry budget — set back to pending
            logger.info("Spec issue reported for phase=%s: %s", phase_id, result.summary)
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.pending,
                f"Spec issue: {result.summary[:100]}",
            )
            self.spec_manager.log_event(
                EventType.retry_launched,
                phase_id=phase_id,
                rationale=f"Spec issue (not counted against retry budget): {result.summary[:100]}",
            )
            return

        # Implementation failure — check retry budget
        if self.failure_handler.can_retry(phase_id):
            retries = self.failure_handler.retries_used(phase_id)
            logger.info(
                "Retrying phase=%s (attempt %d/%d): %s",
                phase_id, retries, self.config.max_retries, result.summary,
            )
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.pending,
                f"Retry {retries}/{self.config.max_retries}: {result.summary[:100]}",
            )
            self.spec_manager.log_event(
                EventType.retry_launched,
                phase_id=phase_id,
                rationale=f"Implementation failure, retry {retries}/{self.config.max_retries}",
            )
        else:
            # Exhausted retries
            logger.error("Phase %s exhausted all retries", phase_id)
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.failed,
                f"Exhausted {self.config.max_retries} retries: {result.summary[:100]}",
            )
