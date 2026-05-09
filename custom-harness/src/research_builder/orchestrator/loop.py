"""Orchestrator execution loop (spec_v4 §4.2).

Runs phases in dependency order, dispatches sub-agents, handles results.
MVP: sequential execution. Parallel dispatch is a Phase 7 enhancement.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import click

from .. import ui
from ..commands import get_inbox
from ..config import Config
from ..events import get_emitter
from ..interaction import UserAction, open_in_editor, prompt_skip_which_phase

if TYPE_CHECKING:
    from ..console import InteractiveConsole
    from ..cloud import CloudProvisioner
from ..models.context import PostMortem, RetryContext
from ..models.claims import ClaimsReport
from ..models.results import ResultStatus, SubAgentResult
from ..models.spec import EventType, FileStatus, PhaseStatus
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
        on_phase_complete: Callable[[str, SubAgentResult], UserAction | asyncio.coroutines] | None = None,
        console: InteractiveConsole | None = None,
        cloud_provisioner: "CloudProvisioner | None" = None,
    ) -> None:
        self.config = config
        self.spec_manager = spec_manager
        self.workspace = workspace
        self.orchestrator_agent = orchestrator_agent
        self.failure_handler = FailureHandler(max_retries=config.max_retries)
        async def _default_phase_complete(pid, r):
            return UserAction.CONTINUE
        self.on_phase_complete = on_phase_complete or _default_phase_complete
        self.console = console
        self.cloud_provisioner = cloud_provisioner
        self.total_cost_usd: float = 0.0
        self.emitter = get_emitter()
        self._files_announced: set[str] = set()
        # Announce initial planned files + DAG so a viewer attached at startup
        # has a complete picture even before the first phase runs.
        if self.emitter:
            self._emit_initial_plan()

    def _emit_initial_plan(self) -> None:
        """Emit file_planned for every file in the plan + an initial dag_updated."""
        plan = self.spec_manager.state.plan
        if plan is not None:
            for f in plan.files:
                if f.file_id in self._files_announced:
                    continue
                self._files_announced.add(f.file_id)
                self.emitter.emit(
                    "file_planned",
                    agent_id=f"phase:{f.owning_phase}",
                    parent_id="orchestrator",
                    file_id=f.file_id,
                    path=f.rel_path,
                    role=f.role.value,
                    status=f.status.value,
                )
        self._emit_dag_snapshot()

    def _emit_dag_snapshot(self) -> None:
        """Emit a full DAG snapshot reflecting current per-phase status."""
        if not self.emitter:
            return
        plan = self.spec_manager.state.plan
        nodes = []
        for p in self.spec_manager.state.phases:
            node = plan.get_node(p.phase_id) if plan else None
            nodes.append({
                "id": f"phase:{p.phase_id}",
                "phase_id": p.phase_id,
                "title": p.title,
                "status": p.status.value,
                "sub_steps": list(node.sub_steps) if node else [],
                "parent_id": "orchestrator",
            })
        edges = [
            {"from": f"phase:{dep}", "to": f"phase:{pid}"}
            for pid, deps in self.spec_manager.state.dependency_graph.items()
            for dep in deps
        ]
        self.emitter.emit(
            "dag_updated",
            agent_id="orchestrator",
            parent_id=None,
            nodes=nodes,
            edges=edges,
        )

    async def _surface_gpu_estimate(self) -> None:
        """Walk every phase through the GPU classifier once and report the projected spend.

        If the projected total exceeds the per-run cap, the provisioner's
        approval callback gets invoked up-front so the operator can either
        raise the cap before any phase starts or accept that some phases will
        be denied GPUs at provision time.
        """
        provisioner = self.cloud_provisioner
        if provisioner is None:
            return
        sub_specs = []
        paper_path = str(self.config.paper_path)
        for phase in self.spec_manager.state.phases:
            try:
                sub_specs.append(self.spec_manager.extract_sub_spec(phase.phase_id, paper_path=paper_path))
            except Exception:
                logger.exception("Failed to extract sub_spec for phase=%s during GPU estimate", phase.phase_id)
        # Load the full canonical spec so the classifier can cross-reference
        # the architecture section when judging eval/inference phases — without
        # this, eval phases whose per-phase slice only says "compute metric X"
        # under-classify as CPU-only.
        try:
            full_spec_md = self.spec_manager.store.load_spec_md()
        except Exception:
            logger.exception("Failed to load full spec.md; GPU classifier will miss cross-phase context")
            full_spec_md = None
        try:
            estimate = await provisioner.estimate_run(sub_specs, full_spec_markdown=full_spec_md)
        except Exception:
            logger.exception("CloudProvisioner.estimate_run failed; skipping upfront estimate")
            return

        gpu_phases = estimate.gpu_phases()
        if not gpu_phases:
            ui.info("GPU estimate: no phases require a GPU.")
            return

        ui.header("GPU spend estimate")
        for p in gpu_phases:
            ui.info(
                f"  {p.phase_id}: {p.instance_type} "
                f"(~${p.hourly_rate_usd:.2f}/hr × {p.estimated_hours:.1f}hrs = ${p.estimated_cost_usd:.2f})"
            )
        ui.info(f"Total projected: ${estimate.total_usd:.2f} | Cap: ${estimate.cap_usd:.2f}")

        if estimate.total_usd > estimate.cap_usd:
            ui.warning(
                f"Projected GPU spend ${estimate.total_usd:.2f} exceeds cap ${estimate.cap_usd:.2f}. "
                "The harness will ask you to approve raising the cap when the first over-budget "
                "provision is requested. You can also approve up-front by passing --gpu-budget."
            )

    def _project_run_complete(self, success: bool) -> None:
        """Append a final summary row to <project_root>/notes/journal.md."""
        try:
            import time
            from ..storage.paper_repo import project_run_complete
            duration = time.time() - getattr(self, "_run_started_at", time.time())
            project_run_complete(
                self.config,
                run_id=getattr(self, "_run_id", "harness-unknown"),
                state=self.spec_manager.state,
                total_cost_usd=self.total_cost_usd,
                duration_seconds=duration,
            )
        except Exception:
            logger.exception("Failed to project run-complete journal row; continuing")

    def _project_phase_outcome(self, phase_id: str, result: SubAgentResult, duration_seconds: float | None) -> None:
        """Append a per-phase row to <project_root>/notes/journal.md."""
        try:
            from ..storage.paper_repo import project_phase_complete
            project_phase_complete(
                self.config,
                phase_id=phase_id,
                result=result,
                duration_seconds=duration_seconds,
                run_id=getattr(self, "_run_id", None),
            )
        except Exception:
            logger.exception("Failed to project phase-complete journal row for %s; continuing", phase_id)

    def _emit_orchestrator_completed(self, status: str, summary: str) -> None:
        """Mark the orchestrator agent as finished in the event stream.

        The orchestrator stays in_progress for the whole run — this fires
        only when the execution loop is about to return.
        """
        if not self.emitter:
            return
        self.emitter.emit(
            "agent_completed",
            agent_id="orchestrator",
            parent_id=None,
            status=status,
            summary=summary,
        )

    async def run(self) -> bool:
        """Execute all phases. Returns True if the run completed successfully."""
        import time
        logger.info("Starting execution loop")
        self._run_id = f"harness-{time.strftime('%Y%m%d-%H%M%S')}"
        self._run_started_at = time.time()

        # Upfront GPU spend estimate (only if a cloud provisioner is wired).
        # Walks every phase through the classifier once and surfaces the
        # projected total. Cached decisions are reused at provision time.
        if self.cloud_provisioner is not None:
            await self._surface_gpu_estimate()

        while True:
            # Check if all phases are done
            all_completed = all(
                p.status == PhaseStatus.completed
                for p in self.spec_manager.state.phases
            )
            if all_completed:
                self.spec_manager.log_event(EventType.run_completed, rationale="All phases completed")
                logger.info("All phases completed successfully")
                self._generate_claims_summary()
                self._project_run_complete(success=True)
                self._emit_orchestrator_completed("completed", "All phases completed")
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
                self._project_run_complete(success=False)
                self._emit_orchestrator_completed("failed", f"Phases exhausted retries: {failed_ids}")
                return False

            # Find runnable phases
            runnable = self.spec_manager.get_runnable_phases()
            if not runnable:
                in_progress = [p for p in self.spec_manager.state.phases if p.status == PhaseStatus.in_progress]
                if in_progress:
                    logger.warning("No runnable phases but %d in progress — possible deadlock", len(in_progress))
                else:
                    logger.error("No runnable phases and nothing in progress — deadlock")
                    self.spec_manager.log_event(EventType.run_failed, rationale="Deadlock: no runnable phases")
                    self._emit_orchestrator_completed("failed", "Deadlock: no runnable phases")
                return False

            # MVP: execute sequentially
            for phase_id in runnable:
                await self._execute_phase(phase_id)
                break  # Re-enter the while loop to recompute runnable

    async def _execute_phase(self, phase_id: str) -> None:
        """Dispatch a sub-agent for a phase and handle the result."""
        phase = self.spec_manager.state.get_phase(phase_id)
        if phase is None:
            return

        # Set up phase directory (single directory per phase, retries work in-place)
        work_dir = self.workspace.create_phase_dir(phase_id)
        self._current_work_dir = work_dir
        retry_num = self.failure_handler.retries_used(phase_id)

        # Mark as in progress
        self.spec_manager.set_phase_status(phase_id, PhaseStatus.in_progress, f"Starting (retry {retry_num})" if retry_num else "Starting")
        self._update_plan_for_phase(phase_id, PhaseStatus.in_progress)

        # Build sub-spec and retry context
        paper_path = str(self.config.paper_path)
        sub_spec = self.spec_manager.extract_sub_spec(phase_id, paper_path=paper_path)

        retry_context = None
        prior_results = self.failure_handler.get_prior_results(phase_id)
        if prior_results:
            retry_context = RetryContext(
                prior_results=prior_results,
                post_mortem=self.failure_handler.get_post_mortem(phase_id),
            )

        # Run sub-agent (with interactive console if available)
        logger.info("Dispatching sub-agent for phase=%s retry=%d", phase_id, retry_num)
        total_phases = len(self.spec_manager.state.phases)
        completed = sum(1 for p in self.spec_manager.state.phases if p.status == PhaseStatus.completed)
        phase_num = completed + 1
        ui.step(f"Phase '{phase_id}' [{phase_num}/{total_phases}]" + (f" — retry {retry_num}" if retry_num else ""))

        # Emit agent_started for the viewer (subagent kind, parented to orchestrator).
        if self.emitter:
            self.emitter.emit(
                "agent_started",
                agent_id=f"phase:{phase_id}",
                parent_id="orchestrator",
                kind="subagent",
                title=phase.title or phase_id,
            )
            self._emit_dag_snapshot()

        emitter = self.emitter
        def _on_activity(kind: str, detail: str) -> None:
            if emitter:
                if kind == "thinking":
                    emitter.emit(
                        "agent_thinking",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        text=detail,
                    )
                elif kind == "tool":
                    emitter.emit(
                        "agent_tool",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        summary=detail,
                    )
                elif kind == "done":
                    emitter.emit(
                        "agent_tool",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        summary=detail,
                    )
            if self.console:
                self.console.print_activity(phase_id, kind, detail)
            elif kind == "done":
                ui.clear_status_line()
                ui.activity_done(phase_id, detail)
            else:
                ui.status_line(phase_id, kind, detail)

        # Drain any chat messages the operator queued for this phase via the
        # agent-terminal viewer (commands.jsonl → Inbox). They get folded into
        # the SubAgent's kickoff prompt as additional user instructions.
        queued_msgs = get_inbox().drain(f"phase:{phase_id}")
        if queued_msgs and self.emitter:
            for m in queued_msgs:
                self.emitter.emit(
                    "agent_message",
                    agent_id=f"phase:{phase_id}",
                    parent_id="orchestrator",
                    role="user",
                    text=m[:2000],
                )

        # Optional: provision a cloud GPU machine for this phase if the
        # provisioner's LLM gate decides this phase needs one. The sub-agent
        # picks up the connection details via work_dir/.cloud + remote_run.sh.
        compute_handle = None
        if self.cloud_provisioner is not None:
            try:
                # Pass the full spec so the classifier has cross-phase context
                # on cache miss (e.g. post spec-amendment). The normal flow
                # hits the cache populated by _surface_gpu_estimate.
                try:
                    full_spec_md = self.spec_manager.store.load_spec_md()
                except Exception:
                    full_spec_md = None
                compute_handle = await self.cloud_provisioner.provision(
                    sub_spec, work_dir, full_spec_markdown=full_spec_md,
                )
                if compute_handle is not None:
                    ui.info(
                        f"    Provisioned Lambda {compute_handle.machine.instance_type} "
                        f"({compute_handle.machine.public_ip}) for phase '{phase_id}'"
                    )
                    if emitter:
                        emitter.emit(
                            "compute_provisioned",
                            agent_id=f"phase:{phase_id}",
                            parent_id="orchestrator",
                            instance_id=compute_handle.machine.id,
                            instance_type=compute_handle.machine.instance_type,
                            public_ip=compute_handle.machine.public_ip,
                        )
            except Exception:
                logger.exception("Cloud provisioning failed for phase=%s; continuing locally", phase_id)
                compute_handle = None

        sub_agent = SubAgent(
            config=self.config,
            sub_spec=sub_spec,
            work_dir=work_dir,
            retry_context=retry_context,
            on_activity=_on_activity,
            extra_user_messages=queued_msgs,
            cloud_provisioner=self.cloud_provisioner,
            compute_handle=compute_handle,
        )

        # Start interactive console alongside agent execution
        console_task = None
        if self.console:
            console_task = asyncio.create_task(self.console.run())

        try:
            result = await sub_agent.run()
            self.total_cost_usd += result.cost_usd
            if result.cost_usd > 0:
                ui.info(f"    Phase cost: ${result.cost_usd:.2f} | Total: ${self.total_cost_usd:.2f}")

            # Log result details
            logger.info(
                "Sub-agent returned: phase=%s status=%s attempts=%d summary=%s",
                phase_id, result.status.value, result.attempts_used, result.summary[:200],
            )
            if result.test_report.tests_run > 0:
                tr = result.test_report
                logger.info(
                    "  Tests: %d/%d passed, %d failed",
                    tr.tests_passed, tr.tests_run, tr.tests_failed,
                )
                for t in tr.test_details:
                    if t.status.value != "passed":
                        logger.warning("  FAIL %s: %s", t.test_name, t.message or t.description)
            if result.outputs:
                logger.info("  Outputs: %s", ", ".join(f"{o.name} ({o.file_path})" for o in result.outputs))
            if result.is_spec_issue:
                logger.warning("  Flagged as SPEC ISSUE")
            if result.diagnostics:
                logger.info("  Diagnostics: %s", result.diagnostics)

            # Record result
            self.failure_handler.record_result(phase_id, result)

            # Handle result (includes acceptance review)
            await self._handle_result(phase_id, result)

            # Append per-phase row to <project_root>/notes/journal.md so the
            # Claude Code skill workflow can see the harness's progress.
            self._project_phase_outcome(phase_id, result, duration_seconds=None)
        finally:
            # Signal console to stop, then wait for it to finish gracefully.
            # If the user has an editor/pager open, this waits for them to
            # close it before proceeding to the human checkpoint.
            if self.console:
                self.console.stop()
            if console_task:
                await console_task
            # Tear down any cloud machine we provisioned for this phase.
            # This MUST run on every exit path — leaked GPU machines are expensive.
            if compute_handle is not None:
                try:
                    await compute_handle.teardown()
                    logger.info("Tore down cloud machine for phase=%s", phase_id)
                    if emitter:
                        emitter.emit(
                            "compute_terminated",
                            agent_id=f"phase:{phase_id}",
                            parent_id="orchestrator",
                            instance_id=compute_handle.machine.id,
                        )
                except Exception:
                    logger.exception(
                        "Failed to teardown cloud machine id=%s for phase=%s; "
                        "manual cleanup may be required",
                        compute_handle.machine.id, phase_id,
                    )

        # Human checkpoint
        result_or_coro = self.on_phase_complete(phase_id, result)
        action = await result_or_coro if asyncio.iscoroutine(result_or_coro) else result_or_coro
        if action == UserAction.ABORT:
            self.spec_manager.log_event(EventType.run_failed, rationale="Aborted by user")
            # Mark all non-completed phases as failed to exit the loop
            for p in self.spec_manager.state.phases:
                if p.status not in (PhaseStatus.completed, PhaseStatus.failed):
                    p.status = PhaseStatus.failed
            self.spec_manager.save()
        elif action == UserAction.EDIT_SPEC:
            spec_path = self.spec_manager.store.spec_md_path
            open_in_editor(spec_path)
            ui.success("Spec updated. Continuing...")
        elif action == UserAction.SKIP:
            runnable = self.spec_manager.get_runnable_phases()
            skip_id = prompt_skip_which_phase(runnable)
            if skip_id:
                self.spec_manager.set_phase_status(
                    skip_id, PhaseStatus.completed, "Skipped by user",
                )
                ui.info(f"Skipped phase '{skip_id}'.")

    def _generate_claims_summary(self) -> None:
        """After all phases complete, aggregate per-phase claims reports into a
        single ``report/claims_verification.md`` and emit a chat summary."""
        from .claims import verify_phase_claims

        ledger = self.spec_manager.store.load_claims()
        if not ledger.claims:
            return

        all_verifications = []
        for phase in self.spec_manager.state.phases:
            work_dir = self.workspace.phase_dir(phase.phase_id)
            if not work_dir.exists():
                work_dir = None

            # Get the latest result from the failure handler (includes successes)
            prior = self.failure_handler.get_prior_results(phase.phase_id)
            latest = prior[-1] if prior else None
            if latest is None:
                continue

            report = verify_phase_claims(phase.phase_id, latest, ledger, work_dir)
            all_verifications.extend(report.verifications)

        if not all_verifications:
            return

        full_report = ClaimsReport(verifications=all_verifications)
        body = (
            "# Claims Verification Report\n\n"
            "Comparison of reproduced results against paper-reported claims.\n\n"
            f"{full_report.to_markdown()}\n"
        )

        try:
            report_path = self.config.report_dir / "claims_verification.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(body)
            logger.info("Claims verification report written to %s", report_path)
        except Exception:
            logger.exception("Failed to write claims verification report")

        if self.emitter:
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="system",
                text=(
                    f"📊 Final claims verification: "
                    f"{full_report.verified_count} verified, "
                    f"{full_report.close_count} close, "
                    f"{full_report.missed_count} missed, "
                    f"{full_report.exceeded_count} suspicious, "
                    f"{full_report.not_checked_count} unchecked — "
                    f"see report/claims_verification.md"
                ),
            )

        ui.info(
            f"Claims: {full_report.verified_count} verified, "
            f"{full_report.close_count} close, "
            f"{full_report.missed_count} missed, "
            f"{full_report.exceeded_count} suspicious"
        )

    async def _refine_spec_for_phase(self, phase_id: str, result: SubAgentResult) -> None:
        """Run the orchestrator spec refinement for a phase flagged as a spec issue.

        Behaviour:
        - If under amendment budget: call refine_spec(), apply the amendment,
          cascade-invalidate downstream completed phases, log to disk, signal chat.
        - If at budget cap: mark the phase failed and let the post-phase
          human checkpoint handle the escalation (edit_spec / skip / abort).
        """
        if not self.spec_manager.can_amend(phase_id):
            logger.warning(
                "Amendment budget exhausted for phase=%s (%d/%d) — escalating to human",
                phase_id,
                self.spec_manager.amendment_count(phase_id),
                self.spec_manager.MAX_AMENDMENTS_PER_PHASE,
            )
            ui.warning(
                f"Phase '{phase_id}' has hit the spec amendment cap "
                f"({self.spec_manager.MAX_AMENDMENTS_PER_PHASE}). "
                f"Escalating to human checkpoint."
            )
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.failed,
                f"Spec amendment cap reached: {result.summary[:100]}",
            )
            self._update_plan_for_phase(phase_id, PhaseStatus.failed)
            if self.emitter:
                self.emitter.emit(
                    "agent_message",
                    agent_id="orchestrator",
                    parent_id=None,
                    role="system",
                    text=(
                        f"⚠️ Phase `{phase_id}` exhausted its spec amendment "
                        f"budget ({self.spec_manager.MAX_AMENDMENTS_PER_PHASE}/"
                        f"{self.spec_manager.MAX_AMENDMENTS_PER_PHASE}). "
                        f"Human intervention required."
                    ),
                )
            return

        # Build trigger diagnostics from the sub-agent's spec_issue report
        # plus the latest post-mortem (if any).
        pm = self.failure_handler.get_post_mortem(phase_id)
        trigger = {
            "sub_agent_summary": result.summary,
            "sub_agent_diagnostics": result.diagnostics or {},
            "post_mortem": pm.model_dump() if pm else None,
        }

        try:
            amendment = await self.orchestrator_agent.refine_spec(
                phase_id, trigger, self.spec_manager, self.config.paper_path,
            )
        except Exception:
            logger.exception("refine_spec crashed for phase=%s", phase_id)
            amendment = None

        if amendment is None or not amendment.succeeded:
            summary = amendment.summary if amendment else "(refinement failed)"
            logger.warning("Spec refinement produced no amendment for phase=%s: %s", phase_id, summary)
            # Reset to pending without burning retry budget so the next attempt
            # can try a different angle. The amendment count is NOT bumped.
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.pending,
                f"Spec issue, refinement no-op: {summary[:100]}",
            )
            self.spec_manager.log_event(
                EventType.retry_launched,
                phase_id=phase_id,
                rationale=f"Spec issue, refinement no-op: {summary[:100]}",
            )
            if self.emitter:
                self.emitter.emit(
                    "agent_message",
                    agent_id="orchestrator",
                    parent_id=None,
                    role="system",
                    text=f"📝 Spec refinement no-op for `{phase_id}`: {summary}",
                )
            return

        # Apply the amendment.
        n = self.spec_manager.record_amendment(phase_id)
        rationale = (
            f"refine_spec amendment {n}/{self.spec_manager.MAX_AMENDMENTS_PER_PHASE} "
            f"for phase '{phase_id}': {amendment.summary}"
        )
        self.spec_manager.amend_spec_md(amendment.amended_spec_md, rationale)
        invalidated = self.spec_manager.invalidate_phase(phase_id, rationale)

        # Persist the full amendment record to logs/spec_amendments/.
        try:
            path = self.workspace.amendment_path(phase_id, n)
            body = (
                f"# Spec amendment {n}/{self.spec_manager.MAX_AMENDMENTS_PER_PHASE} "
                f"— phase `{phase_id}`\n\n"
                f"## Summary\n\n{amendment.summary}\n\n"
                f"## Sections changed\n\n"
                + ("\n".join(f"- {s}" for s in amendment.sections_changed) or "(unspecified)")
                + "\n\n"
                f"## Trigger\n\n```json\n{json.dumps(trigger, indent=2, default=str)}\n```\n\n"
                f"## Invalidated phases (cascade)\n\n"
                + ("\n".join(f"- `{p}`" for p in invalidated) or "(none)")
                + "\n\n"
                f"The full amended spec is in `canonical_spec/spec.md`.\n"
            )
            path.write_text(body)
            try:
                rel = path.relative_to(self.config.project_root)
            except ValueError:
                rel = path
        except Exception:
            logger.exception("Failed to write amendment log")
            rel = "logs/spec_amendments/<unknown>"

        if self.emitter:
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="system",
                text=(
                    f"📝 Spec amended ({n}/{self.spec_manager.MAX_AMENDMENTS_PER_PHASE}) "
                    f"for phase `{phase_id}` — {amendment.summary}. "
                    f"Invalidated: {invalidated or '(none)'}. See `{rel}`"
                ),
            )
            self._emit_dag_snapshot()

    async def _run_post_mortem(
        self,
        phase_id: str,
        failed_result: SubAgentResult,
        work_dir: Path,
        final: bool,
    ) -> PostMortem:
        """Run the orchestrator post-mortem, persist it to disk, signal chat.

        ``final=True`` indicates the phase has exhausted retries — the post-mortem
        is still useful as a written diagnosis for the human inspecting the
        wreckage, but it will not be injected into a next attempt.
        """
        try:
            pm = await self.orchestrator_agent.post_mortem(
                phase_id, failed_result, work_dir, self.spec_manager,
            )
        except Exception as e:
            logger.exception("Post-mortem crashed for phase=%s", phase_id)
            pm = PostMortem(
                failure_hypothesis=f"(post-mortem crashed: {e})",
                confidence="low",
            )

        # Persist the full record. The PostMortem itself is small; we also
        # echo the failed_result summary so the file is self-contained.
        retry_num = self.failure_handler.retries_used(phase_id)
        path = self.workspace.postmortem_path(phase_id, retry_num)
        body = (
            f"# Post-mortem: phase `{phase_id}` retry {retry_num}\n\n"
            f"- **Final attempt:** {final}\n"
            f"- **Confidence:** {pm.confidence}\n"
            f"- **Likely spec issue:** {pm.is_likely_spec_issue}\n\n"
            f"## Failure hypothesis\n\n{pm.failure_hypothesis}\n\n"
            f"## Suggested fix\n\n{pm.suggested_fix or '(none)'}\n\n"
            f"## Sub-agent's own summary\n\n{failed_result.summary}\n\n"
            f"## Sub-agent diagnostics\n\n```json\n"
            f"{json.dumps(failed_result.diagnostics or {}, indent=2)}\n```\n"
        )
        try:
            path.write_text(body)
        except Exception:
            logger.exception("Failed to write post-mortem to %s", path)

        # Project to paper-repo shape (notes/post-mortems/<phase>-retry-<N>.md)
        # so the Claude Code skill workflow's /post-mortem can find it too.
        try:
            from ..storage.paper_repo import project_post_mortem
            project_post_mortem(
                self.config,
                phase_id=phase_id,
                retry_num=retry_num,
                internal_path=path,
            )
        except Exception:
            logger.exception("Failed to project post-mortem to paper-repo shape; continuing")

        # Stash for the next attempt (if any).
        if not final:
            self.failure_handler.set_post_mortem(phase_id, pm)

        # Single chat signal in the orchestrator pane — full detail in the file.
        if self.emitter:
            tag = "📋 Post-mortem"
            if final:
                tag += " (final)"
            try:
                rel = path.relative_to(self.config.project_root)
            except ValueError:
                rel = path
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="system",
                text=(
                    f"{tag} for phase `{phase_id}` retry {retry_num} "
                    f"({pm.confidence} confidence): {pm.failure_hypothesis} — "
                    f"see `{rel}`"
                ),
            )
        return pm

    def _propagate_outputs_to_downstream(self, phase_id: str, outputs) -> None:
        """Update downstream phases' inputs to point at this phase's accepted outputs.

        Matches by artifact name. Logs a warning for any input that references
        this producer but has no matching output name.
        """
        if not outputs:
            return
        outputs_by_name = {o.name: o for o in outputs}
        downstream_ids = self.spec_manager.dep_graph.get_downstream(phase_id)
        for ds_id in downstream_ids:
            ds_phase = self.spec_manager.state.get_phase(ds_id)
            if ds_phase is None:
                continue
            updated = False
            for inp in ds_phase.inputs:
                # Heuristic: if this input's path mentions the producer phase dir
                # OR an output with the same name exists, rewrite it.
                producer_marker = f"phases/{phase_id}/"
                path_matches_producer = producer_marker in inp.file_path
                name_match = outputs_by_name.get(inp.name)
                if path_matches_producer or name_match:
                    if name_match:
                        if inp.file_path != name_match.file_path:
                            logger.info(
                                "Rewriting input '%s' on phase '%s': %s -> %s",
                                inp.name, ds_id, inp.file_path, name_match.file_path,
                            )
                            inp.file_path = name_match.file_path
                            updated = True
                    elif path_matches_producer:
                        logger.warning(
                            "Phase '%s' input '%s' references producer '%s' "
                            "but no matching output name found in %s",
                            ds_id, inp.name, phase_id,
                            list(outputs_by_name.keys()),
                        )
            if updated:
                logger.info("Updated downstream phase '%s' inputs after '%s' acceptance", ds_id, phase_id)

    async def _handle_result(self, phase_id: str, result: SubAgentResult) -> None:
        """Process a sub-agent result: accept, retry, or fail."""
        if result.status == ResultStatus.success:
            # Run acceptance review
            try:
                review_work_dir = getattr(self, "_current_work_dir", None)
                accepted, feedback = await self.orchestrator_agent.acceptance_review(
                    phase_id, result, self.spec_manager, work_dir=review_work_dir,
                )
            except asyncio.TimeoutError:
                logger.warning("Acceptance review timed out for phase=%s. Auto-accepting.", phase_id)
                accepted, feedback = True, None
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
                    # Propagate new output paths to downstream consumers' inputs
                    # so they don't reference stale phases/<id>/1/outputs/ paths
                    # from spec creation time.
                    self._propagate_outputs_to_downstream(phase_id, result.outputs)
                    self.spec_manager.save()
                self._update_plan_for_phase(
                    phase_id, PhaseStatus.completed, outputs=result.outputs,
                )
                return
            else:
                # Rejection counts as a retry
                logger.info("Acceptance rejected for phase=%s: %s", phase_id, feedback)
                ui.warning(f"Acceptance rejected for '{phase_id}': {feedback}")
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
            logger.info("Spec issue reported for phase=%s: %s", phase_id, result.summary)
            await self._refine_spec_for_phase(phase_id, result)
            return

        # Implementation failure — run a post-mortem before deciding what to do.
        # If the orchestrator concludes this is actually a spec issue, route
        # through the spec-issue branch so we trigger amendment (feature #1)
        # rather than burning a retry on the same broken spec.
        work_dir = getattr(self, "_current_work_dir", None)
        if work_dir is not None:
            final = not self.failure_handler.can_retry(phase_id)
            pm = await self._run_post_mortem(phase_id, result, work_dir, final=final)
            if pm.is_likely_spec_issue and not result.is_spec_issue and not final:
                logger.info(
                    "Post-mortem reclassified phase=%s as spec issue: %s",
                    phase_id, pm.failure_hypothesis,
                )
                self.spec_manager.set_phase_status(
                    phase_id, PhaseStatus.pending,
                    f"Post-mortem flagged spec issue: {pm.failure_hypothesis[:100]}",
                )
                self.spec_manager.log_event(
                    EventType.retry_launched,
                    phase_id=phase_id,
                    rationale=f"Spec issue (post-mortem, not counted): {pm.failure_hypothesis[:100]}",
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
            logger.error("Phase %s exhausted all retries", phase_id)
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.failed,
                f"Exhausted {self.config.max_retries} retries: {result.summary[:100]}",
            )
            self._update_plan_for_phase(phase_id, PhaseStatus.failed)

    def _update_plan_for_phase(
        self,
        phase_id: str,
        status: PhaseStatus,
        outputs=None,
    ) -> None:
        """Single write point for PlanDocument node + file lifecycle updates.

        - in_progress: node→in_progress, owned files (planned) → in_progress
        - completed:   node→completed,   owned files matching reported outputs → verified,
                       remaining owned files → written (file exists from report_result)
        - failed:      node→failed (file statuses left as-is for inspection)
        """
        plan = self.spec_manager.state.plan
        if plan is None:
            return
        node = plan.get_node(phase_id)
        if node is None:
            return
        node.status = status
        owned = plan.files_for_phase(phase_id)
        if status == PhaseStatus.in_progress:
            for f in owned:
                if f.status == FileStatus.planned:
                    f.status = FileStatus.in_progress
        elif status == PhaseStatus.completed:
            reported_paths = {o.file_path for o in (outputs or [])}
            reported_names = {o.name for o in (outputs or [])}
            for f in owned:
                # Match by either name (file_id) or by path suffix.
                matched = (
                    f.file_id in reported_names
                    or any(p.endswith(f.rel_path) for p in reported_paths)
                )
                f.status = FileStatus.verified if matched else FileStatus.written
        try:
            self.spec_manager.store.save_plan(plan)
            self.spec_manager.save()
        except Exception as e:
            logger.warning("Failed to persist plan update for phase=%s: %s", phase_id, e)

        # Emit lifecycle events for the viewer.
        if self.emitter:
            for f in owned:
                if f.status in (FileStatus.written, FileStatus.verified):
                    self.emitter.emit(
                        "file_created",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        file_id=f.file_id,
                        path=f.rel_path,
                        role=f.role.value,
                        status=f.status.value,
                    )
            if status in (PhaseStatus.completed, PhaseStatus.failed):
                self.emitter.emit(
                    "agent_completed",
                    agent_id=f"phase:{phase_id}",
                    parent_id="orchestrator",
                    status="completed" if status == PhaseStatus.completed else "failed",
                )
            self._emit_dag_snapshot()
