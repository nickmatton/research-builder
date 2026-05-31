"""Orchestrator execution loop (spec_v4 §4.2).

Runs phases in dependency order, dispatches sub-agents, handles results.
MVP: sequential execution. Parallel dispatch is a Phase 7 enhancement.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import click

from .. import ui
from ..commands import get_inbox
from ..config import Config
from ..events import get_emitter
from ..interaction import UserAction, open_in_editor, prompt_long_running_phase, prompt_skip_which_phase

if TYPE_CHECKING:
    from ..console import InteractiveConsole
    from ..cloud import CloudProvisioner
from ..models.context import PostMortem, RetryContext
from ..models.claims import ClaimsReport
from ..models.results import ResultStatus, SubAgentResult
from ..models.spec import EventType, FileStatus, PhaseStatus
from ..storage.step_records import StepRecord, now as step_now, write_step_record
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
        spec_manager: SpecManager | None,
        workspace: WorkspaceManager,
        orchestrator_agent: OrchestratorAgent,
        on_phase_complete: Callable[[str, SubAgentResult], UserAction | asyncio.coroutines] | None = None,
        on_phase_start: Callable[[str], asyncio.coroutines] | None = None,
        console: InteractiveConsole | None = None,
        cloud_provisioner: "CloudProvisioner | None" = None,
    ) -> None:
        # spec_manager may be None at construction time in the model-orchestrator
        # flow — the orchestrator's write_skeleton tool populates it on its
        # first turn. Methods that need a spec must guard against None.
        self.config = config
        self.spec_manager = spec_manager
        self.workspace = workspace
        self.orchestrator_agent = orchestrator_agent
        self.failure_handler = FailureHandler(max_retries=config.max_retries)
        async def _default_phase_complete(pid, r):
            return UserAction.CONTINUE
        self.on_phase_complete = on_phase_complete or _default_phase_complete
        async def _default_phase_start(pid):
            return None
        self.on_phase_start = on_phase_start or _default_phase_start
        self.console = console
        self.cloud_provisioner = cloud_provisioner
        self.total_cost_usd: float = 0.0
        # Hard LLM-spend cap. Set when total_cost_usd crosses
        # config.llm_spend_cap_usd. The outer loop drains this and exits
        # cleanly, marking remaining phases failed so the run can be resumed
        # later under a fresh budget.
        self._budget_exceeded: bool = False
        self.emitter = get_emitter()
        self._files_announced: set[str] = set()
        # Announce initial planned files + DAG so a viewer attached at startup
        # has a complete picture even before the first phase runs. In the
        # model-orchestrator flow, spec_manager is None at construction time
        # (the orchestrator's write_skeleton tool populates it later); the
        # helper handles that and is also called again from inside that tool.
        if self.emitter:
            self._emit_initial_plan()

    def _emit_initial_plan(self) -> None:
        """Emit file_planned for every file in the plan + an initial dag_updated.

        No-op when ``spec_manager`` is still None (model-orchestrator flow:
        the loop is constructed before write_skeleton runs). Safe to call
        again later — file_planned emissions dedupe via ``_files_announced``.
        """
        if self.spec_manager is None:
            return
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
        if self.spec_manager is None:
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
        # Resolve to absolute — sub-agents run in their own phase work-dir,
        # so a relative paper_path won't resolve. Read tool needs absolute.
        paper_path = str(Path(self.config.paper_path).resolve())
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

    def _check_llm_budget(self, source: str) -> None:
        """If total LLM spend has crossed the configured cap, set the
        abort flag and emit a loud error. The outer loop honours the flag
        between iterations and exits cleanly.

        ``source`` is a short label (e.g. ``"sub-agent"``, ``"orchestrator"``)
        included in the log line so the operator knows which call type
        pushed the total over.
        """
        cap = getattr(self.config, "llm_spend_cap_usd", 0.0) or 0.0
        if cap <= 0 or self._budget_exceeded:
            return
        if self.total_cost_usd < cap:
            return
        self._budget_exceeded = True
        msg = (
            f"LLM SPEND CAP EXCEEDED: total ${self.total_cost_usd:.2f} ≥ "
            f"cap ${cap:.2f} (last cost from {source}). Aborting run — "
            f"remaining phases will be marked failed. Raise --llm-budget / "
            f"RB_LLM_SPEND_CAP_USD and resume to continue."
        )
        logger.error(msg)
        ui.error(f"    {msg}")
        if self.emitter:
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="system",
                text=msg,
            )

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

    # ──────────────────────────────────────────────────────────────────
    # Per-attempt step records (Stage 1)
    # ──────────────────────────────────────────────────────────────────

    def _emit_step_started(self, phase_id: str, retry_num: int, role: str) -> None:
        if not self.emitter:
            return
        self.emitter.emit(
            "step_started",
            agent_id=f"phase:{phase_id}",
            parent_id="orchestrator",
            role=role,
            retry_num=retry_num,
        )

    def _emit_step_completed(
        self,
        phase_id: str,
        retry_num: int,
        role: str,
        *,
        started_at: float,
        cost_usd: float | None = None,
        **extra,
    ) -> None:
        if not self.emitter:
            return
        self.emitter.emit(
            "step_completed",
            agent_id=f"phase:{phase_id}",
            parent_id="orchestrator",
            role=role,
            retry_num=retry_num,
            duration_s=step_now() - started_at,
            cost_usd=cost_usd,
            **extra,
        )

    def _write_query_step_record(
        self,
        *,
        phase_id: str,
        retry_num: int,
        role: str,
        started_at: float,
        parsed: dict | None,
        extra: dict | None = None,
    ) -> None:
        """Persist a step record using ``orchestrator_agent._last_query``.

        Called immediately after a refiner/researcher/verifier invocation —
        ``orchestrator_agent._last_query`` holds the most recent ``_query()``
        record (prompt, response, cost, model, timings). If for some reason
        no query was captured (e.g. an early return on missing phase) we
        still emit a thin record so the manifest reflects the step ran.
        """
        try:
            work_dir = self.workspace.phase_dir(phase_id)
            q = getattr(self.orchestrator_agent, "_last_query", None)
            # Aggregate orchestrator query cost into the global total so the
            # LLM-spend cap covers refiner/verifier/post-mortem calls, not just
            # sub-agent dispatches. Without this the cap would silently miss a
            # meaningful chunk of spend (verifier alone runs once per phase).
            if q is not None and q.cost_usd:
                self.total_cost_usd += float(q.cost_usd)
                self._check_llm_budget(source=f"orchestrator/{role}")
            ended_at = step_now()
            record = StepRecord(
                role=role,
                phase_id=phase_id,
                retry_num=retry_num,
                started_at=started_at,
                ended_at=ended_at,
                duration_s=ended_at - started_at,
                cost_usd=q.cost_usd if q else None,
                model=q.model if q else None,
                prompt_role=q.prompt_role if q else None,
                system_prompt=q.system_prompt if q else None,
                prompt=q.prompt if q else None,
                response_text=q.response_text if q else None,
                parsed=parsed,
                messages_received=list(q.messages_received) if q else [],
                extra=extra or {},
                status=(q.status if q else "ok"),
            )
            write_step_record(work_dir, record)
        except Exception:
            logger.exception("Failed to write step record (role=%s phase=%s)", role, phase_id)

    def _write_builder_step_record(
        self,
        *,
        phase_id: str,
        retry_num: int,
        started_at: float,
        result: SubAgentResult,
    ) -> None:
        """Persist a thin builder-step record pointing at outputs/_result.json.

        The full builder result is already persisted by the sub-agent at
        ``<work_dir>/outputs/_result.json`` — we don't duplicate it here.
        """
        try:
            work_dir = self.workspace.phase_dir(phase_id)
            ended_at = step_now()
            tr = result.test_report
            record = StepRecord(
                role="builder",
                phase_id=phase_id,
                retry_num=retry_num,
                started_at=started_at,
                ended_at=ended_at,
                duration_s=ended_at - started_at,
                cost_usd=result.cost_usd or None,
                model=self.config.model,
                prompt_role=f"builder-{phase_id}",
                response_text=None,
                parsed={
                    "status": result.status.value,
                    "summary": result.summary,
                    "attempts_used": result.attempts_used,
                    "is_spec_issue": result.is_spec_issue,
                    "outputs": [o.model_dump() for o in result.outputs],
                    "test_report": {
                        "tests_run": tr.tests_run,
                        "tests_passed": tr.tests_passed,
                        "tests_failed": tr.tests_failed,
                    },
                },
                extra={
                    "result_json_path": "outputs/_result.json",
                    "diagnostics": result.diagnostics,
                },
                status="ok" if result.status == ResultStatus.success else "error",
            )
            write_step_record(work_dir, record)
        except Exception:
            logger.exception("Failed to write builder step record for %s", phase_id)

    # ──────────────────────────────────────────────────────────────────
    # Operator interventions (Stage 3a)
    # ──────────────────────────────────────────────────────────────────

    def _apply_pre_hook(self, phase_id: str, hook: str) -> int:
        """Drain and apply any operator interventions queued for ``hook``.

        Returns the number of commands applied.
        """
        try:
            cmds = get_inbox().drain_interventions(phase_id, hook)
        except Exception:
            logger.exception("intervention: drain failed for phase=%s hook=%s", phase_id, hook)
            return 0
        applied = 0
        for cmd in cmds:
            try:
                self._apply_intervention(cmd, phase_id=phase_id, hook=hook)
                applied += 1
            except Exception:
                logger.exception("intervention: failed to apply cmd=%s at hook=%s", cmd, hook)
        return applied

    def _apply_between_phases_hook(self) -> bool:
        """Drain force_retry / jump_back commands across all phases.

        Returns True if any phase state was mutated (so callers know to
        re-evaluate the runnable set).
        """
        try:
            # We don't know phase_ids up front, so iterate over current phases
            # and drain each. Also handle wildcard ("*", between_phases).
            mutated = False
            for phase in list(self.spec_manager.state.phases):
                cmds = get_inbox().drain_interventions(phase.phase_id, "between_phases")
                for cmd in cmds:
                    try:
                        self._apply_intervention(cmd, phase_id=phase.phase_id, hook="between_phases")
                        mutated = True
                    except Exception:
                        logger.exception("intervention: failed to apply cmd=%s", cmd)
            # Wildcard bucket
            cmds = get_inbox().drain_interventions("*", "between_phases")
            for cmd in cmds:
                try:
                    pid = cmd.get("payload", {}).get("phase_id") or cmd.get("payload", {}).get("to_phase_id")
                    self._apply_intervention(cmd, phase_id=pid or "*", hook="between_phases")
                    mutated = True
                except Exception:
                    logger.exception("intervention: failed to apply cmd=%s", cmd)
            return mutated
        except Exception:
            logger.exception("intervention: between_phases drain failed")
            return False

    def _apply_intervention(self, cmd: dict, *, phase_id: str, hook: str) -> None:
        """Apply one intervention command. Idempotent on cmd_id (deduped earlier)."""
        ctype = cmd.get("type")
        payload = cmd.get("payload", {}) or {}
        rationale = payload.get("rationale") or f"user {ctype}"
        cmd_id = cmd.get("cmd_id", "")
        target_phase = payload.get("phase_id") or payload.get("to_phase_id") or phase_id

        if ctype == "edit_refined_spec":
            self._apply_edit_refined_spec(target_phase, payload, cmd_id=cmd_id)
        elif ctype == "force_retry":
            self._apply_force_retry(target_phase, payload)
        elif ctype == "inject_note":
            self._apply_inject_note(target_phase, payload, scope=payload.get("scope", "phase"))
        elif ctype == "jump_back":
            self._apply_jump_back(target_phase, payload)
        else:
            logger.warning("intervention: unknown type=%r", ctype)
            return

        # Audit: revision log + event stream.
        try:
            self.spec_manager.log_event(
                EventType.user_intervened,
                phase_id=target_phase if target_phase != "*" else None,
                rationale=f"{ctype}: {rationale}",
            )
        except Exception:
            logger.exception("intervention: failed to append revision_log entry")
        self._emit_user_intervened(ctype, target_phase, hook, payload)

    def _apply_edit_refined_spec(self, phase_id: str, payload: dict, *, cmd_id: str) -> None:
        work_dir = self.workspace.phase_dir(phase_id)
        context_dir = work_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        refined_path = context_dir / "refined_spec.md"
        mode = (payload.get("mode") or "replace").lower()
        new_content = payload.get("content", "")

        # Audit sidecar: save the prior version + the operator's content.
        edits_dir = context_dir / "refined_spec_edits"
        edits_dir.mkdir(parents=True, exist_ok=True)
        try:
            if refined_path.exists():
                (edits_dir / f"{cmd_id or 'edit'}.prev.md").write_text(refined_path.read_text())
            (edits_dir / f"{cmd_id or 'edit'}.new.md").write_text(new_content)
        except Exception:
            logger.exception("intervention: failed to write refined_spec_edits sidecar")

        prior = refined_path.read_text() if refined_path.exists() else ""
        if mode == "patch":
            # Patch mode left intentionally simple: payload contains an
            # ``old``/``new`` pair we substitute exactly once. Operators with
            # complex needs should use replace.
            old = payload.get("old", "")
            if old and old in prior:
                refined_path.write_text(prior.replace(old, new_content, 1))
            else:
                logger.warning("intervention: patch 'old' not found, falling back to append")
                refined_path.write_text(prior + "\n\n" + new_content)
        elif mode == "append":
            refined_path.write_text(prior + "\n\n" + new_content)
        else:  # replace (default)
            refined_path.write_text(new_content)

    def _apply_force_retry(self, phase_id: str, payload: dict) -> None:
        rationale = payload.get("rationale") or "operator force_retry"
        # invalidate_phase already cascades to downstream completed phases.
        # When cascade=False, we re-mark the cascaded ones to their prior
        # status — but for the common case cascade=True is what the operator
        # wants. We keep the simple semantics: always cascade through
        # invalidate_phase; downstream phases that haven't completed yet
        # aren't touched anyway.
        invalidated = self.spec_manager.invalidate_phase(phase_id, rationale=rationale)
        logger.info("force_retry: invalidated %s", invalidated)
        work_dir = self.workspace.phase_dir(phase_id)
        context_dir = work_dir / "context"
        if payload.get("reset_refined_spec"):
            (context_dir / "refined_spec.md").unlink(missing_ok=True)
        if payload.get("reset_research_cache"):
            (context_dir / "research_notes.md").unlink(missing_ok=True)
            (context_dir / "research_questions.json").unlink(missing_ok=True)

    def _apply_inject_note(self, phase_id: str, payload: dict, *, scope: str) -> None:
        text = (payload.get("text") or "").strip()
        if not text:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        note_block = f"### {ts} — operator note\n\n{text}\n"
        if scope == "global":
            target = self.config.spec_dir / "operator_notes.md"
        else:
            target = self.workspace.phase_dir(phase_id) / "context" / "operator_notes.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        prior = target.read_text() if target.exists() else ""
        target.write_text(prior + ("\n" if prior else "") + note_block)

    def _apply_jump_back(self, phase_id: str, payload: dict) -> None:
        rationale = payload.get("rationale") or "operator jump_back"
        invalidated = self.spec_manager.invalidate_phase(phase_id, rationale=rationale)
        logger.info("jump_back: invalidated %s (preserve_artifacts=%s)", invalidated, payload.get("preserve_artifacts", True))

    def _read_operator_notes(self, phase_id: str, context_dir: "Path") -> str:
        """Concatenate phase-scoped + global operator notes for this phase."""
        chunks: list[str] = []
        global_path = self.config.spec_dir / "operator_notes.md"
        if global_path.exists():
            try:
                chunks.append("## Global operator notes\n\n" + global_path.read_text().strip())
            except Exception:
                pass
        phase_path = context_dir / "operator_notes.md"
        if phase_path.exists():
            try:
                chunks.append("## Phase operator notes\n\n" + phase_path.read_text().strip())
            except Exception:
                pass
        return "\n\n".join(chunks)

    def _maybe_prompt_long_phase(
        self,
        phase_id: str,
        context_dir: "Path",
        retry_num: int,
    ) -> "UserAction | None":
        """Return UserAction if the long-phase gate fires, else None.

        The gate consults the refiner's persisted runtime estimate. It fires
        only when the estimate exceeds the configured threshold and only on
        the first attempt of a phase — retries reuse the operator's earlier
        approval. Non-interactive runs log a warning and proceed.
        """
        threshold = self.config.long_phase_threshold_minutes
        if threshold <= 0 or retry_num != 0:
            return None
        est_path = context_dir / "estimated_minutes.txt"
        if not est_path.exists():
            return None
        try:
            est_minutes = int(est_path.read_text().strip())
        except (ValueError, OSError):
            return None
        if est_minutes <= threshold:
            return None
        if not self.config.interactive:
            logger.warning(
                "Phase %s estimated at %d min (threshold %d) — proceeding (non-interactive)",
                phase_id, est_minutes, threshold,
            )
            return None
        return prompt_long_running_phase(phase_id, est_minutes, threshold)

    def _emit_user_intervened(self, ctype: str, phase_id: str, hook: str, payload: dict) -> None:
        if not self.emitter:
            return
        self.emitter.emit(
            "user_intervened",
            agent_id=f"phase:{phase_id}" if phase_id != "*" else "orchestrator",
            parent_id="orchestrator",
            intervention=ctype,
            hook=hook,
            rationale=payload.get("rationale", ""),
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
            # Stage 3a: drain operator interventions that mutate phase state
            # (force_retry, jump_back). Runs every iteration so a command
            # issued during a phase execution is honoured on the next pass.
            self._apply_between_phases_hook()

            # LLM-budget guard: if a prior phase pushed total spend over the
            # cap, abort cleanly here. Marking remaining phases failed (rather
            # than leaving them pending) ensures the outer caller sees a hard
            # stop and the run can be resumed later without re-running what
            # already completed.
            if self._budget_exceeded:
                cap = getattr(self.config, "llm_spend_cap_usd", 0.0) or 0.0
                rationale = (
                    f"LLM spend cap exceeded: ${self.total_cost_usd:.2f} ≥ ${cap:.2f}"
                )
                for p in self.spec_manager.state.phases:
                    if p.status not in (PhaseStatus.completed, PhaseStatus.failed):
                        p.status = PhaseStatus.failed
                self.spec_manager.save()
                self.spec_manager.log_event(EventType.run_failed, rationale=rationale)
                logger.error("Run aborted — %s", rationale)
                self._project_run_complete(success=False)
                self._emit_orchestrator_completed("failed", rationale)
                return False

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
                    # Last-chance drain: an operator may have just queued a
                    # force_retry that would unblock the DAG.
                    if self._apply_between_phases_hook():
                        continue
                    logger.error("No runnable phases and nothing in progress — deadlock")
                    self.spec_manager.log_event(EventType.run_failed, rationale="Deadlock: no runnable phases")
                    self._emit_orchestrator_completed("failed", "Deadlock: no runnable phases")
                return False

            # MVP: execute sequentially
            for phase_id in runnable:
                # Pre-phase callback (optional). In the model-driven
                # orchestrator flow, main.py does NOT call run(); the
                # orchestrator's start_phase tool drives _execute_phase
                # directly and handles approval via request_user_approval.
                # This callback is kept for any future direct-driven usage.
                try:
                    await self.on_phase_start(phase_id)
                except Exception:
                    logger.exception("on_phase_start raised for %s; continuing", phase_id)
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
        # Resolve to absolute — sub-agents run in their own phase work-dir,
        # so a relative paper_path won't resolve. Read tool needs absolute.
        paper_path = str(Path(self.config.paper_path).resolve())
        sub_spec = self.spec_manager.extract_sub_spec(phase_id, paper_path=paper_path)

        # ────────────────────────────────────────────────────────────
        # Per-section agent chain: PLAN REFINER → RESEARCHER → BUILDER
        # The refiner + researcher run ONCE per section (cached on disk),
        # so retries reuse their output. They run only on the first attempt.
        # ────────────────────────────────────────────────────────────
        context_dir = work_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        refined_path = context_dir / "refined_spec.md"
        research_path = context_dir / "research_notes.md"

        # The orchestrator now authors per-section specs UPFRONT (in parallel,
        # before the execution loop starts). If a canonical section spec exists
        # for this phase, use it as the refined spec on first attempt — the
        # Refiner agent is demoted to "verify still valid" rather than "rewrite
        # from scratch", saving an LLM call per phase.
        upfront_spec = None
        if retry_num == 0 and not refined_path.exists():
            try:
                upfront_spec = self.spec_manager.store.load_section_spec(phase_id)
            except Exception:
                logger.exception("Failed to load upfront section spec for %s", phase_id)
                upfront_spec = None

        if upfront_spec is not None:
            # Use the upfront section spec; skip the legacy Refiner authoring.
            ui.info(f"  📝 Using upfront section spec for '{phase_id}' (refiner skipped)")
            refined_path.write_text(upfront_spec.spec_markdown)
            self._emit_step_started(phase_id, retry_num, "refiner")
            self._emit_step_completed(
                phase_id, retry_num, "refiner",
                started_at=step_now(),
                source="upfront_section_spec",
                skipped=True,
            )
            research_questions: list[str] = []
        elif retry_num == 0 and not refined_path.exists():
            self._apply_pre_hook(phase_id, "pre_refiner")
            ui.info(f"  📝 Plan refiner running for section '{phase_id}'...")
            self._emit_step_started(phase_id, retry_num, "refiner")
            refiner_started = step_now()
            refinement = await self.orchestrator_agent.refine_section(
                phase_id, self.spec_manager, Path(paper_path),
            )
            refined_md = refinement.get("refined_spec_md") or sub_spec.spec_markdown
            refined_path.write_text(refined_md)
            research_questions = refinement.get("research_questions") or []
            # Persist the refiner's runtime estimate so the gate can re-check
            # on retries without rerunning the refiner.
            est_raw = refinement.get("estimated_runtime_minutes")
            try:
                est_minutes = max(1, int(est_raw)) if est_raw is not None else None
            except (TypeError, ValueError):
                est_minutes = None
            if est_minutes is not None:
                (context_dir / "estimated_minutes.txt").write_text(str(est_minutes))

            # Sweep-driven best-guess values the refiner is asking the
            # operator to confirm. See Step D of "Kill sweep-shaped
            # acceptance criteria" in REFINER_SYSTEM_PROMPT. Persisted to
            # disk so a retry-from-disk sees them; emitted as an event so
            # the chat surfaces "approval needed" prominently. The actual
            # approve/deny flow runs through the existing pre_builder hook
            # and edit_refined_spec command — the operator either lets the
            # refined spec stand (suggested values are baked in) or edits
            # before the builder reads the spec.
            pending_approvals = list(refinement.get("pending_approvals") or [])
            approvals_path = context_dir / "pending_approvals.json"
            if pending_approvals:
                approvals_path.write_text(json.dumps(pending_approvals, indent=2))
                ui.warning(
                    f"  ⚠️  Refiner flagged {len(pending_approvals)} sweep-driven "
                    f"best-guess value(s) needing operator approval for '{phase_id}':"
                )
                for item in pending_approvals:
                    q = (item.get("question") or "").strip()
                    v = (item.get("suggested_value") or "").strip()
                    ui.warning(f"      • {q} → suggested: {v}")
                ui.warning(
                    f"      Review {refined_path} (look for '❓ APPROVAL PENDING' "
                    f"blockquotes) and edit / approve before the Builder runs."
                )
                emitter = get_emitter()
                if emitter:
                    emitter.emit(
                        "refiner_pending_approvals",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        phase_id=phase_id,
                        approvals=pending_approvals,
                        refined_spec_path=str(refined_path),
                    )
            elif approvals_path.exists():
                # Stale entry from a previous attempt — clear so retries don't
                # re-flag values the operator already approved.
                try:
                    approvals_path.unlink()
                except OSError:
                    pass

            self._write_query_step_record(
                phase_id=phase_id,
                retry_num=retry_num,
                role="refiner",
                started_at=refiner_started,
                parsed=refinement,
                extra={
                    "num_research_questions": len(research_questions),
                    "num_pending_approvals": len(pending_approvals),
                    "estimated_runtime_minutes": est_minutes,
                },
            )
            self._emit_step_completed(phase_id, retry_num, "refiner", started_at=refiner_started)
            if research_questions:
                (context_dir / "research_questions.json").write_text(
                    json.dumps(research_questions, indent=2)
                )
                self._apply_pre_hook(phase_id, "pre_researcher")
                ui.info(f"  🔬 Researcher running ({len(research_questions)} question(s))...")
                self._emit_step_started(phase_id, retry_num, "researcher")
                researcher_started = step_now()
                research = await self.orchestrator_agent.research_for_section(
                    phase_id, research_questions, self.spec_manager, Path(paper_path),
                )
                notes_md = research.get("research_notes_md", "")
                if notes_md.strip():
                    research_path.write_text(notes_md)
                self._write_query_step_record(
                    phase_id=phase_id,
                    retry_num=retry_num,
                    role="researcher",
                    started_at=researcher_started,
                    parsed=research,
                    extra={
                        "num_questions": len(research_questions),
                        "num_sources": len(research.get("sources", [])),
                    },
                )
                self._emit_step_completed(phase_id, retry_num, "researcher", started_at=researcher_started)

        # Stage 3a: operator interventions targeting the builder fire here —
        # right before refined_spec.md / research_notes.md / operator_notes.md
        # are read into the sub_spec. edit_refined_spec rewrites refined_path
        # in place; inject_note appends to operator_notes.md.
        self._apply_pre_hook(phase_id, "pre_builder")

        # If refined / research artifacts exist on disk (from this attempt or
        # a previous retry), use them to enrich the builder's sub_spec.
        if refined_path.exists():
            sub_spec.spec_markdown = refined_path.read_text()
        if research_path.exists():
            sub_spec.spec_markdown += (
                "\n\n## Research Notes (from Researcher agent)\n\n"
                + research_path.read_text()
            )
        # Operator-injected notes (phase-scoped + global) get prepended once
        # at the front of the builder's spec so the agent reads them before
        # the rest of the plan.
        operator_notes = self._read_operator_notes(phase_id, context_dir)
        if operator_notes:
            sub_spec.spec_markdown = (
                "## Operator notes\n\n" + operator_notes + "\n\n---\n\n" + sub_spec.spec_markdown
            )

        retry_context = None
        prior_results = self.failure_handler.get_prior_results(phase_id)
        if prior_results:
            retry_context = RetryContext(
                prior_results=prior_results,
                post_mortem=self.failure_handler.get_post_mortem(phase_id),
            )

        # Long-phase approval gate: if the refiner estimated this phase will
        # run over the threshold, prompt the operator before burning compute.
        # Fires at most once per phase (retries skip the gate — the operator
        # already approved this phase when it first reached the threshold).
        gate_action = self._maybe_prompt_long_phase(phase_id, context_dir, retry_num)
        if gate_action == UserAction.SKIP:
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.completed, "Skipped by operator (long-phase gate)",
            )
            ui.info(f"Skipped phase '{phase_id}' at long-phase gate.")
            return
        if gate_action == UserAction.ABORT:
            self.spec_manager.log_event(
                EventType.run_failed, rationale=f"Aborted by user at long-phase gate ({phase_id})"
            )
            for p in self.spec_manager.state.phases:
                if p.status not in (PhaseStatus.completed, PhaseStatus.failed):
                    p.status = PhaseStatus.failed
            self.spec_manager.save()
            return

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
                retry_num=retry_num,
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
                return
            # The inline viewer (rich-based transcript in the same terminal)
            # already renders the same content via its agent_thinking /
            # agent_tool emitter subscriptions. Writing a status_line on top
            # produces duplicate output. Suppress when the viewer is up.
            from ..viewer.inline import get_active_viewer
            if get_active_viewer() is not None:
                return
            if kind == "done":
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
            except Exception as exc:
                # Fail loud: the prompt rule says GPU-needed phases must NOT
                # silently fall back to local CPU (training quietly runs on
                # the wrong device for hours, burns $ on wasted compute, and
                # the operator only notices when the Lambda dashboard is empty).
                # Mark the phase failed with a clear diagnostic so the loop
                # stops re-picking it and the operator can fix the underlying
                # cause (typically an expired LAMBDA_API_KEY → 401) and resume.
                logger.exception(
                    "Cloud provisioning failed for phase=%s; refusing local fallback",
                    phase_id,
                )
                err_type = type(exc).__name__
                err_msg = str(exc)
                remediation = (
                    "If this is a 401 Unauthorized, your LAMBDA_API_KEY is "
                    "invalid/expired — rotate it and resume the phase. "
                    "Verify with: curl -u \"$LAMBDA_API_KEY:\" "
                    "https://cloud.lambda.ai/api/v1/instance-types"
                )
                ui.error(
                    f"    Cloud GPU provisioning FAILED for '{phase_id}': "
                    f"{err_type}: {err_msg}"
                )
                ui.error(f"    {remediation}")
                ui.error(
                    f"    Phase '{phase_id}' marked failed (will NOT fall back "
                    f"to local CPU). Fix the cloud issue, then resume."
                )
                failure_result = SubAgentResult(
                    status=ResultStatus.failure,
                    phase_id=phase_id,
                    summary=(
                        f"Cloud GPU provisioning failed ({err_type}): {err_msg}. "
                        f"Refusing local-CPU fallback per spec rule."
                    ),
                    diagnostics={
                        "cloud_provisioning_failed": True,
                        "error_type": err_type,
                        "error": err_msg,
                        "remediation": remediation,
                    },
                )
                self.failure_handler.record_result(phase_id, failure_result)
                self.spec_manager.set_phase_status(
                    phase_id, PhaseStatus.failed,
                    f"Cloud provisioning failed: {err_type}",
                )
                self.spec_manager.save()
                if emitter:
                    emitter.emit(
                        "agent_message",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        role="system",
                        text=(
                            f"Cloud GPU provisioning failed: {err_type}: {err_msg}\n\n"
                            f"{remediation}\n\n"
                            f"Phase marked failed — refusing to fall back to local CPU."
                        ),
                    )
                return

        sub_agent = SubAgent(
            config=self.config,
            sub_spec=sub_spec,
            work_dir=work_dir,
            retry_context=retry_context,
            on_activity=_on_activity,
            extra_user_messages=queued_msgs,
            cloud_provisioner=self.cloud_provisioner,
            compute_handle=compute_handle,
            access_approval_callback=getattr(self, "_access_approval_callback", None),
        )

        # Start interactive console alongside agent execution
        console_task = None
        if self.console:
            console_task = asyncio.create_task(self.console.run())

        try:
            self._emit_step_started(phase_id, retry_num, "builder")
            builder_started = step_now()
            result = await sub_agent.run()
            self.total_cost_usd += result.cost_usd
            if result.cost_usd > 0:
                ui.info(f"    Phase cost: ${result.cost_usd:.2f} | Total: ${self.total_cost_usd:.2f}")
            self._check_llm_budget(source=f"sub-agent/{phase_id}")
            self._write_builder_step_record(
                phase_id=phase_id,
                retry_num=retry_num,
                started_at=builder_started,
                result=result,
            )
            self._emit_step_completed(
                phase_id, retry_num, "builder",
                started_at=builder_started,
                cost_usd=result.cost_usd,
            )

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
            # Section Verifier: deterministic checks + tool-free LLM judge.
            # Payload includes ``deterministic_checks`` and the judge's raw
            # JSON; persisted to the section's context dir for diagnostics.
            verifier_retry_num = self.failure_handler.retries_used(phase_id)
            self._apply_pre_hook(phase_id, "pre_verifier")
            try:
                review_work_dir = getattr(self, "_current_work_dir", None)
                ui.info(f"  ✅ Section verifier running for '{phase_id}'...")
                self._emit_step_started(phase_id, verifier_retry_num, "verifier")
                verifier_started = step_now()
                accepted, feedback, verifier_payload = await self.orchestrator_agent.verify_section(
                    phase_id, result, self.spec_manager, work_dir=review_work_dir,
                )
                # Persist verifier output for the journal + post-hoc diagnostics
                if review_work_dir is not None:
                    try:
                        verify_path = review_work_dir / "context" / f"verification_retry_{verifier_retry_num}.json"
                        verify_path.parent.mkdir(parents=True, exist_ok=True)
                        verify_path.write_text(json.dumps(verifier_payload, indent=2))
                    except Exception:
                        logger.exception("Failed to persist verifier payload for %s", phase_id)
                self._write_query_step_record(
                    phase_id=phase_id,
                    retry_num=verifier_retry_num,
                    role="verifier",
                    started_at=verifier_started,
                    parsed=verifier_payload,
                    extra={"accepted": accepted, "feedback": feedback},
                )
                self._emit_step_completed(
                    phase_id, verifier_retry_num, "verifier",
                    started_at=verifier_started,
                    accepted=accepted,
                )
            except asyncio.TimeoutError:
                # Fail-closed: timeout means we couldn't verify, not that
                # the section is good. Counts as a rejection so the section
                # goes back through the retry pipeline rather than slipping
                # through as silently-accepted.
                logger.warning(
                    "Section verifier timed out for phase=%s — REJECTING (fail-closed)",
                    phase_id,
                )
                accepted, feedback = False, "Section verifier timed out (fail-closed)"
            except Exception as e:
                # Fail-closed: any unexpected exception in the verifier
                # itself is treated as rejection, not acceptance. The old
                # auto-accept behavior turned every verifier bug into a
                # rubber-stamp.
                logger.warning(
                    "Section verifier failed for phase=%s: %s — REJECTING (fail-closed)",
                    phase_id, e,
                )
                accepted, feedback = False, f"Section verifier crashed: {e}"
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

    # ─── Per-step methods for the model-driven orchestrator path ─────────
    #
    # These are called by the per-step MCP tools (run_refiner, run_researcher,
    # run_builder, run_verifier) in orchestrator/runtime_tools.py. They run
    # ONE step of the per-section chain, emit the same events as _execute_phase
    # so the chat / trace surfaces light up identically, and update the
    # failure_handler so retry budgets stay accurate.
    #
    # Unlike _execute_phase, these methods do NOT auto-retry on failure,
    # do NOT run the post-mortem / spec-amendment logic, and do NOT advance
    # the phase to its next status. The orchestrator (model) decides those
    # next actions by reading the returned summary and calling the next tool.

    async def _step_refiner(self, phase_id: str) -> dict:
        """Run the refiner step for a phase.

        If an upfront section spec exists at canonical_spec/sections/<id>.md,
        it's used as the refined spec and the LLM refiner is skipped (matches
        _execute_phase's caching). Otherwise calls orchestrator_agent.refine_section.

        Writes refined_spec.md to phases/<id>/context/. Returns
        ``{"source": "upfront"|"refiner_run"|"cached", "research_questions": [...]}``.
        """
        work_dir = self.workspace.create_phase_dir(phase_id)
        self._current_work_dir = work_dir
        context_dir = work_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        refined_path = context_dir / "refined_spec.md"
        retry_num = self.failure_handler.retries_used(phase_id)

        # Already produced on a prior attempt? Reuse.
        if refined_path.exists():
            questions_path = context_dir / "research_questions.json"
            questions: list[str] = []
            if questions_path.exists():
                try:
                    questions = json.loads(questions_path.read_text()) or []
                except Exception:
                    questions = []
            return {
                "source": "cached",
                "research_questions": questions,
                "path": str(refined_path),
            }

        # Try the upfront section spec authored during stage 2.
        try:
            upfront_spec = self.spec_manager.store.load_section_spec(phase_id)
        except Exception:
            logger.exception("_step_refiner: load_section_spec failed for %s", phase_id)
            upfront_spec = None

        if upfront_spec is not None:
            ui.info(f"  📝 Using upfront section spec for '{phase_id}' (refiner skipped)")
            refined_path.write_text(upfront_spec.spec_markdown)
            self._emit_step_started(phase_id, retry_num, "refiner")
            self._emit_step_completed(
                phase_id, retry_num, "refiner",
                started_at=step_now(),
                source="upfront_section_spec",
                skipped=True,
            )
            return {
                "source": "upfront",
                "research_questions": [],
                "path": str(refined_path),
            }

        # Fresh refiner run.
        self._apply_pre_hook(phase_id, "pre_refiner")
        ui.info(f"  📝 Plan refiner running for section '{phase_id}'...")
        self._emit_step_started(phase_id, retry_num, "refiner")
        refiner_started = step_now()
        paper_path = Path(self.config.paper_path).resolve()
        refinement = await self.orchestrator_agent.refine_section(
            phase_id, self.spec_manager, paper_path,
        )
        refined_md = refinement.get("refined_spec_md") or ""
        if not refined_md.strip():
            # Fall back to extracting from sub_spec — matches _execute_phase.
            sub_spec = self.spec_manager.extract_sub_spec(phase_id, paper_path=str(paper_path))
            refined_md = sub_spec.spec_markdown
        refined_path.write_text(refined_md)
        questions: list[str] = refinement.get("research_questions") or []
        if questions:
            (context_dir / "research_questions.json").write_text(
                json.dumps(questions, indent=2)
            )
        self._write_query_step_record(
            phase_id=phase_id,
            retry_num=retry_num,
            role="refiner",
            started_at=refiner_started,
            parsed=refinement,
            extra={"num_research_questions": len(questions)},
        )
        self._emit_step_completed(phase_id, retry_num, "refiner", started_at=refiner_started)
        return {
            "source": "refiner_run",
            "research_questions": questions,
            "path": str(refined_path),
        }

    async def _step_researcher(
        self, phase_id: str, questions: list[str] | None = None,
    ) -> dict:
        """Run the researcher step for a phase.

        Early-returns ``{"skipped": True}`` if no research questions exist
        (either from arg or from cached research_questions.json). Otherwise
        calls orchestrator_agent.research_for_section, writes research_notes.md.

        Returns ``{"skipped": bool, "sources": [...], "num_questions": N}``.
        """
        work_dir = self.workspace.create_phase_dir(phase_id)
        self._current_work_dir = work_dir
        context_dir = work_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        research_path = context_dir / "research_notes.md"
        retry_num = self.failure_handler.retries_used(phase_id)

        # Already done on prior attempt? Reuse.
        if research_path.exists():
            return {
                "skipped": False,
                "cached": True,
                "sources": [],
                "path": str(research_path),
            }

        # Resolve questions: arg first, else look for cached file from the
        # refiner step.
        if questions is None:
            qpath = context_dir / "research_questions.json"
            if qpath.exists():
                try:
                    questions = json.loads(qpath.read_text()) or []
                except Exception:
                    questions = []
            else:
                questions = []

        if not questions:
            return {"skipped": True, "reason": "no research questions"}

        self._apply_pre_hook(phase_id, "pre_researcher")
        ui.info(f"  🔬 Researcher running ({len(questions)} question(s))...")
        self._emit_step_started(phase_id, retry_num, "researcher")
        researcher_started = step_now()
        paper_path = Path(self.config.paper_path).resolve()
        research = await self.orchestrator_agent.research_for_section(
            phase_id, questions, self.spec_manager, paper_path,
        )
        notes_md = research.get("research_notes_md", "") or ""
        if notes_md.strip():
            research_path.write_text(notes_md)
        sources = research.get("sources") or []
        self._write_query_step_record(
            phase_id=phase_id,
            retry_num=retry_num,
            role="researcher",
            started_at=researcher_started,
            parsed=research,
            extra={"num_questions": len(questions), "num_sources": len(sources)},
        )
        self._emit_step_completed(
            phase_id, retry_num, "researcher", started_at=researcher_started,
        )
        return {
            "skipped": False,
            "cached": False,
            "sources": sources,
            "num_questions": len(questions),
            "path": str(research_path) if notes_md.strip() else None,
        }

    async def _step_builder(self, phase_id: str) -> SubAgentResult:
        """Run the Builder sub-agent for a phase.

        Builds enriched sub_spec from refined_spec.md + research_notes.md +
        operator_notes.md, handles cloud GPU provisioning, dispatches the
        SubAgent, records the result via FailureHandler, tears down the
        cloud machine on exit. Returns the structured SubAgentResult.

        Does NOT run the verifier (that's _step_verifier) or trigger any
        retry / spec-amendment logic (that's the orchestrator's decision).
        """
        phase = self.spec_manager.state.get_phase(phase_id)
        if phase is None:
            raise ValueError(f"Unknown phase: {phase_id}")

        work_dir = self.workspace.create_phase_dir(phase_id)
        self._current_work_dir = work_dir
        context_dir = work_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        refined_path = context_dir / "refined_spec.md"
        research_path = context_dir / "research_notes.md"
        retry_num = self.failure_handler.retries_used(phase_id)

        self.spec_manager.set_phase_status(
            phase_id, PhaseStatus.in_progress,
            f"Builder starting (retry {retry_num})" if retry_num else "Builder starting",
        )
        self._update_plan_for_phase(phase_id, PhaseStatus.in_progress)

        paper_path = str(Path(self.config.paper_path).resolve())
        sub_spec = self.spec_manager.extract_sub_spec(phase_id, paper_path=paper_path)

        self._apply_pre_hook(phase_id, "pre_builder")

        # Enrich sub_spec from disk artifacts produced by earlier steps.
        if refined_path.exists():
            sub_spec.spec_markdown = refined_path.read_text()
        if research_path.exists():
            sub_spec.spec_markdown += (
                "\n\n## Research Notes (from Researcher agent)\n\n"
                + research_path.read_text()
            )
        operator_notes = self._read_operator_notes(phase_id, context_dir)
        if operator_notes:
            sub_spec.spec_markdown = (
                "## Operator notes\n\n" + operator_notes
                + "\n\n---\n\n" + sub_spec.spec_markdown
            )

        retry_context = None
        prior_results = self.failure_handler.get_prior_results(phase_id)
        if prior_results:
            retry_context = RetryContext(
                prior_results=prior_results,
                post_mortem=self.failure_handler.get_post_mortem(phase_id),
            )

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
                elif kind in ("tool", "done"):
                    emitter.emit(
                        "agent_tool",
                        agent_id=f"phase:{phase_id}",
                        parent_id="orchestrator",
                        summary=detail,
                    )

        # agent_started event so the chat + trace see the builder light up.
        total_phases = len(self.spec_manager.state.phases)
        completed = sum(
            1 for p in self.spec_manager.state.phases if p.status == PhaseStatus.completed
        )
        phase_num = completed + 1
        ui.step(
            f"Phase '{phase_id}' [{phase_num}/{total_phases}]"
            + (f" — retry {retry_num}" if retry_num else "")
        )
        if emitter:
            emitter.emit(
                "agent_started",
                agent_id=f"phase:{phase_id}",
                parent_id="orchestrator",
                kind="subagent",
                title=phase.title or phase_id,
                retry_num=retry_num,
            )
            self._emit_dag_snapshot()

        queued_msgs = get_inbox().drain(f"phase:{phase_id}")
        if queued_msgs and emitter:
            for m in queued_msgs:
                emitter.emit(
                    "agent_message",
                    agent_id=f"phase:{phase_id}",
                    parent_id="orchestrator",
                    role="user",
                    text=m[:2000],
                )

        # GPU provisioning (optional).
        compute_handle = None
        if self.cloud_provisioner is not None:
            try:
                try:
                    full_spec_md = self.spec_manager.store.load_spec_md()
                except Exception:
                    full_spec_md = None
                compute_handle = await self.cloud_provisioner.provision(
                    sub_spec, work_dir, full_spec_markdown=full_spec_md,
                )
            except Exception as exc:
                logger.exception(
                    "Cloud provisioning failed for phase=%s (_step_builder)", phase_id,
                )
                failure = SubAgentResult(
                    status=ResultStatus.failure,
                    phase_id=phase_id,
                    summary=f"Cloud GPU provisioning failed: {type(exc).__name__}: {exc}",
                    diagnostics={"cloud_provisioning_failed": True, "error": str(exc)},
                )
                self.failure_handler.record_result(phase_id, failure)
                self.spec_manager.set_phase_status(
                    phase_id, PhaseStatus.failed, "Cloud provisioning failed",
                )
                return failure

        sub_agent = SubAgent(
            config=self.config,
            sub_spec=sub_spec,
            work_dir=work_dir,
            retry_context=retry_context,
            on_activity=_on_activity,
            extra_user_messages=queued_msgs,
            cloud_provisioner=self.cloud_provisioner,
            compute_handle=compute_handle,
            access_approval_callback=getattr(self, "_access_approval_callback", None),
        )

        try:
            self._emit_step_started(phase_id, retry_num, "builder")
            builder_started = step_now()
            result = await sub_agent.run()
            self.total_cost_usd += result.cost_usd
            self._check_llm_budget(source=f"sub-agent/{phase_id}")
            self._write_builder_step_record(
                phase_id=phase_id,
                retry_num=retry_num,
                started_at=builder_started,
                result=result,
            )
            self._emit_step_completed(
                phase_id, retry_num, "builder",
                started_at=builder_started,
                cost_usd=result.cost_usd,
            )
            # Record result for retry accounting (orchestrator decides
            # whether to actually retry — we just count the attempt).
            self.failure_handler.record_result(phase_id, result)
            return result
        finally:
            if compute_handle is not None:
                try:
                    await compute_handle.teardown()
                except Exception:
                    logger.exception(
                        "Failed to teardown cloud machine for phase=%s", phase_id,
                    )

    async def _step_verifier(
        self, phase_id: str, builder_result: SubAgentResult,
    ) -> tuple[bool, dict]:
        """Run the Section Verifier on a builder result.

        Calls orchestrator_agent.verify_section, persists verification_retry_N.json,
        emits step events. On accepted=True, updates phase status to completed
        and propagates outputs. On accepted=False, records a failure result
        (so retry budget counts) but does NOT auto-retry — the orchestrator
        decides.

        Returns ``(accepted, payload)``. payload is the structured verifier
        output (deterministic_checks + llm_judge + feedback) — model uses it
        to decide next steps on rejection.
        """
        retry_num = self.failure_handler.retries_used(phase_id)
        work_dir = getattr(self, "_current_work_dir", None) or self.workspace.create_phase_dir(phase_id)
        self._apply_pre_hook(phase_id, "pre_verifier")
        ui.info(f"  ✅ Section verifier running for '{phase_id}'...")
        self._emit_step_started(phase_id, retry_num, "verifier")
        verifier_started = step_now()
        accepted: bool
        feedback: str | None
        verifier_payload: dict
        try:
            accepted, feedback, verifier_payload = await self.orchestrator_agent.verify_section(
                phase_id, builder_result, self.spec_manager, work_dir=work_dir,
            )
            verify_path = work_dir / "context" / f"verification_retry_{retry_num}.json"
            verify_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                verify_path.write_text(json.dumps(verifier_payload, indent=2))
            except Exception:
                logger.exception("Failed to persist verifier payload for %s", phase_id)
            self._write_query_step_record(
                phase_id=phase_id,
                retry_num=retry_num,
                role="verifier",
                started_at=verifier_started,
                parsed=verifier_payload,
                extra={"accepted": accepted, "feedback": feedback},
            )
        except asyncio.TimeoutError:
            logger.warning("Verifier timed out for phase=%s — REJECTING", phase_id)
            accepted, feedback = False, "Section verifier timed out (fail-closed)"
            verifier_payload = {"error": "timeout"}
        except Exception as e:
            logger.warning("Verifier crashed for phase=%s: %s — REJECTING", phase_id, e)
            accepted, feedback = False, f"Section verifier crashed: {e}"
            verifier_payload = {"error": str(e)}
        self._emit_step_completed(
            phase_id, retry_num, "verifier",
            started_at=verifier_started,
            accepted=accepted,
        )

        if accepted:
            # Mark complete + propagate outputs to downstream phases.
            self.spec_manager.set_phase_status(
                phase_id, PhaseStatus.completed,
                f"Accepted: {(builder_result.summary or '')[:100]}",
            )
            phase = self.spec_manager.state.get_phase(phase_id)
            if phase:
                phase.outputs = builder_result.outputs
                self._propagate_outputs_to_downstream(phase_id, builder_result.outputs)
                self.spec_manager.save()
            self._update_plan_for_phase(
                phase_id, PhaseStatus.completed, outputs=builder_result.outputs,
            )
        else:
            # Record as a failure so retry budget tracks the rejection. Do
            # NOT change phase status — the model decides whether to retry,
            # ask the user, or escalate to failed.
            rejection = SubAgentResult(
                status=ResultStatus.failure,
                phase_id=phase_id,
                summary=f"Acceptance rejected: {feedback}",
                test_report=builder_result.test_report,
                outputs=builder_result.outputs,
            )
            self.failure_handler.record_result(phase_id, rejection)

        # Carry feedback into the payload for the tool's text summary.
        verifier_payload = {**verifier_payload, "_accepted": accepted, "_feedback": feedback}
        return accepted, verifier_payload

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
