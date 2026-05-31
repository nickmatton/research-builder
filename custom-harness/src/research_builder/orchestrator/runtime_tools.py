"""In-process custom tools for the model-driven orchestrator runtime.

These are plain async Python functions registered with the Claude Agent SDK
via ``create_sdk_mcp_server``. The "MCP" in that helper is just the SDK's
plugin-registration format — nothing crosses a network or process boundary.
Tool calls dispatch as direct coroutine invocations in our process. The
``mcp__<server>__<tool>`` prefix you see in tool names is the SDK's internal
tool-routing convention; we don't control that part.

The orchestrator is a single long-running ``query()`` loop whose tools wrap
existing Python machinery (skeleton authoring, parallel section spec fan-out,
critique, claims extraction, per-step phase execution). The model decides
ordering, narrates each step, and asks the user for approval via
``request_user_approval`` — Python no longer sequences stages.

Each tool returns a compact text summary the model can read on its next turn
to decide what to do. Heavy lifting (parallel asyncio.gather, retries, file
I/O) stays in Python — tools are thin façades, not where work lives.

Tools provided:

  - write_skeleton: read the paper, write canonical_spec/spec.md + state.json
  - author_section_specs: fan out detailed per-section spec authoring
  - critique_section_specs: critic pass against the paper
  - extract_claims_ledger: extract numerical claims
  - list_pending_phases: enumerate phases ready to execute (DAG order)
  - run_refiner / run_researcher / run_builder / run_verifier: granular
    per-step tools for the per-section chain
  - start_phase: convenience — runs all four sub-steps sequentially
  - request_user_approval: pause for the user's next chat reply
  - pipeline_complete / pipeline_failed: terminal tools, end the loop
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

if TYPE_CHECKING:
    from ..config import Config
    from ..storage.spec_store import SpecStore
    from ..storage.workspace import WorkspaceManager
    from .agent import OrchestratorAgent
    from .loop import ExecutionLoop
    from .spec_manager import SpecManager

logger = logging.getLogger(__name__)


# ─── Runtime state shared across tools ─────────────────────────────────────


@dataclass
class OrchestratorRuntime:
    """Mutable state every orchestrator tool reads/writes.

    Held on the OrchestratorAgent for the lifetime of one ``run_as_orchestrator``
    call. Tools close over this rather than taking it as an arg because the MCP
    tool decorator wraps the function signature.
    """

    config: "Config"
    paper_path: Path
    workspace: "WorkspaceManager"
    store: "SpecStore"
    orchestrator_agent: "OrchestratorAgent"
    execution_loop: "ExecutionLoop"
    spec_manager: "SpecManager | None" = None
    # Approval queue — request_user_approval awaits .get(); the chat handler
    # routes user replies via .put_nowait(text).
    approval_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    # Terminal-tool signal. The orchestrator loop polls this after each turn
    # and exits cleanly when set.
    done: bool = False
    success: bool = True
    final_message: str = ""


def deliver_user_reply(runtime: OrchestratorRuntime, text: str) -> None:
    """Route an inbound chat message to the open approval (if any).

    Called by main.py's chat handler. If no approval is open the message is
    queued anyway — the next request_user_approval call will pick it up. If
    multiple messages arrive between calls, only the latest is meaningful
    (drain old ones).
    """
    # Drain stale replies — the orchestrator only consumes one per gate.
    while not runtime.approval_queue.empty():
        try:
            runtime.approval_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    runtime.approval_queue.put_nowait(text)


# ─── Tool definitions ──────────────────────────────────────────────────────


def create_orchestrator_tools(runtime: OrchestratorRuntime):
    """Register the orchestrator's in-process custom tools.

    Returns an SDK tool-server handle to plug into
    ``ClaudeAgentOptions(mcp_servers={"orchestrator": handle})``. Tool calls
    dispatch as direct coroutines in this process — no IPC, no network.
    """

    @tool(
        "write_skeleton",
        "Read the research paper and write the slim top-level skeleton: "
        "canonical_spec/spec.md (under 200 lines) and state.json (section list "
        "+ DAG). Call this once at the start of the run, after introducing "
        "yourself and getting the user's approval. Returns a summary of what "
        "was found (section IDs, titles, phase count).",
        {"type": "object", "properties": {}, "required": []},
    )
    async def write_skeleton(_args: dict[str, Any]) -> dict[str, Any]:
        try:
            spec_manager = await runtime.orchestrator_agent.create_top_level_spec(
                runtime.paper_path, runtime.store,
            )
        except Exception as e:
            logger.exception("write_skeleton failed")
            return _err(f"write_skeleton failed: {type(e).__name__}: {e}")
        runtime.spec_manager = spec_manager
        # Wire the execution loop with the new spec_manager so subsequent
        # start_phase calls operate on the right state.
        runtime.execution_loop.spec_manager = spec_manager
        # Also re-emit the initial plan now that we have one.
        try:
            runtime.execution_loop._emit_initial_plan()
        except Exception:
            pass

        phases = spec_manager.state.phases
        lines = [f"- {p.phase_id} [{p.kind.value}]: {p.title}" for p in phases]
        summary = (
            f"Skeleton written. Found {len(phases)} section(s):\n"
            + "\n".join(lines)
            + f"\n\nFiles: {runtime.store.spec_md_path}, {runtime.store.state_path}"
        )
        return _ok(summary)

    @tool(
        "author_section_specs",
        "Fan out per-section spec authoring in parallel. One agent per "
        "section reads its paper pages and writes a detailed spec with "
        "cited acceptance criteria. Call this AFTER write_skeleton and "
        "after the user has approved the skeleton. Returns counts of "
        "succeeded vs. failed authorings.",
        {"type": "object", "properties": {}, "required": []},
    )
    async def author_section_specs(_args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("author_section_specs called before write_skeleton — no skeleton on disk.")
        try:
            specs = await runtime.orchestrator_agent.create_section_specs(
                runtime.spec_manager, runtime.paper_path, runtime.store,
            )
        except Exception as e:
            logger.exception("author_section_specs failed")
            return _err(f"author_section_specs failed: {type(e).__name__}: {e}")
        total = len(runtime.spec_manager.state.phases)
        summary = (
            f"Authored {len(specs)}/{total} section spec(s). "
            f"Each lives at {runtime.store.sections_dir}/<phase_id>.md."
        )
        if len(specs) < total:
            missed = {p.phase_id for p in runtime.spec_manager.state.phases} - {
                s.phase_id for s in specs
            }
            summary += f" Missing: {sorted(missed)}."
        return _ok(summary)

    @tool(
        "critique_section_specs",
        "Run the critic pass: for each section spec, a fresh LLM reads the "
        "relevant paper pages and judges whether the spec is grounded "
        "(verified / questionable / missing_citations). Call AFTER "
        "author_section_specs. Returns verdict counts + flagged sections.",
        {"type": "object", "properties": {}, "required": []},
    )
    async def critique_section_specs(_args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("critique_section_specs called before any sections exist.")
        # Reload section specs from disk so we critique what's actually written.
        spec_ids = runtime.store.list_section_spec_ids()
        specs = []
        for pid in spec_ids:
            s = runtime.store.load_section_spec(pid)
            if s is not None:
                specs.append(s)
        if not specs:
            return _err("No section specs found on disk.")
        try:
            critiques = await runtime.orchestrator_agent.critique_section_specs(
                specs, runtime.paper_path, runtime.store,
            )
        except Exception as e:
            logger.exception("critique_section_specs failed")
            return _err(f"critique_section_specs failed: {type(e).__name__}: {e}")
        counts: dict[str, int] = {}
        flagged: list[str] = []
        for c in critiques:
            counts[c.verdict.value] = counts.get(c.verdict.value, 0) + 1
            if c.verdict.value != "verified":
                flagged.append(f"{c.phase_id} ({c.verdict.value})")
        summary = "Critique: " + ", ".join(f"{n} {v}" for v, n in counts.items())
        if flagged:
            summary += f". Flagged: {', '.join(flagged)}"
        return _ok(summary)

    @tool(
        "extract_claims_ledger",
        "Extract the numerical claims (table rows, figure annotations, "
        "headline accuracy numbers) from the paper into a structured ledger "
        "at canonical_spec/claims.json. Each claim is tied to the phase that "
        "should reproduce it. Returns claim count.",
        {"type": "object", "properties": {}, "required": []},
    )
    async def extract_claims_ledger(_args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("extract_claims_ledger called before write_skeleton.")
        from ..llm.paper import get_page_count
        try:
            page_count = get_page_count(runtime.paper_path)
        except Exception:
            page_count = 20
        phase_ids = {p.phase_id for p in runtime.spec_manager.state.phases}
        try:
            ledger = await runtime.orchestrator_agent._extract_claims(
                runtime.paper_path.resolve(), page_count, phase_ids,
            )
        except Exception as e:
            logger.exception("extract_claims_ledger failed")
            return _err(f"extract_claims_ledger failed: {type(e).__name__}: {e}")
        if ledger.claims:
            runtime.store.save_claims(ledger)
        return _ok(
            f"Extracted {len(ledger.claims)} claim(s) → {runtime.store.claims_path}"
        )

    @tool(
        "list_pending_phases",
        "Return the phases that are ready to execute next (DAG dependencies "
        "satisfied, status != completed). Use this between start_phase calls "
        "to know what's runnable. Returns a JSON array of "
        "{phase_id, title, kind, status, depends_on, planned_files}.",
        {"type": "object", "properties": {}, "required": []},
    )
    async def list_pending_phases(_args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("list_pending_phases called before write_skeleton.")
        runnable = runtime.spec_manager.get_runnable_phases()
        out = []
        for pid in runnable:
            phase = runtime.spec_manager.state.get_phase(pid)
            if phase is None:
                continue
            out.append({
                "phase_id": pid,
                "title": phase.title,
                "kind": phase.kind.value,
                "status": phase.status.value,
                "depends_on": runtime.spec_manager.dep_graph.get_dependencies(pid),
                "planned_files": [
                    {"name": a.name, "path": a.file_path} for a in phase.outputs
                ],
            })
        return _ok(json.dumps(out, indent=2))

    @tool(
        "run_refiner",
        "Run ONLY the refiner step for a phase. Uses the upfront section "
        "spec if available (skips the LLM call). Otherwise calls the LLM "
        "refiner. Writes refined_spec.md and surfaces any research_questions "
        "the refiner identified. Returns a summary indicating source "
        "(upfront/refiner_run/cached) and any questions. Fast (skipped path) "
        "or ~10-30s (LLM path). Usually no need to pause before this.",
        {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string", "description": "Phase to refine."},
            },
            "required": ["phase_id"],
        },
    )
    async def run_refiner(args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("run_refiner called before write_skeleton.")
        phase_id = str(args.get("phase_id", "")).strip()
        if not phase_id:
            return _err("run_refiner requires phase_id.")
        if runtime.spec_manager.state.get_phase(phase_id) is None:
            return _err(f"Unknown phase: {phase_id}")
        try:
            result = await runtime.execution_loop._step_refiner(phase_id)
        except Exception as e:
            logger.exception("run_refiner: _step_refiner raised for %s", phase_id)
            return _err(f"refiner crashed: {type(e).__name__}: {e}")
        src = result.get("source", "unknown")
        qs = result.get("research_questions") or []
        msg = f"Refiner {phase_id}: source={src}."
        if qs:
            msg += f" {len(qs)} research question(s): " + "; ".join(
                f"({i+1}) {q[:80]}" for i, q in enumerate(qs[:5])
            )
            if len(qs) > 5:
                msg += f" … +{len(qs)-5} more"
        else:
            msg += " No research questions."
        return _ok(msg)

    @tool(
        "run_researcher",
        "Run ONLY the researcher step for a phase. Skips automatically if "
        "the refiner produced no research_questions (returns 'skipped'). "
        "Otherwise queries citations / WebFetch and writes research_notes.md. "
        "Variable cost (~30s-3min depending on question count). Usually no "
        "need to pause before this.",
        {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string", "description": "Phase to research."},
            },
            "required": ["phase_id"],
        },
    )
    async def run_researcher(args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("run_researcher called before write_skeleton.")
        phase_id = str(args.get("phase_id", "")).strip()
        if not phase_id:
            return _err("run_researcher requires phase_id.")
        if runtime.spec_manager.state.get_phase(phase_id) is None:
            return _err(f"Unknown phase: {phase_id}")
        try:
            result = await runtime.execution_loop._step_researcher(phase_id)
        except Exception as e:
            logger.exception("run_researcher: _step_researcher raised for %s", phase_id)
            return _err(f"researcher crashed: {type(e).__name__}: {e}")
        if result.get("skipped"):
            return _ok(
                f"Researcher {phase_id}: skipped — {result.get('reason', 'no questions')}."
            )
        if result.get("cached"):
            return _ok(f"Researcher {phase_id}: cached notes reused.")
        sources = result.get("sources") or []
        nq = result.get("num_questions", 0)
        return _ok(
            f"Researcher {phase_id}: answered {nq} question(s) using "
            f"{len(sources)} source(s). Notes at {result.get('path')}."
        )

    @tool(
        "run_builder",
        "Run ONLY the Builder sub-agent for a phase. This is the long step "
        "— writes code, runs tests, iterates on failures. Can take "
        "minutes to ~hour depending on phase complexity. Includes GPU "
        "provisioning if configured. Returns {status, summary, tests, "
        "outputs}. ALWAYS call request_user_approval first so the user "
        "knows what's about to run and can preview the planned files.",
        {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string", "description": "Phase to build."},
            },
            "required": ["phase_id"],
        },
    )
    async def run_builder(args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("run_builder called before write_skeleton.")
        phase_id = str(args.get("phase_id", "")).strip()
        if not phase_id:
            return _err("run_builder requires phase_id.")
        if runtime.spec_manager.state.get_phase(phase_id) is None:
            return _err(f"Unknown phase: {phase_id}")
        try:
            result = await runtime.execution_loop._step_builder(phase_id)
        except Exception as e:
            logger.exception("run_builder: _step_builder raised for %s", phase_id)
            return _err(f"builder crashed: {type(e).__name__}: {e}")
        tests = result.test_report
        tests_line = (
            f" Tests: {tests.tests_passed}/{tests.tests_run} passed"
            if tests and tests.tests_run > 0 else ""
        )
        outputs_line = (
            f" Outputs: {', '.join(o.name for o in result.outputs[:5])}"
            if result.outputs else ""
        )
        msg = (
            f"Builder {phase_id}: {result.status.value}.{tests_line}.{outputs_line}\n"
            f"Summary: {(result.summary or '')[:600]}"
        )
        if result.is_spec_issue:
            msg += "\n⚠ Builder flagged this as a SPEC ISSUE (not a code bug)."
        return _ok(msg)

    @tool(
        "run_verifier",
        "Run ONLY the Section Verifier on a phase's last Builder result. "
        "Reads the acceptance criteria and the builder's outputs/tests, "
        "judges accept/reject. On accept, marks the phase completed + "
        "propagates outputs to downstream phases. On reject, records "
        "the failure (retry budget counts) but does NOT auto-retry — YOU "
        "decide whether to call run_builder again, escalate to the user, "
        "or call pipeline_failed. ALWAYS pause after this so the user "
        "sees the verdict.",
        {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string", "description": "Phase to verify."},
            },
            "required": ["phase_id"],
        },
    )
    async def run_verifier(args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("run_verifier called before write_skeleton.")
        phase_id = str(args.get("phase_id", "")).strip()
        if not phase_id:
            return _err("run_verifier requires phase_id.")
        # Load the latest builder result from disk to feed the verifier.
        result_path = (
            runtime.workspace.phase_dir(phase_id) / "outputs" / "_result.json"
        )
        if not result_path.exists():
            return _err(
                f"run_verifier {phase_id}: no builder result on disk yet. "
                f"Run run_builder first."
            )
        try:
            raw = json.loads(result_path.read_text() or "{}")
        except Exception as e:
            return _err(f"Could not parse {result_path}: {e}")
        # Reconstruct a SubAgentResult-shaped object the verifier expects.
        from ..models.results import SubAgentResult as _SR, ResultStatus as _RS
        try:
            builder_result = _SR.model_validate({
                **raw,
                "phase_id": phase_id,
                "status": raw.get("status") or "success",
            })
        except Exception:
            # Fall back: minimal stub so verifier still runs against on-disk artifacts.
            builder_result = _SR(
                status=_RS.success,
                phase_id=phase_id,
                summary=raw.get("summary", ""),
            )
        try:
            accepted, payload = await runtime.execution_loop._step_verifier(
                phase_id, builder_result,
            )
        except Exception as e:
            logger.exception("run_verifier: _step_verifier raised for %s", phase_id)
            return _err(f"verifier crashed: {type(e).__name__}: {e}")
        feedback = payload.get("_feedback") or "(no feedback)"
        if accepted:
            return _ok(
                f"Verifier {phase_id}: ACCEPTED. Phase marked complete. "
                f"Feedback: {feedback}"
            )
        return _ok(
            f"Verifier {phase_id}: REJECTED. Phase not marked complete; "
            f"the failure is recorded against the retry budget. "
            f"Feedback: {feedback}. You can call run_builder again to retry, "
            f"or ask the user how to proceed."
        )

    @tool(
        "start_phase",
        "Convenience: run the full per-section chain (refiner → researcher "
        "→ builder → verifier) sequentially without pausing in between. Use "
        "when the user has explicitly said 'just run them all' or when "
        "running through a series of trivial sections en bloc. For most "
        "phases, prefer calling the per-step tools individually so you can "
        "pause for approval before run_builder. ALWAYS call "
        "request_user_approval before this.",
        {
            "type": "object",
            "properties": {
                "phase_id": {
                    "type": "string",
                    "description": "Phase to execute. Must be in list_pending_phases.",
                },
            },
            "required": ["phase_id"],
        },
    )
    async def start_phase(args: dict[str, Any]) -> dict[str, Any]:
        if runtime.spec_manager is None:
            return _err("start_phase called before write_skeleton.")
        phase_id = str(args.get("phase_id", "")).strip()
        if not phase_id:
            return _err("start_phase requires phase_id.")
        phase = runtime.spec_manager.state.get_phase(phase_id)
        if phase is None:
            return _err(f"Unknown phase: {phase_id}")
        # Bypass the loop's outer scheduler and drive one phase. The loop's
        # internal _execute_phase handles the per-section chain + per-attempt
        # retries via FailureHandler. We poll the phase's status afterwards.
        try:
            await runtime.execution_loop._execute_phase(phase_id)
        except Exception as e:
            logger.exception("start_phase: _execute_phase raised for %s", phase_id)
            return _err(f"start_phase {phase_id} crashed: {type(e).__name__}: {e}")

        # Re-read the phase from state to get terminal status.
        phase = runtime.spec_manager.state.get_phase(phase_id)
        status = phase.status.value if phase else "unknown"
        # Try to load the last attempt result for a summary.
        summary_text = ""
        try:
            result_path = (
                runtime.workspace.phase_dir(phase_id) / "outputs" / "_result.json"
            )
            if result_path.exists():
                raw = json.loads(result_path.read_text() or "{}")
                summary_text = str(raw.get("summary") or "")[:600]
        except Exception:
            pass
        msg = f"Phase {phase_id} → {status}."
        if summary_text:
            msg += f" Summary: {summary_text}"
        return _ok(msg)

    @tool(
        "request_user_approval",
        "Pause the pipeline and wait for the user's next chat reply. The "
        "reply is returned verbatim so YOU can decide whether it's an "
        "approval (proceed with the next action) or a question/edit request "
        "(respond, then call this again). Use this before write_skeleton "
        "(intro confirmation), after the skeleton, before each start_phase, "
        "and after each phase. Always include a clear prompt explaining what "
        "the user is approving and what comes next.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or summary the user sees in the chat panel.",
                },
                "gate_id": {
                    "type": "string",
                    "description": "Stable identifier for this gate (e.g. 'pre_run', 'post_skeleton', 'pre_phase:section_3_2'). Used for tracing and the frontend banner.",
                },
                "open_doc": {
                    "type": "string",
                    "description": "Optional workspace-relative path of a spec doc the user should read while deciding — 'spec.md' for the skeleton, 'sections/<phase_id>.md' for a section spec. When set, the frontend opens this doc in the Docs viewer alongside the approval banner. Omit when the gate isn't about reviewing a written spec (e.g. pre_run, post_verifier summary).",
                },
            },
            "required": ["prompt", "gate_id"],
        },
    )
    async def request_user_approval(args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args.get("prompt") or "")
        gate_id = str(args.get("gate_id") or "gate")
        open_doc = (str(args.get("open_doc") or "").strip()) or None
        emitter = runtime.orchestrator_agent.emitter

        # --auto bypass: when the user opted out of chat-driven gates (e.g.
        # /api/launch?skip_gates=true or `research-builder --auto`), every
        # request_user_approval auto-approves immediately with a synthetic
        # reply. GPU/cost gates live elsewhere and stay independent.
        if not runtime.config.interactive:
            if emitter:
                emitter.emit(
                    "gate_reached",
                    agent_id="orchestrator",
                    parent_id=None,
                    gate_id=gate_id,
                    prompt=prompt,
                    open_doc=open_doc,
                    context={},
                    auto=True,
                )
                emitter.emit(
                    "gate_resolved",
                    agent_id="orchestrator",
                    parent_id=None,
                    gate_id=gate_id,
                    decision="approve",
                    auto=True,
                )
            return _ok("User replied: approve (auto mode — no human in the loop)")

        if emitter:
            emitter.emit(
                "gate_reached",
                agent_id="orchestrator",
                parent_id=None,
                gate_id=gate_id,
                prompt=prompt,
                open_doc=open_doc,
                context={},
                auto=False,
            )
            # Mirror the prompt as an assistant message so it lands in the
            # chat transcript alongside the banner (banner + transcript
            # both anchor the user's attention).
            emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="assistant",
                text=prompt,
            )
        reply = await runtime.approval_queue.get()
        if emitter:
            # The gate "resolves" the moment a user reply arrives — the
            # orchestrator decides whether that reply means approval or
            # not when it reads this tool's result.
            emitter.emit(
                "gate_resolved",
                agent_id="orchestrator",
                parent_id=None,
                gate_id=gate_id,
                decision="user_replied",
                auto=False,
            )
        return _ok(f"User replied: {reply}")

    @tool(
        "pipeline_complete",
        "Terminal tool. Call when the reproduction run is finished — every "
        "phase the user approved is either completed or skipped, claims have "
        "been verified, and there's nothing left to do. Ends the run cleanly.",
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Short summary for the run log + chat (e.g. 'Reproduced 4/4 phases. Headline accuracy 95.1% vs paper's 95.2%.')",
                },
            },
            "required": ["message"],
        },
    )
    async def pipeline_complete(args: dict[str, Any]) -> dict[str, Any]:
        msg = str(args.get("message") or "Pipeline complete.")
        runtime.done = True
        runtime.success = True
        runtime.final_message = msg
        emitter = runtime.orchestrator_agent.emitter
        if emitter:
            emitter.emit(
                "run_completed",
                agent_id="orchestrator",
                parent_id=None,
                message=msg,
            )
        return _ok("Pipeline marked complete. You may stop now.")

    @tool(
        "pipeline_failed",
        "Terminal tool. Call when the run cannot continue — for example a "
        "phase failed irrecoverably and downstream phases depend on it, or "
        "the user explicitly told you to abort. Ends the run with failure.",
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Reason the pipeline is failing (one or two sentences).",
                },
            },
            "required": ["message"],
        },
    )
    async def pipeline_failed(args: dict[str, Any]) -> dict[str, Any]:
        msg = str(args.get("message") or "Pipeline failed.")
        runtime.done = True
        runtime.success = False
        runtime.final_message = msg
        emitter = runtime.orchestrator_agent.emitter
        if emitter:
            emitter.emit(
                "run_failed",
                agent_id="orchestrator",
                parent_id=None,
                message=msg,
            )
        return _ok("Pipeline marked failed. You may stop now.")

    return create_sdk_mcp_server(
        # Short server label; surfaces as the middle segment of the SDK's
        # ``mcp__<server>__<tool>`` routing prefix. Keep it minimal — the
        # frontend strips the prefix for display anyway.
        name="orchestrator",
        version="1.0.0",
        tools=[
            write_skeleton,
            author_section_specs,
            critique_section_specs,
            extract_claims_ledger,
            list_pending_phases,
            run_refiner,
            run_researcher,
            run_builder,
            run_verifier,
            start_phase,
            request_user_approval,
            pipeline_complete,
            pipeline_failed,
        ],
    )


# Fully-qualified tool names the SDK needs in ``allowed_tools``. The
# ``mcp__<server>__<tool>`` prefix is the SDK's tool-routing convention —
# we don't control that part. The server segment matches the ``name=``
# above.
ORCHESTRATOR_TOOL_NAMES = [
    "mcp__orchestrator__write_skeleton",
    "mcp__orchestrator__author_section_specs",
    "mcp__orchestrator__critique_section_specs",
    "mcp__orchestrator__extract_claims_ledger",
    "mcp__orchestrator__list_pending_phases",
    "mcp__orchestrator__run_refiner",
    "mcp__orchestrator__run_researcher",
    "mcp__orchestrator__run_builder",
    "mcp__orchestrator__run_verifier",
    "mcp__orchestrator__start_phase",
    "mcp__orchestrator__request_user_approval",
    "mcp__orchestrator__pipeline_complete",
    "mcp__orchestrator__pipeline_failed",
]


# ─── helpers ───────────────────────────────────────────────────────────────


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}
