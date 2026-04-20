"""Sub-agent: executes a single phase using the Claude Agent SDK (spec_v4 §5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claude_agent_sdk import (
    StreamEvent,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from typing import TYPE_CHECKING, Callable

from ..config import Config
from ..events import get_emitter
from ..models.context import RetryContext, SubSpec
from ..models.results import (
    ResultStatus,
    SubAgentResult,
    TestReport,
    TestResult,
    TestStatus,
)
from ..models.spec import Artifact
from .prompts import build_system_prompt
from .tools import BUILTIN_TOOLS, COMPUTE_TOOL_NAME, CUSTOM_TOOL_NAMES, create_phase_tools

if TYPE_CHECKING:
    from ..cloud import CloudProvisioner, ComputeHandle

logger = logging.getLogger(__name__)


class SubAgent:
    """Runs a single phase by driving a Claude Agent SDK session.

    The sub-agent:
    1. Builds a system prompt from the sub-spec and retry context
    2. Creates custom MCP tools for paper access and result reporting
    3. Runs query() with built-in + custom tools
    4. Reads the structured result from the report_result output file
    5. Returns a SubAgentResult
    """

    def __init__(
        self,
        config: Config,
        sub_spec: SubSpec,
        work_dir: Path,
        retry_context: RetryContext | None = None,
        on_activity: Callable[[str, str], None] | None = None,
        extra_user_messages: list[str] | None = None,
        cloud_provisioner: "CloudProvisioner | None" = None,
        compute_handle: "ComputeHandle | None" = None,
    ) -> None:
        self.config = config
        self.sub_spec = sub_spec
        self.work_dir = work_dir
        self.retry_context = retry_context
        self.result_path = work_dir / "outputs" / "_result.json"
        self.on_activity = on_activity or (lambda kind, detail: None)
        self.extra_user_messages = extra_user_messages or []
        # If both are present, the sub-agent gets a request_compute MCP tool
        # that calls back into the provisioner to swap the machine in place.
        self.cloud_provisioner = cloud_provisioner
        self.compute_handle = compute_handle

    async def run(self) -> SubAgentResult:
        """Execute the phase and return a structured result."""
        phase_id = self.sub_spec.phase.phase_id
        logger.info("SubAgent starting phase=%s in %s", phase_id, self.work_dir)
        emitter = get_emitter()
        agent_id = f"phase:{phase_id}"

        system_prompt = build_system_prompt(self.sub_spec, self.retry_context)

        # If the orchestrator provisioned a cloud GPU machine for this phase,
        # tell the agent how to use it. The wrapper script + .cloud/env are
        # already on disk in work_dir.
        compute_upgrade_fn = None
        if (self.work_dir / "remote_run.sh").exists():
            instance_type = self.compute_handle.machine.instance_type if self.compute_handle else "GPU"
            ledger = self.cloud_provisioner.ledger if self.cloud_provisioner else None
            budget_line = (
                f" Per-run GPU spend cap: ${ledger.cap_usd:.2f} (currently ${ledger.projected_total():.2f} reserved)."
                if ledger else ""
            )
            system_prompt += (
                "\n\n## Remote GPU machine\n"
                f"A Lambda Cloud {instance_type} GPU instance has been provisioned for this phase. "
                "Connection details live in `.cloud/env`. To run any GPU workload "
                '(training, large eval, etc.), invoke: `bash remote_run.sh "python train.py --epochs 10"`. '
                "The wrapper rsyncs the work_dir to the remote box, executes the command "
                "there, and rsyncs results back into your work_dir. Do NOT try to install "
                "CUDA / GPU drivers locally — only the remote machine has a GPU. "
                "Local Bash is still fine for editing code, running unit tests on CPU, "
                "and inspecting outputs after they've been rsynced back."
                f"\n\nIf the machine is too small (OOM, too slow) or you need more wall-clock, "
                f"call `request_compute` with a desired `instance_type` and/or `additional_hours` "
                f"plus a one-line `reason`. The harness swaps the box in place; you re-invoke "
                f"`bash remote_run.sh \"...\"` to use it.{budget_line} "
                f"Upgrades that would exceed the cap require operator approval, so only "
                f"request what you genuinely need."
            )
            # Wire the upgrade hook IFF we have both provisioner and handle.
            if self.cloud_provisioner is not None and self.compute_handle is not None:
                provisioner = self.cloud_provisioner
                handle = self.compute_handle
                async def _upgrade(it: str | None, hrs: float | None, reason: str) -> tuple[bool, str]:
                    return await provisioner.upgrade(
                        handle,
                        new_instance_type=it,
                        additional_hours=hrs,
                        reason=reason,
                    )
                compute_upgrade_fn = _upgrade

        # Create custom MCP tools
        paper_path = Path(self.sub_spec.paper_path) if self.sub_spec.paper_path else Path("paper.pdf")
        phase_tools = create_phase_tools(paper_path, self.result_path, compute_upgrade=compute_upgrade_fn, spec_dir=self.config.spec_dir)
        allowed_custom = list(CUSTOM_TOOL_NAMES)
        if compute_upgrade_fn is not None:
            allowed_custom.append(COMPUTE_TOOL_NAME)

        # Track cost from ResultMessage
        phase_cost: float = 0.0

        # Capture stderr for diagnostics
        stderr_lines: list[str] = []

        def capture_stderr(line: str) -> None:
            stderr_lines.append(line)

        # Build agent options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            cwd=str(self.work_dir),
            allowed_tools=BUILTIN_TOOLS + allowed_custom,
            mcp_servers={"phase_tools": phase_tools},
            permission_mode="bypassPermissions",
            model=self.config.model,
            max_turns=self.sub_spec.phase.max_debug_attempts * 5,
            stderr=capture_stderr,
            # Extended thinking — sub-agents are writing real code (data
            # pipelines, model architectures, training loops). Chain-of-
            # thought materially improves correctness and the ThinkingBlock
            # chunks stream live into the per-phase chat pane.
            thinking={"type": "enabled", "budget_tokens": 12000},
            include_partial_messages=True,
        )

        # User message to kick off the agent
        user_message = (
            f"Implement the **{self.sub_spec.phase.title}** phase. "
            f"Your working directory is `{self.work_dir}`. "
            f"Write source code under `src/` and output artifacts under `outputs/`. "
            f"\n\nIMPORTANT: When you are completely finished, you MUST call the "
            f"`mcp__phase_tools__report_result` tool with your status, summary, "
            f"outputs list, and test report. This is how your results are recorded. "
            f"Do not end your session without calling this tool."
        )

        # Operator messages queued via the agent-terminal chat pane while the
        # previous attempt was running. Folded in here so the agent sees them
        # as additional user direction at the top of this attempt.
        if self.extra_user_messages:
            joined = "\n".join(f"- {m}" for m in self.extra_user_messages)
            user_message += (
                "\n\n## Operator messages\n"
                "The human operator sent the following messages since the last "
                "attempt. Treat them as additional user instructions and "
                "address them as part of this attempt:\n"
                f"{joined}"
            )

        # Persist a context snapshot of exactly what this sub-agent saw, so
        # the operator can browse it via the file tree (context/<phase>/try_N/).
        # Best-effort — never fail the run because we couldn't write a log.
        try:
            self._write_context_snapshot(system_prompt, user_message)
        except Exception:
            logger.exception("Failed to write context snapshot for phase=%s", phase_id)

        # Surface both the actual system_prompt and the kickoff message in
        # the per-phase chat pane so the operator can see how the sub-agent
        # was instructed. Without this the chat pane for sub-agents is empty.
        if emitter:
            emitter.emit(
                "agent_message",
                agent_id=agent_id,
                parent_id="orchestrator",
                role="system",
                text=system_prompt[:4000],
            )
            emitter.emit(
                "agent_message",
                agent_id=agent_id,
                parent_id="orchestrator",
                role="system",
                text=user_message[:2000],
            )

        messages_received: list[str] = []  # trace of message types for crash diagnostics
        try:
            turn_count = 0
            async for message in query(prompt=user_message, options=options):
                msg_type = type(message).__name__
                # Only log non-StreamEvent types to avoid flooding the trace
                if not isinstance(message, StreamEvent):
                    messages_received.append(msg_type)
                    logger.debug("sub_agent[%s]: received %s (#%d)", phase_id, msg_type, len(messages_received))
                if isinstance(message, StreamEvent):
                    evt = message.event
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "thinking_delta":
                            text = delta.get("thinking", "")
                            if text.strip():
                                snippet = text.strip().replace("\n", " ")[:200]
                                self.on_activity("thinking", snippet)
                                if emitter:
                                    emitter.emit(
                                        "agent_thinking",
                                        agent_id=agent_id,
                                        parent_id="orchestrator",
                                        text=text[:500],
                                    )
                    continue
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            # Stream extended-thinking chunks live into the
                            # per-phase chat pane and activity feed.
                            if block.thinking.strip():
                                snippet = block.thinking.strip().replace("\n", " ")
                                if len(snippet) > 200:
                                    snippet = snippet[:200] + "..."
                                self.on_activity("thinking", snippet)
                                if emitter:
                                    emitter.emit(
                                        "agent_thinking",
                                        agent_id=agent_id,
                                        parent_id="orchestrator",
                                        text=block.thinking[:1000],
                                    )
                            continue
                        if isinstance(block, TextBlock) and block.text.strip():
                            # Show a truncated snippet of reasoning
                            snippet = block.text.strip().replace("\n", " ")
                            if len(snippet) > 200:
                                snippet = snippet[:200] + "..."
                            self.on_activity("thinking", snippet)
                            # Also surface as a chat message so the per-phase
                            # chat pane has an assistant trail.
                            if emitter:
                                emitter.emit(
                                    "agent_message",
                                    agent_id=agent_id,
                                    parent_id="orchestrator",
                                    role="assistant",
                                    text=block.text[:2000],
                                )
                        elif isinstance(block, ToolUseBlock):
                            self.on_activity("tool", _format_tool_use(block))
                            if emitter:
                                emitter.emit(
                                    "process_started",
                                    agent_id=agent_id,
                                    parent_id="orchestrator",
                                    process_id=block.id,
                                    tool_name=block.name,
                                    summary=_format_tool_use(block),
                                    command=block.input.get("command", "")[:500] if block.name == "Bash" else None,
                                    file_path=block.input.get("file_path") if block.name in ("Write", "Read", "Edit") else None,
                                )
                        else:
                            logger.warning(
                                "sub_agent[%s]: unhandled assistant block type=%s",
                                phase_id, type(block).__name__,
                            )
                            self.on_activity("tool", f"[unhandled {type(block).__name__}]")
                    turn_count += 1

                elif isinstance(message, UserMessage):
                    # Tool results live here. Errors are surfaced in the
                    # chat / activity panes so failures are visible. All
                    # results (including success) are emitted as
                    # process_result events for the Processes pane.
                    content = getattr(message, "content", None)
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                text = _stringify_tool_result(block.content)
                                is_err = bool(getattr(block, "is_error", False))
                                # Emit process_result for the Processes pane
                                if emitter:
                                    emitter.emit(
                                        "process_result",
                                        agent_id=agent_id,
                                        parent_id="orchestrator",
                                        process_id=block.tool_use_id,
                                        is_error=is_err,
                                        output=(text[:2000] if text else ""),
                                    )
                                # Surface errors in chat/activity (existing behavior)
                                if is_err and text:
                                    summary = "ERR " + text[:300].replace("\n", " ⏎ ")
                                    self.on_activity("tool", summary)
                                    if emitter:
                                        emitter.emit(
                                            "agent_tool",
                                            agent_id=agent_id,
                                            parent_id="orchestrator",
                                            summary=summary,
                                        )

                elif isinstance(message, SystemMessage):
                    logger.debug(
                        "sub_agent[%s]: SystemMessage subtype=%s",
                        phase_id, getattr(message, "subtype", "?"),
                    )

                elif isinstance(message, ResultMessage):
                    phase_cost = message.total_cost_usd or 0.0
                    cost = f"${phase_cost:.2f}" if phase_cost else "n/a"
                    self.on_activity(
                        "done",
                        f"{message.num_turns} turns, cost {cost}",
                    )
                    logger.info(
                        "SubAgent query complete: phase=%s, turns=%d, cost=$%s",
                        phase_id,
                        message.num_turns,
                        message.total_cost_usd,
                    )

            # Read the structured result written by report_result tool
            if self.result_path.exists():
                raw = json.loads(self.result_path.read_text())
                result = _parse_result(phase_id, raw)
            else:
                # Fallback: auto-construct result from outputs directory
                logger.warning("SubAgent for phase=%s did not call report_result, scanning outputs", phase_id)
                result = _build_fallback_result(phase_id, self.work_dir)
            result.cost_usd = phase_cost
            return result

        except Exception as e:
            msg_trace = " → ".join(messages_received) if messages_received else "(no messages received)"
            logger.error(
                "SubAgent for phase=%s crashed: %s\n"
                "  error_type: %s\n"
                "  work_dir: %s\n"
                "  system_prompt: %d chars\n"
                "  user_message: %d chars\n"
                "  max_turns: %d\n"
                "  model: %s\n"
                "  turns_completed: %d\n"
                "  messages_received: %s\n"
                "  result_file_exists: %s",
                phase_id, e,
                type(e).__name__,
                self.work_dir,
                len(system_prompt),
                len(user_message),
                self.sub_spec.phase.max_debug_attempts * 5,
                self.config.model,
                turn_count,
                msg_trace,
                self.result_path.exists(),
            )
            if stderr_lines:
                logger.error("Sub-agent stderr (%d lines):\n%s", len(stderr_lines), "\n".join(stderr_lines[-30:]))
            else:
                logger.error("Sub-agent exited with no stderr")
            logger.exception("Full traceback:")

            # Surface the crash in the TUI chat pane so it's visible
            if emitter:
                emitter.emit(
                    "agent_message",
                    agent_id=agent_id,
                    parent_id="orchestrator",
                    role="system",
                    text=(
                        f"Sub-agent crashed: {type(e).__name__}: {e}\n"
                        f"Messages received: {msg_trace}\n"
                        f"Turns completed: {turn_count}\n"
                        f"Check Output tab / run.log for full details."
                    ),
                )

            return SubAgentResult(
                status=ResultStatus.failure,
                phase_id=phase_id,
                summary=f"Agent crashed: {type(e).__name__}: {e}",
                diagnostics={
                    "error": str(e),
                    "type": type(e).__name__,
                    "messages_received": messages_received,
                    "turns_completed": turn_count,
                    "stderr_tail": stderr_lines[-10:] if stderr_lines else [],
                },
            )


    def _write_context_snapshot(self, system_prompt: str, user_message: str) -> None:
        """Persist what this sub-agent saw to context/<phase>/.

        Lets the operator browse exact prompts/spec slices via the file tree
        instead of having to scroll the chat pane. Best-effort.
        """
        phase_id = self.sub_spec.phase.phase_id
        ctx_dir = self.config.context_dir / phase_id
        ctx_dir.mkdir(parents=True, exist_ok=True)
        (ctx_dir / "system_prompt.md").write_text(system_prompt)
        (ctx_dir / "kickoff.md").write_text(user_message)
        (ctx_dir / "sub_spec.md").write_text(_render_sub_spec_markdown(self.sub_spec))


def _render_sub_spec_markdown(sub_spec) -> str:
    """Pretty-print a SubSpec to standalone markdown for the context snapshot."""
    lines: list[str] = [
        f"# SubSpec: {sub_spec.phase.title} (`{sub_spec.phase.phase_id}`)",
        "",
        f"- Status: `{sub_spec.phase.status.value}`",
        f"- Max debug attempts: `{sub_spec.phase.max_debug_attempts}`",
        f"- Paper: `{sub_spec.paper_path}`",
        "",
    ]
    if sub_spec.phase.inputs:
        lines.append("## Inputs")
        for a in sub_spec.phase.inputs:
            lines.append(f"- **{a.name}**: `{a.file_path}`")
        lines.append("")
    if sub_spec.phase.outputs:
        lines.append("## Expected Outputs")
        for a in sub_spec.phase.outputs:
            lines.append(f"- **{a.name}**: `{a.file_path}`")
        lines.append("")
    if sub_spec.adjacent_phases:
        lines.append("## Adjacent Phases")
        for adj in sub_spec.adjacent_phases:
            lines.append(f"### {adj.title} (`{adj.phase_id}`)")
            if adj.inputs:
                lines.append("- Consumes: " + ", ".join(f"`{a.name}`" for a in adj.inputs))
            if adj.outputs:
                lines.append("- Produces: " + ", ".join(f"`{a.name}`" for a in adj.outputs))
            lines.append("")
    if sub_spec.open_questions:
        lines.append("## Open Questions")
        for q in sub_spec.open_questions:
            lines.append(f"- {q}")
        lines.append("")
    if sub_spec.spec_markdown:
        lines.append("## Detailed Spec (excerpt from spec.md)")
        lines.append("")
        lines.append(sub_spec.spec_markdown)
    return "\n".join(lines)


def _parse_result(phase_id: str, raw: dict) -> SubAgentResult:
    """Parse the raw dict from report_result into a SubAgentResult."""
    status_str = raw.get("status", "failure")
    status = ResultStatus.success if status_str == "success" else ResultStatus.failure

    outputs = [
        Artifact(name=o.get("name", ""), file_path=o.get("file_path", ""))
        for o in raw.get("outputs", [])
    ]

    raw_report = raw.get("test_report", {})
    test_details = [
        TestResult(
            test_name=t.get("test_name", ""),
            status=TestStatus(t.get("status", "error")),
            description=t.get("description", ""),
            message=t.get("message"),
        )
        for t in raw_report.get("test_details", [])
    ]
    test_report = TestReport(
        tests_run=raw_report.get("tests_run", len(test_details)),
        tests_passed=raw_report.get("tests_passed", sum(1 for t in test_details if t.status == TestStatus.passed)),
        tests_failed=raw_report.get("tests_failed", sum(1 for t in test_details if t.status != TestStatus.passed)),
        test_details=test_details,
    )

    return SubAgentResult(
        status=status,
        phase_id=phase_id,
        outputs=outputs,
        summary=raw.get("summary", ""),
        test_report=test_report,
        attempts_used=raw.get("attempts_used", 1),
        is_spec_issue=raw.get("is_spec_issue", False),
        diagnostics=raw.get("diagnostics"),
    )


def _build_fallback_result(phase_id: str, work_dir: Path) -> SubAgentResult:
    """Auto-construct a result by scanning the outputs directory.

    Used when the sub-agent completes without calling report_result.
    If artifacts exist in outputs/, treat it as a success.
    """
    outputs_dir = work_dir / "outputs"
    artifacts = []
    if outputs_dir.exists():
        for f in sorted(outputs_dir.iterdir()):
            if f.is_file() and f.name != "_result.json":
                artifacts.append(Artifact(name=f.stem, file_path=str(f.relative_to(work_dir.parent.parent))))

    if artifacts:
        return SubAgentResult(
            status=ResultStatus.success,
            phase_id=phase_id,
            outputs=artifacts,
            summary=f"Phase completed (auto-detected {len(artifacts)} output artifacts, report_result not called)",
        )
    else:
        return SubAgentResult(
            status=ResultStatus.failure,
            phase_id=phase_id,
            summary="Agent completed without calling report_result and no output artifacts found",
        )


def _stringify_tool_result(content) -> str:
    """Flatten the polymorphic ToolResultBlock.content into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if t:
                    parts.append(str(t))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _format_tool_use(block: ToolUseBlock) -> str:
    """Format a tool use block into a concise one-liner."""
    name = block.name
    inp = block.input or {}

    # Strip MCP prefixes for readability
    short = name.replace("mcp__phase_tools__", "")

    if name == "Bash":
        cmd = inp.get("command", "")
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"Bash: {cmd}"
    elif name in ("Write", "Edit"):
        path = inp.get("file_path", "")
        return f"{name}: {path}"
    elif name == "Read":
        path = inp.get("file_path", "")
        return f"Read: {path}"
    elif name in ("Glob", "Grep"):
        pattern = inp.get("pattern", "")
        return f"{name}: {pattern}"
    else:
        return short
