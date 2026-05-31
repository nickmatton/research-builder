"""Sub-agent: executes a single phase using the Claude Agent SDK (spec_v4 §5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claude_agent_sdk import (
    StreamEvent,
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
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

from ..access_tools import ACCESS_TOOL_NAMES, ApprovalCallback, create_access_server
from ..config import Config
from ..events import get_emitter, maybe_emit_paper_read
from ..events.emitter import capture_file_before, emit_file_write
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
        access_approval_callback: ApprovalCallback | None = None,
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
        # Optional. When None we fall back to click.prompt in the access
        # tool — fine for direct CLI runs, but the orchestrator passes a
        # chat-routing callback so web-spawned sessions don't hang on stdin.
        self.access_approval_callback = access_approval_callback

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
                "Connection details live in `.cloud/env`. "
                "**All training, fine-tuning, and gradient-based fitting MUST go through "
                '`bash remote_run.sh "..."`** — never run training directly under local '
                "`python`/`uv run`, even for smoke runs of a few steps, even for small models. "
                'Example: `bash remote_run.sh "python -m src.train --max-steps 100"`. '
                "Inference / eval is GPU-only when the model is large; small-model "
                "inference and unit tests of non-training code may use local Bash. "
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
        # Ad-hoc filesystem access (paths not in --allow-dir). Per-phase cache
        # so a path approved here doesn't carry across to the next phase. The
        # callback (when supplied by the orchestrator) routes through the
        # chat surface — see make_chat_approval_callback in access_tools.
        access_tools = create_access_server(
            self.config,
            cwd=self.work_dir,
            approval_callback=self.access_approval_callback,
        )
        allowed_custom = list(CUSTOM_TOOL_NAMES) + ACCESS_TOOL_NAMES
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
            # Extend the sandbox to user-allowlisted dirs (--allow-dir). The
            # mcp__access__read_outside_workspace tool covers ad-hoc paths not
            # in this list via an interactive approval prompt.
            add_dirs=[str(p) for p in self.config.extra_allowed_dirs],
            allowed_tools=BUILTIN_TOOLS + allowed_custom,
            mcp_servers={"phase": phase_tools, "access": access_tools},
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
            f"`mcp__phase__report_result` tool with your status, summary, "
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

        # Heartbeat: sub-agent sessions can be 50+ min long. Print a one-line
        # "still alive" update every 30s with elapsed + msg-count + last
        # message type. Bypasses the logger so it shows regardless of
        # --verbose. Cancelled in the finally block below.
        import asyncio as _asyncio
        _hb_start = _asyncio.get_event_loop().time()

        # Live stream-state diagnostics. Tracks which content block the
        # model currently has open and how many of each delta type arrived
        # since the last heartbeat tick. Lets the heartbeat tell a true
        # stall from a long phase of silent activity (input_json_delta for
        # a giant tool call, signature_delta after an extended-thinking
        # block, etc.). messages_received intentionally filters out
        # StreamEvents, which is why the old heartbeat could show "9 msgs"
        # unchanged for minutes while the wire was fully active.
        stream_state: dict = {
            "open_type":  None,
            "open_name":  None,
            "open_id":    None,
            "open_since": 0.0,
            "deltas":     {},
        }

        async def _heartbeat():
            interval = 30.0
            while True:
                await _asyncio.sleep(interval)
                elapsed = _asyncio.get_event_loop().time() - _hb_start
                deltas_this_tick = stream_state["deltas"]
                stream_state["deltas"] = {}
                open_t = stream_state["open_type"]
                if open_t:
                    held_s = _asyncio.get_event_loop().time() - stream_state["open_since"]
                    name = stream_state["open_name"]
                    block_label = f"{open_t}:{name}" if name else open_t
                    block_str = f"open {block_label} ({held_s:.0f}s)"
                else:
                    block_str = "no block open"
                if deltas_this_tick:
                    delta_str = ", ".join(
                        f"+{n} {k}"
                        for k, n in sorted(deltas_this_tick.items(), key=lambda kv: -kv[1])
                    )
                else:
                    delta_str = "no stream events"
                last = messages_received[-1] if messages_received else "(no messages yet)"
                print(
                    f"  💓 [phase:{phase_id}] {elapsed:.0f}s · "
                    f"{block_str} · last {interval:.0f}s: {delta_str} · "
                    f"{len(messages_received)} non-stream msgs (last: {last})",
                    flush=True,
                )
                # Mirror the heartbeat as a structured event so the web UI
                # can render a per-agent "latest tick" strip. Filtered out
                # of the activity firehose by type — see frontend
                # ActivityView RENDER_TYPES.
                if emitter:
                    emitter.emit(
                        "heartbeat",
                        agent_id=agent_id,
                        parent_id="orchestrator",
                        elapsed_s=elapsed,
                        interval_s=interval,
                        open_block=block_str if open_t else None,
                        deltas=deltas_this_tick,
                        last_msg_type=last,
                        msgs_count=len(messages_received),
                    )

        heartbeat_task = _asyncio.create_task(_heartbeat())
        # Track whether we ever saw a ResultMessage. The bundled `claude` CLI
        # sometimes raises ``Exception("Claude Code returned an error result:
        # success")`` *after* the ResultMessage was already delivered — a
        # cleanup-path quirk of the SDK transport. If the sub-agent already
        # called report_result (result file on disk) OR we received the
        # terminal ResultMessage, the work is real; we swallow the late
        # exception, the same way the orchestrator does in _consume_stream.
        result_message_seen = False
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
                    ev_type = evt.get("type")
                    if ev_type == "content_block_start":
                        cb = evt.get("content_block") or {}
                        cb_type = cb.get("type")
                        stream_state["open_type"]  = cb_type
                        stream_state["open_name"]  = cb.get("name")
                        stream_state["open_id"]    = cb.get("id")
                        stream_state["open_since"] = _asyncio.get_event_loop().time()
                        # Operator-visible early announce — Claude Code shows
                        # the tool name the instant content_block_start fires
                        # rather than waiting for the full AssistantMessage.
                        # Full input details (file_path, command, …) still
                        # surface from the AssistantMessage path once
                        # input_json_delta is done.
                        if cb_type == "tool_use":
                            name = cb.get("name") or "?"
                            self.on_activity("tool", f"{name} (input streaming…)")
                    elif ev_type == "content_block_stop":
                        stream_state["open_type"]  = None
                        stream_state["open_name"]  = None
                        stream_state["open_id"]    = None
                    elif ev_type == "content_block_delta":
                        delta = evt.get("delta") or {}
                        dt = delta.get("type") or "?"
                        stream_state["deltas"][dt] = stream_state["deltas"].get(dt, 0) + 1
                        if dt == "thinking_delta":
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
                            # Update the terminal status line so the operator
                            # sees activity, but do NOT emit an agent_thinking
                            # event here. The same content already streamed as
                            # ``thinking_delta`` StreamEvents above; emitting
                            # again would produce a duplicate in the Activity
                            # tab and per-phase chat pane.
                            if block.thinking.strip():
                                snippet = block.thinking.strip().replace("\n", " ")
                                if len(snippet) > 200:
                                    snippet = snippet[:200] + "..."
                                self.on_activity("thinking", snippet)
                            continue
                        if isinstance(block, TextBlock) and block.text.strip():
                            # TextBlock is the assistant's user-facing reply,
                            # distinct from internal thinking. Surface ONLY
                            # via agent_message so subscribers don't double-
                            # render it as "thinking" content. The terminal
                            # status line + the chat pane both see this via
                            # the emitter — no need to also push through
                            # on_activity, which would route TextBlock into
                            # the thinking channel.
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
                            # Snapshot pre-write content so the UI can render
                            # before/after diffs when the user clicks the
                            # tool node. No-ops for non-write tools.
                            capture_file_before(
                                emitter,
                                process_id=block.id,
                                tool_name=block.name,
                                file_path=block.input.get("file_path"),
                                cwd=self.work_dir,
                            )
                            maybe_emit_paper_read(
                                emitter,
                                agent_id=agent_id,
                                parent_id="orchestrator",
                                tool_name=block.name,
                                tool_input=block.input or {},
                                paper_path=paper_path,
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
                                    # Pair with capture_file_before above: emit
                                    # the before/after snapshot for write tools.
                                    # No-ops for non-write tools.
                                    emit_file_write(
                                        emitter,
                                        agent_id=agent_id,
                                        parent_id="orchestrator",
                                        process_id=block.tool_use_id,
                                        is_error=is_err,
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

                elif isinstance(message, RateLimitEvent):
                    # Reuse the orchestrator's structured rate-limit logger.
                    from ..orchestrator.agent import _log_rate_limit
                    _log_rate_limit(
                        f"sub_agent[{phase_id}]",
                        phase_id,
                        getattr(message, "rate_limit_info", None),
                        logger,
                        emitter,
                        trace=None,
                    )

                elif isinstance(message, ResultMessage):
                    result_message_seen = True
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
            # The bundled `claude` CLI can raise ``Exception("Claude Code
            # returned an error result: success")`` AFTER it has already
            # delivered a ResultMessage and the sub-agent has already
            # written its result file via report_result. In that case the
            # work is real — fall through to the normal result-parse path
            # instead of marking the phase as crashed. Same pattern the
            # orchestrator uses in its ``_consume_stream``.
            benign_tail = (result_message_seen or self.result_path.exists()) and (
                "error result: success" in str(e).lower()
            )
            if benign_tail:
                msg_trace = " → ".join(messages_received) if messages_received else "(none)"
                logger.warning(
                    "sub_agent[%s]: CLI emitted late error after delivering result (%s). "
                    "Treating as success; result file %s.",
                    phase_id, e, "exists" if self.result_path.exists() else "missing",
                )
                if self.result_path.exists():
                    raw = json.loads(self.result_path.read_text())
                    result = _parse_result(phase_id, raw)
                else:
                    result = _build_fallback_result(phase_id, self.work_dir)
                result.cost_usd = phase_cost
                return result

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
                # Structured companion to the system-role message above. The
                # Trace view renders this with stderr + last-N tool calls
                # inline; the agent_message stays for legacy tailers.
                emitter.emit(
                    "agent_crashed",
                    agent_id=agent_id,
                    parent_id="orchestrator",
                    error_type=type(e).__name__,
                    error=str(e),
                    messages_received=messages_received[-15:],
                    stderr_tail=stderr_lines[-30:],
                    turns_completed=turn_count,
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
        finally:
            # Cancel the heartbeat task (covers both success-return and
            # exception paths). Print a final summary if the run was nontrivial.
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (_asyncio.CancelledError, Exception):
                pass
            _hb_elapsed = _asyncio.get_event_loop().time() - _hb_start
            if _hb_elapsed >= 5.0:
                print(
                    f"  ✓ [phase:{phase_id}] done in {_hb_elapsed:.1f}s "
                    f"({len(messages_received)} msgs)",
                    flush=True,
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
    short = name.replace("mcp__phase__", "")

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
