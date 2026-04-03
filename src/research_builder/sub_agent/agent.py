"""Sub-agent: executes a single phase using the Claude Agent SDK (spec_v4 §5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from ..config import Config
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
from .tools import BUILTIN_TOOLS, CUSTOM_TOOL_NAMES, create_phase_tools

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
    ) -> None:
        self.config = config
        self.sub_spec = sub_spec
        self.work_dir = work_dir
        self.retry_context = retry_context
        self.result_path = work_dir / "outputs" / "_result.json"

    async def run(self) -> SubAgentResult:
        """Execute the phase and return a structured result."""
        phase_id = self.sub_spec.phase.phase_id
        logger.info("SubAgent starting phase=%s in %s", phase_id, self.work_dir)

        system_prompt = build_system_prompt(self.sub_spec, self.retry_context)

        # Create custom MCP tools
        paper_path = Path(self.sub_spec.paper_path) if self.sub_spec.paper_path else Path("paper.pdf")
        phase_tools = create_phase_tools(paper_path, self.result_path)

        # Build agent options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            cwd=str(self.work_dir),
            allowed_tools=BUILTIN_TOOLS + CUSTOM_TOOL_NAMES,
            mcp_servers={"phase_tools": phase_tools},
            permission_mode="bypassPermissions",
            model=self.config.model,
            max_turns=self.sub_spec.phase.max_debug_attempts * 5,
        )

        # User message to kick off the agent
        user_message = (
            f"Implement the **{self.sub_spec.phase.title}** phase. "
            f"Your working directory is `{self.work_dir}`. "
            f"Write source code under `src/` and output artifacts under `outputs/`. "
            f"When finished, call `report_result` with your results."
        )

        try:
            async for message in query(prompt=user_message, options=options):
                if isinstance(message, ResultMessage):
                    logger.info(
                        "SubAgent query complete: phase=%s, turns=%d, cost=$%s",
                        phase_id,
                        message.num_turns,
                        message.total_cost_usd,
                    )

            # Read the structured result written by report_result tool
            if self.result_path.exists():
                raw = json.loads(self.result_path.read_text())
                return _parse_result(phase_id, raw)
            else:
                logger.warning("SubAgent for phase=%s did not call report_result", phase_id)
                return SubAgentResult(
                    status=ResultStatus.failure,
                    phase_id=phase_id,
                    summary="Agent completed without calling report_result",
                )

        except Exception as e:
            logger.exception("SubAgent for phase=%s crashed", phase_id)
            return SubAgentResult(
                status=ResultStatus.failure,
                phase_id=phase_id,
                summary=f"Agent crashed: {type(e).__name__}: {e}",
                diagnostics={"error": str(e), "type": type(e).__name__},
            )


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
