"""Sub-agent: executes a single phase of the reproduction pipeline (spec_v4 §5)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from ..llm.client import LLMClient, ToolExit
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
from .tools import ALL_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


class SubAgent:
    """Runs a single phase by driving an LLM tool-use loop.

    The sub-agent:
    1. Builds a system prompt from the sub-spec and retry context
    2. Runs the LLM tool loop (plan → implement → test → debug → report)
    3. Parses the structured result from the report_result tool call
    4. Returns a SubAgentResult
    """

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        sub_spec: SubSpec,
        work_dir: Path,
        retry_context: RetryContext | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.sub_spec = sub_spec
        self.work_dir = work_dir
        self.retry_context = retry_context

        self.tool_executor = ToolExecutor(
            work_dir=work_dir,
            paper_path=Path(sub_spec.paper_path) if sub_spec.paper_path else Path("paper.pdf"),
            bash_timeout=config.bash_timeout,
        )

    async def run(self) -> SubAgentResult:
        """Execute the phase and return a structured result."""
        phase_id = self.sub_spec.phase.phase_id
        logger.info("SubAgent starting phase=%s in %s", phase_id, self.work_dir)

        system_prompt = build_system_prompt(self.sub_spec, self.retry_context)

        # Initial user message kicks off the agent
        user_message = (
            f"Implement the **{self.sub_spec.phase.title}** phase. "
            f"Your working directory is `{self.work_dir}`. "
            f"Write source code under `src/` and output artifacts under `outputs/`. "
            f"When finished, call `report_result` with your results."
        )

        messages = [{"role": "user", "content": user_message}]

        try:
            response, history = await self.llm_client.run_tool_loop(
                messages=messages,
                system=system_prompt,
                tools=ALL_TOOLS,
                execute_tool=self.tool_executor.execute,
                max_iterations=self.sub_spec.phase.max_debug_attempts * 5,
            )

            # If we got here without ToolExit, the model stopped calling tools
            # without calling report_result. Extract what we can from the response.
            logger.warning(
                "SubAgent for phase=%s ended without calling report_result (stop_reason=%s)",
                phase_id,
                response.stop_reason,
            )
            return SubAgentResult(
                status=ResultStatus.failure,
                phase_id=phase_id,
                summary=f"Agent stopped without reporting result (stop_reason={response.stop_reason})",
                diagnostics={"stop_reason": response.stop_reason},
            )

        except ToolExit as e:
            return _parse_tool_exit(phase_id, e.result)

        except Exception as e:
            logger.exception("SubAgent for phase=%s crashed", phase_id)
            return SubAgentResult(
                status=ResultStatus.failure,
                phase_id=phase_id,
                summary=f"Agent crashed: {type(e).__name__}: {e}",
                diagnostics={"error": str(e), "type": type(e).__name__},
            )


def _parse_tool_exit(phase_id: str, raw: dict) -> SubAgentResult:
    """Parse the raw dict from a report_result tool call into a SubAgentResult."""
    # Parse status
    status_str = raw.get("status", "failure")
    status = ResultStatus.success if status_str == "success" else ResultStatus.failure

    # Parse outputs
    outputs = [
        Artifact(name=o.get("name", ""), file_path=o.get("file_path", ""))
        for o in raw.get("outputs", [])
    ]

    # Parse test report
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
