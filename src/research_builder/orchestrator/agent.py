"""Orchestrator agent: LLM-driven reasoning via Claude Agent SDK (spec_v4 §4)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from ..config import Config
from ..llm.paper import extract_full_text
from ..models.results import SubAgentResult
from ..models.spec import Artifact, EventType, PhaseState, Revision, SpecMetadata, SpecState
from ..storage.spec_store import SpecStore
from .dependency import DependencyGraph
from .spec_manager import SpecManager

logger = logging.getLogger(__name__)

SPEC_CREATION_SYSTEM_PROMPT = """\
You are an expert research paper analyst. Your job is to read a research paper and \
produce two outputs that will guide a team of implementation agents:

1. **spec.md** — A rich markdown document that serves as the canonical interpretation \
of the paper. It should contain:
   - A global context section with a 2-3 paragraph summary
   - One section per implementation phase (typically: Data, Architecture, Training, Eval, Results)
   - For each phase: a detailed description, acceptance criteria, and any relevant \
     hyperparameters, equations, or implementation details extracted from the paper
   - Flagged ambiguities or unclear details

2. **state** — A structured JSON object with:
   - metadata: paper_id, paper_title
   - phases: list of phase objects with phase_id, title, inputs (list of {name, file_path}), \
     outputs (list of {name, file_path})
   - dependency_graph: dict mapping phase_id to list of phase_ids it depends on

## Output Format

Return your response as a JSON object with exactly two keys:
```json
{
  "spec_md": "the full markdown content...",
  "state": {
    "metadata": {"paper_id": "...", "paper_title": "..."},
    "phases": [...],
    "dependency_graph": {...}
  }
}
```

## Guidelines

- Be thorough. Extract every hyperparameter, every architectural detail, every dataset reference.
- Use the standard phase IDs: "data", "architecture", "training", "eval", "results"
- Artifact file_paths should use the convention: phases/<phase_id>/<try_num>/outputs/<filename>
  Use try_num=1 as placeholder.
- Flag anything ambiguous — the implementation agents will consult the paper for details, \
  but the spec should identify known gaps.
- The dependency graph should reflect which phases need outputs from other phases. \
  Typically: data and architecture are independent, training depends on both, \
  eval depends on training (and sometimes data), results depends on eval and training.
"""

ACCEPTANCE_REVIEW_SYSTEM_PROMPT = """\
You are reviewing a sub-agent's completed work for cross-phase compatibility. \
You are NOT re-implementing or re-testing — the sub-agent already validated its own code. \
Your job is to check that the outputs will work with downstream phases.

Review the sub-agent's result and determine:
1. Are the output artifacts present and in the expected format?
2. Will downstream phases be able to consume these outputs based on their input specs?
3. Are there any cross-phase interface mismatches?

Respond with a JSON object:
```json
{
  "accept": true/false,
  "feedback": "explanation if rejecting, null if accepting"
}
```
"""


class OrchestratorAgent:
    """LLM-driven orchestrator for spec creation and phase review."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def create_spec(self, paper_path: Path, store: SpecStore) -> SpecManager:
        """Ingest a paper and produce the canonical spec (§4.1)."""
        logger.info("Ingesting paper from %s", paper_path)

        paper_text = extract_full_text(paper_path)

        response_text = await self._query(
            system=SPEC_CREATION_SYSTEM_PROMPT,
            prompt=f"Here is the paper:\n\n{paper_text}",
            tools=[],  # Pure reasoning — no tools needed
        )

        parsed = _extract_json(response_text)

        # Build spec.md
        spec_md = parsed.get("spec_md", response_text)

        # Build state
        raw_state = parsed.get("state", {})
        raw_metadata = raw_state.get("metadata", {})
        metadata = SpecMetadata(
            paper_id=raw_metadata.get("paper_id", paper_path.stem),
            paper_title=raw_metadata.get("paper_title", "Unknown"),
        )

        phases = []
        for p in raw_state.get("phases", []):
            phases.append(PhaseState(
                phase_id=p.get("phase_id", ""),
                title=p.get("title", ""),
                inputs=[Artifact(**a) for a in p.get("inputs", [])],
                outputs=[Artifact(**a) for a in p.get("outputs", [])],
                max_debug_attempts=self.config.max_debug_attempts,
            ))

        dep_graph = raw_state.get("dependency_graph", {})

        state = SpecState(
            metadata=metadata,
            phases=phases,
            dependency_graph=dep_graph,
        )

        # Validate
        graph = DependencyGraph.from_spec_state(state)
        phase_ids = {p.phase_id for p in phases}
        errors = graph.validate(phase_ids)
        if errors:
            logger.warning("Dependency graph validation issues: %s", errors)

        # Persist
        store.save_spec_md(spec_md)
        store.save_state(state)
        store.append_revision(Revision(
            event_type=EventType.spec_created,
            rationale=f"Initial spec created from {paper_path.name}",
        ))

        logger.info("Spec created: %d phases, dependency_graph=%s", len(phases), dep_graph)
        return SpecManager(store, state)

    async def acceptance_review(
        self,
        phase_id: str,
        result: SubAgentResult,
        spec_manager: SpecManager,
    ) -> tuple[bool, str | None]:
        """Review a sub-agent's result for cross-phase compatibility (§6.2)."""
        phase = spec_manager.state.get_phase(phase_id)
        if phase is None:
            return False, f"Unknown phase: {phase_id}"

        downstream_ids = spec_manager.dep_graph.get_downstream(phase_id)
        downstream_info = []
        for ds_id in downstream_ids:
            ds = spec_manager.state.get_phase(ds_id)
            if ds:
                downstream_info.append({
                    "phase_id": ds.phase_id,
                    "title": ds.title,
                    "inputs": [a.model_dump() for a in ds.inputs],
                })

        review_context = {
            "phase_id": phase_id,
            "phase_title": phase.title,
            "expected_outputs": [a.model_dump() for a in phase.outputs],
            "actual_outputs": [a.model_dump() for a in result.outputs],
            "sub_agent_summary": result.summary,
            "test_report": {
                "tests_run": result.test_report.tests_run,
                "tests_passed": result.test_report.tests_passed,
                "tests_failed": result.test_report.tests_failed,
            },
            "downstream_consumers": downstream_info,
        }

        response_text = await self._query(
            system=ACCEPTANCE_REVIEW_SYSTEM_PROMPT,
            prompt=f"Review this phase result:\n\n```json\n{json.dumps(review_context, indent=2)}\n```",
            tools=["Read", "Bash", "Glob", "Grep"],
            max_turns=3,
        )

        parsed = _extract_json(response_text)
        accepted = parsed.get("accept", False)
        feedback = parsed.get("feedback")

        logger.info("Acceptance review for phase=%s: accepted=%s", phase_id, accepted)
        return accepted, feedback

    async def _query(
        self,
        system: str,
        prompt: str,
        max_retries: int = 3,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> str:
        """Run a single query and return the text response."""
        import asyncio

        for attempt in range(max_retries):
            try:
                stderr_lines: list[str] = []

                def capture_stderr(line: str) -> None:
                    stderr_lines.append(line)
                    if "error" in line.lower() or "fatal" in line.lower() or "exception" in line.lower():
                        logger.error("CLI stderr: %s", line)

                options = ClaudeAgentOptions(
                    system_prompt=system,
                    model=self.config.model,
                    permission_mode="bypassPermissions",
                    cwd=str(self.config.project_root),
                    max_turns=max_turns,
                    stderr=capture_stderr,
                )
                if tools is not None:
                    options.allowed_tools = tools

                result_text = ""
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                result_text += block.text
                    elif isinstance(message, ResultMessage):
                        if message.result:
                            result_text = message.result

                return result_text

            except Exception as e:
                if stderr_lines:
                    logger.error("CLI stderr (%d lines):\n%s", len(stderr_lines), "\n".join(stderr_lines[-20:]))
                else:
                    logger.error("CLI failed with no stderr output")
                if attempt < max_retries - 1:
                    logger.warning("Query attempt %d failed: %s. Retrying...", attempt + 1, e)
                    await asyncio.sleep(3)
                else:
                    raise


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may contain markdown code blocks."""
    code_block = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning("Could not extract JSON from response")
    return {}
