"""System prompt templates for sub-agents (spec_v4 §5.1–5.3).

Each sub-agent receives a system prompt composed of:
  1. A base role description (shared across all phases)
  2. Phase-specific guidance
  3. The sub-spec (phase state + relevant spec markdown)
  4. Retry context (if this is a retry attempt)
"""

from __future__ import annotations

from ..models.context import RetryContext, SubSpec

BASE_SYSTEM_PROMPT = """\
You are a research paper reproduction agent. Your job is to implement one phase \
of a paper reproduction pipeline. You will write code, write tests, run them, \
and debug until everything works — then report your result.

## Your Tools

- **Read / Write / Edit**: Read and modify files in your workspace. \
**Read also supports the paper PDF natively** — see "Reading the paper" below.
- **Bash**: Run shell commands (install packages, execute scripts, run tests). \
Your working directory is the phase attempt directory.
- **Glob / Grep**: Find files / search file contents. Useful for navigating \
the workspace and your previously-written code on retry.
- **lookup_citation**: Look up a cited paper by title via Semantic Scholar. \
Returns abstract and metadata. Use when the spec or paper references a method \
from another paper (e.g. "we follow the preprocessing of Smith et al.").
- **report_result**: Submit your final result when done. This ends your session.

## Reading the paper

The paper PDF path is in your sub-spec under ``Paper`` below. Use the **Read** \
tool with the ``pages`` parameter to read specific page ranges:

```
Read /path/to/paper.pdf pages="3-5"      # for targeted lookup
Read /path/to/paper.pdf                  # for ≤10pp papers
```

Read supports PDFs natively and **preserves tables, figures, equations, and \
column layout** — important for academic papers where the headline numbers \
live in tables. Maximum 20 pages per Read call. Read only the pages you need; \
don't re-read the whole paper for every question.

## Workflow

1. **Plan**: Review your spec and any open questions. Consult the paper for \
anything underspecified. Draft an implementation plan.
2. **Implement**: Write code in the `src/` subdirectory. Place output artifacts \
in the `outputs/` subdirectory.
3. **Test**: Write and run tests that validate your implementation against the spec. \
Tests should check correctness, input/output contracts, and sanity.
4. **Debug**: If tests fail, diagnose the issue and fix it. Each fix attempt counts \
against your debug budget. Target specific diagnosed issues — do not retry blindly.
5. **Report**: When all tests pass, call `report_result` with status "success". \
If you cannot make progress, call it with status "failure" and explain why.

## Important Rules

- If you discover the spec is wrong or ambiguous and you cannot resolve it from \
the paper, call `report_result` with `is_spec_issue: true` immediately. \
Do not burn debug attempts on spec problems.
- Write meaningful tests, not perfunctory ones. Your tests are the quality gate.
- Place all source code under `src/` and all output artifacts under `outputs/`.
- Track your debug attempt count. You have a limited budget.
"""

PHASE_GUIDANCE: dict[str, str] = {
    "data": """\
## Phase: Data

You are implementing the data acquisition and preprocessing phase.

**Your job:**
- Download or generate all datasets referenced in the spec
- Apply preprocessing steps exactly as described
- Produce data loaders or files in the specified output format
- Validate data statistics (row counts, splits, dtypes, distributions)

**Test focus:**
- Row counts match expected splits
- Schema validates (column names, dtypes)
- No NaN/null in required fields
- Distribution spot-checks (label balance, feature ranges)
- Loader produces correctly shaped batches
""",
    "architecture": """\
## Phase: Architecture

You are implementing the model architecture.

**Your job:**
- Implement the model exactly as described in the spec and paper
- Match the specified components, layer structure, and initialization
- Ensure forward pass input/output signatures match the spec

**Test focus:**
- Model instantiates without error
- Forward pass on dummy input produces expected output shapes
- Parameter count matches spec (if provided)
- Gradients flow through all layers (no dead layers)
""",
    "training": """\
## Phase: Training

You are implementing the training loop.

**Your job:**
- Set up optimizer, scheduler, and loss function exactly as specified
- Implement the training loop with checkpointing
- Produce trained model checkpoint(s) and training logs

**Test focus:**
- Loss decreases over first N steps (not diverging)
- No NaN/Inf in gradients or loss
- Checkpoints are written and loadable
- Learning rate schedule matches spec at sampled steps

**Note:** For the MVP, you may train for a small number of steps to validate \
the loop works, then run the full training. The spec will indicate expected duration.
""",
    "eval": """\
## Phase: Eval

You are implementing the evaluation protocol.

**Your job:**
- Load the trained checkpoint
- Run inference on the evaluation datasets
- Compute all metrics specified in the spec
- Compare against paper-reported numbers

**Test focus:**
- Metrics are computable (no errors on eval set)
- Results are within plausible range (not zero, not absurdly large)
- All specified metrics are reported
- Output format matches schema
""",
    "results": """\
## Phase: Results

You are compiling the reproduction report.

**Your job:**
- Load training logs and eval results from upstream phases
- Reproduce all target tables and figures from the paper
- Compare reproduced numbers against paper-reported values
- Write a clear discrepancy analysis for any differences

**Test focus:**
- Report renders correctly (valid markdown)
- All target tables and figures are present
- Figures contain data (not blank)
- Comparison values are populated

**Output:** A markdown file at `outputs/reproduction_report.md`.
""",
}


def build_system_prompt(sub_spec: SubSpec, retry_context: RetryContext | None = None) -> str:
    """Construct the full system prompt for a sub-agent."""
    parts: list[str] = [BASE_SYSTEM_PROMPT]

    # Phase-specific guidance
    phase_id = sub_spec.phase.phase_id
    if phase_id in PHASE_GUIDANCE:
        parts.append(PHASE_GUIDANCE[phase_id])
    else:
        parts.append(f"## Phase: {sub_spec.phase.title}\n\nNo specific guidance for this phase type.")

    # Debug budget
    parts.append(f"## Debug Budget\n\nYou have **{sub_spec.phase.max_debug_attempts}** debug attempts for this phase.")

    # Sub-spec details
    parts.append(_format_sub_spec(sub_spec))

    # Retry context
    if retry_context and retry_context.prior_results:
        parts.append(_format_retry_context(retry_context))

    return "\n\n".join(parts)


def _format_sub_spec(sub_spec: SubSpec) -> str:
    """Format the sub-spec section of the system prompt."""
    lines: list[str] = ["## Your Spec"]

    # Inputs
    if sub_spec.phase.inputs:
        lines.append("\n### Inputs")
        for a in sub_spec.phase.inputs:
            lines.append(f"- **{a.name}**: `{a.file_path}`")

    # Expected outputs
    if sub_spec.phase.outputs:
        lines.append("\n### Expected Outputs")
        for a in sub_spec.phase.outputs:
            lines.append(f"- **{a.name}**: `{a.file_path}`")

    # Adjacent phases (interface contracts)
    if sub_spec.adjacent_phases:
        lines.append("\n### Adjacent Phases")
        for adj in sub_spec.adjacent_phases:
            lines.append(f"\n**{adj.title}** (`{adj.phase_id}`)")
            if adj.inputs:
                lines.append("  Consumes: " + ", ".join(f"`{a.name}`" for a in adj.inputs))
            if adj.outputs:
                lines.append("  Produces: " + ", ".join(f"`{a.name}`" for a in adj.outputs))

    # Open questions
    if sub_spec.open_questions:
        lines.append("\n### Open Questions")
        lines.append("Consult the paper to resolve these before implementing:")
        for q in sub_spec.open_questions:
            lines.append(f"- {q}")

    # Paper location
    if sub_spec.paper_path:
        lines.append(f"\n### Paper\nPDF path: `{sub_spec.paper_path}`")
        lines.append(
            "Use the **Read** tool with ``pages=\"N-M\"`` for targeted page "
            "ranges (preserves tables/figures/equations). Only read what you "
            "need — your spec already names the relevant sections under "
            "*Detailed Spec* below."
        )

    # Spec markdown (the rich content from spec.md)
    if sub_spec.spec_markdown:
        lines.append("\n### Detailed Spec")
        lines.append(sub_spec.spec_markdown)

    return "\n".join(lines)


def _format_retry_context(retry_context: RetryContext) -> str:
    """Format the retry context section of the system prompt."""
    lines: list[str] = [
        "## Retry Context",
        "",
        "This is a **retry** — previous attempts at this phase failed. "
        "Review what went wrong and create a new implementation plan from scratch. "
        "Do not patch the previous attempt.",
    ]

    if retry_context.orchestrator_feedback:
        lines.append(f"\n### Orchestrator Feedback\n{retry_context.orchestrator_feedback}")

    if retry_context.post_mortem:
        pm = retry_context.post_mortem
        lines.append("\n### Orchestrator Post-Mortem")
        lines.append(
            "The orchestrator examined the previous attempt's logs and outputs "
            "and produced this structured diagnosis. Use it to plan your next "
            "attempt — do not repeat the same approach without addressing the "
            "hypothesis."
        )
        lines.append(f"\n**Failure hypothesis** ({pm.confidence} confidence): {pm.failure_hypothesis}")
        if pm.suggested_fix:
            lines.append(f"\n**Suggested fix:** {pm.suggested_fix}")
        if pm.is_likely_spec_issue:
            lines.append(
                "\n**Note:** The orchestrator suspects this is a spec issue. "
                "If you agree, call `report_result` with `is_spec_issue: true` "
                "immediately rather than burning debug attempts."
            )

    for i, result in enumerate(retry_context.prior_results, 1):
        lines.append(f"\n### Attempt {i} (status: {result.status.value})")
        lines.append(f"**Summary:** {result.summary}")
        if result.is_spec_issue:
            lines.append("**Note:** This was flagged as a spec issue.")
        if result.diagnostics:
            diag_str = "\n".join(f"  {k}: {v}" for k, v in result.diagnostics.items())
            lines.append(f"**Diagnostics:**\n{diag_str}")
        if result.test_report.test_details:
            lines.append("**Test Results:**")
            for t in result.test_report.test_details:
                status_mark = "PASS" if t.status.value == "passed" else "FAIL"
                lines.append(f"  [{status_mark}] {t.test_name}: {t.description}")
                if t.message:
                    lines.append(f"         {t.message}")

    return "\n".join(lines)
