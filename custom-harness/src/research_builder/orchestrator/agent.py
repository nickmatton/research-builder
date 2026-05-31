"""Orchestrator agent: LLM-driven reasoning via Claude Agent SDK (spec_v4 §4)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from ..config import Config
from ..events import emit_artifact_created, get_emitter, maybe_emit_paper_read
from ..events.emitter import capture_file_before, emit_file_write
from ..literature.references import extract_citation_titles
from ..literature.scholar import SemanticScholarClient
from ..llm.paper import extract_full_text
from pydantic import BaseModel

from ..models.claims import Claim, ClaimSource, ClaimsLedger
from ..models.context import PostMortem
from ..models.results import SubAgentResult
from ..models.spec import (
    AcceptanceCriterion,
    Artifact,
    Citation,
    CritiqueVerdict,
    DagNode,
    EventType,
    PhaseKind,
    PhaseState,
    PlanDocument,
    PlannedFile,
    Revision,
    SectionCritique,
    SectionSpec,
    SpecMetadata,
    SpecState,
)
from ..storage.spec_store import SpecStore
from .dependency import DependencyGraph
from .prompts import (
    ACCEPTANCE_REVIEW_SYSTEM_PROMPT,
    POST_MORTEM_SYSTEM_PROMPT,
    SPEC_REFINEMENT_SYSTEM_PROMPT,
    STRUCTURED_JSON_CONTRACT,
)
from .spec_manager import SpecManager

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Metadata captured for a single ``_query()`` call.

    Stashed on the OrchestratorAgent after each call so ExecutionLoop can
    record it as a per-attempt step file. ``response_text`` is the full
    result string ``_query`` returned to its caller.
    """

    prompt_role: str
    system_prompt: str
    prompt: str
    started_at: float
    ended_at: float
    duration_s: float
    response_text: str
    model: str
    messages_received: list[str] = field(default_factory=list)
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "ok"  # ok | error | timeout

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the research-builder orchestrator. Your job is to help a user reproduce \
the results of a research paper, with the user in the loop — they should always \
feel informed and in control, never surprised. The heavy lifting (reading the \
paper, authoring specs, writing code, running tests) is wrapped in tools you \
call. Your job is to choose what to do next, narrate clearly so the user follows \
what's happening, and pause for their input at the right moments.

## The general flow

Read the paper → write a slim skeleton → fan out per-section specs and critique \
them in parallel → extract numerical claims → for each section, run the per-step \
chain (refiner → researcher → builder → verifier) → final summary. You decide \
the ordering and when to pause; this is a guideline, not a contract.

## Tools

Each tool returns a short text summary. Paraphrase results in chat — never dump \
raw JSON.

### Setup (run once near the start)

- ``write_skeleton`` — read the paper, write the top-level spec.md + state.json. \
  Identifies the sections worth reproducing.
- ``author_section_specs`` — fan out per-section spec authoring in parallel. \
  One agent per section.
- ``critique_section_specs`` — critic re-reads the paper against each section \
  spec. Flags hallucinations or missing details.
- ``extract_claims_ledger`` — pull numerical claims (table rows, headline \
  accuracies) into claims.json. Independent of section spec authoring.

### Per-section sub-steps (granular control during execution)

- ``run_refiner(phase_id)`` — fast. Loads the upfront section spec or refines \
  fresh. Returns research_questions the researcher should answer.
- ``run_researcher(phase_id)`` — variable. Skipped if no research questions. \
  Pulls citations and external context.
- ``run_builder(phase_id)`` — **long** (minutes to hours). The Builder \
  sub-agent writes code, runs tests, iterates.
- ``run_verifier(phase_id)`` — judges the builder's output. On accept, marks \
  the phase complete + propagates outputs. On reject, records the failure \
  but does NOT auto-retry — you decide whether to call ``run_builder`` again.

### Convenience

- ``start_phase(phase_id)`` — runs all four sub-steps sequentially without \
  pausing between them. Use only when the user has said "just run it" or \
  for batches of trivial sections.

### Scheduling

- ``list_pending_phases`` — phases ready to execute next (DAG order). \
  Returns phase_id, title, goal, planned_files. Call before each builder \
  invocation so you can show the user what's about to be created.

### User

- ``request_user_approval(prompt, gate_id)`` — pause and wait for the user's \
  next chat reply. The reply is returned verbatim — **you** decide whether \
  it's an approval, a question, or an edit request. If it's not a clear \
  approval, respond in plain text, then call this tool again to keep waiting.

### Terminal

- ``pipeline_complete(message)`` — call when the run is finished.
- ``pipeline_failed(message)`` — call only if you genuinely cannot continue.

## When to pause for approval

You decide when to block; here are sensible defaults. Override them based on \
context.

**Always pause:**
- **At the very start**, before doing any work. Introduce yourself in plain \
  text, then call ``request_user_approval`` with ``gate_id="pre_run"``.
- **After ``write_skeleton``**. The section list is a high-stakes commitment; \
  the user should review it before you spend tokens authoring each section.
- **Before each ``run_builder``** (or ``start_phase``). The builder is the \
  long, expensive step. The user should see the phase_id, goal, and the \
  planned files (from ``list_pending_phases``) and approve before you launch.
- **After each ``run_verifier``** that either accepted (so the user can review \
  the result and decide to keep going) or rejected (so the user can decide \
  whether to retry, edit the spec, or skip).

**Use your judgment:**
- Between ``run_refiner`` and ``run_researcher`` — usually no pause needed. \
  Pause if the refiner surfaced research_questions that seem unusual.
- Between ``run_researcher`` and ``run_builder`` — usually no pause needed. \
  Builder is the next big step; the pre-builder pause covers this.
- Between sections — auto-continue if the previous section verified cleanly \
  and nothing in the chat suggests the user wants to chime in.

**Don't pause for routine narration.** If you just want to tell the user \
what you're doing, emit plain text (an assistant message). Reserve \
``request_user_approval`` for moments when you actually need a decision.

**Point the user at the doc they're reviewing.** When the gate is about \
approving a spec, pass ``open_doc`` so the UI opens it next to the banner: \
``"spec.md"`` for the skeleton (typical for ``post_skeleton``), \
``"sections/<phase_id>.md"`` for a section spec (typical for \
``pre_phase:<id>`` / ``pre_builder``). Skip ``open_doc`` for gates that \
aren't about reading a doc (``pre_run``, ``post_verifier`` summary, etc.).

## When the user replies

The user can chime in any time. Their message arrives as the result of your \
next ``request_user_approval`` call.

- **Clear approval** ("yes", "go", "lgtm", "approve", "looks good"): proceed \
  with the action you were about to take.
- **Question** ("what does X mean?", "why this section?"): answer it in plain \
  text using what you know from prior tool results. If you need to look \
  something up, you can re-call ``list_pending_phases`` for fresh state. Then \
  call ``request_user_approval`` again to keep waiting.
- **Edit request** ("change X to Y", "skip section Z"): acknowledge what you'll \
  change, but the actual edit happens through the operator command channel \
  (handled outside your tools). After acknowledging, call \
  ``request_user_approval`` again to wait for them to either approve \
  proceeding or send a follow-up.
- **Ambiguous** ("fine"): treat as approval; user can always say "wait" if \
  you misread them.

## Style

- Conversational and concise. Short paragraphs, no walls of text.
- Bold the **phase_id** and key noun phrases.
- Bulleted lists for planned files and verdict breakdowns.
- Paraphrase tool results in chat. Don't dump raw JSON.
- No filler ("Great!", "Sure!", "Excellent!") before every action.
- When narrating long stretches between gates, drop short status messages \
  (plain text, not request_user_approval) so the chat doesn't go silent. \
  Don't overdo it — one status message per major sub-step is plenty.

## What you don't do

- Don't try to read or write files directly — you have no Read/Write/Edit/Bash \
  access at this level. The tools own all file I/O.
- Don't fan out parallel work yourself — tools handle parallelism internally.
- Don't skip the pre_run / post_skeleton / pre_builder / post_verifier gates. \
  Even confident users want a chance to review at these high-stakes moments.
- Don't call ``pipeline_complete`` or ``pipeline_failed`` until the work is \
  genuinely done or you genuinely cannot continue.
"""


SPEC_CREATION_SYSTEM_PROMPT = """\
You are an expert research paper analyst. Your job is to read a research paper \
and produce a **slim top-level reproduction skeleton**. Per-section detail is \
authored by a separate fan-out of agents downstream — you do NOT write per-section \
implementation detail. Your output is the scaffolding: which sections matter, how \
they depend on each other, and the shared invariants (hyperparameters, datasets, \
metrics) that those sections will reference.

## Output protocol — IMPORTANT, READ FIRST

You produce the skeleton by **writing two files** to disk via the ``Write`` and
``Edit`` tools. You DO NOT emit the skeleton as a chat response. The user
prompt gives you the exact paths.

  1. ``spec.md`` — slim markdown (target: under 200 lines total). Format below.
  2. ``state.json`` — machine-readable JSON (schema below).

Write/Edit liberally — small incremental writes are fine. Each call to
Write or Edit ends an API request and starts a fresh one (the
checkpointing mechanism that keeps long calls under the stream-idle
timeout).

Keep ``state.json`` **strictly valid JSON** — no comments, no trailing
commas, no single-quoted strings. Inside string values, escape newlines
as ``\\n``, tabs as ``\\t``, and quotes as ``\\"``.

### ``state.json`` schema

```jsonc
{
  "metadata": {
    "paper_id": "lstm_hochreiter_1997",
    // type: string  (stable short id)
    "paper_title": "Long Short-Term Memory",
    // type: string
    "paper_url": null
    // type: string | null
  },
  "phases": [
    {
      "phase_id": "section_3_2_attention",
      // type: string  (one entry per implementation-requiring paper section)
      "title": "<short human title>",
      // type: string
      "kind": "build",
      // type: string  enum: "build" | "experiment"
      "inputs": [
        {"name": "<artifact_name>", "file_path": "phases/<upstream_phase>/outputs/<filename>"}
      ],
      "outputs": [
        {"name": "<artifact_name>", "file_path": "phases/<this_phase>/outputs/<filename>"}
      ]
    }
  ],
  "dependency_graph": {
    "section_3_2_attention": [],
    "section_5_1_data": [],
    "section_5_training": ["section_3_2_attention", "section_5_1_data"]
  },
  // type: dict[string, list[string]]  (phase_id -> upstream phase_ids)
  "plan": {
    // Top-level DAG view. Keep nodes terse — the per-section authors fill in
    // sub_steps and file_ids later. Empty lists are fine here.
    "nodes": [
      {
        "phase_id": "section_5_1_data",
        "title": "Data — WMT 2014 EN-DE",
        "description": "One short sentence — full description lives in the per-section spec.",
        "kind": "build",
        "sub_steps": [],
        "file_ids": [],
        "depends_on": []
      }
    ],
    "files": []
  }
}
```

### ``spec.md`` format (TARGET LENGTH: under 200 lines)

```markdown
# <paper title>

**Paper:** <paper_id>   **Pages:** N

<3-sentence summary of the paper.>

## Sections to reproduce

| Phase ID | Title | Pages | Reproduce? | Notes |
|---|---|---|---|---|
| section_3_2_attention | Multi-Head Attention | 4–5 | yes | core architecture |
| section_5_1_data      | Data preprocessing   | 7   | yes | needed by training |
| section_4_related     | Related Work         | 2–3 | no  | pure prose, skip |
...

## Dependency graph

```
section_5_1_data, section_3_2_attention → section_5_training → section_6_1_results
```
(or any concise textual rendering of the DAG)

## Shared invariants

These values are quoted once here and referenced by phase_id in the per-section \
specs. They are the **canonical source of truth** for downstream agents.

### Hyperparameters
| Name | Value | Source |
|---|---|---|
| learning rate | 1e-4 | §5.1, p.7 |
| batch size    | 32   | §5.1, p.7 |
...

### Datasets
| Name | Description | Source |
|---|---|---|
| WMT 2014 EN-DE | 4.5M sentence pairs | §5.1, p.7 |
...

### Evaluation metrics
| Name | Description | Source |
|---|---|---|
| BLEU | corpus-level BLEU on newstest2014 | §6.1, p.10 |
...

## Claims to reproduce

The numerical claims are extracted in parallel into ``claims.json``. The
per-section specs cross-reference them by ``claim_id``.

## Flagged ambiguities

Anything the per-section authors should investigate. Cite page numbers.
```

## Reading the paper

Use the **Read** tool on the paper PDF path you'll be given. Read supports \
PDFs natively. For papers ≤10 pages, read the whole thing in one Read call. \
For larger papers use the ``pages`` parameter (max 20 pages per call). The page \
count is in the user prompt.

## Identifying sections

Read the paper. Identify each section that REQUIRES IMPLEMENTATION WORK to \
reproduce. Skip pure-prose sections (Introduction, Related Work, Conclusion, \
Acknowledgements) — include them in the "Sections to reproduce" table with \
``Reproduce? = no`` so the reader can see they were considered and dismissed.

Use STABLE, descriptive section_ids derived from the paper's section numbering: \
``section_3_2_attention``, ``section_5_1_data``, ``section_6_1_results_en_de``.

## Phase kinds

  - ``build``      — produces reusable code/infrastructure (model architectures,
                     algorithms, data loaders, training scaffolding, eval utilities).
  - ``experiment`` — runs a specific experiment to validate a paper's numerical
                     claim (headline-table rows, ablations).

Rule of thumb: if the phase exists to produce a number the paper reports, \
it's an experiment. If it exists so other phases can import its code, it's \
a build phase.

## What NOT to do

- **Do not** write per-section implementation detail in ``spec.md`` (no hyperparameter \
  blocks per section, no architectural prose, no acceptance criteria). That is the \
  per-section authors' job downstream. Your spec.md is a navigation map.
- **Do not** embed long verbatim quotes from the paper. Cite by page.
- **Do not** invent file paths beyond the ``phases/<phase_id>/outputs/<filename>`` \
  convention used in ``inputs`` / ``outputs``.
- **Do not** exceed 200 lines in spec.md. If you find yourself doing so, you are \
  writing per-section detail — push that down to the per-section authors.

## Why slim is the goal

A slim skeleton is the single source of truth for shared invariants \
(hyperparameters, datasets, metrics). Per-section specs reference those names \
rather than restating values, so the system can't drift into contradictions. \
The skeleton also gives the user (and downstream agents) a 30-second \
orientation to the whole reproduction.
"""

CLAIMS_EXTRACTION_SYSTEM_PROMPT = """\
You are extracting every numerical/quantitative claim from a research paper. \
These claims will be stored in a structured ledger and verified automatically \
when the paper's code is reproduced.

## Reading the paper

Use the **Read** tool on the paper PDF path in the user prompt. Read supports \
PDFs natively (preserves tables, equations, figures). Tables are usually where \
the headline numerical claims live — read those pages especially carefully. \
For papers ≤10 pages, one Read call covers it; for larger, use ``pages`` \
(max 20 per call). Page count is in the user prompt.

Extract claims from:
- Results tables (every row/column with a numeric metric)
- Figures with reported numbers (axes labels, annotations)
- Inline text ("we achieve 95.2% accuracy on CIFAR-10")
- Ablation tables
- Training details that are verifiable (e.g. "converges in 50 epochs", \
  "final training loss of 0.23")

## Output protocol — IMPORTANT, READ FIRST

You write the claims to a JSON file via the ``Write`` and ``Edit`` tools.
The user prompt gives you the exact output path. Each Write/Edit ends
an API request and starts a fresh one (this is the checkpointing
mechanism that lets long jobs run without hitting the stream-idle
timeout). Small incremental writes / appends are fine — write a chunk
of claims, then Edit to append more as you find them.

Keep the file **strictly valid JSON** — no comments, no trailing commas,
no single quotes. The harness re-parses it; a malformed file is dropped.

### Output file schema

A top-level JSON array of claim objects. Each entry:

```jsonc
[
  {
    "claim_id": "table2_cifar10_top1",
    // type: string  (stable snake_case ID)
    "metric": "top-1 accuracy",
    // type: string  (what's measured)
    "value": 95.2,
    // type: float  (the number; emit as JSON number, NOT a string)
    "tolerance": 0.3,
    // type: float  (± range if stated, else 0)
    "unit": "%",
    // type: string  ("%", "ms", etc.; "" if dimensionless)
    "dataset": "CIFAR-10 test set",
    // type: string
    "condition": "ResNet-50, 200 epochs",
    // type: string
    "source": {
      "table": "Table 2",
      // type: string | null  (OR figure/section; null when not applicable)
      "figure": null,
      // type: string | null
      "section": null,
      // type: string | null
      "page": 7,
      // type: int  (page number, NOT a string)
      "verbatim": "95.2 ± 0.3"
      // type: string (exact quote from the paper)
    },
    "phase_id": "section_6_1_eval",
    // type: string  (which spec phase reproduces this)
    "notes": "mean of 3 runs"
    // type: string  ("" if no notes)
  }
]
```

Be thorough — every number that could be verified is a claim. Include baseline \
comparisons only if the paper reports reproducing them. Skip claims about other \
papers' results unless the paper re-ran them.
"""


SECTION_SPEC_PROMPT = """\
You are a per-section spec author. The orchestrator has already written a slim \
top-level skeleton (``spec.md`` and ``state.json``) that names the sections worth \
reproducing and lists shared invariants (hyperparameters, datasets, eval metrics). \
Your job is to author the **detailed spec for exactly one section** of the paper.

## Output protocol — IMPORTANT, READ FIRST

You write ONE JSON file via the ``Write`` tool to the path the user prompt \
gives you. Schema below. Then you stop. You do NOT emit any other artifact, \
edit other files, or comment in chat.

Use ``Write`` (or ``Edit`` to refine). Each call ends the current API request, \
so small writes are fine. The file must be **strictly valid JSON** — no \
comments, no trailing commas, no single quotes. Inside strings, escape newlines \
as ``\\n``, tabs as ``\\t``, and quotes as ``\\"``.

### Output schema

```jsonc
{
  "phase_id": "section_3_2_attention",
  // type: string  (must match the phase_id from the user prompt)
  "title": "Multi-Head Attention",
  // type: string  (mirror the title from the skeleton)
  "goal": "Implement the multi-head scaled dot-product attention module per §3.2.",
  // type: string  (one sentence — what success looks like for this section)
  "spec_markdown": "<full markdown body for this section — see format below>",
  // type: string  (the human-readable spec the Builder will work from)
  "acceptance_criteria": [
    {
      "text": "Forward pass returns tensor of shape (batch, seq, d_model).",
      "source": { "page": 4, "section": "3.2", "quote": "..." }
    },
    {
      "text": "Multi-head split uses h=8 heads when d_model=512.",
      "source": { "page": 4, "section": "3.2", "quote": "h = 8 parallel attention layers" }
    }
  ],
  // type: list[object]  EVERY entry MUST include a non-null source.page.
  "citations": [
    { "page": 4, "section": "3.2", "quote": "..." },
    { "page": 5, "section": "3.2.1", "quote": null }
  ]
  // type: list[object]  Every page you cite anywhere in spec_markdown or
  //                     acceptance_criteria. Deduplicate by page+section.
}
```

## Mandatory citations

Every entry in ``acceptance_criteria`` must carry a non-null ``source.page``. \
A criterion without a citation will be rejected and you will be asked to \
re-author. Pages are 1-indexed from the paper PDF. ``section`` is the paper's \
own section number (e.g. ``"3.2"``). ``quote`` is the verbatim phrase you're \
relying on — keep it short (under 30 words).

If you can't cite, you can't claim. If a fact isn't anchored in a specific page \
of the paper, do not include it. The downstream Builder agent is allowed to \
make local engineering choices — your job is to fix the things the paper \
actually constrains.

## ``spec_markdown`` format

```markdown
## <Section title>

**Paper:** §<paper section number> (pages N–M)

**Goal:** <one-sentence restatement of `goal` above>

### What to implement
<2-4 paragraphs of prose extracted from the paper, with page citations in
markdown footnote style: "...as described on p. 4.">

### Hyperparameters / equations / details

Use ``invariant://<name>`` to reference a hyperparameter or dataset from the
shared-invariants table in the top-level spec.md — do NOT restate the value
inline (that creates drift). Examples:

  - Learning rate: ``invariant://learning_rate``
  - Optimizer: Adam (β1=0.9, β2=0.98, ε=1e-9), §5.4 p.7
  - Number of heads: h = 8

For anything not in the invariants table, cite the page inline.

### Acceptance criteria
<bulleted list mirroring `acceptance_criteria` above, in human-readable form.>

### File plan
<list of files this phase will create, using the convention
phases/<phase_id>/outputs/<filename>. Brief description of each.>

### Open questions
<anything the researcher agent should investigate later, with page references.>
```

## Reading the paper

Use the **Read** tool on the paper PDF path you'll be given. For a single \
section you usually only need 2–5 pages — use ``pages="N-M"`` (max 20). Read the \
specific section first, then any cross-referenced appendices/tables.

## Reading the top-level skeleton

The user prompt gives you the path to ``spec.md``. Open it first to see the \
shared invariants table — your section spec references these by name rather \
than restating values. This keeps the system from drifting into contradictions \
across sections.

## What NOT to do

- **Do not** restate hyperparameter values that exist in the shared invariants \
  table. Reference by ``invariant://name``.
- **Do not** invent acceptance criteria that aren't grounded in the paper. \
  Every criterion needs a citation.
- **Do not** edit any file other than the JSON output path you were given. \
  Don't edit ``spec.md`` or other section specs.
- **Do not** write code. You are authoring the spec; the Builder writes code.

## Reproduce the result, not the search

If the paper describes a hyperparameter sweep, record the **best configuration \
the paper actually reports** (from the experiment description, figure caption, \
or appendix). Do NOT write acceptance criteria that demand the Builder re-run \
the sweep. The paper already paid that cost.
"""


SPEC_CRITIC_PROMPT = """\
You are a tool-free critic of a single per-section spec. Your job is to re-read \
the relevant pages of the paper and judge whether the section spec is accurate, \
grounded, and complete.

## Output protocol

You write ONE JSON file via the ``Write`` tool to the path the user prompt \
gives you. Schema below. Then you stop. The file must be strictly valid JSON.

### Output schema

```jsonc
{
  "phase_id": "section_3_2_attention",
  // type: string  (must match the phase_id under review)
  "verdict": "verified",
  // type: string  enum: "verified" | "questionable" | "missing_citations"
  "reasons": [
    "All acceptance criteria carry valid page citations.",
    "Hyperparameter values referenced via invariant:// match the top-level spec."
  ]
  // type: list[string]  (1–6 short reasons; required even when verdict is "verified")
}
```

## Verdict definitions

- ``verified``           — every claim is grounded in the cited page, no \
                            hallucinations or contradictions detected.
- ``questionable``       — some claim seems to misread the paper, or a \
                            reference to the invariants table looks wrong, \
                            OR the spec demands a hyperparameter sweep \
                            (see "Sweep check" below). Reasons must \
                            enumerate which.
- ``missing_citations``  — one or more acceptance criteria lack a page \
                            citation, or a citation points to the wrong page.

## How to judge

1. Open the section spec JSON the user prompt points to.
2. Open the paper PDF; read the pages the spec cites (use the ``pages`` param, \
   max 20). Cross-reference each acceptance criterion against the cited page.
3. Open the top-level ``spec.md`` and confirm every ``invariant://name`` \
   reference exists in the shared invariants table.
4. Run the sweep check below.
5. Emit the verdict.

## Sweep check (mandatory)

Hyperparameter sweeps in acceptance criteria are the single biggest source of \
wasted GPU spend in this harness — they turn into ``for lr in [...]: train()`` \
loops that burn hours of A10/A100 time to "find" a value the paper already \
pinned. Catch them at spec time.

Mark the verdict as ``questionable`` and add a ``reasons`` entry if the \
section spec contains any of these patterns in acceptance criteria or "what \
to build" prose:

- The phrases: "grid search", "hyperparameter sweep", "parameter sweep", \
  "sweep over", "search over (learning rates|step sizes|hyperparams)", \
  "tune the learning rate", "lr selection", "step-size selection", \
  "best lr from {…}", "best step-size from {…}", "for each (lr, …) combination"
- Mathematical set notation describing candidates as the input to a search: \
  "α ∈ {…}", "lr ∈ {…}", "learning rate ∈ {…}", "we tried {…}" — treat \
  these as authors' methodology, not reproduction protocol
- An acceptance criterion that says "select the best <hyperparameter>" or \
  "find the optimal <hyperparameter>" by running training

When you flag this, the ``reasons`` entry MUST name the offending phrase and \
recommend the fix in one of two shapes: \
"Criterion <N> instructs a sweep over α; rewrite as N point-sampled trainings \
at the paper's pinned α per cell" \
or \
"Paper's appendix lists α ∈ {0.01, 0.02, 0.1} but doesn't pin a winner for \
Table 2 row K — Builder should be instructed to either hard-code 0.01 \
(Adagrad MNIST default) or call ``report_result(is_spec_issue=true)``."

A criterion that simply *references* the paper's own sweep (e.g. "Figure 1 \
shows the authors' lr sweep") is fine as long as the reproduction \
deliverable is point-sampled. The bar is: does the BUILDER end up writing a \
sweep loop? If yes, flag.

## What NOT to do

- **Do not** edit the section spec — your output is the critique JSON only.
- **Do not** propose new acceptance criteria — flag missing-coverage concerns \
  in ``reasons`` instead.
- **Do not** mark ``verified`` when you didn't actually re-read the cited pages.
- **Do not** mark ``verified`` when the sweep check above caught something — \
  ``questionable`` is the correct verdict for sweep-shaped criteria.
"""


# Sweep-phrase markers we scan section spec markdown for. A hit downgrades
# the critic's verdict to ``questionable`` even if the LLM judged the spec
# fine — multiple incidents (Adam paper, VAE) show the LLM critic can miss
# sweep language that turns into hours of wasted GPU once the Builder reads
# it. Patterns are anchored loosely on purpose: "grid search" in a flowing
# sentence and "Grid Search" in a heading both fire.
_SWEEP_PHRASE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgrid[\s-]?search(?:es|ing)?\b", re.IGNORECASE), "grid search"),
    (re.compile(r"\b(?:hyper[\s-]?parameter|parameter|hp)[\s-]?sweep(?:s|ing)?\b", re.IGNORECASE),
     "hyperparameter sweep"),
    (re.compile(r"\bsweep(?:s|ing)?\s+(?:over|across|through)\b", re.IGNORECASE),
     "sweep over/across"),
    (re.compile(r"\bsearch\s+(?:over|across|the)\s+(?:learning[\s-]?rate|step[\s-]?size|lr|hyperparameter)",
                re.IGNORECASE), "search over hyperparams"),
    (re.compile(r"\b(?:tune|tuning)\s+(?:the\s+)?(?:learning[\s-]?rate|lr|step[\s-]?size|hyperparameter)",
                re.IGNORECASE), "tune the lr/step-size/hyperparameter"),
    (re.compile(r"\blr[\s-]?selection\b|\bstep[\s-]?size[\s-]?selection\b|\blearning[\s-]?rate[\s-]?selection\b",
                re.IGNORECASE), "lr/step-size selection"),
    (re.compile(r"\bselect\s+the\s+best\s+(?:learning[\s-]?rate|lr|step[\s-]?size|hyperparameter|optimizer)",
                re.IGNORECASE), "select the best <hyperparameter>"),
    (re.compile(r"\bbest\s+(?:learning[\s-]?rate|lr|step[\s-]?size|alpha|α)\s+from\b", re.IGNORECASE),
     "best lr/α from {…}"),
    (re.compile(r"\bfor\s+each\s+\(?(?:lr|learning[\s-]?rate|alpha|α)\b", re.IGNORECASE),
     "for each (lr, …)"),
    # Set-notation that's almost always authors' methodology getting copied
    # into reproduction protocol: "α ∈ {0.01, 0.02, 0.1}", "lr in {…}".
    (re.compile(r"(?:α|alpha|lr|learning[\s-]?rate|step[\s-]?size)\s*∈\s*\{", re.IGNORECASE),
     "<hyperparam> ∈ {…}"),
    (re.compile(r"we\s+tried\s+\{[^}]*\}", re.IGNORECASE), "we tried {…}"),
    (re.compile(r"\bhyperparameter[\s-]?search\b|\bhp[\s-]?search\b", re.IGNORECASE),
     "hyperparameter search"),
]


def _find_sweep_phrases(text: str) -> list[str]:
    """Return de-duplicated sweep-phrase markers found in ``text``.

    Used as the deterministic safety net behind the spec critic: even if the
    LLM judges the section spec "verified", we scan the markdown for these
    patterns and force-downgrade the verdict. The strings returned are stable
    labels (not the matched substrings) so the critique reasons look the same
    regardless of which exact wording the spec used.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pattern, label in _SWEEP_PHRASE_PATTERNS:
        if pattern.search(text):
            if label not in seen:
                seen.add(label)
                out.append(label)
    return out


class SpecAmendment(BaseModel):
    """Result of an orchestrator-driven spec refinement pass."""
    amended_spec_md: str | None = None
    summary: str = ""
    sections_changed: list[str] = []

    @property
    def succeeded(self) -> bool:
        return self.amended_spec_md is not None and bool(self.amended_spec_md.strip())


class OrchestratorAgent:
    """LLM-driven orchestrator for spec creation and phase review."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.emitter = get_emitter()
        # Last _query() execution metadata — read by ExecutionLoop after each
        # high-level agent call (refine_section, research_for_section, etc.)
        # to write a per-step record under phases/<id>/attempts/<n>/. None
        # before the first call.
        self._last_query: "QueryRecord | None" = None

    async def _query_json(
        self,
        system: str,
        prompt: str,
        *,
        tools: list[str] | None = None,
        max_turns: int = 1,
        prompt_role: str = "system",
        timeout: float | None = None,
        reprompt_on_parse_failure: bool = True,
        thinking_budget: int = 8000,
    ) -> tuple[dict, str]:
        """Call ``_query`` and parse the response as a JSON object.

        Returns ``(parsed_dict, raw_response_text)``. When parsing fails and
        ``reprompt_on_parse_failure`` is True, sends one short follow-up to
        the model with the broken response and an instruction to emit valid
        JSON only. The reprompt is single-turn and uses the smallest tool
        set — significantly cheaper than the original call but recovers many
        otherwise-silent failures.
        """
        response_text = await self._query(
            system=system,
            prompt=prompt,
            tools=tools,
            max_turns=max_turns,
            prompt_role=prompt_role,
            timeout=timeout,
            thinking_budget=thinking_budget,
        )
        parsed = _extract_json(response_text)
        if parsed:
            return parsed, response_text
        if not response_text.strip():
            return {}, response_text
        if not reprompt_on_parse_failure:
            logger.warning(
                "json reprompt disabled; first response (%d chars) was unparseable: %s",
                len(response_text), response_text[:300].replace("\n", " ⏎ "),
            )
            return {}, response_text

        # Don't run the reformat reprompt on responses that weren't even
        # attempting to be JSON. The reformatter is meant to fix slightly-
        # broken JSON (missing comma, unescaped quote) — given pure prose
        # like "I'll inspect each file..." it hallucinates a JSON shape
        # from nothing. Saw this in practice when a CLI crash truncated
        # a verifier mid-thought; the reformat invented
        # ``{"status": "pending_verification", ...}`` out of narrative.
        # Heuristic: if there's no ``{`` AND no fenced ``json`` block,
        # this is narrative, not broken JSON.
        looks_like_json_attempt = (
            "{" in response_text
            or "```json" in response_text
        )
        if not looks_like_json_attempt:
            logger.warning(
                "json parse failed (%d chars) and response contains no JSON-shaped "
                "content — skipping reformat to avoid hallucinated structure. "
                "Snippet: %s",
                len(response_text),
                response_text[:200].replace("\n", " ⏎ "),
            )
            return {}, response_text

        logger.warning(
            "json parse failed (%d chars); requesting a clean JSON reformat",
            len(response_text),
        )
        reformat_system = (
            "You produce ONLY a single JSON object. No prose. No markdown code "
            "fences. No leading or trailing text. The first character of your "
            "response MUST be `{` and the last MUST be `}`. Inside string "
            "values, every newline must be written as \\n and every tab as "
            "\\t (do not include raw newlines/tabs inside strings). Do not "
            "wrap the JSON in any extra structure."
        )
        reformat_prompt = (
            "The following response was supposed to be a JSON object but could "
            "not be parsed. Re-emit the SAME content as a single, strict, "
            "valid JSON object. Do not change semantics — just fix the JSON.\n\n"
            f"--- BROKEN RESPONSE ---\n{response_text}\n--- END ---\n"
        )
        try:
            retry_text = await self._query(
                system=reformat_system,
                prompt=reformat_prompt,
                tools=None,
                max_turns=1,
                prompt_role=f"{prompt_role}-reformat",
                timeout=min(timeout or 120.0, 120.0),
            )
        except Exception as e:
            logger.warning("json reformat call failed: %s", e)
            return {}, response_text
        parsed_retry = _extract_json(retry_text)
        if parsed_retry:
            logger.info("json reformat recovered (%d chars)", len(retry_text))
            return parsed_retry, retry_text
        logger.warning(
            "json reformat still unparseable (%d chars): %s",
            len(retry_text), retry_text[:300].replace("\n", " ⏎ "),
        )
        return {}, retry_text

    async def create_top_level_spec(self, paper_path: Path, store: SpecStore) -> SpecManager:
        """Stage 1 of 3: author the slim top-level skeleton (spec.md + state.json).

        The skeleton names the implementation-requiring sections, captures the
        DAG between them, and lists shared invariants (hyperparameters, datasets,
        eval metrics). It does NOT contain per-section implementation detail —
        that is authored separately by ``create_section_specs`` once this
        returns.

        Returns the resulting ``SpecManager`` so callers can feed it into the
        section fan-out.
        """
        logger.info("Authoring top-level skeleton from %s", paper_path)
        if self.emitter:
            self.emitter.emit(
                "agent_started",
                agent_id="orchestrator",
                parent_id=None,
                kind="orchestrator",
                title="Orchestrator",
            )
            self.emitter.emit(
                "skeleton_started",
                agent_id="orchestrator",
                parent_id=None,
                paper_path=str(paper_path),
            )

        from ..llm.paper import get_page_count
        try:
            page_count = get_page_count(paper_path)
        except Exception:
            logger.exception("get_page_count failed; defaulting to 20")
            page_count = 20

        store.spec_dir.mkdir(parents=True, exist_ok=True)
        spec_md_path = store.spec_md_path
        state_json_path = store.state_path

        # Pre-clear any stale outputs from a prior failed run so we
        # detect "model never wrote them" cleanly.
        for p in (spec_md_path, state_json_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                logger.warning("create_top_level_spec: could not unlink stale %s", p)

        absolute_paper_path = Path(paper_path).resolve()
        prompt = (
            f"Read the research paper at:\n  {absolute_paper_path}\n\n"
            f"It has {page_count} pages. Use the ``Read`` tool with the "
            f"``pages`` parameter (max 20 pages per call).\n\n"
            f"Then produce these two files exactly:\n"
            f"  - ``{spec_md_path}`` — slim markdown skeleton (format in your system prompt; UNDER 200 LINES).\n"
            f"  - ``{state_json_path}`` — strictly valid JSON matching the schema in your system prompt.\n\n"
            f"Use ``Write`` to create each file. Many small writes are FINE — "
            f"they're the checkpointing mechanism. When both files are written, stop."
        )
        await self._query(
            system=SPEC_CREATION_SYSTEM_PROMPT,
            prompt=prompt,
            tools=["Read", "Write", "Edit", "Glob", "Grep"],
            max_turns=40,
            timeout=600,
            thinking_budget=4000,
        )

        if not spec_md_path.exists():
            raise RuntimeError(
                f"create_top_level_spec: model did not write {spec_md_path}."
            )
        if not state_json_path.exists():
            raise RuntimeError(
                f"create_top_level_spec: model did not write {state_json_path}."
            )
        try:
            raw_state = json.loads(state_json_path.read_text() or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"create_top_level_spec: model wrote invalid JSON to {state_json_path}: {e}"
            )

        raw_metadata = raw_state.get("metadata", {}) or {}
        metadata = SpecMetadata(
            paper_id=raw_metadata.get("paper_id") or paper_path.stem,
            paper_title=raw_metadata.get("paper_title") or "Unknown",
        )
        phases = []
        for p in raw_state.get("phases", []) or []:
            phases.append(PhaseState(
                phase_id=p.get("phase_id", ""),
                title=p.get("title", ""),
                kind=_coerce_kind(p.get("kind")),
                inputs=[Artifact(**a) for a in p.get("inputs", []) or []],
                outputs=[Artifact(**a) for a in p.get("outputs", []) or []],
                max_debug_attempts=self.config.max_debug_attempts,
            ))
        dep_graph = dict(raw_state.get("dependency_graph", {}) or {})
        plan = _parse_plan(raw_state.get("plan"))

        spec_md = spec_md_path.read_text()

        state = SpecState(
            metadata=metadata,
            phases=phases,
            dependency_graph=dep_graph,
            plan=plan,
        )

        graph = DependencyGraph.from_spec_state(state)
        phase_ids = {p.phase_id for p in phases}
        errors = graph.validate(phase_ids)
        if errors:
            logger.warning("Dependency graph validation issues: %s", errors)

        store.save_spec_md(spec_md)
        store.save_state(state)
        if state.plan is not None:
            errors = _validate_plan(state.plan, phase_ids)
            if errors:
                logger.warning("Plan validation issues: %s", errors)
            store.save_plan(state.plan)
        store.append_revision(Revision(
            event_type=EventType.spec_created,
            rationale=f"Initial skeleton created from {paper_path.name}",
        ))

        logger.info(
            "Skeleton created: %d phases, %d lines spec.md",
            len(phases), spec_md.count("\n") + 1,
        )

        if self.emitter:
            self.emitter.emit(
                "skeleton_completed",
                agent_id="orchestrator",
                parent_id=None,
                phase_count=len(phases),
                spec_md_lines=spec_md.count("\n") + 1,
            )
            emit_artifact_created(
                self.emitter,
                agent_id="orchestrator",
                artifact_type="top_level_spec",
                path=spec_md_path,
                producer="skeleton",
            )

        return SpecManager(store, state)

    async def create_section_specs(
        self,
        spec_manager: SpecManager,
        paper_path: Path,
        store: SpecStore,
    ) -> list[SectionSpec]:
        """Stage 2 of 3: fan out to per-section authors (one task per phase).

        Each section author reads the paper section it owns and writes a JSON
        file at ``canonical_spec/sections/<phase_id>.json`` matching the
        ``SectionSpec`` schema. Citations on every acceptance criterion are
        required (validated post-hoc — a missing-citations spec is returned but
        flagged for re-authoring).
        """
        phases = list(spec_manager.state.phases)
        if not phases:
            logger.warning("create_section_specs: no phases in spec — nothing to author")
            return []

        absolute_paper_path = Path(paper_path).resolve()
        sections_dir = store.sections_dir
        sections_dir.mkdir(parents=True, exist_ok=True)

        async def _author_one(phase: PhaseState) -> SectionSpec | None:
            json_path = store._section_json_path(phase.phase_id)
            md_path = store._section_md_path(phase.phase_id)
            # Pre-clear stale outputs so we detect "didn't write" cleanly.
            for p in (json_path, md_path):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

            if self.emitter:
                self.emitter.emit(
                    "section_spec_started",
                    agent_id=f"section_author:{phase.phase_id}",
                    parent_id="orchestrator",
                    phase_id=phase.phase_id,
                    title=phase.title,
                )

            prompt = (
                f"Author the per-section spec for ``{phase.phase_id}`` "
                f"(title: ``{phase.title}``).\n\n"
                f"Paper PDF: ``{absolute_paper_path}``\n"
                f"Top-level skeleton: ``{store.spec_md_path}`` (read first — your spec "
                f"references its shared invariants table by name).\n\n"
                f"Write your JSON output to exactly this path:\n"
                f"  ``{json_path}``\n\n"
                f"Required fields: phase_id, title, goal, spec_markdown, "
                f"acceptance_criteria (every entry MUST have source.page), citations."
            )
            try:
                await self._query(
                    system=SECTION_SPEC_PROMPT,
                    prompt=prompt,
                    tools=["Read", "Write", "Edit", "Glob", "Grep"],
                    max_turns=20,
                    timeout=300,
                    thinking_budget=2000,
                )
            except Exception:
                logger.exception(
                    "create_section_specs: author for %s raised", phase.phase_id,
                )
                return None

            if not json_path.exists():
                logger.warning(
                    "create_section_specs: author for %s did not write %s",
                    phase.phase_id, json_path,
                )
                return None
            try:
                raw = json.loads(json_path.read_text() or "{}")
            except json.JSONDecodeError as e:
                logger.warning(
                    "create_section_specs: %s wrote invalid JSON: %s",
                    phase.phase_id, e,
                )
                return None

            try:
                spec = _build_section_spec(raw, phase)
            except Exception:
                logger.exception(
                    "create_section_specs: could not build SectionSpec from %s output",
                    phase.phase_id,
                )
                return None

            # Persist via the typed storage path (writes both .md and .json).
            store.save_section_spec(spec)

            if self.emitter:
                self.emitter.emit(
                    "section_spec_completed",
                    agent_id=f"section_author:{phase.phase_id}",
                    parent_id="orchestrator",
                    phase_id=phase.phase_id,
                    path=str(store._section_md_path(phase.phase_id)),
                    criteria_count=len(spec.acceptance_criteria),
                )
                emit_artifact_created(
                    self.emitter,
                    agent_id=f"section_author:{phase.phase_id}",
                    parent_id="orchestrator",
                    artifact_type="section_spec",
                    path=store._section_md_path(phase.phase_id),
                    producer="section_author",
                    phase_id=phase.phase_id,
                )

            return spec

        results = await asyncio.gather(*(_author_one(p) for p in phases))
        successful = [s for s in results if s is not None]
        logger.info(
            "Section authoring: %d/%d phases succeeded", len(successful), len(phases),
        )
        return successful

    async def critique_section_specs(
        self,
        section_specs: list[SectionSpec],
        paper_path: Path,
        store: SpecStore,
    ) -> list[SectionCritique]:
        """Stage 3 of 3: critic agent re-reads paper, judges each section spec.

        Runs in parallel — each critique is independent. Verdicts are persisted
        as ``canonical_spec/sections/<phase_id>.critique.json``. Citation
        validation runs first (cheap) — a spec that fails ``validate_citations``
        is auto-marked ``missing_citations`` without a critic call.
        """
        if not section_specs:
            return []

        absolute_paper_path = Path(paper_path).resolve()

        async def _critique_one(spec: SectionSpec) -> SectionCritique | None:
            # Cheap pre-check: enforce citation presence before spending an LLM call.
            try:
                spec.validate_citations()
            except ValueError as e:
                critique = SectionCritique(
                    phase_id=spec.phase_id,
                    verdict=CritiqueVerdict.missing_citations,
                    reasons=[str(e)],
                )
                store.save_section_critique(critique)
                if self.emitter:
                    self.emitter.emit(
                        "section_spec_critiqued",
                        agent_id=f"section_critic:{spec.phase_id}",
                        parent_id="orchestrator",
                        phase_id=spec.phase_id,
                        verdict=critique.verdict.value,
                    )
                return critique

            critique_path = store._section_critique_path(spec.phase_id)
            section_json_path = store._section_json_path(spec.phase_id)
            try:
                if critique_path.exists():
                    critique_path.unlink()
            except OSError:
                pass

            prompt = (
                f"Critique the section spec for ``{spec.phase_id}``.\n\n"
                f"Section spec JSON: ``{section_json_path}``\n"
                f"Top-level spec.md (for invariant:// references): ``{store.spec_md_path}``\n"
                f"Paper PDF: ``{absolute_paper_path}``\n\n"
                f"Write your critique JSON to exactly this path:\n"
                f"  ``{critique_path}``\n\n"
                f"Verdict enum: verified | questionable | missing_citations."
            )
            try:
                await self._query(
                    system=SPEC_CRITIC_PROMPT,
                    prompt=prompt,
                    tools=["Read", "Write"],
                    max_turns=10,
                    timeout=180,
                    thinking_budget=1000,
                )
            except Exception:
                logger.exception(
                    "critique_section_specs: critic for %s raised", spec.phase_id,
                )
                return None

            if not critique_path.exists():
                logger.warning(
                    "critique_section_specs: critic for %s did not write %s",
                    spec.phase_id, critique_path,
                )
                return None
            try:
                raw = json.loads(critique_path.read_text() or "{}")
            except json.JSONDecodeError as e:
                logger.warning(
                    "critique_section_specs: %s critic wrote invalid JSON: %s",
                    spec.phase_id, e,
                )
                return None

            try:
                critique = SectionCritique.model_validate(raw)
            except Exception:
                logger.exception(
                    "critique_section_specs: invalid critique JSON for %s",
                    spec.phase_id,
                )
                return None

            # Deterministic safety net for sweep language. The LLM critic has
            # missed this multiple times (Adam paper, VAE); the regex never
            # does. If we find any sweep phrase in the spec markdown and the
            # critic still verdicted ``verified``, force-downgrade so the
            # Builder is held back until the spec is rewritten.
            sweep_findings = _find_sweep_phrases(spec.spec_markdown)
            if sweep_findings and critique.verdict == CritiqueVerdict.verified:
                logger.warning(
                    "critique_section_specs: %s spec contains sweep phrases %s — "
                    "downgrading verdict from `verified` to `questionable`",
                    spec.phase_id, sweep_findings,
                )
                critique = SectionCritique(
                    phase_id=critique.phase_id,
                    verdict=CritiqueVerdict.questionable,
                    reasons=list(critique.reasons) + [
                        (
                            "Sweep-shaped language detected in spec_markdown: "
                            + ", ".join(sweep_findings)
                            + ". The Builder will turn this into a "
                            "`for lr in [...]: train()` loop and burn GPU. "
                            "Rewrite each instance as N point-sampled "
                            "criteria (one per cell/curve the paper reports) "
                            "at the paper's pinned config — see the "
                            "\"No hyperparameter sweeps\" rule in the "
                            "Builder prompt. If the paper doesn't pin a "
                            "winner, instruct the Builder to call "
                            "`report_result(is_spec_issue=true)` rather "
                            "than search."
                        )
                    ],
                )

            store.save_section_critique(critique)
            if self.emitter:
                self.emitter.emit(
                    "section_spec_critiqued",
                    agent_id=f"section_critic:{spec.phase_id}",
                    parent_id="orchestrator",
                    phase_id=spec.phase_id,
                    verdict=critique.verdict.value,
                )
                emit_artifact_created(
                    self.emitter,
                    agent_id=f"section_critic:{spec.phase_id}",
                    parent_id="orchestrator",
                    artifact_type="section_critique",
                    path=critique_path,
                    producer="section_critic",
                    phase_id=spec.phase_id,
                    verdict=critique.verdict.value,
                )
            return critique

        results = await asyncio.gather(*(_critique_one(s) for s in section_specs))
        return [c for c in results if c is not None]

    #
    # Replaces the Python-sequenced run_pipeline body. One long-running
    # Claude Agent SDK query() loop with orchestrator tools (write_skeleton,
    # author_section_specs, start_phase, request_user_approval, etc.) — the
    # model decides what to do next, narrates each step naturally, and
    # pauses for the user via tool calls rather than wait_for_gate. Heavy
    # lifting (parallel section authoring, per-phase retries) still happens
    # in Python — tools wrap the existing async functions.

    async def run_as_orchestrator(
        self,
        runtime,
    ) -> tuple[bool, str]:
        """Run the model-driven orchestrator loop until terminal.

        ``runtime`` is an OrchestratorRuntime carrying shared state (config,
        paper path, store, workspace, execution loop, approval queue). The
        loop terminates when the model calls ``pipeline_complete`` or
        ``pipeline_failed``, or after a hard turn cap.

        Returns ``(success, final_message)``. Streams every TextBlock as an
        agent_message event (assistant role), every thinking_delta as an
        agent_thinking event, and every tool call as process_started so the
        web chat surface lights up exactly like Claude Code.
        """
        from ..access_tools import (
            ACCESS_TOOL_NAMES,
            create_access_server,
            make_chat_approval_callback,
        )
        from .runtime_tools import (
            ORCHESTRATOR_TOOL_NAMES,
            create_orchestrator_tools,
        )

        tool_server = create_orchestrator_tools(runtime)
        # Route access-tool approvals through the same chat queue as
        # request_user_approval so the web UI + inline viewer get a real
        # gate banner instead of a hung click.prompt. The sub-agent picks
        # up the same callback via runtime.execution_loop below.
        access_approval_callback = make_chat_approval_callback(
            runtime.approval_queue,
            getattr(self, "emitter", None),
        )
        runtime.execution_loop._access_approval_callback = access_approval_callback  # type: ignore[attr-defined]
        access_server = create_access_server(
            self.config,
            cwd=self.config.project_root,
            approval_callback=access_approval_callback,
        )

        # The initial user message kicks off the agent loop. The model's
        # system prompt instructs it to introduce itself + ask "ready to go?"
        # before doing anything.
        user_kickoff = (
            f"Begin the run for paper: {runtime.paper_path.name} "
            f"(at {runtime.paper_path.resolve()}). Introduce yourself in chat "
            f"first, ask the user to confirm they're ready, then follow the "
            f"workflow in your system prompt."
        )

        options = ClaudeAgentOptions(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            # Without this, the SDK inherits the bundled CLI's default model,
            # which currently routes to Opus 4.7 and crashes immediately with
            # "API Error: 400 role 'system' is not supported on this model"
            # because the SDK still posts the system prompt as a message role.
            # See Config.model for the same workaround.
            model=self.config.model,
            # User-allowlisted dirs from --allow-dir; ad-hoc paths go through
            # the mcp__access__read_outside_workspace tool (see runtime_tools).
            add_dirs=[str(p) for p in self.config.extra_allowed_dirs],
            allowed_tools=ORCHESTRATOR_TOOL_NAMES + ACCESS_TOOL_NAMES,
            mcp_servers={"orchestrator": tool_server, "access": access_server},
            max_turns=400,
            # Extended thinking on — the orchestrator's decisions about
            # ordering and gate prompts benefit from a few hundred tokens
            # of reasoning per turn.
            thinking={"type": "enabled", "budget_tokens": 2000},
            include_partial_messages=True,
        )

        agent_id = "orchestrator"
        if self.emitter:
            self.emitter.emit(
                "agent_started",
                agent_id=agent_id,
                parent_id=None,
                kind="orchestrator",
                title="Orchestrator",
            )

        result_message_seen = False
        try:
            async for message in query(prompt=user_kickoff, options=options):
                # Stream events carry thinking deltas + tool I/O lifecycle —
                # we surface thinking_delta to the chat so the user sees
                # reasoning the moment it appears.
                if isinstance(message, StreamEvent):
                    evt = message.event
                    ev_type = evt.get("type")
                    if ev_type == "content_block_delta":
                        delta = evt.get("delta") or {}
                        if delta.get("type") == "thinking_delta":
                            text = delta.get("thinking") or ""
                            if text.strip() and self.emitter:
                                self.emitter.emit(
                                    "agent_thinking",
                                    agent_id=agent_id,
                                    parent_id=None,
                                    text=text[:500],
                                )
                    continue

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            # Already emitted via stream events above; skip
                            # to avoid duplicate chat lines.
                            continue
                        if isinstance(block, TextBlock):
                            if block.text.strip() and self.emitter:
                                self.emitter.emit(
                                    "agent_message",
                                    agent_id=agent_id,
                                    parent_id=None,
                                    role="assistant",
                                    text=block.text[:4000],
                                )
                        elif isinstance(block, ToolUseBlock):
                            if self.emitter:
                                self.emitter.emit(
                                    "process_started",
                                    agent_id=agent_id,
                                    parent_id=None,
                                    process_id=block.id,
                                    tool_name=block.name,
                                    summary=_format_orchestrator_tool(block),
                                )
                                capture_file_before(
                                    self.emitter,
                                    process_id=block.id,
                                    tool_name=block.name,
                                    file_path=(block.input or {}).get("file_path"),
                                    cwd=self.config.project_root,
                                )
                    continue

                if isinstance(message, UserMessage):
                    # Tool results land here. Emit process_result so the chat
                    # can show ✓ / ✗ next to the tool call.
                    for block in message.content:
                        if isinstance(block, ToolResultBlock) and self.emitter:
                            output = _stringify_tool_result(block.content)
                            self.emitter.emit(
                                "process_result",
                                agent_id=agent_id,
                                parent_id=None,
                                process_id=block.tool_use_id,
                                is_error=bool(block.is_error),
                                output=output[:2000],
                            )
                            emit_file_write(
                                self.emitter,
                                agent_id=agent_id,
                                parent_id=None,
                                process_id=block.tool_use_id,
                                is_error=bool(block.is_error),
                            )
                    continue

                if isinstance(message, ResultMessage):
                    result_message_seen = True
                    continue

                if isinstance(message, SystemMessage):
                    continue

                if isinstance(message, RateLimitEvent):
                    logger.warning(
                        "run_as_orchestrator: rate limit event: %s", message,
                    )
                    continue

                logger.debug(
                    "run_as_orchestrator: unhandled message type=%s",
                    type(message).__name__,
                )

                # The model may call pipeline_complete / pipeline_failed at any
                # turn — once they fire, the tool sets runtime.done; we break
                # out of the loop on the next message.
                if runtime.done:
                    break

        except Exception as e:
            # If the loop crashes after the model already called a terminal
            # tool, the work is real — swallow late SDK errors.
            if runtime.done or result_message_seen:
                logger.warning(
                    "run_as_orchestrator: late exception ignored (done=%s, result_seen=%s): %s",
                    runtime.done, result_message_seen, e,
                )
            else:
                logger.exception("run_as_orchestrator: loop crashed")
                runtime.success = False
                runtime.final_message = f"Orchestrator crashed: {type(e).__name__}: {e}"

        if not runtime.done:
            # Model exited the loop without calling a terminal tool — treat
            # as failure with a generic message.
            runtime.success = False
            if not runtime.final_message:
                runtime.final_message = (
                    "Orchestrator stopped without calling pipeline_complete "
                    "or pipeline_failed."
                )

        if self.emitter:
            self.emitter.emit(
                "agent_completed",
                agent_id=agent_id,
                parent_id=None,
                status="success" if runtime.success else "failed",
                message=runtime.final_message,
            )

        return runtime.success, runtime.final_message

    async def chat(self, message: str, spec_manager: "SpecManager | None" = None) -> str:
        """Side-channel conversational query against the orchestrator.

        Used by the agent-terminal viewer's chat pane: the user sends a message
        targeted at ``agent_id="orchestrator"``, the command listener routes it
        here, and the response is emitted as an ``agent_message`` event so the
        viewer renders it in the chat pane.

        This does NOT touch the running phase loop. It is read-only with
        respect to spec/state — purely a conversation about what the
        orchestrator currently knows.
        """
        context = ""
        if spec_manager is not None:
            try:
                phases_summary = "\n".join(
                    f"- {p.phase_id} [{p.status.value}]: {p.title}"
                    for p in spec_manager.state.phases
                )
                context = f"\n\nCurrent phase status:\n{phases_summary}"
            except Exception:
                context = ""

        system = (
            "You are the orchestrator of a multi-agent research-paper "
            "reproduction pipeline. The user is observing the run live "
            "through a terminal UI and is asking questions about the plan, "
            "the current phase status, or what to do next. Answer concisely. "
            "Do not invoke tools."
        )
        prompt = f"User says: {message}{context}"

        if self.emitter:
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="user",
                text=message[:2000],
            )
        try:
            response = await self._query(
                system=system, prompt=prompt, tools=[], max_turns=1, emit_messages=False,
            )
        except Exception as e:
            response = f"(orchestrator chat failed: {e})"
        if self.emitter:
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="assistant",
                text=response[:2000],
            )
        return response

    async def acceptance_review(
        self,
        phase_id: str,
        result: SubAgentResult,
        spec_manager: SpecManager,
        work_dir: Path | None = None,
    ) -> tuple[bool, str | None]:
        """Review a sub-agent's result for cross-phase compatibility + result quality.

        Now includes:
        - Claims verification against the ledger
        - Convergence sanity for training phases
        - Suspicious-result flagging
        """
        from .claims import verify_phase_claims

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

        # Run claims verification
        claims_ledger = spec_manager.store.load_claims()
        claims_report = verify_phase_claims(phase_id, result, claims_ledger, work_dir)

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

        # Add claims verification to the review context
        if claims_report.verifications:
            review_context["claims_verification"] = {
                "summary": {
                    "verified": claims_report.verified_count,
                    "close": claims_report.close_count,
                    "missed": claims_report.missed_count,
                    "exceeded": claims_report.exceeded_count,
                    "not_checked": claims_report.not_checked_count,
                },
                "details": [v.model_dump(mode="json") for v in claims_report.verifications],
            }

        logger.info("Starting acceptance review LLM call for phase=%s", phase_id)
        response_text = await self._query(
            system=ACCEPTANCE_REVIEW_SYSTEM_PROMPT,
            prompt=f"Review this phase result:\n\n```json\n{json.dumps(review_context, indent=2)}\n```",
            tools=["Read", "Bash", "Glob", "Grep"],
            # Bumped 3 → 12. Opus extended thinking burns turns on thinking
            # blocks before producing the JSON verdict — 3 turns guarantees
            # max_turns errors and triggers the doom loop downstream.
            max_turns=12,
            prompt_role=f"subagent-{phase_id}",
            # 300s was tight for paper-grounded JSON reviews; bumped to 600s
            # so a slow CLI stream doesn't time out before the final result.
            timeout=600,
            # Thinking disabled — acceptance review is JSON-output review.
            # Extended thinking risks CLI internal idle timeout (see
            # spec-creation comment for the empirical evidence).
            thinking_budget=0,
        )
        logger.info("Acceptance review LLM call completed for phase=%s (%d chars)", phase_id, len(response_text))

        parsed = _extract_json(response_text)
        accepted = parsed.get("accept", False)
        feedback = parsed.get("feedback")
        claims_notes = parsed.get("claims_notes")

        logger.info("Acceptance review for phase=%s: accepted=%s", phase_id, accepted)

        # Persist claims report to logs/
        if claims_report.verifications:
            try:
                claims_log_dir = self.config.logs_dir / "claims"
                claims_log_dir.mkdir(parents=True, exist_ok=True)
                report_path = claims_log_dir / f"{phase_id}.md"
                body = (
                    f"# Claims verification: phase `{phase_id}`\n\n"
                    f"{claims_report.to_markdown()}\n"
                )
                if claims_notes:
                    body += f"\n## Reviewer notes\n\n{claims_notes}\n"
                report_path.write_text(body)
            except Exception:
                logger.exception("Failed to write claims report for phase=%s", phase_id)

            if self.emitter:
                self.emitter.emit(
                    "agent_message",
                    agent_id="orchestrator",
                    parent_id=None,
                    role="system",
                    text=(
                        f"📊 Claims check for `{phase_id}`: "
                        f"{claims_report.verified_count} verified, "
                        f"{claims_report.close_count} close, "
                        f"{claims_report.missed_count} missed, "
                        f"{claims_report.exceeded_count} suspicious — "
                        f"see logs/claims/{phase_id}.md"
                    ),
                )

        return accepted, feedback

    # ──────────────────────────────────────────────────────────────────────
    # Per-section agent chain (Stage 2 of the redesign):
    #   refine_section → research_for_section (conditional) → builder → verify_section
    # ──────────────────────────────────────────────────────────────────────

    async def refine_section(
        self,
        phase_id: str,
        spec_manager: SpecManager,
        paper_path: Path,
    ) -> dict:
        """Plan Refiner: reads section spec + paper, returns enriched markdown
        + a list of questions for the Researcher.

        Returns dict: {refined_spec_md, summary, research_questions: [...]}.
        On failure, returns the original spec + no research questions.
        """
        from ..sub_agent.prompts import REFINER_SYSTEM_PROMPT
        phase = spec_manager.state.get_phase(phase_id)
        if phase is None:
            return {"refined_spec_md": "", "summary": "(unknown phase)", "research_questions": []}

        sub_spec = spec_manager.extract_sub_spec(phase_id, paper_path=str(paper_path.resolve()))
        try:
            from ..llm.paper import get_page_count
            page_count = get_page_count(paper_path)
        except Exception:
            page_count = 20

        prompt = (
            f"Refine the plan for section '{phase_id}' (paper: {paper_path}, "
            f"{page_count} pages).\n\n"
            f"## Section's current plan (markdown)\n\n{sub_spec.spec_markdown}\n\n"
            "Return the JSON object specified in your system prompt."
        )

        try:
            parsed, _ = await self._query_json(
                system=REFINER_SYSTEM_PROMPT,
                prompt=prompt,
                # Refiner only needs Read (paper PDF + the section spec
                # that's already inlined in the prompt). Bash/Glob/Grep
                # cost system-prompt tokens and tempt the model to
                # explore the filesystem — both slow it down for zero
                # benefit on this task.
                tools=["Read"],
                # 12 → 6: refiner should be "read a few pages, output
                # JSON". 6 turns is plenty (3 Read calls + 1 output =
                # 4 typical, 6 leaves headroom for one retry).
                max_turns=6,
                prompt_role=f"refine-{phase_id}",
                # 300s was tight for paper-grounded refinements + 12-turn
                # JSON output. Bumped to 600s after observed timeouts on
                # multi-page sections that triggered repeated retries.
                timeout=600,
                # Refiner is mechanical (read pages + structured output).
                # Extended thinking is wasted budget here — disabling it
                # cut LSTM refine time materially in early testing.
                thinking_budget=0,
            )
        except Exception as e:
            logger.warning("refine_section crashed for %s: %s", phase_id, e)
            return {
                "refined_spec_md": sub_spec.spec_markdown,
                "summary": f"(refiner crashed: {e})",
                "research_questions": [],
                "pending_approvals": [],
            }

        if not parsed:
            logger.warning("refine_section response could not be parsed for %s", phase_id)
            return {
                "refined_spec_md": sub_spec.spec_markdown,
                "summary": "(refiner response unparsed)",
                "research_questions": [],
                "pending_approvals": [],
            }
        return {
            "refined_spec_md": parsed.get("refined_spec_md") or sub_spec.spec_markdown,
            "summary": parsed.get("summary", ""),
            "research_questions": list(parsed.get("research_questions", [])),
            # Best-guess hyperparameter values the refiner flagged when the
            # paper didn't pin a winner — operator must confirm before the
            # Builder runs (see Step D of the "Kill sweep-shaped acceptance
            # criteria" workflow in REFINER_SYSTEM_PROMPT). Each entry is
            # {question, suggested_value, rationale, criterion, paper_pages_checked}.
            "pending_approvals": list(parsed.get("pending_approvals") or []),
            "estimated_runtime_minutes": parsed.get("estimated_runtime_minutes"),
        }

    async def research_for_section(
        self,
        phase_id: str,
        questions: list[str],
        spec_manager: SpecManager,
        paper_path: Path,
    ) -> dict:
        """Researcher: answers refiner's questions using citations + web.

        Returns dict: {research_notes_md, sources, summary}.
        On failure, returns empty notes.
        """
        from ..sub_agent.prompts import build_researcher_system_prompt
        if not questions:
            return {"research_notes_md": "", "sources": [], "summary": "(no questions)"}

        sub_spec = spec_manager.extract_sub_spec(phase_id, paper_path=str(paper_path.resolve()))
        system = build_researcher_system_prompt(sub_spec, questions)

        # Two-call sequence — same pattern as post_mortem / refine_spec:
        #   1. gather: tools enabled (Read/WebFetch/WebSearch), thinking=0.
        #      Each tool call checkpoints the API request. Output =
        #      raw findings per question as JSON.
        #   2. synthesize: no tools, thinking enabled. Consolidate the
        #      findings into the research_notes_md markdown.
        questions_block = "\n".join(f"- {q}" for q in questions)
        gather_prompt = (
            f"Gather raw findings for these research questions on section "
            f"'{phase_id}':\n\n{questions_block}\n\n"
            f"Use Read on the paper, WebFetch on URLs, WebSearch for "
            f"open queries, and lookup_citation for cited papers. "
            f"Iterate one question at a time.\n\n"
            f"## Schema\n\n"
            f"```jsonc\n"
            f"{{\n"
            f"  \"findings\": [\n"
            f"    {{\n"
            f"      \"question\": \"<the question being answered>\",\n"
            f"      // type: string\n"
            f"      \"answer\": \"<the synthesized answer>\",\n"
            f"      // type: string\n"
            f"      \"sources\": [\"<url or citation>\"]\n"
            f"      // type: list[string] (each item a single url/citation string)\n"
            f"    }}\n"
            f"  ]\n"
            f"  // type: list[object]\n"
            f"}}\n```"
        )
        try:
            gather_parsed, _ = await self._query_json(
                system=(
                    "You are a research agent gathering raw findings to "
                    "support a paper-reproduction task. Use tools liberally "
                    "to investigate. A separate synthesis call writes the "
                    "polished notes — your job is the raw findings.\n\n"
                    f"{STRUCTURED_JSON_CONTRACT}"
                ),
                prompt=gather_prompt,
                tools=["Read", "WebFetch", "WebSearch"],
                max_turns=20,
                prompt_role=f"research-gather-{phase_id}",
                thinking_budget=0,
                timeout=900,
            )
        except Exception as e:
            logger.warning("research-gather crashed for %s: %s", phase_id, e)
            return {"research_notes_md": "", "sources": [], "summary": f"(researcher gather crashed: {e})"}

        synthesize_prompt = (
            f"Synthesize the following raw findings into the research_notes_md "
            f"+ sources structure your system prompt specifies. Use ONLY "
            f"the findings below — you have no tools.\n\n"
            f"## Raw findings\n```json\n{json.dumps(gather_parsed, indent=2)[:60000]}\n```\n\n"
            f"Return the JSON object specified in your system prompt."
        )
        try:
            parsed, _ = await self._query_json(
                system=system,
                prompt=synthesize_prompt,
                tools=[],
                max_turns=1,
                prompt_role=f"research-synthesize-{phase_id}",
                thinking_budget=4000,
                timeout=600,
            )
        except Exception as e:
            logger.warning("research-synthesize crashed for %s: %s", phase_id, e)
            return {"research_notes_md": "", "sources": [], "summary": f"(researcher synth crashed: {e})"}

        if not parsed:
            return {"research_notes_md": "", "sources": [], "summary": "(researcher unparsed)"}
        return {
            "research_notes_md": parsed.get("research_notes_md", ""),
            "sources": list(parsed.get("sources", [])),
            "summary": parsed.get("summary", ""),
        }

    async def verify_section(
        self,
        phase_id: str,
        builder_result: SubAgentResult,
        spec_manager: SpecManager,
        work_dir: Path | None = None,
    ) -> tuple[bool, str | None, dict]:
        """Verify a section in two layers: deterministic checks, then a
        tool-free LLM judge.

        Returns ``(accept, feedback, verifier_payload)``. The payload always
        includes ``deterministic_checks`` (list of CheckResult dicts) and
        ``llm_judge`` (the raw JSON the judge emitted, or ``None`` when the
        judge wasn't reached because deterministic checks already failed).

        Fail-closed throughout: a verifier that crashes, times out, or emits
        unparseable JSON is treated as rejection, not acceptance. The old
        behavior auto-accepted on every failure mode and made the verifier
        a no-op exactly when it broke.
        """
        from .section_verifier import (
            build_judge_user_prompt,
            run_deterministic_checks,
            summarize_failures,
        )
        from ..sub_agent.prompts import VERIFIER_SYSTEM_PROMPT

        phase = spec_manager.state.get_phase(phase_id)
        if phase is None:
            return False, f"Unknown phase: {phase_id}", {
                "status": "unknown_phase",
                "accept": False,
                "deterministic_checks": [],
                "llm_judge": None,
            }

        # ── Layer 1: deterministic checks ─────────────────────────────────
        det_checks = run_deterministic_checks(
            phase_outputs=phase.outputs,
            builder_result=builder_result,
            work_dir=work_dir,
        )
        det_payload = [c.to_dict() for c in det_checks]
        det_failures = [c for c in det_checks if not c.passed]
        if det_failures:
            feedback = summarize_failures(det_checks)
            logger.info(
                "verify_section: %s rejected by deterministic checks (%d failure(s))",
                phase_id, len(det_failures),
            )
            if self.emitter:
                try:
                    self.emitter.emit(
                        "agent_tool",
                        agent_id="orchestrator",
                        parent_id=None,
                        summary=(
                            f"✗ verifier rejected {phase_id}: "
                            f"{len(det_failures)} deterministic check(s) failed"
                        ),
                    )
                except Exception:
                    pass
            return False, feedback, {
                "status": "missed",
                "accept": False,
                "feedback": feedback,
                "deterministic_checks": det_payload,
                "llm_judge": None,
            }

        # ── Layer 2: tool-free LLM judge ──────────────────────────────────
        sub_spec = spec_manager.extract_sub_spec(
            phase_id, paper_path=str(Path(self.config.paper_path).resolve()),
        )
        user_prompt = build_judge_user_prompt(
            phase_outputs=phase.outputs,
            builder_result=builder_result,
            work_dir=work_dir,
            acceptance_criteria_md=sub_spec.spec_markdown or "",
        )

        try:
            parsed, _ = await self._query_json(
                system=VERIFIER_SYSTEM_PROMPT,
                prompt=user_prompt,
                # tools=[] disables tools entirely — no Read/Bash loop, no
                # max_turns runaway. The judge sees everything inline.
                tools=[],
                max_turns=1,
                prompt_role=f"verify-{phase_id}",
                timeout=180,
                # Thinking disabled — verifier judge is structured JSON
                # output. Extended thinking risks tripping the bundled
                # CLI's ~5min internal stream-idle timeout (see
                # spec-creation comment for the full diagnosis).
                thinking_budget=0,
            )
        except Exception as e:
            err_descr = _describe_exception(e)
            logger.warning(
                "verify_section: LLM judge crashed for %s: %s — REJECTING (fail-closed)",
                phase_id, err_descr,
            )
            feedback = f"Verifier judge crashed: {err_descr}"
            return False, feedback, {
                "status": "judge_crashed",
                "accept": False,
                "feedback": feedback,
                "deterministic_checks": det_payload,
                "llm_judge": None,
            }

        if not parsed or "accept" not in parsed:
            payload_snippet = json.dumps(parsed, default=str)[:300] if parsed else "(empty)"
            logger.warning(
                "verify_section: LLM judge produced unparseable verdict for %s — "
                "REJECTING (fail-closed). Payload: %s",
                phase_id, payload_snippet,
            )
            return False, (
                "Verifier judge did not return a parseable verdict "
                "(missing or malformed 'accept' field)"
            ), {
                "status": "judge_unparsed",
                "accept": False,
                "feedback": "Verifier judge did not return a parseable verdict",
                "deterministic_checks": det_payload,
                "llm_judge": parsed or {},
            }

        accept = bool(parsed.get("accept", False))
        feedback = (parsed.get("feedback") or "").strip()
        if not accept and not feedback:
            # Reject without articulating why. Still reject (fail-closed),
            # but flag the verifier itself so an operator can investigate.
            logger.warning(
                "verify_section: %s rejected without feedback — rejecting anyway "
                "with placeholder feedback. Payload: %s",
                phase_id, json.dumps(parsed, default=str)[:300],
            )
            feedback = (
                "Verifier rejected this section but provided no feedback. "
                "Treating as a verifier failure — investigate the judge's "
                "prompt/response in the trace before retrying the section."
            )

        return accept, (None if accept else feedback), {
            **parsed,
            "deterministic_checks": det_payload,
            "llm_judge": parsed,
        }

    # ──────────────────────────────────────────────────────────────────────

    async def _extract_claims(
        self,
        paper_path: Path,
        page_count: int,
        valid_phase_ids: set[str],
    ) -> ClaimsLedger:
        """Extract structured numerical claims from the paper (second LLM pass).

        The orchestrator reads the PDF directly via Read tool — preserves
        tables (where most headline numerical claims live).

        Best-effort: returns an empty ledger on any failure.
        """
        if self.emitter:
            self.emitter.emit(
                "agent_thinking",
                agent_id="orchestrator",
                parent_id=None,
                text="Extracting numerical claims from paper...",
            )

        # File-driven extraction: model writes claims.json directly using
        # the plain Write/Edit tools. Each Write/Edit ends the current
        # API request and starts a fresh one — same checkpointing as
        # sub-agents. We use a draft path so an invalid model write
        # doesn't clobber an existing valid claims.json from a prior run.
        draft_path = self.config.spec_dir / "_claims_draft.json"
        self.config.spec_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale drafts so we detect "model never wrote anything" cleanly.
        try:
            if draft_path.exists():
                draft_path.unlink()
        except OSError:
            logger.warning("_extract_claims: could not unlink stale %s", draft_path)

        valid_ids_block = ", ".join(sorted(valid_phase_ids)) or "(none)"
        prompt = (
            f"Extract every numerical/quantitative claim from the research "
            f"paper at:\n  {paper_path}\n\n"
            f"It has {page_count} pages. Use the ``Read`` tool with the "
            f"``pages`` parameter (max 20 pages per call). Tables hold the "
            f"headline claims — read those pages carefully.\n\n"
            f"Write the claims to ``{draft_path}`` using ``Write`` (and "
            f"``Edit`` to append as you find more). Top-level JSON array, "
            f"schema in your system prompt. Many small writes/edits are "
            f"fine — each one checkpoints the API request.\n\n"
            f"Valid phase_ids to use when assigning claims to phases: "
            f"{valid_ids_block}. Leave phase_id empty if no phase fits."
        )
        try:
            await self._query(
                system=CLAIMS_EXTRACTION_SYSTEM_PROMPT,
                prompt=prompt,
                tools=["Read", "Write", "Edit"],
                # Plenty of turns for paper reads + Write/Edit iteration.
                max_turns=80,
                prompt_role="claims-extraction",
                timeout=900,
                # Thinking safe — Write/Edit checkpoints the request.
                thinking_budget=4000,
            )
        except Exception as e:
            logger.warning("Claims extraction LLM call failed: %s", e)
            # Fall through; we may still have a partial draft on disk.

        if not draft_path.exists():
            logger.warning(
                "_extract_claims: model did not write %s — returning empty ledger",
                draft_path,
            )
            return ClaimsLedger()

        try:
            raw = json.loads(draft_path.read_text() or "[]")
        except json.JSONDecodeError as e:
            logger.warning(
                "_extract_claims: %s contains invalid JSON (%s) — returning empty ledger",
                draft_path, e,
            )
            return ClaimsLedger()
        if not isinstance(raw, list):
            logger.warning(
                "_extract_claims: %s top-level is %s, expected list — returning empty ledger",
                draft_path, type(raw).__name__,
            )
            return ClaimsLedger()

        claims: list[Claim] = []
        for entry in raw:
            if not isinstance(entry, dict):
                logger.warning("Skipping non-dict claim entry: %r", entry)
                continue
            # Coerce stray phase_id to "" if it doesn't match an actual phase.
            raw_phase = (entry.get("phase_id") or "").strip()
            if raw_phase and raw_phase not in valid_phase_ids:
                entry["phase_id"] = ""
            try:
                claims.append(Claim.model_validate(entry))
            except Exception as e:
                logger.warning(
                    "Skipping invalid claim %r: %s",
                    entry.get("claim_id"), e,
                )
        return ClaimsLedger(claims=claims)

    async def _build_literature_context(
        self,
        paper_text: str,
        store: SpecStore,
    ) -> str:
        """Extract citations and resolve them via Semantic Scholar.

        Returns a markdown block to append to the spec-creation prompt, and
        also persists the full list to ``context/references.md``. Best-effort:
        returns empty string on any failure.
        """
        try:
            titles = extract_citation_titles(paper_text)
        except Exception:
            logger.exception("Failed to extract citations from paper text")
            return ""

        if not titles:
            logger.info("No citation titles extracted — skipping literature context")
            return ""

        # Cap to avoid excessive API calls — the first ~30 refs cover the
        # core related work section and methodological citations.
        titles = titles[:30]
        logger.info("Resolving %d cited papers via Semantic Scholar", len(titles))

        if self.emitter:
            self.emitter.emit(
                "agent_thinking",
                agent_id="orchestrator",
                parent_id=None,
                text=f"Resolving {len(titles)} cited papers via Semantic Scholar...",
            )

        try:
            client = SemanticScholarClient()
            papers = await client.resolve_citations(titles, max_concurrent=2)
        except Exception:
            logger.exception("Semantic Scholar batch resolve failed")
            return ""

        if not papers:
            logger.info("No cited papers resolved — skipping literature context")
            return ""

        logger.info("Resolved %d / %d cited papers", len(papers), len(titles))

        # Build the markdown block
        lines = [
            "## Literature Context",
            "",
            "The following abstracts were resolved from the paper's reference list. "
            "Use them to fill in details the paper leaves to cited work.",
            "",
        ]
        for p in papers:
            lines.append(p.to_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")

        block = "\n".join(lines)

        # Persist to context/references.md so the operator can browse it.
        try:
            ctx_dir = self.config.context_dir if hasattr(self.config, "context_dir") else store.spec_dir.parent / "context"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            (ctx_dir / "references.md").write_text(block)
        except Exception:
            logger.exception("Failed to write context/references.md")

        return block

    async def refine_spec(
        self,
        phase_id: str,
        trigger_diagnostics: dict,
        spec_manager: SpecManager,
        paper_path: Path,
    ) -> SpecAmendment:
        """Re-read the paper and produce an amended spec.md for one phase.

        ``trigger_diagnostics`` is whatever the caller wants the LLM to see —
        typically the sub-agent's spec_issue summary, the post-mortem hypothesis,
        and the relevant phase markdown excerpt.
        """
        phase = spec_manager.state.get_phase(phase_id)
        phase_title = phase.title if phase else phase_id
        current_spec = spec_manager.store.load_spec_md()

        prompt_payload = {
            "phase_id": phase_id,
            "phase_title": phase_title,
            "trigger": trigger_diagnostics,
            "paper_path": str(paper_path),
            "current_spec_md": current_spec,
        }

        # Two-call sequence — same shape as ``post_mortem``:
        #   1. evidence: read the paper sections relevant to the trigger,
        #      tools=Read, thinking=0, output=structured "what's wrong"
        #      dump as JSON. Each Read checkpoints the API request.
        #   2. verdict: no tools, bounded input, thinking enabled. Emit
        #      the SpecAmendment JSON. Fast because no tool loop.
        evidence_prompt = (
            f"You are gathering evidence for a spec amendment to phase "
            f"'{phase_id}'. Read the paper sections relevant to the "
            f"trigger below to identify what the paper actually says. "
            f"Do NOT propose an amendment yet.\n\n"
            f"## Trigger and current spec\n```json\n{json.dumps(prompt_payload, indent=2)[:60000]}\n```\n\n"
            f"Paper: ``{paper_path}``\n\n"
            f"## Schema\n\n"
            f"```jsonc\n{{\n"
            f"  \"current_spec_problem\": \"<what's ambiguous/contradictory/missing>\",\n"
            f"  // type: string  (required, non-empty)\n"
            f"  \"paper_says\": \"<verbatim or paraphrased text from the paper>\",\n"
            f"  // type: string\n"
            f"  \"page_refs\": [\"p.N\"],\n"
            f"  // type: list[string]  (each item: one page ref as a single string, e.g. \"p.5\")\n"
            f"  \"resolvable\": true,\n"
            f"  // type: bool\n"
            f"  \"reason_if_unresolvable\": \"\"\n"
            f"  // type: string  (empty string when resolvable=true)\n"
            f"}}\n```"
        )
        try:
            evidence_text = await self._query(
                system=(
                    "You are gathering evidence to support a spec "
                    "amendment. Read the paper sections that the trigger "
                    "implicates. Do NOT propose an amendment — a separate "
                    "call does that.\n\n"
                    f"{STRUCTURED_JSON_CONTRACT}"
                ),
                prompt=evidence_prompt,
                tools=["Read"],
                max_turns=8,
                prompt_role=f"refine-spec-evidence-{phase_id}",
                thinking_budget=0,
                timeout=600,
            )
        except Exception as e:
            logger.warning(
                "Spec-refinement evidence call failed for phase=%s: %s", phase_id, e,
            )
            return SpecAmendment(summary=f"(refinement evidence crashed: {e})")

        verdict_prompt = (
            f"Produce the amended spec.md for phase '{phase_id}'. Use ONLY "
            f"the evidence below — you have no tools.\n\n"
            f"## Trigger and current spec\n```json\n{json.dumps(prompt_payload, indent=2)[:60000]}\n```\n\n"
            f"## Evidence gathered\n```json\n{evidence_text}\n```\n\n"
            f"Return ONLY the JSON object specified in your system prompt."
        )
        try:
            response_text = await self._query(
                system=SPEC_REFINEMENT_SYSTEM_PROMPT,
                prompt=verdict_prompt,
                tools=[],
                max_turns=1,
                prompt_role=f"refine-spec-verdict-{phase_id}",
                # Deep reasoning IS the work here. Bounded input + no
                # tools means the call finishes well under the timeout
                # even with thinking enabled.
                thinking_budget=4000,
                timeout=600,
            )
        except Exception as e:
            logger.warning("Spec refinement verdict call failed for phase=%s: %s", phase_id, e)
            return SpecAmendment(summary=f"(refinement verdict crashed: {e})")

        parsed = _extract_json(response_text)
        if not parsed:
            return SpecAmendment(summary="(refinement response could not be parsed)")
        return SpecAmendment(
            amended_spec_md=parsed.get("amended_spec_md"),
            summary=parsed.get("summary", ""),
            sections_changed=list(parsed.get("sections_changed", [])),
        )

    async def post_mortem(
        self,
        phase_id: str,
        failed_result: SubAgentResult,
        work_dir: Path,
        spec_manager: SpecManager,
    ) -> PostMortem:
        """Diagnose a failed sub-agent attempt and produce a structured hypothesis.

        The full text response is persisted by the caller; this method only
        returns the parsed PostMortem. On any failure (LLM error, parse error)
        we return a low-confidence fallback so the loop never blocks.
        """
        phase = spec_manager.state.get_phase(phase_id)
        phase_title = phase.title if phase else phase_id

        context = {
            "phase_id": phase_id,
            "phase_title": phase_title,
            "attempt_dir": str(work_dir),
            "sub_agent_summary": failed_result.summary,
            "is_spec_issue_flag": failed_result.is_spec_issue,
            "diagnostics": failed_result.diagnostics or {},
            "test_report": {
                "tests_run": failed_result.test_report.tests_run,
                "tests_passed": failed_result.test_report.tests_passed,
                "tests_failed": failed_result.test_report.tests_failed,
                "failures": [
                    {"name": t.test_name, "message": t.message, "description": t.description}
                    for t in failed_result.test_report.test_details
                    if t.status.value != "passed"
                ],
            },
            "outputs_reported": [a.model_dump() for a in failed_result.outputs],
        }

        # Two-call sequence to keep individual API requests well under
        # the bundled CLI's ~5min stream-idle timeout:
        #
        #   1. evidence: read work_dir with tools, no thinking. Each
        #      file Read is a tool call that checkpoints the stream.
        #      Output = structured evidence dump as JSON.
        #
        #   2. verdict: no tools, bounded input (the evidence dump),
        #      thinking enabled for actual diagnosis. Bounded input
        #      means this call also finishes fast.
        #
        # Splitting like this lets us keep extended thinking on the part
        # of the work that actually needs it (the diagnosis) while
        # keeping every individual HTTP request short.

        evidence_prompt = (
            f"Gather forensic evidence on the failed attempt at phase "
            f"'{phase_id}'. Inspect ``{work_dir}`` — read relevant source "
            f"files, any logs/stderr, partial outputs, and the sub-agent's "
            f"own ``outputs/_result.json``. Do NOT diagnose yet — just evidence.\n\n"
            f"Sub-agent's own report:\n```json\n{json.dumps(context, indent=2)}\n```\n\n"
            f"## Schema\n\n"
            f"```jsonc\n{{\n"
            f"  \"key_files_inspected\": [\"src/path.py\"],\n"
            f"  // type: list[string]  (each item: one file path as a single string)\n"
            f"  \"observed_errors\": [\n"
            f"    {{\"file\": \"<path>\", \"line\": 0, \"message\": \"<text>\"}}\n"
            f"  ],\n"
            f"  // type: list[object]  (each: file string, line int, message string)\n"
            f"  \"failing_test_signatures\": [\"<test name or signature>\"],\n"
            f"  // type: list[string]\n"
            f"  \"partial_outputs_found\": [\"outputs/<path>\"],\n"
            f"  // type: list[string]\n"
            f"  \"suspicious_code_snippets\": [\n"
            f"    {{\"file\": \"<path>\", \"snippet\": \"<code>\", \"why\": \"<reason>\"}}\n"
            f"  ]\n"
            f"  // type: list[object]\n"
            f"}}\n```\n"
            f"Be terse. This is input to a follow-up diagnosis call — "
            f"include only what a diagnoser will need."
        )
        try:
            evidence_text = await self._query(
                system=(
                    "You are gathering evidence on a failed agent attempt. "
                    "Inspect files, identify what went wrong factually, and "
                    "return a structured evidence dump. Do NOT diagnose or "
                    "prescribe a fix — a separate call does that.\n\n"
                    f"{STRUCTURED_JSON_CONTRACT}"
                ),
                prompt=evidence_prompt,
                tools=["Read", "Glob", "Grep"],
                max_turns=15,
                prompt_role=f"postmortem-evidence-{phase_id}",
                thinking_budget=0,
                timeout=600,
            )
        except Exception as e:
            err_descr = _describe_exception(e)
            logger.warning(
                "Post-mortem evidence call failed for phase=%s: %s", phase_id, err_descr,
            )
            return PostMortem(
                failure_hypothesis=f"(post-mortem evidence unavailable: {err_descr})",
                confidence="low",
            )

        # Verdict call — no tools, bounded input, thinking allowed.
        verdict_prompt = (
            f"Diagnose the failure at phase '{phase_id}'. Use the evidence "
            f"below; do NOT inspect files (you have no tools). Produce a "
            f"focused hypothesis quoting specific symptoms from the evidence.\n\n"
            f"## Sub-agent's own report\n```json\n{json.dumps(context, indent=2)}\n```\n\n"
            f"## Evidence gathered\n```json\n{evidence_text}\n```\n\n"
            f"Return ONLY the JSON object specified in your system prompt."
        )
        try:
            response_text = await self._query(
                system=POST_MORTEM_SYSTEM_PROMPT,
                prompt=verdict_prompt,
                tools=[],
                max_turns=1,
                prompt_role=f"postmortem-verdict-{phase_id}",
                thinking_budget=4000,
                timeout=600,
            )
        except Exception as e:
            err_descr = _describe_exception(e)
            logger.warning("Post-mortem verdict call failed for phase=%s: %s", phase_id, err_descr)
            return PostMortem(
                failure_hypothesis=f"(post-mortem verdict unavailable: {err_descr})",
                confidence="low",
            )

        parsed = _extract_json(response_text)
        if not parsed:
            return PostMortem(
                failure_hypothesis="(post-mortem could not be parsed; see raw log)",
                suggested_fix=response_text[:500],
                confidence="low",
            )
        try:
            return PostMortem(
                failure_hypothesis=parsed.get("failure_hypothesis", ""),
                suggested_fix=parsed.get("suggested_fix", ""),
                is_likely_spec_issue=bool(parsed.get("is_likely_spec_issue", False)),
                confidence=parsed.get("confidence", "medium"),
            )
        except Exception as e:
            logger.warning("Post-mortem parse failed: %s", e)
            return PostMortem(
                failure_hypothesis="(post-mortem malformed)",
                confidence="low",
            )

    async def _query(
        self,
        system: str,
        prompt: str,
        # Reduced 3 → 1. SDK-level retries compound with phase-level retries
        # in FailureHandler (default 3) — worst case 3×3 = 9 full re-runs of a
        # phase on transient failures. The phase retry already re-dispatches
        # the sub-agent fresh, so SDK retries here are double-counting.
        max_retries: int = 1,
        tools: list[str] | None = None,
        max_turns: int = 1,
        emit_messages: bool = True,
        prompt_role: str = "system",
        timeout: float | None = None,
        # Extended thinking budget (input tokens the model can burn on
        # internal reasoning before producing output). Default 8000 is
        # right for hard decisions (acceptance review, post-mortem, plan
        # rewrites). For mechanical tasks (refine = "read pages → output
        # JSON") thinking is wasted budget — pass 0 to disable.
        thinking_budget: int = 8000,
        # Optional MCP servers to expose to the model. Used by spec
        # creation (and other future calls) to break monolithic JSON
        # output into a series of tool calls — each tool call ends the
        # current API request, avoiding the bundled CLI's ~5min idle
        # timeout that fires during long silent synthesis phases.
        mcp_servers: dict | None = None,
    ) -> str:
        """Run a single query and return the text response.

        ``timeout`` caps the wall-clock time (seconds) for a single attempt.
        On timeout the accumulated text is returned (or "" if nothing yet).

        ``prompt_role`` controls how the prompt is tagged in the chat pane.
        Default ``"system"`` covers harness-built instructions (paper ingest,
        spec creation). Callers relaying a sub-agent result up for review
        should pass e.g. ``"subagent-training"``.
        """
        import asyncio

        wall_started_at = time.time()
        captured_cost_usd: float | None = None
        captured_input_tokens: int | None = None
        captured_output_tokens: int | None = None

        if self.emitter and emit_messages:
            # Show the actual system prompt in the chat pane so the operator
            # can see exactly how the orchestrator was instructed.
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role="system",
                text=system[:4000],
            )
            # Then the user-turn prompt (paper text, acceptance review JSON,
            # etc.) tagged with whatever role the caller requested.
            self.emitter.emit(
                "agent_message",
                agent_id="orchestrator",
                parent_id=None,
                role=prompt_role,
                text=prompt[:2000],
            )
            # Heartbeat: emit a visible "working…" line right away so the chat
            # pane shows activity during long LLM calls.
            self.emitter.emit(
                "agent_thinking",
                agent_id="orchestrator",
                parent_id=None,
                text="✻ Calling LLM…",
            )
        # Trace writer for this call. Each attempt opens its own file so a
        # retried call doesn't clobber the failed attempt's trace.
        from ..storage.traces import TraceWriter
        traces_dir = self.config.project_root / "traces"

        # ``empty_timeout_retried`` is a one-shot bonus retry budget that
        # fires ONLY when the CLI streamed for the full timeout window but
        # produced zero usable text (no result_text, no ResultMessage). That
        # specific shape is transient infra (Anthropic API hiccup, bundled
        # CLI socket reset) and almost always succeeds on a fresh attempt.
        # It does NOT count against ``max_retries`` so callers' phase-level
        # retry math is unchanged. The same one-shot budget is shared with
        # the stall-watchdog retry path below — a query that was stall-
        # killed AND then empty-timed-out on the bonus attempt has hit
        # genuinely bad infra; we stop and let the caller degrade.
        empty_timeout_retried = False
        attempt = 0
        while attempt < max_retries:
            trace = TraceWriter(traces_dir, role="orchestrator", prompt_role=prompt_role, model=self.config.model)
            trace.open(system_prompt=system, prompt=prompt)
            # Hoist these to the outer scope so the except handler can read
            # them when the try body raises before they're populated.
            result_text = ""
            messages_received: list[str] = []
            # The CLI may stream AssistantMessage chunks then crash before
            # delivering the terminal ResultMessage. ``result_text`` is
            # non-empty in that case but it represents a partial reply, not
            # finished work. Tracking this separately lets the crash handler
            # distinguish "got real result" from "got partial mid-stream".
            result_message_seen = False
            # Set True when the stall-watchdog cancels the consume task —
            # see the watchdog body below. Hoisted so the outer except
            # handler can branch on it for the bonus retry.
            stall_killed = False
            try:
                stderr_lines: list[str] = []

                def capture_stderr(line: str) -> None:
                    stderr_lines.append(line)
                    # Persist EVERY stderr line to the trace, not just
                    # error-keyword matches — Node-side diagnostics
                    # (warnings about deprecated APIs, undici socket
                    # close codes, etc.) live in non-error lines and
                    # are exactly what we need to diagnose CLI crashes.
                    try:
                        trace.stderr_chunk(line)
                    except Exception:
                        pass
                    if "error" in line.lower() or "fatal" in line.lower() or "exception" in line.lower():
                        logger.error("CLI stderr: %s", line)

                # Build options. Extended thinking only when budget>0 — for
                # mechanical refine/extract calls we skip it to halve latency.
                _opts_kwargs: dict = dict(
                    system_prompt=system,
                    model=self.config.model,
                    permission_mode="bypassPermissions",
                    cwd=str(self.config.project_root),
                    add_dirs=[str(p) for p in self.config.extra_allowed_dirs],
                    max_turns=max_turns,
                    stderr=capture_stderr,
                    include_partial_messages=True,
                )
                if thinking_budget > 0:
                    _opts_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                if mcp_servers is not None:
                    _opts_kwargs["mcp_servers"] = mcp_servers
                options = ClaudeAgentOptions(**_opts_kwargs)
                if tools is not None:
                    options.allowed_tools = tools

                # result_text and messages_received are now hoisted to the
                # enclosing scope (above the try). The inner async closure
                # below uses ``nonlocal`` for result_text.

                # Live stream-state diagnostics. Tracks which content block
                # the model currently has open and how many of each delta
                # type arrived since the last heartbeat tick. Lets the
                # heartbeat tell a true stall from a long phase of silent
                # activity (input_json_delta for a giant tool call,
                # signature_delta after an extended-thinking block, etc.).
                stream_state: dict = {
                    "open_type":  None,   # "text" | "thinking" | "tool_use" | None
                    "open_name":  None,   # tool name when open_type == "tool_use"
                    "open_id":    None,   # tool_use id when open_type == "tool_use"
                    "open_since": 0.0,    # event-loop time when block opened
                    "deltas":     {},     # delta-type -> count since last tick
                }

                async def _consume_stream():
                    nonlocal result_text, result_message_seen
                    nonlocal captured_cost_usd, captured_input_tokens, captured_output_tokens
                    async for message in query(prompt=prompt, options=options):
                        msg_type = type(message).__name__
                        messages_received.append(msg_type)
                        # Record arrival time for stream-health diagnostics.
                        # Cheap (just a tuple) — gives us per-chunk gap data
                        # in the trace footer so we can see whether crashes
                        # follow a long gap (idle timeout) or arrive mid-
                        # stream (max-duration cut, RST).
                        trace.chunk_arrived(msg_type)
                        logger.debug("orchestrator _query: received %s (#%d)", msg_type, len(messages_received))
                        if isinstance(message, StreamEvent):
                            evt = message.event
                            ev_type = evt.get("type")
                            if ev_type == "content_block_start":
                                cb = evt.get("content_block") or {}
                                cb_type = cb.get("type")
                                stream_state["open_type"]  = cb_type
                                stream_state["open_name"]  = cb.get("name")
                                stream_state["open_id"]    = cb.get("id")
                                stream_state["open_since"] = asyncio.get_event_loop().time()
                                # Announce tool intent the instant the block
                                # opens, so the operator doesn't have to wait
                                # for the full AssistantMessage to land. The
                                # AssistantMessage path still fires later with
                                # the resolved file_path / command / etc.
                                if cb_type == "tool_use":
                                    name = cb.get("name") or "?"
                                    if self.emitter:
                                        self.emitter.emit(
                                            "agent_tool",
                                            agent_id="orchestrator",
                                            parent_id=None,
                                            summary=f"{name} (input streaming…)",
                                        )
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
                                        trace.thinking(text)
                                        if self.emitter:
                                            self.emitter.emit(
                                                "agent_thinking",
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                text=text[:500],
                                            )
                            continue
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, ThinkingBlock):
                                    # No emit — the consolidated ThinkingBlock
                                    # is the same content the streaming
                                    # ``thinking_delta`` path above already
                                    # surfaced piece by piece. Emitting both
                                    # produces a duplicate copy in any
                                    # subscriber that renders ``agent_thinking``
                                    # (e.g. the Activity tab).
                                    pass
                                elif isinstance(block, ToolUseBlock):
                                    trace.mark_first_tool()
                                    trace.tool_call(block.name, _format_tool_use(block))
                                    if self.emitter:
                                        self.emitter.emit(
                                            "agent_tool",
                                            agent_id="orchestrator",
                                            parent_id=None,
                                            summary=_format_tool_use(block),
                                        )
                                        # Surface tool uses as process_started so the
                                        # FilesView / Processes pane sees orchestrator
                                        # writes the same way it sees sub-agent writes.
                                        if block.name in ("Write", "Read", "Edit", "Bash"):
                                            self.emitter.emit(
                                                "process_started",
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                process_id=block.id,
                                                tool_name=block.name,
                                                summary=_format_tool_use(block),
                                                command=(block.input or {}).get("command", "")[:500] if block.name == "Bash" else None,
                                                file_path=(block.input or {}).get("file_path") if block.name in ("Write", "Read", "Edit") else None,
                                            )
                                        capture_file_before(
                                            self.emitter,
                                            process_id=block.id,
                                            tool_name=block.name,
                                            file_path=(block.input or {}).get("file_path"),
                                            cwd=self.config.project_root,
                                        )
                                    maybe_emit_paper_read(
                                        self.emitter,
                                        agent_id="orchestrator",
                                        parent_id=None,
                                        tool_name=block.name,
                                        tool_input=block.input or {},
                                        paper_path=self.config.paper_path,
                                    )
                                elif isinstance(block, TextBlock):
                                    result_text += block.text
                                    if block.text.strip():
                                        trace.mark_first_text()
                                        trace.assistant_text(block.text)
                                        if self.emitter:
                                            self.emitter.emit(
                                                "agent_thinking",
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                text=block.text[:500],
                                            )
                                else:
                                    # Unknown assistant block — surface so we
                                    # know about it instead of silently dropping.
                                    logger.warning(
                                        "orchestrator: unhandled assistant block type=%s",
                                        type(block).__name__,
                                    )
                                    if self.emitter:
                                        self.emitter.emit(
                                            "agent_tool",
                                            agent_id="orchestrator",
                                            parent_id=None,
                                            summary=f"[unhandled {type(block).__name__}]",
                                        )
                        elif isinstance(message, UserMessage):
                            # Tool results (Bash stdout, file contents, etc.) ride
                            # in here. Intentionally NOT surfaced into the chat —
                            # they'd flood the pane. We do still surface tool
                            # ERRORS so the operator notices breakage.
                            content = getattr(message, "content", None)
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, ToolResultBlock):
                                        text = _stringify_tool_result(block.content)
                                        is_err = bool(getattr(block, "is_error", False))
                                        if text:
                                            trace.tool_result(text, is_error=is_err)
                                        if self.emitter:
                                            # Pair with the earlier process_started
                                            # so the Trace view can show full tool
                                            # I/O on click — same shape sub-agent
                                            # emits.
                                            self.emitter.emit(
                                                "process_result",
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                process_id=block.tool_use_id,
                                                is_error=is_err,
                                                output=(text[:2000] if text else ""),
                                            )
                                            emit_file_write(
                                                self.emitter,
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                process_id=block.tool_use_id,
                                                is_error=is_err,
                                            )
                                            if is_err and text:
                                                self.emitter.emit(
                                                    "agent_tool",
                                                    agent_id="orchestrator",
                                                    parent_id=None,
                                                    summary="ERR " + text[:300],
                                                )
                        elif isinstance(message, ResultMessage):
                            result_message_seen = True
                            trace.mark_result()
                            if message.result:
                                result_text = message.result
                            msg_cost = getattr(message, "total_cost_usd", None)
                            if msg_cost is not None:
                                captured_cost_usd = float(msg_cost)
                            usage = getattr(message, "usage", None)
                            if usage is not None:
                                in_tok = usage.get("input_tokens") if isinstance(usage, dict) else getattr(usage, "input_tokens", None)
                                out_tok = usage.get("output_tokens") if isinstance(usage, dict) else getattr(usage, "output_tokens", None)
                                if in_tok is not None:
                                    captured_input_tokens = int(in_tok)
                                if out_tok is not None:
                                    captured_output_tokens = int(out_tok)
                        elif isinstance(message, SystemMessage):
                            logger.debug("orchestrator: SystemMessage subtype=%s", getattr(message, "subtype", "?"))
                        elif isinstance(message, RateLimitEvent):
                            info = getattr(message, "rate_limit_info", None)
                            _log_rate_limit("orchestrator", prompt_role, info, logger, self.emitter, trace)
                        else:
                            logger.warning(
                                "orchestrator: unhandled message type=%s",
                                type(message).__name__,
                            )

                # Heartbeat: a long LLM call (acceptance review, post-mortem,
                # spec creation) can be silent for minutes — print a one-line
                # "still alive" update every 20s so the user knows it's working.
                _hb_start = asyncio.get_event_loop().time()

                async def _heartbeat():
                    interval = 20.0
                    while True:
                        await asyncio.sleep(interval)
                        elapsed = asyncio.get_event_loop().time() - _hb_start
                        # Snapshot+reset the per-delta-type counts for this tick.
                        deltas_this_tick = stream_state["deltas"]
                        stream_state["deltas"] = {}
                        # Describe the currently-open content block, if any.
                        open_t = stream_state["open_type"]
                        if open_t:
                            held_s = asyncio.get_event_loop().time() - stream_state["open_since"]
                            name = stream_state["open_name"]
                            block_label = f"{open_t}:{name}" if name else open_t
                            block_str = f"open {block_label} ({held_s:.0f}s)"
                        else:
                            block_str = "no block open"
                        # Per-type delta tally — lets the operator tell
                        # "stalled" from "streaming input_json".
                        if deltas_this_tick:
                            delta_str = ", ".join(
                                f"+{n} {k}"
                                for k, n in sorted(deltas_this_tick.items(), key=lambda kv: -kv[1])
                            )
                        else:
                            delta_str = "no stream events"
                        last_msg = messages_received[-1] if messages_received else "(no messages yet)"
                        # Use print() so it bypasses logger filters and shows
                        # regardless of --verbose. Single line, easy to grep.
                        print(
                            f"  💓 [{prompt_role}] {elapsed:.0f}s · "
                            f"{block_str} · last {interval:.0f}s: {delta_str} · "
                            f"{len(messages_received)} msgs (last: {last_msg}) · "
                            f"{len(result_text)} chars",
                            flush=True,
                        )
                        # Also emit as a structured event so the web UI's
                        # heartbeat strip can replace its per-agent latest
                        # tick. Filtered out of the activity firehose by
                        # type — see frontend ActivityView RENDER_TYPES.
                        if self.emitter:
                            self.emitter.emit(
                                "heartbeat",
                                agent_id="orchestrator",
                                parent_id=None,
                                role=prompt_role,
                                elapsed_s=elapsed,
                                interval_s=interval,
                                open_block=block_str if open_t else None,
                                deltas=deltas_this_tick,
                                last_msg_type=last_msg,
                                msgs_count=len(messages_received),
                                result_chars=len(result_text),
                            )

                heartbeat_task = asyncio.create_task(_heartbeat())

                # Stream-stall watchdog. Logs progressive gap warnings AND
                # aborts the stream when the gap exceeds STALL_KILL_S. The
                # bundled CLI can wedge mid tool-call (input_json_delta
                # stops arriving and the stream just sits there until the
                # outer timeout fires). Without a hard kill, a 600s post-
                # mortem call eats the full 600s even when the stream has
                # been silent for 9+ minutes. With a kill, we abort fast
                # and let the caller's fail-closed path take over.
                stall_thresholds_s = (15, 30, 60, 120)
                STALL_KILL_S = 180.0
                last_chunk_count = 0
                last_chunk_at = asyncio.get_event_loop().time()
                consume_task: asyncio.Task | None = None

                async def _stall_watchdog():
                    nonlocal last_chunk_count, last_chunk_at, stall_killed
                    fired: set[int] = set()
                    while True:
                        await asyncio.sleep(2.0)
                        cur_count = len(messages_received)
                        now = asyncio.get_event_loop().time()
                        if cur_count > last_chunk_count:
                            last_chunk_count = cur_count
                            last_chunk_at = now
                            fired.clear()
                            continue
                        gap_s = now - last_chunk_at
                        for thr in stall_thresholds_s:
                            if gap_s >= thr and thr not in fired:
                                fired.add(thr)
                                last_msg = messages_received[-1] if messages_received else "(none)"
                                logger.warning(
                                    "stall-watchdog [%s]: no new chunks for %ds "
                                    "(last: %s, total: %d, text_so_far: %d chars)",
                                    prompt_role, thr, last_msg, cur_count, len(result_text),
                                )
                                try:
                                    trace.note(
                                        f"stall-watchdog: {thr}s gap (last={last_msg}, "
                                        f"chunks={cur_count}, text={len(result_text)})"
                                    )
                                except Exception:
                                    pass
                        if gap_s >= STALL_KILL_S and not stall_killed:
                            stall_killed = True
                            last_msg = messages_received[-1] if messages_received else "(none)"
                            logger.error(
                                "stall-watchdog [%s]: stream silent for >%.0fs — "
                                "aborting query (last: %s, %d chars accumulated)",
                                prompt_role, STALL_KILL_S, last_msg, len(result_text),
                            )
                            try:
                                trace.note(
                                    f"stall-watchdog: ABORT at {gap_s:.0f}s gap "
                                    f"(last={last_msg}, text={len(result_text)})"
                                )
                            except Exception:
                                pass
                            if consume_task is not None and not consume_task.done():
                                consume_task.cancel()
                            return

                watchdog_task = asyncio.create_task(_stall_watchdog())
                consume_task = asyncio.create_task(_consume_stream())
                try:
                    try:
                        if timeout is not None:
                            await asyncio.wait_for(consume_task, timeout=timeout)
                        else:
                            await consume_task
                    except asyncio.CancelledError:
                        # Stall watchdog killed the consume task. Translate
                        # to TimeoutError so the existing fail-closed paths
                        # apply uniformly. If it was outer cancellation
                        # (someone aborted us from above), propagate.
                        if stall_killed:
                            logger.warning(
                                "orchestrator _query [%s] aborted by stall-watchdog "
                                "(%d chars, %d msgs)",
                                prompt_role, len(result_text), len(messages_received),
                            )
                            raise asyncio.TimeoutError(
                                f"stream stalled for >{STALL_KILL_S:.0f}s (watchdog abort)"
                            )
                        raise
                    except asyncio.TimeoutError:
                        logger.warning(
                            "orchestrator _query timed out after %.0fs (had %d chars of text so far)",
                            timeout, len(result_text),
                        )
                        if not result_text:
                            raise
                    except Exception as stream_err:
                        # The CLI process may exit non-zero after it already
                        # delivered a ResultMessage (e.g. post-run cleanup
                        # failure). If we got usable text, log the error but
                        # return the result instead of discarding it and
                        # retrying — the model's work is done.
                        logger.warning(
                            "CLI stream error after %d messages [%s]: %s",
                            len(messages_received),
                            " → ".join(messages_received) or "(none)",
                            stream_err,
                        )
                        if stderr_lines:
                            logger.warning(
                                "CLI stderr at crash (%d lines):\n%s",
                                len(stderr_lines), "\n".join(stderr_lines[-30:]),
                            )
                        # "Got usable text" requires that the CLI actually
                        # delivered a ResultMessage. Otherwise result_text is
                        # a mid-stream AssistantMessage chunk the CLI crashed
                        # halfway through — semantically incomplete (e.g. a
                        # verifier that said "Let me inspect..." and then
                        # died is NOT a verification result, even though it
                        # looks like one). Treat that as a real failure.
                        if result_text and result_message_seen:
                            logger.warning(
                                "CLI crashed after delivering result (%d chars); "
                                "using collected response",
                                len(result_text),
                            )
                        else:
                            if result_text and not result_message_seen:
                                logger.warning(
                                    "CLI crashed BEFORE delivering ResultMessage "
                                    "(%d chars of partial text discarded). Treating "
                                    "as failure — partial mid-stream output is not "
                                    "a real response.",
                                    len(result_text),
                                )
                            raise
                finally:
                    # Always cancel the heartbeat + stall-watchdog tasks
                    # — both on success and on exception bubbling up to
                    # the outer retry loop.
                    heartbeat_task.cancel()
                    watchdog_task.cancel()
                    for t in (heartbeat_task, watchdog_task):
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    elapsed_total = asyncio.get_event_loop().time() - _hb_start
                    if elapsed_total >= 5.0:  # only print summary for non-trivial calls
                        print(
                            f"  ✓ [{prompt_role}] done in {elapsed_total:.1f}s "
                            f"({len(messages_received)} msgs, {len(result_text)} chars)",
                            flush=True,
                        )

                if self.emitter and emit_messages and result_text:
                    self.emitter.emit(
                        "agent_message",
                        agent_id="orchestrator",
                        parent_id=None,
                        role="assistant",
                        text=result_text[:2000],
                    )
                trace.close(
                    status="ok",
                    result_text=result_text,
                    cost_usd=captured_cost_usd,
                    input_tokens=captured_input_tokens,
                    output_tokens=captured_output_tokens,
                    messages_received=list(messages_received),
                )
                _ended_at = time.time()
                self._last_query = QueryRecord(
                    prompt_role=prompt_role,
                    system_prompt=system,
                    prompt=prompt,
                    started_at=wall_started_at,
                    ended_at=_ended_at,
                    duration_s=_ended_at - wall_started_at,
                    response_text=result_text,
                    model=self.config.model,
                    messages_received=list(messages_received),
                    cost_usd=captured_cost_usd,
                    input_tokens=captured_input_tokens,
                    output_tokens=captured_output_tokens,
                    status="ok",
                )
                return result_text

            except Exception as e:
                msg_trace = " → ".join(messages_received) if messages_received else "(no messages received)"
                try:
                    trace.note(f"Exception: {type(e).__name__}: {e}")
                    trace.close(
                        status=f"failed: {type(e).__name__}",
                        result_text=result_text,
                        cost_usd=captured_cost_usd,
                        input_tokens=captured_input_tokens,
                        output_tokens=captured_output_tokens,
                        messages_received=list(messages_received),
                    )
                except Exception:
                    pass
                logger.error(
                    "Query failed (attempt %d/%d): %s\n"
                    "  error_type: %s\n"
                    "  system_prompt: %d chars (%.50s...)\n"
                    "  prompt: %d chars\n"
                    "  tools: %s\n"
                    "  max_turns: %d\n"
                    "  cwd: %s\n"
                    "  model: %s\n"
                    "  result_text_so_far: %d chars\n"
                    "  messages_received: %s",
                    attempt + 1, max_retries, e,
                    type(e).__name__,
                    len(system), system[:50],
                    len(prompt),
                    tools,
                    max_turns,
                    self.config.project_root,
                    self.config.model,
                    len(result_text),
                    msg_trace,
                )
                if stderr_lines:
                    logger.error("CLI stderr (%d lines):\n%s", len(stderr_lines), "\n".join(stderr_lines[-30:]))
                else:
                    logger.error("CLI exited with no stderr — process may have crashed on startup")
                # Transparent retry for two transient infra shapes:
                #
                #   1. Empty timeout: the CLI streamed N status frames but
                #      never produced a ResultMessage OR any text. Almost
                #      always a hiccup that succeeds on retry.
                #
                #   2. Stall-killed: the watchdog aborted the consume task
                #      after >180s of stream silence (typically a wedged
                #      tool call). Partial result_text from a half-streamed
                #      tool input is incomplete and useless — retry from
                #      scratch even when accumulated text is non-zero.
                #
                # Both share a single one-shot budget — if the bonus retry
                # also fails, we degrade and let the caller handle it.
                is_empty_timeout = (
                    isinstance(e, asyncio.TimeoutError) and len(result_text) == 0
                )
                if (is_empty_timeout or stall_killed) and not empty_timeout_retried:
                    empty_timeout_retried = True
                    reason = "Stall-watchdog retry" if stall_killed else "Empty-timeout retry"
                    logger.warning(
                        "%s [%s]: CLI streamed %d messages, %d chars accumulated. "
                        "Retrying once in 5s (does not count against max_retries=%d).",
                        reason, prompt_role, len(messages_received),
                        len(result_text), max_retries,
                    )
                    await asyncio.sleep(5)
                    continue
                if attempt < max_retries - 1:
                    logger.warning("Retrying in 3s...")
                    await asyncio.sleep(3)
                    attempt += 1
                    continue
                else:
                    logger.error("All %d attempts exhausted for this query", max_retries)
                    _ended_at = time.time()
                    self._last_query = QueryRecord(
                        prompt_role=prompt_role,
                        system_prompt=system,
                        prompt=prompt,
                        started_at=wall_started_at,
                        ended_at=_ended_at,
                        duration_s=_ended_at - wall_started_at,
                        response_text=result_text,
                        model=self.config.model,
                        messages_received=list(messages_received),
                        cost_usd=captured_cost_usd,
                        input_tokens=captured_input_tokens,
                        output_tokens=captured_output_tokens,
                        status="error",
                    )
                    raise


def _describe_exception(e: BaseException) -> str:
    """Render an exception for operator-facing feedback.

    ``asyncio.TimeoutError`` and ``CancelledError`` have empty ``str(e)`` —
    formatting them as ``f"crashed: {e}"`` produces a useless ``"crashed: "``
    with no indication of what went wrong. This helper always surfaces the
    exception's class name and only appends the message when non-empty.
    """
    msg = str(e).strip()
    name = type(e).__name__
    return f"{name}: {msg}" if msg else name


def _log_rate_limit(
    source: str,
    prompt_role: str,
    info,                                  # RateLimitInfo
    log,                                   # logging.Logger
    emitter,                               # EventEmitter | None
    trace=None,                            # TraceWriter | None
) -> None:
    """Log + surface a CLI RateLimitEvent in a usable form.

    The bundled CLI emits ``RateLimitEvent`` whenever the rate-limit state
    transitions — *both* on hitting a soft warning and on full rejection.
    The previous handling was a generic "unhandled message type" warning
    that gave the operator no info. Now we log structured fields and
    optionally bubble an ``agent_tool`` event so the UI's Activity tab
    shows that the run is being throttled by Anthropic (not the harness).
    """
    if info is None:
        log.info("%s [%s]: rate-limit event (no info attached)", source, prompt_role)
        return
    status = getattr(info, "status", "?")
    rate_type = getattr(info, "rate_limit_type", None) or "?"
    util = getattr(info, "utilization", None)
    util_str = f"{util * 100:.0f}%" if util is not None else "?"
    resets_at = getattr(info, "resets_at", None)
    reset_str = ""
    if resets_at:
        from datetime import datetime as _dt
        try:
            reset_str = " · resets " + _dt.fromtimestamp(int(resets_at)).strftime("%H:%M:%S")
        except (ValueError, OSError, TypeError):
            reset_str = ""
    msg = f"rate-limit {status} ({rate_type} · {util_str} used{reset_str})"
    # "allowed_warning" is the soft heads-up — log at INFO.
    # "rejected" means the SDK call will fail — log at WARNING.
    level = log.warning if status == "rejected" else log.info
    level("%s [%s]: %s", source, prompt_role, msg)
    if emitter is not None:
        try:
            emitter.emit(
                "agent_tool",
                agent_id=source,
                parent_id=None,
                summary="⚠ " + msg if status == "rejected" else msg,
            )
        except Exception:
            pass
    if trace is not None:
        try:
            trace.note(msg)
        except Exception:
            pass


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM text. See ``_json_candidates`` for the
    candidate-generation order and the repair passes.

    LLMs frequently emit valid-looking JSON that fails strict ``json.loads``
    because of three recurring bugs: nested code fences inside a string
    value's markdown, raw newlines/tabs inside string values, and trailing
    commas. Each candidate is tried both as-is and through a repair pass.
    """
    for candidate in _json_candidates(text):
        for src in (candidate, _repair_json_strings(_strip_trailing_commas(candidate))):
            try:
                obj = json.loads(src)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    logger.warning("Could not extract JSON from response")
    return {}


def _json_candidates(text: str):
    """Yield candidate JSON substrings in priority order."""
    # 1. Fenced ```json ... ``` — GREEDY match. With re.DOTALL + .*, this
    #    catches the LAST closing fence in the response, not the first one
    #    appearing inside a string value's nested code block.
    fenced = re.search(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        yield fenced.group(1).strip()

    # 2. Whole-text parse, stripped of any leading/trailing prose. Many models
    #    emit pure JSON with no fences.
    stripped = text.strip()
    if stripped:
        yield stripped

    # 3. String-aware balanced-brace scan: find the first `{` and the matching
    #    `}` at depth 0, ignoring braces inside JSON string values.
    span = _balanced_object_span(text)
    if span:
        yield text[span[0]:span[1]]


def _balanced_object_span(text: str) -> tuple[int, int] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
    return None


def _repair_json_strings(s: str) -> str:
    """Escape raw control chars (newline/tab/cr) appearing inside JSON strings.

    LLMs frequently emit multi-line markdown inside a JSON string value with
    real newlines instead of ``\\n`` escapes — invalid JSON, but easy to
    repair with a one-pass scanner that tracks quote/escape state.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for c in s:
        if in_string:
            if escaped:
                out.append(c)
                escaped = False
                continue
            if c == "\\":
                out.append(c)
                escaped = True
                continue
            if c == '"':
                out.append(c)
                in_string = False
                continue
            if c == "\n":
                out.append("\\n")
                continue
            if c == "\r":
                out.append("\\r")
                continue
            if c == "\t":
                out.append("\\t")
                continue
            out.append(c)
        else:
            if c == '"':
                in_string = True
            out.append(c)
    return "".join(out)


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(s: str) -> str:
    """Strip ``, }`` and ``, ]`` patterns — common LLM JSON typo."""
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM text. Uses the same repair passes as
    ``_extract_json`` but yields list candidates instead of object ones."""
    for candidate in _json_array_candidates(text):
        for src in (candidate, _repair_json_strings(_strip_trailing_commas(candidate))):
            try:
                result = json.loads(src)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
    return []


def _json_array_candidates(text: str):
    fenced = re.search(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        yield fenced.group(1).strip()
    stripped = text.strip()
    if stripped:
        yield stripped
    span = _balanced_array_span(text)
    if span:
        yield text[span[0]:span[1]]


def _balanced_array_span(text: str) -> tuple[int, int] | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
    return None


def _coerce_kind(raw: object) -> PhaseKind:
    # Tolerate missing / unknown values from older state.json files and from
    # the planner LLM (which sometimes substitutes near-synonyms like "infra"
    # or "eval"). Default to ``build``.
    if isinstance(raw, PhaseKind):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("experiment", "exp", "eval", "evaluation", "benchmark"):
            return PhaseKind.experiment
        if s in ("build", "implement", "infra", "infrastructure", "code"):
            return PhaseKind.build
    return PhaseKind.build


def _parse_plan(raw: dict | None) -> PlanDocument | None:
    if not raw:
        return None
    nodes = []
    for n in raw.get("nodes", []):
        try:
            nodes.append(DagNode.model_validate(n))
        except Exception as e:
            logger.warning("Skipping invalid plan node %s: %s", n, e)
    files = []
    for f in raw.get("files", []):
        try:
            files.append(PlannedFile.model_validate(f))
        except Exception as e:
            logger.warning("Skipping invalid plan file %s: %s", f, e)
    if not nodes and not files:
        return None
    return PlanDocument(nodes=nodes, files=files)


def _build_section_spec(raw: dict, phase: PhaseState) -> SectionSpec:
    """Parse a per-section author's JSON output into a typed SectionSpec.

    Tolerates the LLM emitting partial or near-correct shapes. Acceptance
    criteria without a valid ``source.page`` are dropped (the citation
    validator will flag them downstream — better than silently failing the
    whole spec).
    """
    criteria_in = raw.get("acceptance_criteria") or []
    criteria: list[AcceptanceCriterion] = []
    citations: list[Citation] = []
    for c in criteria_in:
        if not isinstance(c, dict):
            continue
        src = c.get("source") or {}
        page = src.get("page") if isinstance(src, dict) else None
        if page is None:
            logger.warning(
                "section spec %s: dropping uncited criterion %r",
                phase.phase_id, c.get("text", "")[:80],
            )
            continue
        try:
            criterion = AcceptanceCriterion(
                text=str(c.get("text", "")),
                source=Citation(
                    page=int(page),
                    section=src.get("section"),
                    quote=src.get("quote"),
                ),
            )
        except Exception:
            logger.warning(
                "section spec %s: invalid criterion %r", phase.phase_id, c,
            )
            continue
        criteria.append(criterion)

    for cite in raw.get("citations") or []:
        if not isinstance(cite, dict):
            continue
        page = cite.get("page")
        if page is None:
            continue
        try:
            citations.append(Citation(
                page=int(page),
                section=cite.get("section"),
                quote=cite.get("quote"),
            ))
        except Exception:
            continue

    return SectionSpec(
        phase_id=raw.get("phase_id") or phase.phase_id,
        title=raw.get("title") or phase.title,
        goal=raw.get("goal") or "",
        spec_markdown=raw.get("spec_markdown") or "",
        acceptance_criteria=criteria,
        citations=citations,
    )


def _validate_plan(plan: PlanDocument, phase_ids: set[str]) -> list[str]:
    errors: list[str] = []
    node_ids = {n.phase_id for n in plan.nodes}
    for n in plan.nodes:
        if n.phase_id not in phase_ids:
            errors.append(f"Plan node '{n.phase_id}' not in spec phases")
        for dep in n.depends_on:
            if dep not in node_ids:
                errors.append(f"Plan node '{n.phase_id}' depends on unknown node '{dep}'")
    file_ids = {f.file_id for f in plan.files}
    for f in plan.files:
        if f.owning_phase not in node_ids:
            errors.append(f"File '{f.file_id}' owned by unknown phase '{f.owning_phase}'")
        for dep in f.depends_on:
            if dep not in file_ids:
                errors.append(f"File '{f.file_id}' depends on unknown file '{dep}'")
    return errors


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
    """One-line summary of a tool use for the activity / chat feed."""
    name = block.name
    inp = block.input or {}
    if name == "Bash":
        cmd = inp.get("command", "")
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"Bash: {cmd}"
    if name in ("Write", "Edit"):
        return f"{name}: {inp.get('file_path', '')}"
    if name == "Read":
        return f"Read: {inp.get('file_path', '')}"
    if name in ("Glob", "Grep"):
        return f"{name}: {inp.get('pattern', '')}"
    return name


def _format_orchestrator_tool(block: ToolUseBlock) -> str:
    """One-line summary of an orchestrator tool call for the chat feed.

    Orchestrator tools have human-friendly names (write_skeleton,
    author_section_specs, request_user_approval, …); we strip the MCP prefix
    and add the most useful arg inline so the chat shows e.g.
    ``start_phase: section_3_2`` rather than just ``start_phase``.
    """
    raw = block.name or ""
    name = raw.split("__")[-1] if "__" in raw else raw
    inp = block.input or {}
    if name == "start_phase" and inp.get("phase_id"):
        return f"start_phase: {inp['phase_id']}"
    if name == "request_user_approval" and inp.get("gate_id"):
        return f"request_user_approval: {inp['gate_id']}"
    if name in ("pipeline_complete", "pipeline_failed") and inp.get("message"):
        msg = str(inp["message"])
        return f"{name}: {msg[:120]}" + ("…" if len(msg) > 120 else "")
    return name
