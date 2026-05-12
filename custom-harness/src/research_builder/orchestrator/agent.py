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
from ..events import get_emitter
from ..literature.references import extract_citation_titles
from ..literature.scholar import SemanticScholarClient
from ..llm.paper import extract_full_text
from pydantic import BaseModel

from ..models.claims import Claim, ClaimSource, ClaimsLedger
from ..models.context import PostMortem
from ..models.results import SubAgentResult
from ..models.spec import (
    Artifact,
    DagNode,
    EventType,
    FileRole,
    FileStatus,
    PhaseState,
    PlanDocument,
    PlannedFile,
    Revision,
    SpecMetadata,
    SpecState,
)
from ..storage.spec_store import SpecStore
from .dependency import DependencyGraph
from .spec_manager import SpecManager

logger = logging.getLogger(__name__)

SPEC_CREATION_SYSTEM_PROMPT = """\
You are an expert research paper analyst. Your job is to read a research paper and \
produce two outputs that will guide a team of implementation agents.

## Reading the paper

Use the **Read** tool on the paper PDF path you'll be given. Read tool supports \
PDFs natively (preserves tables, figures, equations, layout). For papers ≤10 \
pages you can read the whole thing in one Read call. For larger papers, use the \
``pages`` parameter (e.g. ``pages="1-10"`` then ``pages="11-20"``); maximum 20 \
pages per Read call. The page count for this paper is in the user prompt below.

## Outputs you must produce

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

3. **plan** — An explicit DAG + file plan that downstream tools and the UI consume:
   - nodes: list of {phase_id, title, description, sub_steps (list of short bullet strings, \
     3–6 items describing the concrete work in that phase), file_ids (list of file_id strings \
     this phase OWNS as outputs/intermediates), depends_on (list of phase_ids)}
   - files: list of {file_id (stable dotted id like "data.train_loader"), rel_path \
     (e.g. "outputs/train_loader.pt"), owning_phase, role ("input"|"output"|"intermediate"), \
     description (one-line purpose), depends_on (list of file_ids consumed)}

## Output Format

Return your response as a JSON object with exactly three keys:
```json
{
  "spec_md": "the full markdown content...",
  "state": {
    "metadata": {"paper_id": "...", "paper_title": "..."},
    "phases": [...],
    "dependency_graph": {...}
  },
  "plan": {
    "nodes": [
      {"phase_id": "data", "title": "Data", "description": "...", \
"sub_steps": ["download X", "tokenize", "build loader"], \
"file_ids": ["data.train_loader"], "depends_on": []}
    ],
    "files": [
      {"file_id": "data.train_loader", "rel_path": "outputs/train_loader.pt", \
"owning_phase": "data", "role": "output", "description": "Serialized training DataLoader", \
"depends_on": []}
    ]
  }
}
```

## Guidelines

- Be thorough. Extract every hyperparameter, every architectural detail, every dataset reference.
- Use the standard phase IDs: "data", "architecture", "training", "eval", "results"
- Artifact file_paths should use the convention: phases/<phase_id>/outputs/<filename>
- Flag anything ambiguous — the implementation agents will consult the paper for details, \
  but the spec should identify known gaps.
- The dependency graph should reflect which phases need outputs from other phases. \
  Typically: data and architecture are independent, training depends on both, \
  eval depends on training (and sometimes data), results depends on eval and training.
- **Cited papers:** If the prompt includes a "Literature Context" section with abstracts \
  of cited papers, use them to resolve ambiguities. When the paper says something like \
  "we follow the same preprocessing as [Smith et al.]", look up the cited abstract for \
  specifics and include them in the spec rather than marking them as ambiguous. \
  Implementation agents also have a `lookup_citation` tool for runtime lookups.
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

For each claim, produce:
- **claim_id**: stable snake_case ID, e.g. "table2_cifar10_accuracy"
- **metric**: what is being measured (accuracy, F1, BLEU, loss, perplexity, etc.)
- **value**: the number (as a float)
- **tolerance**: ± range if stated, otherwise 0
- **unit**: "%", "ms", "perplexity points", etc. (empty if dimensionless)
- **dataset**: which dataset/split this was measured on
- **condition**: model variant, hyperparameter setting, etc.
- **source**: {table, figure, section, page, verbatim} — where in the paper
- **phase_id**: which implementation phase should produce this result \
  (typically "eval" or "results", sometimes "training" for convergence claims)
- **notes**: any caveats (e.g. "paper reports median of 5 runs")

Return a JSON array of claim objects:
```json
[
  {
    "claim_id": "table2_cifar10_top1",
    "metric": "top-1 accuracy",
    "value": 95.2,
    "tolerance": 0.3,
    "unit": "%",
    "dataset": "CIFAR-10 test set",
    "condition": "ResNet-50, 200 epochs",
    "source": {"table": "Table 2", "page": 7, "verbatim": "95.2 ± 0.3"},
    "phase_id": "eval",
    "notes": "mean of 3 runs"
  }
]
```

Be thorough — every number that could be verified is a claim. Include baseline \
comparisons only if the paper reports reproducing them. Skip claims about other \
papers' results unless the paper re-ran them.
"""

SPEC_REFINEMENT_SYSTEM_PROMPT = """\
You are amending the canonical spec for a research-paper reproduction pipeline. \
A sub-agent (or the orchestrator post-mortem) has identified that the current \
spec for one phase is ambiguous, contradictory, or under-specified — and no \
amount of retrying will fix it without changing the spec itself.

Your job:
1. Read the relevant section of `canonical_spec/spec.md` and the trigger diagnostics.
2. Consult the paper (use Read on the paper text dump or read_paper_section if available) \
   to find the authoritative answer.
3. Produce an amended `spec.md` that resolves the specific ambiguity. Quote paper text \
   in the rationale to justify the change.
4. Be surgical. Do NOT rewrite unrelated sections. Preserve the rest of the document verbatim.

Tools available: Read, Bash, Glob, Grep.

Respond with a JSON object:
```json
{
  "amended_spec_md": "the FULL new contents of spec.md (not a diff)",
  "summary": "one-sentence description of what changed and why",
  "sections_changed": ["Phase: Training", "Hyperparameters"]
}
```

If after consulting the paper you genuinely cannot resolve the ambiguity (the paper \
itself is silent or contradictory), respond with:
```json
{
  "amended_spec_md": null,
  "summary": "Cannot resolve: <reason>",
  "sections_changed": []
}
```
"""

POST_MORTEM_SYSTEM_PROMPT = """\
You are diagnosing a failed sub-agent attempt at one phase of a research-paper \
reproduction pipeline. The sub-agent already gave up — your job is NOT to fix \
the code. Your job is to read the wreckage and produce a structured hypothesis \
the next attempt can use to plan smarter.

You have read-only tools (Read, Bash, Glob, Grep). Use them to inspect:
- The attempt's `outputs/_result.json` (the sub-agent's own report)
- Any logs / stderr / test output in the attempt directory
- The source code the sub-agent wrote under `src/`
- Any partial outputs in `outputs/`

You should produce ONE focused hypothesis. Avoid generic advice ("add more tests"). \
Quote specific error lines or symptoms when possible.

Decide whether this is most likely:
- An **implementation issue** the sub-agent could fix on retry (wrong API call, \
  shape mismatch in code it wrote, missed edge case) → ``is_likely_spec_issue: false``
- A **spec issue** where the spec is genuinely under-specified or contradictory and \
  no amount of retrying the same spec will help → ``is_likely_spec_issue: true``

Respond with a JSON object:
```json
{
  "failure_hypothesis": "single sentence — what you think actually went wrong",
  "suggested_fix": "concrete next-attempt direction (one or two sentences)",
  "is_likely_spec_issue": true/false,
  "confidence": "low" | "medium" | "high"
}
```
"""

ACCEPTANCE_REVIEW_SYSTEM_PROMPT = """\
You are reviewing a sub-agent's completed work for cross-phase compatibility \
AND result quality. You are NOT re-implementing or re-testing — the sub-agent \
already validated its own code. Your job is a three-part review:

## Part 1: Cross-phase compatibility
1. Are the output artifacts present and in the expected format?
2. Will downstream phases be able to consume these outputs based on their input specs?
3. Are there any cross-phase interface mismatches?

## Part 2: Numerical claims verification
If a claims report is provided, review it carefully:
- **verified** claims: good, expected.
- **close** claims: acceptable but note the deviation.
- **missed** claims: investigate — is it a real failure or a measurement mismatch?
- **exceeded** claims: suspicious — the result is significantly better than the paper \
  reports. This often indicates a data leak, wrong eval split, or metric mismatch. \
  Flag these for investigation. Do NOT accept if the exceeded margin is large \
  unless you can explain why.
- **not_checked** claims: note these but don't reject solely because a claim \
  couldn't be auto-verified.

## Part 3: Convergence sanity (training phase only)
If this is a training phase, check:
- Did loss decrease monotonically (roughly)? Or did it diverge/plateau immediately?
- Did training run for the expected number of steps/epochs?
- Are there any NaN/Inf warnings?

Respond with a JSON object:
```json
{
  "accept": true/false,
  "feedback": "explanation if rejecting, null if accepting",
  "claims_notes": "optional notes on claims verification results"
}
```
"""


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

    async def create_spec(self, paper_path: Path, store: SpecStore) -> SpecManager:
        """Ingest a paper and produce the canonical spec (§4.1)."""
        logger.info("Ingesting paper from %s", paper_path)
        if self.emitter:
            self.emitter.emit(
                "agent_started",
                agent_id="orchestrator",
                parent_id=None,
                kind="orchestrator",
                title="Orchestrator",
            )

        # Step 0 — Get page count so we can tell the LLM how to page Read calls
        from ..llm.paper import get_page_count
        try:
            page_count = get_page_count(paper_path)
        except Exception:
            logger.exception("get_page_count failed; defaulting to 20")
            page_count = 20

        # Step 1 — Draft spec.md (the long LLM call). The orchestrator reads
        # the paper PDF directly via the Read tool — preserves tables/figures/
        # equations vs the old pdfplumber → text path.
        absolute_paper_path = Path(paper_path).resolve()
        prompt = (
            f"Read the research paper at:\n  {absolute_paper_path}\n\n"
            f"It has {page_count} pages. Use the Read tool with the ``pages`` "
            f"parameter (max 20 pages per call) to read it. After reading, "
            f"produce the JSON output specified in your system prompt."
        )
        response_text = await self._query(
            system=SPEC_CREATION_SYSTEM_PROMPT,
            prompt=prompt,
            tools=["Read", "Bash", "Glob", "Grep"],
            # Reading the paper takes several turns (multiple Read calls for
            # large papers) before the JSON synthesis. Plus extended thinking.
            max_turns=20,
            timeout=600,
        )

        # Step 2 — Build phase DAG (parse + validate)
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

        plan = _parse_plan(parsed.get("plan"))

        state = SpecState(
            metadata=metadata,
            phases=phases,
            dependency_graph=dep_graph,
            plan=plan,
        )

        # Validate
        graph = DependencyGraph.from_spec_state(state)
        phase_ids = {p.phase_id for p in phases}
        errors = graph.validate(phase_ids)
        if errors:
            logger.warning("Dependency graph validation issues: %s", errors)

        # Step 3 — Persist canonical state (write to disk)
        store.save_spec_md(spec_md)
        store.save_state(state)
        if state.plan is not None:
            errors = _validate_plan(state.plan, {p.phase_id for p in phases})
            if errors:
                logger.warning("Plan validation issues: %s", errors)
            store.save_plan(state.plan)
        store.append_revision(Revision(
            event_type=EventType.spec_created,
            rationale=f"Initial spec created from {paper_path.name}",
        ))

        logger.info("Spec created: %d phases, dependency_graph=%s", len(phases), dep_graph)

        # Step 4 — Extract claims ledger (second focused LLM pass)
        claims_ledger = await self._extract_claims(absolute_paper_path, page_count, {p.phase_id for p in phases})
        if claims_ledger.claims:
            store.save_claims(claims_ledger)
            logger.info("Claims ledger: %d claims extracted", len(claims_ledger.claims))
        else:
            logger.warning("No claims extracted from paper")

        # Step 4.5 — Project to paper-repo shape (CLAUDE.md + notes/claims.yaml)
        # so the Claude Code skill workflow can read the same artifacts.
        try:
            from ..storage.paper_repo import (
                project_spec_to_claude_md,
                project_claims_to_notes,
                ensure_journal_header,
            )
            project_spec_to_claude_md(self.config, spec_md, state)
            if claims_ledger.claims:
                project_claims_to_notes(self.config, claims_ledger)
            ensure_journal_header(self.config)
            logger.info(
                "Projected spec → %s, claims → %s",
                self.config.claude_md_path, self.config.claims_yaml_path,
            )
        except Exception:
            logger.exception("Failed to project spec/claims to paper-repo shape; continuing")

        # Step 5 — Build paper search index for sub-agent semantic search.
        # Optional: requires the [rag] extra (torch + sentence-transformers).
        # Skip with a warning if not installed; sub-agents can still use
        # read_paper_section + lookup_citation.
        try:
            from ..rag.agent import build_paper_index
            build_paper_index(paper_path, store.spec_dir)
        except ImportError as e:
            logger.warning(
                "Paper search index skipped — install custom-harness[rag] for semantic search "
                "in sub-agents (got: %s)", e,
            )
        except Exception:
            logger.exception("build_paper_index failed; sub-agents will use page-based reads only")

        # NB: do NOT emit agent_completed here. The orchestrator stays
        # alive supervising every phase — completion is signalled from
        # ExecutionLoop.run() once the entire run finishes.
        return SpecManager(store, state)

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
            timeout=300,
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

        prompt = (
            f"Extract every numerical/quantitative claim from the research "
            f"paper at:\n  {paper_path}\n\n"
            f"It has {page_count} pages. Use the Read tool with ``pages`` "
            f"(max 20 pages per call). Tables typically hold the headline "
            f"claims — read those carefully. Return the JSON array specified "
            f"in your system prompt."
        )
        try:
            response_text = await self._query(
                system=CLAIMS_EXTRACTION_SYSTEM_PROMPT,
                prompt=prompt,
                tools=["Read", "Bash", "Glob", "Grep"],
                # Reading + extracting claims takes multiple turns.
                max_turns=15,
                prompt_role="claims-extraction",
                timeout=600,
            )
        except Exception as e:
            logger.warning("Claims extraction LLM call failed: %s", e)
            return ClaimsLedger()

        parsed = _extract_json_array(response_text)
        if not parsed:
            # Try extracting a JSON object with a "claims" key
            obj = _extract_json(response_text)
            parsed = obj.get("claims", [])

        claims: list[Claim] = []
        for raw in parsed:
            try:
                source_raw = raw.get("source", {})
                source = ClaimSource(
                    table=source_raw.get("table"),
                    figure=source_raw.get("figure"),
                    section=source_raw.get("section"),
                    page=source_raw.get("page"),
                    verbatim=source_raw.get("verbatim", ""),
                )
                phase_id = raw.get("phase_id", "")
                # Validate phase_id — fall back to "eval" if not recognized
                if phase_id not in valid_phase_ids:
                    phase_id = "eval" if "eval" in valid_phase_ids else ""
                claims.append(Claim(
                    claim_id=raw.get("claim_id", f"claim_{len(claims)}"),
                    metric=raw.get("metric", ""),
                    value=float(raw.get("value", 0)),
                    tolerance=float(raw.get("tolerance", 0)),
                    unit=raw.get("unit", ""),
                    dataset=raw.get("dataset", ""),
                    condition=raw.get("condition", ""),
                    source=source,
                    phase_id=phase_id,
                    notes=raw.get("notes", ""),
                ))
            except Exception as e:
                logger.warning("Skipping malformed claim %s: %s", raw, e)

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
        prompt = (
            f"Amend the spec for phase '{phase_id}'. Trigger and current spec:\n\n"
            f"```json\n{json.dumps(prompt_payload, indent=2)[:60000]}\n```\n\n"
            f"Use Read on `{paper_path}` (or its extracted text) to consult the paper. "
            f"Return ONLY the JSON object specified in your system prompt."
        )

        try:
            response_text = await self._query(
                system=SPEC_REFINEMENT_SYSTEM_PROMPT,
                prompt=prompt,
                tools=["Read", "Bash", "Glob", "Grep"],
                # Bumped 6 → 15. Spec refinement reads paper + emits a full
                # amended spec; needs more headroom on opus extended thinking.
                max_turns=15,
                prompt_role=f"refine-spec-{phase_id}",
            )
        except Exception as e:
            logger.warning("Spec refinement LLM call failed for phase=%s: %s", phase_id, e)
            return SpecAmendment(summary=f"(refinement crashed: {e})")

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

        prompt = (
            f"Diagnose the failed attempt at phase '{phase_id}'.\n\n"
            f"Sub-agent's own report:\n```json\n{json.dumps(context, indent=2)}\n```\n\n"
            f"Inspect `{work_dir}` (the attempt directory) for source code, logs, "
            f"and partial outputs. Return ONLY the JSON object specified in your "
            f"system prompt."
        )

        try:
            response_text = await self._query(
                system=POST_MORTEM_SYSTEM_PROMPT,
                prompt=prompt,
                tools=["Read", "Bash", "Glob", "Grep"],
                # Bumped 4 → 12. Same extended-thinking issue as
                # acceptance_review — 4 turns guarantees max_turns errors and
                # the harness loops through SDK retries × phase retries
                # (~hours wasted per phase failure).
                max_turns=12,
                prompt_role=f"postmortem-{phase_id}",
            )
        except Exception as e:
            logger.warning("Post-mortem LLM call failed for phase=%s: %s", phase_id, e)
            return PostMortem(
                failure_hypothesis=f"(post-mortem unavailable: {e})",
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
                    # Extended thinking — orchestrator decisions (spec
                    # creation, acceptance review, plan rewrites) all benefit
                    # from chain-of-thought, and the ThinkingBlock chunks
                    # stream live so the chat pane shows real progress
                    # during long calls.
                    thinking={"type": "enabled", "budget_tokens": 8000},
                    include_partial_messages=True,
                )
                if tools is not None:
                    options.allowed_tools = tools

                result_text = ""
                messages_received: list[str] = []  # trace of message types for crash diagnostics

                async def _consume_stream():
                    nonlocal result_text
                    async for message in query(prompt=prompt, options=options):
                        msg_type = type(message).__name__
                        messages_received.append(msg_type)
                        logger.debug("orchestrator _query: received %s (#%d)", msg_type, len(messages_received))
                        if isinstance(message, StreamEvent):
                            evt = message.event
                            if evt.get("type") == "content_block_delta":
                                delta = evt.get("delta", {})
                                if delta.get("type") == "thinking_delta":
                                    text = delta.get("thinking", "")
                                    if text.strip() and self.emitter:
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
                                    if self.emitter and block.thinking.strip():
                                        self.emitter.emit(
                                            "agent_thinking",
                                            agent_id="orchestrator",
                                            parent_id=None,
                                            text=block.thinking[:1000],
                                        )
                                elif isinstance(block, ToolUseBlock):
                                    if self.emitter:
                                        self.emitter.emit(
                                            "agent_tool",
                                            agent_id="orchestrator",
                                            parent_id=None,
                                            summary=_format_tool_use(block),
                                        )
                                elif isinstance(block, TextBlock):
                                    result_text += block.text
                                    if self.emitter and block.text.strip():
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
                                    if isinstance(block, ToolResultBlock) and getattr(block, "is_error", False):
                                        text = _stringify_tool_result(block.content)
                                        if self.emitter and text:
                                            self.emitter.emit(
                                                "agent_tool",
                                                agent_id="orchestrator",
                                                parent_id=None,
                                                summary="ERR " + text[:300],
                                            )
                        elif isinstance(message, ResultMessage):
                            if message.result:
                                result_text = message.result
                        elif isinstance(message, SystemMessage):
                            logger.debug("orchestrator: SystemMessage subtype=%s", getattr(message, "subtype", "?"))
                        else:
                            logger.warning(
                                "orchestrator: unhandled message type=%s",
                                type(message).__name__,
                            )

                try:
                    if timeout is not None:
                        await asyncio.wait_for(_consume_stream(), timeout=timeout)
                    else:
                        await _consume_stream()
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
                    if result_text:
                        logger.warning(
                            "CLI crashed after delivering result (%d chars); "
                            "using collected response",
                            len(result_text),
                        )
                    else:
                        raise

                if self.emitter and emit_messages and result_text:
                    self.emitter.emit(
                        "agent_message",
                        agent_id="orchestrator",
                        parent_id=None,
                        role="assistant",
                        text=result_text[:2000],
                    )
                return result_text

            except Exception as e:
                msg_trace = " → ".join(messages_received) if messages_received else "(no messages received)"
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
                if attempt < max_retries - 1:
                    logger.warning("Retrying in 3s...")
                    await asyncio.sleep(3)
                else:
                    logger.error("All %d attempts exhausted for this query", max_retries)
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


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from text that may contain markdown code blocks."""
    # Try code block first
    code_block = re.search(r"```(?:json)?\s*\n(\[.*?\])\n```", text, re.DOTALL)
    if code_block:
        try:
            result = json.loads(code_block.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    bracket_start = text.find("[")
    if bracket_start >= 0:
        depth = 0
        for i in range(bracket_start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[bracket_start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return []


def _parse_plan(raw: dict | None) -> PlanDocument | None:
    if not raw:
        return None
    nodes = []
    for n in raw.get("nodes", []):
        try:
            nodes.append(DagNode(
                phase_id=n.get("phase_id", ""),
                title=n.get("title", ""),
                description=n.get("description", ""),
                sub_steps=list(n.get("sub_steps", [])),
                file_ids=list(n.get("file_ids", [])),
                depends_on=list(n.get("depends_on", [])),
            ))
        except Exception as e:
            logger.warning("Skipping invalid plan node %s: %s", n, e)
    files = []
    for f in raw.get("files", []):
        try:
            files.append(PlannedFile(
                file_id=f.get("file_id", ""),
                rel_path=f.get("rel_path", ""),
                owning_phase=f.get("owning_phase", ""),
                role=FileRole(f.get("role", "output")),
                description=f.get("description", ""),
                depends_on=list(f.get("depends_on", [])),
                status=FileStatus.planned,
            ))
        except Exception as e:
            logger.warning("Skipping invalid plan file %s: %s", f, e)
    if not nodes and not files:
        return None
    return PlanDocument(nodes=nodes, files=files)


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
