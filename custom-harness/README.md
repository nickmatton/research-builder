# custom-harness/ — the original research-builder agent harness

Frozen but tracked. The 9.5k LoC custom orchestrator + sub-agent + MCP-tools system that came before the Claude-Code-native template at the repo root. Browseable as proof-of-concept; not installed by the toolkit's `pyproject.toml` anymore.

To actually run it: `git checkout 90076ce` (or earlier — that commit still has it at `src/research_builder/` with the original deps in `pyproject.toml`).

## What's here

- **`src/research_builder/`** — the harness source. 57 modules grouped into:
  - `orchestrator/` — outer Python state machine (`loop.py`), LLM-driven `agent.py` (1250 LoC; five LLM call sites: spec creation, claims extraction, acceptance review, post-mortem, spec refinement), `dependency.py` (DAG), `failure.py` (retry budgets), `spec_manager.py` (sub-spec extraction).
  - `sub_agent/` — Claude Agent SDK session wrapper, custom MCP tools (`read_paper_section`, `search_paper`, `lookup_citation`, `report_result`, `request_compute`), per-phase system prompts.
  - `models/` — Pydantic schemas for spec / state / results / claims / context.
  - `storage/` — `spec_store.py` (`spec.md` + `state.yaml` + `revision_log.yaml`), `workspace.py` (per-attempt isolation `phases/<id>/<try>/`).
  - `cloud/` — Lambda Cloud API client, `BudgetLedger` with hard cap, ephemeral SSH keypair, `remote_run.sh` template.
  - `viewer/` — Textual-based real-time TUI: phase status, file tree, activity feed, chat pane.
  - `events/`, `commands/`, `chat.py` — JSONL event stream + side-channel operator chat with the orchestrator.
  - `literature/`, `rag/` — Semantic Scholar resolver + sentence-transformers paper search index.
  - `resume.py` — detect existing canonical_spec/, offer resume vs fresh.

- **`tests/`** — 137 unit + e2e tests covering models, storage, dependency graph, failure handling, execution loop, and end-to-end pipeline (with mocked LLM).

- **`examples/attention_run/`** — a real harness run on Attention Is All You Need. `canonical_spec/`, per-phase generated code under `phases/<id>/<try>/{src,outputs}/`, and the final `report/reproduction_report.md`. Concrete proof the harness produced working transformer code end-to-end (`attention.py`, `transformer.py`, `feed_forward.py`, `positional_encoding.py`).

- **`paper/`** — PDFs the harness was tested against (Attention Is All You Need + a small synthetic test paper).

- **`spec_v4.md`** — the internal design spec the harness was built from. The document `orchestrator/agent.py` + `sub_agent/agent.py` trace back to.

## The agentic loop, briefly

Two LLM-in-the-loop tiers:

```
Tier 1: Python orchestrator
  while phases not done:
    decision = await llm.needs_gpu(phase)        ← LLM
    handle = cloud.provision(phase) if gpu
    result = await spawn_subagent(phase)
    accept = await llm.acceptance_review(result) ← LLM
    if not accept and budget.has_retries():
      pm = await llm.post_mortem(result)         ← LLM
      if pm.is_spec_issue:
        await llm.refine_spec(phase)             ← LLM
      retry

Tier 2: Sub-agent (Claude Agent SDK session per phase)
  for attempt in range(10):
    write/edit code → run tests
    if pass: report_result(success); exit
    if spec_unfixable: report_result(spec_issue)
    diagnose + fix
```

Both tiers ARE Claude. Tier 1 is `claude_agent_sdk.query()` calls from Python; Tier 2 is per-phase Claude sessions with custom MCP tools. Per-phase retry budget (3) at Tier 1; internal debug budget (10) at Tier 2; spec-issue returns escape both budgets.

## Why this still lives in the repo

1. **Engineering proof.** The harness shows the complex version is achievable. The new template is the deliberate simplification, not the only thing we know how to build.
2. **Methodology provenance.** Patterns like `claims-ledger`, `verification-ladder`, `post-mortem`, and `compare-to-paper` were validated against the harness's runs before being extracted into `paper-template/.claude/skills/`. The skills you see in the new template are condensed from this code.
3. **Side-by-side comparison.** When the new template's `papers/attention-is-all-you-need/` reproduction completes on GPU, the old harness's `examples/attention_run/` provides a concrete before/after baseline (cost, time, output quality).

## Why we stopped using it

Documented in [`MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) at the repo root. Short version: every Claude Code feature shipped (plan mode, hooks, sub-agents, compaction, memory, `/loop`, scheduled triggers) made the custom orchestration a little less load-bearing. The methodology was the durable contribution; the orchestration code was duplicating Claude Code's own loop.
