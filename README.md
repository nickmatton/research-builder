# research-builder

A toolkit for reproducing research-paper results using Claude Code. One paper = one repo, scaffolded from a template, driven by an opinionated methodology: **claims-first, verification-ladder, post-mortem on every failure.**

```
research-builder/
├── paper-template/             # cp -r per paper; self-contained
│   ├── CLAUDE.md               # reproduction spec scaffold
│   ├── notes/{claims.yaml, plan.md, journal.md}
│   ├── .claude/skills/         # verification-ladder, post-mortem, compare-to-paper
│   ├── .claude/commands/       # /reproduce, /compare, /verify, /post-mortem
│   └── scripts/                # extract-paper-text, compare-claims, lookup-citation, smoke/overfit/reproduce
└── papers/
    └── attention-is-all-you-need/   # worked example, scaffolded from paper-template/
```

To start a new paper: `cp -r paper-template/ ~/papers/<slug>`, drop the PDF, run `extract-paper-text.py`, open Claude Code. Three Python scripts (~200 LoC) replace what was originally a 9.5k LoC custom agent harness. **No MCP servers. No protocol layer. No toolkit install in the paper repo.**

## How it works (current architecture)

Each paper is its own Claude Code project. The template ships with:

- **`CLAUDE.md`** — the reproduction spec. Citation, summary, headline claims (with table/page refs), hyperparameters with paper-section references, dataset locations, compute budget, commands. Claude reads it on every session. Discrepancies with the paper get resolved *here*, not in chat.

- **`notes/claims.yaml`** — machine-readable ledger of the paper's headline numerical claims (e.g. "BLEU 28.4 on WMT14 EN-DE — Table 2, p.8"). Verified after each run by `scripts/compare-claims.py`.

- **`notes/journal.md`** — append-only log. One row per run: git SHA, config hash, hardware, key metrics, claims-verification summary, one-sentence note. Survives sessions; survives context resets.

- **`.claude/skills/`** — the methodology, encoded as durable patterns:
  - `verification-ladder.md` — unit → overfit-one-batch → smoke → short → full. Cheapest gate first. Don't skip rungs.
  - `post-mortem.md` — one focused hypothesis per failed run. Spec-issue vs implementation-issue classification.
  - `compare-to-paper.md` — `verified | close | missed | exceeded | not_checked` rubric. **Exceeded = red flag** (data leak, wrong split, metric mismatch), not a win.

- **`.claude/commands/`** — slash commands wiring it together: `/reproduce`, `/compare`, `/verify`, `/post-mortem`.

- **`scripts/`** — three small Python helpers and three shell scaffolds:
  - `extract-paper-text.py` — one-shot PDF→text with `--- Page N ---` markers (pdfplumber).
  - `compare-claims.py` — verify run metrics against `claims.yaml`, emit markdown table.
  - `lookup-citation.py` — Semantic Scholar wrapper handling the API key.
  - `smoke.sh`, `overfit-one-batch.sh`, `reproduce.sh` — verification-ladder rungs.

## Worked example

`papers/attention-is-all-you-need/` is the worked example. As of this commit, **the verification ladder is green through smoke run on synthetic data**:

- ✅ Scaffolded from `paper-template/`. PDF extracted to `paper/paper.txt` (15 pages).
- ✅ `CLAUDE.md` + `notes/claims.yaml` populated from the paper (6 headline claims from Table 2 + §5.2).
- ✅ Real PyTorch implementation: `src/attention.py`, `src/positional.py`, `src/transformer.py`, `src/train.py` — paper-faithful (Adam β2=0.98, §5.3 warmup-then-decay LR, label smoothing ε=0.1, shared input/output embedding per §3.4 with √d_model scaling).
- ✅ **15 unit tests pass** (shape, mask semantics, PE formula, gradient flow): `uv run pytest`.
- ✅ **Overfit-one-batch**: loss → 0.0000 in ~250 steps with LS=0; → 0.78 with paper-faithful LS=0.1, which is exactly H(smoothed_dist) ≈ 0.77 — a label-smoothing artifact, not a model defect (verified by re-running with LS=0). See `notes/journal.md` for the analysis.
- ✅ **Smoke run** (200 steps, fresh synthetic batches): pipeline executes end-to-end, no NaN, loss decreases.
- ⏳ Real WMT 2014 EN-DE loader (Phase 2 of `notes/plan.md`) — pending. Then a base-model run on a single A100 (~3–6 GPU-h) attempts the headline `table2_base_en_de_bleu` claim. The big-model claims are explicitly out-of-budget without multi-GPU compute and will likely land as `not_checked`.

See `papers/attention-is-all-you-need/CLAUDE.md` for the full spec, `notes/plan.md` for the implementation plan, and `notes/journal.md` for the run log.

## Architecture decision: built-in tools over MCP

The first cut of the toolkit ([Phase 2 commit](../../commit/8b96670)) used three MCP servers — `paper`, `arxiv`, `claims` — wired in via `.mcp.json`. They worked. They were also overkill for a 3-server / ~10-tool toolkit used inside a single harness (Claude Code).

The pivot ([Phase 2.5](../../commit/d2014ca)) replaces them with three small Python scripts called via Bash, plus direct Read/Write/Grep on `paper.txt` and `claims.yaml`. Net change: ~410 LoC of MCP servers + 40 transitive deps → ~200 LoC of self-contained scripts and zero new deps.

Why:
- **Tool-discovery / typed-schema / persistent-process wins are marginal** when there are only ~10 tools used by one client.
- **Costs are real**: a `mcp` dependency, ~40 transitive packages, subprocess management, stdout-discipline risk, and a per-paper `uv pip install -e <toolkit>` step that breaks portability.
- **Built-in tools are more native** to Claude Code than any third-party protocol. A Read on `paper/paper.txt` is faster, simpler, more debuggable than a JSON-RPC roundtrip.

Full reasoning, including a full ratings comparison and the discarded MCP implementation, is in [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) § "Architecture decision: built-in tools over MCP."

## The original harness

Before this template existed, `research-builder` was a 9.5k LoC custom agent harness:

- **Orchestrator** (Python) read the paper via Claude Agent SDK, drafted a canonical spec (`spec.md` + `state.yaml`), built a phase dependency graph, and ran an execution loop.
- **Sub-agents** (one per phase: data, architecture, training, eval, results) each spawned a Claude Agent SDK session with custom MCP tools (`read_paper_section`, `lookup_citation`, `report_result`).
- **Failure handling**: per-phase retry budgets, spec-issue vs implementation-issue classification, LLM-driven post-mortems, structured spec amendments.
- **TUI viewer** (Textual) for live monitoring: phase status, file lifecycle, chat pane.
- **Cloud GPU provisioning** (Lambda Labs) with per-run spend caps.

It worked. It was also a small platform to maintain — every Claude Agent SDK upgrade meant re-validating ~9.5k LoC of orchestration, and every feature Claude Code shipped (plan mode, hooks, sub-agents, compaction, memory, `/loop`, scheduled triggers) made the custom orchestration a little less load-bearing.

The harness source still lives at `src/research_builder/` (committed at [`e20cd93`](../../commit/e20cd93)). It will move to `.archive/research-builder-v1/` once an end-to-end reproduction in `papers/attention-is-all-you-need/` validates the lighter-weight template stack. The methodology — claims ledger schema, verification ladder, post-mortem template, the structured `report_result` artifact format — was extracted before the rewrite ([Phase 1 commit](../../commit/e6f5c3d)) and now lives in `paper-template/.claude/skills/`.

The harness was the right thing to build first. It validated the methodology end-to-end. Then deleting most of it was the right next move.

## Migration history

| Commit | Phase | Lines |
|---|---|---|
| [`e20cd93`](../../commit/e20cd93) | WIP checkpoint of the original harness | +9797 |
| [`99294c6`](../../commit/99294c6) | Migration plan committed | +193 |
| [`e6f5c3d`](../../commit/e6f5c3d) | Phase 1: extract methodology to skills + templates | +456 |
| [`8b96670`](../../commit/8b96670) | Phase 2: MCP servers + paper template | +830 |
| [`d2014ca`](../../commit/d2014ca) | Phase 2.5: drop MCP, switch to built-in tools | -249 (net) |

See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the phase plan, gates, and what's next.

## Use the template

```bash
git clone https://github.com/nickmatton/research-builder.git
cp -r research-builder/paper-template ~/papers/<paper-slug>
cd ~/papers/<paper-slug>

mkdir -p paper && cp /path/to/paper.pdf paper/paper.pdf
uv pip install pdfplumber pyyaml          # only deps for the helper scripts
python scripts/extract-paper-text.py      # → paper/paper.txt

claude .                                  # open Claude Code
# First conversation: "read paper/paper.txt, fill in CLAUDE.md and notes/claims.yaml"
```

The per-paper README (`paper-template/README.md`) walks through the rest of the workflow.

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover the original harness's models, storage, dependency graph, failure handling, and execution loop. The new template's `compare-claims.py` is smoke-tested in `papers/attention-is-all-you-need/` against real claims from `notes/claims.yaml`. Tests for the harness will move with the source to `.archive/` in Phase 4.
