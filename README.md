# research-builder

[![tests](https://github.com/nickmatton/research-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/nickmatton/research-builder/actions/workflows/tests.yml)

A toolkit for reproducing research-paper results. **Two interfaces, one methodology**: an interactive Claude Code skill (slash commands + skills + scripts), and an autonomous Python harness (CLI that drives the same workflow unattended). Both share skill files, slash commands, helper tools, and the per-paper artifact format.

The opinionated methodology: **claims-first, verification-ladder, post-mortem on every failure.**

## File tree

```
research-builder/
├── bin/                       # toolkit-wide CLIs
│   ├── lambda                 # Lambda Cloud GPU provisioning with hard spend cap
│   ├── new-paper              # scaffold a new paper repo from skills/+commands/+tools/+paper-skeleton/
│   └── research-builder       # autonomous-harness CLI entry (Phase 1, in progress)
├── skills/                    # canonical methodology — copied to <paper-repo>/.claude/skills/
│   ├── verification-ladder.md
│   ├── post-mortem.md
│   └── compare-to-paper.md
├── commands/                  # canonical slash commands — copied to <paper-repo>/.claude/commands/
│   ├── reproduce.md           # /reproduce
│   ├── compare.md             # /compare
│   ├── verify.md              # /verify
│   └── post-mortem.md         # /post-mortem
├── tools/                     # canonical reusable Python helpers — copied to <paper-repo>/scripts/
│   ├── extract-paper-text.py  # PDF → paper.txt (one-shot)
│   ├── compare-claims.py      # verify run metrics vs claims.yaml, emit markdown table
│   └── lookup-citation.py     # Semantic Scholar wrapper
├── paper-skeleton/            # per-paper PLACEHOLDERS — copied verbatim per paper
│   ├── CLAUDE.md              # <PAPER TITLE> spec template
│   ├── pyproject.toml         # per-paper Python deps template
│   ├── notes/{claims, plan, journal}.{yaml, md}
│   ├── scripts/{smoke,overfit-one-batch,reproduce}.sh   # paper-specific shell scaffolds
│   └── (configs/, src/, tests/, .gitignore, README.md)
├── custom-harness/            # the original 9.5k LoC Python orchestrator (peer interface)
│   └── src/research_builder/
└── papers/<slug>/             # produced by EITHER interface, identical layout
```

Each top-level dir has its own README explaining its role.

## Two interfaces

Both interfaces produce paper repos with identical layout — `notes/claims.yaml`, `notes/journal.md`, `notes/post-mortems/`, `runs/<run-id>/`, `src/`, `tests/`, `configs/`. Either workflow can pick up where the other left off.

### Interactive — Claude Code skill workflow

```bash
bin/new-paper bert ~/papers/bert /path/to/bert.pdf
cd ~/papers/bert
uv sync                                  # install per-paper deps
python scripts/extract-paper-text.py     # → paper/paper.txt

claude .                                 # open Claude Code in the paper repo
# /reproduce, /compare, /verify, /post-mortem  ← drive interactively, with judgment
```

The methodology in `<paper-repo>/.claude/skills/`. The slash commands in `<paper-repo>/.claude/commands/`. You drive each rung of the verification ladder.

### Autonomous — custom harness workflow

```bash
bin/new-paper bert ~/papers/bert /path/to/bert.pdf
bin/research-builder ~/papers/bert       # drives unattended
```

The harness reads `~/papers/bert/CLAUDE.md` and `~/papers/bert/notes/claims.yaml` as inputs, spawns sub-agents per phase via the Claude Agent SDK, runs the same retry/post-mortem/spec-amendment loop the original orchestrator did, writes to the same per-paper artifact paths.

**The harness is currently being resurrected** — the migration moved its source from `src/research_builder/` to `custom-harness/src/research_builder/` and several external integrations have rotted (Lambda API host migrated, Cloudflare User-Agent block, Claude Agent SDK version drift). See `MIGRATION_PLAN.md` for status. The interactive skill workflow is fully working today.

## How the toolkit composes

`bin/new-paper <slug>` produces a self-contained paper repo by layering:

```
<paper-repo>/
├── (placeholders from paper-skeleton/)
├── scripts/
│   ├── (shell scaffolds from paper-skeleton/scripts/)
│   └── (Python tools from tools/)              # e.g. compare-claims.py
└── .claude/
    ├── skills/   ← from skills/
    └── commands/ ← from commands/
```

The toolkit-root directories are the **canonical masters**. Per-paper repos hold copies — paper repos stay self-contained (you can email one to a colleague who doesn't have the toolkit and they can run it). Updates to a master propagate to existing paper repos via `bin/new-paper-resync` (TODO; for now, `cp tools/*.py papers/<slug>/scripts/` etc.).

## Worked example

`papers/attention-is-all-you-need/` is a worked reproduction of Vaswani et al. 2017. As of this commit, **the verification ladder is green through smoke run on synthetic data**:

- ✅ Scaffolded; PDF extracted (15 pages); CLAUDE.md + notes/claims.yaml populated from the paper (6 headline claims).
- ✅ Real PyTorch implementation: `src/{attention, positional, transformer, train, eval, wmt, tokenize}.py` — paper-faithful (Adam β2=0.98, §5.3 warmup-then-decay LR, label smoothing ε=0.1, shared input/output embedding per §3.4 with √d_model scaling, beam search with Wu et al. length penalty, sacrebleu BLEU).
- ✅ **32 unit tests pass**: `cd papers/attention-is-all-you-need && uv run pytest`.
- ✅ **Overfit-one-batch**: loss → 0.0000 in ~250 steps with LS=0; → 0.78 with paper-faithful LS=0.1, which is exactly H(smoothed_dist) ≈ 0.77 — a label-smoothing artifact, not a model defect (verified by re-running with LS=0).
- ✅ **Smoke run** (200 steps, fresh synthetic batches): pipeline executes end-to-end, no NaN, loss decreases.
- ⏳ Real WMT 2014 EN-DE training run on a single A100 (~3–6 GPU-h) attempts the headline `table2_base_en_de_bleu = 27.3` claim. Pending GPU provision via `bin/lambda`. The big-model claims (28.4 / 41.8 BLEU) need 8x GPU × 3.5 days — explicitly out-of-budget, will land as `not_checked`.

See `papers/attention-is-all-you-need/CLAUDE.md` for the full spec, `notes/plan.md` for the implementation plan, `notes/journal.md` for the run log.

## Cloud GPU access (Lambda Labs)

`bin/lambda` provisions Lambda Cloud instances under a hard cumulative spend cap.

```bash
export LAMBDA_API_KEY=secret_xxx          # https://cloud.lambda.ai/api-keys
export LAMBDA_BUDGET_USD=20               # default 20; cumulative across all launches

bin/lambda price gpu_1x_a100              # → "$1.29/hr"
bin/lambda budget                         # cap, cumulative, remaining
bin/lambda provision gpu_1x_a100 --max-hours 6 --work-dir papers/attention-is-all-you-need
# → writes .lambda/env + remote_run.sh into the work dir, schedules nohup auto-teardown

cd papers/attention-is-all-you-need
bash remote_run.sh "uv sync && bash scripts/reproduce.sh"   # rsync work-dir → remote, run, rsync runs/ back

bin/lambda list                           # show ledger
bin/lambda teardown <id>                  # explicit teardown (auto-fires at deadline too)
bin/lambda check                          # cron-friendly: terminates overdue instances
```

State at `~/.lambda/state.json`. Each provision *eagerly* commits `max_hours × hourly_rate` to the ledger; teardown replaces the estimate with actual elapsed cost. `would_exceed` check is plain arithmetic — provision is refused, not warned, if it would breach `LAMBDA_BUDGET_USD`. `bin/lambda` is stdlib-only Python.

## The original harness (now a peer interface)

Before this restructure, `research-builder` was a 9.5k LoC custom Python harness. The orchestrator + sub-agent + retry-budget machinery preserved at [`custom-harness/`](custom-harness/) (with its own README); the **methodology** that was encoded inside it is now extracted to top-level `skills/` + `commands/` + `tools/`, available to both interfaces.

The harness was the right thing to build first. It validated the methodology end-to-end. Then extracting the durable patterns and making them shareable across two interfaces was the right next move.

## Migration history

| Commit | Phase | Lines |
|---|---|---|
| [`e20cd93`](../../commit/e20cd93) | WIP checkpoint of the original harness | +9797 |
| [`99294c6`](../../commit/99294c6) | Migration plan committed | +193 |
| [`e6f5c3d`](../../commit/e6f5c3d) | Phase 1: extract methodology to skills + templates | +456 |
| [`8b96670`](../../commit/8b96670) | Phase 2: MCP servers + paper template | +830 |
| [`d2014ca`](../../commit/d2014ca) | Phase 2.5: drop MCP, switch to built-in tools | −249 net |
| [`7d8e083`](../../commit/7d8e083) | Phase 3 (partial): scaffold attention-is-all-you-need + README rewrite | +1901 |
| [`bb3af4b`](../../commit/bb3af4b) | Phase 3: real implementation plan for the paper | +59 |
| [`1ad9c17`](../../commit/1ad9c17) | Phase 3: real PyTorch transformer + verification ladder green through smoke | +765 |
| [`d6d96e7`](../../commit/d6d96e7) | CI workflow + bin/new-paper scaffolder | +136 |
| [`fd37797`](../../commit/fd37797) | bin/lambda — Lambda Cloud GPU provisioning with hard cap | +653 |
| [`762f646`, `441263e`, `cc333ba`](../../commits) | Archive harness → custom-harness/ | rename |
| [`c497b0f`](../../commit/c497b0f) | WMT loader + BPE tokenizer + beam-search eval (32 tests) | +3327 |
| _(this commit)_ | Restructure toolkit: split paper-template → skills/+commands/+tools/+paper-skeleton/ | rename + new |

See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the phase plan and what's next.

## Tests

The current architecture's tests live with the worked example:

```bash
cd papers/attention-is-all-you-need
uv sync && uv run pytest tests/ -v
```

32 unit tests across attention/positional/transformer/tokenize/wmt/eval. CI runs them + an `--overfit-one-batch` assertion (final loss must be < 0.05 with LS=0) on every push. The original harness's 137 tests are preserved at [`custom-harness/tests/`](custom-harness/tests/) but no longer run in CI.
