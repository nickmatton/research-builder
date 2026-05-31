# research-builder

[![tests](https://github.com/nickmatton/research-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/nickmatton/research-builder/actions/workflows/tests.yml)

Reproduce the numerical claims of a research paper end to end, from PDF to verified results. One methodology, two interfaces:

1. **Interactive** Claude Code skill workflow. Slash commands, skills, and helper scripts copied into a per-paper repo. You drive each rung of the verification ladder.
2. **Autonomous** Python harness ([`custom-harness/`](custom-harness/)). A CLI that runs the same workflow unattended, with a single LLM-driven orchestrator and per-section sub-agent chains.

Both interfaces share skills, slash commands, helper tools, and the per-paper artifact format. Output repos are interchangeable.

## Quick start

### Autonomous harness (zero setup)

```bash
cd custom-harness
uv sync
uv run research-builder --test
```

`--test` runs against the bundled 4-page test paper, writes to `/tmp/rb-test`, and implies `--auto` (no prompts) and `--dev` (uses your Claude Code subscription instead of an API key).

For a real paper:

```bash
uv run research-builder /path/to/paper.pdf -o ./workspace --auto
```

### Interactive skill workflow

```bash
bin/new-paper bert ~/papers/bert /path/to/bert.pdf
cd ~/papers/bert
uv sync
python scripts/extract-paper-text.py     # paper/paper.txt
claude .                                  # open Claude Code
# /reproduce, /compare, /verify, /post-mortem
```

The skills and slash commands land at `~/papers/bert/.claude/{skills,commands}/`.

## Methodology

Claims-first, verification-ladder, post-mortem on every failure.

- **Claims-first**: extract every numerical assertion from the paper before writing any code. Each claim has a source (table, figure, section, page), an expected value, and a tolerance.
- **Verification ladder**: smoke run on synthetic data, overfit one batch, full reproduction. Earlier rungs catch bugs cheaply.
- **Post-mortem on failure**: every failed phase produces a structured analysis. The orchestrator decides whether to retry, amend the spec, or mark the claim impossible.

Reference docs: [`skills/`](skills/).

## Repo layout

```
research-builder/
  bin/                  toolkit CLIs (new-paper, lambda, research-builder shim)
  skills/               canonical methodology, copied into per-paper repos
  commands/             canonical slash commands (/reproduce, /compare, /verify, /post-mortem)
  tools/                reusable Python helpers (extract-paper-text, compare-claims, lookup-citation)
  paper-skeleton/       per-paper placeholder layout (CLAUDE.md, notes/, scripts/, configs/)
  custom-harness/       autonomous Python harness (see its own README)
  papers/<slug>/        per-paper repos produced by either interface
  MIGRATION_PLAN.md     phase history and roadmap
```

Each top-level directory has its own README.

## How the toolkit composes

`bin/new-paper <slug>` produces a self-contained paper repo by layering:

```
<paper-repo>/
  (placeholders from paper-skeleton/)
  scripts/
    (shell scaffolds from paper-skeleton/scripts/)
    (Python tools from tools/)
  .claude/
    skills/      from skills/
    commands/    from commands/
```

The toolkit-root directories are canonical. Per-paper repos hold copies so they stay portable (you can email one to a colleague who does not have the toolkit). Updates to a master propagate via `cp tools/*.py papers/<slug>/scripts/` for now (resync command is TODO).

## Worked example

`papers/attention-is-all-you-need/` reproduces Vaswani et al. 2017. As of this commit the verification ladder is green through smoke run on synthetic data:

- Paper-faithful PyTorch implementation: attention, positional encoding, transformer, training, eval, WMT loader, BPE tokenizer.
- 32 unit tests pass: `cd papers/attention-is-all-you-need && uv run pytest`.
- Overfit-one-batch: loss converges to 0.0000 in ~250 steps with label smoothing off, to 0.78 with paper-faithful LS=0.1 (matches `H(smoothed_dist)`).
- Smoke run: 200 steps on synthetic batches, pipeline runs end to end, no NaN.
- Pending: WMT 2014 EN-DE training run on a single A100 (~3 to 6 GPU-h) targeting the headline `table2_base_en_de_bleu = 27.3` claim. Big-model claims (28.4 / 41.8 BLEU) need 8x GPU x 3.5 days and are explicitly out of budget.

See `papers/attention-is-all-you-need/CLAUDE.md` for the spec, `notes/plan.md` for the implementation plan, `notes/journal.md` for the run log.

## Cloud GPU access

`bin/lambda` provisions Lambda Cloud instances under a hard cumulative spend cap.

```bash
export LAMBDA_API_KEY=secret_xxx
export LAMBDA_BUDGET_USD=20

bin/lambda price gpu_1x_a100
bin/lambda budget
bin/lambda provision gpu_1x_a100 --max-hours 6 --work-dir papers/attention-is-all-you-need

cd papers/attention-is-all-you-need
bash remote_run.sh "uv sync && bash scripts/reproduce.sh"

bin/lambda list
bin/lambda teardown <id>
bin/lambda check                   # cron-friendly; terminates overdue instances
```

State at `~/.lambda/state.json`. Each provision eagerly commits `max_hours x hourly_rate` to the ledger; teardown replaces the estimate with actual elapsed cost. `would_exceed` checks are plain arithmetic, so provisioning is refused (not warned) if it would breach `LAMBDA_BUDGET_USD`. `bin/lambda` is stdlib-only Python.

## Tests

Worked example:

```bash
cd papers/attention-is-all-you-need
uv sync && uv run pytest tests/ -v
```

32 unit tests across attention, positional, transformer, tokenize, WMT, eval. CI runs them plus an `--overfit-one-batch` assertion (final loss < 0.05 with LS=0) on every push.

Autonomous harness:

```bash
cd custom-harness
uv run pytest
```

## History

See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the phase plan and the original-harness-to-toolkit migration.
