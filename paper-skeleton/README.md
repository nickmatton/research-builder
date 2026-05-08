# Paper reproduction skeleton

The **placeholder layer** of a paper repo. `bin/new-paper <slug>` copies these contents into the new paper repo, then layers in the toolkit's canonical `skills/`, `commands/`, and `tools/` on top.

## What's in here

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Reproduction-spec template with `<PAPER TITLE>` placeholder. Filled in per paper: citation, summary, headline claims, hyperparameters, datasets, compute budget, gotchas. Claude reads this on every session. |
| `notes/claims.yaml` | Claims ledger template (one example claim + delete-me comment). Filled in per paper from Table 2 / Table 3 / §results. |
| `notes/plan.md` | Implementation plan template — the standard 6 phases (analysis → data → model → training → eval → reproduce). Filled in after the first plan-mode session. |
| `notes/journal.md` | Empty append-only journal with the row format documented at the top. Per-run entries get appended automatically by `/reproduce`. |
| `notes/post-mortems/.gitkeep` | One file per failed run lands here, written by `/post-mortem`. |
| `scripts/{smoke,overfit-one-batch,reproduce}.sh` | **Paper-specific** shell scaffolds with `TODO` comments where you wire up your real entry points (`src.train`, `src.eval`, etc.). The verification-ladder pattern; you fill in the implementation. |
| `pyproject.toml` | Per-paper Python deps template. Default: torch + pyyaml + pdfplumber + pytest. Comments list common additions per paper-type (NLP / vision / RL / generic). |
| `configs/`, `src/`, `tests/` | Empty (`.gitkeep` placeholders). Filled in per paper. |
| `.gitignore` | Standard ML repo ignores: `data/`, `runs/`, `*.pt`, `.venv`, `__pycache__/`, `.lambda/`, `remote_run.sh`. |

## What's NOT in here

Everything that's the **same across every paper** — the methodology, the slash commands, the reusable Python helpers — lives at the toolkit root and gets copied in by `bin/new-paper`:

| Toolkit root | → Per-paper repo at | What it is |
|---|---|---|
| `skills/` | `.claude/skills/` | Methodology (verification-ladder, post-mortem, compare-to-paper) |
| `commands/` | `.claude/commands/` | Slash commands (/reproduce, /compare, /verify, /post-mortem) |
| `tools/` | `scripts/` | Reusable Python helpers (compare-claims, extract-paper-text, lookup-citation) |

So the per-paper repo, after scaffolding, has:

```
papers/<slug>/
├── CLAUDE.md, notes/, configs/, src/, tests/   ← from paper-skeleton/
├── pyproject.toml, .gitignore, README.md       ← from paper-skeleton/
├── scripts/
│   ├── smoke.sh, overfit-one-batch.sh, reproduce.sh   ← from paper-skeleton/
│   └── compare-claims.py, extract-paper-text.py, lookup-citation.py  ← from tools/
└── .claude/
    ├── skills/    ← from skills/
    └── commands/  ← from commands/
```

## Why split this way

Earlier versions had everything bundled into a single `paper-template/` directory. The split makes the toolkit's surface area visible at the repo root: `skills/` (methodology), `commands/` (interactive), `tools/` (computational), `paper-skeleton/` (per-paper placeholders). Each is a peer with a clear role. See the top-level README for the full architecture.
