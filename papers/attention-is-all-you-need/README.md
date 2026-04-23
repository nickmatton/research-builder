# Paper reproduction template

A self-contained Claude Code project for reproducing one research paper. Built on Claude Code's built-in tools (Read, Write, Bash, Grep) plus three small Python scripts. No MCP servers, no toolkit install, no protocol layer.

## What's here

- `CLAUDE.md` — the reproduction spec. **Edit this first.** Citation, summary, headline claims, hyperparameters, dataset locations, commands, gotchas. This is what Claude reads on every session.
- `notes/claims.yaml` — machine-readable ledger of the paper's headline numerical claims. Edited directly with Read/Write.
- `notes/plan.md` — implementation plan (filled in once after a plan-mode session).
- `notes/journal.md` — append-only log of every meaningful run.
- `notes/post-mortems/` — one file per failed run.
- `.claude/skills/` — the methodology: verification ladder, post-mortem format, compare-to-paper rubric. Read these before working.
- `.claude/commands/` — slash commands: `/reproduce`, `/compare`, `/verify`, `/post-mortem`.
- `scripts/`:
  - `extract-paper-text.py` — one-time PDF→text extraction (uses pdfplumber).
  - `compare-claims.py` — verify a run's metrics against `notes/claims.yaml`.
  - `lookup-citation.py` — Semantic Scholar lookup (handles `SEMANTIC_SCHOLAR_API_KEY` env var).
  - `smoke.sh`, `overfit-one-batch.sh`, `reproduce.sh` — verification ladder. Edit to point at your real entry points.
- `paper/` — drop the PDF here as `paper/paper.pdf`. After extraction, also `paper/paper.txt`.
- `src/`, `tests/`, `configs/` — your implementation.
- `data/`, `runs/` — gitignored. Datasets and run artifacts.

## Bootstrap a new paper

```bash
# Clone the template:
cp -r paper-template/ ~/papers/<paper-slug>
cd ~/papers/<paper-slug>

# Drop the PDF and extract:
mkdir -p paper && cp /path/to/paper.pdf paper/paper.pdf
uv pip install pdfplumber pyyaml      # only deps for the helper scripts
python scripts/extract-paper-text.py  # → paper/paper.txt

# Open in Claude Code:
claude .
# First conversation: "read paper/paper.txt and fill in CLAUDE.md + notes/claims.yaml"
```

That's the whole setup. Claude reads `paper/paper.txt` with built-in Read/Grep, edits `notes/claims.yaml` directly, runs `scripts/compare-claims.py` for verification.

## Workflow expectations

See `CLAUDE.md` § "Workflow expectations". Short version:

1. Read `notes/plan.md` first.
2. Walk the verification ladder. Don't skip rungs.
3. Every full run writes to `notes/journal.md` via `/reproduce`.
4. Every failed run gets a `/post-mortem` before retrying.
5. Every paper-discrepancy resolution lives in `CLAUDE.md`, not in the chat.
