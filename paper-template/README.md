# Paper reproduction template

A self-contained Claude Code project for reproducing one research paper.

## What's here

- `CLAUDE.md` — the reproduction spec. **Edit this first.** Filled-in citation, summary, headline claims, hyperparameters, dataset locations, commands, gotchas.
- `notes/claims.yaml` — machine-readable ledger of the paper's headline numerical claims. The `claims` MCP server reads/writes this.
- `notes/plan.md` — implementation plan (filled in once after a plan-mode session).
- `notes/journal.md` — append-only log of every meaningful run.
- `notes/post-mortems/` — one file per failed run.
- `.claude/skills/` — the methodology: verification ladder, post-mortem format, compare-to-paper rubric. Read these before working.
- `.claude/commands/` — slash commands: `/reproduce`, `/compare`, `/verify`, `/post-mortem`.
- `.mcp.json` — wires up the three MCP servers (paper, arxiv, claims).
- `scripts/{smoke,overfit-one-batch,reproduce}.sh` — verification ladder as runnable scripts. Edit to point at your real entry points.
- `paper/` — drop the PDF here as `paper/paper.pdf` (or set `PAPER_PDF_PATH` in `.mcp.json`).
- `src/`, `tests/`, `configs/` — your implementation.
- `data/`, `runs/` — gitignored. Datasets and run artifacts.

## Bootstrap a new paper

```bash
# From the toolkit repo:
cp -r paper-template/ ~/papers/<paper-slug>
cd ~/papers/<paper-slug>

# Install the toolkit so rb-mcp-* console scripts are on PATH for this repo:
uv pip install -e /path/to/research-builder

# Drop the PDF:
mkdir -p paper && cp /path/to/paper.pdf paper/paper.pdf

# Open in Claude Code:
claude .
# First conversation: "read the paper and fill in CLAUDE.md + notes/claims.yaml"
```

## Workflow expectations

See `CLAUDE.md` § "Workflow expectations". Short version:

1. Read `notes/plan.md` first.
2. Walk the verification ladder. Don't skip rungs.
3. Every run writes to `notes/journal.md`.
4. Every failed run gets a `/post-mortem` before retrying.
5. Every paper-discrepancy resolution lives in `CLAUDE.md`, not in the chat.
