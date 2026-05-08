# commands/

Slash commands invoked from a Claude Code session inside a paper repo. **These are the canonical masters.** Each new paper repo gets its own copy at `<paper-repo>/.claude/commands/` via `bin/new-paper`.

## Contents

| Command | What it does |
|---|---|
| `/reproduce` | Drives the full reproduction pipeline: pre-flight tests → run `scripts/reproduce.sh` → invoke `scripts/compare-claims.py` against `notes/claims.yaml` → append a journal row → verdict. |
| `/compare` | Verifies a completed run's metrics against the claims ledger. Reads `runs/<id>/metrics.json`, runs `scripts/compare-claims.py`, surfaces any `missed`/`exceeded` claims with red-flag treatment per `.claude/skills/compare-to-paper.md`. |
| `/verify` | Walks the next rung of the verification ladder (`.claude/skills/verification-ladder.md`). Reads recent journal + git state, picks the cheapest next check, runs it, reports + recommends the next rung. |
| `/post-mortem` | Diagnoses a failed run. Inspects logs + metrics + source, produces ONE focused hypothesis classified as implementation-issue vs spec-issue, writes to `notes/post-mortems/<run-id>.md` per the rubric in `.claude/skills/post-mortem.md`. |

## How they're used

In a paper repo, type `/reproduce` (or any of the others). Claude Code loads the markdown file as a prompt and executes the steps in your interactive session.

The commands are written so that:
- They reference `.claude/skills/<skill>.md` for the methodology rules (commands ARE NOT the methodology — they invoke it).
- They invoke `scripts/compare-claims.py` and other tools by relative path.
- They expect to find a `paper-template`-shaped paper repo (`CLAUDE.md` at root, `notes/`, `runs/`, `scripts/` etc.).

## Authoring rule

Each command:
- Has a **frontmatter `description:`** that shows in the `/`-menu.
- Has a **numbered Steps section** — explicit, executable.
- **Defers the methodology rules to a skill file.** A command should never re-explain the verification-ladder rubric; it should say "see `.claude/skills/verification-ladder.md`".
- Is short — ~30–50 lines.
