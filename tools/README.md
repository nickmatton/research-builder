# tools/

Reusable Python helpers — **the canonical masters.** Each new paper repo gets its own copy at `<paper-repo>/scripts/` via `bin/new-paper` (alongside the paper-specific shell scripts in `paper-skeleton/scripts/`). Same content every paper.

## Contents

| Tool | Purpose | Deps |
|---|---|---|
| `extract-paper-text.py` | One-shot PDF → text extraction with `--- Page N ---` markers. Output: `paper/paper.txt`. After this runs, Claude reads/greps the text file with built-in tools — no MCP server needed for paper access. | `pdfplumber` |
| `compare-claims.py` | Verify a run's `metrics.json` against `notes/claims.yaml`. Emits markdown table per the `.claude/skills/compare-to-paper.md` rubric (`verified` / `close` / `missed` / `exceeded` / `not_checked`). Exit code 1 if any claim missed or suspicious. | `pyyaml` |
| `lookup-citation.py` | Semantic Scholar API wrapper. Takes a title or paper ID, returns JSON with abstract, year, venue, authors. Handles `SEMANTIC_SCHOLAR_API_KEY` env var so WebFetch doesn't have to. Stdlib only. | (none) |

## Why these are tools and not skills

These do *concrete computation* that doesn't require an LLM — text extraction, statistical comparison, HTTP API calls. They produce structured output that slash commands and the harness can parse.

Skills (in `skills/`) are *prose patterns* the LLM reads and follows. Tools are *executables* it runs.

## Adding a new tool

If a slash command keeps re-implementing the same parsing or comparison logic in prose, it probably wants to be a tool here. Candidate signals:
- The command tells Claude to "compute X across N items" (loop logic → tool)
- The command says "this is sensitive to off-by-one errors, check carefully" (parsing logic → tool)
- The command involves an HTTP API call (network access → tool)

## Authoring rule

- **Stdlib first** if possible — keeps per-paper deps minimal. (`lookup-citation.py` uses `urllib`, not `httpx`.)
- **JSON to stdout** for structured output — slash commands can parse it.
- **Exit code reflects success** — `0` for happy path, non-zero for actionable failure (missed claim, no PDF found, etc.).
- **Self-contained** — each tool is one Python file, no shared `lib/`. Simplifies the per-paper copy.
