# MCP Servers (design notes)

Three servers ship with the toolkit. Each is a small Python process exposing a focused set of tools to Claude Code via MCP. The goal: pull the *useful* parts of `src/research_builder/{sub_agent/tools.py, literature/, rag/, llm/paper.py}` into a form any paper repo can wire up with a single `.mcp.json` entry.

## 1. `paper_reader_server.py`

Reads the paper PDF. Ports `sub_agent/tools.py:read_paper_section` + `search_paper` + `llm/paper.py:extract_full_text`.

Tools:

- `read_paper_section(start_page, end_page?)` — extract raw text from a page range. 1-indexed, inclusive. Omit `end_page` for a single page.
- `search_paper(query, top_k?)` — semantic search over a pre-built paper index. Returns chunks with page numbers and section headings. Use before `read_paper_section` to locate the right pages.
- `extract_full_text()` — full paper as one blob. Expensive; use `search_paper` unless you really need everything.

Config (via env / CLI args):
- `PAPER_PDF_PATH` — absolute path to the PDF.
- `PAPER_INDEX_PATH` — path to the built search index (optional; falls back to page-based reads only).

Port source: `src/research_builder/sub_agent/tools.py` and `src/research_builder/rag/`.

## 2. `arxiv_server.py`

Looks up cited papers. Ports `sub_agent/tools.py:lookup_citation` + `literature/scholar.py:SemanticScholarClient` + `literature/references.py:extract_citation_titles`.

Tools:

- `lookup_citation(query)` — search Semantic Scholar by title or citation string. Returns title, abstract, year, venue, link for up to 3 matches.
- `extract_citations(paper_path)` — pull citation titles out of a paper's references section. Useful once per paper at scaffold time to pre-populate `notes/related.md`.

Config:
- `SEMANTIC_SCHOLAR_API_KEY` — optional; unauthenticated calls rate-limit aggressively under load.

Port source: `src/research_builder/literature/`.

## 3. `claims_server.py`

CRUD over `notes/claims.md` + runs the compare-to-paper logic from `skills/compare-to-paper.md`. This is the one *new* server — the current harness does all of this in-process in `orchestrator/claims.py` and `models/claims.py`.

Tools:

- `list_claims()` — list all claims from `notes/claims.md`.
- `add_claim(claim_id, metric, value, tolerance?, unit?, dataset?, condition?, source?, phase?, notes?)` — append a claim row.
- `get_claim(claim_id)` — fetch one.
- `verify_claim(claim_id, actual_value)` — classify as verified / close / missed / exceeded (returns status + delta + delta_pct).
- `verify_run(results_json_path)` — verify every claim against a run's metrics file, return the claims-verification table as markdown.

Data model (ported from `src/research_builder/models/claims.py`):

```python
Claim = {
    claim_id: str,        # "table2_cifar10_top1"
    metric: str,          # "accuracy", "F1", "BLEU"
    value: float,
    tolerance: float,     # 0 if not stated in paper
    unit: str,            # "%", "ms", "perplexity"
    dataset: str,         # "CIFAR-10 test set"
    condition: str,       # "ResNet-50, 200 epochs"
    source: {table|figure|section?, page?, verbatim},
    phase: str,           # "eval" | "training" | ...
    notes: str,
}
```

Storage: `notes/claims.md` in the paper repo. Machine-readable YAML frontmatter per row, human-readable table body. The server parses and writes the table.

Port source: `src/research_builder/models/claims.py` + `src/research_builder/orchestrator/claims.py` (which implements the verification logic — status assignment, tolerance/relative-delta rules).

## Wiring in a paper repo

Each paper's `.mcp.json`:

```json
{
  "mcpServers": {
    "paper": {
      "command": "python",
      "args": ["/path/to/research-builder/mcp/paper_reader_server.py"],
      "env": { "PAPER_PDF_PATH": "paper/paper.pdf" }
    },
    "arxiv": {
      "command": "python",
      "args": ["/path/to/research-builder/mcp/arxiv_server.py"]
    },
    "claims": {
      "command": "python",
      "args": ["/path/to/research-builder/mcp/claims_server.py"],
      "env": { "CLAIMS_PATH": "notes/claims.md" }
    }
  }
}
```

## What's explicitly NOT here

- **No `report_result` tool.** In the harness, sub-agents called `report_result` to write a JSON to a known path so the parent could read it. In the native approach, Claude Code *is* the parent — there's no cross-process handoff. Structured run outputs live in `runs/<run-id>/result.json` and are inspected in-session.
- **No `request_compute` tool.** Out of scope for v1; `cloud/` provisioning stays archived until we decide whether GPU provisioning is a first-class MCP server or just a shell script.
