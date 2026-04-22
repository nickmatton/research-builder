"""MCP server: read the paper PDF.

Exposes page-range extraction, page count, and a simple substring/regex search.
Path is auto-discovered from the paper repo's working directory (``paper/*.pdf``)
or overridden via the ``PAPER_PDF_PATH`` env var.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pdfplumber
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("paper")


def _paper_path() -> Path:
    env = os.environ.get("PAPER_PDF_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    paper_dir = Path.cwd() / "paper"
    if paper_dir.is_dir():
        pdfs = sorted(paper_dir.glob("*.pdf"))
        if len(pdfs) == 1:
            return pdfs[0]
        if len(pdfs) > 1:
            raise RuntimeError(
                f"Multiple PDFs in {paper_dir}; set PAPER_PDF_PATH to disambiguate"
            )
    raise FileNotFoundError(
        "No paper PDF found. Set PAPER_PDF_PATH or place a single .pdf in ./paper/"
    )


@mcp.tool()
def read_paper_section(start_page: int, end_page: int | None = None) -> str:
    """Read specific pages from the paper PDF. 1-indexed, inclusive.

    Omit ``end_page`` for a single page. Use for targeted section / figure /
    table retrieval when you know roughly where to look.
    """
    path = _paper_path()
    end = end_page if end_page is not None else start_page
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        if start_page < 1 or end > total:
            raise ValueError(f"Page range {start_page}-{end} out of bounds (paper has {total} pages)")
        parts: list[str] = []
        for i in range(start_page - 1, end):
            text = pdf.pages[i].extract_text() or ""
            parts.append(f"--- Page {i + 1} ---\n{text}")
    header = f"[Paper: pages {start_page}-{end} of {total}]\n\n"
    return header + "\n\n".join(parts)


@mcp.tool()
def page_count() -> int:
    """Return the number of pages in the paper PDF."""
    with pdfplumber.open(_paper_path()) as pdf:
        return len(pdf.pages)


@mcp.tool()
def search_paper(pattern: str, context_chars: int = 200, max_matches: int = 20) -> list[dict]:
    """Search the paper for a regex pattern. Returns up to ``max_matches``
    hits with a context snippet and page number.

    Use this before ``read_paper_section`` when you know what you're looking
    for (e.g. "learning rate", "β1 = 0\\.9") but not the page. Case-insensitive.
    """
    path = _paper_path()
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}") from e

    results: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for m in regex.finditer(text):
                start = max(0, m.start() - context_chars // 2)
                end = min(len(text), m.end() + context_chars // 2)
                snippet = text[start:end].replace("\n", " ").strip()
                results.append({
                    "page": i,
                    "match": m.group(0),
                    "snippet": snippet,
                })
                if len(results) >= max_matches:
                    return results
    return results


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
