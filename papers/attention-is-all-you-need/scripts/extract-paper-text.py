#!/usr/bin/env python3
"""Extract a PDF to plain text with page markers.

Run once after dropping the PDF into ``paper/``:

    python scripts/extract-paper-text.py

Output: ``paper/paper.txt`` with ``--- Page N ---`` markers between pages.
After this, Claude reads/greps the text file with built-in tools — no MCP
server needed for paper access.

Dependency: pdfplumber. Install: ``uv pip install pdfplumber``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber


def extract(pdf_path: Path, out_path: Path) -> int:
    """Extract pages with markers. Returns page count."""
    with pdfplumber.open(pdf_path) as pdf:
        parts = []
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            parts.append(f"--- Page {i} ---\n{text}")
        out_path.write_text("\n\n".join(parts))
        return len(pdf.pages)


def main() -> int:
    paper_dir = Path("paper")
    if not paper_dir.is_dir():
        print(f"error: {paper_dir}/ not found. Drop the PDF there first.", file=sys.stderr)
        return 1

    pdfs = sorted(paper_dir.glob("*.pdf"))
    if not pdfs:
        print(f"error: no .pdf in {paper_dir}/", file=sys.stderr)
        return 1
    if len(pdfs) > 1:
        print(f"error: multiple PDFs in {paper_dir}/. Keep only one.", file=sys.stderr)
        return 1

    pdf = pdfs[0]
    out = paper_dir / "paper.txt"
    n = extract(pdf, out)
    print(f"extracted {n} pages: {pdf.name} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
