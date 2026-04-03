"""PDF paper extraction for targeted section retrieval (spec_v4 §5.2)."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_pages(pdf_path: str | Path, start: int, end: int | None = None) -> str:
    """Extract text from a range of pages in a PDF.

    Args:
        pdf_path: Path to the PDF file.
        start: First page number (1-indexed).
        end: Last page number (1-indexed, inclusive). If None, extracts only `start`.

    Returns:
        Extracted text with page markers.
    """
    if end is None:
        end = start

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Paper not found: {pdf_path}")

    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        if start < 1 or end > total:
            raise ValueError(f"Page range {start}-{end} out of bounds (paper has {total} pages)")

        for i in range(start - 1, end):  # pdfplumber uses 0-indexed
            page = pdf.pages[i]
            text = page.extract_text() or ""
            parts.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(parts)


def get_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_full_text(pdf_path: str | Path) -> str:
    """Extract all text from a PDF. Use sparingly — prefer extract_pages."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Paper not found: {pdf_path}")

    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            parts.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(parts)
