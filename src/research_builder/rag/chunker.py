"""Chunk research paper text into passages with metadata.

Operates on text already extracted by pdfplumber (with --- Page N --- markers).
Produces chunks suitable for embedding and semantic search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A passage from the paper with location metadata."""

    text: str
    chunk_id: int
    page_start: int  # 1-indexed
    page_end: int  # 1-indexed
    section_heading: str

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.chunk_id}, pages {self.page_start}-{self.page_end}, '{preview}...')"


# Regex for section headings common in academic papers.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:abstract|introduction|related work|background|method|methods|methodology|"
    r"approach|experiments?|results?|discussion|conclusion|conclusions|"
    r"acknowledgements?|references|appendix|supplementary)"
    r"|"
    r"(?:\d+\.[\d.]*\s+\S.*)"  # numbered: "3.2 Multi-Head Attention"
    r"|"
    r"(?:[A-Z]\.[\d.]*\s+\S.*)"  # appendix-style: "A.1 Details"
    r")$",
    re.IGNORECASE,
)

# Page marker inserted by extract_full_text / extract_pages.
_PAGE_MARKER_RE = re.compile(r"^---\s*Page\s+(\d+)\s*---$")

# Target chunk size in characters (~300-500 tokens).
_TARGET_CHUNK_SIZE = 1500
_MAX_CHUNK_SIZE = 2500


def chunk_paper(full_text: str) -> list[Chunk]:
    """Split extracted paper text into chunks with metadata.

    Args:
        full_text: Paper text with ``--- Page N ---`` markers (from extract_full_text).

    Returns:
        List of Chunk objects ordered by position in the paper.
    """
    # Parse into (page_number, line) pairs.
    lines_with_pages = _parse_pages(full_text)
    if not lines_with_pages:
        return []

    # Group lines into sections by detecting headings.
    sections = _split_into_sections(lines_with_pages)

    # Chunk each section by paragraph boundaries, respecting size limits.
    chunks: list[Chunk] = []
    chunk_id = 0
    for heading, section_lines in sections:
        section_chunks = _chunk_section(section_lines, heading, chunk_id)
        chunks.extend(section_chunks)
        chunk_id += len(section_chunks)

    return chunks


def _parse_pages(full_text: str) -> list[tuple[int, str]]:
    """Parse text into (page_number, line) pairs using page markers."""
    result: list[tuple[int, str]] = []
    current_page = 1

    for line in full_text.split("\n"):
        m = _PAGE_MARKER_RE.match(line.strip())
        if m:
            current_page = int(m.group(1))
            continue
        result.append((current_page, line))

    return result


def _split_into_sections(
    lines_with_pages: list[tuple[int, str]],
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Group lines into sections by heading detection."""
    sections: list[tuple[str, list[tuple[int, str]]]] = []
    current_heading = "Preamble"
    current_lines: list[tuple[int, str]] = []

    for page, line in lines_with_pages:
        stripped = line.strip()
        if stripped and _is_heading(stripped):
            # Save current section if it has content.
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append((page, line))

    # Don't forget the last section.
    if current_lines:
        sections.append((current_heading, current_lines))

    return sections


def _is_heading(line: str) -> bool:
    """Detect whether a line is a section heading."""
    # Skip very long lines — headings are short.
    if len(line) > 100:
        return False
    return bool(_HEADING_RE.match(line))


def _chunk_section(
    lines: list[tuple[int, str]],
    heading: str,
    start_id: int,
) -> list[Chunk]:
    """Split a section's lines into chunks at paragraph boundaries."""
    # Collect paragraphs (separated by blank lines).
    paragraphs: list[tuple[str, int, int]] = []  # (text, page_start, page_end)
    current_para_lines: list[str] = []
    para_page_start: int = lines[0][0] if lines else 1
    para_page_end: int = para_page_start

    for page, line in lines:
        if not line.strip():
            if current_para_lines:
                text = "\n".join(current_para_lines).strip()
                if text:
                    paragraphs.append((text, para_page_start, para_page_end))
                current_para_lines = []
                para_page_start = page
        else:
            if not current_para_lines:
                para_page_start = page
            current_para_lines.append(line)
            para_page_end = page

    # Flush remaining.
    if current_para_lines:
        text = "\n".join(current_para_lines).strip()
        if text:
            paragraphs.append((text, para_page_start, para_page_end))

    if not paragraphs:
        return []

    # Merge small paragraphs into chunks up to target size.
    chunks: list[Chunk] = []
    chunk_id = start_id
    buf_text = ""
    buf_page_start = paragraphs[0][1]
    buf_page_end = paragraphs[0][2]

    for para_text, p_start, p_end in paragraphs:
        candidate = (buf_text + "\n\n" + para_text).strip() if buf_text else para_text

        if len(candidate) > _MAX_CHUNK_SIZE and buf_text:
            # Flush buffer, start new chunk with this paragraph.
            chunks.append(Chunk(
                text=buf_text,
                chunk_id=chunk_id,
                page_start=buf_page_start,
                page_end=buf_page_end,
                section_heading=heading,
            ))
            chunk_id += 1
            buf_text = para_text
            buf_page_start = p_start
            buf_page_end = p_end
        elif len(candidate) > _TARGET_CHUNK_SIZE and buf_text:
            # Target reached — flush and start new.
            chunks.append(Chunk(
                text=buf_text,
                chunk_id=chunk_id,
                page_start=buf_page_start,
                page_end=buf_page_end,
                section_heading=heading,
            ))
            chunk_id += 1
            buf_text = para_text
            buf_page_start = p_start
            buf_page_end = p_end
        else:
            buf_text = candidate
            buf_page_end = p_end
            if not buf_text or buf_text == para_text:
                buf_page_start = p_start

    # Flush remaining buffer.
    if buf_text.strip():
        chunks.append(Chunk(
            text=buf_text,
            chunk_id=chunk_id,
            page_start=buf_page_start,
            page_end=buf_page_end,
            section_heading=heading,
        ))

    return chunks
