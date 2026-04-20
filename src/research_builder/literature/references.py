"""Extract references/bibliography from extracted paper text.

Heuristic-based: looks for a "References" or "Bibliography" heading and
extracts numbered or bracketed citation entries. Good enough for the
common formats — not a full BibTeX parser.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that signal the start of the references section.
_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+)?(?:references|bibliography|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Individual reference entries: [1] ..., 1. ..., or lines starting with an author list.
_ENTRY_RE = re.compile(r"^\s*\[?\d{1,3}\]?\s*[.\s]")


def extract_reference_strings(paper_text: str) -> list[str]:
    """Return a list of raw citation strings from the paper text.

    Each entry is the cleaned text of one reference (e.g. "Smith et al. (2023).
    Attention is all you need. NeurIPS."). Returns an empty list if the
    references section can't be located.
    """
    # Find the references heading
    m = _HEADING_RE.search(paper_text)
    if not m:
        logger.debug("No references heading found in paper text")
        return []

    ref_section = paper_text[m.end():]

    # Stop at the next section heading (if any) — heuristic: a line that
    # starts with "# " or all-caps short line or "--- Page" marker at the end.
    stop = re.search(
        r"^(?:#{1,3}\s+\S|[A-Z][A-Z ]{5,}\s*$|--- Page \d+)",
        ref_section,
        re.MULTILINE,
    )
    if stop:
        ref_section = ref_section[:stop.start()]

    # Split into individual references.
    entries: list[str] = []
    current: list[str] = []

    for line in ref_section.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                entries.append(" ".join(current))
                current = []
            continue

        # New entry starts with [N] or N. pattern
        if _ENTRY_RE.match(stripped):
            if current:
                entries.append(" ".join(current))
            # Strip the leading number/bracket
            cleaned = re.sub(r"^\s*\[?\d{1,3}\]?\s*[.\s]*", "", stripped).strip()
            current = [cleaned] if cleaned else []
        else:
            current.append(stripped)

    if current:
        entries.append(" ".join(current))

    # Filter out very short entries (likely parsing noise)
    entries = [e for e in entries if len(e) > 15]

    logger.info("Extracted %d reference entries from paper", len(entries))
    return entries


def extract_citation_titles(paper_text: str) -> list[str]:
    """Best-effort extraction of just the paper titles from reference strings.

    Uses a heuristic: the title is typically the text before the first period
    that follows the author list, or the text in quotes/italics. Falls back
    to the first ~100 chars of each entry (still a decent search query).
    """
    entries = extract_reference_strings(paper_text)
    titles: list[str] = []

    for entry in entries:
        # Try "Title." pattern: authors (year). Title. Venue.
        # Look for a year-like pattern, then grab the next sentence.
        year_m = re.search(r"\(?\d{4}[a-z]?\)?[.,]?\s*", entry)
        if year_m:
            after_year = entry[year_m.end():]
            # Title is typically the text up to the next period
            dot = after_year.find(".")
            if dot > 5:
                titles.append(after_year[:dot].strip())
                continue

        # Fallback: use first 100 chars as a search query
        titles.append(entry[:100].strip())

    return titles
