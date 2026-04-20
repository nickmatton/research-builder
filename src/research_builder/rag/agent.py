"""Build the paper search index during spec creation.

This module is kept thin — it just calls PaperIndex.load_or_build
and handles errors. No sub-agent needed; the embedding runs directly
in the harness process.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .index import PaperIndex

logger = logging.getLogger(__name__)

INDEX_FILENAME = "paper_index.pkl"


def build_paper_index(paper_path: Path, spec_dir: Path) -> PaperIndex | None:
    """Build (or load cached) paper search index.

    Called from OrchestratorAgent.create_spec(). Non-fatal — returns None
    on failure so the pipeline continues without semantic search.

    Args:
        paper_path: Path to the research paper PDF.
        spec_dir: Directory where the index is cached (canonical_spec/).

    Returns:
        PaperIndex on success, None on failure.
    """
    index_path = spec_dir / INDEX_FILENAME
    try:
        index = PaperIndex.load_or_build(paper_path, index_path)
        logger.info(
            "Paper index ready: %d chunks, saved at %s",
            len(index.chunks),
            index_path,
        )
        return index
    except Exception as e:
        logger.warning(
            "Failed to build paper search index (search_paper tool will be unavailable): %s",
            e,
        )
        return None
