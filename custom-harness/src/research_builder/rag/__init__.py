"""RAG system for semantic paper search.

Builds a vector index over the research paper during spec creation.
Phase sub-agents query it via the search_paper MCP tool.
"""

from .index import PaperIndex, SearchResult

__all__ = ["PaperIndex", "SearchResult"]
