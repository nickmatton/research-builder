"""MCP server: look up cited papers via Semantic Scholar.

Use when the paper (or your spec) references a method, dataset, or technique
from another paper and you need implementation details from the abstract /
venue / author list.

Authenticated calls via ``SEMANTIC_SCHOLAR_API_KEY`` env var are much less
rate-limited than unauthenticated ones.
"""

from __future__ import annotations

import asyncio
import os

import httpx
from mcp.server.fastmcp import FastMCP


BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,venue,externalIds,url,authors"

mcp = FastMCP("arxiv")


def _headers() -> dict[str, str]:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


async def _get(path: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=15.0, headers=_headers()) as client:
        for attempt in range(3):
            resp = await client.get(f"{BASE_URL}{path}", params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5 * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        return {}


def _parse(raw: dict) -> dict:
    authors = [a.get("name", "") for a in (raw.get("authors") or [])]
    ext = raw.get("externalIds") or {}
    return {
        "paper_id": raw.get("paperId", ""),
        "title": raw.get("title", ""),
        "abstract": raw.get("abstract"),
        "year": raw.get("year"),
        "venue": raw.get("venue", ""),
        "url": raw.get("url", ""),
        "authors": authors,
        "arxiv_id": ext.get("ArXiv"),
        "doi": ext.get("DOI"),
    }


@mcp.tool()
async def lookup_citation(query: str, limit: int = 3) -> list[dict]:
    """Search Semantic Scholar by title or citation string. Returns up to
    ``limit`` matches with title, abstract, year, venue, authors, URL.

    Examples:
        lookup_citation("Attention Is All You Need")
        lookup_citation("BERT pre-training deep bidirectional transformers")
    """
    data = await _get("/paper/search", {"query": query, "limit": limit, "fields": FIELDS})
    return [_parse(p) for p in (data.get("data") or [])]


@mcp.tool()
async def get_paper(paper_id: str) -> dict | None:
    """Fetch a single paper by Semantic Scholar ID, DOI, or ArXiv ID.

    Accepts identifiers like ``ArXiv:2301.12345`` or ``DOI:10.18653/...``.
    Returns None if the paper isn't found.
    """
    data = await _get(f"/paper/{paper_id}", {"fields": FIELDS})
    if not data or "paperId" not in data:
        return None
    return _parse(data)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
