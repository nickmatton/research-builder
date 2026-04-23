"""Thin Semantic Scholar API client for citation lookups.

Uses the free (unauthenticated) API. Rate limits are generous for our use case
(~100 req/5min). All methods are async and return plain dicts — callers decide
what to render.

Docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,venue,externalIds,url,authors"
REF_FIELDS = "title,abstract,year,venue,externalIds,url"


@dataclass
class PaperInfo:
    """Minimal paper metadata returned by the client."""

    paper_id: str = ""
    title: str = ""
    abstract: str | None = None
    year: int | None = None
    venue: str = ""
    url: str = ""
    authors: list[str] = field(default_factory=list)
    arxiv_id: str | None = None

    def short_citation(self) -> str:
        first_author = self.authors[0] if self.authors else "Unknown"
        return f"{first_author} et al., {self.year or '?'}"

    def to_markdown(self) -> str:
        lines = [f"### {self.title}"]
        lines.append(f"*{self.short_citation()}* — {self.venue}")
        if self.url:
            lines.append(f"Link: {self.url}")
        if self.abstract:
            # Trim very long abstracts
            abstract = self.abstract if len(self.abstract) <= 800 else self.abstract[:800] + "..."
            lines.append(f"\n{abstract}")
        return "\n".join(lines)


class SemanticScholarClient:
    """Async Semantic Scholar API client.

    Best-effort: all methods swallow network errors and return empty results
    so the pipeline is never blocked by a flaky external API.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def search_by_title(self, title: str, limit: int = 3) -> list[PaperInfo]:
        """Search for papers by title. Returns up to ``limit`` results."""
        params = {"query": title, "limit": limit, "fields": FIELDS}
        data = await self._get("/paper/search", params=params)
        return [_parse_paper(p) for p in (data.get("data") or [])]

    async def get_paper(self, paper_id: str) -> PaperInfo | None:
        """Get a single paper by Semantic Scholar ID, DOI, or ArXiv ID.

        Accepts IDs like ``"DOI:10.18653/..."`` or ``"ArXiv:2301.12345"``.
        """
        data = await self._get(f"/paper/{paper_id}", params={"fields": FIELDS})
        if not data or "paperId" not in data:
            return None
        return _parse_paper(data)

    async def get_references(self, paper_id: str, limit: int = 50) -> list[PaperInfo]:
        """Get papers cited BY this paper (its reference list)."""
        data = await self._get(
            f"/paper/{paper_id}/references",
            params={"fields": REF_FIELDS, "limit": limit},
        )
        results = []
        for item in data.get("data") or []:
            cited = item.get("citedPaper")
            if cited and cited.get("title"):
                results.append(_parse_paper(cited))
        return results

    async def resolve_citations(
        self,
        citation_strings: list[str],
        max_concurrent: int = 5,
    ) -> list[PaperInfo]:
        """Best-effort batch resolve: search each citation string by title.

        Returns one result per citation (the top hit) or skips on miss.
        Limits concurrency so we don't hammer the API.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _resolve_one(cite: str) -> PaperInfo | None:
            async with sem:
                results = await self.search_by_title(cite, limit=1)
                return results[0] if results else None

        tasks = [_resolve_one(c) for c in citation_strings]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in settled if isinstance(r, PaperInfo)]

    # ── Internal ──────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for attempt in range(3):
                    resp = await client.get(f"{BASE_URL}{path}", params=params)
                    if resp.status_code == 429:
                        backoff = 5 * (2 ** attempt)  # 5s, 10s, 20s
                        logger.warning(
                            "Semantic Scholar rate limit hit; backing off %ds (attempt %d/3)",
                            backoff, attempt + 1,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    return resp.json()
                # All retries hit 429
                logger.warning("Semantic Scholar rate limit exhausted for %s after 3 attempts", path)
                return {}
        except httpx.HTTPStatusError as e:
            logger.warning("Semantic Scholar HTTP %s for %s: %s", e.response.status_code, path, e)
            return {}
        except Exception as e:
            logger.warning("Semantic Scholar request failed for %s: %s", path, e)
            return {}


def _parse_paper(raw: dict) -> PaperInfo:
    authors = [a.get("name", "") for a in (raw.get("authors") or [])]
    ext = raw.get("externalIds") or {}
    return PaperInfo(
        paper_id=raw.get("paperId", ""),
        title=raw.get("title", ""),
        abstract=raw.get("abstract"),
        year=raw.get("year"),
        venue=raw.get("venue", ""),
        url=raw.get("url", ""),
        authors=authors,
        arxiv_id=ext.get("ArXiv"),
    )
