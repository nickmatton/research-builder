#!/usr/bin/env python3
"""Look up a cited paper via Semantic Scholar. Prints JSON to stdout.

    python scripts/lookup-citation.py "Attention Is All You Need"
    python scripts/lookup-citation.py --limit 5 "BERT pre-training"
    python scripts/lookup-citation.py --id ArXiv:1706.03762

The script handles the API key (``SEMANTIC_SCHOLAR_API_KEY`` env var) so
WebFetch doesn't have to. Output is JSON with title, abstract, year, venue,
authors, url, arxiv_id, doi for each hit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,venue,externalIds,url,authors"


def _request(path: str, params: dict) -> dict:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        req.add_header("x-api-key", key)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"error: HTTP {e.code} from {url}", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            return {}
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", help='Title / citation string, OR a paper ID with --id')
    p.add_argument("--id", action="store_true", help="Treat query as a paper ID (ArXiv:..., DOI:..., S2 paperId)")
    p.add_argument("--limit", type=int, default=3, help="Max results for title search")
    args = p.parse_args()

    if args.id:
        data = _request(f"/paper/{args.query}", {"fields": FIELDS})
        out = [_parse(data)] if data and data.get("paperId") else []
    else:
        data = _request("/paper/search", {"query": args.query, "limit": args.limit, "fields": FIELDS})
        out = [_parse(p) for p in (data.get("data") or [])]

    print(json.dumps(out, indent=2))
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
