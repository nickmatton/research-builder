"""MCP server: claims ledger CRUD + run verification.

Wraps ``notes/claims.yaml`` — the canonical list of numerical claims extracted
from the paper. Provides:

- list/get/add operations for editing the ledger
- verify(claim_id, actual_value) — classify one metric
- verify_run(metrics) — bulk-compare a run's output against all claims, return
  the standard markdown table

Storage format matches ``paper-template/notes/claims.yaml``. Auto-discovers
``./notes/claims.yaml`` from cwd; override via ``CLAIMS_PATH`` env var.

Status rubric (see skills/compare-to-paper.md):
    verified | close | missed | exceeded | not_checked
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("claims")


def _claims_path() -> Path:
    env = os.environ.get("CLAIMS_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else Path.cwd() / p
    return Path.cwd() / "notes" / "claims.yaml"


def _load() -> list[dict]:
    path = _claims_path()
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    return list(raw.get("claims") or [])


def _save(claims: list[dict]) -> None:
    path = _claims_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"claims": claims}, sort_keys=False))


def _classify(expected: float, actual: float | None, tolerance: float) -> dict:
    """Compute status + delta per skills/compare-to-paper.md."""
    if actual is None:
        return {"status": "not_checked", "expected": expected, "actual": None,
                "delta": None, "delta_pct": None,
                "note": "No matching value supplied for verification."}

    delta = actual - expected
    delta_pct = (delta / expected * 100) if expected != 0 else None

    tol = tolerance if tolerance > 0 else max(abs(expected * 0.05), 0.5)
    abs_delta = abs(delta)

    if abs_delta <= tol:
        status = "verified"
        note = ""
    elif abs_delta <= tol * 2:
        status = "close"
        note = f"Outside tolerance but within 2×. Tolerance: ±{tol:.4g}."
    elif delta > 0 and abs_delta > tol * 2:
        status = "exceeded"
        note = (f"Result ({actual:.4g}) exceeds paper ({expected:.4g}) by more than "
                f"2× tolerance — suspect data leak, wrong eval split, or metric mismatch.")
    else:
        status = "missed"
        note = f"Outside 2× tolerance (±{tol:.4g})."

    return {"status": status, "expected": expected, "actual": actual,
            "delta": delta, "delta_pct": delta_pct, "note": note}


@mcp.tool()
def list_claims() -> list[dict]:
    """Return all claims in the ledger."""
    return _load()


@mcp.tool()
def get_claim(claim_id: str) -> dict | None:
    """Fetch a single claim by ID. Returns None if not found."""
    for c in _load():
        if c.get("claim_id") == claim_id:
            return c
    return None


@mcp.tool()
def add_claim(
    claim_id: str,
    metric: str,
    value: float,
    tolerance: float = 0.0,
    unit: str = "",
    dataset: str = "",
    condition: str = "",
    source: dict | None = None,
    phase: str = "",
    notes: str = "",
) -> dict:
    """Append or update a claim in the ledger. Idempotent on claim_id.

    ``source`` is an optional dict with keys ``table``, ``figure``, ``section``,
    ``page``, ``verbatim`` — where in the paper the claim appears.
    """
    claims = _load()
    new = {
        "claim_id": claim_id,
        "metric": metric,
        "value": float(value),
        "tolerance": float(tolerance),
        "unit": unit,
        "dataset": dataset,
        "condition": condition,
        "source": source or {},
        "phase": phase,
        "notes": notes,
    }
    replaced = False
    for i, c in enumerate(claims):
        if c.get("claim_id") == claim_id:
            claims[i] = new
            replaced = True
            break
    if not replaced:
        claims.append(new)
    _save(claims)
    return new


@mcp.tool()
def verify(claim_id: str, actual_value: float) -> dict:
    """Classify one actual value against a claim. Does NOT mutate the ledger.
    Returns {status, expected, actual, delta, delta_pct, note}.
    """
    claim = get_claim(claim_id)
    if claim is None:
        return {"status": "not_checked", "expected": None, "actual": actual_value,
                "delta": None, "delta_pct": None, "note": f"Unknown claim_id: {claim_id}"}
    return {"claim_id": claim_id,
            **_classify(float(claim["value"]), actual_value, float(claim.get("tolerance", 0)))}


@mcp.tool()
def verify_run(metrics: dict[str, Any]) -> dict:
    """Verify all claims against a dict of results.

    ``metrics`` is looked up by ``claim_id`` first, then by ``metric`` name.
    Missing entries map to ``not_checked``. Returns a dict with:

    - ``table``: a markdown table of per-claim results
    - ``summary``: counts of each status
    - ``details``: full per-claim verification list
    """
    claims = _load()
    details = []
    for c in claims:
        cid = c.get("claim_id", "")
        actual = metrics.get(cid)
        if actual is None:
            actual = metrics.get(c.get("metric", ""))
        try:
            actual_f = float(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_f = None
        verif = _classify(float(c["value"]), actual_f, float(c.get("tolerance", 0)))
        details.append({"claim_id": cid, **verif})

    status_icon = {"verified": "✓", "close": "~", "missed": "✗",
                   "exceeded": "⚠", "not_checked": "—"}
    header = ("| Status | Claim | Expected | Actual | Delta |\n"
              "|--------|-------|----------|--------|-------|")
    rows = []
    for d in details:
        actual_s = f"{d['actual']:.4g}" if d['actual'] is not None else "—"
        delta_s = ""
        if d['delta'] is not None:
            sign = "+" if d['delta'] >= 0 else ""
            delta_s = f"{sign}{d['delta']:.4g}"
            if d['delta_pct'] is not None:
                delta_s += f" ({sign}{d['delta_pct']:.1f}%)"
        rows.append(f"| {status_icon.get(d['status'], '?')} | {d['claim_id']} | "
                    f"{d['expected']:.4g} | {actual_s} | {delta_s} |")

    summary = {s: sum(1 for d in details if d["status"] == s)
               for s in ("verified", "close", "missed", "exceeded", "not_checked")}
    summary_line = (f"**Summary:** {summary['verified']} verified, "
                    f"{summary['close']} close, {summary['missed']} missed, "
                    f"{summary['exceeded']} suspicious, {summary['not_checked']} unchecked")

    return {
        "table": "\n".join([header, *rows, "", summary_line]),
        "summary": summary,
        "details": details,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
