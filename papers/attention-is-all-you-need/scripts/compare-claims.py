#!/usr/bin/env python3
"""Compare run metrics against the claims ledger. Prints markdown table.

    python scripts/compare-claims.py runs/<run-id>/metrics.json
    python scripts/compare-claims.py --claims notes/claims.yaml runs/<id>/metrics.json

Reads:
    notes/claims.yaml — the ledger (default; override with --claims)
    <metrics.json>    — flat JSON dict of { claim_id_or_metric_name: value }

Writes (to stdout):
    Markdown table per the rubric in .claude/skills/compare-to-paper.md
    plus a JSON summary of {verified, close, missed, exceeded, not_checked}
    on the last line, prefixed ``SUMMARY: ``.

Exit code: 0 if no claims missed/exceeded; 1 otherwise.

Dependency: PyYAML. ``uv pip install pyyaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def classify(expected: float, actual: float | None, tolerance: float) -> dict:
    if actual is None:
        return {"status": "not_checked", "expected": expected, "actual": None,
                "delta": None, "delta_pct": None, "note": "no matching value in metrics"}
    delta = actual - expected
    delta_pct = (delta / expected * 100) if expected != 0 else None
    tol = tolerance if tolerance > 0 else max(abs(expected * 0.05), 0.5)
    abs_delta = abs(delta)
    if abs_delta <= tol:
        status, note = "verified", ""
    elif abs_delta <= tol * 2:
        status, note = "close", f"outside tolerance, within 2× (±{tol:.4g})"
    elif delta > 0 and abs_delta > tol * 2:
        status = "exceeded"
        note = (f"actual ({actual:.4g}) exceeds paper ({expected:.4g}) by >2× tolerance — "
                "suspect data leak / wrong split / metric mismatch")
    else:
        status, note = "missed", f"outside 2× tolerance (±{tol:.4g})"
    return {"status": status, "expected": expected, "actual": actual,
            "delta": delta, "delta_pct": delta_pct, "note": note}


def render_table(details: list[dict]) -> str:
    icon = {"verified": "✓", "close": "~", "missed": "✗",
            "exceeded": "⚠", "not_checked": "—"}
    rows = ["| Status | Claim | Expected | Actual | Delta |",
            "|--------|-------|----------|--------|-------|"]
    for d in details:
        actual = f"{d['actual']:.4g}" if d['actual'] is not None else "—"
        delta = ""
        if d['delta'] is not None:
            sign = "+" if d['delta'] >= 0 else ""
            delta = f"{sign}{d['delta']:.4g}"
            if d['delta_pct'] is not None:
                delta += f" ({sign}{d['delta_pct']:.1f}%)"
        rows.append(f"| {icon.get(d['status'], '?')} | {d['claim_id']} | "
                    f"{d['expected']:.4g} | {actual} | {delta} |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", type=Path, help="Path to run metrics JSON")
    p.add_argument("--claims", type=Path, default=Path("notes/claims.yaml"))
    args = p.parse_args()

    if not args.claims.exists():
        print(f"error: {args.claims} not found", file=sys.stderr)
        return 2
    if not args.metrics.exists():
        print(f"error: {args.metrics} not found", file=sys.stderr)
        return 2

    claims = (yaml.safe_load(args.claims.read_text()) or {}).get("claims") or []
    metrics = json.loads(args.metrics.read_text())
    if not isinstance(metrics, dict):
        print("error: metrics JSON must be a flat dict { name: value }", file=sys.stderr)
        return 2

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
        details.append({"claim_id": cid,
                        **classify(float(c["value"]), actual_f, float(c.get("tolerance", 0)))})

    summary = {s: sum(1 for d in details if d["status"] == s)
               for s in ("verified", "close", "missed", "exceeded", "not_checked")}

    print(render_table(details))
    print()
    print(f"**Summary:** {summary['verified']} verified, {summary['close']} close, "
          f"{summary['missed']} missed, {summary['exceeded']} suspicious, "
          f"{summary['not_checked']} unchecked")
    print()
    print(f"SUMMARY: {json.dumps(summary)}")

    return 0 if summary["missed"] == 0 and summary["exceeded"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
