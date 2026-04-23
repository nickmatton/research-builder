---
description: Verify a run's metrics against the claims ledger.
---

Compare a completed run's metrics against `notes/claims.yaml`.

## Steps

1. **Locate the run**: If the user supplied a run-id (`$1`), use `runs/$1/metrics.json`. Otherwise pick the most recent `runs/*/metrics.json`.

2. **Verify**: Run `python scripts/compare-claims.py runs/<run-id>/metrics.json | tee runs/<run-id>/claims-report.md`. The script prints the markdown table to stdout and writes it to `claims-report.md` via tee.

3. **Parse**: The last line of stdout is `SUMMARY: {...}` JSON — read it for per-status counts.

4. **Read the rubric**: status meanings and the "exceeded = red flag" rule come from `.claude/skills/compare-to-paper.md`. Apply them — don't paper over `exceeded` results.

5. **Recommendation**:
   - All `verified` / `close`: green light.
   - Any `missed`: suggest `/post-mortem <run-id>` for the failed claims.
   - Any `exceeded` with margin > 2× tolerance: flag loudly. Suggest investigating before accepting (data leak, eval split, metric definition).
   - All `not_checked`: the metric names in the run output don't line up with the claims ledger. Help the user fix the metric names or the claim_ids so they match.
