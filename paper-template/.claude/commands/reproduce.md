---
description: Run a full reproduction, verify against claims, append to journal.
---

Run the full reproduction pipeline for this paper and log the result.

## Steps

1. **Pre-flight**: Confirm cheaper rungs of the verification ladder pass first. Run `uv run pytest` (unit tests). If anything fails, **STOP** and ask the user before proceeding — running a full reproduction over a broken codebase wastes time and compute.

2. **Run**: Execute `bash scripts/reproduce.sh ${1:-configs/full.yaml}`. Stream output to the user as it runs.

3. **Verify**: Once `runs/<run-id>/metrics.json` is written, call `claims.verify_run(metrics=<the dict>)`. Save the resulting markdown table to `runs/<run-id>/claims-report.md`.

4. **Journal**: Append a new run block to `notes/journal.md` using the template at the top of that file. Include:
   - run-id (matches the directory name)
   - ISO 8601 timestamp
   - Git SHA (from `runs/<run-id>/git_sha.txt`)
   - Config path
   - Hardware (best guess from environment)
   - Wall-clock duration (from train.log timestamps)
   - Top-line metrics
   - Claims summary counts
   - One-sentence note on what this run proved or failed to prove

5. **Verdict**:
   - If all headline claims are `verified` or `close`: report success.
   - If any are `missed`: tell the user, point at `claims-report.md`, and ask whether to invoke `/post-mortem` next.
   - If any are `exceeded`: this is a **red flag** (data leak, wrong eval split, metric mismatch). Do NOT report success. Ask the user to investigate before accepting.

## Notes

- Don't tweak hyperparameters silently to make claims pass. If results don't match, that's information — log it, post-mortem it, then iterate.
- One full reproduction run can take hours. Confirm with the user before starting if the config implies a long run.
