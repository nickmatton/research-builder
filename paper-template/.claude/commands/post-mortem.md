---
description: Diagnose a failed run. One focused hypothesis, code-vs-spec classified.
---

Produce a structured post-mortem for a failed run. **Goal: one focused hypothesis the next attempt can plan against. Not fix the code right now.**

Read `.claude/skills/post-mortem.md` first — the rules below are a summary; the skill has the full rubric and reasoning.

## Steps

1. **Locate the run**: if the user supplied a run-id (`$1`), use `runs/$1/`. Otherwise pick the most recent failing run (look for `runs/*/train.log` with errors, or `claims-report.md` with `missed` / `exceeded` results).

2. **Inspect**: read the run directory in this order:
   - `runs/<run-id>/result.json` or `metrics.json` (if present)
   - `runs/<run-id>/train.log`, `eval.log` (tail for stack traces / divergence)
   - `runs/<run-id>/claims-report.md` (if from a full run)
   - The source files in `src/` that ran in this attempt
   - Any partial outputs in `runs/<run-id>/`

3. **Diagnose**: produce ONE focused hypothesis. Quote specific error lines or symptom numbers — generic advice is forbidden ("add more tests", "tune hyperparameters" — no).

4. **Classify**:
   - **Implementation issue** (`is_likely_spec_issue: false`) — wrong API, shape mismatch, missed edge case. Retry with the same spec will likely succeed.
   - **Spec issue** (`is_likely_spec_issue: true`) — `CLAUDE.md` is genuinely under-specified or contradictory. Going back to the paper and amending CLAUDE.md is the next move, NOT another retry.

5. **Confidence**: high (specific line, obvious fix), medium (one most-likely cause, needs a targeted experiment), low (multiple plausible causes).

6. **Write** the post-mortem to `notes/post-mortems/<run-id>.md` using the template in `.claude/skills/post-mortem.md`. Include the failure hypothesis, suggested fix, classification, confidence, and evidence (with file:line where possible).

7. **Recommend next step**: the cheapest experiment that distinguishes this hypothesis from the alternatives. If it's a spec issue, recommend the specific CLAUDE.md section to amend with the paper reference.
