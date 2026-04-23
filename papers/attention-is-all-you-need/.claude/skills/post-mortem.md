---
name: post-mortem
description: Produce a structured diagnosis after a failed run — one focused hypothesis, with the code-vs-spec classification.
---

# Post-Mortem

Run this after any failed experiment before attempting another. The goal is **one focused hypothesis the next attempt can plan against**, not fix the code right now.

## Inputs

- The attempt's output/result file (`runs/<run-id>/result.json` or equivalent).
- Logs, stderr, test output in the run directory.
- The source code that was run.
- Any partial outputs.

## Output

Write to `notes/post-mortems/<run-id>.md` with this frontmatter + body:

```markdown
---
run_id: <run-id>
timestamp: <ISO 8601>
phase: data | architecture | training | eval | results
confidence: low | medium | high
is_likely_spec_issue: true | false
---

## Failure hypothesis
One sentence. What you think actually went wrong. Quote specific error lines or symptoms.

## Suggested fix
One or two sentences. Concrete next-attempt direction.

## Evidence
- Relevant log lines (with file:line if possible)
- Relevant output numbers
- What you ruled out
```

## Rules

- **ONE focused hypothesis.** If you have three, pick the most likely one and note the others as alternatives in Evidence. Don't enumerate ten possibilities — that's not a diagnosis.
- **No generic advice.** "Add more tests", "tune hyperparameters", "check your data" are not hypotheses. They're fillers.
- **Quote specifics.** `ValueError: expected shape (B, H, L, D), got (B, L, H, D) at model.py:142` is a hypothesis. "Shape mismatch" is not.
- **Classify: spec vs. code.**
  - **Implementation issue** (`is_likely_spec_issue: false`) — wrong API call, shape mismatch in code you wrote, missed edge case. Retry with the same spec will likely succeed.
  - **Spec issue** (`is_likely_spec_issue: true`) — the spec (or `CLAUDE.md`) is genuinely under-specified or contradictory. No amount of retrying against the current spec will fix it. Go back to the paper and amend `CLAUDE.md` first.

## Confidence rubric

- **high** — you can point to a specific line of code or log that caused the failure, and the fix is obvious.
- **medium** — you have a single most-likely cause, but verifying requires a targeted experiment.
- **low** — multiple plausible causes, no clear signal. Design a distinguishing experiment before retrying.

## Why this matters

The post-mortem is a discipline against the LLM tendency to "try something different and see." Every retry that doesn't start with a hypothesis is information-destroying — you burn compute without learning. A weak hypothesis that turns out wrong is still better than no hypothesis, because the wrongness *is* the learning.

## Connection to the verification ladder

If the post-mortem hypothesis is "we jumped to full training without passing overfit-one-batch", the suggested fix is always: **drop back to the lowest failed rung of the verification ladder.** Don't retry at the same level.
