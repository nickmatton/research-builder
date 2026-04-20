---
name: compare-to-paper
description: Verify a run's numeric results against the claims ledger. Flags suspicious "too good" results as loudly as misses.
---

# Compare to Paper

This is the acceptance gate. Every full reproduction run feeds through it. Inputs: the run's result metrics + `notes/claims.md`. Output: a claims-verification table and a go/no-go.

## Status rubric

For each claim, assign one status:

| Status | Meaning | Action |
|---|---|---|
| **verified** | Within stated tolerance. | Accept. |
| **close** | Within 2× tolerance or 5% relative. | Accept, but note the deviation in the journal. |
| **missed** | Outside tolerance. | Investigate. Write a post-mortem. Do not silently accept. |
| **exceeded** | Suspiciously better than the paper. | **Red flag.** Do NOT accept without explaining why. |
| **not_checked** | No matching result in this run. | Note. Don't reject solely on this. |

## Why "exceeded" is a red flag

A model that beats the paper by a lot is almost always one of:
- **Data leak** — eval set contaminated training.
- **Wrong eval split** — evaluating on train or dev by mistake.
- **Metric mismatch** — computing top-5 when paper reports top-1, or vice versa.
- **Different normalization** — your accuracy is over a different denominator.

Flag `exceeded` claims with a margin > 2× tolerance for manual inspection before accepting the run.

## Output format

Produce a markdown table:

```markdown
| Status | Claim | Expected | Actual | Delta |
|--------|-------|----------|--------|-------|
| ✓ | table2_cifar10_top1 | 95.2 | 95.4 | +0.2 (+0.2%) |
| ~ | table3_latency_ms | 12.3 | 13.1 | +0.8 (+6.5%) |
| ✗ | table4_transfer_f1 | 78.5 | 72.1 | -6.4 (-8.2%) |
| ⚠ | table2_imagenet_top1 | 76.0 | 82.3 | +6.3 (+8.3%) |

**Summary:** 1 verified, 1 close, 1 missed, 1 suspicious, 0 unchecked
```

Append this to `notes/journal.md` after each run, alongside the run's git SHA and config hash.

## Convergence sanity (training phase specifically)

Before comparing final metrics, sanity-check the training run itself:

- Did loss decrease roughly monotonically? Or did it diverge / plateau immediately?
- Did training actually run for the expected number of steps/epochs?
- Any NaN/Inf warnings anywhere?

A run that produces "good" final metrics but has a divergent loss curve is almost certainly broken. Don't accept it.

## Workflow

1. Run `scripts/reproduce.sh` (or equivalent full reproduction).
2. Load `runs/<run-id>/metrics.json`.
3. Load `notes/claims.md`.
4. For each claim, find the matching metric in the run output. Compute delta and assign status.
5. Render the table above.
6. If any `missed` or `exceeded (margin > 2×)`: write a post-mortem before accepting.
7. Append the table + verdict to `notes/journal.md`.

## Claim ID discipline

Claims should have stable, descriptive IDs (snake_case). Good:
- `table2_cifar10_top1`
- `section4_2_bleu_wmt14`
- `figure3_attention_entropy_layer6`

Bad:
- `claim_0`, `claim_1` (ordering can shift)
- `best_accuracy` (which dataset? which table?)

Stable IDs let you track the same claim across runs in `notes/journal.md`.
