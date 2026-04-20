---
name: verification-ladder
description: Gate each reproduction run through cheap→expensive checks. Each rung must pass before you pay for the next.
---

# Verification Ladder

**Rule: do not move to the next rung until the current one passes.** This is the single biggest lever for not wasting days on training runs that never had a chance of working.

Ordered cheapest → most expensive:

## 1. Unit tests on individual components
- Attention mask shape, loss numerics on a toy input, layer init stats, data-loader output shapes.
- Runnable in seconds. Run on every change.
- Reference: `tests/` in the paper repo.

## 2. Replicate a toy example or Figure 1 demo exactly
- If the paper shows a didactic example, reproduce it bit-for-bit first.
- Often catches a misread equation before you ever touch the real model.

## 3. Overfit a single batch
- Train on one batch, watch loss → ~0.
- If loss won't collapse on a single batch, the model, loss, or optimizer is wrong — not the data scale, not the hyperparameters.
- Reference script: `scripts/overfit-one-batch.sh`.

## 4. Smoke run
- ~100 training steps on a tiny dataset slice.
- Purpose: end-to-end pipeline confirmation. Does data → model → loss → step → checkpoint all work together?
- Does NOT prove the model is right, only that nothing is broken.
- Reference script: `scripts/smoke.sh`.

## 5. Short training run
- 1–10% of full training.
- Compare early-training loss curve *shape* to the paper's if available (Figure X often shows this).
- Catches learning-rate / schedule / batch-size bugs that smoke runs miss.

## 6. Full reproduction
- Only after 1–5 have all passed.
- Log the exact config alongside the checkpoint. Fix seeds. Pin CUDA/PyTorch versions.
- Use `scripts/reproduce.sh`.

## Per-phase verification focus

When a phase finishes, verify these before moving on:

**Data phase**
- Row counts match expected splits.
- Schema validates (column names, dtypes).
- No NaN/null in required fields.
- Distribution spot-checks: label balance, feature ranges.
- Loader produces correctly shaped batches.

**Architecture phase**
- Model instantiates without error.
- Forward on dummy input → expected output shapes.
- Parameter count matches paper (if reported).
- Gradients flow through all layers (no dead layers).

**Training phase**
- Loss decreases over first N steps (not diverging).
- No NaN/Inf in gradients or loss.
- Checkpoints are written and loadable.
- Learning-rate schedule matches spec at sampled steps.

**Eval phase**
- Metrics are computable (no errors on eval set).
- Results are within plausible range (not zero, not absurdly large).
- All specified metrics are reported.
- Output format matches schema.

**Results phase**
- Report renders (valid markdown).
- All target tables and figures present.
- Figures contain data (not blank).
- Comparison values populated.

## When results don't match the paper

Resist the urge to tweak hyperparameters. Instead:

1. **State the specific discrepancy as a number.** "Paper reports 84.2%, we get 79.6% — delta 4.6pp."
2. **List plausible causes ranked by likelihood** in `notes/journal.md`.
3. **Design the cheapest experiment that distinguishes the top two.**
4. **Run it. Update the ranking.**
5. **Repeat.**

Keeping the hypothesis list *externalized* (in a file, not in conversation) is what makes this loop work across sessions.
