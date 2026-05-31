---
name: reproduce-not-search
description: When authoring a spec or implementing a phase, reproduce the paper's reported result with the paper's reported configuration — never redo the hyperparameter search the paper already paid for.
---

# Reproduce, Don't Re-Search

**Rule: reproduce the result, not the search.** The paper already ran the experiments and reports a winner. Your job is to reproduce *the figure / table / number*, using the configuration the paper says won. Re-running the grid search locally is wasted compute and almost never produces a meaningfully different answer.

This applies twice:
1. **Spec authoring** — write acceptance criteria that pin concrete hyperparameters, not "search the grid."
2. **Phase implementation** — when you read the spec or the paper, if it mentions a sweep, look for the winning config and run *just that*.

## Where the paper's winning config usually lives

In order of likelihood:
- The experiment section's text (e.g. "we used α=0.001, β₁=0.9").
- A figure caption ("Adam with α=0.001 in red").
- An appendix table of final hyperparameters.
- A "Default values" block in the methodology / algorithm box.
- The official code release referenced in the paper (last resort — check the paper text first).

If the paper paraphrases its methodology as *"we searched over a dense grid of learning rates and report the best,"* that's a description of how they got the numbers — not an instruction for you to redo it. Skip to where they report the winning value.

## The spec-authoring rule

When you write per-section acceptance criteria:

- **Do** record the concrete winning hyperparameters as numbers ("`α = 0.001`, `batch_size = 128`, `epochs = 45`").
- **Do** flag the config as a "Flagged ambiguity" if the paper genuinely doesn't pin it down, so a downstream refiner or researcher can find it from the appendix or the official code.
- **Don't** write criteria like "searched over dense grid; report best setting for each optimizer." That ports the paper's methodology into the spec and tells the builder to do a search.
- **Don't** write "tune learning rate" / "grid search" / "ablation over batch sizes" as acceptance criteria. If you need a sweep, see the carve-out below.

## The implementer's rule

When you (a sub-agent) see a spec or paper that mentions a sweep:

- Find the paper's reported best config for the experiment you're reproducing.
- Run **that one config**, end-to-end, and produce the deliverable.
- If you can't find a clear winner in the paper text / caption / appendix, call `report_result` with `is_spec_issue: true` rather than burning hours doing a local search.

A sweep across `N optimizers × M learning rates × K epochs × multiple datasets` will turn a "trains in seconds" model into hours of CPU wall-clock. The cloud provisioner judges per-trial cost, not sweep total — so a runaway sweep silently eats local CPU.

## The one legitimate exception: the sweep IS the deliverable

Some figures plot a sensitivity ablation — the curves themselves are the result, not just illustration of how the winner was chosen. Examples:

- A bias-correction ablation where the figure shows loss across `β₁ × β₂` values (e.g. Adam paper §6.4, Figure 4: a 2×3 grid).
- An optimizer-comparison figure where each curve IS one optimizer at its best, and the figure exists to compare them.
- An ablation table where each row is a different ablated variant.

In these cases the sweep IS reproduction. The rule still applies though: **match the paper's grid, don't invent a denser one.** If the paper sweeps β₁ over `{0.9, 0.99}`, you sweep `{0.9, 0.99}` — not `{0.5, 0.6, 0.7, ..., 0.99}`.

## Failure mode to recognize

The Adam paper §6.1 spec authored under the old prompt said:

> "Hyperparameters searched over dense grid; report best setting for each optimizer."

That's a verbatim paraphrase of the paper's methodology. Following it produced ~14 hours of CPU wall-clock for logistic-regression curves the paper already published. The correct spec would have been:

> "Run Adam (α=0.001, β₁=0.9, β₂=0.999), SGD+Nesterov (paper-reported best LR), and Adagrad (paper-reported best LR) on MNIST logistic regression. Plot training cost vs. epoch. Match Figure 1 left panel."

The figure is the deliverable. The search that picked the LRs is not.
