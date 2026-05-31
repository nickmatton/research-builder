# Example run: test paper

Snapshot of a successful end-to-end harness run against the bundled 4-page test paper at `custom-harness/paper/test_paper.pdf`.

## Reproduce locally

```bash
cd custom-harness
uv sync
uv run research-builder --test
```

`--test` auto-selects the bundled paper, writes to `/tmp/rb-test`, and implies `--auto --dev --wipe`.

## Paper

A minimal study of "widget performance":

- 3-layer neural network trained with AdamW (lr=0.001, batch_size=32)
- 100 epochs of training
- Reported accuracy: 95.2%

## Result

Pipeline reached **95.5% test accuracy** (within 0.3% of the reported 95.2%).

Run stats: ~25 minutes, ~$1.15 in Claude tokens, 5 phases sequential.

## Layout

```
canonical_spec/
  spec.md              LLM-authored canonical interpretation
  state.yaml           machine-readable phase state and dependency graph
  revision_log.yaml    append-only event log

phases/
  data/1/              synthetic dataset generation (10/10 tests passing)
  architecture/1/      MLP model code + tests
  training/1/          training loop + tests (100-epoch log)
  eval/1/              eval script + tests (metrics JSON)
  results/1/           report generation + tests
```

## Key files

- [`phases/results/1/outputs/reproduction_report.md`](phases/results/1/outputs/reproduction_report.md): final report with training curves, comparison tables, discrepancy analysis
- [`canonical_spec/spec.md`](canonical_spec/spec.md): LLM's interpretation of the paper
- [`phases/architecture/1/src/model.py`](phases/architecture/1/src/model.py): generated model architecture
- [`phases/training/1/src/train.py`](phases/training/1/src/train.py): generated training loop

Note: this snapshot predates the `compute_setup` phase that newer runs include.
