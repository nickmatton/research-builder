# Example Run: Test Paper

This directory contains the complete output of a successful end-to-end run of the research-builder pipeline against a simple 3-page test paper (`tests/fixtures/test_paper.pdf`).

## Paper

The test paper describes a study of "widget performance" — a minimal paper with:
- A 3-layer neural network trained with AdamW (lr=0.001, batch_size=32)
- 100 epochs of training
- Reported accuracy: 95.2%

## Results

The pipeline reproduced the paper's results, achieving **95.5% test accuracy** (within 0.3% of the reported 95.2%).

**Run stats:** ~25 minutes, ~$1.15 in Claude tokens, 5 phases executed sequentially.

## Directory Structure

```
canonical_spec/
├── spec.md              # LLM-authored canonical interpretation of the paper
├── state.yaml           # Machine-readable phase state and dependency graph
└── revision_log.yaml    # Append-only event log

phases/
├── data/1/
│   ├── src/             # Data generation code + tests (10/10 passing)
│   └── outputs/         # Synthetic dataset (train + test splits)
├── architecture/1/
│   ├── src/             # MLP model code + tests
│   └── outputs/         # Model class file
├── training/1/
│   ├── src/             # Training loop + tests
│   └── outputs/         # Training log (100 epochs)
├── eval/1/
│   ├── src/             # Evaluation script + tests
│   └── outputs/         # Metrics JSON (95.5% accuracy)
└── results/1/
    ├── src/             # Report generation + tests
    └── outputs/         # Reproduction report with training curves
```

## Key Files

- **[Reproduction Report](phases/results/1/outputs/reproduction_report.md)** — The final output with training curves, comparison tables, and discrepancy analysis
- **[Canonical Spec](canonical_spec/spec.md)** — How the LLM interpreted the paper
- **[Model Code](phases/architecture/1/src/model.py)** — The generated model architecture
- **[Training Script](phases/training/1/src/train.py)** — The generated training loop
