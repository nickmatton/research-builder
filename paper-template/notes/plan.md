# Implementation Plan

Filled in once, after the first plan-mode session in this repo. Updated as you discover the paper has more (or fewer) phases than you thought.

## Phases

Each phase has explicit success criteria. Don't move to the next phase until the previous one's gate is green. (See `.claude/skills/verification-ladder.md`.)

### 1. Paper analysis & claims extraction
- [ ] Read paper end-to-end. Update `CLAUDE.md` Summary + Hyperparameters.
- [ ] Populate `notes/claims.yaml` with 3–10 headline claims.
- [ ] Look up any opaque cited methods via the `arxiv` MCP server.

**Gate:** `claims.list_claims()` returns the headline list. CLAUDE.md is no longer mostly placeholders.

### 2. Scaffolding & data pipeline
- [ ] Project skeleton (`src/data.py`, `src/model.py`, `src/train.py`, `src/eval.py`).
- [ ] Data download / preprocessing scripts (`scripts/fetch-*.sh`).
- [ ] Unit tests for shapes, loader output, label distribution.

**Gate:** `uv run pytest` passes.

### 3. Model implementation
- [ ] Architecture matches paper §<X>.
- [ ] Forward pass on dummy input → expected output shapes.
- [ ] Parameter count matches paper (if reported).
- [ ] Gradients flow through all layers.

**Gate:** `uv run pytest tests/test_model.py` passes.

### 4. Training loop
- [ ] Optimizer / scheduler / loss exactly as spec.
- [ ] Checkpointing.
- [ ] Loss decreases over first 100 steps.
- [ ] No NaN/Inf in gradients or loss.

**Gate:** `bash scripts/overfit-one-batch.sh` collapses loss to ~0.

### 5. Reproduce headline number
- [ ] `bash scripts/smoke.sh` runs end-to-end.
- [ ] Short training run loss curve roughly matches paper Figure <Y>.
- [ ] Full reproduction run.
- [ ] `claims.verify_run(...)` returns mostly verified/close.

**Gate:** Top headline claim is verified or close.

### 6. Ablations & secondary claims
- [ ] One ablation per claim that isn't covered by the headline run.

**Gate:** All headline claims either verified, close, or have a documented post-mortem explaining why not.

## Open questions

<List of things you don't know yet. Move resolved items into CLAUDE.md's hyperparameters / architecture sections; delete from this list when settled.>

- ...
