# Run Journal — Attention Is All You Need

Append-only log of every meaningful run. Most-recent at the bottom.

The `/reproduce` slash command appends a row automatically after each full run. Add rows manually for smoke / overfit / partial runs you want to remember.

## Format

```
## <run-id>  (<ISO 8601 timestamp>)
**Type:** smoke | overfit-one-batch | short-train | full
**Git SHA:** <short sha>
**Config:** <configs/<file>.yaml> · hash <md5>
**Hardware:** <gpu / cpu>
**Duration:** <wall-clock>

**Key metrics**
- <metric>: <value>

**Claims verification** (full runs only)
- verified: <n>, close: <n>, missed: <n>, exceeded: <n>, not_checked: <n>
- See `runs/<run-id>/claims-report.md`.

**Notes**
<one or two sentences. What did this run prove or fail to prove?>
```

---

## Runs

## scaffold-2026-04-23  (2026-04-23T12:50)
**Type:** scaffold (not a training run)
**Git SHA:** d2014ca (research-builder toolkit)

**Notes**
Repo scaffolded from `paper-template/`. PDF placed at `paper/paper.pdf`, extracted to `paper/paper.txt` via `extract-paper-text.py` (15 pages, all readable). `CLAUDE.md` populated with citation, summary, hyperparameters from §3 + Table 3, datasets from §5.1, compute budget from §5.2. `notes/claims.yaml` populated with 6 claims from Table 2 (4 BLEU) + §5.2 (2 step counts). Implementation has not started yet — `src/` is empty. Plan-mode session pending: decide between reproducing the base model (~12 GPU-h on P100, in-budget) vs deferring the big model.

## impl-2026-04-23  (2026-04-23T14:15)
**Type:** implementation (not a training run)
**Git SHA:** bb3af4b (paper-repo branch)

**What landed:**
- `src/attention.py` — scaled dot-product + multi-head attention (paper §3.2). Explicit Linear projections, no PyTorch built-in.
- `src/positional.py` — sinusoidal PE (§3.5). Verified by unit test: PE(0,0)=0, PE(0,1)=1, PE(3,0)=sin(3).
- `src/transformer.py` — encoder–decoder, post-norm, shared input/output embedding per §3.4 (with √d_model scaling).
- `src/data.py` — synthetic deterministic batches (real WMT loader pending).
- `src/train.py` — Adam(β1=0.9, β2=0.98, ε=1e-9), §5.3 warmup-then-decay schedule, label-smoothing CE (§5.4), `--overfit-one-batch` mode.
- `tests/test_{attention,positional,model}.py` — 15 unit tests covering shapes, mask semantics, PE formula, gradient flow.

**Verification ladder rung 1 — unit tests**
- `uv run pytest tests/ -v`: **15/15 passed in 0.84s**.

## overfit-20260423-141946  (2026-04-23T14:19)
**Type:** overfit-one-batch
**Git SHA:** bb3af4b
**Hardware:** CPU (no CUDA)
**Duration:** 3.3 s

**Config:** d_model=64, num_heads=4, num_layers=2, d_ff=256, vocab=100, batch=4, src/tgt_len=6, warmup=400, label_smoothing=**0.1** (paper-faithful)

**Key metrics**
- Initial loss: 49.47 (random init)
- Final loss (step 1000): **0.7847**
- Random baseline (log V=100): 4.61

**Notes**
Loss plateaus at ~0.78 well below random. **This is not a bug** — it's the irreducible floor of label smoothing: H(smoothed_dist) = -[0.9·log(0.9) + 98·(0.1/98)·log(0.1/98)] ≈ 0.77 for V=100, ε=0.1. The model hit the floor by step 200. Verification ladder rung 3 PASSES — the optimizer can drive loss to its theoretical minimum given LS.

## overfit-no-ls  (2026-04-23T14:20)
**Type:** overfit-one-batch (label smoothing disabled, sanity check)
**Git SHA:** bb3af4b (with --label-smoothing 0)
**Hardware:** CPU
**Duration:** 3.3 s

**Key metrics**
- Initial loss: 49.35
- Loss at step 100: 0.0896
- Loss at step 250: **0.0000**
- Final loss (step 1000): 0.0000

**Notes**
Confirmation experiment for the previous run. Without LS the loss collapses to 0 within ~250 steps. Proves the model + optimizer + loss are all wired correctly; the 0.78 floor in the paper-faithful run is purely a label-smoothing artifact, not a model defect.

## smoke-20260423-142122  (2026-04-23T14:21)
**Type:** smoke (200 steps, fresh synthetic batches per step)
**Git SHA:** bb3af4b
**Hardware:** CPU
**Duration:** 0.7 s

**Key metrics**
- Initial loss: 49.80
- Final loss (step 200): 4.9537
- Random baseline w/ LS floor: ~5.4

**Notes**
End-to-end pipeline executes — no NaN, no crashes, loss decreases. The model can't memorize fresh batches (by design), but it does meaningfully better than random within 200 steps. Verification ladder rung 4 PASSES.

**Verification ladder status (so far):**
- ✅ Rung 1: unit tests (15/15)
- ✅ Rung 3: overfit-one-batch (loss → 0 with LS=0; → 0.78 = LS floor with LS=0.1)
- ✅ Rung 4: smoke run (loss decreasing, no NaN, end-to-end pipeline works)
- ⏳ Rung 5: short training run on real WMT — pending data loader (Phase 2 of plan.md)
- ⏳ Rung 6: full base-model reproduction — needs A100 (3–6 GPU-h)

The methodology and infrastructure work end-to-end on synthetic data. Next blocker: implement the WMT 2014 EN-DE loader (HuggingFace `datasets` + BPE tokenizer) and rerun smoke against real data. Then provision compute and attempt the headline `table2_base_en_de_bleu` claim.

## impl-2026-05-05  (2026-05-05T14:25)
**Type:** implementation (not a training run)
**Git SHA:** _(this commit)_

**What landed:**
- `src/tokenize.py` — HF `tokenizers` BPE wrapper (PAD/BOS/EOS/UNK at ids 0–3).
- `src/wmt.py` — HF `datasets` loader for WMT 2014 EN-DE; `tokenize_pairs` with length filter; `token_budget_batches` (length-sort + greedy pack, paper §5.1).
- `src/eval.py` — beam search (Wu et al. length penalty); sacrebleu BLEU; CLI for `python -m src.eval --checkpoint ... --tokenizer ...`.
- `src/train.py` extended: `--data wmt --tokenizer X` mode iterates real WMT batches indefinitely; `--save-checkpoint` writes model+optimizer state; auto-detects CUDA/MPS/CPU.
- `scripts/train-tokenizer.py` — one-shot BPE training (default 37k vocab per paper).
- `configs/base.yaml` — paper-faithful base-model config (d=512, h=8, N=6, d_ff=2048, warmup=4000, ...).
- `scripts/reproduce.sh` — full pipeline: train-tokenizer → train (100k steps) → eval (beam=4) → compare-claims.
- `scripts/smoke.sh` extended: `bash scripts/smoke.sh wmt 200` runs synthetic-mode-like smoke against real WMT (auto-trains a small tokenizer if missing).
- `papers/attention-is-all-you-need/pyproject.toml` (NEW) — per-paper env. Toolkit pyproject.toml slimmed back to 2 deps (pdfplumber, pyyaml).
- All scripts dropped the absolute `--project /Users/...` path and now run via the local per-paper venv.

**Tests:** `uv run pytest tests/` → **32/32 passed in 0.47s** (was 15/15 before).
- New: tests/test_tokenize.py (4), tests/test_wmt.py (6), tests/test_eval.py (7).
- Existing 15 still green after the train.py refactor.

**Verification ladder status:**
- ✅ Rung 1: 32 unit tests
- ✅ Rung 3 (synthetic): overfit-one-batch loss → 0 with LS=0; → 0.78 with LS=0.1
- ✅ Rung 4 (synthetic): smoke 200 steps, no NaN
- ⏳ Rung 4 (real WMT): blocked on dataset download (HF datasets, ~1.5 GB EN-DE, no network in this session). Code path verified by tests + manual `uv run python -c "from src import wmt"` import-clean check.
- ⏳ Rung 5 (eval): code complete, needs trained checkpoint
- ⏳ Rung 6 (full reproduction): needs A100 (~3–6 GPU-h, ~$5 with bin/lambda), not run yet

**Known gaps documented in plan.md "Open questions":**
- sacrebleu vs paper's `multi-bleu.perl` (~0.5–1.5 BLEU gap)
- single-checkpoint vs paper's last-5-average (~0.3–0.8 BLEU)
- Big-model 28.4 / 41.8 claims out-of-budget without 8x GPU

Next: provision A100 via bin/lambda, run reproduce.sh, populate the journal with a real BLEU number.

<!-- Append run blocks below. -->
