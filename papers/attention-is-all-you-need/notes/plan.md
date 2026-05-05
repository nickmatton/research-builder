# Implementation Plan — Attention Is All You Need

Updated 2026-05-05. The big-model claims (28.4 / 41.8 BLEU) are out-of-budget without multi-GPU compute; the **base-model EN-DE reproduction** is the realistic primary target. Phases 1–5 done; only the GPU run (Phase 6) remains.

## Phases

Each phase has explicit success criteria. Don't move to the next until the previous one's gate is green. (See `.claude/skills/verification-ladder.md`.)

### 1. Paper analysis & claims extraction ✅

- [x] `python scripts/extract-paper-text.py` produced `paper/paper.txt`.
- [x] Read paper end-to-end. CLAUDE.md filled with citation, summary, hyperparameters from §3 + Table 3, datasets from §5.1, compute budget from §5.2.
- [x] `notes/claims.yaml` populated with 6 claims from Table 2 + §5.2.

**Gate:** ✅ headline claims registered; CLAUDE.md is no longer placeholders.

### 2. Scaffolding & data pipeline ✅ done

- [x] `src/data.py` synthetic batches (overfit + smoke)
- [x] `src/wmt.py` — WMT 2014 EN-DE loader via HuggingFace `datasets` (`wmt14`).
- [x] `src/tokenize.py` — shared BPE via HF `tokenizers`. `scripts/train-tokenizer.py` produces `data/wmt14_en_de/tokenizer.json` (default 37k vocab per paper §5.1).
- [x] `src/wmt.py:token_budget_batches` returns batches whose padded token count ≤ 25k (paper §5.1). Length-sorted then greedy-packed.
- [x] `tests/test_wmt.py` (6 tests): drops too-long pairs, direction swap, batch shapes/dtypes, budget respected, no pairs lost.
- [x] `tests/test_tokenize.py` (4 tests): BPE training, encode/decode round-trip, BOS/EOS handling, special-skip on decode.

**Gate met:** `uv run pytest tests/test_{wmt,tokenize}.py` → 10/10 passed. Real WMT iteration end-to-end pending Phase 6 (needs network + CPU/GPU minutes for the dataset download).

### 3. Model implementation ✅ done

- [x] `src/attention.py` — scaled dot-product + multi-head (paper §3.2). Mask zeros attention verified.
- [x] `src/positional.py` — sinusoidal positional encoding (§3.5). PE(pos, 2i) formula verified at sample points.
- [x] `src/transformer.py` — encoder + decoder layers + full model. d_model + N configurable (smoke uses d=64, N=2; base config will be d=512, N=6).
- [x] Embedding sharing per §3.4.

**Gate met:** `uv run pytest tests/` → 15/15 passed. `bash scripts/overfit-one-batch.sh` → loss → 0.0000 with LS=0 (or 0.78 = LS floor with paper-faithful LS=0.1).

### 4. Training loop ✅ done (modulo checkpoint averaging)

- [x] `src/train.py` — Adam, §5.3 warmup-then-decay LR, label-smoothing CE.
- [x] `--overfit-one-batch` mode and `--label-smoothing` override (used to verify the LS floor is the actual floor).
- [x] `--data wmt --tokenizer ...` mode iterates `src/wmt.token_budget_batches` indefinitely.
- [x] `--save-checkpoint` writes `runs/<id>/checkpoint.pt` with model + optimizer state + config.
- [x] Auto-detects CUDA / MPS / CPU device.
- [x] `configs/base.yaml` documents the canonical paper config.
- [ ] Checkpoint averaging at eval time (paper averages last 5 base / 20 big). Skipped for v1 — single final checkpoint used.
- [ ] Warmup-schedule unit tests at step 1, 100, 4000, 10000. Skipped — formula is one line, manually verified at step 1 (lr matches).

**Gate met:** synthetic overfit loss → 0 with LS=0; → 0.78 (LS floor) with LS=0.1.

### 5. Eval ✅ done (sacrebleu only; multi-bleu.perl gap documented)

- [x] `src/eval.py` — beam search (beam=4 default), Wu et al. length penalty (α=0.6), max output length = input + 50.
- [x] `python -m src.eval --checkpoint X --tokenizer Y` CLI loads checkpoint, decodes WMT test set, writes metrics.json with `table2_base_en_de_bleu` claim_id.
- [x] sacrebleu BLEU computation (modern standard).
- [x] `tests/test_eval.py` (7 tests): length penalty math, beam search structural correctness, sacrebleu plumbing.
- [ ] `multi-bleu.perl` parity (paper-compatible BLEU). **Gap documented in CLAUDE.md "Open questions":** sacrebleu BLEU differs from paper BLEU by typically 0.5–1.5 points due to tokenization. Future work to add the perl-compatible variant for true paper-faithful comparison; for now, sacrebleu is the comparison metric and we accept the gap explicitly in any reproduction report.

**Gate:** code complete + tests green. Final-number gate is part of Phase 6.

### 6. Reproduce base-model EN-DE ⏳ pending GPU

Code is done; this is execution. The full pipeline lives in `scripts/reproduce.sh`.

- [ ] `export LAMBDA_API_KEY=... LAMBDA_BUDGET_USD=20`
- [ ] `bin/lambda provision gpu_1x_a100 --max-hours 8 --work-dir papers/attention-is-all-you-need` (~$10 budget commitment, real ≈ $4–8)
- [ ] Inside the work dir on the remote (or locally with rsync via `remote_run.sh`): `bash scripts/reproduce.sh configs/base.yaml`. Pipeline: train-tokenizer → train (100k steps) → eval (beam=4) → compare-claims.
- [ ] Append journal row with run_id, BLEU, deltas vs `table2_base_en_de_bleu = 27.3`, `table2_base_en_fr_bleu = 38.1`.
- [ ] If `verified` or `close`: success. If `missed`: `/post-mortem`. Note that sacrebleu BLEU is expected ~0.5–1.5 lower than paper BLEU — interpret accordingly.
- [ ] `bin/lambda teardown <id>` (auto-teardown also fires at the deadline)

**Gate:** `claim_id table2_base_en_de_bleu` is `verified` or `close` (within tolerance + LS-floor margin) in the comparison report.

### 7. (Optional) Base-model EN-FR + ablations

- [ ] EN-FR base run. Same hyperparameters except dataset.
- [ ] Pick 1–2 ablations from Table 3 (e.g., row B: smaller d_k, d_v) and reproduce.

**Gate:** All non-big-model headline claims either verified, close, or have a documented post-mortem.

### 8. (Out of budget) Big-model runs

The 28.4 / 41.8 BLEU big-model claims need ~3.5 days on 8× P100 (or equivalent). Likely `not_checked` for this reproduction unless we secure multi-GPU compute. Document the gap in `notes/journal.md` and CLAUDE.md.

## Open questions

Resolved during implementation:

- ~~**BPE vocab construction order**~~: implemented as shared EN+DE BPE training on interleaved corpus (`src/wmt.train_tokenizer_from_pairs` → yields `en` then `de` per pair). Matches "shared source-target vocabulary" per §5.1.
- ~~**Token batching**~~: implemented as length-sort then greedy-pack (`src/wmt.token_budget_batches`). Tested: budget always respected, no pairs lost.
- ~~**Length penalty α=0.6**~~: implemented as `lp(Y) = (5+|Y|)^α / (5+1)^α` (Wu et al. 2016 form). Tested: longer beam at equal log-prob scores higher, α=0 → no penalty.

Remaining open:

- **Eval tokenization**: sacrebleu uses its own tokenizer (default `13a`). Paper used `multi-bleu.perl` which is tokenization-sensitive. Expect 0.5–1.5 BLEU gap vs paper. Future: add a `--bleu-impl multi-bleu` option. For now, accept the gap and document in journal.
- **Checkpoint averaging**: paper averages last 5 (base) / 20 (big) checkpoints written every 10 min. We use single final checkpoint. Likely costs 0.3–0.8 BLEU. Future work.
- **Big-model claims (28.4 / 41.8 BLEU)**: need ~3.5 days on 8× P100. Out of budget without multi-GPU compute. Will likely land as `not_checked` in the comparison report.
