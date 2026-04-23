# Implementation Plan — Attention Is All You Need

Updated 2026-04-23. The big-model claims (28.4 / 41.8 BLEU) are out-of-budget without multi-GPU compute; the **base-model EN-DE reproduction** is the realistic primary target.

## Phases

Each phase has explicit success criteria. Don't move to the next until the previous one's gate is green. (See `.claude/skills/verification-ladder.md`.)

### 1. Paper analysis & claims extraction ✅

- [x] `python scripts/extract-paper-text.py` produced `paper/paper.txt`.
- [x] Read paper end-to-end. CLAUDE.md filled with citation, summary, hyperparameters from §3 + Table 3, datasets from §5.1, compute budget from §5.2.
- [x] `notes/claims.yaml` populated with 6 claims from Table 2 + §5.2.

**Gate:** ✅ headline claims registered; CLAUDE.md is no longer placeholders.

### 2. Scaffolding & data pipeline ⏳ partial

`src/data.py` currently provides only **synthetic** deterministic batches — used to validate the model + training loop end-to-end without paying for the real WMT loader first. Real WMT loader pending below.

- [x] `src/data.py` synthetic batches (overfit + smoke)
- [ ] `src/data.py` — WMT 2014 EN-DE loader. Use HuggingFace `datasets` (`wmt14`) for the raw corpus.
- [ ] BPE tokenization. The paper uses ~37k shared source-target vocab. Use `subword-nmt` or `tokenizers`. Pin the version. Save the vocab to `data/wmt14_en_de/bpe.codes` so it's reproducible.
- [ ] `src/data.py` returns batches of ~25k source + 25k target tokens (paper §5.1, p.7). Group sentence pairs by approximate length.
- [ ] Unit tests (`tests/test_data.py`): vocab size matches expected, batch shapes are right, no NaN/null in input ids, BOS/EOS handling correct.
- [ ] `configs/smoke.yaml` — tiny WMT slice (10 sentence pairs) for `scripts/smoke.sh`.

**Gate:** `uv run pytest tests/test_data.py` passes. Loader produces batches matching paper specs.

### 3. Model implementation ✅ done

- [x] `src/attention.py` — scaled dot-product + multi-head (paper §3.2). Mask zeros attention verified.
- [x] `src/positional.py` — sinusoidal positional encoding (§3.5). PE(pos, 2i) formula verified at sample points.
- [x] `src/transformer.py` — encoder + decoder layers + full model. d_model + N configurable (smoke uses d=64, N=2; base config will be d=512, N=6).
- [x] Embedding sharing per §3.4.

**Gate met:** `uv run pytest tests/` → 15/15 passed. `bash scripts/overfit-one-batch.sh` → loss → 0.0000 with LS=0 (or 0.78 = LS floor with paper-faithful LS=0.1).

### 4. Training loop ⏳ partial

- [x] `src/train.py` — Adam, §5.3 warmup-then-decay LR, label-smoothing CE.
- [x] `--overfit-one-batch` mode and `--label-smoothing` override (used to verify the LS floor is the actual floor).
- [ ] Checkpointing every N steps + checkpoint averaging at eval time. Currently train.py just runs to max_steps and saves metrics.json — no model checkpoint persistence yet.
- [ ] `configs/base.yaml` — base-model config (waits on real data).
- [ ] Warmup-schedule unit tests at step 1, 100, 4000, 10000.

**Gate met (synthetic):** `bash scripts/overfit-one-batch.sh` → loss collapses. `bash scripts/smoke.sh` → loss decreases, no NaN.
**Gate pending (real data):** smoke against real WMT slice — blocked on Phase 2.

### 5. Eval

- [ ] `src/eval.py` — load checkpoint(s), do checkpoint averaging, run beam search (beam=4, length penalty α=0.6, max len = input + 50).
- [ ] BLEU computation: **first** with `multi-bleu.perl` (paper-compatible), **then** with sacrebleu. Document the gap.
- [ ] Output `runs/<run-id>/metrics.json` with keys matching `notes/claims.yaml` claim_ids.

**Gate:** `python scripts/compare-claims.py runs/<run-id>/metrics.json` returns mostly verified/close for the base-model claims (`table2_base_en_de_bleu`, `train_steps_base`).

### 6. Reproduce base-model EN-DE

- [ ] Provision a single A100 (Lambda Labs / RunPod). Estimated 3–6 GPU-hours.
- [ ] `bash scripts/reproduce.sh configs/base.yaml`.
- [ ] `/compare` after eval. Append a row to `notes/journal.md`.
- [ ] If `table2_base_en_de_bleu` lands within ±0.3 BLEU: success. If outside: `/post-mortem` and iterate.

**Gate:** `claim_id table2_base_en_de_bleu` is `verified` or `close` in the comparison report.

### 7. (Optional) Base-model EN-FR + ablations

- [ ] EN-FR base run. Same hyperparameters except dataset.
- [ ] Pick 1–2 ablations from Table 3 (e.g., row B: smaller d_k, d_v) and reproduce.

**Gate:** All non-big-model headline claims either verified, close, or have a documented post-mortem.

### 8. (Out of budget) Big-model runs

The 28.4 / 41.8 BLEU big-model claims need ~3.5 days on 8× P100 (or equivalent). Likely `not_checked` for this reproduction unless we secure multi-GPU compute. Document the gap in `notes/journal.md` and CLAUDE.md.

## Open questions (resolve before phases 3–4)

- **BPE vocab construction order**: paper says shared source-target ~37k tokens. Construct vocab on combined EN+DE corpus or separately and merge? The author code (tensor2tensor) is authoritative — read it first.
- **Token batching**: §5.1 says "approximately 25000 source tokens and 25000 target tokens." Implementation detail: bin sentence pairs by length first, then pack to fit token budget. Reference: tensor2tensor's `data_generators`.
- **Length penalty α=0.6**: which Wu et al. (2016) variant? The paper cites [38]; check whether their `lp(Y) = (5+|Y|)^α / (5+1)^α` or a simpler form.
- **Eval set**: newstest2014, but is it the raw text or detokenized? `multi-bleu.perl` expects specific tokenization. Use the tensor2tensor eval pipeline as ground truth.
