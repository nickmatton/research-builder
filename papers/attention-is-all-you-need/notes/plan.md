# Implementation Plan — Attention Is All You Need

Updated 2026-04-23. The big-model claims (28.4 / 41.8 BLEU) are out-of-budget without multi-GPU compute; the **base-model EN-DE reproduction** is the realistic primary target.

## Phases

Each phase has explicit success criteria. Don't move to the next until the previous one's gate is green. (See `.claude/skills/verification-ladder.md`.)

### 1. Paper analysis & claims extraction ✅

- [x] `python scripts/extract-paper-text.py` produced `paper/paper.txt`.
- [x] Read paper end-to-end. CLAUDE.md filled with citation, summary, hyperparameters from §3 + Table 3, datasets from §5.1, compute budget from §5.2.
- [x] `notes/claims.yaml` populated with 6 claims from Table 2 + §5.2.

**Gate:** ✅ headline claims registered; CLAUDE.md is no longer placeholders.

### 2. Scaffolding & data pipeline

- [ ] `src/data.py` — WMT 2014 EN-DE loader. Use HuggingFace `datasets` (`wmt14`) for the raw corpus.
- [ ] BPE tokenization. The paper uses ~37k shared source-target vocab. Use `subword-nmt` or `tokenizers`. Pin the version. Save the vocab to `data/wmt14_en_de/bpe.codes` so it's reproducible.
- [ ] `src/data.py` returns batches of ~25k source + 25k target tokens (paper §5.1, p.7). Group sentence pairs by approximate length.
- [ ] Unit tests (`tests/test_data.py`): vocab size matches expected, batch shapes are right, no NaN/null in input ids, BOS/EOS handling correct.
- [ ] `configs/smoke.yaml` — tiny WMT slice (10 sentence pairs) for `scripts/smoke.sh`.

**Gate:** `uv run pytest tests/test_data.py` passes. Loader produces batches matching paper specs.

### 3. Model implementation

Reference: `tensor2tensor/models/transformer.py` (author code), Annotated Transformer (Sasha Rush). Read both before implementing.

- [ ] `src/attention.py` — scaled dot-product + multi-head (paper §3.2, p.4–5). Tests: shape on (B, L, d), mask zeros out future positions, output is convex combo of values.
- [ ] `src/positional.py` — sinusoidal positional encoding (§3.5, p.6). Tests: PE(pos, 2i) matches the paper's formula at sample positions, encoding is deterministic.
- [ ] `src/transformer.py` — encoder layer (self-attn + FFN), decoder layer (masked self-attn + cross-attn + FFN), full encoder–decoder. d_model=512, d_ff=2048, h=8, N=6 (base config). Tests: forward on dummy (B, L, d) inputs gives expected shapes; param count within 5% of paper-reported (paper doesn't list exact param count for base, but typical impls ≈ 65M).
- [ ] Embedding sharing per §3.4: input embeddings, output embeddings, and pre-softmax linear share weights (multiplied by √d_model in embeddings).

**Gate:** `uv run pytest tests/test_model.py` passes. `bash scripts/overfit-one-batch.sh` collapses loss to ~0.

### 4. Training loop

- [ ] `src/train.py` — Adam (β1=0.9, β2=0.98, ε=1e-9), the §5.3 warmup-then-decay LR schedule with `warmup_steps=4000`, label smoothing ε_ls=0.1.
- [ ] Loss: cross-entropy with label smoothing.
- [ ] Checkpoints written every 10 minutes (§6.1, p.8). At eval time, average the last 5 (base) / last 20 (big) checkpoints.
- [ ] `configs/base.yaml` — base-model full config: d_model=512, d_ff=2048, h=8, N=6, P_drop=0.1, ε_ls=0.1, batch ~25k+25k tokens, 100,000 steps.
- [ ] Tests: warmup schedule matches paper formula at step 1, 100, 4000, 10000; loss is finite for first 10 steps; checkpoint round-trips.

**Gate:** `bash scripts/overfit-one-batch.sh` → loss ~0. `bash scripts/smoke.sh` (100 steps, tiny WMT slice) → loss decreases, no NaN.

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
