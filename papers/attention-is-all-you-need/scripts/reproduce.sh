#!/usr/bin/env bash
# Full reproduction pipeline for Attention Is All You Need (base model, WMT EN-DE).
# Only run after every cheaper rung passes:
#   1. uv run pytest                      # unit tests (32 of them)
#   2. bash scripts/overfit-one-batch.sh  # loss → 0
#   3. bash scripts/smoke.sh wmt 500      # real WMT, end-to-end check
# Estimated cost on a single A100: 3–6 GPU-hours, $4–8 with bin/lambda.
#
# Rung 6 of the verification ladder. The killer demo.
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
MAX_STEPS="${MAX_STEPS:-100000}"
RUN_ID="full-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" data/wmt14_en_de

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
cp "${CONFIG}" "${RUN_DIR}/config.yaml"
echo "${GIT_SHA}" > "${RUN_DIR}/git_sha.txt"

echo "[reproduce] config=${CONFIG} run_dir=${RUN_DIR} git=${GIT_SHA} max_steps=${MAX_STEPS}"

# 1. Train tokenizer (if not cached). Paper §5.1: shared 37k vocab.
TOKENIZER="data/wmt14_en_de/tokenizer.json"
if [ ! -f "${TOKENIZER}" ]; then
    echo "[reproduce] training BPE tokenizer (37k vocab, full WMT)..."
    uv run python scripts/train-tokenizer.py --vocab-size 37000
else
    echo "[reproduce] reusing cached tokenizer ${TOKENIZER}"
fi

# 2. Train base model. Paper §3 + Table 3: d_model=512, h=8, N=6, d_ff=2048.
echo "[reproduce] training base model (${MAX_STEPS} steps, paper config)..."
uv run python -m src.train \
    --data wmt \
    --tokenizer "${TOKENIZER}" \
    --max-steps "${MAX_STEPS}" \
    --d-model 512 \
    --num-heads 8 \
    --num-layers 6 \
    --d-ff 2048 \
    --warmup 4000 \
    --label-smoothing 0.1 \
    --token-budget 25000 \
    --save-checkpoint \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

# 3. Eval — beam search on newstest2014 (HF "test" split). Paper §6.1.
echo "[reproduce] evaluating with beam=4, α=0.6..."
uv run python -m src.eval \
    --checkpoint "${RUN_DIR}/checkpoint.pt" \
    --tokenizer "${TOKENIZER}" \
    --split test \
    --beam-size 4 \
    --length-penalty 0.6 \
    --output "${RUN_DIR}/metrics.json" \
    2>&1 | tee "${RUN_DIR}/eval.log"

# 4. Compare against claims.yaml.
echo "[reproduce] comparing to claims..."
uv run python scripts/compare-claims.py "${RUN_DIR}/metrics.json" \
    | tee "${RUN_DIR}/claims-report.md"

echo
echo "[reproduce] done. Append a row to notes/journal.md with run_id=${RUN_ID}"
echo "[reproduce] then teardown the GPU: bin/lambda teardown <id>"
