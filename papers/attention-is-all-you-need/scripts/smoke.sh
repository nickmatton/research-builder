#!/usr/bin/env bash
# Smoke run: end-to-end pipeline check. Two flavors:
#   bash scripts/smoke.sh                 # synthetic data, 200 steps, ~1s
#   bash scripts/smoke.sh wmt 200         # real WMT (1000 pair limit), tokenizer reused
# Verifies the pipeline executes — does NOT prove the model converges.
set -euo pipefail

MODE="${1:-synthetic}"
MAX_STEPS="${2:-200}"
RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "${GIT_SHA}" > "${RUN_DIR}/git_sha.txt"

echo "[smoke] mode=${MODE} max_steps=${MAX_STEPS} run_dir=${RUN_DIR} git=${GIT_SHA}"

if [ "${MODE}" = "wmt" ]; then
    TOKENIZER="data/wmt14_en_de/tokenizer.json"
    if [ ! -f "${TOKENIZER}" ]; then
        echo "[smoke] no tokenizer at ${TOKENIZER}; training one (limit 5000 pairs)..."
        uv run python scripts/train-tokenizer.py --vocab-size 8000 --limit 5000
    fi
    uv run python -m src.train \
        --data wmt \
        --tokenizer "${TOKENIZER}" \
        --wmt-limit 1000 \
        --max-steps "${MAX_STEPS}" \
        --d-model 128 --num-heads 4 --num-layers 2 --d-ff 256 \
        --token-budget 4000 \
        --output-dir "${RUN_DIR}" \
        2>&1 | tee "${RUN_DIR}/train.log"
else
    uv run python -m src.train \
        --max-steps "${MAX_STEPS}" \
        --output-dir "${RUN_DIR}" \
        2>&1 | tee "${RUN_DIR}/train.log"
fi

echo "[smoke] done. Logs at ${RUN_DIR}/train.log"
