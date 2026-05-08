#!/usr/bin/env bash
# Full reproduction run. Only run after every cheaper rung passes:
#   1. uv run pytest                      # unit tests
#   2. bash scripts/overfit-one-batch.sh  # overfit
#   3. bash scripts/smoke.sh              # smoke
#   4. (optional) short training run
# Rung 6 of the verification ladder.
set -euo pipefail

CONFIG="${1:-configs/full.yaml}"
RUN_ID="full-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# Pin the config + git SHA next to the checkpoint for reproducibility.
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
cp "${CONFIG}" "${RUN_DIR}/config.yaml"
echo "${GIT_SHA}" > "${RUN_DIR}/git_sha.txt"

echo "[reproduce] config=${CONFIG} run_dir=${RUN_DIR} git=${GIT_SHA}"

# TODO: replace with your real entry point.
uv run python -m src.train \
    --config "${CONFIG}" \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

# Eval after training. Should write metrics.json into RUN_DIR.
uv run python -m src.eval \
    --checkpoint "${RUN_DIR}/checkpoint.pt" \
    --output "${RUN_DIR}/metrics.json" \
    2>&1 | tee "${RUN_DIR}/eval.log"

echo "[reproduce] done. Now run /compare to verify against the claims ledger."
