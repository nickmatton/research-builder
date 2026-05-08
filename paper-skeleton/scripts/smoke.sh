#!/usr/bin/env bash
# Smoke run: ~100 steps on a tiny dataset slice. End-to-end pipeline check.
# Does NOT prove the model is right — only that nothing is broken.
# Rung 4 of the verification ladder. Run before full training, every time.
set -euo pipefail

CONFIG="${1:-configs/smoke.yaml}"
RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[smoke] config=${CONFIG} run_dir=${RUN_DIR}"

# TODO: replace with your real entry point.
uv run python -m src.train \
    --config "${CONFIG}" \
    --max-steps 100 \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "[smoke] done. Logs at ${RUN_DIR}/train.log"
