#!/usr/bin/env bash
# Smoke run: fresh synthetic batches per step, end-to-end pipeline check.
# Does NOT prove the model is right — only that nothing is broken.
# Rung 4 of the verification ladder. Run before any longer training.
set -euo pipefail

MAX_STEPS="${1:-200}"
RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "${GIT_SHA}" > "${RUN_DIR}/git_sha.txt"

echo "[smoke] max_steps=${MAX_STEPS} run_dir=${RUN_DIR} git=${GIT_SHA}"

uv run --project /Users/nickmatton/repos/research-builder \
    python -m src.train \
    --max-steps "${MAX_STEPS}" \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "[smoke] done. Logs at ${RUN_DIR}/train.log"
echo "[smoke] Note: loss won't go to 0 — fresh random batches each step. We're"
echo "[smoke] verifying the pipeline executes, not that the model converges."
