#!/usr/bin/env bash
# Overfit a single batch. Loss should collapse to ~0.
# If it doesn't, the model, loss, or optimizer is wrong — not the data scale.
# Rung 3 of the verification ladder. Cheap. Run before any longer training.
set -euo pipefail

CONFIG="${1:-configs/smoke.yaml}"
RUN_ID="overfit-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[overfit] config=${CONFIG} run_dir=${RUN_DIR}"

# TODO: replace with your real entry point. The point is to repeatedly train
# on a single batch — your training script needs to support that mode.
uv run python -m src.train \
    --config "${CONFIG}" \
    --overfit-one-batch \
    --max-steps 500 \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

# Sanity-check the final loss. Adjust threshold per task.
LOSS=$(grep -E "^step 500" "${RUN_DIR}/train.log" | awk '{print $NF}' || echo "?")
echo "[overfit] final loss = ${LOSS}  (should be ~0)"
