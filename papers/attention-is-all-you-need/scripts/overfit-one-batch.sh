#!/usr/bin/env bash
# Overfit a single batch. Loss should collapse to ~0.
# If it doesn't, the model, loss, or optimizer is wrong — not the data scale.
# Rung 3 of the verification ladder. Cheap. Run before any longer training.
set -euo pipefail

MAX_STEPS="${1:-1000}"
RUN_ID="overfit-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "${GIT_SHA}" > "${RUN_DIR}/git_sha.txt"

echo "[overfit] max_steps=${MAX_STEPS} run_dir=${RUN_DIR} git=${GIT_SHA}"

# Toolkit env (parent toolkit has torch + pytest installed). Per-paper repos
# would normally use their own .venv; we share the toolkit env for the demo.
uv run --project /Users/nickmatton/repos/research-builder \
    python -m src.train \
    --overfit-one-batch \
    --max-steps "${MAX_STEPS}" \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

# Verdict — loss should be ~0 (well under the random baseline of log(vocab_size)).
FINAL_LOSS=$(python -c "import json; print(json.load(open('${RUN_DIR}/metrics.json'))['final_loss'])")
echo "[overfit] final loss = ${FINAL_LOSS}  (should be ≪ log(100)=4.6 for vocab=100; ideally < 0.5)"
