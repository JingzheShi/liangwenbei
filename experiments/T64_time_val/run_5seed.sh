#!/bin/bash
# Run a single strategy across 5 seeds. Sequential GPU.
set -euo pipefail
STRATEGY="${1:-V4}"
TAG="${2:-5seed}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HERE"

for SEED in 1 7 13 42 100; do
    echo "===== ${STRATEGY} seed=${SEED} ====="
    CUDA_VISIBLE_DEVICES=0 python3 "$HERE/train_v_time_val.py" \
        --strategy "$STRATEGY" --seed "$SEED" --out-tag "$TAG" --no-wandb \
        2>&1 | tee "$LOG_DIR/train_${STRATEGY}_seed${SEED}.log"
done
echo "DONE: ${STRATEGY} 5-seed"
