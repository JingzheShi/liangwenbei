#!/bin/bash
# T72 cascade 5-seed: Stage A (full) + Stage B (active-only)
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

LOG=run_5seed.log
echo "=== T72 cascade 5-seed start: $(date -u +%FT%TZ) ===" > "$LOG"

for STAGE in A B; do
  for SEED in 42 1 7 13 100; do
    echo "--- stage=$STAGE seed=$SEED ---" | tee -a "$LOG"
    python3 train_cascade.py \
      --stage "$STAGE" \
      --seed "$SEED" \
      --num-boost-round 600 \
      --early-stopping 40 \
      --learning-rate 0.05 \
      --min-data-in-leaf 100 \
      --bagging-freq 5 \
      --num-threads 18 \
      --use-gpu \
      --aug-ratio 1.0 \
      >> "$LOG" 2>&1
  done
done
echo "=== T72 cascade 5-seed done: $(date -u +%FT%TZ) ===" >> "$LOG"
