#!/bin/bash
# T69 5-seed run: V4 walk-forward val on full 0-95 data
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

LOG=run_5seed.log
echo "=== T69 5-seed start: $(date -u +%FT%TZ) ===" > "$LOG"

for SEED in 42 1 7 13 100; do
  echo "--- seed=$SEED ---" | tee -a "$LOG"
  python3 train_t69_full.py \
    --seed "$SEED" \
    --num-boost-round 600 \
    --early-stopping 40 \
    --variant aug_a \
    --aug-ratio 1.0 \
    --learning-rate 0.05 \
    --min-data-in-leaf 100 \
    --bagging-freq 5 \
    --num-threads 18 \
    --use-gpu \
    --n-drop-tail 3 \
    --out-tag full95 \
    >> "$LOG" 2>&1
done
echo "=== T69 5-seed done: $(date -u +%FT%TZ) ===" >> "$LOG"
