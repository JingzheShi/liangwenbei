#!/bin/bash
# T71 5-seed: V4 walk-forward val + Stage 5 features (schemeP cache), horizon=40
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

LOG=run_5seed.log
echo "=== T71 5-seed h40 start: $(date -u +%FT%TZ) ===" > "$LOG"

for SEED in 42 1 7 13 100; do
  echo "--- seed=$SEED ---" | tee -a "$LOG"
  python3 train_h40_v4_stage5.py \
    --seed "$SEED" \
    --horizon 40 \
    --num-boost-round 600 \
    --early-stopping 40 \
    --variant aug_a \
    --aug-ratio 1.0 \
    --learning-rate 0.05 \
    --min-data-in-leaf 100 \
    --bagging-freq 5 \
    --num-threads 18 \
    --use-gpu \
    --out-tag t71_h40 \
    >> "$LOG" 2>&1
done
echo "=== T71 5-seed h40 done: $(date -u +%FT%TZ) ===" >> "$LOG"
