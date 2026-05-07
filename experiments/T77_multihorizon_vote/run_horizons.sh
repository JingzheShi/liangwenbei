#!/bin/bash
# T77: Train h_5/10/20/40 with V4 walk-forward + Stage 5 features, seed=42
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

LOG=run_horizons.log
echo "=== T77 multi-horizon train start: $(date -u +%FT%TZ) ===" > "$LOG"

for H in 5 10 20 40; do
  echo "--- horizon=$H seed=42 ---" | tee -a "$LOG"
  python3 train_horizon.py \
    --seed 42 \
    --horizon $H \
    --num-boost-round 600 \
    --early-stopping 40 \
    --variant aug_a \
    --aug-ratio 1.0 \
    --learning-rate 0.05 \
    --min-data-in-leaf 100 \
    --bagging-freq 5 \
    --num-threads 18 \
    --use-gpu \
    --out-tag t77 \
    >> "$LOG" 2>&1
done
echo "=== T77 multi-horizon train done: $(date -u +%FT%TZ) ===" >> "$LOG"
