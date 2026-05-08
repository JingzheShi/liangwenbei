#!/bin/bash
set -e
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE/../.."

# baseline seed42 already done; do 1, 7
for SEED in 1 7; do
  echo "=== baseline seed=$SEED ==="
  CUDA_VISIBLE_DEVICES=0 python3 experiments/T122_retest_wins_on_T75/train_t122.py \
    --seed $SEED --variant baseline --no-wandb 2>&1 \
    | tee experiments/T122_retest_wins_on_T75/train_baseline_seed${SEED}.log | tail -5
done

for VARIANT in hyd lag monotone all3; do
  for SEED in 42 1 7; do
    echo "=== $VARIANT seed=$SEED ==="
    CUDA_VISIBLE_DEVICES=0 python3 experiments/T122_retest_wins_on_T75/train_t122.py \
      --seed $SEED --variant $VARIANT --no-wandb 2>&1 \
      | tee experiments/T122_retest_wins_on_T75/train_${VARIANT}_seed${SEED}.log | tail -5
  done
done
echo "ALL DONE"
