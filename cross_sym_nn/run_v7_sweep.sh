#!/bin/bash
# V7 5-seed sweep — V6_PairwiseInteract + selected interaction features
# Identical training protocol as v6_pairwise SOTA.

set -e
cd "$(dirname "$0")"
mkdir -p logs_v7

for seed in 0 1 2 3 4; do
    LOG="logs_v7/v7_pairwise_inter_s${seed}.log"
    echo "=== seed=$seed → $LOG ==="
    CUDA_VISIBLE_DEVICES=0 python3 train_v7.py \
        --arch v6_pairwise --seed $seed \
        --keep-idx keep_idx_259d.npy \
        --sel-idx  selected_features.npy \
        --zstats   inter_zstats.npz \
        --wandb-project liangwenbei-cross-sym \
        > "$LOG" 2>&1
    tail -3 "$LOG"
done
echo "=== ALL SEEDS DONE ==="
