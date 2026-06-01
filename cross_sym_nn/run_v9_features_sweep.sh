#!/bin/bash
# V9 features 5-seed sweep — v6_pairwise + 60 features (40 v7 + 20 NF) + D no_aug
# Run seeds 1-4 sequentially (seed 0 already launched separately).

set -e
cd "$(dirname "$0")"
mkdir -p logs_v9 runs_v9

START_SEED=${1:-1}
END_SEED=${2:-4}

for seed in $(seq $START_SEED $END_SEED); do
    LOG="logs_v9/v9_features60_s${seed}.log"
    echo "=== seed=$seed → $LOG ==="
    CUDA_VISIBLE_DEVICES=0 python3 -u train_v9_features.py \
        --arch v6_pairwise --seed $seed \
        --keep-idx keep_idx_259d.npy \
        --sel-idx selected_features_v9_60.npy \
        --zstats inter_zstats_v9.npz \
        --out-root runs_v9 --out-tag v9_features60 \
        > "$LOG" 2>&1
    tail -3 "$LOG"
done
echo "=== SEEDS $START_SEED-$END_SEED DONE ==="
