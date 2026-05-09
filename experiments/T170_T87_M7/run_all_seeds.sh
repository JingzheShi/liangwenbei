#!/bin/bash
# T170: Train all 5 seeds sequentially with M7 trick
set -e
WORKDIR="$(dirname "$0")"
LOG="$WORKDIR/train_all.log"
echo "=== T170 run_all_seeds start $(date) ===" | tee -a "$LOG"

for SEED in 1 7 13 42 100; do
    echo "--- seed=$SEED start $(date) ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 "$WORKDIR/train_T170.py" \
        --seed $SEED \
        --epochs 11 \
        --lr 3e-5 \
        --lambda-spo 30.0 \
        --batch-size 4096 \
        --no-wandb \
        2>&1 | tee -a "$LOG"
    echo "--- seed=$SEED done $(date) ---" | tee -a "$LOG"
done

echo "=== T170 all seeds done $(date) ===" | tee -a "$LOG"
