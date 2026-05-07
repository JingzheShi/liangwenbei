#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

LOG=run_5seed.log
echo "=== T74 5-seed run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee $LOG

for SEED in 42 1 7 13 100; do
    echo "" | tee -a $LOG
    echo ">>> seed=$SEED $(date -u +%H:%M:%S)" | tee -a $LOG
    python3 train_v4_time.py --seed $SEED 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== T74 5-seed DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a $LOG
