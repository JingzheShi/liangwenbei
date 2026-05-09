#!/bin/bash
# T155: Reproduce T140c v2 training exactly
# Seeds 1,7: train_full.py --no-gpu --num-threads 18 (with data_random_seed=seed+3)
# Seeds 13,42,100: train_full_memeff.py --num-threads 8 (no data_random_seed)

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/train_T155.log"
echo "=== T155 v2 replica training ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

cd "$HERE"

echo "=== Seed 1 (train_full --no-gpu --num-threads 18) ===" | tee -a "$LOG"
python3 train_full.py --seed 1 --no-gpu --num-threads 18 --no-wandb 2>&1 | tee -a "$LOG"

echo "=== Seed 7 (train_full --no-gpu --num-threads 18) ===" | tee -a "$LOG"
python3 train_full.py --seed 7 --no-gpu --num-threads 18 --no-wandb 2>&1 | tee -a "$LOG"

echo "=== Seed 13 (train_full_memeff --num-threads 8) ===" | tee -a "$LOG"
python3 train_full_memeff.py --seed 13 --num-threads 8 --no-wandb 2>&1 | tee -a "$LOG"

echo "=== Seed 42 (train_full_memeff --num-threads 8) ===" | tee -a "$LOG"
python3 train_full_memeff.py --seed 42 --num-threads 8 --no-wandb 2>&1 | tee -a "$LOG"

echo "=== Seed 100 (train_full_memeff --num-threads 8) ===" | tee -a "$LOG"
python3 train_full_memeff.py --seed 100 --num-threads 8 --no-wandb 2>&1 | tee -a "$LOG"

echo "End: $(date)" | tee -a "$LOG"
echo "All 5 seeds done." | tee -a "$LOG"
