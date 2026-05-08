#!/bin/bash
set -e
cd /root/projects/liangwenbei_workdir/experiments/R_hawkes

echo "[$(date +%H:%M:%S)] Starting Hawkes L2 experiment (local)"
export CUDA_VISIBLE_DEVICES=0

# Train baseline (3 seeds)
for seed in 1 7 42; do
    echo "[$(date +%H:%M:%S)] >>> baseline_seed${seed}"
    python3 train_hawkes_l2_local.py --seed $seed --variant baseline
done

# Train hawkes (3 seeds)
for seed in 1 7 42; do
    echo "[$(date +%H:%M:%S)] >>> hawkes_seed${seed}"
    python3 train_hawkes_l2_local.py --seed $seed --variant hawkes
done

# DE + LOSO evaluation
echo "[$(date +%H:%M:%S)] >>> DE + LOSO evaluation"
python3 eval_de_loso.py

echo "[$(date +%H:%M:%S)] DONE"
