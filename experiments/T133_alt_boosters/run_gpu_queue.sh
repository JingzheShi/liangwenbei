#!/bin/bash
# Sequential GPU queue: DART seed 7, 42 + Tweedie seed 1, 7, 42
# (DART seed 1 already running; this kicks off after it finishes)

set -e
cd /root/projects/liangwenbei_workdir/experiments/T133_alt_boosters
export CUDA_VISIBLE_DEVICES=0

# wait for DART seed 1 to finish
while pgrep -f "train_lgb_alt.py --seed 1 --variant dart" > /dev/null; do
    sleep 5
done
echo "DART seed 1 done at $(date)"

for seed in 7 42; do
    echo "=== DART seed=$seed start at $(date) ==="
    python3 train_lgb_alt.py --seed $seed --variant dart --no-wandb \
        > train_dart_seed${seed}.log 2>&1
    echo "=== DART seed=$seed done at $(date) ==="
done

for seed in 1 7 42; do
    echo "=== Tweedie seed=$seed start at $(date) ==="
    python3 train_lgb_alt.py --seed $seed --variant tweedie --no-wandb \
        > train_tweedie_seed${seed}.log 2>&1
    echo "=== Tweedie seed=$seed done at $(date) ==="
done

echo "ALL GPU JOBS DONE at $(date)"
touch /root/projects/liangwenbei_workdir/experiments/T133_alt_boosters/.gpu_queue_done
