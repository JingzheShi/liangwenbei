#!/bin/bash
# Sequential NGBoost queue (CPU): seed 1 already started; seeds 7 and 42 follow.
# CPU-only; runs after seed 1 finishes to avoid sklearn parallelism contention.

cd /root/projects/liangwenbei_workdir/experiments/T133_alt_boosters
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# wait for ngboost seed 1 to finish
while pgrep -f "train_ngboost.py --seed 1" > /dev/null; do
    sleep 10
done
echo "NGB seed 1 done at $(date)"

for seed in 7 42; do
    echo "=== NGB seed=$seed start at $(date) ==="
    python3 train_ngboost.py --seed $seed --no-wandb \
        > train_ngboost_seed${seed}.log 2>&1
    echo "=== NGB seed=$seed done at $(date) ==="
done

echo "ALL NGBOOST DONE at $(date)"
touch /root/projects/liangwenbei_workdir/experiments/T133_alt_boosters/.ngb_queue_done
