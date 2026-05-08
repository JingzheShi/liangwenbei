#!/bin/bash
set -e
cd /root/lwb_work_multitick

mkdir -p logs

# Sequential to avoid GPU contention; 6 runs total, ~12s each = ~75s
for seed in 1 7 42; do
  for variant in baseline trick; do
    echo "=== seed=$seed variant=$variant ==="
    python3 train_multitick.py --seed $seed --variant $variant --no-wandb 2>&1 | tee logs/log_${variant}_seed${seed}.log | tail -20
    echo ""
  done
done

echo "ALL TRAINING DONE"
ls -la model_*.txt summary_*.json pred_*.parquet
