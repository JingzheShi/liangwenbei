#!/bin/bash
set -e
cd /root/lwb_work_v3
for variant in zero min max; do
  for seed in 1 7 42; do
    if [ -f "summary_cb_huber_a0.001_${variant}_seed${seed}.json" ]; then
      echo "skip variant=$variant seed=$seed (already done)"
      continue
    fi
    echo "=== variant=$variant seed=$seed ==="
    python3 train_cb_nan.py --seed $seed --variant $variant --no-wandb 2>&1 | tee log_${variant}_s${seed}.log
  done
done
echo 'ALL DONE'
