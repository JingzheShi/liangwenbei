#!/bin/bash
set -e
cd /root/projects/liangwenbei_workdir/experiments/R_full_retrain
mkdir -p logs
# seed=1 already done in CPU mode. Run remaining 4 in CPU mode.
for seed in 7 13 42 100; do
  echo "=== seed=$seed start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  CUDA_VISIBLE_DEVICES= python3 -u train_full.py --seed $seed --num-boost-round 330 --no-wandb --no-gpu 2>&1 | tee -a logs/train_seed${seed}.log
  echo "=== seed=$seed end $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done

echo "All seeds complete."
ls -la model_h60_seed*.txt summary_h60_seed*.json
