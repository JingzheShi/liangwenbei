#!/bin/bash
# T93 sweep runner: runs each config (single seed=42) sequentially.
# Usage: ./run_sweep.sh [config_idx_start] [config_idx_end]
set -e
cd "$(dirname "$0")"

START=${1:-0}
END=${2:-99}

CONFIGS=$(python3 -c "import json; cfgs=json.load(open('sweep_configs.json')); print(len(cfgs))")
echo "Total configs: $CONFIGS"

for i in $(seq $START $((END < CONFIGS-1 ? END : CONFIGS-1))); do
  CONFIG=$(python3 -c "
import json
cfgs = json.load(open('sweep_configs.json'))
c = cfgs[$i]
out = []
for k, v in c.items():
    flag = '--' + k.replace('_', '-')
    out.append(f'{flag}={v}')
print(' '.join(out))
")
  echo "===================================================="
  echo "[Config $i/$((CONFIGS-1))] $CONFIG"
  echo "===================================================="
  TAG=$(python3 -c "import json; print(json.load(open('sweep_configs.json'))[$i]['tag'])")
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL python3 train_t93.py \
    --seed 42 \
    --no-wandb \
    $CONFIG \
    > "train_${TAG}_seed42.log" 2>&1 || echo "  FAILED config $i"
  echo "  -> done $TAG"
  tail -3 "train_${TAG}_seed42.log"
done
echo "ALL DONE"
