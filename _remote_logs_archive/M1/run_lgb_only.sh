#!/bin/bash
set -e
SEEDS=(1 7 13 42 100)
for S in "${SEEDS[@]}"; do
  echo "=== LGB seed $S $(date -Iseconds) ==="
  T191_CACHE_SP=/root/T191/cache_sp T191_CACHE_TR=/root/T191/cache_trend \
    python3 /root/T191/scripts/train_T191_lgb_seed.py --seed $S 2>&1 | tail -10
done
echo "=== ALL LGB DONE $(date -Iseconds) ==="
