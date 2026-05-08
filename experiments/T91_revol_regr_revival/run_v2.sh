#!/bin/bash
set -e
cd /root/projects/liangwenbei_workdir/experiments/T91_revol_regr_revival
for SEED in 42 1 7 13 100; do
  echo "==== v2 seed $SEED start $(date) ===="
  python3 train_regr.py --seed $SEED --variant v2_revol12 --no-gpu --no-wandb --num-threads 14 \
    > logs/train_v2_seed${SEED}.log 2>&1
  echo "==== v2 seed $SEED done $(date) ===="
done
echo "ALL V2 DONE $(date)"
