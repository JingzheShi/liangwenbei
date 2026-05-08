#!/bin/bash
set -e
cd /root/projects/liangwenbei_workdir/experiments/T91_revol_regr_revival
for SEED in 42 1 7 13 100; do
  echo "==== seed $SEED start $(date) ===="
  python3 train_regr.py --seed $SEED --variant v1_all32 --no-gpu --no-wandb --num-threads 14 \
    > logs/train_seed${SEED}.log 2>&1
  echo "==== seed $SEED done $(date) ===="
done
echo "ALL DONE $(date)"
