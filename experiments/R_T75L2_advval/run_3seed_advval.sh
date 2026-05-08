#!/bin/bash
# Run 3 seeds × T75 LGB L2 with adversarial-val reweight (trick)
# Use existing baseline preds from /root/lwb_remote_pkg/preds/ for comparison
set -e
cd /root/lwb_work_t75_advval
SEEDS=(42 7 13)
for s in "${SEEDS[@]}"; do
  echo "=== seed=$s trick ===" | tee -a multi_train.log
  date | tee -a multi_train.log
  python3 train_T75_l2_advval.py --seed $s --no-wandb 2>&1 | tee -a multi_train_seed${s}.log | tail -100 >> multi_train.log
  echo "=== seed=$s DONE ===" | tee -a multi_train.log
done
echo "ALL_TRAIN_DONE" | tee -a multi_train.log
