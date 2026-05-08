#!/bin/bash
# 3 seeds × T75 LGB L2 with milder advval weights (sw_advval_mild.npy)
set -e
cd /root/lwb_work_t75_advval
SEEDS=(42 7 13)
for s in "${SEEDS[@]}"; do
  echo "=== seed=$s mild trick ===" | tee -a multi_train_mild.log
  date | tee -a multi_train_mild.log
  python3 train_T75_l2_advval.py --seed $s --no-wandb --sw-advval-file sw_advval_mild.npy 2>&1 | tee multi_train_mild_seed${s}.log | tail -50 >> multi_train_mild.log
  # Rename pred and summary to mark as mild
  mv pred_T75L2_advval_seed${s}.parquet pred_T75L2_mild_seed${s}.parquet
  mv summary_T75L2_advval_seed${s}.json summary_T75L2_mild_seed${s}.json
  mv model_T75L2_advval_seed${s}.txt model_T75L2_mild_seed${s}.txt
  echo "=== seed=$s DONE ===" | tee -a multi_train_mild.log
done
echo "ALL_MILD_DONE" | tee -a multi_train_mild.log
