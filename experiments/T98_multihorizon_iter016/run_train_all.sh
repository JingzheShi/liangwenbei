#!/bin/bash
# Run training for all h ∈ {5, 10, 20, 40} × 5 seeds × {LGB, CB}.
# Sequential because GPU is shared. Logs to train_<model>_h<H>_seed<S>.log
set -e
cd /root/projects/liangwenbei_workdir
DIR=experiments/T98_multihorizon_iter016
SEEDS=(42 1 7 13 100)
HORIZONS=(5 10 20 40)

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    LOG=$DIR/train_lgb_h${H}_seed${S}.log
    if [ -f $DIR/summary_lgb_h${H}_seed${S}.json ]; then
      echo "[skip] LGB h=$H seed=$S exists"
      continue
    fi
    echo "[lgb] h=$H seed=$S -> $LOG"
    CUDA_VISIBLE_DEVICES=0 python3 $DIR/train_lgb_h.py --seed $S --horizon $H --no-wandb > $LOG 2>&1
    tail -2 $LOG
  done
done

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    LOG=$DIR/train_cb_h${H}_seed${S}.log
    if [ -f $DIR/summary_cb_h${H}_seed${S}.json ]; then
      echo "[skip] CB h=$H seed=$S exists"
      continue
    fi
    echo "[cb]  h=$H seed=$S -> $LOG"
    CUDA_VISIBLE_DEVICES=0 python3 $DIR/train_cb_h.py --seed $S --horizon $H --no-wandb > $LOG 2>&1
    tail -2 $LOG
  done
done

echo "ALL DONE"
