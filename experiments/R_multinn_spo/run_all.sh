#!/bin/bash
# Run all 12 trains: 4 archs × 3 seeds.
# Within each seed batch, run 4 archs in parallel (different processes share GPU).
# Sequential across seeds to avoid OOM with too many parallel.
set -u
cd /root/lwb_work_multinn

ARCHS=(t87_baseline wide_mlp deep_resnet glu_mlp)
SEEDS=(1 7 42)

LOG=/root/lwb_work_multinn/run_all.log
echo "=== run_all started $(date) ===" > $LOG

for SEED in "${SEEDS[@]}"; do
  echo "--- launching seed=$SEED batch ---" >> $LOG
  PIDS=()
  for ARCH in "${ARCHS[@]}"; do
    if [ -f "pred_${ARCH}_seed${SEED}.parquet" ]; then
      echo "  SKIP $ARCH seed=$SEED (pred exists)" >> $LOG
      continue
    fi
    echo "  launch $ARCH seed=$SEED" >> $LOG
    python3 -u train_multinn_spo.py --arch $ARCH --seed $SEED --no-wandb \
        > train_${ARCH}_seed${SEED}.log 2>&1 &
    PIDS+=($!)
  done
  echo "  waiting on PIDs: ${PIDS[*]}" >> $LOG
  for P in "${PIDS[@]}"; do wait $P; echo "  pid $P done" >> $LOG; done
  echo "--- seed=$SEED batch done $(date) ---" >> $LOG
done

echo "=== run_all finished $(date) ===" >> $LOG
ls -la /root/lwb_work_multinn/pred_*.parquet >> $LOG
