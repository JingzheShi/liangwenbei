#!/bin/bash
# Run 3 arch × 5 seeds sequentially. Logs in logs_v2/{arch}_s{seed}.log
set -u
cd "$(dirname "$0")"
mkdir -p logs_v2

export WANDB_API_KEY="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"

ARCHS=("isab" "sae_mlp" "deepsets")
SEEDS=(0 1 2 3 4)

for ARCH in "${ARCHS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOG="logs_v2/${ARCH}_s${SEED}.log"
    echo "[$(date +%H:%M:%S)] START arch=$ARCH seed=$SEED -> $LOG"
    python3 train_v2.py \
      --arch "$ARCH" --seed "$SEED" --cuda 0 \
      --max-epochs 20 --eval-every 200 --patience-steps 1500 \
      --batch-size 1024 --lr 1e-4 --weight-decay 1e-3 \
      --warmup-steps 200 --grad-clip 1.0 \
      > "$LOG" 2>&1
    EC=$?
    echo "[$(date +%H:%M:%S)] DONE  arch=$ARCH seed=$SEED ec=$EC"
  done
done

echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE"
