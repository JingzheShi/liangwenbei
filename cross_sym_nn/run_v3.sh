#!/bin/bash
# v3 sae_mlp HP sweep: 4 configs × 3 seeds
set -u
cd "$(dirname "$0")"
mkdir -p logs_v3 runs_v3

export WANDB_API_KEY="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"
export CUDA_VISIBLE_DEVICES=0

CONFIGS=("A" "B" "C" "D")
SEEDS=(0 1 2)
ARCH="sae_mlp"

for CFG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOG="logs_v3/${ARCH}_cfg${CFG}_s${SEED}.log"
    echo "[$(date +%H:%M:%S)] START arch=$ARCH config=$CFG seed=$SEED -> $LOG"
    python3 train_v3.py \
      --arch "$ARCH" --config "$CFG" --seed "$SEED" --cuda 0 \
      --max-epochs 20 --batch-size 1024 --grad-clip 1.0 \
      > "$LOG" 2>&1
    EC=$?
    RESULT=$(tail -3 "$LOG" | grep "DONE:")
    echo "[$(date +%H:%M:%S)] DONE  arch=$ARCH config=$CFG seed=$SEED ec=$EC | $RESULT"
  done
done

echo "[$(date +%H:%M:%S)] v3 ALL RUNS COMPLETE"
