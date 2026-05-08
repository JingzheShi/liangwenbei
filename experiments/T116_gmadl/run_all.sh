#!/bin/bash
# T116: Train 3 seeds × 3 GMADL configs = 9 trains.
# Configs are *calibrated* for our return scale (std~2e-3). The literature uses
# a∈{500,1000,2000} for unscaled returns at scale ~1e-2, so we scale a up by
# (1e-2/2e-3)^2 = 25 → a_calibrated ≈ {25k, 50k, 50k}, and empirically a~5e5 is
# our practical sweet spot (see REPORT.md for scan results).
set -e
cd "$(dirname "$0")"

CONFIGS=(
  "500000 1.0"
  "500000 1.5"
  "1000000 1.0"
)
SEEDS=(1 7 42)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    A=$(echo $CFG | cut -d' ' -f1)
    B=$(echo $CFG | cut -d' ' -f2)
    echo "===== seed=$SEED a=$A b=$B ====="
    LOG="train_seed${SEED}_a${A}_b${B}.log"
    python3 train_gmadl.py \
      --seed $SEED --gmadl-a $A --gmadl-b $B \
      --num-boost-round 600 --early-stopping 40 \
      --num-threads 18 \
      --no-wandb 2>&1 | tee "$LOG" | tail -4
  done
done
echo "ALL DONE"
