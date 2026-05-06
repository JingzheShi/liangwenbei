#!/bin/bash
# Run remaining LOSO folds sequentially.
# Assumes fold 0 was launched separately; this picks up from fold 1.
set -euo pipefail

cd /root/projects/liangwenbei_workdir
mkdir -p experiments/T19_NN_regularized

START_FOLD=${1:-1}
END_FOLD=${2:-4}

for K in $(seq $START_FOLD $END_FOLD); do
  echo "=== Launching fold $K at $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 python3 experiments/T19_NN_regularized/train_loso.py \
    --held $K --epochs 5 --batch_size 1024 --num_workers 4 \
    > experiments/T19_NN_regularized/fold${K}.out 2>&1
  echo "=== Done fold $K at $(date -Is) ==="
done
echo "ALL FOLDS DONE at $(date -Is)"
