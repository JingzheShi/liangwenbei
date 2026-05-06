#!/bin/bash
# T32: Run folds 1..4 sequentially after fold 0 completes.
# Each fold logs to fold{K}.out
set -e
cd "$(dirname "$0")"
for K in 1 2 3 4; do
  echo "=== Starting fold $K at $(date '+%H:%M:%S') ==="
  CUDA_VISIBLE_DEVICES=0 python3 train_loso.py --held $K --epochs 6 --batch_size 1024 > fold${K}.out 2>&1
  echo "=== Fold $K done at $(date '+%H:%M:%S') ==="
done
echo "=== ALL FOLDS DONE at $(date '+%H:%M:%S') ==="
