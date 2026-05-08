#!/bin/bash
# TOD experiment runner: build features, train trick (3 seeds), eval ensemble.
# Reuses R4 baseline preds from /root/lwb_work/.
set -e

OUT=/root/lwb_work_tod
mkdir -p "$OUT"
cd "$OUT"

echo "=== Step 1: build TOD features ==="
python3 /root/lwb_work_tod/build_tod_features.py 2>&1 | tee build_tod.log

echo ""
echo "=== Step 2: sanity (TOD trick must NOT contain forbidden cols) ==="
ls "$OUT"/tod_*.npz

echo ""
echo "=== Step 3: train trick (3 seeds with --use-tod) on GPU ==="
for s in 1 7 42; do
  echo "--- seed $s trick ---"
  CUDA_VISIBLE_DEVICES=0 python3 /root/lwb_work_tod/tod_train.py \
    --seed "$s" --use-tod 2>&1 | tee "tod_trick_s${s}.log"
done

echo ""
echo "=== Step 4: verify R4 baseline preds available for reuse ==="
for s in 1 7 42; do
  ls /root/lwb_work/pred_R4_baseline_seed${s}.parquet
done

echo ""
echo "=== Step 5: ensemble eval ==="
python3 /root/lwb_work_tod/tod_eval.py 2>&1 | tee tod_eval.log

echo ""
echo "DONE - results at $OUT/results.json"
