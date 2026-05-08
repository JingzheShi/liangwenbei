#!/bin/bash
set -e
cd /root/projects/liangwenbei_workdir/experiments/R_Hawkes_OFI

export WANDB_API_KEY="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"
export WANDB_PROJECT="liangwenbei"
export WANDB_ENTITY="cjxh21-Tsinghua University"

SCHEME_DIR=/root/projects/liangwenbei_workdir/experiments/T68_stage5_features/cache
HK_DIR=/root/projects/liangwenbei_workdir/experiments/R_Hawkes_OFI
OUT=$HK_DIR
LOG=$OUT/train_all.log

: > "$LOG"
T0=$(date +%s)

VARIANTS=(baseline hawkes)
SEEDS=(1 7 42)

for V in "${VARIANTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    TAG="${V}_seed${S}"
    if [[ -f "$OUT/summary_${TAG}.json" ]]; then
      echo "[$(date +%H:%M:%S)] SKIP $TAG (already exists)" | tee -a "$LOG"
      continue
    fi
    echo "[$(date +%H:%M:%S)] >>> $TAG" | tee -a "$LOG"
    python3 train_hawkes.py \
      --variant "$V" --seed "$S" \
      --scheme-train "$SCHEME_DIR/schemeP_train.npz" \
      --scheme-test  "$SCHEME_DIR/schemeP_test.npz" \
      --scheme-feat-names "$SCHEME_DIR/schemeP_feat_names.txt" \
      --hawkes-train "$HK_DIR/hawkes_train.npy" \
      --hawkes-test  "$HK_DIR/hawkes_test.npy" \
      --hawkes-names "$HK_DIR/hawkes_train_names.txt" \
      --num-threads 8 \
      --out-dir "$OUT" \
      2>&1 | tee -a "$LOG"
    DT=$(($(date +%s) - T0))
    echo "[$(date +%H:%M:%S)] elapsed=${DT}s" | tee -a "$LOG"
  done
done

echo "[$(date +%H:%M:%S)] all done in $(($(date +%s) - T0))s" | tee -a "$LOG"
