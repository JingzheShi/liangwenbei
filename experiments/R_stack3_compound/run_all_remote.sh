#!/bin/bash
# Run all 5 variants × 3 seeds = 15 LGB Huber trains sequentially on r4.
set -e
cd /root/lwb_work_v3

export WANDB_API_KEY="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"
export WANDB_PROJECT="liangwenbei"
export WANDB_ENTITY="cjxh21-Tsinghua University"

SCHEME_DIR=/root/lwb_remote_pkg/cache
HYD_DIR=/root/lwb_work_v3/hyd
LAG_DIR=/root/lwb_work_v3/lagret
OUT=/root/lwb_work_v3
LOG=/root/lwb_work_v3/train_all.log

: > "$LOG"
T0=$(date +%s)

VARIANTS=(V0 V1 Vlogret V2 V3)
SEEDS=(1 7 42)

for V in "${VARIANTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    TAG="${V}_seed${S}"
    if [[ -f "$OUT/summary_${TAG}.json" ]]; then
      echo "[$(date +%H:%M:%S)] SKIP $TAG (already exists)" | tee -a "$LOG"
      continue
    fi
    echo "[$(date +%H:%M:%S)] >>> $TAG" | tee -a "$LOG"
    python3 /root/lwb_work_v3/train_stack3.py \
      --variant "$V" --seed "$S" \
      --scheme-train "$SCHEME_DIR/schemeP_train.npz" \
      --scheme-test  "$SCHEME_DIR/schemeP_test.npz" \
      --scheme-feat-names "$SCHEME_DIR/schemeP_feat_names.txt" \
      --hyd-train "$HYD_DIR/hyd_train.npy" \
      --hyd-test  "$HYD_DIR/hyd_test.npy" \
      --hyd-names "$HYD_DIR/hyd_feat_names.txt" \
      --lagret-train "$LAG_DIR/lagret_train.npz" \
      --lagret-test  "$LAG_DIR/lagret_test.npz" \
      --num-threads 4 \
      --out-dir "$OUT" \
      >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] <<< $TAG" | tee -a "$LOG"
  done
done

echo "[$(date +%H:%M:%S)] all training done in $(( $(date +%s) - T0 ))s" | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] running eval" | tee -a "$LOG"
python3 /root/lwb_work_v3/eval_stack3.py --pred-dir "$OUT" --out-json "$OUT/eval_results.json" >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
