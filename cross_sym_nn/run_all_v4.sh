#!/bin/bash
# Run all 4 V4 variants x 5 seeds sequentially on GPU 0.
# Each run ~12-15 min. Total ~4-5 hr.
set -u
cd "$(dirname "$0")"
LOG=logs_v4
mkdir -p "$LOG" runs_v4

for arch in multi_h_sae sae_cross_sym sae_wider vae_mlp; do
    for seed in 0 1 2 3 4; do
        OUT="runs_v4/${arch}_pruned259_s${seed}"
        LOGF="$LOG/${arch}_s${seed}.log"
        if [ -f "${OUT}/results.json" ]; then
            echo "[skip] ${arch} s${seed} (already done)"
            continue
        fi
        echo "[run] ${arch} s${seed} → ${LOGF}"
        CUDA_VISIBLE_DEVICES=0 python3 train_v4.py \
            --arch "$arch" --seed "$seed" \
            --keep-idx keep_idx_259d.npy \
            --out-root runs_v4 \
            > "$LOGF" 2>&1
        EC=$?
        if [ $EC -ne 0 ]; then
            echo "[FAIL] ${arch} s${seed} exit=${EC}"
            # If first seed of an arch fails (per the task discipline), skip the rest of this arch
            if [ "$seed" = "0" ]; then
                echo "[skip-arch] ${arch} broken on first seed → skipping seeds 1-4"
                break
            fi
        fi
    done
done
echo "all_done"
