#!/bin/bash
# T176: Train all HP variants sequentially (one seed at a time to avoid OOM)
# S1: epoch sweep (lr=3e-5, lam=30)
# S2: lr sweep (ep=11, lam=30)
# S3: lambda_spo sweep (ep=11, lr=3e-5)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$SCRIPT_DIR/train_variant.py"
SEEDS="1 7 13 42 100"
CUDA=0

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SCRIPT_DIR/train_all.log"; }

declare -A VARIANTS
# name -> epochs:lr:lambda_spo
VARIANTS["S1_ep7"]="7:3e-5:30"
VARIANTS["S1_ep11"]="11:3e-5:30"   # T170 baseline - skip if already exists
VARIANTS["S1_ep15"]="15:3e-5:30"
VARIANTS["S1_ep20"]="20:3e-5:30"
VARIANTS["S2_lr1e-5"]="11:1e-5:30"
VARIANTS["S2_lr1e-4"]="11:1e-4:30"
VARIANTS["S3_lam10"]="11:3e-5:10"
VARIANTS["S3_lam100"]="11:3e-5:100"

# S1_ep11 = T170 baseline (already trained in experiments/T170_T87_M7); skip to save time
for variant_name in S1_ep7 S1_ep15 S1_ep20 S2_lr1e-5 S2_lr1e-4 S3_lam10 S3_lam100; do
    params="${VARIANTS[$variant_name]}"
    epochs=$(echo "$params" | cut -d: -f1)
    lr=$(echo "$params" | cut -d: -f2)
    lam=$(echo "$params" | cut -d: -f3)

    OUT_DIR="$SCRIPT_DIR/variants/$variant_name"
    mkdir -p "$OUT_DIR"

    # Check if already done (all 5 seeds have npz)
    done_count=$(ls "$OUT_DIR"/nn_h60_seed*.npz 2>/dev/null | wc -l)
    if [ "$done_count" -ge 5 ]; then
        log "SKIP $variant_name (already has $done_count seeds)"
        continue
    fi

    log "=== START $variant_name (ep=$epochs lr=$lr lam=$lam) ==="
    for seed in $SEEDS; do
        # Skip if seed already done
        if [ -f "$OUT_DIR/nn_h60_seed${seed}.npz" ]; then
            log "  skip seed=$seed (already done)"
            continue
        fi
        log "  training seed=$seed ..."
        python3 "$TRAIN" \
            --seed "$seed" \
            --epochs "$epochs" \
            --lr "$lr" \
            --lambda-spo "$lam" \
            --out-dir "$OUT_DIR" \
            --variant-name "$variant_name" \
            --cuda "$CUDA" \
            >> "$SCRIPT_DIR/train_all.log" 2>&1
        log "  done seed=$seed"
    done
    log "=== DONE $variant_name ==="
done

log "All variants trained."
