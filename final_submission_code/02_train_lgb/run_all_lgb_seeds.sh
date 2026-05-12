#!/bin/bash
# Train all 50 LGB models (5 HP configs x 10 seeds each).
# Usage: bash run_all_lgb_seeds.sh <CACHE_DIR> <OUT_DIR> [GPU_FLAG]
# Example: bash run_all_lgb_seeds.sh ./outputs/cache ./outputs/models
set -e

CACHE_DIR=${1:-./outputs/cache}
OUT_DIR=${2:-./outputs/models}
GPU_FLAG=${3:-""}  # pass "--gpu" to use GPU training

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== Training 50 LGB models ==="
echo "  cache: $CACHE_DIR"
echo "  output: $OUT_DIR"
echo "  gpu: ${GPU_FLAG:-none}"
echo ""

for SEED in $(seq 1 50); do
    echo "--- LGB seed=$SEED ---"
    python3 "$SCRIPT_DIR/train_T188v2_lgb_seed.py" \
        --seed $SEED \
        --cache-dir "$CACHE_DIR" \
        --out-dir "$OUT_DIR" \
        $GPU_FLAG
done

echo ""
echo "=== All 50 LGB models done ==="
ls -1 "$OUT_DIR"/model_h60_seed*.txt 2>/dev/null | wc -l
