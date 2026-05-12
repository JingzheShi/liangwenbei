#!/bin/bash
# Train all 50 NN models (T81 pretrain + T87 SPO+ M7 fine-tune).
# Usage: bash run_all_nn_seeds.sh <CACHE_DIR> <OUT_DIR> [CUDA_DEVICE]
# Example: bash run_all_nn_seeds.sh ./outputs/cache ./outputs/models 0
set -e

CACHE_DIR=${1:-./outputs/cache}
OUT_DIR=${2:-./outputs/models}
CUDA=${3:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== Training 50 NN models ==="
echo "  cache: $CACHE_DIR"
echo "  output: $OUT_DIR"
echo "  CUDA device: $CUDA"
echo ""

for SEED in $(seq 1 50); do
    echo "--- NN seed=$SEED ---"
    python3 "$SCRIPT_DIR/train_T188v2_nn_seed.py" \
        --seed $SEED \
        --cache-dir "$CACHE_DIR" \
        --out-dir "$OUT_DIR" \
        --cuda "$CUDA" \
        --skip-phase1
done

echo ""
echo "=== All 50 NN models done ==="
ls -1 "$OUT_DIR"/nn_h60_seed*.npz 2>/dev/null | wc -l
