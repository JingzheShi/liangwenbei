#!/bin/bash
# T172: Precompute data once, then train 5 seeds sequentially
set -e
export CACHE_DIR=/root/lwb_remote_pkg/cache
OUTDIR="/root/T172_out"
mkdir -p "$OUTDIR"

echo "=== Step 1: Precompute data (once) ==="
python3 /root/T172/precompute_data.py 2>&1 | tee "$OUTDIR/precompute.log"

echo "=== Step 2: Train 5 seeds ==="
for S in 1 7 13 42 100; do
    echo "--- Seed $S ---"
    python3 /root/T172/train_transformer.py \
        --seed $S \
        --horizon 60 \
        --n-tokens 32 \
        --d-token 64 \
        --n-heads 4 \
        --depth 2 \
        --ffn-mult 4.0 \
        --dropout 0.1 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --batch-size 4096 \
        --epochs 15 \
        --cuda 0 \
        --out-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/train_seed${S}.log"
    echo "Seed $S complete"
done

echo "=== ALL DONE ==="
ls -lh "$OUTDIR"/*.npz
