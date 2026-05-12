#!/bin/bash
# Usage: run_T190_remote.sh <seed_start> <seed_end>
# Trains LGB seed_start..seed_end, then NN seed_start..seed_end
# Both use GPU 0. LGB is fast (~12s each), NN is slow (~80s each).

SEED_START=${1:-51}
SEED_END=${2:-75}
WORKDIR=/root/T190

echo "=== T190 Remote Runner: seeds ${SEED_START}-${SEED_END} ==="
echo "WORKDIR=$WORKDIR"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd $WORKDIR || { echo "ERROR: $WORKDIR not found"; exit 1; }

# WandB login
export WANDB_API_KEY="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"

# === Phase A: Train LGB (fast, GPU) ===
echo ""
echo "=== PHASE A: LGB seeds ${SEED_START}-${SEED_END} ==="
for seed in $(seq $SEED_START $SEED_END); do
    echo "--- LGB seed=$seed ---"
    python3 train_T190_lgb_seed.py --seed $seed --horizon 60 --no-wandb 2>&1 | tail -4
done
echo "LGB done at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "LGB count: $(ls model_h60_seed*.txt 2>/dev/null | wc -l)"

# === Phase B: Train NN (slow, GPU) ===
echo ""
echo "=== PHASE B: NN seeds ${SEED_START}-${SEED_END} ==="
for seed in $(seq $SEED_START $SEED_END); do
    echo "--- NN seed=$seed ---"
    python3 train_T190_nn_seed.py --seed $seed --horizon 60 --cuda 0 --no-wandb 2>&1 | grep -E "(p1 ep [0-9]+ |p2 ep [0-9]+ |DONE|Phase [12] done|best_ep|WARNING|ERROR)"
done
echo "NN done at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NN count: $(ls nn_h60_seed*.npz 2>/dev/null | wc -l)"

echo ""
echo "=== ALL DONE: seeds ${SEED_START}-${SEED_END} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "LGB files: $(ls model_h60_seed*.txt 2>/dev/null | wc -l)"
echo "NN files: $(ls nn_h60_seed*.npz 2>/dev/null | wc -l)"
