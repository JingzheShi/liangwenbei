#!/bin/bash
# T188 Remote runner: Phase 1+2 (50 NN seeds) then Phase 3 (50 LGB seeds)
# Run on vastai-r2 (RTX 3080)
set -e

WORKDIR="/root/projects/liangwenbei_workdir/experiments/T188_mega_ensemble"
cd "$WORKDIR"

echo "=== T188 Remote Training ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Login WandB
wandb login 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 --relogin 2>/dev/null || true

echo ""
echo "=== PHASE 1+2: Training 50 NN seeds ==="
PHASE12_START=$(date +%s)

for S in $(seq 1 50); do
    echo ""
    echo "--- NN seed $S / 50 ---"
    python3 train_T188_nn_seed.py --seed $S --cuda 0 --no-wandb 2>&1 | tee -a logs/nn_seed${S}.log
    echo "NN seed $S done: $(date)"
done

PHASE12_END=$(date +%s)
PHASE12_MIN=$(( (PHASE12_END - PHASE12_START) / 60 ))
echo ""
echo "=== Phase 1+2 done in ${PHASE12_MIN} min ==="

echo ""
echo "=== PHASE 3: Training 50 LGB seeds ==="
PHASE3_START=$(date +%s)

for S in $(seq 1 50); do
    echo ""
    echo "--- LGB seed $S / 50 ---"
    python3 train_T188_lgb_seed.py --seed $S --no-wandb 2>&1 | tee -a logs/lgb_seed${S}.log
    echo "LGB seed $S done: $(date)"
done

PHASE3_END=$(date +%s)
PHASE3_MIN=$(( (PHASE3_END - PHASE3_START) / 60 ))
echo ""
echo "=== Phase 3 done in ${PHASE3_MIN} min ==="

echo ""
echo "=== ALL TRAINING DONE ==="
echo "Finished: $(date)"
echo "Phase 1+2: ${PHASE12_MIN} min"
echo "Phase 3: ${PHASE3_MIN} min"

# Count outputs
NN_COUNT=$(ls nn_h60_seed*.npz 2>/dev/null | wc -l)
LGB_COUNT=$(ls model_h60_seed*.txt 2>/dev/null | wc -l)
echo "NN npz files: $NN_COUNT / 50"
echo "LGB txt files: $LGB_COUNT / 50"
