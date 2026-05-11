#!/bin/bash
# T188v2 CORRECTED mega ensemble — Phase 1+2 (50 NN) then Phase 3 (50 LGB)
# Run on vastai-r2 (RTX 3080)
set -e

WORKDIR="/root/projects/liangwenbei_workdir/experiments/T188v2_mega_proper"
cd "$WORKDIR"
mkdir -p logs T81_pretrained

echo "=== T188v2 CORRECTED Remote Training ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Phase 1 val split: schemeP_val.npz (dates 80-95)"

wandb login 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 --relogin 2>/dev/null || true

echo ""
echo "=== PHASE 1+2: Training 50 NN seeds (CORRECTED pretrain) ==="
PHASE12_START=$(date +%s)

FAILED_SEEDS=""
for S in $(seq 1 50); do
    echo ""
    echo "--- NN seed $S / 50 ---"
    if [ -f "nn_h60_seed${S}.npz" ]; then
        echo "  SKIP: nn_h60_seed${S}.npz already exists"
        continue
    fi
    python3 train_T188v2_nn_seed.py --seed $S --cuda 0 --no-wandb \
        2>&1 | tee logs/nn_seed${S}.log

    # Check Phase 1 convergence
    BEST_EP=$(python3 -c "
import json, sys
try:
    with open('summary_nn_seed${S}.json') as f:
        d = json.load(f)
    print(d.get('phase1_best_epoch', -1))
except:
    print(-1)
" 2>/dev/null)

    echo "NN seed $S done: best_epoch=$BEST_EP $(date)"

    if [ "$BEST_EP" -le 0 ] 2>/dev/null; then
        echo "  WARNING: seed $S best_epoch=$BEST_EP <= 0 !"
        FAILED_SEEDS="$FAILED_SEEDS $S"
    fi
done

PHASE12_END=$(date +%s)
PHASE12_MIN=$(( (PHASE12_END - PHASE12_START) / 60 ))
echo ""
echo "=== Phase 1+2 done in ${PHASE12_MIN} min ==="

# Check convergence
NN_COUNT=$(ls nn_h60_seed*.npz 2>/dev/null | wc -l)
echo "NN seeds done: $NN_COUNT / 50"

# Report convergence stats
echo "Checking convergence..."
python3 -c "
import json, os, glob
summaries = sorted(glob.glob('summary_nn_seed*.json'))
best_epochs = []
for f in summaries:
    try:
        d = json.load(open(f))
        be = d.get('phase1_best_epoch', -1)
        best_epochs.append(be)
    except:
        pass
if best_epochs:
    n_converged = sum(1 for be in best_epochs if be > 5)
    n_partial = sum(1 for be in best_epochs if 0 < be <= 5)
    n_failed = sum(1 for be in best_epochs if be <= 0)
    print(f'  Total seeds: {len(best_epochs)}')
    print(f'  Converged (best_ep > 5): {n_converged}')
    print(f'  Partial (0 < best_ep <= 5): {n_partial}')
    print(f'  Failed (best_ep <= 0): {n_failed}')
    print(f'  best_epochs: {best_epochs}')
    if n_failed > len(best_epochs) * 0.2:
        print('ABORT SIGNAL: >20% seeds failed to converge!')
        exit(1)
    else:
        print('OK: <20% failed')
else:
    print('No summaries found')
"

echo ""
echo "=== PHASE 3: Training 50 LGB seeds ==="
PHASE3_START=$(date +%s)

for S in $(seq 1 50); do
    echo ""
    echo "--- LGB seed $S / 50 ---"
    if [ -f "model_h60_seed${S}.txt" ]; then
        echo "  SKIP: model_h60_seed${S}.txt already exists"
        continue
    fi
    python3 train_T188v2_lgb_seed.py --seed $S --no-wandb \
        2>&1 | tee logs/lgb_seed${S}.log
    echo "LGB seed $S done: $(date)"
done

PHASE3_END=$(date +%s)
PHASE3_MIN=$(( (PHASE3_END - PHASE3_START) / 60 ))
echo ""
echo "=== Phase 3 done in ${PHASE3_MIN} min ==="

echo ""
echo "=== TRAINING SUMMARY ==="
echo "Finished: $(date)"
echo "Phase 1+2 NN: ${PHASE12_MIN} min"
echo "Phase 3 LGB: ${PHASE3_MIN} min"

NN_COUNT=$(ls nn_h60_seed*.npz 2>/dev/null | wc -l)
LGB_COUNT=$(ls model_h60_seed*.txt 2>/dev/null | wc -l)
echo "NN npz files: $NN_COUNT / 50"
echo "LGB txt files: $LGB_COUNT / 50"
