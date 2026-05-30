#!/bin/bash
# Run all 6 cross-sym experiments: 3 archs x 2 seeds
set -e
cd "$(dirname "$0")"

CUDA=0
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_exp() {
    ARCH=$1
    SEED=$2
    LOG="$LOG_DIR/${ARCH}_s${SEED}.log"
    RESULT_DIR="runs/${ARCH}_s${SEED}"

    if [ -f "$RESULT_DIR/results.json" ]; then
        PNL=$(python3 -c "import json; r=json.load(open('$RESULT_DIR/results.json')); print(f\"{r.get('test_pnl','N/A'):.4f}\" if r.get('test_pnl') else 'FAILED')")
        echo "[SKIP] ${ARCH}_s${SEED} already done: test_pnl=$PNL"
        return
    fi

    echo "[RUN] ${ARCH}_s${SEED} -> $LOG"
    CUDA_VISIBLE_DEVICES=$CUDA python3 train.py \
        --arch "$ARCH" --seed "$SEED" \
        --epochs 50 --patience 10 --batch-size 256 \
        --wandb-project liangwenbei-cross-sym \
        > "$LOG" 2>&1

    PNL=$(python3 -c "import json; r=json.load(open('$RESULT_DIR/results.json')); print(f\"{r.get('test_pnl',0):.4f}\")" 2>/dev/null || echo "ERR")
    echo "[DONE] ${ARCH}_s${SEED} test_pnl=$PNL"
}

echo "=== Starting cross-sym experiments ==="
date

# Architecture A: Cross-Concat MLP (simplest, runs first as sanity check)
run_exp cross_mlp 0
run_exp cross_mlp 1

# Architecture B: Sym-Token Self-Attention (core)
run_exp sym_attn 0
run_exp sym_attn 1

# Architecture C: Hybrid Cross-Attn
run_exp hybrid 0
run_exp hybrid 1

echo ""
echo "=== ALL DONE ==="
date
echo ""
echo "=== Results Summary ==="
python3 -c "
import json
from pathlib import Path

archs = ['cross_mlp', 'sym_attn', 'hybrid']
seeds = [0, 1]
baseline_mlp = 31.44
baseline_ft  = 26.73

print(f'  baseline MLP={baseline_mlp:+.2f}  FT={baseline_ft:+.2f}')
print()
for arch in archs:
    vals = []
    for seed in seeds:
        rf = Path(f'runs/{arch}_s{seed}/results.json')
        if rf.exists():
            r = json.load(open(rf))
            if r.get('test_pnl') is not None:
                vals.append(r['test_pnl'])
    if vals:
        m = sum(vals)/len(vals)
        print(f'  {arch:15s} n={len(vals)} mean={m:+.4f}  vs_mlp={m-baseline_mlp:+.4f}')
    else:
        print(f'  {arch:15s} no results')
"
