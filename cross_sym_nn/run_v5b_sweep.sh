#!/bin/bash
# Phase 2 sweep: 5 archs × 3 seeds = 15 runs
# Skip arch if seed 0 < 33 (save GPU time)

set -e
WORKDIR="/root/projects/liangwenbei_workdir/cross_sym_nn"
CUDA_VISIBLE_DEVICES=0
LOGDIR="$WORKDIR/logs_v5b"
mkdir -p "$LOGDIR"

cd "$WORKDIR"

ARCHS=(v5_diff_attn v5_mask_ssl v5_glu_v_attn v5_market_adaln v5_diff_swiglu)
SEEDS=(0 1 2)

for arch in "${ARCHS[@]}"; do
    first_seed_pnl=""
    for seed in "${SEEDS[@]}"; do
        rundir="$WORKDIR/runs_v5/${arch}_pruned259_s${seed}"
        if [ -f "$rundir/results.json" ]; then
            echo "[SKIP] $arch seed=$seed already done"
            if [ "$seed" = "0" ]; then
                first_seed_pnl=$(python3 -c "import json; r=json.load(open('$rundir/results.json')); print(r['test_pnl'])" 2>/dev/null || echo "0")
            fi
            continue
        fi

        echo "=== START: $arch seed=$seed ===" | tee -a "$LOGDIR/${arch}_s${seed}.log"
        CUDA_VISIBLE_DEVICES=0 python3 train_v5.py \
            --arch "$arch" --seed "$seed" \
            --keep-idx keep_idx_259d.npy \
            >> "$LOGDIR/${arch}_s${seed}.log" 2>&1

        # Read test_pnl from results
        if [ -f "$rundir/results.json" ]; then
            pnl=$(python3 -c "import json; r=json.load(open('$rundir/results.json')); print(r.get('test_pnl', 0))" 2>/dev/null || echo "0")
            echo "=== DONE: $arch seed=$seed test_pnl=$pnl ===" | tee -a "$LOGDIR/${arch}_s${seed}.log"
            if [ "$seed" = "0" ]; then
                first_seed_pnl="$pnl"
                # Check if first seed is too bad (< 33)
                skip=$(python3 -c "print('yes' if float('$pnl') < 33.0 else 'no')" 2>/dev/null || echo "no")
                if [ "$skip" = "yes" ]; then
                    echo "[EARLY_SKIP] $arch seed0=$pnl < 33.0, skipping remaining seeds"
                    break
                fi
            fi
        else
            echo "=== FAILED: $arch seed=$seed ==="
        fi

        # Intermediate commit every arch after seed 2
        if [ "$seed" = "2" ]; then
            git add "runs_v5/${arch}_pruned259_s"*/results.json "logs_v5b/${arch}_s"*.log 2>/dev/null || true
            git commit -m "v5 Phase 2: ${arch} 3-seed done (pnl0=$first_seed_pnl)" 2>/dev/null || true
            git push origin master 2>/dev/null || true
        fi
    done
done

echo "=== Phase 2 sweep COMPLETE ==="
