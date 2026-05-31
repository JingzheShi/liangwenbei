#!/usr/bin/env bash
# V6 sweep: 5 paradigm-shift archs × 3 seeds = 15 runs
# Each ~3 min on RTX 3090 → ~45 min total.
#
# Usage:
#   bash run_v6_sweep.sh                    # all archs, seeds 0-2
#   bash run_v6_sweep.sh v6_gat             # single arch, seeds 0-2
#   bash run_v6_sweep.sh v6_gat 3 4         # single arch, seeds 3-4 (extension)

set -e
cd "$(dirname "$0")"

mkdir -p logs_v6 runs_v6

KEEP_IDX="keep_idx_259d.npy"
CUDA="${CUDA_VISIBLE_DEVICES:-0}"

# Default arch list and seed list
if [ -z "$1" ]; then
    ARCHS=(v6_gat v6_pairwise v6_bottleneck v6_moe v6_crossfeat)
    SEEDS=(0 1 2)
else
    ARCHS=("$1")
    shift
    if [ -z "$1" ]; then
        SEEDS=(0 1 2)
    else
        SEEDS=("$@")
    fi
fi

# Per-arch dropout overrides (bigger archs get higher dropout)
declare -A DROPOUT_OVERRIDE
DROPOUT_OVERRIDE[v6_crossfeat]=0.35   # 236k params, +88% over baseline
DROPOUT_OVERRIDE[v6_moe]=0.32         # 186k params, +48% over baseline

echo "[sweep] archs=${ARCHS[*]} seeds=${SEEDS[*]} CUDA=$CUDA"

for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="logs_v6/${arch}_s${seed}.log"
        result="runs_v6/${arch}_pruned259_s${seed}/results.json"
        if [ -f "$result" ]; then
            echo "[skip] $result already exists"
            continue
        fi
        echo
        echo "===================================================================="
        echo "[run] arch=$arch seed=$seed log=$log"
        echo "===================================================================="
        extra_args=""
        if [ -n "${DROPOUT_OVERRIDE[$arch]}" ]; then
            extra_args="--dropout ${DROPOUT_OVERRIDE[$arch]}"
            echo "[opt] dropout override = ${DROPOUT_OVERRIDE[$arch]}"
        fi
        CUDA_VISIBLE_DEVICES=$CUDA python3 train_v6.py \
            --arch "$arch" --seed "$seed" \
            --keep-idx "$KEEP_IDX" \
            $extra_args \
            > "$log" 2>&1 || echo "[fail] $arch seed=$seed (see $log)"
        tail -3 "$log"
    done
done

echo
echo "[sweep] done."
