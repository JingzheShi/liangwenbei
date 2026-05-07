#!/bin/bash
# T45 tau ablation: fold-2-only single-seed sweep over tau ∈ {0.2, 0.5, 1.0, 2.0}
set -e
cd /root/projects/liangwenbei_workdir

mkdir -p experiments/T45_group_dro/logs

for tau in 0.2 0.5 1.0 2.0; do
    tag="tau_$(echo $tau | sed 's/\./_/')_fold2"
    echo "=========================================="
    echo "Running tau=$tau tag=$tag (fold 2 only, seed 42)"
    echo "=========================================="
    python3 experiments/T45_group_dro/train_loso.py \
        --seeds 42 --syms 2 --tag "$tag" \
        --tau "$tau" --K 200 \
        --num-boost-round 600 --early-stopping 40 \
        2>&1 | tee experiments/T45_group_dro/logs/${tag}.log
done

echo
echo "===== TAU ABLATION DONE ====="
for tau in 0.2 0.5 1.0 2.0; do
    tag="tau_$(echo $tau | sed 's/\./_/')_fold2"
    echo "tau=$tau:"
    grep -E "HELD-OUT TEST|cum_pnl|accuracy" "experiments/T45_group_dro/logs/${tag}.log" | head -4
done
