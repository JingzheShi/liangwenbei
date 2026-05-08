#!/usr/bin/env bash
# T94 single-seed architecture/regularization sweep, seed=42 only.
# Each run logs to logs/<tag>.log and saves model_<tag>_seed42.pt + pred_<tag>_seed42.parquet.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"
cd "$HERE/../.."

run() {
    local tag="$1"; shift
    echo "=== $(date +%H:%M:%S) running $tag ==="
    CUDA_VISIBLE_DEVICES=0 python3 experiments/T94_nn_arch_resnet_dcn/train_T94.py \
        --tag "$tag" --seed 42 --no-wandb \
        "$@" > "$HERE/logs/${tag}.log" 2>&1
    tail -6 "$HERE/logs/${tag}.log" | sed 's/^/  /'
}

# --- Architectures (all with default reg=none) ---
run mlp_med        --arch mlp_med    --dropout 0.15 --epochs 25 --patience 5
run mlp_wide       --arch mlp_wide   --dropout 0.20 --epochs 25 --patience 5
run mlp_xwide      --arch mlp_xwide  --dropout 0.25 --epochs 25 --patience 5
run mlp_deep       --arch mlp_deep   --dropout 0.20 --epochs 25 --patience 5
run resnet_3       --arch resnet     --resnet-hidden 512 --resnet-blocks 3 --dropout 0.20 --epochs 25 --patience 5
run resnet_5       --arch resnet     --resnet-hidden 512 --resnet-blocks 5 --dropout 0.20 --epochs 25 --patience 5
run dcn_v2_3       --arch dcn_v2     --n-cross 3 --dcn-deep "256,128,64" --dropout 0.15 --epochs 25 --patience 5
run dcn_v2_5       --arch dcn_v2     --n-cross 5 --dcn-deep "512,256,128" --dropout 0.20 --epochs 25 --patience 5
run wide_deep      --arch wide_deep  --wd-hidden "512,256,128" --dropout 0.15 --epochs 25 --patience 5
run geglu          --arch geglu      --geglu-hidden "512,256,128" --dropout 0.10 --epochs 25 --patience 5
run snn            --arch snn        --dropout 0.10 --epochs 25 --patience 5

# --- Regularization on mlp_med (medium-capacity baseline for fair comparison) ---
run mlp_med_mixup4    --arch mlp_med --reg mixup       --mixup-alpha 0.4 --dropout 0.10 --epochs 25 --patience 5
run mlp_med_swap15    --arch mlp_med --reg swapnoise   --swap-p 0.15 --dropout 0.10 --epochs 25 --patience 5
run mlp_med_indrop10  --arch mlp_med --reg input_dropout --input-dropout-p 0.10 --dropout 0.10 --epochs 25 --patience 5
run mlp_med_swa       --arch mlp_med --reg swa         --swa-start 8 --dropout 0.15 --epochs 20 --patience 100

echo "=== sweep done $(date +%H:%M:%S) ==="
