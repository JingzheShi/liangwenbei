#!/usr/bin/env bash
# After single-seed sweep, take top-K tags and run remaining 4 seeds (1, 7, 13, 100) for each.
# Usage: bash run_topk_5seed.sh tag1 tag2 tag3 ...

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"
cd "$HERE/../.."

if [ "$#" -lt 1 ]; then
    echo "usage: $0 tag1 [tag2 ...]"
    exit 1
fi

# Read CLI args (tag-only -> use saved args from summary_<tag>_seed42.json)
for tag in "$@"; do
    summary="$HERE/summary_${tag}_seed42.json"
    if [ ! -f "$summary" ]; then
        echo "  ERR: $summary not found"; continue
    fi
    # extract args
    arch=$(python3 -c "import json;d=json.load(open('$summary'));print(d['arch'])")
    reg=$(python3 -c "import json;d=json.load(open('$summary'));print(d['reg'])")
    dropout=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['dropout'])")
    epochs=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['epochs'])")
    patience=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['patience'])")
    extra=""
    if [ "$arch" = "resnet" ]; then
        rh=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['resnet_hidden'])")
        rb=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['resnet_blocks'])")
        extra="--resnet-hidden $rh --resnet-blocks $rb"
    fi
    if [ "$arch" = "dcn_v2" ]; then
        nc=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['n_cross'])")
        dd=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['dcn_deep'])")
        extra="--n-cross $nc --dcn-deep $dd"
    fi
    if [ "$arch" = "wide_deep" ]; then
        wh=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['wd_hidden'])")
        extra="--wd-hidden $wh"
    fi
    if [ "$arch" = "geglu" ]; then
        gh=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['geglu_hidden'])")
        extra="--geglu-hidden $gh"
    fi
    if [ "$reg" = "mixup" ] || [ "$reg" = "mixup_swa" ]; then
        ma=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['mixup_alpha'])")
        extra="$extra --mixup-alpha $ma"
    fi
    if [ "$reg" = "swapnoise" ]; then
        sp=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['swap_p'])")
        extra="$extra --swap-p $sp"
    fi
    if [ "$reg" = "input_dropout" ]; then
        ip=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['input_dropout_p'])")
        extra="$extra --input-dropout-p $ip"
    fi
    if [ "$reg" = "swa" ] || [ "$reg" = "mixup_swa" ]; then
        ss=$(python3 -c "import json;d=json.load(open('$summary'));print(d['params']['swa_start'])")
        extra="$extra --swa-start $ss"
    fi

    for s in 1 7 13 100; do
        echo "=== $(date +%H:%M:%S) running $tag seed=$s arch=$arch reg=$reg ==="
        CUDA_VISIBLE_DEVICES=0 python3 experiments/T94_nn_arch_resnet_dcn/train_T94.py \
            --tag "$tag" --seed "$s" --no-wandb \
            --arch "$arch" --reg "$reg" --dropout "$dropout" \
            --epochs "$epochs" --patience "$patience" $extra \
            > "$HERE/logs/${tag}_seed${s}.log" 2>&1
        tail -4 "$HERE/logs/${tag}_seed${s}.log" | sed 's/^/  /'
    done
done

echo "=== topK 5-seed done $(date +%H:%M:%S) ==="
