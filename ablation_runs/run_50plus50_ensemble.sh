#!/bin/bash
# Run 50+50 ensemble (50 NN + 50 LGB), w_nn=1.0, w_lgb=1.5, asym gate.
set -euo pipefail
cd /root/projects/liangwenbei_workdir

NN_DIRS=()
LGB_DIRS=()
for s in $(seq 1 50); do
    NN_DIRS+=("ablation_runs/big50_nn/seed${s}")
    LGB_DIRS+=("ablation_runs/big50_lgb/seed${s}")
done

OUT=ablation_runs/big50_ensemble
mkdir -p "$OUT"

# sym
python3 ablation_runs/ensemble_55_ablation.py \
    --nn-dirs "${NN_DIRS[@]}" \
    --lgb-dirs "${LGB_DIRS[@]}" \
    --w-nn 1.0 --w-lgb 1.5 --gate sym \
    --out "$OUT/sym" \
    --wandb-name "ens_50plus50_sym" --wandb-project liangwenbei-50plus50-final \
    2>&1 | tee "$OUT/sym.log"

# asym
python3 ablation_runs/ensemble_55_ablation.py \
    --nn-dirs "${NN_DIRS[@]}" \
    --lgb-dirs "${LGB_DIRS[@]}" \
    --w-nn 1.0 --w-lgb 1.5 --gate asym \
    --out "$OUT/asym" \
    --wandb-name "ens_50plus50_asym" --wandb-project liangwenbei-50plus50-final \
    2>&1 | tee "$OUT/asym.log"

echo "=== DONE ==="
echo "sym val/test:"; grep -E 'val_pnl_h60|test_pnl_h60' "$OUT/sym/results.json"
echo "asym val/test:"; grep -E 'val_pnl_h60|test_pnl_h60' "$OUT/asym/results.json"
