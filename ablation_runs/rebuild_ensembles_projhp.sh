#!/bin/bash
# Rebuild row 12 (sym) + row 13 (asym) ensembles using:
#   - NN: ablation_runs/big_table/r10_nn_spo_359  (unchanged)
#   - LGB: ablation_runs/big_table_projhp/r08_lgb_l2_359_zscore_mirror_log1p  (new projhp)

set -euo pipefail
cd /root/projects/liangwenbei_workdir

OUT_BASE=ablation_runs/big_table_projhp
NN_DIR=ablation_runs/big_table/r10_nn_spo_359
LGB_DIR="$OUT_BASE/r08_lgb_l2_359_zscore_mirror_log1p"

# Row 12: sym EV gate
python3 ablation_runs/ensemble_ablation.py \
    --nn-dir "$NN_DIR" --lgb-dir "$LGB_DIR" \
    --out "$OUT_BASE/r12_nnspo_lgb_sym" \
    --w-nn 1.0 --w-lgb 1.5 --gate sym \
    --wandb-project liangwenbei-ablation-rerun \
    --wandb-name r12_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r12.log"

# Row 13: asym EV gate (DE)
python3 ablation_runs/ensemble_ablation.py \
    --nn-dir "$NN_DIR" --lgb-dir "$LGB_DIR" \
    --out "$OUT_BASE/r13_nnspo_lgb_asym" \
    --w-nn 1.0 --w-lgb 1.5 --gate asym \
    --wandb-project liangwenbei-ablation-rerun \
    --wandb-name r13_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r13.log"

echo "=== ENSEMBLE DONE ==="
grep -h "RESULT_LINE" "$OUT_BASE"/r12.log "$OUT_BASE"/r13.log
