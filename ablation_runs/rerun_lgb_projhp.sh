#!/bin/bash
# Rerun all 8 LGB rows + row 11 fulltrain with project HP_CONFIGS[0].
# HP defaults in train_lgb_ablation.py already match project: lr=0.05, num_leaves=127,
# min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0.
# Key diff vs. previous big_table runs: --rounds 330 (project max), --early-stop 200
# (effectively allows training to 330 rounds; ES rarely fires).

set -euo pipefail
cd /root/projects/liangwenbei_workdir

OUT_BASE=ablation_runs/big_table_projhp
mkdir -p "$OUT_BASE"

CACHE_RAW=ablation_runs/cache_raw154_rawamt
CACHE_P=ablation_runs/cache_log1p

COMMON="--rounds 330 --early-stop 200 --lr 0.05 --seed 1 --gpu --save-preds \
        --wandb-project liangwenbei-ablation-rerun"

# Row 1: 154 raw + LGB CE (multiclass)
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_RAW" --objective multiclass --feature-mode full \
    --out "$OUT_BASE/r01_lgb_ce_154raw" --wandb-name r01_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r01.log"

# Row 2: 154 raw + LGB regression L2
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_RAW" --objective regression_l2 --feature-mode full \
    --out "$OUT_BASE/r02_lgb_l2_154raw" --wandb-name r02_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r02.log"

# Row 3: 226-d SchemeC + LGB L2
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode schemeC72 \
    --out "$OUT_BASE/r03_lgb_l2_226c" --wandb-name r03_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r03.log"

# Row 4: 370-d SchemeP no drop
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode full \
    --out "$OUT_BASE/r04_lgb_l2_370nodrop" --wandb-name r04_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r04.log"

# Row 5: 359-d SchemeP drop11
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode drop11 \
    --out "$OUT_BASE/r05_lgb_l2_359drop" --wandb-name r05_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r05.log"

# Row 6: row 5 + window-z
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode drop11 \
    --window-z \
    --out "$OUT_BASE/r06_lgb_l2_359_zscore" --wandb-name r06_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r06.log"

# Row 7: row 6 + mirror flip
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode drop11 \
    --window-z --mirror-flip \
    --out "$OUT_BASE/r07_lgb_l2_359_zscore_mirror" --wandb-name r07_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r07.log"

# Row 8: row 7 + log1p rawlast (full pipeline)
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode drop11 \
    --window-z --mirror-flip --log1p-rawlast \
    --out "$OUT_BASE/r08_lgb_l2_359_zscore_mirror_log1p" --wandb-name r08_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r08.log"

# Row 11: row 8 + M7 fulltrain (train+val 0-95, fixed 330 rounds)
python3 ablation_runs/train_lgb_ablation.py $COMMON \
    --cache-dir "$CACHE_P" --objective regression_l2 --feature-mode drop11 \
    --window-z --mirror-flip --log1p-rawlast \
    --fulltrain --fulltrain-rounds 330 \
    --out "$OUT_BASE/r11_lgb_l2_359_fulltrain" --wandb-name r11_projhp_v2 \
    2>&1 | tee "$OUT_BASE/r11.log"

echo "=== ALL DONE ==="
grep -h "RESULT_LINE" "$OUT_BASE"/*.log
