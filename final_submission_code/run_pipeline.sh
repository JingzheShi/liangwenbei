#!/bin/bash
# T188v2 50+50 Full-Horizon Submission Pipeline
# Usage: ./run_pipeline.sh [DATA_DIR] [CUDA_DEVICE] [LGB_GPU_FLAG]
# Default DATA_DIR = ./data  (put raw parquet files here)
# Default CUDA = 0
# Default LGB_GPU_FLAG = "--gpu" (matches original training; pass "" to use CPU LightGBM)
#
# End-to-end: raw parquets -> feature cache -> 50 LGB + 50 NN -> submission zip
set -e

DATA_DIR=${1:-./data}
CUDA=${2:-0}
LGB_GPU_FLAG=${3:---gpu}
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$WORKDIR/outputs"
CACHE_DIR="$OUT/cache"
MODELS_DIR="$OUT/models"
PKG_DIR="$OUT/pkg"
ZIP_OUT="$OUT/submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip"

echo "============================================"
echo "T188v2 50+50 Fullhorizon Pipeline"
echo "  DATA_DIR : $DATA_DIR"
echo "  CUDA     : $CUDA"
echo "  OUTPUT   : $OUT"
echo "============================================"
echo ""

mkdir -p "$OUT" "$CACHE_DIR" "$MODELS_DIR"

# -------------------------------------------------------
# Step 1: Build schemeP feature cache (370-d)
# -------------------------------------------------------
echo "=== Step 1: Build schemeP features (370-d) ==="
echo "    Input:  $DATA_DIR/*.parquet"
echo "    Output: $CACHE_DIR/schemeP_{train,val,test}.npz"
echo ""
python3 "$WORKDIR/01_build_features/build_schemeP_cache.py" \
    --data_dir "$DATA_DIR" \
    --out "$CACHE_DIR"
echo ""

# -------------------------------------------------------
# Step 2: Train 50 LGB models (5 HP configs x 10 seeds)
# -------------------------------------------------------
echo "=== Step 2: Train 50 LGB models ==="
echo "    M7 full-retrain on dates 0-119, num_boost_round=330"
echo "    ~30 min on CPU / ~12 min with GPU LightGBM"
echo ""
# Use --gpu flag if LightGBM GPU is available (optional)
bash "$WORKDIR/02_train_lgb/run_all_lgb_seeds.sh" "$CACHE_DIR" "$MODELS_DIR" "$LGB_GPU_FLAG"
echo ""

# -------------------------------------------------------
# Step 3: Train 50 NN models (T81 pretrain + T87 SPO+ M7)
# -------------------------------------------------------
echo "=== Step 3: Train 50 NN models (T81 pretrain + T87 SPO+ fine-tune) ==="
echo "    Phase 1: L2 pretrain on dates 0-79, val on 80-95, early stop"
echo "    Phase 2: SPO+ DFL fine-tune on dates 0-119, fixed 11 epochs"
echo "    ~90 min on single GPU / ~18 min on 5 GPUs in parallel"
echo ""
bash "$WORKDIR/03_train_nn/run_all_nn_seeds.sh" "$CACHE_DIR" "$MODELS_DIR" "$CUDA"
echo ""

# -------------------------------------------------------
# Step 4: Assemble package + zip
# -------------------------------------------------------
echo "=== Step 4: Assemble package + zip ==="
python3 "$WORKDIR/04_build_pkg/build_pkg.py" \
    --models "$MODELS_DIR" \
    --pkg "$PKG_DIR"

echo ""
echo "Creating submission zip..."
cd "$PKG_DIR" && zip -qr "$ZIP_OUT" .
cd "$WORKDIR"

echo ""
echo "============================================"
echo "DONE: $ZIP_OUT"
echo "Size: $(du -sh "$ZIP_OUT" | cut -f1)"
echo "MD5:  $(md5sum "$ZIP_OUT" | cut -d' ' -f1)"
echo "============================================"
