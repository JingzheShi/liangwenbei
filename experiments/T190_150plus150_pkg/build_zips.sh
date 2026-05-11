#!/bin/bash
# Build submission zip files for T190 after models are in T190_PKG.
# Creates both CUDA-torch and CPU-torch versions.

set -e
WORKDIR=/root/projects/liangwenbei_workdir
T190_PKG=$WORKDIR/experiments/T190_150plus150_pkg
OUT_DIR=$WORKDIR

LGB_TOTAL=$(ls $T190_PKG/model_h60_seed*.txt 2>/dev/null | wc -l)
NN_TOTAL=$(ls $T190_PKG/nn_h60_seed*.npz 2>/dev/null | wc -l)
echo "Building zips with $NN_TOTAL NN + $LGB_TOTAL LGB models"

if [ "$LGB_TOTAL" -lt 100 ] || [ "$NN_TOTAL" -lt 100 ]; then
  echo "WARNING: fewer than 100 of each model type — likely not all training is done"
fi

# === CPU-torch version (recommended: avoids 2GB CUDA download on platform) ===
ZIP_CPU="$OUT_DIR/submission_050912_iter019_v2N_150plus150_cputorch.zip"
echo "--- Building CPU-torch zip: $ZIP_CPU ---"
cd "$T190_PKG"
zip -q "$ZIP_CPU" \
  Predictor.py \
  fast_features.py \
  fast_features_batch.py \
  config.json \
  thresholds.json \
  requirements.txt \
  model_h60_seed*.txt \
  nn_h60_seed*.npz
echo "CPU zip size: $(du -sh $ZIP_CPU | cut -f1)"
echo "CPU zip MD5: $(md5sum $ZIP_CPU | cut -d' ' -f1)"

# === CUDA-torch version (with CUDA deps for platform GPU) ===
echo "--- Building CUDA-torch zip ---"
# Create temp requirements with cuda index
TEMP_REQ=$(mktemp /tmp/req_cuda_XXXXX.txt)
cat > "$TEMP_REQ" <<'REQEOF'
--extra-index-url https://download.pytorch.org/whl/cu124
numpy==2.4.4
pandas==2.3.3
lightgbm==4.6.0
scipy==1.17.1
torch==2.5.1+cu124
REQEOF
ZIP_CUDA="$OUT_DIR/submission_050912_iter019_v2N_150plus150_optimized.zip"
cd "$T190_PKG"
zip -q "$ZIP_CUDA" \
  Predictor.py \
  fast_features.py \
  fast_features_batch.py \
  config.json \
  thresholds.json \
  model_h60_seed*.txt \
  nn_h60_seed*.npz
# Add cuda requirements under correct name
zip "$ZIP_CUDA" -j "$TEMP_REQ" --junk-paths >/dev/null
cd "$(dirname $ZIP_CUDA)"
unzip -o "$ZIP_CUDA" "$(basename $TEMP_REQ)" -d /tmp/t190_temp/ >/dev/null 2>&1 || true
rm -f "$TEMP_REQ"
echo "CUDA zip size: $(du -sh $ZIP_CUDA | cut -f1)"
echo "CUDA zip MD5: $(md5sum $ZIP_CUDA | cut -d' ' -f1)"

echo ""
echo "=== Zip verification ==="
echo "CPU zip contents (model count):"
unzip -l "$ZIP_CPU" | grep "model_h60" | wc -l
unzip -l "$ZIP_CPU" | grep "nn_h60" | wc -l
echo "Files: Predictor.py fast_features.py fast_features_batch.py config.json thresholds.json requirements.txt"
unzip -l "$ZIP_CPU" | grep -E "Predictor|fast_feat|config|thresholds|requirements" | awk '{print $NF}'
