#!/bin/bash
set -e
PKG_DIR="$(dirname "$(realpath "$0")")"
WORKDIR="$(realpath "${PKG_DIR}/../../")"
ZIP_OUT="${WORKDIR}/submission_050818_iter019_v9_TOD.zip"

echo "=== Building iter_019 v9 TOD zip ==="
echo "PKG: $PKG_DIR"
echo "OUT: $ZIP_OUT"

# Verify all 5 models present
for S in 1 7 13 42 100; do
  M="${PKG_DIR}/model_h60_seed${S}.txt"
  if [ ! -f "$M" ]; then
    echo "ERROR: Missing $M"
    exit 1
  fi
  echo "  model_h60_seed${S}.txt  $(du -sh $M | cut -f1)"
done

# Build zip from pkg dir
rm -f "${ZIP_OUT}"
cd "${PKG_DIR}"
zip -r "${ZIP_OUT}" \
  Predictor.py \
  config.json \
  thresholds.json \
  requirements.txt \
  fast_features.py \
  fast_features_batch.py \
  model_h60_seed1.txt \
  model_h60_seed7.txt \
  model_h60_seed13.txt \
  model_h60_seed42.txt \
  model_h60_seed100.txt \
  nn_h60_seed1.npz \
  nn_h60_seed7.npz \
  nn_h60_seed13.npz \
  nn_h60_seed42.npz \
  nn_h60_seed100.npz

echo ""
echo "=== ZIP built ==="
echo "Size: $(du -sh ${ZIP_OUT} | cut -f1)"
echo "MD5:  $(md5sum ${ZIP_OUT})"
echo "Files:"
unzip -l "${ZIP_OUT}" | tail -n +4 | head -30
