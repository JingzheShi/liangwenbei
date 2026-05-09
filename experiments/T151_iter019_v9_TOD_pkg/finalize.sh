#!/bin/bash
set -e
T151_DIR="/root/projects/liangwenbei_workdir/experiments/T151_TOD_v9"
PKG_DIR="/root/projects/liangwenbei_workdir/experiments/T151_iter019_v9_TOD_pkg"
WORKDIR="/root/projects/liangwenbei_workdir"

echo "=== Finalizing iter_019 v9 TOD package ==="
echo "Current time: $(date)"

# Step 1: Copy remaining models
for S in 42 100; do
  SRC="${T151_DIR}/model_h60_seed${S}.txt"
  DST="${PKG_DIR}/model_h60_seed${S}.txt"
  if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
    cp "$SRC" "$DST"
    echo "  Copied seed $S model"
  elif [ -f "$DST" ]; then
    echo "  seed $S already in pkg"
  else
    echo "ERROR: $SRC not found!"
    exit 1
  fi
done

# Step 2: Run full smoke test
echo ""
echo "[Smoke test]"
cd "$PKG_DIR"
python3 smoke_test.py
echo "Smoke test PASSED"

# Step 3: Build zip
echo ""
echo "[Build zip]"
bash "$PKG_DIR/build_zip.sh"

# Step 4: Copy to output dir
mkdir -p /tmp/metabot-outputs/worker-f064911a/
cp "${WORKDIR}/submission_050818_iter019_v9_TOD.zip" /tmp/metabot-outputs/worker-f064911a/
echo "Copied to output dir"

# Step 5: Report
ZIP="${WORKDIR}/submission_050818_iter019_v9_TOD.zip"
MD5=$(md5sum "$ZIP" | cut -d' ' -f1)
SIZE=$(du -sh "$ZIP" | cut -f1)
echo ""
echo "=== DONE ==="
echo "ZIP: $ZIP"
echo "Size: $SIZE"
echo "MD5: $MD5"
