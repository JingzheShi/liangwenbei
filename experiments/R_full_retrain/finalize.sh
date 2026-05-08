#!/bin/bash
# Finalize: copy full-retrain LGB models into both pkg dirs, zip.
set -e
cd /root/projects/liangwenbei_workdir/experiments/R_full_retrain

V1=pkg_iter019_v1
V2=pkg_iter019_v2

echo "=== Verify all 5 LGB models present ==="
for s in 1 7 13 42 100; do
  if [ ! -f "model_h60_seed${s}.txt" ]; then
    echo "MISSING: model_h60_seed${s}.txt"
    exit 1
  fi
done
ls -la model_h60_seed*.txt

echo ""
echo "=== Copy LGB models to v1 and v2 ==="
for s in 1 7 13 42 100; do
  cp -f model_h60_seed${s}.txt $V1/
  cp -f model_h60_seed${s}.txt $V2/
done

echo "v1 contents:"; ls $V1/
echo "v2 contents:"; ls $V2/

echo ""
echo "=== Zip ==="
DATESTAMP=050818
ZIP1=/root/projects/liangwenbei_workdir/submission_${DATESTAMP}_iter019_v1_fullretrain.zip
ZIP2=/root/projects/liangwenbei_workdir/submission_${DATESTAMP}_iter019_v2_fullretrain_conformal.zip
rm -f $ZIP1 $ZIP2
cd $V1 && zip -q -j $ZIP1 . && cd ..
cd $V2 && zip -q -j $ZIP2 . && cd ..
ls -la $ZIP1 $ZIP2
echo ""
echo "=== Verify zips ==="
unzip -l $ZIP1 | tail -25
echo "---"
unzip -l $ZIP2 | tail -25
