#!/bin/bash
# Run ablation training: AF1, AF12, AF123 (AF1234 done separately with --save-models)
# A baseline already exists from T4.
set -e
cd /root/projects/liangwenbei_workdir

for VAR in AF1 AF12 AF123; do
    echo ""
    echo "############################################"
    echo "### Training variant $VAR ###"
    echo "############################################"
    python3 experiments/T9_schemeE/train_loso.py --features $VAR \
      2>&1 | tee experiments/T9_schemeE/logs/train_${VAR}.log
done
echo "ABLATIONS DONE"
