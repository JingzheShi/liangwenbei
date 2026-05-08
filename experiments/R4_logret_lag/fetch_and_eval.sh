#!/bin/bash
# Fetch all R4 prediction parquets from remote + run DE eval locally.
set -e
HERE=$(dirname $0)
cd $HERE
echo "=== fetch parquets from remote ==="
for SEED in 1 7 42; do
  for COND in baseline trick; do
    F=pred_R4_${COND}_seed${SEED}.parquet
    scp -q -i /root/.ssh/vastai_pm -o StrictHostKeyChecking=no -o ConnectTimeout=10 -P 14818 \
        root@ssh3.vast.ai:/root/lwb_work/$F ./$F
    ls -la ./$F
  done
done
echo "=== run DE eval ==="
python3 eval_de.py
