#!/bin/bash
# Syncs trained models from all 4 remote machines, copies existing 50+50,
# then builds the T190 pkg and zip submissions.
# Run AFTER training completes on all machines.

set -e
WORKDIR=/root/projects/liangwenbei_workdir
T190_PKG=$WORKDIR/experiments/T190_150plus150_pkg
T188_PKG=$WORKDIR/experiments/T188v3_optimized_pkg
T188V2_ORIG=$WORKDIR/experiments/T188v2_mega_proper/pkg_T188v2_50plus50

SSH_KEY=~/.ssh/vastai_pm
MACHINES=(
  "ssh1.vast.ai:10814:51:75"
  "ssh3.vast.ai:14818:76:100"
  "ssh6.vast.ai:27152:101:125"
  "ssh3.vast.ai:27154:126:150"
)

echo "=== Step 1: Sync models from remote machines ==="
for m in "${MACHINES[@]}"; do
  HOST=${m%%:*}; rest=${m#*:}
  PORT=${rest%%:*}; rest=${rest#*:}
  S_START=${rest%%:*}
  S_END=${rest#*:}
  echo "--- Syncing $HOST:$PORT seeds $S_START-$S_END ---"
  rsync -az -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT" \
    "root@$HOST:/root/T190/nn_h60_seed*.npz" \
    "root@$HOST:/root/T190/model_h60_seed*.txt" \
    "$T190_PKG/" 2>&1 || echo "WARNING: rsync failed for $HOST:$PORT"
done

echo ""
echo "=== Step 2: Count synced models ==="
LGB_COUNT=$(ls $T190_PKG/model_h60_seed*.txt 2>/dev/null | wc -l)
NN_COUNT=$(ls $T190_PKG/nn_h60_seed*.npz 2>/dev/null | wc -l)
echo "New models in T190_PKG: $NN_COUNT NN, $LGB_COUNT LGB"

echo ""
echo "=== Step 3: Copy existing 50+50 models from T188v2 (seeds 1-50) ==="
for s in $(seq 1 50); do
  nn_src="$T188_PKG/nn_h60_seed${s}.npz"
  lgb_src="$T188_PKG/model_h60_seed${s}.txt"
  if [ -f "$nn_src" ] && [ ! -f "$T190_PKG/nn_h60_seed${s}.npz" ]; then
    cp "$nn_src" "$T190_PKG/"
    echo "  Copied NN seed $s"
  fi
  if [ -f "$lgb_src" ] && [ ! -f "$T190_PKG/model_h60_seed${s}.txt" ]; then
    cp "$lgb_src" "$T190_PKG/"
    echo "  Copied LGB seed $s"
  fi
done

echo ""
echo "=== Step 4: Final model count ==="
LGB_TOTAL=$(ls $T190_PKG/model_h60_seed*.txt 2>/dev/null | wc -l)
NN_TOTAL=$(ls $T190_PKG/nn_h60_seed*.npz 2>/dev/null | wc -l)
echo "Total in T190_PKG: $NN_TOTAL NN, $LGB_TOTAL LGB"
echo "Expected: 150 NN, 150 LGB"

# Check for missing models
echo "Missing NN seeds:"; for s in $(seq 1 150); do
  [ ! -f "$T190_PKG/nn_h60_seed${s}.npz" ] && echo -n "$s "
done; echo ""
echo "Missing LGB seeds:"; for s in $(seq 1 150); do
  [ ! -f "$T190_PKG/model_h60_seed${s}.txt" ] && echo -n "$s "
done; echo ""

echo ""
echo "=== Done. Run eval_holdout_150plus150.py next, then build zips. ==="
