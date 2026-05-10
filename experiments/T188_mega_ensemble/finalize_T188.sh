#!/bin/bash
# Finalize T188: SCP models from remote, build package, evaluate, zip
set -e

REMOTE="root@ssh1.vast.ai"
REMOTE_PORT=10814
SSH_KEY="$HOME/.ssh/vastai_pm"
REMOTE_DIR="/root/projects/liangwenbei_workdir/experiments/T188_mega_ensemble"
LOCAL_DIR="/root/projects/liangwenbei_workdir/experiments/T188_mega_ensemble"

echo "=== T188 Finalize ==="
echo "Started: $(date)"

# Check training status on remote
echo ""
echo "--- Checking remote training status ---"
ssh -p $REMOTE_PORT -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $REMOTE "
    cd $REMOTE_DIR &&
    NN_COUNT=\$(ls nn_h60_seed*.npz 2>/dev/null | wc -l) &&
    LGB_COUNT=\$(ls model_h60_seed*.txt 2>/dev/null | wc -l) &&
    echo \"NN: \$NN_COUNT / 50\" &&
    echo \"LGB: \$LGB_COUNT / 50\" &&
    tail -3 logs/run_all.log 2>/dev/null
" 2>/dev/null

echo ""
echo "--- SCPing NN models from remote ---"
# SCP all nn_h60_seed*.npz
scp -P $REMOTE_PORT -i $SSH_KEY -o ConnectTimeout=10 \
    "$REMOTE:$REMOTE_DIR/nn_h60_seed*.npz" \
    "$LOCAL_DIR/" 2>/dev/null || true
NN_LOCAL=$(ls "$LOCAL_DIR"/nn_h60_seed*.npz 2>/dev/null | wc -l)
echo "  NN files local: $NN_LOCAL"

echo ""
echo "--- SCPing LGB models from remote ---"
scp -P $REMOTE_PORT -i $SSH_KEY -o ConnectTimeout=10 \
    "$REMOTE:$REMOTE_DIR/model_h60_seed*.txt" \
    "$LOCAL_DIR/" 2>/dev/null || true
LGB_LOCAL=$(ls "$LOCAL_DIR"/model_h60_seed*.txt 2>/dev/null | wc -l)
echo "  LGB files local: $LGB_LOCAL"

echo ""
echo "--- Building package ---"
cd "$LOCAL_DIR"
python3 build_T188_pkg.py

echo ""
echo "=== FINALIZE DONE ==="
echo "Finished: $(date)"
