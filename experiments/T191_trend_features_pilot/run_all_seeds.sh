#!/bin/bash
# T191: train 5 NN seeds + 5 LGB seeds on M-vast
set -e
SEEDS=(1 7 13 42 100)
LOG_DIR=/root/T191/logs
mkdir -p $LOG_DIR

echo "=== T191 training START $(date -Iseconds) ==="

# Verify env
nvidia-smi | head -10
python3 -c "import torch, lightgbm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('lgb', lightgbm.__version__)"

# Train NN seeds sequentially (single GPU)
for S in "${SEEDS[@]}"; do
  echo "=== NN seed $S $(date -Iseconds) ==="
  python3 /root/T191/scripts/train_T191_nn_seed.py --seed $S --cuda 0 2>&1 | tee $LOG_DIR/nn_seed${S}.log
done

# Train LGB seeds sequentially (also uses GPU; GPU LGB on RTX_3080)
for S in "${SEEDS[@]}"; do
  echo "=== LGB seed $S $(date -Iseconds) ==="
  python3 /root/T191/scripts/train_T191_lgb_seed.py --seed $S 2>&1 | tee $LOG_DIR/lgb_seed${S}.log
done

echo "=== T191 training DONE $(date -Iseconds) ==="
ls -la /root/T191/*.npz /root/T191/*.txt 2>/dev/null
