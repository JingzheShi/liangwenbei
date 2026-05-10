#!/bin/bash
# T184: Train 3 seeds, then eval + build zip

set -e
cd /root/projects/liangwenbei_workdir/experiments/T184_deep_CNN_raw

# Login WandB
wandb login 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 2>/dev/null || true

echo "=== T184: Train seed=1 ===" && python3 train_t184.py --seed 1 --cuda 0 2>&1 | tee logs/seed1.log
echo "=== T184: Train seed=42 ===" && python3 train_t184.py --seed 42 --cuda 0 2>&1 | tee logs/seed42.log
echo "=== T184: Train seed=100 ===" && python3 train_t184.py --seed 100 --cuda 0 2>&1 | tee logs/seed100.log

echo "=== T184: Eval + build zip ===" && python3 eval_t184.py 2>&1 | tee logs/eval.log

echo "DONE"
