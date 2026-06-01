#!/bin/bash
# V13 sweep: 5 archs × 5 seeds = 25 runs. Each ~25 min on a 3090. Sequential.
set -u
cd /root/projects/liangwenbei_workdir/cross_sym_nn
mkdir -p logs_v13 runs_v13

# 4 archs with 60 features
for arch in v13_a_spo v13_b_gated v13_d_contrastive v13_e_multihorizon; do
  for seed in 0 1 2 3 4; do
    LOG="logs_v13/${arch}_s${seed}.log"
    echo "[$(date +%H:%M:%S)] launching ${arch} seed=${seed}" | tee -a logs_v13/sweep_main.log
    CUDA_VISIBLE_DEVICES=0 timeout 1800 python3 train_v13_arch.py --arch $arch --seed $seed \
      --keep-idx keep_idx_259d.npy --sel-idx selected_features_v9_60.npy \
      --zstats inter_zstats_v9.npz \
      --out-root runs_v13 --out-tag v13_${arch}_60feat \
      --no-wandb > "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] finished ${arch} seed=${seed} rc=$?" | tee -a logs_v13/sweep_main.log
  done
done

# V13.C uses 80 features
for seed in 0 1 2 3 4; do
  LOG="logs_v13/v13_c_80feat_s${seed}.log"
  echo "[$(date +%H:%M:%S)] launching v13_c_80feat seed=${seed}" | tee -a logs_v13/sweep_main.log
  CUDA_VISIBLE_DEVICES=0 timeout 1800 python3 train_v13_arch.py --arch v13_c_80feat --seed $seed \
    --keep-idx keep_idx_259d.npy --sel-idx selected_features_v9_80.npy \
    --zstats inter_zstats_v9.npz \
    --out-root runs_v13 --out-tag v13_c_80feat \
    --no-wandb > "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] finished v13_c_80feat seed=${seed} rc=$?" | tee -a logs_v13/sweep_main.log
done

echo "[v13 done $(date)]" | tee -a logs_v13/sweep_main.log
