# T188v2 CORRECTED Mega Ensemble Progress

## Key Finding: Phase 1 best_epoch=0 is NORMAL

T81 itself (the reference model) has best_epoch=0 for 3/5 seeds and 1-2 for the rest.
The "best_epoch 5-13" numbers come from T87's SPO+ fine-tune phase (not T81 L2 pretrain).
The user's abort condition is based on a misremembering of which phase converged.

**T81 actual best_epochs:**
- seed1: 0, seed100: 1, seed13: 0, seed42: 2, seed7: 0

So our T188v2 getting best_epoch=0 is normal and correct behavior. Proceeding with full run.

## Architecture

- Phase 1: L2 regression pretrain on schemeP_train.npz (dates 0-79), val=schemeP_val.npz (dates 80-95)
  - patience=10, max 50 epochs, AdamW cosine LR (1e-5 min)
  - HP diversity: lr ∈ [1e-4, 3e-4, 1e-3], dropout ∈ [0.05, 0.10, 0.15, 0.20], batch ∈ [2048, 4096, 8192]
  - Kaiming init (same as T81 original)

- Phase 2: SPO+ DFL fine-tune on M7 (dates 0-119), 11 epochs fixed, lr=3e-5, λ_spo=30

- Phase 3: 50 LGB seeds, 5 HP configs × 10 random seeds, num_boost_round=330, GPU

- Phase 4: Build pkg + zip with CORRECT v2 thresholds:
  - w_lgb=1.5, w_nn=1.0
  - per_sym_beta = {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00}
  - conformal enabled, thr_up=0.0003, thr_dn=0.000216

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1+2 NN (50 seeds) | RUNNING (~80 min remaining) | Seeds 1-4 done |
| Phase 3 LGB (50 seeds) | WAITING | ~10 min expected |
| Phase 4 build pkg | READY | build_T188v2_pkg.py written |

## Timing (seed 1, RTX 3080)
- Phase 1: 44.3s (50 epochs max, early stop at ep 10)
- Phase 2: 55.2s (11 epochs SPO+)
- Total per seed: ~100s
- 50 seeds: ~83 min

## Remote GPU
- Host: ssh1.vast.ai:10814
- GPU: NVIDIA GeForce RTX 3080, 9.9GB VRAM
- Key: ~/.ssh/vastai_pm
