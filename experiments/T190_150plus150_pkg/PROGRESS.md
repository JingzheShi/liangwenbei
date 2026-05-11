# T190: 150 NN + 150 LGB Mega Ensemble Progress

## Status: TRAINING IN PROGRESS

Started: 2026-05-11 05:15 UTC

## Machine Assignment (4 active, machine 2 ssh9.vast.ai:11332 was DOWN)

| Machine | Seeds | Address | Status |
|---------|-------|---------|--------|
| M1 (lwb-r2-3080) | 51-75 | ssh1.vast.ai:10814 | ✅ Running |
| M3 (lwb-r4-3080) | 76-100 | ssh3.vast.ai:14818 | ✅ Running (slower, ~120s/LGB) |
| M4 (lwb-r21-evening) | 101-125 | ssh6.vast.ai:27152 | ✅ Running (fastest, ~45s/LGB) |
| M5 (lwb-r48-evening) | 126-150 | ssh3.vast.ai:27154 | ✅ Running (~77s/LGB) |

## Training Sequence (per machine)
1. Phase A: LGB seeds Sx-Sy (25 seeds × ~40-120s each = 17-50 min)
2. Phase B: NN seeds Sx-Sy (25 seeds × ~80s each = ~33 min)

## Extended HP Grid

### NN (seeds 51-150)
- 10 groups of 10 seeds each
- hidden variants: (256,128,64), (384,192,96), (320,160,80), (512,256,128), (192,96,48)
- weight_decay: 1e-5, 1e-4, 5e-4 (by group)
- lr_min: 1e-6, 1e-5 (by group)
- lr, dropout, batch: cycle within group

### LGB (seeds 51-150)
- 10 new HP configs (index 5-14)
- max_depth: 8, 12, 16
- min_data_in_leaf: 10, 50, 100, 200, 1000
- min_gain_to_split: 0.0, 0.001, 0.01
- lr: 0.03, 0.05, 0.1 (cycling by seed)
- num_boost_round: 330 (fixed M7 protocol)

## Key Architecture Decisions
- Predictor.py updated: _HeterogeneousBatchedEnsemble groups NNs by hidden architecture
  so BMM batching still works for each architecture group
- T188v2 thresholds inherited: thr_up=0.0003, thr_dn=0.000216, w_lgb=1.5
- Seeds 1-50: copied from T188v2_mega_proper (not retrained)

## Checkpoints
- 05:15 UTC: Training launched on all 4 machines
- 05:25 UTC: LGB progress — M1: 6 done, M3: 3 done, M4: 9 done, M5: 5 done
- ETA LGB complete: ~05:45-06:05 UTC
- ETA NN complete: ~06:30-06:45 UTC

## Files Created
- train_T190_nn_seed.py — extended NN training script (seeds 1-150)
- train_T190_lgb_seed.py — extended LGB training script (seeds 1-150)
- Predictor.py — modified for heterogeneous NN architectures
- thresholds.json — ensemble_seeds [1..150] for h=60
- requirements.txt — CPU-torch build (same as T188v3)
