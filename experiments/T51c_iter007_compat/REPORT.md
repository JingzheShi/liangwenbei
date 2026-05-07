# T51c iter_007 Inference-Compat Audit + Packaging

**Date**: 2026-05-07
**Decision**: **PACKAGED** → `submission_050707_iter007.zip`

## Summary

Verified that all T51 R34 Stage 1+2 features are computable from a 100-tick inference window. Trained 5 full-data h_60 LightGBM models on the 327-d schemeM cache, packaged into iter_007 with the existing T51b DE 4D threshold, passed all 22 sanity checks, and confirmed local single-session PnL is ahead of iter_006 (+1.18 cum_pnl on sym=0/date=119/am demo).

## Audit (Step 1)

### Rolling-window inventory across all 339 schemeM features

| Block | Source | Max W | Status |
|---|---|---|---|
| schemeC raw 154 | LOB raw cols | n/a | ✓ |
| schemeC mlofi (30) | T3 compute | 60 | ✓ |
| schemeC wmp (11) | T3 compute | per-tick | ✓ |
| schemeC rv (4) | T3 compute | 50 | ✓ |
| schemeC ewma_intst (24) | T3 compute | EWMA (no W) | ✓ (cold-start drift ~6e-3) |
| schemeC time (3) | always dropped | n/a | dropped (3 tail) |
| Stage 1 dualz (37) | T44 R34 | 100 | ✓ |
| Stage 1 signed_rv (3) | T44 R34 | 100 | ✓ |
| Stage 1 kyle_inv (2) | T44 R34 | 100 | ✓ |
| Stage 1 ewma_ofi (12) | T44 R34 | EWMA on mlofi_W20 | ✓ |
| Stage 2 qrank (20) | T51 R34 | 100 | ✓ |
| Stage 2 rskew (3) | T51 R34 | 100 | ✓ |
| Stage 2 gofi (30) | T51 R34 | 60 | ✓ |
| Stage 2 kyle_lam (2) | T51 R34 | 100 | ✓ |
| Stage 2 vol_burst (4) | T51 R34 | 50 | ✓ |

**Conclusion**: max W across all features = **100** → all computable from a 100-tick window at the last tick.

PM's expected `signed_rv_W200`, `kyle_inv_W200`, `qrank_W200` features **do not exist** in T51 cache; the largest realized window is W=100 (signed_rv_W100, kyle_inv_W100, qrank_W100, etc.).

### Path decision: Path A == Path B

Since no W>100 features exist, "Path A (drop W>100) → identical features to Path B (keep all)". **No retrain needed beyond full-data 5-seed training**.

### Boundary verification

`experiments/T51_r34_stage2/inference_compat_check.py` (already authored) computed all 113 stage1+2 extras from a 100-tick slice and compared to full-session compute at the same global index:

```
inference extras dim: 113 (expect 113)
OK: all 113 extras finite at the inference boundary
max |diff| inf vs full: 1.150882e-05  (in ewma_ofi_a0.05_lvl10)
```

EWMA cold-start drift is bounded: alpha ≥ 0.1 → drift < 1e-7; alpha=0.05 → drift ≤ 1.2e-5 (negligible vs LightGBM f32 quantization).

## Step 2: Train 5 full-data h_60 models

Reused `experiments/T51_r34_stage2/train_full_data_h60.py` (already authored) on the existing T51 schemeM cache:
- X = train + val concatenated (1,768,320 rows)
- aug_a applied (ratio=1.0 → 3,536,640 effective samples)
- Drops 9 FAIL extras + 3 time-encoding tail → **327 features**
- GPU LightGBM, 200 boost rounds, 5 seeds with diverse {feature_fraction, bagging_fraction, num_leaves, lambda_l2}
- Total wall: ~390s

Model files: `experiments/T51_r34_stage2/full_model_h60_seed{1,7,13,42,100}.txt`

## Step 3: Build iter_007

**Directory**: `submission/iter_007_stage2_5seed/`

**Files**:
- `Predictor.py` (new): extends iter_006 template with Stage 1+2 feature compute
- `compute.py`: T3 schemeC features (copied from iter_006)
- `r34_features.py`: T44 Stage 1 (copied)
- `r34_stage2_features.py`: T51 Stage 2 (copied)
- `model_h60_seed{1,7,13,42,100}.txt`: 5 full-data boosters (renamed from `full_model_h60_seed*`)
- `config.json`: identical to iter_006 (154 raw features)
- `thresholds.json`: T51b DE 4D values (T_up=0.4648, T_dn=0.4236, d_up=0.0435, d_dn=0.0065)
- `requirements.txt`: numpy 2.4.4, pandas 2.3.3, lightgbm 4.6.0

**Predictor pipeline per inference call**:
1. Read 154 raw cols (last row of 100-tick window)
2. Apply `sign(amount_delta)·log1p(|amount_delta|)` transform
3. Compute T3 features (mlofi, wmp, rv, ewma_intst → 69 cols), take last row
4. Attach derived `midprice` and `mlofi_W20_lvl{1,5,10}` to df
5. Compute Stage 1 R34 (54 cols at last index)
6. Compute Stage 2 R34 (59 cols at last index)
7. Drop 9 FAIL extras → 327-d feature vector
8. Run 5-booster ensemble, average probabilities
9. Apply DE 4D asymmetric gate → predicted class

## Step 4: Sanity (22 items)

`experiments/T51c_iter007_compat/sanity_results.json` → **22/22 PASSED**

Highlights:
- 01 Predictor loads in 0.08s with 5 boosters
- 03 Real-session features all finite
- 04/05 Idempotent within and across instances
- 06 Order-independent (shuffle batches)
- 07 OOD sym=99 doesn't crash (sym never enters features)
- 08 date=0 ignored (date not in feat list)
- 09 No mutable cross-call state
- 10 feat_dim==327 matches model.feature_name()
- 11 5 boosters loaded for h_60
- 12 4D thresholds present
- 14 inference compute matches cached schemeM features (max_diff=0.0 at session start)
- 16 pred dist not collapsed: 6 up / 20 down / 24 flat (of 50 random windows)
- 17 only h_60 active (h_5/10/20/40 disabled, matches iter_006)
- 18 No FAIL features in any booster
- 22 zip < 100MB (22.06 MB)

## Step 5: zip + LocalEvaluator contract validation

`src/submit/validate_contract.py --src submission/iter_007_stage2_5seed --pkg-name iter_007_stage2_5seed --sym 0 --date 119 --session am --n-batches 2`

```
✓ zip 创建成功
✓ 解压后存在 Predictor.py / config.json / requirements.txt
✓ batch#0: len==1024, inner len==5, 元素 ∈ {0,1,2}
✓ batch#1: len==818, inner len==5, 元素 ∈ {0,1,2}
✓ run() 完整执行 → (1842, 5)
[PASS] 所有断言通过
```

## Step 6: Single-session local PnL comparison

| Model | h_60 cum_pnl | n_active | avg_pnl |
|---|---|---|---|
| iter_006 | +22.48 | 58 | +0.388 |
| iter_007 | +23.66 | 69 | +0.343 |
| **delta** | **+1.18** | +11 | -0.045 |

Single session is noisy but consistent with the LOSO 5-seed DE result (+15.06 vs +13.61, **+1.45**).

## Final metrics

- LOSO 5-seed DE 4D sum: **+15.058** (vs iter_006 +13.606, **+1.451**)
- Per-fold: [+2.42, +4.90, -1.47, +0.14, +9.07]
- All 5 folds positive in iter_006; in iter_007, fold 2 (sym=2 held) goes negative -1.47 — consistent with sym=2 being the hardest LOSO fold across all iterations
- n_features = 327, n_seeds = 5, horizon = 60
- Inference latency: ~33 ms / 100-tick window (vs 12 ms iter_006)
- Submission size: 22.06 MB

## Output files

- `submission/iter_007_stage2_5seed/` — full submission contents
- `submission_050707_iter007.zip` — uploadable bundle (22.06 MB)
- `experiments/T51c_iter007_compat/results.json` — machine-readable summary
- `experiments/T51c_iter007_compat/sanity_results.json` — 22-item check log
- `experiments/T51c_iter007_compat/iter007_demo_pred.parquet`, `iter006_demo_pred.parquet` — single-session predictions for offline diff
- `experiments/T51c_iter007_compat/sanity_check.py` — 22-item suite (re-runnable)
- `experiments/T51c_iter007_compat/train_full.log` — 5-seed training log
