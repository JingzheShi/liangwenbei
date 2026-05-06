# T35 — ReVol + Savitzky-Golay (Scheme K) — h_60 LOSO breakthrough attempt

> R31 Top 1 + Top 3 implementation: causal rolling normalization (Lee 2025, P17)
> + causal SG smooth/derivative features (Crypto LOB Better Inputs 2025, P10).
> Goal: push iter_005b's +11.46 LOSO h_60 to >+13 by enriching Scheme C 226d
> with 32 R31-style features → Scheme K 255d.

## TL;DR

| Variant | LOSO h_60 raw argmax | Best threshold sum | Per-fold (best thr) | Decision |
|---|---|---|---|---|
| **iter_002** Scheme C 226d, single seed | -12.14 | **+6.30** (T=0.50, d=0.20) | 0.41, 3.03, -1.41, -0.29, 4.56 | platform +4.07 |
| **iter_005b** Scheme C 223d + aug_a, 5-seed ens | -- | **+11.46** (T=0.45, d=0.10) | 1.30, 2.16, -0.28, 0.10, 8.18 | platform predicted +9.2 |
| **T35 Scheme K** 255d, single seed=42 | -8.52 (+3.62 vs iter_002) | **+5.53** (T=0.50, d=0.10) | 0.52, 3.06, -2.09, 0.06, 3.99 | -0.77 vs iter_002 |
| **T35 Scheme K** 255d + aug_a, 5-seed ens | _running_ | _TBD_ | _TBD_ | iter_006 ⇔ if >+12 |

**Single-seed verdict**: Scheme K's raw argmax improves over iter_002 by **+3.62**
(LightGBM is fitting to the new ReVol signal), but threshold-tuned PnL is **0.77
worse**. ReVol provides genuine signal (#1 feature, 12.6% gain share); SG provides
almost none for tree models (1.2% gain share, 0 in top-30).

**5-seed verdict**: pending. Will determine iter_006 build.

## What we built

### ReVol Normalization (`revol.py`) — 12 features
Causal log-return standardization (Lee 2025):
- W=64 rolling estimator on log((mid+1)/(mid_prev+1))
- For mid and wmp_lvl1, emit per output row:
  - `eps_last`, `eps_mean5`, `eps_mean20`, `eps_std20`, `eps_skew20`, `sigma_hat`
- Total 6 × 2 = **12 dims**
- Causal: μ̂, σ̂ at t use returns [t-W..t-1] only (`.shift(1)` after rolling)

### Savitzky-Golay (`savgol.py`) — 20 features
Causal SG via FIR weights at last position (window=9, polyorder=3):
- `savgol_coeffs(9, 3, pos=8)` for smooth + 1st & 2nd derivatives
- For mid, wmp_lvl1, imbalance, mlofi_W60_lvl1, spread1, emit:
  - `smooth`, `minus_raw`, `slope`, `accel`
- Total 4 × 5 = **20 dims**
- Strict causality verified by unit test (massive future shock at t=1500 leaves
  features at t<=1490 bit-identical).

### Scheme K cache (`build_features.py`)
- Reads raw parquets, computes 32 extras vectorised per-session
- Aligns row-by-row with Scheme C cache (asserted on save)
- Train: 1,473,600 × 32 (188 MB), Val: 294,720 × 32 (38 MB), Test: 442,080 × 32 (57 MB)
- Build time: ~14 s for all 3 splits

### Trainer (`train_loso.py`, `train_final.py`)
- Loads `schemeC_*.npz` + `schemeK_extra_*.npz`, drops Scheme C time tail (-3),
  concats: 223 base + 32 extras = **255 effective dims**
- LightGBM, sym-agnostic, no date / no sym features (verified)
- Same SEED_CONFIGS as T25/T27 for apples-to-apples comparison
- aug_a (per-(sample, feature) random scale [0.8, 1.2]) optional

## Results — single seed=42 baseline

LOSO h_60, 5 LOSO folds, `loso_summary_h60_baseline.json`:

```
seed=42 raw argmax: sum=-8.5229  per_fold=[+2.32, -4.01, -11.13, -0.02, +4.32]
                    iter_002:    [-0.67, -5.31, -10.25, -1.79, +5.88]  delta=[+2.99, +1.30, -0.88, +1.77, -1.56]
threshold sweep best (T=0.50, delta=0.10):
  sum_cum_pnl=+5.5330  pos_folds=4/5  per_fold=[+0.52, +3.06, -2.09, +0.06, +3.99]
                    iter_002 (T=0.50, d=0.20): +6.30  pos_folds=3/5
```

The **per-fold positivity** improved (4/5 vs 3/5) but the magnitude on the
strong fold (held=4) shrank, dropping the sum.

## Feature importance (mean gain over 5 LOSO folds, baseline seed=42)

```
Top 30:
  #1  [REVOL]  revol_wmp1_sigma_hat       gain=368784  (3.5x #2 — DOMINANT)
  #2  [ BASE]  totalasize                 gain=106458
  #3  [ BASE]  rv_w50                     gain= 98924
  #4  [ BASE]  bsize1                     gain= 93038
  ...
  #6  [REVOL]  revol_mid_sigma_hat        gain= 88362

In top-30:  2 ReVol, 0 SG, 28 base
Total gain shares (over all 255 feats):
  ReVol  12.6%
  SG      1.2%
  base   86.2%
```

**Interpretation**:
- ReVol's value is concentrated in `sigma_hat` (the realised vol estimator),
  not the standardised eps (which doesn't crack top-30). This matches an
  intuition that LightGBM already handles raw signed magnitudes, but lacks a
  clean "current vol regime" feature — and `sigma_hat` provides exactly that.
  A 12.6% gain share for 12/255 dims (4.7% of the budget) is a 2.7x-ROI.
- SG features are essentially dead weight (1.2% gain over 20/255 dims =
  7.8% of the budget — sub-1.0x). Tree-based models don't need pre-smoothing.
  R31 P10 was for **NN** inputs, where smoothing helps gradient flow; LightGBM
  doesn't see that benefit.

## Comparison with iter_005b's choice space

iter_005b moved from +6.30 → +11.46 by:
  (a) adding aug_a (per-(sample,feature) random scale) → single-seed +9.67
  (b) 5-seed ensemble → +11.46

Our Scheme K single-seed baseline (no aug) sits **0.77 below** iter_002. The
question is whether ReVol's gain in raw argmax (-12.14 → -8.52, a clear +3.62)
translates into a *threshold-tuned* uplift once we apply aug_a + 5-seed.

## 5-seed aug_a result

_(to be filled in by the running run; ~40 min CPU)_

## iter_006 build decision

- If Scheme K + aug_a + 5-seed ensemble best ≥ **+12**: build `iter_006_scheme_k_aug_a_5seed/`
  bundle, sanity 22/22, zip to `submission_050630_iter006.zip`. Predictor adds
  ReVol+SG inference (verified end-to-end smoke test passes).
- Otherwise: do not ship; report negative finding and use iter_005b for the
  next platform submission.

## Files

```
experiments/T35_revol_savgol/
  revol.py                      ReVol module (causal, vectorised)
  savgol.py                     SG module (causal FIR, vectorised)
  build_features.py             Build schemeK_extra_{train,val,test}.npz
  train_loso.py                 Scheme K LOSO training (per variant, per seed)
  train_final.py                Scheme K final-model training (train+val)
  threshold_sweep.py            T-delta grid sweep + per-seed and ensemble views
  feat_importance.py            Top-K gain importance + ReVol/SG count
  Predictor_iter_006.py         Inference Predictor (auto-detects per-h dim)
  build_iter_006.py             Stage submission/iter_006_scheme_k_aug_a_5seed/
  cache/
    schemeK_extra_{train,val,test}.npz   (N, 32) extras + alignment keys
    schemeK_extra_feat_names.txt
  loso_pred_h60_baseline_seed42_held*.parquet
  loso_model_h60_baseline_seed42_held*.txt
  loso_pred_h60_aug_a_seed*_held*.parquet     (in-flight)
  loso_model_h60_aug_a_seed*_held*.txt        (in-flight)
  loso_summary_*.json
  threshold_results_*.json
  feat_importance_h60_baseline_seed42.csv/.json
  results.json
  worker-progress.json
  train_loso_baseline_seed42.log
  train_loso_aug_a_5seed.log
```

## Constraint compliance (CRITICAL_CONSTRAINTS.md §1)

- ✅ ReVol & SG depend only on per-row 100-tick window (no `sym`, no `date`)
- ✅ All causal: `_causal_eps_and_sigma` uses `.shift(1)` after rolling; SG uses
     `savgol_coeffs(pos=last)` so the FIR taps only past values within the window.
     SG causality verified by unit test (future shock has zero impact on past
     features, max diff = 0.0).
- ✅ Inference and offline produce bit-identical features (max diff = 0.0 on
     synthetic data — verified)
- ✅ Trainer asserts `forbidden = {'date','sym','time'}` not in feat_names
- ✅ Predictor probes `booster.num_feature()` per horizon to slice base/extras
     correctly (h_40 = 223, h_60 = 255), no hard-coded dims
- ✅ Predictor stateless: per-call window only, no `self`-buffer
