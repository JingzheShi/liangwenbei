# T22 — WorldQuant Alpha101 + 国泰君安 Alpha191 (natural-fit subset)

## Executive summary

Faithful implementation of WorldQuant **Alpha101** (33 / 101 = 32.7%) and 国泰君安
**Alpha191** (124 / 191 = 64.9%) factor libraries, filtered to the natural-fit subset
per user requirement *"如果是用不了的就不要用了"*.

**Result: no iter_004**. Best LOSO h=10 sum across 3 schemes is **+21.80** (Scheme Ic),
which is `-0.06` vs iter_002 (+21.86) and `-0.42` vs iter_003 (+22.22). All schemes
fall under the +22.5 decision threshold for building iter_004.

The alpha factors, in their natural-fit single-stock time-series form, do **not** add
incremental predictive signal beyond the existing Scheme C 226d (raw 154 + T3 69
MLOFI/WMP/RV/EWMA features). LightGBM finds redundant overlap.

## Coverage

### Alpha101: 33 / 101 implemented (32.7%)

Implemented: 6, 9, 10, 12, 19, 21, 22, 23, 24, 26, 32, 35, 38, 41, 43, 44, 45, 46, 49,
51, 52, 53, 54, 55, 57, 71, 83, 84, 88, 92, 96, 99, 101.

The full IMPL/SKIP table for all 101 alphas is in `alpha101.py`'s top docstring.
Skipped reasons:
- **rank()** outer cross-section (44 alphas)
- **IndNeutralize** (22 alphas)
- **adv volume cross-section** (2 alphas)

A few "outer-rank" alphas were KEPT after dropping the outermost rank (e.g. Alpha38,
Alpha57, Alpha88) — only when the inner expression is purely single-stock time-series.

### Alpha191: 124 / 191 implemented (64.9%)

Implemented: 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 26, 27,
28, 29, 31, 33, 34, 38, 40, 42, 43, 44, 46, 47, 49, 50, 52, 53, 54, 57, 58, 59, 60, 62,
63, 65, 66, 67, 68, 70, 71, 72, 76, 78, 79, 80, 81, 82, 84, 85, 86, 88, 89, 93, 94, 95,
96, 97, 98, 100, 102, 103, 104, 106, 109, 110, 111, 112, 116, 117, 118, 122, 124, 126,
127, 128, 129, 132, 133, 134, 135, 137, 139, 144, 145, 146, 147, 150, 151, 152, 153,
155, 158, 159, 160, 161, 162, 163, 164, 166, 167, 168, 169, 171, 172, 173, 174, 175,
177, 178, 180, 186, 187, 188, 189, 191.

Full IMPL/SKIP table in `alpha191.py`'s top docstring.
Skipped reasons:
- **rank()** outer cross-section (~50 alphas)
- **benchmark index** dependency (7 alphas — #30, 69, 75, 149, 181, 182, 183)
- **other cross-section flag** (~10 alphas)

GTJA Alpha191 is more "natural-fit friendly" than WorldQuant Alpha101 because GTJA
designed many alphas as pure single-stock time-series operations
(`SMA`, `WMA`, `DELAY`, `DELTA`, `CORR(x,y,n)`, `TSRANK(x,n)`, `DECAY_LINEAR(x,n)`,
`SUMIF`, `COUNT`, `HIGHDAY`, `LOWDAY`).

## Schemes & feature dimensions

| Scheme | feat dim | composition |
|---|---|---|
| iter_002 baseline | 223 | raw 154 + T3 69 (no time enc) |
| **Scheme I_a (alpha-only)** | **157** | a101 33 + a191 124 |
| **Scheme I_b (alpha+raw)** | **380** | raw 154 + T3 69 + a101 33 + a191 124 |
| **Scheme I_c (curated)** | **247** | raw 154 + T3 69 + 24 top alphas (gain > 10k from Ib) |

## LOSO h=10 results (5-fold leave-one-sym-out, GPU LightGBM, single seed 42)

### Argmax baseline (no threshold)

| Scheme | fold0 | fold1 | fold2 | fold3 | fold4 | **sum** |
|---|---|---|---|---|---|---|
| Ia (alpha-only) | -3.73 | -6.24 | -4.80 | -8.45 | -1.29 | **-24.50** |
| Ib (alpha+raw 380d) | -4.10 | -0.33 | +3.11 | +2.16 | +6.88 | **+7.72** |
| Ic (curated 247d) | -3.55 | -0.23 | +2.94 | +2.12 | +6.73 | **+8.01** |

### Best (T, δ) threshold

| Scheme | best (T, δ) | fold0 | fold1 | fold2 | fold3 | fold4 | **sum** | n_active |
|---|---|---|---|---|---|---|---|---|
| iter_002 (Scheme C 223d, ref) | (0.50, 0.15) | +5.0 | +3.0 | +4.0 | +5.0 | +4.9 | **+21.86** | ~120k |
| iter_003 (5-seed Scheme C ensemble) | varies | — | — | — | — | — | **+22.22** | — |
| **T22 Ia (alpha-only)** | (0.55, 0.00) | +0.25 | -0.26 | +0.86 | +0.08 | +2.22 | **+3.16** | 18 975 |
| **T22 Ib (alpha+raw 380d)** | (0.55, 0.10) | +2.15 | +5.06 | +4.48 | +1.43 | +8.68 | **+21.79** | 89 996 |
| **T22 Ic (curated 247d)** | (0.50, 0.00) | +2.30 | +4.10 | +4.42 | +1.85 | +9.13 | **+21.80** | 119 818 |

### Comparison

| Scheme | LOSO h_10 sum | Δ vs iter_002 | Δ vs iter_003 | iter_004 ? |
|---|---|---|---|---|
| iter_002 (Scheme C 223d) | +21.86 | — | -0.36 | baseline |
| iter_003 (5-seed Scheme C ensemble) | +22.22 | +0.36 | — | current best |
| T22 Ia (alpha-only) | **+3.16** | -18.70 | -19.06 | **No** |
| T22 Ib (alpha+raw 380d) | **+21.79** | -0.07 | -0.43 | **No** |
| T22 Ic (curated 247d) | **+21.80** | -0.06 | -0.42 | **No** |

**Decision: no iter_004 from T22.** Best is `+21.80 < +22.5` threshold.

## Why alphas didn't help

1. **Cross-section ranks dropped.** Most useful WorldQuant alphas combine ts ops with
   cross-sectional `rank()` for portfolio neutralization. Without that step, the
   informational content of the inner expression alone is mostly already captured by
   the simpler raw + T3 features (mid-price, OFI, RV).
2. **High collinearity with T3 + raw.** Alpha191 features like #79, #67, #63, #102
   (RSI-style EMA of `max(close-delay(close,1), 0)`) are tightly correlated with
   existing OFI/intensity features. LightGBM picks one or the other but the total
   information is conserved.
3. **Numerical instability.** Several alphas produce extreme magnitudes (e.g. Alpha
   191 #10's `signedpower(stddev_squared, 2)`, Alpha 191 #54's `(low-close)*open^5 /
   (low-high)*close^5`) that interact poorly with LightGBM's GPU `max_bin=63`.

## Top alpha features by gain (from Ib)

The 24 highest-gain alphas (gain > 10 000) make up Ic. Top 10:

| rank | feat | gain | formula (Alpha191 reference) |
|---|---|---|---|
| #12 overall | a191_174 | 103 184 | sma_gtja(close>delay(close,1)?stddev(close,20):0, 20, 1) |
| #32 overall | a191_160 | 40 647 | sma_gtja(close<delay(close,1)?stddev(close,20):0, 20, 1) |
| #38 overall | a191_024 | 31 685 | sma_gtja(close - delay(close, 5), 5, 1) |
| #43 overall | a191_081 | 27 051 | sma_gtja(volume, 21, 2) |
| #44 overall | a191_188 | 26 851 | (high-low - sma_gtja(high-low,11,2)) / sma_gtja(...) * 100 |
| #46 overall | a191_132 | 22 626 | sma(amount, 20) |
| #49 overall | a191_068 | 19 833 | sma_gtja(((H+L)/2 - delay(H+L,1)/2)*(H-L)/V, 15, 2) |
| ... | ... | ... | (see `feature_importance_Ib.csv`) |

Top alphas are mostly **EMA-style smoothers** of close-direction-conditioned
volatility / volume — these provide marginal regularization but are largely
captured by existing `mlofi_W*`, `rv_w*`, `ewma_a*` features.

## Files

- `helpers.py` — vectorized time-series operators
  (`sma`, `stddev`, `correlation`, `decay_linear`, `ts_rank`, `wma`, `sma_gtja`,
  `regbeta`, `sumif`, `count_if`, `highday`, `lowday`, etc.)
- `alpha101.py` — 33 implemented alphas + IMPL/SKIP table for all 101
- `alpha191.py` — 124 implemented alphas + IMPL/SKIP table for all 191
- `build_features.py` — caches `schemeI_{train,val,test}.npz` (380 + 3 time enc cols)
- `loso_train.py --scheme {Ia, Ib, Ic}` — per-scheme 5-fold LOSO h=10 trainer
- `threshold_sweep.py --scheme X` — (T, δ) grid sweep per scheme
- `feature_importance_{Ia,Ib,Ic}.csv` — gain-based feature importance per scheme
- `loso_summary_{Ia,Ib,Ic}.json` — summary metrics
- `threshold_results_{Ia,Ib,Ic}.json` — threshold sweep best
- `curated_features.txt` — 247 features used in scheme Ic

## Hard constraints satisfied

- ✅ All features are sym-agnostic (no cross-sectional rank, no per-sym normalization).
- ✅ All features computed within 100-tick window (max lookback in alphas ≤ 50 ticks).
- ✅ No date/sym in feature list.
- ✅ Stateless within session (no cross-call buffers).

## Conclusion

The WorldQuant Alpha101 and 国泰君安 Alpha191 factor libraries, in their natural-fit
single-stock time-series form (i.e. cross-section operations dropped per user policy),
**do not exceed Scheme C's existing 226d feature set** for the 100-tick high-frequency
LOB prediction task. iter_002/iter_003 remain the best.

**Path forward**: future improvement should target either (a) per-stock or per-window
stochastic-state estimation that's sym-agnostic, (b) NN-based encoder over LOB
sequence rather than tabular last-tick, or (c) deeper window-temporal features beyond
the 100-tick budget if the platform allows. None of these are addressed by Alpha101/191.
