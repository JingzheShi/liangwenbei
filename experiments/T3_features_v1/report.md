# T3 — Feature_v1 Module Report

**Worker**: T3-features-v1
**Date**: 2026-05-06
**Goal**: Implement the research short-list top-5 features as a reusable cache for downstream training workers.

## TL;DR

| Item | Value |
|---|---|
| New features added | **72** (30 MLOFI + 11 WMP + 4 RV + 24 EWMA + 3 time) |
| Sessions cached | **1200 / 1200** (5 sym × 120 date × 2 sess) |
| Cache total size | **1.54 GB** (`data/features_v1/`) |
| Build time (8 workers) | **10.4 s** |
| Per-session compute | **~22 ms** (single-process) |
| Unit tests passing | **8 / 8** |
| Top single-feature predictor | `ewma_a0.5_mb_intst` (Pearson r = +0.129 vs. label_5) |

---

## 1. Feature Definitions

### 1.1 MLOFI — Multi-Level Order Flow Imbalance (30 cols)

Following Cont-Stoikov-Kukanov 2014 + Kolm 2023. For level k = 1..10 at tick t:

```
e_k(t) = I(b_k_t ≥ b_k_{t-1}) · bs_k_t  - I(b_k_t ≤ b_k_{t-1}) · bs_k_{t-1}
       - I(a_k_t ≤ a_k_{t-1}) · as_k_t  + I(a_k_t ≥ a_k_{t-1}) · as_k_{t-1}
```

Then rolling-sum over W ∈ {5, 20, 60} ⇒ **`mlofi_W{W}_lvl{k}`**, 30 features.

**First W ticks NaN** by construction (shift + rolling warmup).

### 1.2 WMP — Weighted Mid-Price (11 cols)

Stoikov microprice. For level k = 1..10:

```
WMP_k = a_k · bs_k / (bs_k + as_k) + b_k · as_k / (bs_k + as_k)
```

Note bid is weighted by **ask** size (and vice versa) — imbalance pulls price toward the side with more pressure.

Fallback when `bs_k + as_k = 0`: `WMP_k = (a_k + b_k) / 2`.

Plus `wmp_balance_12 = wmp_lvl1 − wmp_lvl2` (Optiver Vol Forecasting standard).

### 1.3 RV — Multi-window Realized Volatility (4 cols)

```
log_return(t) = log((WMP1(t) + 1) / (WMP1(t-1) + 1))
RV_W(t) = sqrt( Σ_{s=t-W+1..t} log_return(s)² )
```

W ∈ {5, 10, 20, 50}. Uses WMP1 as the de-noised mid (smoother than `(bid1+ask1)/2`).

(`+1` because price fields are pct-changes ∈ (-1,1), so `safe = price+1 > 0`.)

### 1.4 EWMA Intensities — Hawkes proxy (24 cols)

For each `*_intst` column (lb / la / mb / ma / cb / ca) and α ∈ {0.05, 0.1, 0.3, 0.5}:

```
EWMA_α(intst)[t] = α · intst[t] + (1 − α) · EWMA_α[t-1]
```

Half-lives: α=0.05→14 ticks, α=0.1→7, α=0.3→2, α=0.5→1.4.

### 1.5 Time-in-session encoding (3 cols)

- `time_minutes_since_session_start`: int 0..99 (clipped at 99 for the 11:20:00 / 14:50:00 close)
- `time_session_progress`: float 0..1 (= minutes / 99)
- `time_is_pm`: 0 (AM) or 1 (PM)

---

## 2. Implementation Notes

### Vectorization & leakage

All operations are vectorized via `pd.shift / rolling / ewm`. **Zero Python `for` loops over rows.** Per-session compute is 22 ms.

The MLOFI uses `shift(1)` and rolling sums with `min_periods=W`, so feature at time t depends only on ticks ≤ t. Same for RV (shifts WMP1 once, rolling sum) and EWMA (causal by construction). **No look-ahead bias.**

### Cross-session isolation

Each `(sym, date, session)` parquet is processed independently — no rolling/EWMA state crosses files. Verified by `test_session_isolation`: EWMA[0] of each session equals the raw `intst[0]` (no carry-over).

### Numerical edge cases

- **Zero LOB depth at level k** (`bs_k + as_k = 0`) → WMP falls back to mid `(a_k + b_k) / 2`.
- **Floating-point fuzz in RV** — rolling sum of squared returns can produce ~1e-22 negatives when most returns are 0; clipped to `[0, ∞)` before sqrt.
- **No `±inf` anywhere** in the 72 outputs — verified by `test_no_inf_in_output`.
- **NaN policy**: only at the warmup head of MLOFI (≤60 ticks) and RV (≤50 ticks); 34 cols × 60 head rows ≈ 2040 NaN cells per session out of 469k, which downstream LGBM consumes natively and any NN trainer can mask via `fillna(0)`.

### Cache schema

`data/features_v1/snapshot_sym{X}_date{Y}_{am,pm}.parquet`:
- 163 original columns preserved (no field dropped)
- 72 new columns appended (in canonical order — see `feature_v1_columns()`)
- 235 cols × 2001 rows × 1200 files
- Snappy compression, ~1.28 MB / file, **1.54 GB total**

`data/features_v1/` is covered by `.gitignore` (matches `*.parquet` and `/data/`).

---

## 3. Unit Tests (8 / 8 passing)

| Test | Verifies |
|---|---|
| `test_mlofi_formula_toy` | Hand-computed 3-tick MLOFI matches; constant LOB → 0 after warmup |
| `test_wmp_fallback_on_zero_size` | (bs+as)=0 → (a+b)/2 fallback; balance_12 correct |
| `test_rv_non_negative` | All RV outputs ≥ 0; first W rows NaN |
| `test_ewma_converges_to_constant` | Constant input → EWMA tail within 1e-6 of constant |
| `test_time_encoding_range` | minutes ∈ [0, 99]; is_pm correct AM/PM; auto-detect works |
| `test_session_isolation` | Two distinct sessions: EWMA[0] = raw intst[0]; no state leak |
| `test_no_inf_in_output` | Stress with zero sizes → no inf; time encoding never NaN |
| `test_columns_order_and_count` | Exactly 72 cols matching `feature_v1_columns()` |

Run: `python experiments/T3_features_v1/test_compute.py`

---

## 4. Build Performance

Full sweep over 1200 parquets:

```
[T3-build_cache] 1200 files; workers=8; force=True
  [1200/1200] ok=1200 skip=0 fail=0 elapsed=10.4s
  out_dir size: 1464.9 MB
```

- **~8.7 ms / file** with 8 workers (vs. 22 ms single-process — good parallel scaling on I/O-bound merge+write workload).
- **Idempotent**: re-running without `--force` validates the schema and skips matching outputs.

---

## 5. Pearson Correlation — Top 20 Across All Labels

Computed on a stratified subsample of 30 sessions (60 030 rows post-warmup) per symbol-balanced sampling. Labels mapped 0/1/2 → −1/0/+1. Aggregate ranking by `max(|r|)` across the 5 horizon labels:

| feature | r_label_5 | r_label_10 | r_label_20 | r_label_40 | r_label_60 | max_|r| |
|---|---:|---:|---:|---:|---:|---:|
| `ewma_a0.5_mb_intst`  | **+0.1293** | +0.1080 | +0.0834 | +0.0655 | +0.0583 | 0.1293 |
| `ewma_a0.5_lb_intst`  | +0.1258 | **+0.1138** | +0.0796 | +0.0740 | +0.0600 | 0.1258 |
| `ewma_a0.3_mb_intst`  | +0.1205 | +0.1022 | +0.0735 | +0.0619 | +0.0564 | 0.1205 |
| `ewma_a0.3_lb_intst`  | +0.1202 | +0.1085 | +0.0726 | +0.0723 | +0.0594 | 0.1202 |
| `ewma_a0.5_la_intst`  | −0.0958 | −0.0779 | −0.0499 | −0.0199 | +0.0034 | 0.0958 |
| `ewma_a0.3_la_intst`  | −0.0864 | −0.0718 | −0.0428 | −0.0109 | +0.0154 | 0.0864 |
| `ewma_a0.1_lb_intst`  | +0.0807 | +0.0700 | +0.0444 | +0.0553 | +0.0527 | 0.0807 |
| `mlofi_W5_lvl1`       | +0.0792 | +0.0754 | +0.0500 | +0.0318 | +0.0174 | 0.0792 |
| `ewma_a0.1_mb_intst`  | +0.0763 | +0.0627 | +0.0364 | +0.0454 | +0.0481 | 0.0763 |
| `ewma_a0.5_ma_intst`  | −0.0723 | −0.0534 | −0.0299 | −0.0068 | +0.0005 | 0.0723 |
| `ewma_a0.05_ca_intst` | +0.0056 | +0.0101 | +0.0190 | +0.0390 | **+0.0636** | 0.0636 |
| `mlofi_W60_lvl8`      | −0.0079 | −0.0218 | −0.0461 | **−0.0618** | −0.0558 | 0.0618 |
| `mlofi_W20_lvl1`      | +0.0611 | +0.0453 | +0.0160 | +0.0074 | +0.0024 | 0.0611 |
| `mlofi_W60_lvl2`      | −0.0244 | −0.0340 | −0.0511 | −0.0610 | −0.0513 | 0.0610 |
| `ewma_a0.05_la_intst` | −0.0268 | −0.0202 | −0.0011 | +0.0394 | +0.0607 | 0.0607 |
| `ewma_a0.3_ma_intst`  | −0.0600 | −0.0436 | −0.0238 | +0.0007 | +0.0124 | 0.0600 |
| `mlofi_W60_lvl6`      | −0.0108 | −0.0218 | −0.0417 | −0.0591 | −0.0523 | 0.0591 |
| `mlofi_W20_lvl8`      | −0.0003 | −0.0113 | −0.0401 | −0.0550 | −0.0589 | 0.0589 |
| `mlofi_W20_lvl2`      | −0.0222 | −0.0303 | −0.0474 | −0.0562 | −0.0548 | 0.0562 |
| `mlofi_W20_lvl5`      | −0.0115 | −0.0258 | −0.0540 | −0.0525 | −0.0534 | 0.0540 |

**Per-label top-1**:
- label_5  → `ewma_a0.5_mb_intst` (r = +0.1293)
- label_10 → `ewma_a0.5_lb_intst` (r = +0.1138)
- label_20 → `ewma_a0.5_mb_intst` (r = +0.0834)
- label_40 → `ewma_a0.5_lb_intst` (r = +0.0740)
- label_60 → `ewma_a0.05_ca_intst` (r = +0.0636)

### Interpretation

1. **Buy-side intensities (lb, mb) dominate short-horizon up-moves.** The fastest EWMA (α=0.5) of market-buy intensity is the single best predictor of label_5 (3% correlation gain over raw `mb_intst`).
2. **Mirror symmetry on the sell side.** `la_intst` and `ma_intst` EWMAs flip sign — strong negative predictors of label_5/10.
3. **Long-horizon predictability shifts to slow signals.** label_60's top features are slow-EWMA cancel-ask (`ewma_a0.05_ca_intst`) and deep-book MLOFI (`mlofi_W60_lvl8`, lvl 6, lvl 2). Deep-level negative MLOFI ⇒ sellers shrinking on the deep ask side ⇒ supply contraction ⇒ price up — economically sensible.
4. **MLOFI W=5 lvl1 is the best non-EWMA signal** for label_5/10 (r ≈ +0.08), confirming the Cont-Stoikov-Kukanov insight that level-1 OFI dominates short-horizon information flow.
5. **WMP and RV did not crack top 20.** Pearson r is weak in raw form because both quantities are heavily contemporaneous with the label denominator (price level / volatility); their value to the model will come through interactions and lagged differences, which downstream feature engineering or NN architectures can exploit.

Per-label CSVs: `pearson_top20_label_{5,10,20,40,60}.csv`. Aggregate: `pearson_top20_aggregate.csv`.

---

## 6. Sample-Session Visualisations

Sample: `sym=0, date=0, am`. All four panels stored under `experiments/T3_features_v1/figures/`.

### `mlofi.png`
- **Top**: MLOFI W=20 across levels 1-3 oscillates around 0 — economically required (no persistent imbalance).
- **Bottom**: MLOFI level 1 with W ∈ {5, 20, 60} — longer W is visibly smoother (window-averaging confirmed).

### `wmp.png`
- **Top**: WMP_lvl1 / WMP_lvl2 closely track the official `midprice`, deviating modestly when bid/ask sizes are imbalanced. 20 sample-tick markers overlaid.
- **Bottom**: `wmp_balance_12` oscillates around 0 with brief excursions during quote-shift events — the Optiver-style microprice slope.

### `rv.png`
- All four `rv_w{5,10,20,50}` rise sharply during turbulent windows (t ≈ 600 spike) and decay smoothly otherwise. Short W (rv_w5) is reactive; long W (rv_w50) is steady-state.

### `ewma_intst.png`
- **Top**: `lb_intst` raw vs. four EWMAs — α=0.05 is visibly smoothest, α=0.5 tracks raw closely.
- **Bottom**: Cross-class comparison at α=0.5 (lb / la / mb / ma) showing the four order-class intensity tracks evolve together but with class-specific spikes around news ticks.

---

## 7. Files Produced

```
experiments/T3_features_v1/
├── compute.py                       # 5 feature classes + compute_all
├── test_compute.py                  # 8 unit tests
├── build_cache.py                   # 1200-file batch builder
├── analyze.py                       # Pearson correlation + figures
├── report.md                        # this file
├── pearson_top20_label_5.csv
├── pearson_top20_label_10.csv
├── pearson_top20_label_20.csv
├── pearson_top20_label_40.csv
├── pearson_top20_label_60.csv
├── pearson_top20_aggregate.csv
├── build.log
└── figures/
    ├── mlofi.png
    ├── wmp.png
    ├── rv.png
    └── ewma_intst.png
```

`data/features_v1/snapshot_sym{X}_date{Y}_{am,pm}.parquet` — 1200 cached parquets (1.54 GB, gitignored).

---

## 8. How Downstream Workers Should Consume

```python
import pandas as pd
df = pd.read_parquet("data/features_v1/snapshot_sym0_date0_am.parquet")
# 235 cols total: 163 official + 72 new
# label cols still at the end (label_5..label_60)

# Filter to feature cols (excluding date/sym/time/midprice/labels):
from experiments.T3_features_v1.compute import feature_v1_columns
new_feats = feature_v1_columns()  # 72 names
```

For training:
- **LGBM**: feed all 235 cols minus excluded; LGBM handles NaN natively.
- **NN**: warmup 60 rows of MLOFI/RV are NaN — either skip those samples (LOBDataset already does via `window` arg) or `fillna(0)` after standardization.
- **Caution**: `time_minutes_since_session_start` is int — embed it, don't z-score.

Recommend creating `experiments/T3_features_v1/dataset_v1.py` (LOBDatasetV1) for downstream NN workers; not in scope for this T3.

---

## 9. Acceptance Criteria — All Met

- [x] 5 feature modules implemented; **8/8 unit tests passing**
- [x] 1200 / 1200 cached files in `data/features_v1/`
- [x] Sample-session figures show economically reasonable trajectories (oscillating MLOFI, WMP near mid, RV spikes during turbulence, EWMA monotone-smoother as α↓)
- [x] Pearson top-20 reported with interpretation
- [x] Code committed (data files gitignored)
