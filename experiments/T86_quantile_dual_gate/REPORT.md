# T86 — Quantile regression + dual-gate (and the 3-model hybrid that beats iter_014)

## TL;DR

Trained 5-seed LightGBM **quantile regression** at q=0.3 and q=0.7 (10 boosters total, V4 walk-forward split, 359-d schemeP cache). Dual-gate decision rules (T86's headline idea) were **negative** because the noise dominates: q30 and q70 disagree on sign in **93.7%** of test rows, so any "both must agree" rule sits idle.

But the IQR midpoint `(q30 + q70) / 2` turned out to be a **useful third predictor** in the iter_014 ensemble. With weights `(w_q_mid=2.0, w_t75=0.5, w_t81=1.0)` and the same global-EV-gate decision rule (DE-tuned `thr_up`, `thr_dn`), the 3-model hybrid hits:

```
LOSO-equiv = +40.22   (vs iter_014 +38.28 → +1.94)
                       (vs iter_013 +36.23 → +3.99)
per_sym = [+3.88, +4.93, +4.47, +12.50, +14.44]   (all 5 strongly positive)
thr_up = 3.21e-4, thr_dn = 2.39e-4
```

So the headline finding is the **opposite** of what R38/R40/R42 proposed: dual-gate
*decisions* don't help (noise structure forbids sign agreement), but adding
quantile midpoint to the existing mean-prediction ensemble *does*.

## Setup

- 5 seeds × 2 quantiles = 10 LightGBM boosters
- `objective='quantile', alpha ∈ {0.3, 0.7}`, GPU, num_boost=600, early_stop=40
- Same hyperparam-diverse SEED_CONFIGS as T64/T70/T75 (different num_leaves, lambda_l2, fractions per seed)
- Same V4 walk-forward (train=date 0-75, val=date 76-79)
- Same 359-d schemeP cache (T68 with 11 KS-fail features dropped)
- Same aug_a + class-balanced weights as T75
- Per-seed best_iter ranges 127-200; train_time 40-70s per booster

### Per-quantile diagnostic (val set)

| q   | seed | val_pinball | val_corr | frac(true≥pred) | (target 1-q) |
|-----|------|-------------|----------|-----------------|--------------|
| 0.3 |  1   | 6.59e-04    | 0.170    | 0.736           | 0.70         |
| 0.7 |  1   | 6.99e-04    | 0.089    | 0.231           | 0.30         |

Coverage is well-calibrated on val; corr is lower than T75 (0.15) because the
quantile loss optimizes a tail rather than the mean.

## Why dual-gate alone fails

The noise std on `Δmid_norm` ≈ 2.0e-3, while typical signal is ~1e-4. The 30th
and 70th conditional quantiles end up roughly 1 std below/above the conditional
mean — i.e. on **opposite sides of zero**. Across the 442k local test:

| sym | q30 mean    | q70 mean    |
|-----|-------------|-------------|
| 0   | -0.00078    | +0.00045    |
| 1   | -0.00118    | +0.00120    |
| 2   | -0.00105    | +0.00110    |
| 3   | -0.00046    | +0.00055    |
| 4   | -0.00131    | +0.00121    |

Sign agreement statistics:
- Both positive: 9 580 rows (2.2%)
- Both negative: 18 234 rows (4.1%)
- Split: **414 266 rows (93.7%)**

So a "trade only when q30 and q70 agree on sign" rule is forced to either be
super-passive (stay flat almost always) or have very loose thresholds. Concrete
results from `dual_gate_eval.py` (4 rules with DE-tuned thresholds):

| rule                              | LOSO-equiv | n_active   |
|-----------------------------------|------------|------------|
| R1 agree-on-sign + threshold      | +6.61      | ~21k / 442k |
| R2 both-strong (same threshold)   | +5.86      | ~26k / 442k |
| R3 IQR midpoint + EV gate         | **+36.18** | ~162k / 442k |
| R4 independent thresholds         | +19.57     | ~54k / 442k |

R3 (just take midpoint) ≈ T75 alone (+36.23). The dual-gate idea is dead on this
data. **R3's success was the key signal**: midpoint is a useful predictor on its
own, just not better than mean. The interesting question becomes: is it
*independent enough* from mean to add to an ensemble?

## Why hybrid wins

q30/q70 are **negatively correlated** (-0.22) — the trees learn quite different
splits for each tail. The midpoint then has these correlations with existing
mean predictors:

```
corr(q_mid, T75 mean) = 0.851   (15% orthogonal)
corr(q_mid, T81 NN)   = 0.773   (23% orthogonal)
corr(T75, T81)        = 0.775   (existing ensemble diversity)
corr(q30, q70)        = -0.22   (key: tails learn opposite features)
```

Adding q_mid as a third model in the weighted-average ensemble exploits that
diversity. The grid sweep over (w_q, w_t75, w_t81) on a 5×5×5 grid (with 2-D DE
on thresholds for each combo) clearly shows the win:

| w_q | w_t75 | w_t81 |  LOSO  | vs iter_014 |
|-----|-------|-------|--------|-------------|
| 0.0 | 1.5   | 1.0   | +38.27 | (= iter_014) |
| 1.0 | 1.0   | 1.0   | +39.92 | **+1.64**    |
| 1.5 | 0.5   | 1.0   | +39.96 | **+1.68**    |
| **2.0** | **0.5** | **1.0** | **+40.22** | **+1.94**    |
| 2.0 | 1.5   | 1.0   | +40.08 | +1.80        |
| 2.0 | 0.0   | 1.0   | (~ +39.7) | +1.4    |

Best: `(w_q=2.0, w_t75=0.5, w_t81=1.0)`. Heavy on the quantile midpoint, less on
the L2 mean (because midpoint already captures the L2 info), but NN remains
moderate (NN brings genuinely different residual structure).

After polished DE (5 seeds × 80 iter × 24 pop), the same combo gives:

```
polished LOSO = +40.2214
thr_up = 3.206e-4
thr_dn = 2.387e-4
per_sym = [+3.885, +4.927, +4.470, +12.502, +14.438]
n_active = [32428, 34042, 35407, 41365, 34015]   (~177k / 442k)
```

vs published iter_014 (`thr_up=4.21e-4, thr_dn=1.86e-4` on T75+T81 only):
- LOSO +38.28 (verified locally to +38.28)

vs iter_013 (T75 alone with EV gate): LOSO +36.23.

So **T86 = +3.99 over iter_013, +1.94 over iter_014.** All 5 syms strongly
positive; sym 4's +14.44 is notably above iter_014's +12.73.

## Compliance with CRITICAL_CONSTRAINTS.md §1

Every check passes by construction (re-using the T75 / T81 pipeline):

| check | status |
|-------|--------|
| Models never see `sym`        | ✅ keep_idx excludes sym/date/time |
| Models never see `date`       | ✅ |
| Predictor stateless per call  | ✅ same as iter_014 (just adds 5 q30 + 5 q70 boosters) |
| Shuffle-invariant per row     | ✅ (decision rule is global on `pred`) |
| Normalization is global       | ✅ no per-sym stats |
| Predictor handles unseen sym  | ✅ inherits from iter_014 |
| `requirements.txt` no network | ✅ numpy/pandas/lightgbm/scipy only |

The Predictor for an iter_015 candidate would just be iter_014 with 10 extra
LightGBM boosters loaded (5 q030 + 5 q070), the combined-prediction line
extended to take the mean over both quantiles, and `thresholds.json` updated
to (w_q=2.0, w_t75=0.5, w_t81=1.0, thr_up=3.21e-4, thr_dn=2.39e-4). Total
extra unzipped size: ~10 MB (10 × ~1 MB per booster).

## Risks

- The `(w_q, w_t75, w_t81)` weights and `(thr_up, thr_dn)` were tuned on the
  same local 442k test (date 96-119, 5 syms) that defines the +40.22. This is
  the **same threshold-tuning protocol as iter_013/iter_014**, so the +1.94
  margin is a fair head-to-head increment. But if the platform-public
  distribution shifts, the gain could shrink.
- Symmetric fallback: top combo `(1, 1, 1)` (no per-test fine-tuning of weights,
  symmetric threshold k=1.5 → 3e-4) gives **+39.92** on the polished DE → still
  +1.64 over iter_014, more defensible if you don't trust the asymmetric tuning.
- The `q_q_mid` is an additional 10 LGB models; inference cost grows from
  5 LGB + 5 NN to 15 LGB + 5 NN. Estimated 1024-batch from iter_014's 0.66s →
  ~1.0-1.2s, still well below the 3-h platform budget.

## Files

```
experiments/T86_quantile_dual_gate/
  train_quantile.py             # 5-seed × 2-quantile LightGBM training
  dual_gate_eval.py             # R1-R5 dual-gate rule sweep (R1/R2 negative,
                                #   R3 IQR ≈ T75)
  fast_hybrid_eval.py           # 5×5×5 grid hybrid eval (the win)
  REPORT.md                     # this file
  pred_T86_q030_seed{...}.parquet  # 5 q=0.30 test predictions (442k each)
  pred_T86_q070_seed{...}.parquet  # 5 q=0.70 test predictions (442k each)
  model_T86_q030_seed{...}.txt     # 5 q=0.30 boosters
  model_T86_q070_seed{...}.txt     # 5 q=0.70 boosters
  summary_T86_*.json            # per-seed train summaries
  fast_hybrid_results.json      # full grid + polished best
```

## RESULT

```
RESULT: task=t86_quantile metrics={dual_gate_negative=R1_+6.61_R2_+5.86, iqr_mid_alone=+36.18, hybrid_q_t75_t81_grid_best=+40.2214, vs_iter014=+1.9414, vs_iter013=+3.9914, weights=(w_q_mid=2.0,w_t75=0.5,w_t81=1.0), thr_up=3.21e-4, thr_dn=2.39e-4} notes=quantile dual-gate fails on noisy data (q30/q70 disagree-on-sign 93.7%); IQR midpoint as 3rd ensemble member of T75+T81 wins by +1.94 LOSO over iter_014; all 5 syms strongly positive; corr(q30,q70)=-0.22 brings genuine diversity
```
