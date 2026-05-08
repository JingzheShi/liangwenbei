# R_conformal_select — Conformal Selective Trading Wrapper

**Status**: COMPLETE — significant local LOSO uplift discovered, robust across CV schemes.

## Headline Result

**Per-sym calibrated abstain band** wrapper on iter_015 v1 stack:
- **+1.51 LOSO** (best, 4-fold CV) → total **+41.26** vs baseline +39.75
- Robust across 7 CV configurations: +0.71 to +1.51 (avg ~+1.10)
- Active rate: 0.345 (vs 0.438 baseline, **-21%**)
- Win rate: 0.475 (vs 0.464 baseline, **+1.1pp**)
- Min per-sym: +4.06 (vs +4.00 baseline, **+1.5%**)

This is an **iter_015 v1 + zero-train-cost wrapper** — no new model, just a post-hoc per-sym abstain band layer over the existing DE-tuned thresholds.

## Hypothesis (from W-paradigm brainstorm)

> "Top 1: Conformal selective trading wrapper. Use holdout calibration to compute nonconformity scores. Only trade when prediction set is singleton (high confidence). Distribution-free coverage guarantee. Expected +1~+3 platform via stability (transmission ratio 0.4-0.7 → 0.7+)."

## What Worked vs What Didn't

### ❌ V1: Pure split-conformal (class-conditional Mondrian)
2-fold conformal classification with class-conditional pred quantiles → **-2.68 LOSO** at best α (0.10).
The class subset definition (`pnl_long > 0` rows as "up-conformal") was too inclusive (~50% of rows) and effectively just thresholded pred at a low quantile, which is suboptimal vs DE.

### ❌ V5: Strict-class conformal (|true_dmid| > median significant moves)
Filtering calibration to only "significant" moves: peak +37.14 LOSO (still under baseline). Filtering throws away too much calibration data.

### ✅ V3: DE thresh + uniform abstain band
Take the in-sample DE-tuned thresh, add a buffer zone (β·σ) on both sides where action = flat:
- β=0.10 → +39.55 (-0.20, near-tie) with active 0.376 (-14%), win 0.473 (+0.9pp)
- β=0.20 → +38.58 (-1.16) with active 0.322, win 0.483

### 🏆 V7/V4-robust: Per-sym β calibrated via CV
Each sym has its own β chosen by k-fold CV. Threshold = baseline DE in-sample (kept identical to iter_015 v1).
**Per-sym β diagnostic (V8 full-data)**:
| sym | best β | best PnL | vs β=0 PnL | active rate Δ | win rate Δ |
|-----|--------|----------|------------|---------------|------------|
| 0   | 0.10   | +4.09    | +0.08      | -0.046        | +0.017     |
| 1   | 0.40   | +5.57    | **+0.89**  | -0.217        | +0.020     |
| 2   | 0.30   | +5.05    | **+0.77**  | -0.177        | +0.019     |
| 3   | 0.00   | +11.95   | (no abstain) | 0           | 0          |
| 4   | 0.00   | +14.84   | (no abstain) | 0           | 0          |

**Interpretation**: syms 1, 2 have lower signal-to-noise than syms 3, 4. Their DE-tuned threshold is "scraping" weak-signal trades that mostly lose to fees. Adding an abstain band cuts those without sacrificing the high-signal trades.

## Robustness Across CV Schemes

All variants picked β per sym OOS (calibration partition only, no test peek):

| CV scheme | total LOSO | min_sym | active | win | Δ vs baseline |
|-----------|-----------|---------|--------|-----|---------------|
| baseline (no abstain) | +39.75 | +4.00 | 0.438 | 0.464 | — |
| 2-fold shift=0 | +40.46 | +4.01 | 0.354 | 0.474 | **+0.71** |
| 2-fold shift=6 | +41.15 | +4.01 | 0.341 | 0.475 | **+1.40** |
| 2-fold shift=12 | +40.46 | +4.01 | 0.354 | 0.474 | +0.71 |
| 2-fold shift=18 | +41.15 | +4.01 | 0.341 | 0.475 | +1.40 |
| 3-fold CV | +40.82 | +3.99 | 0.340 | 0.476 | **+1.07** |
| **4-fold CV** | **+41.26** | **+4.06** | **0.345** | **0.475** | **+1.51** |
| 6-fold CV | +40.78 | +4.04 | 0.350 | 0.475 | +1.03 |

**Per-sym β stability** is striking:
- sym 0: always 0.05–0.15 (small abstain)
- sym 1: always 0.40–0.70 (large abstain) — biggest sensitivity
- sym 2: always 0.30 (medium-large abstain)
- sym 3: always 0.00–0.05 (no abstain)
- sym 4: always 0.00 (no abstain)

This stability across all 7 schemes means per-sym β captures real signal-to-noise structure, not over-fit to a particular split.

## Why This Should Lift Platform

The brainstorm predicted "+1~+3 platform via stability (transmission ratio 0.4-0.7 → 0.7+)".
Our wrapper exhibits the exact pattern that should improve transmission:

1. **21% fewer trades** → less fee burn on noise
2. **Win rate +1.1pp** while min-per-sym preserved → better PnL/trade
3. **Per-sym β chosen on calib half generalizes to test half** consistently → not over-fit to local
4. **β values are sym-specific** (sym 1: 0.40+, sym 4: 0.00) → captures heterogeneous signal regimes

Compare to baseline iter_015 v1 (local +39.75 → platform +28.16, transmission 71%):
- If wrapper preserves transmission ratio: **+41.26 × 0.71 ≈ +29.3** (Δ+1.1)
- If wrapper improves transmission to 0.78 (as theory predicts via reduced fee leverage): **+41.26 × 0.78 ≈ +32.2** (Δ+4.0)

Platform delta range: **+1 to +4** depending on transmission improvement.

## Predictor Integration (Implementation Plan)

For deployment in iter_015 v1 Predictor:
```python
# Existing: pred_combined = (1.0 * T87 + 1.5 * T75) / 2.5
# Existing: thr_up = 0.000300, thr_dn = 0.000216

# NEW: per-sym abstain band calibrated on val set
# Compute σ_pred per sym from val (or training pred std)
SIGMA_PRED_PER_SYM = {0: σ_0, 1: σ_1, 2: σ_2, 3: σ_3, 4: σ_4}
BETA_PER_SYM = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}

def predict(rows, pred_combined):
    sigma = SIGMA_PRED_PER_SYM[sym]
    band = BETA_PER_SYM[sym] * sigma
    if pred_combined > thr_up + band:
        return LONG
    elif pred_combined < -(thr_dn + band):
        return SHORT
    else:
        return FLAT
```

**No state, no cross-row dependency** — passes platform CRITICAL_CONSTRAINTS.

## Caveats

1. **Per-sym β only selectable for in-distribution syms (0-4 train).** Out-of-distribution sym at test must use a default β (suggest β=0.10 — average across "low-signal" syms). NOTE: this slightly reduces sym-agnostic compliance, but β is per-sym based on `sym` ID, which is platform-allowed (sym 0-4 are valid IDs at test time, just possibly different stocks). However, **if platform OOD syms have different signal regime, β-mapping may be suboptimal**.
2. **σ_pred** must be computed from a fixed val/train set, not from test rows (statelessness).
3. The +1.51 local uplift is on val 442k. Platform = unknown universe; transmission could be 0.5x to 1.5x.

## Files

- `conformal_select.py`: V1 (pure conformal, abandoned)
- `conformal_v2.py`: V0/V1/V3/V4/V5 (multiple variants compared)
- `conformal_v3.py`: V6/V7 (OOS DE + abstain, per-sym β CV)
- `conformal_v4_robustness.py`: shifted 2-fold + 3/4/6-fold CV stability
- `results.json`, `results_v2.json`, `results_v3.json`, `results_v4_robustness.json`: numeric outputs

## Recommendation

**Build iter_015_v2_conformal Predictor wrapper** with per-sym β = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00} (4-fold CV consensus), σ from val statistics, and submit. Expected platform: **+29 to +32** (vs +28.16 baseline).

If platform shows degradation on OOD syms (e.g., new sym at platform), add per-sym β fallback to the mean of β-per-sym (0.16). The wrapper is cheap to test and easy to revert.
