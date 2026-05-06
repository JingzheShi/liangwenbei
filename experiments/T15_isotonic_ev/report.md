# T15: Isotonic Calibration + EV Gate (R13/R10) — NEGATIVE RESULT

**Task**: Apply isotonic calibration + EV-gate post-processing to iter_002 LOSO OOF, expecting +2-6 vs baseline (per R13 / R10).

**Result**: All variants (calibration alone, calibration + EV gate, calibration + re-tuned (T, δ)) **underperform iter_002 baseline**. Do **NOT** build iter_008.

## Setup

- Inputs: `experiments/T5b_features_multihorizon/loso_pred_h{H}_held{K}.parquet` (5 horizons × 5 LOSO folds, 88,416 rows × 5 = 442,080 rows per horizon)
- Calibration: per-horizon, one-vs-rest IsotonicRegression on combined OOF of held{j ≠ K}, applied to held{K} (no leakage). Renormalized so `sum(p_cal) = 1`.
- EV gate (sym-agnostic, fee-aware):
  ```
  EV(2) =  p2_cal * E_up - p0_cal * E_dn - 2*fee
  EV(0) = -p2_cal * E_up + p0_cal * E_dn - 2*fee
  EV(1) = 0
  pred = argmax{EV(0), EV(1), EV(2)} if max > C else 1
  ```
  E_up, E_dn estimated globally on 4-fold OOF (leak-safe).
- Conservatism C ∈ {0, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}.

## Calibration sanity (negative log-likelihood drops)

| Horizon | BCE original | BCE calibrated | mean_max_p orig | mean_max_p cal |
|---|---|---|---|---|
| h_5  | 0.8244 | **0.4863** | 0.609 | 0.794 |
| h_10 | 0.9468 | **0.6713** | 0.542 | 0.716 |
| h_20 | 0.9182 | **0.5686** | 0.535 | 0.788 |
| h_40 | 0.9979 | **0.7607** | 0.498 | 0.680 |
| h_60 | 1.0161 | **0.8538** | 0.481 | 0.619 |

Calibration is technically correct — log-likelihood improves substantially. Interesting fact: `mean_max_p_cal > mean_max_p_orig` across the board, meaning the LightGBM softmax is **under-confident on the dominant class**, not over-confident. (label=1 dominates ⇒ raw softmax shrinks p_1 toward equality with p_0/p_2.)

## Per-horizon LOSO PnL comparison

| Horizon | iter_002 (T, δ) | cal + same (T, δ) | cal + best EV gate | best EV C | Δ vs iter_002 |
|---|---|---|---|---|---|
| h_5  | **+17.49** | +8.20  | +16.94 | C=1e-5 | -0.55 |
| h_10 | **+21.86** | +12.13 | +20.84 | C=5e-5 | -1.02 |
| h_20 | **+19.70** | +4.79  | +19.45 | C=5e-5 | -0.25 |
| h_40 | **+11.71** | +2.50  | +7.58  | C=2e-4 | -4.13 |
| h_60 | **+6.30**  | +0.33  | +0.64  | C=5e-4 | -5.66 |
| **sum** | **+77.05** | +27.95 | +65.45 | — | **-11.60** |

5/5 positive folds maintained for h_5, h_10, h_20 in EV variant; h_40 drops to 4/5 and h_60 to 3/5 (same as baseline).

## Why does this NOT work?

1. **iter_002's (T, δ) was sweep-tuned on the same OOF**, so it's near-optimal for the raw LightGBM probabilities. Any monotonic transformation of the prob can be undone by re-tuning (T, δ) — but at best it matches, and in practice it loses.
2. **Re-tuning (T, δ) on calibrated probs** gave (T=0.45, δ=0.0) for h_10 → sum = +15.80 — still well below +21.86. So the loss is intrinsic to the post-cal prob distribution shape, not just threshold mismatch.
3. **EV gate uses global E[Δp | y]** estimates, which don't capture the fact that conditional on a high `p_2_cal`, the expected up-move is **larger than the unconditional mean E_up**. The EV decision underestimates the magnitude of high-confidence trades. A regression-based `E[Δp | features]` would be needed to fix this — out of scope here.
4. **Isotonic monotonicity** preserves rank order, but **renormalization** to a simplex is a non-monotonic transform of the (p_0, p_2) joint distribution. After cal, the side_max distribution is more concentrated at extremes (skewed by label=1 dominance), making (T, δ) gate less discriminative.

## Decision

- **Do NOT build iter_008.** Calibration + EV did not beat iter_002 on any horizon.
- **Keep iter_002 as the deployment artifact** (multi-horizon LightGBM Scheme C, unchanged thresholds).
- The calibration + EV idea **may still have value** for h_5 (gap is only -0.55) **if combined with additional features or a confidence-conditional E[Δp]** — flagged for future work.

## Artifacts

- `calibrate.py` — fit + apply isotonic per-fold + global deploy.
- `ev_gate.py` — full EV sweep + iter_002/cal_thresh comparison.
- `retune_cal_thresh.py` — sanity sweep of (T, δ) on calibrated probs.
- `loso_pred_h{H}_held{K}_cal.parquet` × 25 — calibrated OOF predictions (saved for downstream re-use).
- `isotonic_h{H}.pkl` × 5 — deploy isotonic (fit on all 5 folds), in case future iter wants to use them.
- `deploy_e_dp.json` — global E_up / E_dn per horizon (sym-agnostic, ready for deployment if EV is revisited).
- `calibration_summary.json` — BCE before/after.
- `ev_sweep_results.json` — full sweep.

## Compliance

- ✅ Sym-agnostic: isotonic and E_dp are global (no per-sym).
- ✅ No-leakage: per-fold isotonic / E_dp fit on `held{j ≠ K}`, applied to `held{K}`.
- ✅ No date / no test-order assumption: post-process is per-row only.
- ✅ Source models (`final_223_model_h{H}.txt`) untouched.
