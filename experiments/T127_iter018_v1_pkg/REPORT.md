# T127 — iter_018 v1: iter_015 v1 + per-sym β-conformal abstain wrapper

**Status**: COMPLETE — package built, contract validated, verify confirms +41.49 LOSO-equiv (Δ +1.75 over iter_015 v1 baseline).

## Headline

- **Package**: `submission_050817_iter018_v1.zip` (5.6 MB, 16 files at top level)
- **Verify**: total LOSO-equiv = **+41.4939** (vs baseline +39.7467, **Δ +1.7473**)
- **Min per-sym**: +4.0855 (vs baseline +4.0039, +2.0%)
- **Active rate**: 0.345 (vs 0.438 baseline, **-21%** trades)
- **Win rate**: 0.475 (vs 0.464 baseline, **+1.1pp**)

Models are byte-identical to iter_015 v1 (5×T87 SPO+ NN + 5×T75 LGB regression). Only the gate has changed: the asymmetric EV gate now uses a per-sym **β·σ abstain band** layered on top of the DE-tuned thresholds.

## Architecture

| Component | Source | Status |
|---|---|---|
| 5×T87 SPO+ NN (h=60) | iter_015 v1 `nn_h60_seed{1,7,13,42,100}.npz` | byte-identical |
| 5×T75 LGB L2 (h=60) | iter_015 v1 `model_h60_seed{1,7,13,42,100}.txt` | byte-identical |
| Stack weights | w_nn=1.0, w_lgb=1.5 | iter_015 v1 same |
| DE thresh (h=60) | thr_up=2.9995e-4, thr_dn=2.1585e-4 | from R_conformal_select DE optimum (matches iter_015 v1 baseline +39.75) |
| **NEW: per-sym β·σ abstain band** | from R_conformal_select REPORT.md | β = {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00}; σ from local val 442k std(combined_pred) per sym |

## Wrapper logic

For each row at the last tick of the 100-tick window:
1. Look up `sym` from input DataFrame (NEW: `sym` added to `config.feature` so platform passes it)
2. Lookup `band = β_sym × σ_sym` (precomputed at build time, stored in `thresholds.json`)
3. Apply asymmetric gate with band:
   - `pred > thr_up + band` → action 2 (long)
   - `pred < -(thr_dn + band)` → action 0 (short)
   - else → action 1 (flat)

OOD safety: if sym ∉ {0..4} or `sym` column missing, fall back to `default_band = 0.16 × 4.0e-4 = 6.4e-5` (mean β × mean σ).

## Per-sym β values (from R_conformal_select V8 / 4-fold CV consensus)

| sym | β | σ_pred (val 442k) | abstain band β·σ | rationale |
|---|---|---|---|---|
| 0 | 0.10 | 2.40e-4 | 2.40e-5 | small abstain — tighter signal-to-noise but worth slight filtering |
| 1 | 0.40 | 4.71e-4 | 1.88e-4 | **biggest abstain** — DE thresh "scrapes" weak-signal trades |
| 2 | 0.30 | 4.52e-4 | 1.36e-4 | medium abstain — similar low-signal regime to sym 1 |
| 3 | 0.00 | 4.24e-4 | 0 | high-signal regime, no abstain (DE thresh already optimal) |
| 4 | 0.00 | 4.28e-4 | 0 | highest-signal regime, no abstain |

## Local 442k verification

Computed via `compute_sigma_and_verify.py` on cached T87 + T75 5-seed predictions:

```
=== VERIFY: per-sym β-band wrapper ===
  baseline (no abstain): total=+39.7467
  per-sym β-band wrapper: total=+41.4939
  delta: +1.7473
  per_sym_pnl: ['+4.0855', '+5.5662', '+5.0500', '+11.9498', '+14.8425']
  per_sym_active: ['0.251', '0.266', '0.293', '0.483', '0.460']
  per_sym_win: ['0.502', '0.539', '0.548', '0.231', '0.556']
  min_per_sym: +4.0855
```

Compared to R_conformal_select 4-fold CV result (+41.26): **+0.23 better** because we use V8 (full-data) β values directly rather than per-fold CV β. This is in-sample fit, so platform performance may be slightly lower (between 4-fold +41.26 and V8 +41.49).

## E2E spot-check

Ran the actual Predictor on 5 sessions × 50 windows from val (date 100, sym 0–4):

| sym | β | σ | band | active rate | comment |
|---|---|---|---|---|---|
| 0 | 0.10 | 2.40e-4 | 2.40e-5 | 18% | small filtering applied |
| 1 | 0.40 | 4.71e-4 | 1.88e-4 | 22% | strong filtering, vs ~48% baseline |
| 2 | 0.30 | 4.52e-4 | 1.36e-4 | 26% | strong filtering, vs ~47% baseline |
| 3 | 0.00 | 4.24e-4 | 0 | 54% | identical to baseline |
| 4 | 0.00 | 4.28e-4 | 0 | 58% | identical to baseline |

Active-rate reduction matches the wrapper's intended behavior: high-β syms abstain more.

OOD spot-checks:
- `sym=99` → uses default_band = 6.4e-5 (no error)
- `sym` column missing → uses default_band, no exception

## Compliance with CRITICAL_CONSTRAINTS

| Constraint | Status |
|---|---|
| 1. `date` not used as feature | ✓ |
| 2. No cross-call state on `self`; each `predict()` independent | ✓ |
| 3. sym not fed to model.forward (only used as lookup key for β/σ) | ✓ |
| 3. Fallback for OOD sym (∉ {0..4}): default β=0.16, default σ=4e-4 | ✓ |
| W ≤ 100 for every rolling | ✓ (inherited from iter_015 v1) |
| Stateless / shuffle-invariant | ✓ |

## Compatibility risk

**`sym` added to `config.feature`** — platform now passes `sym` column in input DataFrame. Per `submission/RULES.md` §3.5, the platform sends "列名 = config.feature 顺序严格", so listing sym there should make platform send it. We rely on:
1. Platform-side does emit the sym column when listed in feature.
2. sym IDs at platform are still 0..4 (per `data_schema.md` and CRITICAL_CONSTRAINTS).

If either fails, the OOD-default fallback ensures the Predictor still returns valid predictions (just with the OOD band ≈ 6.4e-5 ≈ middle of per-sym values).

## Expected platform PnL

Following the transmission analysis in `experiments/R_conformal_select/REPORT.md`:
- Baseline iter_015 v1 platform: +28.16 (transmission 0.71×)
- iter_018 expected: +41.49 × ~0.70 ≈ **+29 to +30** (Δ +1 to +2 vs iter_015 v1)
- If wrapper improves transmission (theory: less fee leverage, fewer noisy trades): up to **+32**

## Files

```
submission_050817_iter018_v1.zip   (5.6 MB, top-level flat)
├── Predictor.py                   (NEW: per-sym β-band wrapper)
├── fast_features.py               (iter_015 v1 same)
├── fast_features_batch.py         (iter_015 v1 same)
├── config.json                    (NEW: 'sym' added to feature list)
├── thresholds.json                (NEW: conformal_wrapper section)
├── requirements.txt               (numpy, pandas, lightgbm, scipy — no torch)
├── nn_h60_seed{1,7,13,42,100}.npz (iter_015 v1 byte-identical)
└── model_h60_seed{1,7,13,42,100}.txt (iter_015 v1 byte-identical)
```

## Reproducibility

- `compute_sigma_and_verify.py` — computes σ per sym, verifies wrapper PnL on cached pred
- `e2e_spot_check.py` — runs actual Predictor on real session windows, confirms gate logic
- `pkg/` — staged package contents before zipping
