# T11 — Scheme C × 5-seed ensemble (h=10) → **iter_003**

**Status**: complete — submission package built and 22/22 ✓
**Run window**: 2026-05-06 07:05 – 08:32 UTC (~1.5 h)

## Goal
Build a multi-seed ensemble on top of the best Scheme C horizon (h_10, +21.86
LOSO OOF in iter_002). Force diversity across seeds via different
(`feature_fraction`, `bagging_fraction`, `num_leaves`, `lambda_l2`) — T6b
confirmed that same-hyperparam multi-seed gives no improvement (low diversity).

## Method

### LOSO setup (matches T5b / iter_002)
- Train: (sym ≠ held, dates 0..79)
- Val:   (sym ≠ held, dates 80..95) — early stopping
- Test:  (sym = held, dates 96..119) — held-out / OOF source

### Feature space
**223-d** Scheme C (T2 raw 154 + T3 69, dropping 3 time-encoding cols) — same
as iter_002. Cache: `experiments/T5b_features_multihorizon/cache/schemeC_*.npz`.

### Diversity-forcing seed configs
| seed | feature_fraction | bagging_fraction | num_leaves | lambda_l2 |
|---:|---:|---:|---:|---:|
| 42  | 0.80 | 0.80 | 127 | 1.0 |
| 1   | 0.60 | 0.70 | 127 | 1.0 |
| 7   | 0.70 | 0.85 |  63 | 1.0 |
| 13  | 0.60 | 0.70 | 127 | 2.0 |
| 100 | 0.50 | 0.60 | 255 | 1.0 |

Common: lr=0.05, min_data_in_leaf=100, bagging_freq=5, num_threads=18,
num_boost_round=600 + early_stopping=40, class-balanced sample weights.

### Aggregation
Per (h, fold), arithmetic mean of `prob_0/1/2` across the 5 seed boosters.

### Threshold sweep (per horizon)
Grid: T ∈ {0.35, 0.40, …, 0.70} (8) × δ ∈ {0.00, 0.05, …, 0.25} (6).
Decision rule: `side_max = max(p0,p2); take iff (side_max ≥ T) ∧ (side_max > p1+δ); else hold`.

## Results

### LOSO h_10 — ensemble vs single-seed vs iter_002

| Variant | (T, δ) | Sum cum_pnl | n_pos_folds | Per-fold PnL |
|---|---|---|---|---|
| iter_002 single-model (seed 42 from T5b) | (0.55, 0.10) | +21.8595 | 5/5 | (1.93, 5.03, 4.33, 1.77, 8.65) |
| seed 42 alone | (0.55, 0.15) | +21.7153 | 5/5 | (1.93, 5.03, 4.33, 1.77, 8.65) |
| seed 1  alone | (0.55, 0.10) | +21.8716 | 5/5 | (1.94, 5.02, 4.37, 1.76, 8.79) |
| seed 7  alone | (0.55, 0.10) | +22.1012 | 5/5 | (2.09, 5.15, 4.35, 1.78, 8.72) |
| seed 13 alone | (0.55, 0.10) | +21.8597 | 5/5 | (1.98, 5.01, 4.36, 1.91, 8.60) |
| seed 100 alone | (0.55, 0.10) | +21.7249 | 5/5 | (1.90, 5.12, 4.30, 1.90, 8.50) |
| **5-seed ensemble** | **(0.55, 0.10)** | **+22.2241** | **5/5** | **(2.07, 5.19, 4.37, 1.79, 8.80)** |

- **Ensemble vs best single seed (seed 7)**: +0.123 uplift (real but modest)
- **Ensemble vs iter_002**: +0.365 uplift, all 5 folds positive

Argmax baseline (no threshold):
- Ensemble argmax = +6.7404 vs best single seed = +7.2198 → averaging actually
  *reduces* raw argmax PnL slightly; the gain comes from threshold-rule
  interaction with smoother probabilities.

## Final-model training (for iter_003)

Recipe identical to T5b's iter_002 final train: train on (train + val) full data
(all 5 syms, dates 0..95 = 1.77M rows × 223-d), with the held-out test set
(dates 96..119, 442k rows) used as early-stopping validation. Same 5 seed
configs as LOSO.

| seed | best_iter | train_time |
|---:|---:|---:|
| 42  | 245 | 158.5 s |
| 1   | 233 | 125.6 s |
| 7   | 279 | 134.7 s |
| 13  | 238 | 127.9 s |
| 100 | 178 | 118.5 s |

Total: 5 boosters in ~11 min. Saved as `final_223_model_h10_seed{S}.txt`.

## Submission package — iter_003

`submission/iter_003_lgbm_schemeC_5seed/`:
- `Predictor.py` — supports `seeds` field per horizon (ensemble) OR single model
- `compute.py`, `config.json`, `requirements.txt` — copied from iter_002
- `model_h{5,20,40,60}.txt` — copied from iter_002 (verbatim)
- `model_h10_seed{42,1,7,13,100}.txt` — 5 final 223-d boosters
- `thresholds.json` — h_10 (T=0.55, δ=0.10) + iter_002 thresholds for others

**Validation**:
- `submission/scripts/sanity_check.py`: **22/22 ✓ 0 fail 0 warn**
- `submission/scripts/prepare.py` (validate_contract end-to-end): all asserts pass
  with shuffle=True, anonymize=True
- Zip: 34.59 MB, 14 files, no nested dirs

`submission_050614_iter003.zip` copied to workdir root.

## Verdict

**iter_003 ships.** OOF uplift over iter_002 is small (+0.36 / +1.7%) but all 5
LOSO folds are positive, so the ensemble strictly dominates iter_002 on every
held-out sym. The expected leaderboard delta is modest but the risk is low.

The companion T16 experiment (T11+T14 mixup 6-model ensemble) was tested and
**rejected**: 6-model OOF best = +22.2454, only +0.0213 over T11-only — does not
clear the user-specified +22.5 gate. T11 5-seed already saturates seed-driven
diversity. See `experiments/T16_combine_T11_T14/report.md`.

## Files

```
experiments/T11_schemeC_multiseed/
├── loso_train.py                            (LOSO train script — 5 seeds × 5 folds × 1 horizon)
├── loso_train_T11_summary.json              (LOSO results, 25 entries)
├── loso_train_h10.log                       (LOSO training log)
├── loso_model_h10_seed{S}_held{K}.txt       (25 LOSO models)
├── loso_pred_h10_seed{S}_held{K}.parquet    (25 LOSO OOF parquets)
├── aggregate_ensemble.py
├── loso_pred_ensemble_h10_held{0..4}.parquet (5 averaged OOF)
├── threshold_sweep.py
├── threshold_results.json                   (sweep result + per-seed comparison)
├── sweep_results_*.csv                      (per-seed and ensemble grids)
├── train_final_223d_seeds.py                (final model train script)
├── train_final_h10_seeds.log
├── final_223_model_h10_seed{S}.txt          (5 final boosters → iter_003)
├── Predictor.py                             (iter_004 ensemble Predictor template)
├── build_iter_003.py                        (this iter's build script)
└── build_iter_004.py                        (legacy template — not used)
```
