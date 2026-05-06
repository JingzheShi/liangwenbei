# T16: T11 (5-seed Scheme C) + T14 (cross-sym mixup) 6-model ensemble

## Goal
Test whether combining T11's 5 standard-seed Scheme C models with T14's mixup-
augmented model (α=0.20) gives meaningful uplift over T11-only ensemble at h=10.

## Inputs (all reused — no new training)
- `experiments/T11_schemeC_multiseed/loso_pred_h10_seed{42,1,7,13,100}_held{0..4}.parquet` (25 OOF parquets)
- `experiments/T14_cross_sym_mixup/loso_pred_h10_alpha0p20_held{0..4}.parquet` (5 OOF parquets)

## Method
- For each LOSO fold (held=0..4), simple mean of `prob_0/prob_1/prob_2` across 6 models.
- Threshold sweep over T ∈ {0.35..0.70 step 0.05}, δ ∈ {0..0.25 step 0.05}.
- Compare best 6-model ensemble vs T11-only 5-seed ensemble best.

## Results (h=10, fee=1bp)

| Variant | Best (T, δ) | Sum cum_pnl | n_pos_folds | Per-fold |
|---|---|---|---|---|
| iter_002 single-model baseline | T=0.55, δ=0.10 | **+21.8595** | 5/5 | (1.93, 5.03, 4.33, 1.77, 8.65) |
| T11 5-seed Scheme C ensemble | T=0.55, δ=0.10 | **+22.2241** | 5/5 | (2.07, 5.19, 4.37, 1.79, 8.80) |
| T11+T14 6-model (this) | T=0.55, δ=0.10 | **+22.2454** | 5/5 | (2.08, 5.23, 4.39, 1.83, 8.71) |

- 6-model **uplift over T11-only**: **+0.0213** (≈ +0.1%)
- 6-model **uplift over iter_002**: **+0.3858**

Argmax baselines (no threshold):
- T11 ensemble argmax = +6.7404
- 6-model ensemble argmax = +7.2221 (slight bump)

## Decision
The user's gate was `6-model > +22.5`. We got **+22.2454** — does **not** clear the gate,
gain over T11-only is essentially noise (+0.02 PnL on a +22 base). Evidence:
- per-fold PnL is nearly identical across the two ensembles
- T11 already saturates the diversity-from-seeds mechanism at 5 seeds; one more
  member (mixup) does not add meaningful information that the base 5 don't already span.

**Action**: stick with T11-only 5-seed ensemble for `iter_003`. No iter_003b.

## Files

- `combine_oof.py` — averaging + sweep code (~190 lines)
- `loso_pred_combined_h10_held{0..4}.parquet` — averaged OOF
- `sweep_combined_h10.csv` — full T×δ grid
- `results.json` — top-10 grid + best vs T11 baseline

## Notes

- The ensemble averaging is naïve (uniform weight). A weighted average favoring
  T11 (e.g., 5/6 T11 + 1/6 T14) would compute identically here since T14's
  contribution is already 1/6.
- Stacking (LR/LGBM meta on top) was tried in T12 and **failed** (-4 PnL).
  Probability averaging was correct here.
