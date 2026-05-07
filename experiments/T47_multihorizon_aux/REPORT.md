# T47 — Multi-horizon Aux Target Stacking — Report

> **Status:** _draft, will be updated after training + stacking complete_
>
> **Goal:** R33 #1 — Train aux LightGBM heads on horizons {30, 120, 240} alongside
> the main h_60 head, then meta-stack the 12-d (4 horizon × 3 prob) outputs with a
> Ridge regression targeting label_60. Expected gain per Volkova 8th JS2024:
> +0.001 on LB (≈ +0.5–1.5 LOSO).

## Setup

- **Features:** Scheme C 223-d (drops trailing 3 time-encoding cols, matches iter_006).
- **Augmentation:** aug_a (per-(sample, feature) multiplicative scale `U[0.80, 1.20]`,
  ratio 1.0 — same as iter_005b/iter_006).
- **Training:** Single seed=42, 5 LOSO folds, LightGBM CPU.
  - num_boost_round=600, lr=0.05, num_leaves=127, ff=0.8, bf=0.8, λ_l2=1.0
  - early stopping=40 on `multi_logloss` against val (sym != held, dates 80–95)
- **Horizons:** {30, 60, 120, 240}.
  - h_60 labels: from existing T5b cache (threshold T=0.001, reverse-engineered).
  - h_30/120/240 labels: built fresh in `cache/schemeC_*_extras.npz` using same
    rule `label = 2 if Δmp > 0.001 else 0 if Δmp < -0.001 else 1`.
  - For h=120/240, rows where `t + H > T - 1` get y=-1 (filtered from training).
- **LOSO test:** sym==held, dates 96–119. Held syms 0..4.
- **Stacking:** Per fold, predict probs on val (sym != held) and test (sym == held)
  with all 4 horizon models. Concatenate to 12-d. Fit `RidgeCV` (per-class with
  alphas {0.1, 1, 10, 100}) on val OOF with one-hot label_60 target.
  Apply to test → 3-d stacked probs (clipped + renormalized to simplex).
- **Threshold:** 4-d asymmetric DE (T_up, T_dn, d_up, d_dn) over stacked probs
  on the 5 held-out test folds — same protocol as T30 (in-sample optimum on the
  LOSO concat, identical to iter_006 evaluation).

## Reference baselines

| Setup | LOSO h_60 sum cum_pnl |
|---|---|
| iter_006 (5-seed aug_a + DE 4D thresh) | **+13.61** |
| Single-seed-42 aug_a h_60 argmax | _-4.23 (T26)_ |
| Single-seed-42 aug_a h_60 + DE 4D thresh (T47 baseline) | _**TBD**_ |

## Per-horizon LOSO h-target argmax PnL (single seed=42)

| Horizon | sum cum_pnl | per-fold |
|---|---|---|
| h=30 | _TBD_ | _TBD_ |
| h=60 | _TBD_ | _TBD_ |
| h=120 | _TBD_ | _TBD_ |
| h=240 | _TBD_ | _TBD_ |

> Note: PnL for non-60 horizons is computed against y_h with `mp_t{h}` label horizon
> just as a sanity readout — the actual evaluation metric is h_60 LOSO sum after
> stacking.

## Stacking Results

_TBD_

| Stage | LOSO h_60 sum cum_pnl |
|---|---|
| Single-seed h_60 + DE thresh (baseline) | _TBD_ |
| Stacked 4-horizon + DE thresh | _TBD_ |
| Δ vs single-seed h_60 | _TBD_ |
| iter_006 (5-seed h_60 + DE thresh) | +13.61 |
| Δ vs iter_006 | _TBD_ |

## Decision

_TBD_

## Files

- `cache/schemeC_*_extras.npz` — h_30/120/240 labels aligned to T5b cache
- `loso_model_h{H}_held{K}.txt` — LightGBM models, 4 horizons × 5 folds = 20 models
- `loso_pred_h{H}_held{K}.parquet` — held-out test predictions
- `loso_pred_h{H}_val_held{K}.parquet` — val predictions (stacking input)
- `loso_train_T47_summary.json` — per-horizon argmax LOSO results
- `baseline_h60_de_results.json` — single-seed h_60 + DE thresh baseline
- `stacking_de_results.json` — stacked LOSO + DE thresh result
- `stacking_oof.parquet` — stacked OOF predictions across all 5 folds
