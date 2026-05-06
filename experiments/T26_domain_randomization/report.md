# T26 Domain Randomization — LOSO h=60 — Report

> Goal: harden h=60 LightGBM models against OOD `sym` by augmenting training-time
> features (per-feature random scale / Gaussian noise / random feature dropout).
> Hypothesis: training-time domain randomization shrinks the LOSO ↔ platform gap
> that we observed for short horizons (h_5: +17.49 LOSO vs −7.68 platform; h_10:
> +21.86 vs −8.64). h_60 was already well-calibrated (+6.30 LOSO vs +4.07
> platform), so we use it as a sanity-check horizon: if DR breaks the LOSO score
> we know the augment hurts; if it lifts LOSO without losing OOD calibration,
> iter_006 is justified.

## Setup

- Cache: `experiments/T5b_features_multihorizon/cache/schemeC_{train,val,test}.npz`
- Feature space: Scheme C, 226-d → drop trailing 3 time-encoding cols → **223-d**
  (matches iter_002 / T11 baseline exactly).
- Folds: standard LOSO 5-fold (held=0..4); train (sym ≠ k, dates 0..79), val
  (sym ≠ k, dates 80..95) for early stop, test (sym == k, dates 96..119).
- Hyperparams: lr=0.05, num_leaves=127, min_data=100, ff=0.8, bf=0.8, λ_l2=1.0,
  num_boost=600, early_stop=40, **GPU LightGBM**, seed=42.
- Augment ratio: 1.0 → for non-baseline variants the merged batch is
  `concat(X_orig, X_aug)` with class-balanced weights recomputed.
- Augment pipeline (training only — val/test untouched):
  - **A (scale)**: `X *= U[0.8, 1.2]` per (sample, feature)
  - **B (noise)**: `X += N(0, σ·train_std)` with σ=0.05
  - **C (dropout)**: per-sample mask 5–10 % of features → set to feature mean
  - **A+B+C** (combined): each step at reduced strength (scale [0.9,1.1],
    σ=0.03, drop 3–7 %) so total perturbation stays comparable to a single aug

## Hard-constraint compliance

- ✅ `date` never used (already absent from cache feature set)
- ✅ `sym` only used to slice folds at training time; **never** a feature
- ✅ Augment is *per-sample stateless* — Predictor never reproduces it; inference
  is deterministic and shuffle-invariant
- ✅ Per-feature mean/std for augment scaling is computed *per fold* on training
  data only (no leakage)

## Results — h_60 LOSO over 5 folds

(filled in after threshold sweep; raw argmax + per-variant best (T, δ))

| Variant | Raw argmax sum | Best (T, δ) | Best LOSO sum | std/fold | n_pos/5 | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline (no aug) | TBD | TBD | TBD | TBD | TBD | reference |
| aug_a (scale)     | TBD | TBD | TBD | TBD | TBD | TBD |
| aug_b (noise)     | TBD | TBD | TBD | TBD | TBD | TBD |
| aug_c (dropout)   | TBD | TBD | TBD | TBD | TBD | TBD |
| aug_abc (combo)   | TBD | TBD | TBD | TBD | TBD | TBD |

Reference: iter_002 h_60 LOSO sum = +6.300 (T=0.5, δ=0.2; raw argmax=−12.14).

### Per-fold table at each variant's best (T, δ)

(filled in after threshold sweep)

## Feature importance — diff under augment

(comparison of `fi_gain` distribution; how many features change rank by >10 ranks)

## iter_006 decision

Threshold for promotion: best variant LOSO h_60 > **+7.5** AND std per fold
≤ baseline std → build iter_006 with that variant.

- Decision: TBD
- If built: see `submission/iter_006_h60_aug/` and `submission_050625_iter006.zip`

## Notes / caveats

- LOSO is an *underestimate* of true OOD performance because the held-out sym is
  one of the 5 known training syms, not a brand-new sym. DR's purpose is to
  improve performance on *brand-new* syms — LOSO can only proxy it.
- h_60 was already calibrated; the bigger win for DR (if any) would be h_5 / h_10
  where LOSO ↔ platform gap is −25 to −30. Future work: extend T26 to short
  horizons.
- mmpc_demo calibration check is deferred (low ROI given h_60 is already
  calibrated; would help more for h_5/h_10).
