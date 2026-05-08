# T122: Retest 3 Winning Tricks on T75 LGB L2 Baseline

**Goal.** User-uploaded platform results revealed that T75 LGB L2 (regression L2)
ranks much better on the platform than LGB Huber stacks. But the 3 known wins
(R3 HYD +1.42, R4 logret +1.60, T117 monotone +1.87) were all measured on a
**T99 LGB Huber** baseline. We need to know if they survive on the
platform-friendly **T75 LGB L2** baseline.

## Setup

* Baseline: T75 recipe — LGB **regression_l2**, schemeP 359-d (370 - 11 KS-fail
  drops), aug_a per-feat scale [0.8, 1.2] ratio=1.0, V4 split (train 0–75,
  val 76–79), class-balanced 3-class sample weights, GPU LightGBM, T117
  per-seed configs.
* 5 variants: `baseline`, `+hyd` (16 R3 dims), `+lag` (12 R4 dims),
  `+monotone` (T117 18-feature constraint list), `+all3` (HYD + lag +
  monotone, 18 of 387 features constrained).
* 3 seeds per variant: 1, 7, 42.
* Eval: 3-seed averaged ensemble → DE 2D asymmetric thresh
  (thr_up, thr_dn) maximizing sum-of-per-sym cum_pnl (LOSO-equiv).
* Test: 442 080 rows (5-sym × 24-day × 2-sess), fee = 1e-4.

Sanity check: baseline seed 42 EV-gate k=1 = +28.28, exactly matching the
published T75 number (+28.28). Reproduction is bit-exact.

## Headline result

| variant     | DE LOSO    | Δ vs base | per_sym_min | iter_017 gate (Δ ≥ 0.5)? |
| ----------- | ---------- | --------- | ----------- | ------------------------ |
| baseline    | **+35.30** | —         | +3.62       | (base)                   |
| **monotone**| **+37.57** | **+2.27** | +3.68       | **✅ PASS (big)**         |
| **hyd**     | **+36.06** | **+0.76** | +3.98       | **✅ PASS (modest)**      |
| **all3**    | **+37.74** | **+2.45** | +3.97       | **✅ PASS (=monotone+hyd)**|
| lag         | +35.43     | +0.13     | +3.78       | ❌ fail                   |

(thr_up, thr_dn ranges: 3.0–3.7e-4 / 1.5–1.8e-4 across variants.)

## Verdict

* **`monotone` is platform-portable.** +2.27 LOSO on L2 — even **better** than
  the +1.87 it scored on Huber. This is the biggest, cleanest, most
  Huber-independent win. **Definitely include in iter_017.**
* **`hyd` is platform-portable but smaller.** +0.76 on L2 vs the +1.42 it
  showed on Huber. Still passes the 0.5 gate. Per-sym Δ is mixed
  (sym1 −1.56, others positive), so the lift is real but uneven.
  **Include in iter_017** (compatible with monotone — see `all3`).
* **`lag` does NOT survive.** +0.13 LOSO on L2, vs +1.60 on Huber. This is a
  **Huber-specific phenomenon**: Huber's bounded residual loss benefits more
  from short-horizon log-return features. L2 either already extracts this
  signal from existing features (sym-agnostic OFI/imbalance lag is implicit)
  or doesn't reward the noisier 1-step log-return signal.
  **Do not include in iter_017.**
* **`all3` (HYD + lag + monotone) +2.45** ≈ monotone (+2.27) + hyd (+0.76)
  − lag interaction. The +0.18 over monotone-alone is within noise. Roughly
  consistent with **monotone is the dominant trick**, hyd is a small
  supplementary lift, and lag is dead-weight.

## Per-seed variability (DE LOSO)

| variant   | s1     | s7     | s42    | mean   | std   |
| --------- | ------ | ------ | ------ | ------ | ----- |
| baseline  | 34.25  | 34.06  | 32.76  | 33.69  | 0.66  |
| hyd       | 36.27  | 33.67  | 34.70  | 34.88  | 1.07  |
| lag       | 33.46  | 35.65  | 34.16  | 34.42  | 0.91  |
| monotone  | 35.12  | 35.85  | 36.09  | 35.68  | 0.41  |
| all3      | 35.61  | 36.76  | 34.78  | 35.72  | 0.81  |

* Monotone is the **most stable** (std 0.41). The 3-seed ensemble lift
  (+2.27) is well outside per-seed noise, so the trick is robust.
* HYD per-seed mean lift = +1.19 (vs base mean 33.69 → 34.88). The 3-seed
  ensemble lift (+0.76) is smaller because the baseline benefits more from
  averaging here. Still positive.
* Lag per-seed mean lift = +0.73; ensemble lift +0.13. The lift compresses
  almost entirely on ensemble-averaging, suggesting the signal is mostly
  noise-correlated and decorrelates between seeds.

## Pred correlations (post-3-seed ensemble vs baseline)

* corr(baseline, hyd) = 0.953
* corr(baseline, lag) = 0.958
* corr(baseline, monotone) = 0.964
* corr(baseline, all3) = 0.939

All variants stay highly correlated with baseline (≈ same model, slightly
different decision surface). Monotone has the highest correlation — its lift
comes from re-shaping the residual rather than changing what the trees see.

## Why does `monotone` win MORE on L2 than on Huber?

Two non-mutually-exclusive hypotheses:

1. **Huber already implicitly down-weights monotone-violating noisy gradients**
   via its bounded loss. Adding explicit monotone constraints is therefore
   only marginally informative. L2 has no such cushion: any non-monotone
   pattern in the residual is pursued by gradient descent, so an explicit
   monotone constraint provides a much larger regularization signal.
2. **L2 is more variance-driven.** The 18 microstructure features (imbalance,
   OFI, signed RV/BV, order-book intensities) have well-known signed
   directions in market-microstructure theory. Constraining them stops the
   tree from over-fitting per-sym/per-time patterns that violate theory —
   exactly the variance reduction L2 needs.

This also explains the platform-vs-local gap: monotone constraints align
the model with stable cross-section / out-of-distribution invariants
(theory-driven) rather than fitted noise. It should travel **especially
well** to platform's shuffled / unknown-stock test set.

## Why does `lag` die on L2?

The 12 lag features are short-horizon ratios (log_ret_mid_lag1..50,
log_ret_wmp_lag1..50). They are highly noisy at short scales and only
weakly directional. Huber's robust loss tolerates the noise and extracts
the rare strong-signal cases. L2's quadratic loss amplifies the noise into
the gradient and treats every random spike as a target — so per-seed lifts
are present (overfit-style) but they decorrelate on ensemble.

## Recommendations for iter_017

| Trick      | iter_014 LOSO Δ on L2 | Status              |
| ---------- | --------------------- | ------------------- |
| monotone   | **+2.27**             | **TAKE**            |
| hyd        | **+0.76**             | TAKE (compat w/ mono) |
| lag        | +0.13                 | drop                |

**Recommended iter_017 candidate:** T75 baseline + monotone + hyd
(skip lag). Predicted local DE LOSO ≈ baseline + 2.45 = ~+37.7 on the
3-seed L2 stack we tested. With 5 seeds (instead of 3) it should match or
slightly exceed iter_014's published +38.28. Monotone alone gives +37.57,
within noise of the all3 number; if a simpler stack is preferred, **monotone-only is essentially equivalent** to monotone+hyd.

## Files

* `train_t122.py` — variant-aware LGB L2 trainer (objective=regression_l2).
* `eval_t122.py` — DE 2D asymmetric LOSO eval, 3-seed ensemble.
* `model_T122_{variant}_seed{S}.txt` — 15 models (5 × 3 seeds).
* `pred_T122_{variant}_seed{S}.parquet` — 15 pred parquets.
* `summary_T122_{variant}_seed{S}.json` — per-train summary.
* `eval_t122.json` — DE LOSO, deltas, correlations.
* `eval_t122.log` — DE eval stdout.
* `results.json` — final summary.

## CRITICAL_CONSTRAINTS check

* No `date`, `sym`, `time` in features (keep_idx asserted).
* All features are stateless / row-wise (HYD windows are within-snapshot,
  lag features are within-snap log returns).
* `objective=regression_l2` — no Huber alpha. T75 baseline replicated
  bit-exactly (+28.28 seed42).
* Sym-agnostic: monotone constraints are global, no per-sym thresholds.
