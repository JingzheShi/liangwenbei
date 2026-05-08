# T92 — Magnitude-weight regression revival (R44 prediction → confirmed FALSE)

## TL;DR

R44 hypothesised that T57's "PnL-aware sample weight = f(|Δmid|)" — which lost
–3 to –11 under CE because class collapse killed magnitude — would **revive
under L2 regression**, because under L2 the weight directly multiplies (Δmid)²
loss and so should drive the regressor toward larger-PnL samples (predicted
gain: +0.5 ~ +2.0 LOSO-equiv over iter_014 +38.28).

**Empirical result: every magnitude-weight variant (7 of them) underperforms
the T75 baseline at single-seed and at 5-seed DE-asym ensemble.** Best variant
(`G_sqrt_offset` = √(|Δmid| + q50_|Δmid|)) gives 5-seed DE-asym **+32.58**
vs iter_013 +36.23 (−3.65) and vs iter_014 +38.28 (−5.70). The hypothesis is
**falsified at the magnitude-weight level — recommended reverification at the
class-balanced-weight level, which T75 already uses and which already encodes
a coarse magnitude prior**.

No new submission produced.

## Setup (identical to T75 except for sample weight)

- **Cache**: T68 schemeP, drop 11 KS-fail features → 359-d
- **Target**: y_regr = (mp_t60 − mp_t)/(mp_t + 1)
- **Split**: V4 walk-forward, train=date 0-75, val=date 76-79, test=local 442 080
- **Model**: LightGBM regression_l2, GPU, num_boost=600, early_stop=40
- **Aug**: aug_a per-(sample, feat) scale [0.80, 1.20] @ aug_ratio=1.0 (concat)
- **Seeds**: {42, 1, 7, 13, 100}, same SEED_CONFIGS as T64/T70/T75

## Sample-weight schemes tested

All weights have a small floor `eps_w=1e-6` (LightGBM ignores w=0 rows) and are
mean-normalized to 1.0. Train-statistic q50/q95 used to compute val/test weights
(no leakage from val distribution).

| variant | formula | rationale |
|---|---|---|
| **T75 baseline** | `class_balanced_weight(y_cls)` | flat≈0.55x, up/down≈1.7x — coarse magnitude prior |
| A_linear | `\|y\|` | direct magnitude, R44's primary suggestion |
| B_sqrt | `√\|y\|` | milder root-power |
| C_capped | `min(\|y\|, q95)` | clip the tail to reduce dynamic range |
| D_offset | `\|y\| + q50` | additive offset → low-\|y\| retain ≈ q50 weight |
| E_floor_q50 | `max(\|y\|, q50)` | floor at median, no upper clip |
| F_class_mag | `class_balanced(y_cls) · (1 + α·\|y\|/q95)` | T75 + finer magnitude (α∈{0.1, 0.5}) |
| **G_sqrt_offset** | `√(\|y\| + q50)` | smooth combination of B and D — softest |

## Pilot results (single seed=42, EV-gate k=1, thr=2·FEE)

| variant | best_iter | val mse | test corr | cum_pnl @ k=1 | Δ vs T75 |
|---|---|---|---|---|---|
| T75 (class-bal)  | 124 | 5.4e-6 | 0.146 | **+28.28** | (baseline) |
| **G_sqrt_offset** | 115 | 5.0e-6 | 0.135 | +27.00 | −1.28 |
| F_class_mag α=0.1 | 162 | 5.0e-6 | 0.140 | +25.42 | −2.86 |
| B_sqrt           | 115 | 5.2e-6 | 0.138 | +23.08 | −5.20 |
| F_class_mag α=0.5 |  58 | 5.1e-6 | 0.139 | +22.55 | −5.73 |
| C_capped         |  38 | 5.1e-6 | 0.145 | +21.09 | −7.19 |
| D_offset         | 116 | 5.2e-6 | 0.129 | +18.42 | −9.86 |
| E_floor_q50      |  90 | 5.3e-6 | 0.126 | +16.09 | −12.19 |
| A_linear         |  91 | 5.4e-6 | 0.133 | +12.61 | −15.67 |

All 8 magnitude variants **lose** vs class-balanced T75 at single-seed.

## 5-seed ensemble + EV-gate (top-2 variants)

For G_sqrt_offset and B_sqrt, retrained on seeds {1, 7, 13, 100} and ran the
T75 ev_gate_eval pipeline (symmetric k sweep + DE asymmetric).

### G_sqrt_offset 5-seed (best variant)

| metric | T92 G_sqrt | T75 / iter_013 | iter_014 | Δ vs iter_013 | Δ vs iter_014 |
|---|---|---|---|---|---|
| best symmetric k             | k=1.50, +31.68 | k=1.25, +33.72 | n/a   | −2.04 | n/a |
| **DE asymmetric (per-sym)**  | **+32.58** | **+36.23** | **+38.28** | **−3.65** | **−5.70** |
| best DE thr_up / thr_dn      | 3.42e-4 / 2.00e-4 | 3.72e-4 / 1.61e-4 | n/a | — | — |

Per-sym (DE):  G_sqrt = [+3.59, +6.37, +3.59, +9.42, +9.62]
T75 / iter_013 = [+3.65, +6.65, +4.24, +10.43, +11.25].

### B_sqrt 5-seed

| metric | T92 B_sqrt | iter_013 | iter_014 | Δ vs iter_013 |
|---|---|---|---|---|
| best symmetric k       | k=1.50, +29.85 | k=1.25, +33.72 | n/a   | −3.87 |
| **DE asymmetric**      | **+32.24**  | **+36.23** | **+38.28** | **−3.99** |

## Why R44's prediction fails — interpretation

R44 reasoned: under L2, weight = |Δmid| multiplies the (pred − Δmid)² loss
directly, so up-weighting large-|Δmid| samples drives the regressor where PnL
lives. Empirically this is wrong, for at least three compounding reasons:

1. **L2 already weights by squared residual.** The L2 gradient is
   `2w(pred − y)`. For a fresh booster `pred ≈ 0` and the gradient for sample i
   is `≈ −2·w_i·y_i`. Even with `w ≡ 1`, large-|y| samples already contribute
   gradients proportional to |y|. Adding `w = |y|` multiplies again → effective
   weighting `|y|²`, which is *more aggressive than necessary* and concentrates
   the fit on the tail (where label noise is largest). All hard-magnitude
   variants (A, C, D, E) collapse fast (best_iter ∈ [38, 116]) and overfit.

2. **Class-balanced weight is already a coarse magnitude prior.** Class labels
   come from τ-thresholding |Δmid|. So `class_balanced(y_cls)` upweights the
   directional class samples (≈large |Δmid|) by ~3× vs flat (≈small |Δmid|).
   T75 already exploits the magnitude/direction asymmetry — finer magnitude
   weighting (B, F-α=0.1, G) doesn't add a second-order signal, just noise.

3. **PnL is decided by SIGN + threshold-crossing, not magnitude precision.**
   The EV gate fires at `pred > thr_up`; once crossed, it doesn't matter how
   far above the threshold pred is. So getting `pred = 0.005` exactly right at
   a high-|y| sample buys nothing over `pred = 0.001` (both trigger the gate).
   Magnitude weighting biases the fit toward overshooting at the tail —
   reduces the model's ability to correctly classify mid-magnitude samples,
   which is where the bulk of marginal PnL trades live.

## Implication for R44's other revivals

R44's "F-C" pathway ("sample-weight × CE non-alignment → revive under L2")
should be **rated lower-confidence** for any other proposal that uses the same
mechanic (e.g. T45 group-DRO, T66 adversarial reweighting):

- **T45 Group DRO** (R44 estimate +0.5 ~ +2.5): probably similar negative.
  Per-sym MSE-loss reweighting will fight the same L2-already-weights-residuals
  effect, and the LightGBM 5-tree-batch DRO converges far slower than fixed
  weights.
- **T66 adversarial reweighting** (R44 estimate +0.3 ~ +1.5): some independent
  shot — `w = is_test_posterior` is unrelated to magnitude, so doesn't suffer
  the double-count. But the bound is small and stacking with other ideas is
  uncertain.

R44's "F-A" (magnitude collapse) and "F-B" (decision bottleneck) pathways
remain valid for ideas where the failure mode is *information loss at the
label*, not weighting — e.g. T22 alpha101/191, T35 ReVol/Savgol, T46 CatBoost
regression. Those are still recommended.

## Compliance (CRITICAL_CONSTRAINTS.md §3)

| check | status |
|---|---|
| Model forward does not receive `sym`         | ✅ |
| Model forward does not receive `date`        | ✅ |
| Predictor.predict stateless                   | ✅ (would inherit from T75 if shipped) |
| shuffle-invariant                             | ✅ |
| Normalization global, not per-sym             | ✅ |
| Sample weight depends only on (mp_t, mp_t60), available at training time, not at inference | ✅ |

## Files

```
experiments/T92_magnitude_weight_regr/
  train_t92.py                        # 7-variant magnitude-weight regression trainer
  ev_gate_eval.py                     # 5-seed EV-gate + DE-asym sweep
  pilot_{A_linear,B_sqrt,C_capped,D_offset,E_floor_q50,
         F_class_mag_a05,F_class_mag_a01,G_sqrt_offset}.log    # single-seed pilot
  pilot_B_sqrt_seed{1,7,13,100}.log           # B 5-seed
  pilot_G_sqrt_offset_seed{1,7,13,100}.log    # G 5-seed
  ev_eval_{B_sqrt,G_sqrt_offset}.log
  pred_T92_*_seed*_pilot.parquet              # per-(variant, seed) test preds
  model_T92_*_seed*_pilot.txt
  summary_T92_*_seed*_pilot.json
  ev_gate_results_{B_sqrt,G_sqrt_offset}_pilot.json
  results.json
  worker-progress.json
  REPORT.md   (this file)
```

## RESULT

```
RESULT: task=t92_magnitude_weight metrics={best_variant=G_sqrt_offset, loso_equiv=32.58, vs_iter014=-5.70, vs_iter013=-3.65} notes=R44 hypothesis falsified — magnitude sample weight hurts L2 regression vs class-balanced baseline; reasoning: L2 already weights by residual², class-bal already encodes coarse magnitude prior, PnL decided by sign+threshold not magnitude precision
```
