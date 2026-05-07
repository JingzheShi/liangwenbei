# T40 — Probability Calibration on iter_006 base — Negative Result

> **TL;DR**: Tested Platt / Isotonic / Temperature calibration on the iter_006
> 5-seed h_60 ensemble OOFs, then re-ran DE 4D thresh on each variant.
> **All three calibrations *hurt* LOSO sum cum_pnl** (best calibrated −1.42 vs
> baseline +13.59). User's hypothesis ("low recall = bad calibration") doesn't
> hold — the low active rate is the DE thresh's PnL-optimal choice, not a
> calibration artifact. **Do not package iter_007.**

## Setup

**Base preds**: iter_006 = 5-seed avg of iter_005b h_60 OOFs (seed 42 from T26,
seeds 1/7/13/100 from T27). Per-fold ensemble = mean of 5 seed probs. 5 LOSO
folds × 88,416 rows each = 442,080 rows.

**Three calibration methods**, all fit cross-fold (calibrator for fold k is
trained on OOFs from the other 4 folds, never sees fold k labels):

- **Platt** — per-class one-vs-rest LogisticRegression on logit(prob_c) →
  LR.predict_proba, then renormalize so probs sum to 1.
- **Isotonic** — per-class one-vs-rest IsotonicRegression on prob_c → calibrated
  prob_c, renormalize.
- **Temperature** — single scalar T per fold; minimizes NLL of softmax(log_p / T)
  on the other 4 folds. Preserves rank order across classes within a sample.

**DE 4D thresh**: identical to T37/T38 — minimize neg sum cum_pnl over
(T_up, T_dn, d_up, d_dn) ∈ ([0.34,0.75]² × [0,0.30]²), 3 DE seeds × maxiter=60
× popsize=20 × Sobol init.

## Results

### Main table (5-fold LOSO h_60, fee=0.0001)

| variant         | sum cum_pnl | n_pos folds | active count | active rate | F0.5 macro | accuracy | vs iter_006 |
|---|---|---|---|---|---|---|---|
| **iter_006 baseline (raw)** | **+13.59** | **5/5** | 131,876 | **29.83%** | **0.4516** | 0.5990 | — |
| Platt           | +2.79  | 4/5 | 5,867  | 1.33%  | 0.2724 | 0.6238 | **−10.81** |
| Isotonic        | +6.13  | 4/5 | 29,490 | 6.67%  | 0.3683 | 0.6271 | **−7.46**  |
| Temperature     | +12.17 | 5/5 | 148,654 | 33.63% | 0.4502 | 0.5866 | **−1.42**  |

**(Per-fold T for temperature scaling: [1.27, 0.99, 0.89, 2.53, 1.02])**

### Per-fold ECE (one-vs-rest, 15 bins, min-bin=50)

| fold | class | raw   | Platt | Iso   | Temp  |
|---|---|---|---|---|---|
| 0 | 0 (down) | 0.216 | 0.131 | 0.127 | 0.221 |
| 0 | 1 (flat) | 0.425 | 0.220 | 0.215 | 0.438 |
| 0 | 2 (up)   | 0.210 | 0.090 | 0.090 | 0.218 |
| 1 | 0 | 0.049 | 0.092 | 0.093 | 0.050 |
| 1 | 1 | 0.165 | 0.146 | 0.124 | 0.165 |
| 1 | 2 | 0.120 | 0.054 | 0.048 | 0.120 |
| 2 | 0 | 0.141 | 0.038 | 0.043 | 0.142 |
| 2 | 1 | 0.370 | 0.127 | 0.154 | 0.379 |
| 2 | 2 | 0.229 | 0.083 | 0.110 | 0.238 |
| 3 | 0 | 0.035 | 0.055 | 0.026 | 0.149 |
| 3 | 1 | 0.073 | 0.104 | 0.077 | 0.302 |
| 3 | 2 | 0.038 | 0.049 | 0.051 | 0.153 |
| 4 | 0 | 0.096 | 0.038 | 0.039 | 0.095 |
| 4 | 1 | 0.202 | 0.112 | 0.100 | 0.201 |
| 4 | 2 | 0.106 | 0.070 | 0.070 | 0.106 |

**Observation**: Platt and Isotonic *do* lower ECE on most folds (especially
folds 0 and 2 — the syms with most miscalibrated probs). Temperature actually
*raises* ECE on folds 3 (heavy class-1 dominance, T=2.53 over-softens). And yet
**lower ECE → worse PnL**. Why?

### Why calibration hurts PnL

**1. DE 4D thresh is already a learned calibrator.** The decision rule
`(p2 ≥ T_up) ∧ (p2 > p1 + d_up) ∧ (p2 > p0)` (and symmetric for class 0) does
not need probabilistically meaningful inputs — it only needs the *ranking* and
*magnitude gap* of the probs to be PnL-monotonic with respect to the gate. The
DE search optimizes the gates end-to-end on PnL, so any miscalibration is
absorbed into (T, δ).

**2. Per-class one-vs-rest calibration breaks joint ordering.** Platt and
Isotonic map each prob_c through an independent monotonic function. After
renormalization, the relative ordering (p2 > p1 vs p0 > p1) **can flip** for
samples near the boundary. The PnL-relevant inequality `p2 > p1 + d_up` is no
longer a simple function of the original probs — DE has to re-discover the
right gate, but the new gate works on a worse signal.

**3. Temperature scaling preserves rank order**, so it should be safer. And
indeed Temperature's drop is the smallest (−1.42 vs −10.81). But cross-fold T
is still a **bad estimator of per-fold T** — fold 3 needs a softening T (its
labels are 80% class 1 → calibrated probs should be flatter), while fold 1
needs sharpening (T<1). The cross-fold-trained T per fold = average over
*other* folds, which under-fits each individual fold's miscalibration.

**4. Per-fold prevalence shift (LOSO across syms)** is the root cause. Each
held-out sym has very different label prior:
- fold 0: 81.7% class-1 (low signal, very few moves)
- fold 1: 42.8% class-1 (active sym)
- fold 2: 60.8% class-1
- fold 3: 79.8% class-1 (low signal)
- fold 4: 45.5% class-1 (active sym)

A calibrator fit on the avg of 4 other folds **always fits the wrong prior**
for fold k. Per-class Platt with logit input ≈ a global "boost class-1 prob,
shrink class-0/2 prob" — when applied to fold 0 or 3 (very high class-1 rate
already), it shrinks p2/p0 too aggressively → DE can only fire ~1% of trades.

### What about active rate / recall?

User's hypothesis: "low active rate ~24% means low recall, calibration would
help spread the gates better". The data refutes this:
- **Temperature (active 33.6% > baseline 29.8%)** — slightly lower F0.5 (0.450
  vs 0.452), more activity but lower PnL. Activity ≠ profitable activity.
- **Platt (active 1.3%)** — DE search responds to the calibrated probs by
  collapsing the gate (T_up=0.34 ≈ floor of the search). Even at the floor,
  only 1.3% of samples cross — this means after Platt, p2 (and p0) values for
  most samples drop below 0.34. The calibrator collapsed the prob distribution.

### Active per fold (illustrates the constraint)

| fold | label dist (0/1/2 %) | baseline active | baseline PnL |
|---|---|---|---|
| 0 | 9.6 / 81.7 / 8.7  | 6,013   | +1.81 |
| 1 | 32.3 / 42.8 / 24.8 | 42,527 | +2.61 |
| 2 | 21.0 / 60.8 / 18.2 | 39,703 | +1.48 |
| 3 | 10.7 / 79.8 / 9.5  | **245** | +0.10 |
| 4 | 29.3 / 45.5 / 25.2 | 43,105 | +7.59 |

**Fold 3 only fires 245 trades (0.28% of fold)** — the DE thresh has *already*
learned to gate it almost off, because the model has near-zero edge on this
sym. This is correct behavior — pushing more trades into fold 3 (regardless of
calibration) destroys PnL.

**Fold 4 alone produces +7.59** — over half the total PnL. The active rate
is 24-30% because that's where the marginal PnL hits zero on the most-active
folds. Increasing recall beyond that point is a *negative-EV* expansion.

## Decision

**Do NOT package iter_007 from calibrated probs.**

Reasons:
1. All 3 methods produce strictly worse LOSO sum than baseline.
2. The "low recall" the user observed is the **PnL-optimal choice** of the
   joint (model, DE thresh) system, not a calibration bug.
3. If we want to lift PnL, the bottleneck is **alpha on fold 3** (and to a
   lesser extent fold 0), not probability calibration. Those syms genuinely
   look unpredictable from 100-tick LOB windows alone.

## What this rules out

- ❌ Platt / Isotonic / Temperature scaling on the existing 5-seed ensemble
  cannot improve LOSO h_60 PnL.
- ❌ The "low active rate ~24%" hypothesis is wrong — it is a **feature**,
  not a bug. DE thresh chose this because forcing higher activity is
  net-negative on the low-signal folds.
- ❌ Adding any monotone post-hoc transformation to probs is unlikely to help
  unless paired with a fundamentally better thresh objective (e.g. per-fold
  thresholds at inference, but that violates "sym-agnostic" constraint).

## What might still work (suggestions for follow-up T41+)

These are NOT dispatched here — only documented for future PM consideration:

1. **Better alpha on fold 3 / fold 0** — the bottleneck is signal, not
   calibration. Fold 3 = 0.10 PnL on baseline. Even doubling its PnL (to 0.20)
   only adds +0.10. The high-leverage move is a feature/model that lifts
   fold 1/2 (currently +1.48 / +2.61) closer to fold 4 (+7.59).
2. **Per-trade gate model** — replace 4D DE thresh with a learned gate (small
   LR or LightGBM) that takes (raw probs, time, recent vol) → "trade or not".
   This is more expressive than a 4D rule and works around the calibration
   problem.
3. **Ensemble across model families** (CatBoost, NN) for orthogonal signal —
   diversity, not calibration, is the gap. T38 already showed within-family
   ensembles top out at ~+13.6.
4. **EV-weighted thresh objective** — instead of cum_pnl direct, optimize a
   risk-adjusted objective (e.g. Sharpe per fold, then sum). This may also
   choose more activity, but for the right reason.

## Files

- `calibrate_and_thresh.py` — full implementation
- `results.json` — per-variant DE best + per-fold metrics + ECE summary
- `reliability_diag_fold{0..4}.png` — combined raw/Platt/Iso/Temp diagrams
- `reliability_diag_fold{k}_{platt,isotonic,temperature}.png` — per-method
- `run.log` — full stdout from the run

## Compliance with hard constraints

- ✅ `date` not used as feature anywhere (calibrators only see prob_c values)
- ✅ Calibration is run **offline only**, on OOF preds. No state added to
  Predictor — we never reached the "package iter_007" step.
- ✅ Calibrators fit cross-fold (no leakage of fold-k labels into fold-k probs).
- ✅ Sym-agnostic — calibrators don't see sym ID, only probs.

## Final result line

`RESULT: task=calibration metrics={best_variant=iter_006_baseline_raw, best_sum=+13.5949, uplift=+0.0000, baseline_sum=+13.5949}`
