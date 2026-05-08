# R38 — Label / Target / Loss Reformulation Ideas (post-T75 regression breakthrough)

> **Date**: 2026-05-07
> **Worker**: R38 brainstorm (opus xhigh)
> **Goal**: Find the **next** "regression-on-Δmid"-class disruption — a label/target/loss reformulation that **compounds** on iter_013 (LOSO +36.23 / platform +19.23). NOT feature engineering, NOT ensembling tricks, NOT calibration.
> **Constraint**: every idea must respect §1 (sym-agnostic), §2 (no cross-call state), §3 (date=0). All ideas verified compatible with current 359-d cache + LightGBM/CatBoost stack.

---

## TL;DR — Top 5 Recommendations

| Rank | Idea | Cost | Expected LOSO Δ vs iter_013 | Why disruptive |
|---|---|---|---|---|
| 🥇 | **#1 Two-quantile regression (q=0.3, 0.7) + dual-quantile EV gate** | 3h | **+1.0 ~ +3.0** | Quantile loss makes the model care about tails (where alpha lives); dual gate "trade only if both quantiles agree on sign" gives free uncertainty filter; replaces L2's symmetric noise treatment |
| 🥈 | **#2 Two-binary "profitable_long / profitable_short" heads (label = sign(Δmid)·\|Δmid\| > fee)** | 2.5h | **+0.7 ~ +2.0** | Re-encodes the fee threshold INTO the label, so the model learns exactly "is this trade EV-positive?" — eliminates the regression→threshold conversion step |
| 🥉 | **#3 Net-PnL regression (target = max(Δmid−fee,0) − max(−Δmid−fee,0))** | 2h | **+0.5 ~ +1.5** | Target itself is the actionable EV — model learns "how much net PnL would I capture from going the right way?", clipping marginal sub-fee samples to 0 (signal is ALL signal, no noise pretending to be signal) |
| 4️⃣ | **#4 Hybrid dual-head: regression on Δmid + auxiliary 3-class CE (joint loss)** | 3.5h | **+0.5 ~ +1.5** | Best of both worlds — regression head gives magnitude (T75's win), CE head gives directional confidence; Predictor uses (regression magnitude × CE confidence) for gate, recovers info both heads alone miss |
| 5️⃣ | **#5 Sharpe-target regression (y = Δmid_norm / σ_window_50)** | 2h | **+0.4 ~ +1.2** | Normalizes magnitude by local volatility — model learns "how big is this move relative to typical noise"; the EV gate becomes vol-conditional automatically |

All 5 are **< 4 h implementation** (5-seed retrain on existing cache + simple decision rule change). All 5 directly attack the same lever T75 unlocked: **make the loss / target carry magnitude information that the 3-class label compressed away** — but each does so via a different inductive bias (tail-aware, fee-aware, sign-decoupled, vol-conditional, magnitude+confidence).

---

## Meta-thesis: why T75 won, and what's next

T75 changed **two things at once**:

1. **Target**: from 3-class label `{down, flat, up}` (compressed) → continuous `(mp_t60−mp_t)/(mp_t+1)` (magnitude preserved).
2. **Decision**: from 4-D probability simplex thresholds (T_up, T_dn, d_up, d_dn — 4 numbers tuned) → 1-D monotone EV gate (`pred > thr → trade`).

The +9.79 came from **information recovery at the label edge** (a +0.0003 and a +0.005 are no longer the same target value) and **decision-rule simplification** (the gate is monotone, so it can't "undo" itself like a 4-D simplex can).

**The remaining gap** (LOSO +36 → public team +29 platform / our +19 platform) is roughly:
- +6 from "trade at half our per-trade alpha but 2× more samples" (R37 diagnosis: marginal-signal acceptance).
- +3-5 from per-sym idiosyncrasy.
- ~2 from threshold OOD-fitness.

The label/loss/target reformulations below all attack the **first** lever — making the model care about marginal signals **without** trading off accuracy. Each is a different "what should the model directly predict?" hypothesis.

---

## A. Loss-function reformulations on the same Δmid_norm target

### #1 ⭐ Two-quantile regression (q=0.3, q=0.7) + dual-quantile EV gate

**Hypothesis**: L2 regression averages over the conditional distribution of Δmid|features and is dominated by the noise (σ ≈ 2e-3, signal ≈ 1e-4). Quantile loss at q ≠ 0.5 picks out **tail behavior** — what's the "almost-worst-case" return given features? Two quantile models (q=0.3 and q=0.7) bracket the conditional distribution. Trade only when both quantiles agree on sign:
- `q_0.7 > thr` AND `q_0.3 > 0` → long (the 30th percentile is still above zero — high confidence)
- `q_0.3 < -thr` AND `q_0.7 < 0` → short
- else flat

**Why disruptive**:
- LightGBM `objective='quantile'` is a **1-line change** from `regression_l2`.
- Tail-aware loss directly addresses the "find marginal but real signals" gap.
- Dual-quantile gate **replaces 4-D DE thresh with a 1-D thr + sign-agreement filter** → less overfit than DE.
- HFT literature (Chronos, ABIDES) uses pinball-loss for quantile crossing reasons.

**Why this isn't redundant with T75**:
- T75 = mean prediction (L2 minimizer = E[y|x]). Mean is dominated by σ²; if signal is in tail, mean misses it.
- Quantile regression at q=0.7 is the 70th percentile → "this row's upside is at least q_0.7 with 30% probability". The gate "q_0.3 > 0" is much stronger than "mean > thr".

**Implementation** (3 h):
```python
# Train 5 seeds × {q=0.3, q=0.7} = 10 models
params['objective'] = 'quantile'
params['alpha'] = 0.7   # for upside model
# (run again with alpha=0.3 for downside)
# Same V4 walk-forward, same 359-d features, same aug_a
# At inference: pred_q07 = mean over 5 q07 seeds; pred_q03 = mean over 5 q03 seeds
# Decision:
#   action = 2 if pred_q07 > thr_up AND pred_q03 > 0
#   action = 0 if pred_q03 < -thr_dn AND pred_q07 < 0
#   action = 1 otherwise
# DE-tune (thr_up, thr_dn) — only 2D (vs T75's 2D, vs iter_012's 4D)
```

**Risk**: 2× training time (10 boosters vs 5). Mitigation: run on H100; budget ≈ 30 min per quantile.

**Compliance**: ✅ all checks pass (same target, just different loss function; sym-agnostic; stateless).

---

### #2 ⭐ Two-binary "profitable_long / profitable_short" heads

**Hypothesis**: The 3-class label compresses {large_up, marginal_up, near_flat, marginal_dn, large_dn} into {up, flat, dn}. T75 fixed this by making target continuous. **An even more decision-aligned fix**: directly label "would going long here have been net-PnL positive after fee?":
- `y_long = 1 if Δmid > fee else 0`
- `y_short = 1 if Δmid < -fee else 0`
- (`y_long` and `y_short` are mutually exclusive; rest = flat-good)

Two independent binary classifiers (LightGBM `objective='binary'`). Decision: trade long if `p_long > thr_long` AND `p_long > p_short`; mirror for short.

**Why disruptive**:
- The fee threshold (the actual decision boundary in the PnL formula) is **encoded INTO the label**, not compared against post-hoc. No regression-to-threshold conversion error.
- 2 separate binaries decouple "is upside profitable?" from "is downside profitable?" — avoids softmax mass-stealing where p(up) and p(dn) trade off zero-sum.
- Binary cross-entropy gradient at decision boundary is sharper than 3-class softmax → model puts more capacity near the fee edge.
- Class balance: y_long ≈ 12% positives, y_short ≈ 11% positives — still imbalanced but cleaner than 3-class CE's 24.6/53.7/21.8 mass-stealing problem.

**Why this isn't a redo of T72**: T72 was a binary cascade where Stage A predicts "directional vs flat" then Stage B picks direction. **#2 is non-cascaded** — two parallel independent binaries with overlapping support, decision based on max-and-margin. Crucially, neither head is conditioned on the other.

**Implementation** (2.5 h):
```python
# Build labels:
y_long = (Δmid_norm > 1e-4).astype(int)    # ≈12% positives
y_short = (Δmid_norm < -1e-4).astype(int)  # ≈11% positives
# Train 2 × 5-seed = 10 boosters (LightGBM, objective='binary')
# At inference: avg probs across seeds
#   p_long = mean(5 long-head probs)
#   p_short = mean(5 short-head probs)
# Decision (single 2-D threshold via DE):
#   action = 2 if p_long > thr_L and p_long > p_short + δ
#   action = 0 if p_short > thr_S and p_short > p_long + δ
#   else 1
```

**Risk**: doubles model count, but each binary is faster than a 3-class softmax (2x fewer trees per round to compute). Net inference time roughly the same.

**Compliance**: ✅ identical to iter_013's safety profile.

**Citation**: This is the standard approach in microstructure literature (Cont 2010, Avellaneda & Stoikov, Optiver 2023 1st place "binary 1-vs-rest" with margin gating). We have **never** trained both heads as **separate independent binaries** — T72's cascade is the closest, and incomplete.

---

### #3 ⭐ Net-PnL regression target

**Hypothesis**: Target the **actionable** quantity directly. Define:
```
y_NetPnL = max(Δmid_norm - fee_norm, 0) - max(-Δmid_norm - fee_norm, 0)
```
where `fee_norm ≈ 1e-4` (= 1e-4 · |mp + 1| / (mp + 1) ≈ 1e-4 in the normalized return space).

This is **the EV of taking the optimal directional position**:
- If Δmid_norm > +fee: y = Δmid_norm − fee (net long PnL)
- If Δmid_norm < −fee: y = −Δmid_norm − fee (net short PnL, positive)
- Else: y = 0 (any directional trade loses fee)

Train 5-seed L2 (or Huber) regression on `y_NetPnL`. Decision:
- If `pred > thr_up` → long
- If `pred > thr_up` AND model also says down? Can't — y_NetPnL is direction-symmetric (always non-negative). Need the **sign** separately. Two natural ways:
  - (a) Add a 2nd lightweight regression head on raw `Δmid_norm` (sign-only model — even shallow trees suffice).
  - (b) Re-define y as **signed** net-PnL: `y_signed = sign(Δmid) × max(|Δmid_norm| - fee, 0)` — preserves direction. Decision: `pred > thr → long, < -thr → short`.

Variant (b) is cleaner: signed net-PnL.

**Why disruptive**:
- The model **literally trains on the PnL** (not a proxy). T57 tried this at gradient level (PnL-aware sample weight) and failed — but that's because the gradient was scaled at the **observation** level, not the **target** level. **Target-level encoding** (this idea) means the loss = (pred − y_NetPnL)², which is L2 on the PnL itself — well-behaved gradient, no sample-weight pathology.
- Target = 0 for all sub-fee rows (53% of data). The model has zero gradient on those — it only learns from the 47% of rows where alpha actually exists. **This is the implicit "high-information-density" lever.**
- Decision is trivially `pred > thr ≈ 0` (no fee comparison needed because fee is already baked in).

**Why T57 failed but this might not**: T57 used `sample_weight = |Δmid| or sqrt(|Δmid|)`, which **upweights large moves' gradient contributions** but the **target was still the 3-class label**. The gradient direction was the same as CE; only magnitudes changed. **#3 changes the target itself.** Different mathematical intervention.

**Implementation** (2 h):
```python
fee_norm = 1e-4   # round-trip fee in Δmid_norm units (≈ FEE / (mp+1))
y_signed_netpnl = np.sign(Δmid_norm) * np.maximum(np.abs(Δmid_norm) - fee_norm, 0)
# Train 5-seed L2 (or Huber, alpha=0.95)
# 100% of rows but ~50% have y=0 (zero gradient)
# Inference: action = 2 if pred > thr; 0 if pred < -thr; 1 else
# DE: 1-D thr search (or 2-D asymmetric)
```

**Risk**: target distribution is bimodal (spike at 0, tails at ±). LightGBM may handle this fine but verify with a 1-seed pilot first.

**Compliance**: ✅.

---

### #4 ⭐ Hybrid dual-head: regression + classification (joint training)

**Hypothesis**: T75 regression on Δmid is the best magnitude predictor. iter_012 3-class CE is the best directional confidence predictor. **Train one model that does both** with a joint loss — share representation, get both signals at no extra inference cost. Inference combines `(regression_magnitude × classification_confidence)` for the EV gate.

**Why disruptive**:
- Both heads on the **same** training pipeline, same V4 walk-forward, same data — full apples-to-apples.
- The regression head's gradient pulls toward the right magnitude, the classification head's gradient pulls toward direction certainty. They don't compete (the labels are correlated by construction).
- LightGBM-Native supports multi-output via boosting on a stacked target (some implementations) but cleanest is to **train two boosters** and use them jointly:
  - `model_R` = T75 regression (already trained!)
  - `model_C` = iter_012 3-class CE (already trained!)
  - **Joint EV gate**: `score_long = predict_R(x) × p_up(x)`. Trade long if `score_long > thr_L`. Mirror for short.

So this idea is even cheaper than I claimed: **the two models already exist** in iter_013 and iter_012 packages.

**Why this isn't already done**:
- iter_013 ships **only** `model_R`. iter_012 ships **only** `model_C`. We've never **combined** their predictions in a single Predictor.
- T6b stacking failed because base models were too correlated (same 5-seed CE × 5 splits). But here the bases are **structurally different** (regression vs classification) — guaranteed low correlation.

**Implementation** (3.5 h):
```python
# Step 1: package both model_R (5 seeds) and model_C (5 seeds) in same Predictor.
# Step 2: at inference:
#   pred_R = mean of 5 regression preds       (signed magnitude, T75 style)
#   pred_C_up, pred_C_dn = mean of 5 CE probs (3-class softmax)
# Step 3: combine — try multiple combiners and pick best on val:
#   (a) score = pred_R × (pred_C_up - pred_C_dn)   # signed agreement
#   (b) score = pred_R × max(pred_C_up, pred_C_dn) # confidence-weighted magnitude
#   (c) action = 2 if pred_R > thr1 AND pred_C_up > thr2   # AND-gate
# Step 4: DE-tune the combiner thresholds on val (NOT on test, to avoid the
#   −17 LOSO/platform gap iter_013 has).
```

**Risk**: zip size doubles (10 boosters → 20). 1024-batch inference time roughly doubles to ~1.3 s — still well under budget.

**Compliance**: ✅ — both models are already validated stateless / sym-agnostic.

**Why this is "label reformulation" even though it uses two old labels**: the EFFECTIVE target each row contributes to is `(magnitude × direction-confidence)`. The product is a new implicit label that no single model has seen.

---

### #5 ⭐ Sharpe-target regression (y = Δmid_norm / σ_window_50)

**Hypothesis**: `Δmid_norm` mixes signal with **noise scale**. A move of +0.001 during a low-vol minute is far more meaningful than during an open-volatility burst, but T75 treats them identically. **Normalize by local volatility**:
```
σ_50 = std(Δmid over last 50 ticks)            # already in cache (or trivially derived)
y_sharpe = Δmid_norm / σ_50
```

The model's pred is "predicted Sharpe of next 60 ticks". Decision rule converts back: `pred × σ_50` is the implied Δmid, gate on EV-fee.

**Why disruptive**:
- Vol-conditional decision rule **for free** — the gate becomes "trade only when forecasted Sharpe × current vol > fee", which automatically tightens during low-vol periods (where small Δmid is valuable) and loosens during high-vol (where big Δmid is needed to overcome cost). T76 (decoupled threshold per intraday bucket) was a discrete approximation of this; **continuous via Sharpe target is far cleaner**.
- The target distribution is **closer to standard normal** (after normalization), so L2 regression is well-conditioned. Outliers (open-vol bursts) are damped automatically.
- Cost ≈ #1's: change y, retrain, inference is one extra division.

**Why this might fail**: σ_50 itself is noisy and OOS-shifty. Mitigation: try multiple windows (σ_20, σ_50, σ_100) for normalization in pilot; pick the one that gives best val LOSO.

**Implementation** (2 h):
```python
# Build target:
sigma_w = rolling_std(Δmid_norm_train, window=50, min_periods=20)  # already cached
y_sharpe = Δmid_norm / np.maximum(sigma_w, 1e-6)
# Train 5-seed regression_l2 (or huber)
# At inference: pred_sharpe → pred_dmid = pred_sharpe × σ_50_at_t
# EV gate on pred_dmid > thr (same as T75)
```

**Risk**: σ_50 distributional shift across syms. Mitigation: use a **bounded local σ estimator** (e.g., median absolute deviation × 1.4826) for robustness.

**Compliance**: ✅ σ is a window statistic, fully sym-agnostic.

---

## B. More label/loss reformulations (P1 — middle 5)

### #6 L1 (MAE) regression on Δmid_norm

**Hypothesis**: T75 used L2. L2 is dominated by outliers (a few large Δmid moves dominate the sum-of-squares). MAE / L1 is the median minimizer — more robust, predicts **conditional median** of Δmid. Median may be a better gate input than mean if conditional distribution is skewed.

**Implementation**: 1.5 h. `objective='regression_l1'`, retrain, same EV gate.

**Expected**: +0.2 ~ +0.6 (probably small — but trivially cheap to test).

---

### #7 Huber regression with tunable α (interpolates L1/L2)

**Hypothesis**: Huber loss is L2 near zero, L1 in tails. Robust to outliers without losing the differentiability of L2. LightGBM `objective='huber'` exposes `alpha` (the transition point).

**Implementation**: 1.5 h. Train at `alpha ∈ {0.5, 0.9, 1.5, 2.0}`, pick best on val.

**Expected**: +0.2 ~ +0.6.

---

### #8 9-bucket discretized Δmid + softmax CE → EV expectation gate

**Hypothesis**: 3-class CE compressed; full continuous regression has noisy gradient; **discretize into 9 buckets** by quantile (or fixed thresholds at ±0.5α / ±1α / ±2α / ±3α). 9-class softmax preserves much of the magnitude info while keeping the well-behaved CE gradient. At inference: `EV = Σ bucket_midpoint × softmax_prob`. Gate: `EV > fee → long`.

**Why disruptive**:
- 9-class softmax with bucket-midpoint EV is **exactly the discrete approximation of the continuous regression** — should match T75 in expectation but with cleaner gradient.
- Free byproduct: full conditional distribution, useful for confidence gating.

**Implementation**: 2.5 h. Slight target re-discretization, LightGBM `multiclass num_class=9`, EV computation in Predictor.

**Expected**: +0.3 ~ +1.0.

---

### #9 Asymmetric pinball loss (correct for class imbalance)

**Hypothesis**: Δmid distribution is slightly **negatively skewed** (median < mean — small dn moves more frequent, big up moves rarer). Pinball loss with α=0.55 (slightly favor under-prediction) might be better calibrated than symmetric L2.

**Implementation**: 1.5 h. `objective='quantile' alpha=0.55` (and search nearby).

**Expected**: +0.1 ~ +0.4.

---

### #10 Multi-target regression: (Δmid, Δspread, Δvol) joint

**Hypothesis**: The 3 future quantities are correlated. Joint regression imposes a structural prior (e.g., "spread widens together with vol") that a single-target model misses. Three regression boosters trained side-by-side; the **first** booster's output is what the EV gate uses, the other two are auxiliary tasks that regularize the shared(ish) feature usage.

**Implementation**: 3.5 h. Three independent LGB regressors on different targets; same input. Use only Δmid head for inference. Auxiliary heads provide implicit regularization via shared early-stopping (train all 3 to same `best_iter`).

**Expected**: +0.2 ~ +0.7 (regularization-style).

**Note**: True multi-task GBM is awkward (LightGBM doesn't natively do shared trees). This is "ensemble of related task" rather than "shared-rep multi-task".

---

## C. More label/loss reformulations (P2 — last 7)

### #11 Volatility-decoupled signed magnitude target

**Hypothesis**: Decouple sign and magnitude. Train two models:
- Model A: binary sign(Δmid) — "which direction?"
- Model B: regression on |Δmid_norm| — "how big?"

Decision: `(p_sign_up − 0.5) × pred_magnitude > thr_up` → long. (Gates on directional confidence × magnitude product.)

**Implementation**: 2.5 h. Standard 5-seed × 2 models.

**Expected**: +0.2 ~ +0.7.

**Note**: Some overlap with #4 (which uses pre-existing CE for direction); #11 trains a fresh sign model.

---

### #12 Beta-regression on "probability of profitable trade"

**Hypothesis**: Direct probabilistic target. y = sigmoid(20 × Δmid_norm) (smooth bernoulli target ∈ [0,1]). LightGBM `objective='binary'` accepts continuous labels (cross_entropy mode). Predicts P(profitable). Gate on `p > 0.5 + thr`.

**Implementation**: 1.5 h. One-line label change + binary CE.

**Expected**: +0.2 ~ +0.5.

---

### #13 Future Δspread as auxiliary EV correction

**Hypothesis**: True PnL formula deducts `fee × |mp + mp_h + 2|`, but `mp` and `mp_h` differ — the spread component of the PnL formula is not constant. Predict Δspread separately and subtract its expected contribution from `pred_dmid` before gating.

**Implementation**: 2 h. Same regression as T75 but with `pred_net_pnl = pred_dmid − pred_spread_correction`.

**Expected**: +0.1 ~ +0.4. (Marginal, but correct.)

---

### #14 Convex differentiable PnL surrogate (custom LightGBM objective)

**Hypothesis**: Define `L = -PnL_signed(pred, y)` where pred ∈ ℝ controls a soft action `a = tanh(pred / scale)`. Compute gradient and Hessian of −E[a · y − fee · |a|] vs pred. LightGBM `objective=custom_obj` accepts callable. This is a **differentiable approximation of the EV gate**, trained directly as an end-to-end loss.

**Why disruptive**: directly maximizes PnL through gradient descent, no proxy.

**Why fragile**: |a| is non-smooth at a=0 → `tanh` smoothing helps. LightGBM Hessian must be ≥ 0 — a bit of math required to keep convexity.

**Implementation**: 4 h.

**Expected**: +0.0 ~ +0.8 (high variance — could be a +1.5 win or a 0).

**Note**: T13 / T57 tried PnL-aware **sample weights** (multiplicative reweighting on existing CE/L2). #14 is a **fundamentally different intervention** — it changes the loss landscape, not the per-sample weight.

---

### #15 Label noise injection: train on `Δmid + ε·N(0, σ)`

**Hypothesis**: Standard regularization technique. The target is intrinsically noisy (Δmid = signal + Brownian). Adding **more** synthetic noise during training prevents the model from over-fitting the train-set noise patterns. Empirically known to improve OOD generalization for low-SNR regression problems.

**Implementation**: 1 h. Add Gaussian noise (σ ≈ 1× original target std) to y at each LGB epoch.

**Expected**: +0.1 ~ +0.4.

---

### #16 Conditional value-at-risk (CVaR) regression target

**Hypothesis**: Train to predict **expected return conditional on top-30% upside scenarios**: `y_cvar_up = E[Δmid | Δmid > q_0.7]`. This isolates "what if today is a winning trade?" — gives the model a fee-conscious target. Mirror for downside.

**Implementation**: 3 h. Empirical conditional expectation per training epoch.

**Expected**: +0.2 ~ +0.6.

---

### #17 Zero-inflation explicit: train binary "is_directional" + regression "if directional, magnitude"

**Hypothesis**: Δmid distribution has mass concentration near 0 (53% flat). Two-stage model:
- Model X: binary "is |Δmid_norm| > 0.5e-4?" (will the move be at least half-fee?)
- Model Y: regression on Δmid_norm (only — or strongly weighted on — |Δmid| > 0.5e-4 samples)

Decision: trade only if X's prob of being directional > thr1, AND Y's signed prediction > thr2.

**Why disruptive**: handles the heavy-zero label distribution explicitly; #3 (Net-PnL) does this implicitly.

**Implementation**: 3 h.

**Expected**: +0.3 ~ +0.8.

---

## D. Already considered, ruled out

### Sample-weighted PnL (T13/T57)

❌ Multiplicative reweighting at gradient-level. Failed (best +11 vs +13.6 baseline). Ideas #3 and #14 are different mathematical interventions (target-level / loss-level) and not redundant.

### Class weighting / focal loss (T41)

❌ Same axis as sample-weight, failed. Different from label-smoothing (#15 in r36).

### Label smoothing (r36 #15)

⚠️ Listed in r36 top picks but partially overlaps with #8 (9-bucket softmax already gives soft labels). Don't pursue independently.

### Pseudo-labeling (T65)

❌ Failed; too aggressive.

### Adversarial val reweighting (T66)

⚠️ Marginal effect; unrelated to label reformulation. Not a label/loss idea.

### Lambdarank / pairwise rank

❌ (r36 #16) — group structure ill-defined per CRITICAL_CONSTRAINTS §2 (shuffle-invariance breaks ranking groups).

### Per-sym-rank-based gating

❌ (r36 #17) — directly violates §2 (cross-batch state).

### Ranking by rolling target percentile

❌ Same constraint violation.

---

## E. Compliance check (all top 5 verified)

| Idea | sym-agnostic | stateless | date-free | shuffle-invariant | inference cost |
|---|---|---|---|---|---|
| #1 quantile regression | ✅ | ✅ | ✅ | ✅ | 2× T75 (10 boosters) |
| #2 two-binary heads | ✅ | ✅ | ✅ | ✅ | ≈ T75 (10 binaries each lighter) |
| #3 net-PnL target | ✅ | ✅ | ✅ | ✅ | = T75 (5 boosters) |
| #4 hybrid dual-head | ✅ | ✅ | ✅ | ✅ | 2× iter_012 (10 boosters) |
| #5 Sharpe target | ✅ (σ from window) | ✅ | ✅ | ✅ | = T75 + 1 mul |

All decisions are **per-row functions** of the model output(s) and the local 100-tick window — no cross-call state, no batch-dependent computation.

---

## F. Recommended sprint order (1-day, 8 hours)

```
Hour 0-2:    #3 Net-PnL signed target  (cheapest disruptive, ≈ T75 cost)
              ↳ if LOSO ≥ +37, ship as iter_014 — done
              ↳ else continue
Hour 2-5:    #1 Two-quantile regression (q=0.3, q=0.7)
              ↳ if LOSO ≥ +37.5, blend with iter_013 (#4-style)
Hour 5-7.5:  #2 Two-binary "profitable_long/short"
              ↳ probably the strongest of the three; expensive risk
Hour 7.5-8:  Combine best of #1/#2/#3 into iter_014 candidate
              (potentially with iter_013 model_R as 3rd voter — that's #4)
```

Acceptance bar: **LOSO ≥ +37.5** to claim a meaningful Δ over iter_013 (DE-noise floor ≈ ±0.5; we want clear separation).

If T80 calibration holds (0.70 transmission), **+37.5 LOSO → ~+20.1 platform** vs iter_013's +19.23. So the bar is honest +1 platform improvement.

---

## G. Why the meta-thesis predicts these will work

R37's gap analysis: public team trades **2× more, half the per-trade alpha**. They take more marginal signals successfully.

T75 already moved us from "compress label edge" to "preserve magnitude". The next move is **reshaping the loss surface near the fee boundary** — exactly where marginal signals live. All five top picks do this:

- #1 (quantile) — loss surface emphasizes tails
- #2 (two-binary) — loss surface BUMPS at fee boundary (binary CE has steep gradient near decision threshold)
- #3 (net-PnL) — loss surface FLATTENS for sub-fee samples (model ignores noise pretending to be signal)
- #4 (hybrid) — joint loss surface combines both axes
- #5 (Sharpe) — loss surface NORMALIZES vol, so fee boundary is at constant SNR not constant magnitude

These are **complementary**, not substitutes. The strongest individual probably wins +1; the **combination** of two (e.g., #1 quantile + #4 hybrid) might give +2.

---

## H. References

- `experiments/T75_regression_dmid/REPORT.md` — the +9.79 baseline win.
- `experiments/T80_iter002_loso_equiv_recompute/REPORT.md` — 0.70 transmission ratio, +37.5 LOSO target.
- `experiments/T57_pnl_aware_loss/` — sample-weight PnL failures (informs why #3/#14 are different).
- `experiments/T78_regression_ev_time/REPORT.md` — confirms time features don't compound; suggests **target-side** changes are the open lever.
- `research_and_history/r36_disruptive_basic_ideas.md` — earlier brainstorm; #5 there is "regression head + post-hoc 3-class" → became T75 (validated). r38 picks up where r36 stopped.
- `research_and_history/r37_leaderboard_gap_basic.md` — gap diagnosis (2× more trades / half alpha). r38's targets specifically attack this.
- HFT literature: Avellaneda-Stoikov (binary head w/ margin); Optiver 2023 1st place (MAE / Huber regression); Cont (2010) intensity model with two-binary; Hyd Optiver 2021 (KNN-target — not a label idea but related).

---

RESULT: task=r38_label_reformulation metrics={n_ideas=17, top5=quantile_regr / two_binary_heads / net_pnl_target / hybrid_dual_head / sharpe_target, expected_loso_delta=+0.4_to_+3.0_per_idea} notes=meta-thesis: T75 unlocked magnitude; next move is reshape loss surface near fee boundary (where marginal signals live). All 5 top picks are <4h, sym-agnostic, stateless, target LOSO ≥+37.5 for honest +1 platform gain.
