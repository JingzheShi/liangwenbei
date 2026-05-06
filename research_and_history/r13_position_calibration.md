# r13 — Probability Calibration + Position Sizing 进阶方案

> **Author**: T13 research worker | **Date**: 2026-05-06
> **Goal**: Find post-processing methods that beat our current `T=0.55, δ=0.10` argmax+gating rule, given LOSO h_10 +21.86 with single-trade pnl ≈ +1.6e-4 vs fee ±2e-4.
>
> **Hard constraint reminder** (from `CRITICAL_CONSTRAINTS.md`): predict output must be discrete `{0, 1, 2}`. Predictor stateless (test order shuffled). sym-agnostic (sym 0–4 may include unseen stocks).
>
> **What this doc is NOT**: a list of NN architectures, feature ideas, or training tricks. Strictly post-processing on top of the trained LightGBM 3-class output.

---

## Executive summary — at a glance

| # | Scheme | Mechanism | Complexity | Expected LOSO Δ | Risk |
|---|---|---|---|---|---|
| **A** | **Temperature scaling (Guo 2017)** | 1-param softmax rescale on val NLL → calibrated p | LOW (10 lines) | **+1 ~ +5** | very low |
| **B** | **Beta calibration (Kull 2017)** per-class | 3 logistic regressions on (log(p), log(1-p)) | LOW (15 lines) | +1 ~ +4 | low |
| **C** | **Isotonic regression** per-class OvR | non-parametric monotone fit on val (s, y) | LOW (sklearn) | +0 ~ +3 | mild overfit |
| **D** | **LdP z-score gating** (replace T+δ) | `z = (p − 1/3)/√(p·(1−p))`; gate on z | LOW (5 lines) | **+2 ~ +6** | very low |
| **E** | **Bayes EV decision rule** (cost matrix) | `argmax_a Σ_y R[a,y]·p_y`; trade if EV > 0 | MED (needs reward matrix) | **+3 ~ +8** | low |
| **F** | **Meta-labeling (LdP §3.6, §10.7)** | secondary classifier on "M1 right?" | HIGH (new model) | +5 ~ +12 | medium (more code, tune) |
| **G** | **Conformal prediction sets** (Mondrian) | abstain unless prediction set is singleton | LOW-MED | **+1 ~ +5** | low |
| **H** | **Multi-horizon agreement gating** | trade only if h_5 + h_10 + h_20 agree | LOW | **+2 ~ +7** | low |
| **I** | **Vol-regime adaptive threshold** | threshold scales with rolling vol | MED | +0 ~ +3 | medium |
| **J** | **Distributional regression head (NGBoost / quantile-LGBM)** | predict P(Δmid > +α), bypass classifier | HIGH (new model) | +2 ~ +8 | medium |

**If only 2 schemes**: **D + E (LdP z-gating + Bayes EV)** — both are 1-day implementations on top of existing pipeline, both are hyperparameter-light, and both have clear theoretical reasons why they should beat hand-tuned T+δ. **D** alone is cheaper than calibration (no held-out fit) and may be enough; **E** is the principled upgrade. See §11.

---

## 0. Background — why current T=0.55, δ=0.10 is suboptimal

Our current rule:

```
pred = 2  if p_2 > 0.55 AND p_2 > p_1 + 0.10
pred = 0  if p_0 > 0.55 AND p_0 > p_1 + 0.10
pred = 1  otherwise
```

This rule has 4 weaknesses:

1. **Raw `p_k` from LightGBM is not calibrated** — the model's "60%" is not actually 60% empirical hit rate. Trees with class-balanced reweighting are systematically over-confident on minority classes. → addressed by §1, §2, §3.
2. **Threshold ignores class prior offset and Bernoulli variance** — same `p_2 = 0.55` means very different things when `p_1 = 0.40` (high uncertainty) vs `p_1 = 0.05` (clean signal). → addressed by §4.
3. **Treats all observations the same** — high-vol regime trades have different EV than low-vol; the rule doesn't adapt. → addressed by §6, §9.
4. **Discards multi-horizon info** — h_10 is currently used alone even though we have h_5/h_20 from the same backbone. → addressed by §8.

Each of §1–§9 below tackles one of these.

---

## 1. Probability calibration — Scheme A: Temperature scaling

### The math

LightGBM 3-class outputs raw scores `z = (z_0, z_1, z_2) ∈ R^3` per sample (use `model.predict(X, raw_score=True)`). Default softmax: `p = exp(z) / Σ exp(z_k)`.

Temperature scaling applies a single scalar `T > 0`:

```
p_calibrated = softmax(z / T)
T* = argmin_T  -Σ_i log [softmax(z_i / T)]_{y_i}      # NLL on val set
```

Convex 1-D optimization (5 lines with `scipy.optimize.minimize_scalar`).

### Properties

- **Argmax preserved**: dividing all logits by the same T does not change which class is largest. Existing argmax-based predictions are byte-identical. **Only the probabilities change** — meaning the threshold T=0.55 has a different operational meaning (now it actually corresponds to ~55% empirical accuracy after calibration).
- **Universal**: works for any number of classes, doesn't assume calibration curve shape.
- **Cheap**: `T*` is a single number, fit in milliseconds.

### Implementation on our pipeline

```python
# After LightGBM training
val_z = model.predict(val_X, raw_score=True)         # (n_val, 3)
val_y = val_labels                                    # (n_val,)

from scipy.optimize import minimize_scalar
def nll(T):
    log_p = (val_z / T) - logsumexp(val_z / T, axis=1, keepdims=True)
    return -log_p[np.arange(len(val_y)), val_y].mean()
res = minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
T_star = res.x

# At inference:
def predict_calibrated(X):
    z = model.predict(X, raw_score=True)
    p = softmax(z / T_star, axis=1)
    return apply_threshold_gating(p, T=0.55, delta=0.10)
```

### Compatibility with hard constraints

✅ Predictor stateless (T_star is a constant in __init__)
✅ sym-agnostic (T_star fit globally on all sym in val)
✅ output is discrete

### Source

- [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) — On Calibration of Modern Neural Networks
- See `r13_papers/guo_temperature_scaling_2017.md`

### Expected gain

If LightGBM is over-confident by ~10% on extremes (typical for class-balanced GBDT), `T*` ≈ 1.2-1.5. After calibration, our existing T=0.55, δ=0.10 grid was tuned on **mis-calibrated** probabilities; re-grid-searching on calibrated probabilities should find a slightly different (T_new, δ_new) that PnL-dominates the old. Realistic LOSO h_10 gain: **+1 ~ +5**.

### Complexity / Implementation cost

**1 day**: 30-line patch to training/predictor + grid-search re-run. Almost zero risk because argmax is preserved (worst case: identical predictions to current).

---

## 2. Probability calibration — Scheme B: Beta calibration

### The math (Kull 2017)

Per class `k` (one-vs-rest):

```
score: s_k = p_k_raw                              # uncalibrated softmax output
features: x1 = log(s_k), x2 = -log(1 - s_k)
fit: logistic regression  log[p̂/(1−p̂)] = c + a·x1 + b·x2     on  y_binary = (y == k)
```

Output `(a_k, b_k, c_k)` per class. At inference: 3 logistic regression evaluations, then renormalize.

### Why over Platt

Platt assumes per-class scores are normally distributed. GBDT output is **bimodal** (pushes to 0 / 1). Beta calibration fits a richer family that includes Platt + asymmetric distortions. Empirically improves Brier score 5-15% over Platt on tabular data.

### Implementation

Use the [betacal pip package](https://betacal.github.io/) or hand-roll (10 lines, sklearn LogisticRegression).

### Source

- [Kull et al. 2017](https://proceedings.mlr.press/v54/kull17a.html)
- See `r13_papers/kull_beta_calibration_2017.md`

### When to prefer over Scheme A

- If our reliability diagram on val shows non-sigmoidal distortion (likely for class 1 / "flat" given heavy class-imbalance reweighting)
- If we have ≥1k val samples per class (we do)
- If we want **per-class** calibration that lets `p_2` and `p_0` be calibrated differently from `p_1`

Schemes A and B can be combined: temperature first (global), then beta per-class. Diminishing returns but not zero.

### Expected gain

**+1 ~ +4**, slightly less than A because A captures the dominant calibration error (over-confidence). B fixes residual asymmetric distortion.

### Complexity

**1 day** with `betacal` package. Very low risk.

---

## 3. Probability calibration — Scheme C: Isotonic regression OvR

### The math

Non-parametric. For class k:

```
fit f_k: monotone non-decreasing s.t. minimize Σ_i (1{y_i == k} - f_k(s_k_i))^2
                                                 subject to f_k monotone increasing
```

PAV (pool-adjacent-violators) algorithm — O(n log n).

### Properties

- Most powerful family (any monotonic distortion correctable)
- Prone to overfit on small calibration sets — needs ≥1k samples per class
- May affect ranking metrics (ties from step function)
- Multi-class: per-class OvR + renormalize

### Source

- [scikit-learn `CalibratedClassifierCV` docs](https://scikit-learn.org/stable/modules/calibration.html) (method='isotonic')

### When to use

- If we have lots of calibration data (~50k/class in our LOSO setup is plenty)
- If reliability diagram shows highly non-monotonic-sigmoidal distortion
- **Don't use** if val set is <1k samples per class

Generally **B (beta) is the safer bet** because it has 3 params instead of "many breakpoints" — less risk of overfitting on a particular val fold.

### Expected gain

Similar to B (+0 ~ +3), with slightly higher variance.

---

## 4. Decision-theoretic gating — Scheme D: LdP z-score (replace T+δ)

**This is the cheapest "obvious upgrade" to our current rule.**

### The math (Lopez de Prado 2018, AFML §10.3)

For a 3-class classifier, with `p_top = max(p_0, p_2)` and `side = sign(argmax)`:

```
z = (p_top − 1/3) / sqrt(p_top · (1 − p_top))
m = side · (2 · Φ(z) − 1)              # bet size in [-1, +1]
pred = 2  if m > +threshold_m
pred = 0  if m < −threshold_m
pred = 1  otherwise
```

where `Φ` is the standard normal CDF.

### Why over T+δ

Our current rule treats `p_top = 0.55` and `p_top = 0.95` the same (both above T). LdP's rule:
- `p_top = 0.55`: `z = (0.55 - 0.333)/sqrt(0.55·0.45) = 0.217/0.497 = 0.436` → `m = 0.337` → small bet
- `p_top = 0.95`: `z = (0.95 - 0.333)/sqrt(0.95·0.05) = 0.617/0.218 = 2.83` → `m = 0.995` → near-full bet

The `m` value naturally separates **weak** from **strong** signals on the same probability scale, accounting for **class prior** (1/3 baseline) and **Bernoulli variance** (`p(1-p)` is max at p=0.5, so threshold automatically becomes more conservative there).

For our **discrete 0/1/2** output, we don't actually use the continuous `m`; we threshold it. **But** we can choose `threshold_m` directly in m-units, which is more interpretable than (T, δ) pair: `threshold_m = 0.5` means "bet only when m > 50% of full conviction".

### Implementation

```python
import numpy as np
from scipy.stats import norm

def ldp_predict(p, threshold_m=0.5):
    """p: shape (n, 3). Returns shape (n,)."""
    K = 3
    p_top = np.max(p[:, [0, 2]], axis=1)
    side = np.where(p[:, 2] > p[:, 0], +1, -1)        # +1 if long, -1 if short
    eps = 1e-6
    z = (p_top - 1.0/K) / np.sqrt(np.maximum(p_top * (1 - p_top), eps))
    m = side * (2 * norm.cdf(z) - 1)
    pred = np.full(len(p), 1)                          # default flat
    pred[m > threshold_m] = 2
    pred[m < -threshold_m] = 0
    return pred
```

**Tune `threshold_m`** via grid search on LOSO OOF PnL (same way we tuned T, δ).

### Compatibility

✅ stateless ✅ sym-agnostic ✅ discrete output

### Source

- López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 10 §10.3
- See `r13_papers/lopez_de_prado_bet_sizing_ch10.md`
- [MQL5 Article 21824](https://www.mql5.com/en/articles/21824)

### Expected gain

**+2 ~ +6** on LOSO h_10. The z-score adjustment for class prior is essentially free; the variance adjustment helps in the hard regime where `p` is moderately above prior but uncertain.

### Complexity

**1 day**. ~20-line patch to predictor. **No retraining required**. **Extremely safe** — degenerate case (T=0.55, δ=0.10) maps to a specific `threshold_m` value that we can pin if we want to fall back.

---

## 5. Decision-theoretic gating — Scheme E: Bayes EV with cost matrix

### The math

Reward matrix `R[a, y]` over actions × states (see `r13_papers/cost_sensitive_threshold_tuning_2024.md` for full table). For our HFT setting:

```
Estimate from training data:
    E_pos_move  := mean(Δmid | y == 2)              # avg up-move when label is "long"
    E_neg_move  := mean(Δmid | y == 0)              # avg down-move when label is "short"  (negative number)
    fee = 2e-4

R[a=long]:  R[long, neg=0]  = E_neg_move - fee     (loss: down-move costs us)
            R[long, flat=1] = -fee                  (no move, pay fee)
            R[long, pos=2]  = E_pos_move - fee     (gain - fee)
R[a=short]: R[short, neg=0] = -E_neg_move - fee    (gain on down-move)
            R[short, flat=1]= -fee
            R[short, pos=2] = -E_pos_move - fee    (negative: up-move costs us)
R[a=flat]:  R[flat, *] = 0                          (no trade, no PnL)

Decision rule:
    EV[a] = Σ_y R[a, y] · p_y
    a* = argmax_a EV[a]
```

### Why over D

D's `m` is monotone in `p_top` and ignores `p_1`. E uses **full probability vector** including `p_1` (flat probability) — natural because flat is an action with `EV = 0`, and the cost matrix tells us EV[long] vs EV[flat] not just on `p_2` but on whether "high `p_1`" pulls EV down because flat is heavily weighted.

### Implementation

```python
def bayes_ev_predict(p, R):
    """p: (n,3). R: (3,3) reward matrix R[a,y]."""
    EV = p @ R.T                                       # (n, 3)
    return np.argmax(EV, axis=1)
```

3 lines. Pre-compute R once from train-set conditional means.

### Refinement: per-window E_pos/E_neg via regression head

If we add a separate regression head predicting `µ̂ = E[Δmid | x]`:

```
EV[long] = µ̂ - fee
EV[short] = -µ̂ - fee
EV[flat] = 0
```

Even simpler — just compare `|µ̂|` to `fee`. **This is the cleanest decision rule**: trade if and only if predicted alpha exceeds round-trip transaction cost.

### Source

- [scikit-learn cost-sensitive tuning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
- See `r13_papers/cost_sensitive_threshold_tuning_2024.md`
- Bayes Decision Theory — [Duda, Hart, Stork lecture notes](https://faculty.cc.gatech.edu/~hic/CS7616/pdf/lecture2.pdf)

### Expected gain

**+3 ~ +8** on LOSO h_10. The biggest win comes from auto-handling asymmetric `E_pos_move` vs `E_neg_move` (some sym have skewed return distributions; the rule auto-adjusts).

### Complexity

**2 days** if we add the regression head; **0.5 day** if we use only `R · p` with empirical means. Risk: low (degenerate to current behavior with right matrix).

### Combinable with calibration (A/B/C)

E uses `p_y` as input. Calibrating p first (Schemes A/B/C) makes E's decisions more accurate. **Recommended pipeline**: A → E.

---

## 6. Meta-labeling — Scheme F: secondary classifier (LdP §3.6 + §10.7)

### Setup

1. **Primary M1**: existing LightGBM-3class on (X, y_3class). Output: `pred_M1, p_M1`.
2. **Meta-target**: on out-of-fold predictions, compute `was_M1_right = 1[realized PnL of M1 trade > 0]`. Only defined when M1 made a directional bet (pred_M1 ∈ {0, 2}).
3. **Secondary M2**: binary classifier on (X, M1_features, M1_confidence) → predicts `q = P(M1 is right)`.
4. **Decision**: trade only if `q > q_threshold` AND `pred_M1 != 1`. Use M1's direction; M2's confidence determines whether to act.

### Why this works

M1 and M2 see different problems:
- M1 minimizes 3-class log-loss on direction.
- M2 sees "given M1 says long, when is it right?" — has access to M1's own confidence and can learn "M1 fails when X" patterns.

Importantly, M2 has access to **M1's own probability vector as a feature**, plus all the LOB features. M2 effectively learns a **non-linear combination** of `(p_M1, X)` for the gating decision — strictly more expressive than any threshold on p_M1.

### Implementation outline

```python
# Phase 1: train M1, get OOF predictions
m1_oof_preds, m1_oof_probs = oof_predict(M1, X_train, y_train_3class)

# Phase 2: build meta-target (only on rows where M1 made a directional bet)
mask = m1_oof_preds != 1
M1_right = (m1_oof_preds[mask] == y_train_3class[mask]).astype(int)
# OR: use realized PnL directly
M1_pnl = compute_pnl(m1_oof_preds, deltamid_train)
M1_profitable = (M1_pnl > 0).astype(int)
y_meta = M1_profitable[mask]
X_meta = np.hstack([X_train[mask], m1_oof_probs[mask]])

# Phase 3: train M2 (binary classifier)
M2 = LGBMClassifier(...)
M2.fit(X_meta, y_meta)

# Phase 4: at inference
def predict(x):
    p_M1 = M1.predict_proba(x.reshape(1, -1))[0]
    pred_M1 = np.argmax(p_M1)
    if pred_M1 == 1:                                  # M1 says flat -> we agree
        return 1
    x_meta = np.hstack([x, p_M1])
    q = M2.predict_proba(x_meta.reshape(1, -1))[0, 1]
    if q > q_threshold:
        return pred_M1
    else:
        return 1
```

### Compatibility

✅ stateless (M2 weights are frozen)
✅ sym-agnostic (M2 doesn't use sym either)
✅ discrete output

### Source

- [Lopez de Prado AFML §3.6, §10.7](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [Wikipedia Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling)
- [Hudson & Thames meta-labeling article](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- See `r13_papers/lopez_de_prado_bet_sizing_ch10.md`

### Expected gain

**+5 ~ +12** on LOSO h_10. Highest expected benefit because M2 can use signal that M1 doesn't (e.g. spread quantile, regime indicators) to filter false-confidence trades.

### Complexity

**2-3 days**. Need OOF prediction infrastructure (we have it for LOSO), need to train and tune M2, need to grid-search `q_threshold`. Medium risk: M2 can overfit if its features are too colinear with M1's; mitigate with strong regularization + drop M1's softmax entropy as M2 input.

---

## 7. Conformal prediction sets — Scheme G

### The math (split conformal, APS variant)

```
Given: trained M1 + held-out cal set (x_i, y_i)_{i=1..n}, miscoverage α
Step 1: s_i = 1 - p̂(y_i | x_i)                    # non-conformity scores
Step 2: q̂ = (⌈(n+1)·(1-α)⌉)-th smallest of {s_i}
Step 3: at test:  C(x_test) = {y : p̂(y | x_test) ≥ 1 - q̂}

Coverage guarantee: P(y_test ∈ C(x_test)) ≥ 1-α
```

For our 3-class:

| C(x_test) | action |
|---|---|
| {0} | pred = 0 (short) |
| {2} | pred = 2 (long) |
| {1} | pred = 1 (flat) |
| {0, 1}, {1, 2}, {0, 2}, {0,1,2} | pred = 1 (abstain) |

### Mondrian variant — for distribution shift (LOSO!)

Compute separate `q̂_g` per group `g` (e.g. per sym, or per vol-regime tier):

```
q̂_g = (1-α) quantile of {s_i : group(i) = g}
```

This gives **group-conditional coverage** — exactly what we want for LOSO where one sym is held out and we don't trust global calibration. **Caveat**: the test sym may be a new sym (CRITICAL_CONSTRAINTS §3); to handle this, group on **observable signals** like rolling spread quantile or rolling vol decile, not sym ID.

### Implementation

```python
# Cal phase
val_p = M1.predict_proba(val_X)
s_cal = 1 - val_p[np.arange(len(val_y)), val_y]
q_hat = np.quantile(s_cal, 1 - alpha, method='higher')

# Inference
def conformal_predict(x):
    p = M1.predict_proba(x.reshape(1, -1))[0]
    pred_set = np.where(p >= 1 - q_hat)[0].tolist()
    if pred_set == [0]: return 0
    if pred_set == [2]: return 2
    return 1                                          # singleton-{1} or any non-singleton-direction
```

### Source

- [BBVA AI Factory — Conformal Prediction intro](https://www.bbvaaifactory.com/conformal-prediction-an-introduction-to-measuring-uncertainty/)
- [Selective Conformal Risk Control (arXiv 2512.12844)](https://www.arxiv.org/pdf/2512.12844)
- See `r13_papers/conformal_prediction_2024.md`

### Expected gain

**+1 ~ +5**. The principle is sound but in our 3-class setup the gains are marginal vs Schemes D/E because there are only ~3-4 distinct prediction-set patterns, so it's effectively a particular threshold rule. Mondrian-conditional version should help LOSO more (+2 ~ +5 vs +1 ~ +3 for vanilla).

### Complexity

**1 day** vanilla, **2 days** for Mondrian (need to bin val data by regime).

---

## 8. Multi-horizon agreement — Scheme H

### Idea

We have h_5, h_10, h_20, h_40, h_60 trained from the same backbone (or separate). Even though the platform rule is "best of 5 = the strongest single horizon", we can use **agreement across horizons** as a confidence signal *for our chosen horizon*.

For each test point, evaluate all 5 horizons → 5 predictions in `{0, 1, 2}`. Define:

```
agree_long  = sum(pred_h == 2 for h in horizons)
agree_short = sum(pred_h == 0 for h in horizons)
agree_flat  = 5 - agree_long - agree_short
```

**Gating rule**:

```
final = pred_h_10                                    # primary horizon
if agree_long < 3 AND agree_short < 3:               # no horizon majority
    final = 1                                        # abstain
```

Or stronger:

```
if pred_h_10 == 2 AND agree_long < 3:  final = 1     # not enough cross-horizon support for long
if pred_h_10 == 0 AND agree_short < 3: final = 1     # ditto short
```

### Variant — soft agreement on probability

Instead of hard predictions, average softmax probabilities across horizons:

```
p_avg = (p_h5 + p_h10 + p_h20) / 3                   # only short horizons (long ones diverge in regime)
apply standard threshold gating to p_avg
```

This is **ensemble averaging across horizons**. Different from regular ensemble (different models on same target) — here we average **different targets' probabilities** that should agree on direction in the persistent-trend case.

### Compatibility

✅ stateless ✅ sym-agnostic ✅ discrete output. Requires running all 5 horizon models at predict time → 5× inference cost. With LightGBM this is still milliseconds.

### Expected gain

**+2 ~ +7**. The gain comes from filtering "noisy h_10 conviction" — when h_10 says long but no other horizon agrees, the signal is likely a fluke and we shouldn't trade.

**Key risk**: requires that we ship **all 5 horizon models** in the submission zip (size budget). Currently iter_002 only ships h_10 (~1.6MB). All 5 → ~8MB still fine.

### Complexity

**1 day** if all 5 models already trained (we have them). Just modify Predictor to load all 5 and compute agreement.

### Source

- [MDPI — Ensemble Multi-Expert Forecasting (2025)](https://www.mdpi.com/1911-8074/18/6/296)
- [Adaptive Ensemble Learning Hyper-network framework](https://www.mdpi.com/2075-1680/14/8/597)

---

## 9. Volatility-regime adaptive threshold — Scheme I

### Idea

At each test point, compute a **rolling volatility** from the 100-tick window itself (we have it for free):

```
sigma_window = std(diff(midprice_window))            # tick-to-tick mid std
```

Then scale threshold:

```
T_adaptive = T_base + γ · max(0, sigma_window - sigma_median)
delta_adaptive = delta_base * (1 + max(0, sigma_window/sigma_median - 1))
```

Higher vol → tighter gate → trade less. Lower vol → looser gate → trade more (but be careful: low vol might mean low |Δmid| in absolute terms, also bad).

### Refinement — vol-normalized expected reward

Better: use vol to **rescale the EV computation in Scheme E**. If `sigma_window` is high, expected `|Δmid|` is also high → fee is **proportionally smaller** → more trades EV-positive.

```
expected_pos_move = E_pos_move_base * (sigma_window / sigma_median)    # rough
EV[long] ≈ p_2 · (expected_pos_move - fee) + ...
```

### Compatibility

✅ stateless (sigma_window is from current window only) ✅ sym-agnostic (no sym in formula)

### Risk

- **Overfitting `γ`** — we'd be tuning a new hyperparameter from limited LOSO folds.
- **High-vol regimes are often low-edge** (noisy regime) — the relationship between vol and per-trade alpha is empirical, not always positive.

### Source

- [QuantifiedStrategies — Volatility-Based Position Sizing](https://www.quantifiedstrategies.com/volatility-based-position-sizing/)
- [LSEG — Market regime detection](https://developers.lseg.com/en/article-catalog/article/market-regime-detection)
- [Order Book Filtration and Directional Signal Extraction (arXiv 2507.22712)](https://arxiv.org/html/2507.22712v1)

### Expected gain

**+0 ~ +3**, more variance than other schemes. Good for **risk management** (smaller drawdown variance) but not necessarily a +PnL lever.

### Complexity

**2 days**. Tune γ via grid on LOSO PnL.

---

## 10. Distributional regression head — Scheme J

### Idea

Instead of the 3-class classifier, train a regressor that predicts the **distribution** of forward Δmid:

**Option J1 — Quantile regression LightGBM**:
```python
model_q10 = LGBMRegressor(objective='quantile', alpha=0.10).fit(X, deltamid_h10)
model_q50 = LGBMRegressor(objective='quantile', alpha=0.50).fit(X, deltamid_h10)
model_q90 = LGBMRegressor(objective='quantile', alpha=0.90).fit(X, deltamid_h10)
```

**Option J2 — NGBoost**:
```python
from ngboost import NGBRegressor
from ngboost.distns import Normal
model = NGBRegressor(Dist=Normal).fit(X, deltamid_h10)    # outputs (mu, sigma)
```

**Decision rule**:

```
P_long  = 1 - Φ((+α - mu) / sigma)              # P(Δmid > +α)  [Normal assumption]
P_short = Φ((-α - mu) / sigma)
EV[long]  = mu - fee   if mu > 0   else  -fee
EV[short] = -mu - fee  if mu < 0   else  -fee
trade if max(EV) > 0
```

### Why over Scheme E

Scheme E uses **discrete** class probabilities (3 classes from binning Δmid into ±α). Scheme J uses **continuous** Δmid prediction → directly predicts the quantity we care about (alpha), no info-loss from discretization.

### Hard-constraint compatibility

✅ stateless ✅ sym-agnostic
⚠️ requires retraining a new model — bigger lift than Schemes A-G

### Risks

- High-frequency Δmid is **heavy-tailed**. Normal assumption in NGBoost will mis-estimate tail probabilities. Use `LogNormal` or `t-distribution` (NGBoost supports both) or use quantile approach (no distributional assumption).
- Quantile regression in LightGBM doesn't share trees across quantiles → **3× model size**. Submission zip budget (size limits unknown but conservative) may be tight.

### Source

- [Duan et al. 2020 NGBoost](https://arxiv.org/abs/1910.03225)
- [LightGBM Quantile Regression](https://www.geeksforgeeks.org/machine-learning/lightgbm-for-quantile-regression/)
- See `r13_papers/duan_ngboost_2020.md`

### Expected gain

**+2 ~ +8**. Compelling because it directly optimizes the regression target (Δmid) instead of a thresholded version, which retains more information.

### Complexity

**3-4 days**. New model, new training pipeline. Risk: medium (model may not converge or may overfit; LightGBM regression on noisy returns is hard).

---

## 11. Final recommendation — if only 2 schemes, try these

### #1 priority — Scheme D: LdP z-score gating

**Why first**: cheapest to implement (20 LoC, no retrain), highest "free lunch" potential. Replaces `T+δ` with a single `threshold_m` that has cleaner theoretical meaning. **No risk of regression** — degenerate parameter values reproduce current behavior.

**Recipe**:
```python
# In Predictor.predict, replace existing thresholding with:
import numpy as np
from scipy.stats import norm
def ldp_gate(p, threshold_m=0.5):
    K = 3
    p_top = np.max(p[:, [0, 2]], axis=1)
    side  = np.where(p[:, 2] > p[:, 0], +1, -1)
    z = (p_top - 1.0/K) / np.sqrt(np.clip(p_top * (1 - p_top), 1e-6, None))
    m = side * (2 * norm.cdf(z) - 1)
    return np.where(m > threshold_m, 2, np.where(m < -threshold_m, 0, 1))
```
Tune `threshold_m` on LOSO OOF PnL.

**Validation hook**: at threshold_m = 0.36, behavior should approximately match T=0.55, δ=0.10 (reverse-engineered). Make sure it does on val before trusting LOSO numbers.

---

### #2 priority — Scheme E: Bayes EV decision rule

**Why second**: principled, clean upgrade. Replaces "argmax of p" with "argmax of expected reward" — this is what the math says we should do. Combines naturally with calibration (Scheme A) for stronger signal.

**Minimal recipe**:
```python
# Pre-compute on training data:
E_pos = mean(deltamid_train[y_train == 2])     # avg up-move when label was up
E_neg = mean(deltamid_train[y_train == 0])     # avg down-move (negative)
fee = 2e-4

# Reward matrix (rows = action, cols = true class)
R = np.array([
    [-E_neg - fee, -fee, -E_pos - fee],   # a=0 short
    [0,            0,    0          ],    # a=1 flat
    [+E_neg - fee, -fee, +E_pos - fee],   # a=2 long
])

# At inference:
def bayes_ev_predict(p):
    EV = p @ R.T                                # (n,3) @ (3,3) -> (n,3)
    return np.argmax(EV, axis=1)
```

**Even more minimal**: train a regression head predicting `µ̂ = E[Δmid | x]`. Then `pred = 2 if µ̂ > +fee else 0 if µ̂ < -fee else 1`. This bypasses class-probability calibration entirely.

---

### Combined recipe (recommended sequence)

1. **Step 1** (1 day): Implement Scheme A (temperature scaling) — get calibrated `p`. Re-grid (T, δ) on calibrated p; if PnL improves, ship.
2. **Step 2** (1 day): Replace gate with Scheme D (LdP z-score). Grid `threshold_m`. Compare LOSO PnL.
3. **Step 3** (2 days): Implement Scheme E (Bayes EV). Compare to D. Pick winner.
4. **Step 4** (3 days, if 1-3 hit ceiling): Implement Scheme F (meta-labeling).

Total: ~7 days for tier 1 (A+D+E). Each step is a single ablation that either ships or rolls back.

---

## 12. What we did NOT recommend (and why)

| Idea | Why skipped |
|---|---|
| Direct Kelly bet sizing | Platform requires discrete 0/1/2 — Kelly's continuous fraction can't ship; at best we use Kelly to set the threshold (already implicit in Scheme E). |
| Full distributional ensemble (Bayesian model averaging) | Submission zip size budget; computational cost at predict time. Marginal vs Scheme F. |
| Online calibration (recalibrate per session) | Predictor stateless → cannot maintain online statistics across calls. Hard constraint. |
| Per-sym calibration | Sym is unreliable at test time (CRITICAL_CONSTRAINTS §3 — sym 0-4 may be new stocks). Use observable regime instead (Scheme I). |
| Stacking M1's class probs into a deep meta-net | Overkill; meta-labeling (Scheme F) gives 80% of the benefit at 20% of the complexity. |

---

## 13. Bibliography (paper summaries in `r13_papers/`)

- `r13_papers/guo_temperature_scaling_2017.md` — Guo et al. 2017
- `r13_papers/kull_beta_calibration_2017.md` — Kull et al. 2017
- `r13_papers/lopez_de_prado_bet_sizing_ch10.md` — AFML Ch10
- `r13_papers/duan_ngboost_2020.md` — NGBoost
- `r13_papers/conformal_prediction_2024.md` — Conformal + selective prediction
- `r13_papers/cost_sensitive_threshold_tuning_2024.md` — Bayes EV decision rule

## 14. Sources cited

- [Guo et al. 2017 — On Calibration of Modern NN](https://arxiv.org/abs/1706.04599)
- [Kull et al. 2017 — Beta calibration](https://proceedings.mlr.press/v54/kull17a.html)
- [scikit-learn Probability Calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- [Duan et al. 2020 — NGBoost](https://arxiv.org/abs/1910.03225)
- [Lopez de Prado AFML notes (Reasonable Deviations)](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [MQL5 Article 21824 — Bet Sizing for Financial ML](https://www.mql5.com/en/articles/21824)
- [Hudson & Thames — Meta-labeling](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Wikipedia — Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling)
- [BBVA AI Factory — Conformal Prediction](https://www.bbvaaifactory.com/conformal-prediction-an-introduction-to-measuring-uncertainty/)
- [Selective Conformal Risk Control (arXiv 2512.12844)](https://www.arxiv.org/pdf/2512.12844)
- [scikit-learn — Cost-sensitive threshold tuning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
- [Cost-sensitive 2D thresholding (Höppner 2024)](https://www.sciencedirect.com/science/article/pii/S0020025523015414)
- [LightGBM Quantile Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/lightgbm-for-quantile-regression/)
- [QuantifiedStrategies — Volatility-Based Position Sizing](https://www.quantifiedstrategies.com/volatility-based-position-sizing/)
- [Order Book Filtration & Directional Signal (arXiv 2507.22712)](https://arxiv.org/html/2507.22712v1)
- [MDPI — Ensemble Multi-Expert Forecasting](https://www.mdpi.com/1911-8074/18/6/296)
