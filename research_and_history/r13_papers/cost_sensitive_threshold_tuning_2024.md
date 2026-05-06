# Cost-Sensitive Threshold Tuning + Bayes Risk Decision Rule

**Why we read it**: Our trading task has an **explicit cost matrix** (fee, gain). Bayes-optimal action is not argmax of probabilities — it's argmax of *expected reward*. Most researchers under-utilize this when they use a generic argmax/threshold post-processor.

---

## The decision-theoretic optimum

Setup for our problem:
- Three actions: `a = 0` (short), `a = 1` (no trade), `a = 2` (long)
- Three states (true forward Δmidprice ranges): `y ∈ {neg, near0, pos}`
- Reward matrix `R[a, y]`: gain if (we take action a) and (truth is y)

```
                    y=neg(short)    y=near0       y=pos(long)
       a=0 (short)   +Δmid_neg-fee    -fee          -fee+Δmid_pos
       a=1 (flat)        0              0              0
       a=2 (long)   -fee+Δmid_neg     -fee          +Δmid_pos-fee
```

In our concrete numbers (fee ≈ 2e-4, typical |Δmid| ≈ 1.6e-4):
```
                    y=neg          y=near0      y=pos
       a=0 (short)  +1.6e-4        -2e-4         -2e-4 - 1.6e-4 ≈ -3.6e-4
       a=1 (flat)   0              0             0
       a=2 (long)   -3.6e-4        -2e-4         +1.6e-4 - 2e-4 ≈ -2e-4 (??? — actually -fee+pos*move)
```

(Numbers are illustrative — actual reward depends on the realized Δmid magnitude, not the binned class.)

**Bayes-optimal rule**: choose `a* = argmax_a Σ_y R[a, y] · P(y | x)`.

For us, `P(y | x)` is the calibrated 3-class probability `p_0, p_1, p_2`. So:

```
EV[short]  = R[0,0]·p_0 + R[0,1]·p_1 + R[0,2]·p_2   ≈ +1.6e-4·p_0 - 2e-4·p_1 - 3.6e-4·p_2
EV[flat]   = 0
EV[long]   = R[2,0]·p_0 + R[2,1]·p_1 + R[2,2]·p_2   ≈ -3.6e-4·p_0 - 2e-4·p_1 + (avg_pos - 2e-4)·p_2
            (approx -3.6e-4·p_0 - 2e-4·p_1 - 0.4e-4·p_2 if pos move ~ 1.6e-4)
a* = argmax of {EV[short], EV[flat], EV[long]}
```

**Notice**: EV[long] is **negative** when pos-move expected return barely covers fee. The optimal rule says "trade only when expected reward > 0", which is more restrictive than `p_top > T`.

---

## Reward matrix in continuous form

The above uses binned y. We can also use continuous Δmid:

```
EV[a = long] = E[Δmid | x] · 1   − fee
EV[a = short] = E[−Δmid | x] · 1 − fee
EV[a = flat] = 0
a* = argmax
```

This is exactly the **expected-return decision rule**. It collapses 3-class classification to a single conditional-expectation regression on Δmid. **A regression head predicting E[Δmid | x] is sufficient**; no class probabilities needed.

But: Δmid is heavy-tailed and noisy. A classifier on `sign(Δmid − fee)` is empirically more robust because it ignores magnitude (which is hard to predict). The *practical* hybrid:

1. Classifier gives `p_0, p_1, p_2`
2. Quantile/regression head gives `μ̂ = E[Δmid | x]` (point prediction)
3. Decision rule: `argmax(EV)` with `R` defined in terms of µ̂ and per-class hit-rates

---

## TunedThresholdClassifierCV (sklearn)

`sklearn.model_selection.TunedThresholdClassifierCV` (1.5+) wraps a binary classifier and finds the **decision threshold that maximizes a user-defined metric** via cross-validation. We can't use it directly (it's binary), but the pattern is cheap to replicate:

```python
def cumulative_pnl(y_true, y_pred, deltamid):
    pnl = np.zeros_like(deltamid, dtype=float)
    pnl[(y_pred == 2)] = deltamid[(y_pred == 2)] - fee
    pnl[(y_pred == 0)] = -deltamid[(y_pred == 0)] - fee
    return pnl.sum()

best = -np.inf
for T in np.linspace(0.4, 0.7, 20):
    for delta in np.linspace(0.0, 0.3, 10):
        y_pred = apply_threshold(p_oof, T, delta)
        pnl = cumulative_pnl(y_oof, y_pred, deltamid_oof)
        if pnl > best:
            best, bestT, bestD = pnl, T, delta
```

This is what we already do (T=0.55, δ=0.10 from grid search). The Bayes-EV approach replaces the grid search with the **direct optimization** in cost-aware form.

---

## Two-dimensional decision region (Höppner et al. 2024)

Recent fraud-detection paper (extends to trading): instead of one threshold T, use a **2-D decision region in (p_long, p_short) space** that maximizes EV. The optimal region is bounded by hyperplanes derived from the cost matrix. For us, this is equivalent to expressing the EV-positive region in the (p_0, p_2) plane:

```
EV[long] > 0    iff    p_2 · (Δmid_pos − fee) > p_0 · |Δmid_neg + fee| + p_1 · fee
                iff    A·p_2 − B·p_0 − C·p_1 > 0    (some constants)
```

This is a hyperplane gating in probability simplex, more flexible than `max(p_0,p_2) > T`. Specifically, it's the same family as our current rule but with **3 parameters per direction (one per class) tied by the cost matrix**, not 2 hand-set knobs.

---

## When does cost-sensitive trump simple thresholding?

Our current rule already implicitly encodes cost-awareness because we tune T on the PnL metric. **But**:

1. **It's a coarse search** — only 2 hyperparams over a 200-point grid.
2. **It assumes symmetric costs for short/long** — fee is symmetric, but the empirical conditional |Δmid| is *not* symmetric across sym (some sym have skewed return distributions).
3. **It doesn't adapt to per-window EV** — every prediction uses the same threshold even if the predicted EV varies wildly.

**Cost-sensitive expected-return decision rule fixes all three** by directly using `R · p` for every test point.

---

## Application to our project

If we have:
- Calibrated `p_0, p_1, p_2` from LightGBM-3class
- Calibrated `µ̂ = E[Δmid | x]` from LightGBM-regression (already in the multi-horizon pipeline)

Then:

```python
def decide(p, mu_hat, fee=2e-4):
    EV_long  = mu_hat - fee     if mu_hat > 0 else -fee     # if µ̂ predicts up, expect to gain µ̂; pay fee in any case
    EV_short = -mu_hat - fee    if mu_hat < 0 else -fee
    # Or: combine with class probabilities
    EV_long_class  = p[2] * E_pos_move - p[0] * E_neg_move - fee
    EV_short_class = p[0] * E_pos_move - p[2] * E_neg_move - fee   # E_pos_move ≈ avg(|Δmid| | y=pos)
    EV_long_combined = 0.5 * EV_long + 0.5 * EV_long_class
    if max(EV_long, EV_short, 0) == EV_long:  return 2
    if max(EV_long, EV_short, 0) == EV_short: return 0
    return 1
```

`E_pos_move` and `E_neg_move` are pre-computed conditional means from training data: `mean(Δmid | y == 2 in train) ≈ +1.6e-4` and `mean(Δmid | y == 0 in train) ≈ −1.6e-4`.

---

## Sources

- [scikit-learn — Post-tuning the decision threshold for cost-sensitive learning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
- [Cost-sensitive thresholding 2-D decision region (Höppner 2024)](https://www.sciencedirect.com/science/article/pii/S0020025523015414)
- [Reproducible ML for Fraud Detection — Cost-sensitive learning](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html)
- [Optimal strategies for reject-option classifiers (JMLR 2023)](https://jmlr.org/papers/volume24/21-0048/21-0048.pdf)
- [Bayes Decision Theory — Duda Hart Stork lecture notes](https://faculty.cc.gatech.edu/~hic/CS7616/pdf/lecture2.pdf)
