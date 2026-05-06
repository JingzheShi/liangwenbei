# Duan, Avati et al. (2020) — NGBoost: Natural Gradient Boosting for Probabilistic Prediction

**Citation**: Duan, T. et al. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction. ICML 2020.
**arXiv**: 1910.03225
**Why we read it**: Drop-in GBDT-style library that outputs **a full distribution per prediction** (mean + variance for normal; α/β for beta; etc.). If we want the model to give us its own **uncertainty estimate** alongside the point prediction, this is the simplest path beyond LightGBM.

---

## What NGBoost predicts

Standard LightGBM regressor: outputs single number `ŷ`.
NGBoost: outputs distribution parameters, e.g. `(μ̂, σ̂)` for Normal — so for each input you get a full `N(μ̂, σ̂²)`.

This means at inference time you can compute:

```
P(Δmid > +α | features) = 1 − Φ((α − μ̂) / σ̂)
P(Δmid < −α | features) = Φ((−α − μ̂) / σ̂)
```

These are the **probabilities of moves big enough to overcome fee** — exactly the quantity we need for trading, more direct than discretized 3-class probabilities.

---

## Algorithm

Three modular components:
- **Base learner**: typically decision trees (depth 3-5), like a vanilla GBDT
- **Distribution**: Normal, LogNormal, Bernoulli, Categorical, Beta, Poisson
- **Scoring rule**: NLL or CRPS (Continuous Ranked Probability Score)

Trained by **natural gradient descent** on distribution parameters (the natural gradient is the steepest descent direction in distribution space, accounting for the geometry of the parameter space — important because distribution parameters like (μ, σ) have very different scales and ordinary gradient descent tends to dominate updates to μ).

---

## Comparison to LightGBM/XGBoost

| | LightGBM | NGBoost |
|---|---|---|
| Output | point prediction | full distribution (mean + var + ...) |
| Speed | very fast | ~5-10× slower |
| Tunability | many hyperparams | similar |
| Library maturity | extreme | moderate (Stanford ML group) |
| Quantile prediction | needs K separate models (one per α) | free from one model |
| API | sklearn-compat | sklearn-compat |

For our 154-feature dataset and ~50k LOSO training rows, NGBoost is feasible (training takes a few minutes per fold).

---

## Application to our setting — Distributional Direct Predictor

Instead of 3-class classification, train NGBoost to predict the **distribution of forward Δmidprice** conditional on the 100-tick window features:

```
NGBoost(X) → (μ̂, σ̂)      # Normal model
P_long  = 1 − Φ((+α − μ̂) / σ̂)   # P(Δmid > +α)
P_short = Φ((−α − μ̂) / σ̂)       # P(Δmid < −α)
P_flat  = 1 − P_long − P_short

Expected return if we go long  = E[Δmid · 1{Δmid > +α}] − fee · 1
Expected return if we go short = E[−Δmid · 1{Δmid < −α}] − fee · 1
Expected return if we abstain  = 0

Optimal action = argmax over {long, short, abstain} of the expected return
```

This is the **decision-theoretic optimal rule** — it directly maximizes expected PnL per trade, no hand-tuned thresholds.

**Caveat**: assumes Normal distribution of forward returns. Empirically high-frequency Δmid is heavy-tailed; consider **t-distribution** or **mixture of normals** distribution in NGBoost. NGBoost supports custom distributions but they need to be derivable.

---

## Quantile alternative (lighter-weight)

If we don't want to install NGBoost, train **3 LightGBM quantile regressors** on `Δmid` with `objective='quantile'`:

```python
model_lo = LGBMRegressor(objective='quantile', alpha=0.1)
model_md = LGBMRegressor(objective='quantile', alpha=0.5)
model_hi = LGBMRegressor(objective='quantile', alpha=0.9)
```

→ at inference get q10, q50, q90 of forward Δmid → compute "P(Δmid > +α)" by interpolation between quantiles. Crude but free.

---

## Sources

- [NGBoost arXiv 1910.03225](https://arxiv.org/abs/1910.03225)
- [Stanford ML Group project page](https://stanfordmlgroup.github.io/projects/ngboost/)
- [GitHub stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost)
- [GeeksforGeeks LightGBM Quantile Regression](https://www.geeksforgeeks.org/machine-learning/lightgbm-for-quantile-regression/)
