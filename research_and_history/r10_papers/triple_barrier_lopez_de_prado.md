# Triple-Barrier Method — López de Prado, *Advances in Financial Machine Learning*, Ch 3
**Coined**: 2018 in López de Prado's book (Wiley)
URLs:
* mlfinpy: https://mlfinpy.readthedocs.io/en/latest/Labelling.html
* Quantreo newsletter: https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco
* Hudson & Thames meta-labeling: https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/

## Summary

A **labelling** technique (not a loss) that creates training labels reflecting
real trading outcomes. Three barriers per sample t:

| Barrier | Type | Trigger |
|---|---|---|
| Upper | profit-taking | price hits `+α·σ_t` first → label +1 |
| Lower | stop-loss | price hits `-α·σ_t` first → label -1 |
| Vertical | timeout | neither hit within `n_max` ticks → label 0 |

`σ_t` is the EWMA of recent returns volatility — so barriers are
**volatility-adjusted** (heteroscedasticity-aware).

## Why it matters for our problem

Our current labels (`label_h` from competition) use a **fixed alpha threshold**
on `Δp_{t+h}`. Specifically (verified from `examples/`):
```
label = 0 if Δp < -α
        = 2 if Δp > +α
        = 1 otherwise
```
where α is some fixed constant.

This is the "naive fixed-horizon labelling" that López de Prado argues
**underperforms** triple-barrier:

* Fixed α ignores that `Δp` is heteroscedastic — same |Δp| means different
  things in low-vol vs high-vol regimes.
* Fixed h ignores that a winning trade might have hit "+2·fee" at h=20 then
  fully reversed by h=60 (path-dependent outcome).

## Cannot apply directly — but can extend a feature

We **cannot relabel** the competition data — `label_h` is given. But we CAN
use triple-barrier-style thinking to:

1. **Build new features** like "would a profit-take of 2·fee have been hit
   before timeout" — this is essentially "first-passage time" features.
2. **Build per-sample weights** from local realised volatility:
   `sample_w_t = |Δp_t| / σ_t^{ewma}`
   This is a **vol-normalised** version of S1 from `r10_pnl_loss.md`. May
   work better than raw `|Δp|` in heteroscedastic regimes (sym=2 specifically).

## Empirical evidence

López de Prado Ch 3 reports that triple-barrier labelling improves out-of-sample
accuracy and Sharpe of ML models by ~5-15% over fixed-horizon labelling on
S&P futures.

Genetic-algorithm extensions (Park et al. 2024,
https://www.mdpi.com/2227-7390/12/5/780) push this to +25% accuracy in
crypto pair-trading.

## What we'd borrow

1. **Vol-normalised sample weights** as a refinement of S1 — try both raw |Δp|
   and |Δp|/σ as competing sample-weight schemes.
2. **First-passage features**: for each (sym, t), compute "did mid hit
   ±2·fee before h=10 ticks?" as a binary feature. Cheap NumPy.

## What we'd skip

* Replacing the competition labels — not allowed.
* Genetic-algorithm tuning of barriers — overkill at this stage.
