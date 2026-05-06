# Lopez de Prado (2018) — Advances in Financial ML, Chapter 10: Bet Sizing

**Citation**: López de Prado, M. (2018). Advances in Financial Machine Learning, Wiley. Chapter 10: Bet Sizing.
**Why we read it**: This is the canonical reference for **converting classifier confidence into a position size** in quant finance. Defines the math behind "predict 0/1/2 with margin" and bridges to Kelly / cost-aware sizing.

---

## The core problem

A classifier outputs `p_k = P(class = k | features)`. Naive argmax discards confidence. A naive thresholding (T=0.55) is binary on / off and discards the "how much" information. Lopez de Prado (LdP) gives a principled mapping from `p` to a continuous bet size in `[-1, +1]`.

---

## Probability-to-bet-size formula (LdP §10.3)

For a primary classifier predicting one of `K` classes with probability `p` for the *predicted* class:

**z-statistic**:

```
z = (p − 1/K) / sqrt(p · (1 − p))
```

`p − 1/K` is the deviation from the uniform prior (random guess); `sqrt(p(1−p))` is the Bernoulli SD. So `z` is a "signal-to-noise" ratio in std-units.

**Bet size** (signed, in [-1, +1]):

```
m = side · (2 · Φ(z) − 1)
```

where `Φ` is the standard normal CDF and `side ∈ {-1, +1}` is the predicted direction (LdP works in 2-class +1/-1 framing; for our 3-class, `side = +1` if argmax = 2 (long), `−1` if argmax = 0 (short), and we don't trade if argmax = 1).

**Properties**:
- `p = 1/K` (random guess) → `z = 0` → `m = 0` (no bet)
- `p = 1` (max conviction) → `z → ∞` → `m → 1` (full bet)
- The 2·Φ−1 transform compresses unbounded z into [-1, 1]
- Monotone increasing in p — bigger probability → bigger bet

For our **3-class** case (K = 3, classes are short/flat/long with the active classes being short and long):
- Treat as 2-class on `{short, long}` conditional on "we trade": consider `p_act = p_long / (p_long + p_short)` — but this ignores p_flat
- Better: use the **margin** `p_long − p_short` directly, normalize with Bernoulli proxy std `sqrt(p_long·p_short)`
- LdP's formula effectively: `z = (p_top − 1/K) / sqrt(p_top·(1−p_top))` with K=3 baseline; gives smooth continuous size

---

## Discretization (LdP §10.4) — required for our platform

Continuous `m ∈ [-1, +1]` cannot be submitted: platform requires `pred ∈ {0, 1, 2}`. LdP's discretization rule:

```
m_d = round(m / d) · d
m_d = clip(m_d, −1, +1)
```

with step size `d`. For our case the natural step is **1** (full bet or no bet), and the rule degenerates to:

```
pred = 0  if m < −threshold        (short with conviction)
pred = 1  if |m| < threshold       (no trade)
pred = 2  if m > +threshold        (long with conviction)
```

This **looks like our current T+δ rule** but with a key difference: `m` already accounts for class-prior offset and Bernoulli variance, so the threshold has a probabilistic meaning instead of being a raw `p_0, p_2` cut. Equivalently, our current rule `max(p_0, p_2) > T AND > p_1 + δ` is a piecewise-linear approximation; LdP's rule is the smooth analytic version.

---

## Sigmoid form (LdP §10.5) for forecast-price strategies

If you have a regression that gives a target price `f̂` (or equivalently, an expected ∆mid), LdP recommends:

```
m = (f̂ − f) / sqrt(w + (f̂ − f)²)
```

with `w > 0` calibration parameter controlling how aggressively size grows with divergence. **This is directly applicable to us if we add a quantile-regression head predicting expected ∆mid** (see r13 §3 distributional regression).

---

## Power form (LdP §10.5) — non-linear curvature

```
m = sign(f̂ − f) · |f̂ − f|^w
```

- `w < 1`: concave (momentum: large divergence saturates)
- `w = 1`: linear
- `w > 1`: convex (mean-reversion: only act on extreme divergence)

For our HFT setting where alpha decay is fast and over-trading is fee-deadly, `w > 1` (convex) is reasonable — only act on big margins.

---

## Meta-labeling (LdP §3.6, §10.7)

LdP separates the problem into two models:

1. **M1 (primary)**: predicts direction (long/short/flat) — we have this (LightGBM).
2. **M2 (meta / secondary)**: binary classifier — given that M1 says "trade in direction d", will the trade be profitable?
   - Target: `1{realized PnL of M1's signal > fee}` on training set
   - Features: M1's confidence + same LOB features + maybe additional regime features (vol, OFI, time-of-day)
   - Output `q = P(M1's trade is profitable | features)`
3. **Bet size**: scale by `q` (or use LdP's z-formula on q)

**Why it works**: M1 minimizes log-loss on direction. M2 separately learns *when M1 is right*. M2 has access to features that explain when M1 fails — which in our LOSO setup might be "high vol", "near-open", "low OFI persistence". The two-stage decoupling helps because direction and conviction are driven by *different* signals.

For our pipeline this is **an explicit ML upgrade** to threshold gating: instead of a hand-set T, M2 learns its own threshold from data.

---

## Kelly criterion connection (LdP defers to ch.11+)

If we know `p` (calibrated win prob), `b` (gain on win), `a` (loss on loss):

```
f* = (p · b − (1 − p) · a) / (a · b)         (asymmetric Kelly)
   = p / a − (1 − p) / b
```

For our scenario:
- Predict 2 (long), realized ∆mid > 0: gain = ∆mid − fee  ≈  +1.6e-4 expected
- Predict 2, realized ∆mid ≤ 0: loss = ∆mid + fee   ≈  −2e-4 (rough)
- Kelly fraction: bet only when `p > a / (a + b)`

Concrete: `a/b ≈ 1.25` → Kelly says trade if `p_long > 0.555`. **This is almost exactly our current T = 0.55!** It is not coincidental: our PnL-based threshold tuning has converged (empirically) to the Kelly-optimal hit rate. So the obvious lever is **not** lowering T but **calibrating p first** so that `p = 0.55` actually means 55% hit rate (currently it might mean 50% real hit rate due to over-confidence).

**Fractional Kelly**: half-Kelly `f*/2` is typical to handle parameter uncertainty. If we add this as a multiplicative shrink on bet size, we should expect smaller drawdown and slightly lower expected PnL — for our discrete 0/1/2 it manifests as raising T slightly above the LdP/Kelly-optimal level.

---

## Application to our project — key takeaway

Our current `T = 0.55, δ = 0.10` rule is **a special case of LdP's discrete bet-sizing rule**, equivalent to a particular setting of (z-threshold, step size). The path forward:

1. **Calibrate p** (temperature / beta) so threshold has true probabilistic meaning
2. **Replace raw threshold with z-statistic gating** `z(p) > z_threshold` — same complexity, but accounts for class prior and Bernoulli noise
3. **Add meta-labeling M2** trained to predict "M1 is right" on out-of-fold predictions; use `q` from M2 instead of (or alongside) `p` from M1

---

## Sources

- López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 10
- [Reasonable Deviations — AFML notes](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [MQL5 Article 21824 — AFML Ch10 implementation](https://www.mql5.com/en/articles/21824)
- [Hudson & Thames — Meta-labeling](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Wikipedia — Meta-labeling](https://en.wikipedia.org/wiki/Meta-Labeling)
