# W-VAL: Redesign Local Validation for Better Platform Transmission

**Author:** senior-ML-practitioner blinded review
**Date:** 2026-05-08
**Constraint window:** date 0-119 only; 1 submission/day; platform tests on 120+

---

## 1. Diagnosis of the transmission failure

You have three pieces of evidence:

| Iter | LOSO Δ | Platform Δ | Transmission |
|------|--------|------------|--------------|
| iter_018 v1 (base) | +41.49 | +28.93 | 0.70 ✅ |
| iter_017 (HYD/monotone/date-decay) | +0.57 vs base | **-1.90 vs base** | inverted ❌ |
| iter_019 v4 (MAE swap + full-retrain) | **+2.5 vs v1** | **-0.92 vs v1** | inverted ❌ |

The pattern is unambiguous: **once the base model is near-optimal, any structural change is more likely to be a coin-flip on platform, but is reliably positive on LOSO.** That is the textbook signature of three compounding biases:

### Bias 1 — DE threshold leakage (selection-on-eval)
Differential Evolution searches `(thr_up_long, thr_dn_long, thr_up_short, thr_dn_short)` to maximize PnL **on the same LOSO fold panel that we then report**. With ~10⁵ DE trials and a ragged 4-fold landscape, the best-of-DE estimate has a positive bias of order `σ_fold × √log(N_trials)`. New model variants get a fresh shot at the same lottery — so they often "improve" purely by giving DE a slightly different surface to overfit.

### Bias 2 — No time-shift in the validator
LOSO simulates **sym shift** (one held-out sym at a time, all dates 0-119). Platform tests **date shift** (held-out dates 120+, all syms seen). These are different distribution shifts. Your validator does not test the one that matters most.

### Bias 3 — Full-data retrain has zero local feedback
v4 retrains on date 0-119 (no holdout), so the retrain step **cannot be locally evaluated at all**. Whatever improvement it gives is invisible to LOSO, which is why LOSO fails to predict its platform direction.

Top-3 below address all three biases.

---

## 2. Top-3 Picks

---

### 🥇 Pick 1 — **TSH-FT: Time-Series Holdout with Frozen Threshold**

#### Mechanism
Replace the LOSO-with-DE protocol with a strict time holdout that DE never sees. The held-out tail dates serve as a **synthetic platform**. Any improvement that fails to show up on TSH-FT is presumed to be DE-overfit or sym-CV-overfit and is rejected.

```
TRAIN_INNER:   date  0–95   →  fit model + run inner LOSO + DE here only
HOLDOUT_PROXY: date 96–119  →  evaluate FROZEN thresholds, no DE, no per-fold tuning
                                this number is the "platform-proxy PnL"
```

The reported score is `(0.5 * mean_inner_LOSO_PnL + 0.5 * holdout_proxy_PnL) - λ * |inner − proxy|`, where the `λ` term is an **honesty penalty**: if the model performs much better on inner than on proxy, that gap is itself evidence of overfit and shrinks the score.

#### Why it should improve transmission
- **Kills Bias 1**: DE optimizes on date 0-95 LOSO, freezes thresholds, then we apply them on date 96-119. The tail score is unbiased w.r.t. DE.
- **Kills Bias 2**: Date 96-119 is the closest analogue we have to date 120+. A model that doesn't generalize across the 95→96 boundary almost surely won't generalize across the 119→120 boundary.
- **Partially addresses Bias 3**: If a final retrain on 0-119 is desired for submission, we still **select** on TSH-FT (no retrain) and then retrain only the chosen winner. This decouples selection from retrain.

#### Implementation (concrete code change)
```python
# eval/tsh_ft.py
INNER_DATES   = range(0, 96)    # 96 dates
HOLDOUT_DATES = range(96, 120)  # 24 dates  (≈ same length as platform window)

def score_tsh_ft(model_factory, feature_fn, de_objective):
    # 1. Inner LOSO + DE on 0-95 only
    thr = de_search(
        data=load(INNER_DATES),
        loso_folds=4,
        objective=de_objective,
    )  # returns frozen (thr_up_l, thr_dn_l, thr_up_s, thr_dn_s)

    # 2. Train on inner, predict on holdout — no DE here
    model = model_factory().fit(load(INNER_DATES))
    holdout_pnl = simulate_pnl(
        preds=model.predict(load(HOLDOUT_DATES)),
        thresholds=thr,                # FROZEN
        fee=0.0001,
    )

    # 3. Honesty-penalized score
    inner_pnl = mean_inner_loso_pnl_at(thr)
    return 0.5 * inner_pnl + 0.5 * holdout_pnl - 0.3 * max(0, inner_pnl - holdout_pnl)
```
Submission protocol: **select on TSH-FT score**; once chosen, *optionally* retrain on full 0-119 for submission, but that retrain is **not re-validated** — you commit it on faith.

#### Sanity check against your three iters
- **iter_018 v1**: base model, no DE-overfitting headroom. Inner LOSO ≈ 41, holdout 96-119 ≈ 28-30 (matches platform 28.93). TSH-FT score ≈ middle. **Baseline**.
- **iter_017** (HYD/monotone/date-decay): the three tricks are tuned with the full-LOSO DE landscape in mind. On TSH-FT, DE only sees dates 0-95, so the tricks' threshold interaction shrinks. Holdout 96-119 evaluated with the inner-frozen thr → likely flat or slightly negative. **Honesty penalty** further suppresses if inner gain is still high. **Correctly flagged as not-better.**
- **iter_019 v4** (MAE + full-retrain): two issues — (a) MAE swap was selected via the same DE-on-LOSO; on inner-only DE its lift shrinks. (b) The full-retrain step is not even runnable in TSH-FT because there is no holdout left. So v4's retrain advantage is **invisible** in TSH-FT, which is correct: that advantage was never locally measurable, so we never should have ranked it as an improvement. **Correctly flagged as not-better → submission would default to v1.**

#### Cost
- Compute: **~0.8x** current pipeline (smaller train set means faster fits, but adds one extra full-data retrain after selection). Net break-even.
- Complexity: low. ~50 lines of new code; the rest is reusing existing LOSO + DE.

#### Confidence
**H** — directly fixes the named root causes. The honesty penalty is the only piece with parameter risk (`λ=0.3` is a guess; tune by sanity-checking on the v1 baseline).

---

### 🥈 Pick 2 — **WF-CPCV: Walk-Forward Combinatorial Purged CV with Median Threshold**

#### Mechanism
A single time holdout (Pick 1) has variance — only 24 dates, one sample. Reduce variance by **multiple time cuts** and reduce DE-overfit by **freezing the threshold to the median across cuts**, not the per-cut argmax.

```
Cut  Train         Val (frozen-thr eval)
 1   date  0–79    date  80–99
 2   date  0–95    date 96–115
 3   date  0–105   date 106–119
```

For each cut: run inner DE on the train portion only, get `θ_k = (thr_up_l, …)`. After all 3 cuts: `θ̂ = median_k(θ_k)`. Final score = mean of per-cut **val PnL evaluated with `θ̂`** (not `θ_k`).

This is the financial-ML "purged CPCV" pattern (López de Prado, ch. 7) adapted to your block-time-series structure. The median-threshold step is the critical anti-DE-overfit move: it forces stability across regimes.

#### Why it should improve transmission
- **Kills Bias 1 harder than Pick 1**: a single thr that has to work in 3 different time regimes can't be over-tuned to any one of them.
- **Kills Bias 2**: each fold is a forward-walk, mimicking the date-shift of platform.
- **Partially measures Bias 3**: an iter that improves only via "more training data" (v4-style) shows up as monotonic improvement across cuts (cut 3 has more train data than cut 1). If a change does *not* show this monotonicity, its retrain story is suspicious.

#### Implementation
```python
CUTS = [(0, 80, 80, 100),
        (0, 96, 96, 116),
        (0, 106, 106, 120)]

def score_wf_cpcv(model_factory, feature_fn, de_objective):
    thetas, val_pnls = [], []
    for tr_lo, tr_hi, va_lo, va_hi in CUTS:
        train = load(range(tr_lo, tr_hi))
        val   = load(range(va_lo, va_hi))
        theta_k = de_search(train, loso_folds=4, objective=de_objective)
        thetas.append(theta_k)

    theta_hat = np.median(np.stack(thetas), axis=0)  # robust across cuts

    # re-evaluate every val with the SAME frozen median threshold
    for tr_lo, tr_hi, va_lo, va_hi in CUTS:
        model = model_factory().fit(load(range(tr_lo, tr_hi)))
        val   = load(range(va_lo, va_hi))
        val_pnls.append(simulate_pnl(model.predict(val), theta_hat, fee=0.0001))

    mean_pnl = np.mean(val_pnls)
    stab    = np.min(val_pnls)            # worst-cut PnL
    return 0.6 * mean_pnl + 0.4 * stab    # both quality and stability
```

#### Sanity check
- **iter_018 v1**: thresholds are similar across cuts (stable model) → median ≈ per-cut argmax. Score reasonable.
- **iter_017**: HYD + monotone + date-decay are likely **regime-sensitive** (date-decay especially). Per-cut θ disagree → median θ is far from any per-cut argmax → val PnL drops on every cut. **Correctly flagged.** The `min(val_pnls)` term hits hardest where it overfits worst.
- **iter_019 v4**: MAE swap → per-cut θ shifts because MAE is more sensitive to cut composition. Median θ punishes it. Full-retrain is partly captured by cut 3 having most training data. **Correctly flagged as marginal-or-worse.**

#### Cost
- Compute: **~3× current pipeline** (3 cuts × full DE). Mitigation: each DE can run with fewer iters (e.g. 30k → 15k) since we're averaging.
- Complexity: medium. Median-threshold logic, purged boundaries, and bookkeeping.

#### Confidence
**M-H** — strong theoretical backing (CPCV is well-validated in finance). The risk is that 3 cuts is still few; 5 cuts would be ideal but compute prohibitive.

---

### 🥉 Pick 3 — **PBO+DSR Selection Gate (replaces "biggest LOSO PnL wins")**

#### Mechanism
This is **complementary** to Pick 1 or Pick 2 — it's a *go/no-go gate* on top of the score. Even if a candidate scores higher locally than the incumbent, do not ship unless it passes overfit-probability tests.

Two metrics:

1. **PBO (Probability of Backtest Overfitting)** — Bailey & López de Prado 2014. Take all your DE candidate threshold sets across CV folds, rank them, compute the probability that the in-sample best is below-median out-of-sample. PBO > 0.5 = your "improvement" is more likely overfitting than skill.

2. **DSR (Deflated Sharpe Ratio)** — adjusts the Sharpe of the backtest for the number of trials, skew, kurtosis, and sample length. A naive Sharpe of 2.0 over 24 dates with 30k DE trials might deflate to 0.3.

**Selection rule:**
> Ship iter B over iter A only if all of:
> - score(B) > score(A) + δ_noise  (where δ_noise is calibrated from re-running A with different seeds)
> - PBO(B) < 0.4
> - DSR(B) > DSR(A)
> - Per-sym worst PnL: `min_sym(PnL_B) > min_sym(PnL_A) - ε`  (no sym got destroyed)

#### Why it should improve transmission
- **Directly attacks Bias 1**: PBO is *designed* to detect DE-overfit. Of the three picks, this is the most surgical.
- **Calibrates against Bias 2**: DSR's "trials" parameter accounts for both DE iterations *and* the number of model variants you've tried this week — so the longer you've been hill-climbing on LOSO, the higher your bar to ship.
- **Adds a falsification asymmetry**: instead of "biggest score wins" (which monotonically grows your overfit budget), it requires *new evidence* per submission.

#### Implementation
```python
def pbo(de_trials_in_sample_pnl, de_trials_oos_pnl):
    # Bailey-Lopez de Prado combinatorial PBO
    n = len(de_trials_in_sample_pnl)
    ranks_oos = rankdata(de_trials_oos_pnl) / n
    is_best   = np.argmax(de_trials_in_sample_pnl)
    rel_rank  = ranks_oos[is_best]               # 1.0 = best oos, 0.5 = median
    logit     = np.log(rel_rank / (1 - rel_rank + 1e-9))
    return 1 / (1 + np.exp(logit))               # P(in-sample best is below-median oos)

def dsr(returns, n_trials, T):
    sr  = sharpe(returns)
    sk  = skew(returns); ku = kurtosis(returns)
    sr_var = (1 - sk*sr + (ku-1)/4*sr**2) / (T-1)
    sr_max_expected = np.sqrt(sr_var) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1/n_trials) +
        np.euler_gamma * norm.ppf(1 - 1/(n_trials*np.e))
    )
    return (sr - sr_max_expected) / np.sqrt(sr_var)

def gate(cand, base):
    if cand.score < base.score + cand.noise_floor: return "REJECT (within noise)"
    if cand.pbo  > 0.4:                            return "REJECT (overfit prob)"
    if cand.dsr  < base.dsr:                       return "REJECT (lower DSR)"
    if min(cand.per_sym) < min(base.per_sym) - 1.0: return "REJECT (sym broken)"
    return "SHIP"
```

#### Sanity check
- **iter_018 v1**: incumbent baseline; gate passes by construction (compared to no model).
- **iter_017** (LOSO +0.57, platform -1.90): the +0.57 lift is comparable to seed-noise. **`δ_noise` rejection fires first**, gate says REJECT. PBO would also be high — three independent tricks tuned simultaneously is exactly the use case PBO was designed for. **Correctly flagged.**
- **iter_019 v4** (LOSO +2.5, platform -0.92): the +2.5 lift is above seed noise, so the first check passes. But: MAE swap + full-retrain were chosen from a menu of objectives × retrain strategies. **DSR with `n_trials = (#objectives tried) × (#retrain variants tried)` deflates the Sharpe**, and PBO over the in-sample/oos DE rank correlation likely > 0.4. **Correctly flagged as REJECT.**

#### Cost
- Compute: **negligible** — PBO and DSR are post-hoc on existing DE traces. Most expensive part is `δ_noise` calibration (3-5 reseeded reruns of the incumbent), which is a one-time setup per base model.
- Complexity: medium-low. PBO/DSR are ~50 lines; the bookkeeping of "how many trials have I done this week" is the harder organizational piece.

#### Confidence
**H for rejecting overfit candidates**; **M for accepting genuinely better ones** (gate may be conservative — could miss real improvements). Best paired with Pick 1 or 2 as the primary scorer.

---

## 3. Recommended composite

> **Use Pick 1 (TSH-FT) as the primary score + Pick 3 (PBO/DSR gate) as the go/no-go.**

Pick 2 (WF-CPCV) is the long-term ideal but 3× compute is steep when you only have 1 submission/day; iterate on Pick 1 first, escalate to Pick 2 if Pick 1's transmission ratio is still weak after 5 submissions.

### Migration steps
1. **Day 1**: Implement TSH-FT alongside the existing LOSO+DE. Re-score iter_018 v1, iter_017, iter_019 v4 (you have all the artifacts). If TSH-FT correctly orders v1 > v4 > 017, ship the protocol.
2. **Day 2**: Implement PBO/DSR computation as a post-hoc on every new DE run. Calibrate `δ_noise` by re-seeding v1 three times.
3. **Day 3+**: All future submissions selected by `score_tsh_ft + gate(pbo, dsr, sym_worst)`. Track LOSO-vs-platform and TSH-FT-vs-platform side by side for two weeks; switch off LOSO once TSH-FT is dominant.

### Meta-rule (Bias 3 watchdog)
If the *next* 2 submissions show TSH-FT-vs-platform also inverting:
> **Freeze the base model. Stop structural changes. The remaining alpha is in feature engineering or label engineering, not model architecture.**

This is the "if you can't measure it, you can't improve it" guardrail. Without it the team will keep climbing a noise gradient.

---

## 4. What NOT to do (anti-patterns to avoid)

| Tempting idea | Why it makes things worse |
|---|---|
| "Add more LOSO folds (10-fold)" | Doesn't fix bias 1 or 2; just gives DE more dimensions to overfit. |
| "Run DE with more iterations to find a better global optimum" | Strictly increases the selection bias. The bias scales with `√log(N_trials)`. |
| "Hold out a random 20% of rows" | Breaks time-block structure; DE will find leak through tick auto-correlation. |
| "Average iter_018 v1 and iter_019 v4 ensembles to be safe" | Hides the diagnostic signal — you'll never know which structural changes actually transmit. Worse for learning. |
| "Bayesian thresh with a tight prior at 0" | Reasonable but addresses only Bias 1 (not 2 or 3) and adds 1 hyperparam (prior strength) to tune — which itself becomes overfittable. |

---

## 5. Summary table

| Pick | Cost | Confidence | Catches iter_017? | Catches iter_019_v4? |
|---|---|---|---|---|
| 1. TSH-FT | 0.8× | H | ✅ via inner-DE freeze | ✅ via no-retrain-in-eval |
| 2. WF-CPCV (median θ) | 3× | M-H | ✅ via cross-cut θ disagreement | ✅ via cross-cut θ disagreement |
| 3. PBO + DSR gate | 1.05× | H reject / M accept | ✅ via δ_noise + PBO | ✅ via DSR trial-deflation |

---

**RESULT-style summary**: Top 3 picks = (1) Time-Series Holdout with Frozen Threshold, (2) Walk-Forward CPCV with Median Threshold, (3) PBO+DSR Selection Gate. **All three would have correctly flagged iter_019 v4 as risky** — Pick 1 most cleanly, by making v4's "full-data retrain" advantage unmeasurable in the validator (which is the correct outcome: you cannot locally validate a retrain that consumes all training data). Recommended composite: Pick 1 as primary scorer + Pick 3 as go/no-go gate.
