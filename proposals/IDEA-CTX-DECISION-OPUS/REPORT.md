# IDEA-CTX-DECISION-OPUS — Decision-Layer Brainstorm

**Author**: opus xhigh worker
**Date**: 2026-05-08
**Mandate**: Find decision-layer ideas (ℝ → {-1, 0, +1}) that no one has tried but that are first-principles important.
**No training, no GPU, no web search.**
**Baseline**: iter_014 = 5-seed LGB regression on Δmid_norm + EV-gate (T83 robust thresh) — LOSO +36.22, platform +19.23.
**Oracle**: +380. Capture rate **3.6%**. Of the 96.4% gap: **49.6% missed (model abstained)**, **27.8% wrong direction**.

---

## 0. The Cross-Cutting Insight (read first)

Looking at the constraint surface holistically reveals one **structural blind spot** that drives several ideas below:

> The platform feeds `Predictor.predict(List[DataFrame])` with a batch of **N≈1024** windows, shuffled across days/syms. Stateless-across-batches is enforced. **But within a batch, you have 1024 samples to play with.**

Almost every decision-layer experiment to date treats each row independently: `action_i = thr(pred_i)`. But the batch is a **sample of size 1024 drawn from the test distribution.** That gives free distributional moments — sample mean, sample std, sample quantiles, **sample rank** — every one of which is computable per-call and forgotten when the call returns. **None violates statelessness, none uses sym IDs, none uses date.** It is the most underexplored axis in this project.

This shows up in **IDEA #1 (BAT)** and **IDEA #2 (Cross-Sample Sigma Calibration)**, which are first-principles new and not present in any of T1–T98.

A second cross-cutting theme: **`time` is allowed** (it stays in the window even if `date=0`), so any rule that varies decision policy by intraday clock is fair game and largely unused. This is **IDEA #7 (Intraday Schedule)**.

A third theme: the **49.6% missed-trade gap is the dominant contributor**, not wrong-direction. Most of our threshold work has been about *raising* the bar (preventing wrong trades). The bigger lever is **lowering the bar in the right contexts** — which is exactly what a **σ-aware gate** does: in low-noise regimes, take more shots. This drives **IDEAS #3, #4, #5**.

---

## 1. TOP 5 RECOMMENDATIONS (ranked by upside × feasibility / time)

| # | Name | Angle | Est LOSO Δ | Hours | No retrain? | Risk |
|---|---|---|---|---|---|---|
| 1 | **BAT — Batch-Adaptive Threshold (rank-based gate)** | F+A | **+1.5 ~ +3.0** | 3-5 | ✅ | medium |
| 2 | **QSA — Quantile-Spread Abstain (t-stat gate over T86 q-models)** | A+E | +1.0 ~ +2.0 | 3-4 | ✅ | low |
| 3 | **IDS — Intraday Schedule Threshold (3-knot piecewise on `time`)** | F | +0.5 ~ +1.5 | 2-3 | ✅ | very low |
| 4 | **SCG — Spread-Conditioned EV Gate (`thr = α + β·spread1`)** | B | +0.5 ~ +1.5 | 2-3 | ✅ | low |
| 5 | **HMD — Hybrid Multi-Model Direction-Veto (sign(LGB)≡sign(NN))** | E | +0.5 ~ +1.2 | 4-6 | needs T87 NN | medium |

All five are **Predictor-only** changes (no GPU retrain needed) and **fully red-line compliant** (sym-agnostic, stateless across batches, no `date`). Combined upside (with appropriate decorrelation) is likely +2.5 ~ +5 LOSO over iter_014, mapping to roughly **+1.5 ~ +3 platform PnL** at the 0.70 transmission ratio.

---

## 2. IDEAS BY ANGLE

### Angle A — Bayesian / Decision Theory

#### IDEA #1 (★ TOP PICK) — BAT: Batch-Adaptive Threshold

**Problem this solves.** EV gate uses *absolute* thresholds (`pred > thr_up`). If the model's output distribution drifts on test (e.g., new sym pushes the regression mean up by 2σ), the gate over-triggers buys and under-triggers sells. We have no online recalibration mechanism. The 49.6% "didn't dare" rate strongly suggests under-triggering somewhere.

**Mechanism.**
For each `predict()` call with batch of N predictions `{p_1, ..., p_N}`:
```
Q_up   = quantile(preds, 1 - α_up)        # α_up ∈ [0.05, 0.20]
Q_dn   = quantile(preds, α_dn)
action_i = +1   if p_i > max(Q_up, thr_abs_up)
         = -1   if p_i < min(Q_dn, -thr_abs_dn)
         =  0   otherwise
```
The `max/min` clamps protect against pathological batches (e.g., all-flat regimes where Q_up is essentially 0 — we want to abstain not chase rank). Calibrate `α_up, α_dn, thr_abs_*` jointly on K-fold OOF (T83 method), with batches simulated by random shuffling of OOF preds in groups of 1024.

**Why it differs from EV gate.** EV gate is location-invariant: a +0.0001 shift in the entire pred distribution is invisible to it. BAT is **rank-based within the batch** — robust to bias drift. It also adapts: in noisy batches the top quantile threshold rises automatically.

**Why it differs from per-sym/per-regime threshold (which is banned).**
Per-sym uses `sym` as a covariate (banned because test sym ∉ train syms). BAT uses **batch-empirical statistics** which are completely sym-blind — you don't even know which sym each row is. This is the legal substitute.

**First principles.** Conformal prediction (T30 attempt) builds a calibration set from training data. BAT builds an "instant" calibration set from the test batch itself. Because N=1024, the empirical quantile has CV ≈ 1/√(α(1-α)N) ≈ 9% for α=0.1 — tighter than per-fold variance.

**Implementation (≈4h).**
- Modify only `Predictor.predict`. Read `len(input_list)`; compute predictions; compute `np.quantile`; combine with absolute fallback.
- Calibrate `α_up, α_dn` on T75 OOF preds simulated as 1024-batches (run 100 random partitions, mean PnL).
- **No retraining**. Pure Predictor.

**Upside estimate.** If BAT corrects even 5% of the 49.6% missed-trades while costing 1% extra wrong trades: +5%·190 - 1%·100 ≈ +8.5 raw → after capture-rate compression ≈ +1.5 ~ +3.0 LOSO, +1 ~ +2 platform.

**Risks.** (i) Test batch composition — if the platform feeds intentionally homogeneous batches (e.g., all sym=0), within-batch ranks may be collinear with sym → leakage indirectly. Mitigation: monitor empirical batch homogeneity in audit. (ii) If a batch is small (last partial), fall back to absolute. (iii) Calibration overfit to OOF — mitigate via 50/50 split of OOF for α-search vs validation.

**Compliance check.** No `sym` use ✅. No `date` use ✅. Each call is independent ✅. Robust to row shuffle (only ranks matter) ✅.

#### IDEA #2 — Cross-Sample Sigma Calibration (CSSC)

**Mechanism.** Within batch, compute `σ_batch = std(preds)`. Define z-score `z_i = p_i / σ_batch`. Threshold on z: act if `|z_i| > z*`. Equivalent to BAT for symmetric Gaussian batches but smoother.

**Why it differs from BAT.** BAT is rank-based (robust to outliers); CSSC is moment-based (smoother but vulnerable to outliers). They can be combined: `z_i = (p_i - median(B)) / MAD(B)`.

**First principles.** This is amortized empirical Bayes: σ_batch is an in-batch estimator of the predictive uncertainty scale. For risk-averse decisions under unknown σ, the t-statistic is the sufficient statistic.

**Upside.** +0.5 ~ +1.5 LOSO. Slightly less than BAT because z* is harder to calibrate than quantile rank.

**Implementation.** Same as BAT but use `(median, MAD)` instead of quantiles. ~2h.

**Risk.** Outlier-prone if any pred is extreme. Use median/MAD or trimmed-mean to fix.

#### IDEA #3 — Heteroskedastic Bayes-Optimal Threshold (HBT)

**Problem.** EV gate assumes homoskedastic noise. Reality: `σ²(x|features)` varies by 5-10× across regimes. In low-σ regimes we should *lower* the threshold (taking small but reliable signals). This directly attacks the 49.6% missed-trade gap.

**Mechanism.** Train a small variance head `σ̂(x)` (or use ensemble disagreement as proxy). Decision rule:
```
t_i = pred_i / σ̂_i
action_i = sign(t_i) iff |t_i| > z*  AND  |pred_i| > fee_floor
```
The `fee_floor` is the absolute minimum to cover 2bp fee at h=60.

**First principles.** Bayes-optimal under Gaussian posterior with linear utility:
- Action +1 has E[U|+1] = μ - fee
- Switching point μ* = fee. **No σ dependence.**
- BUT under risk-aversion (CRRA-like) or empirical Bayes shrinkage with prior σ_0:
  - Posterior mean shrinks: μ̃ = μ · σ_0² / (σ_0² + σ²)
  - Effective threshold becomes σ-dependent.
- More importantly, under **model uncertainty about whether the regression is biased**, the t-stat is the right invariant.

**Why differs from current.** EV gate is `|pred| > thr` — magnitude-only. HBT is `|pred|/σ > z` — t-stat. In high-σ regimes the t-stat threshold becomes a higher absolute threshold (good — those are noisy regimes); in low-σ regimes it becomes lower (good — those are reliable signals we currently miss).

**Implementation.** Either:
- **Cheap path**: use `σ̂_i = std(pred_seed_1..5)` from existing 5-seed ensemble (zero retrain).
- **Quality path**: use T86 quantile spread `σ̂ = (q70 - q30) / (2·0.524)` (the 0.524 maps IQR to σ for Gaussian).

**Upside.** +1.0 ~ +2.0 LOSO. The cheap path alone should give +0.5 because seed-disagreement *is* a useful uncertainty signal.

**Risks.** Seed disagreement is dominated by where leaves split → can be biased low for high-confidence wrong predictions. Quantile-spread version (IDEA #4) is more principled.

---

### Angle E — Information-Theoretic Abstain

#### IDEA #4 (★ TOP-2) — QSA: Quantile-Spread Abstain (T86 reuse)

**Problem.** T86 trained q30 and q70 quantile regressors and used q_mid = (q30+q70)/2 as a *point estimate* in the ensemble. **The spread (q70 - q30) was discarded** — but the spread is exactly the model's self-reported uncertainty. We have a free uncertainty signal sitting unused on disk.

**Mechanism.**
```
mu_i      = (q30_i + q70_i) / 2
spread_i  = q70_i - q30_i                    # IQR
sigma_i   = spread_i / 1.349                 # IQR → σ for Gaussian
t_i       = mu_i / sigma_i
action_i  = sign(mu_i) iff |t_i| > z*  AND  |mu_i| > thr_floor
```
Calibrate `(z*, thr_floor)` jointly on T83 kfold5_date method.

**Why it differs from EV gate and from T86's existing usage.** T86 used q_mid as *just another predictor* in a 3-way ensemble. It never used the **spread as a gate**. This is the cheapest unrealized win in the project — the models exist.

**Math sanity.** For pred ~ N(μ, σ²):
- Bayes decision under absolute loss + fee: act iff |μ| > fee AND P(sign(Δmid) = sign(μ)) > p*
- P(correct sign) = Φ(|μ|/σ) → t-stat threshold
- Equivalent to acting iff `t > Φ⁻¹(p*) = z*`. For p*=0.6, z*=0.25; for p*=0.7, z*=0.52.

**First principles.** This is the textbook "measurement-error-aware decision rule." Without it, the 49.6% missed-trade gap is partly explained by treating low-σ regimes the same as high-σ ones (raising thresholds defensively).

**Implementation (≈3-4h).**
1. Predictor.py: load 10 q-models (`model_T86_q030_seed*.txt`, `model_T86_q070_seed*.txt`) — they exist in `experiments/T86_quantile_dual_gate/`.
2. Compute (mu, sigma) from per-row q30, q70 averaged over seeds.
3. Add t-gate to existing T75 LGB EV gate as an *additional veto*: act iff EV-gate fires AND t-gate fires.
4. Calibrate z* on OOF using same T83 method.

**Upside.** +1.0 ~ +2.0 LOSO. Conservatively **additive** with BAT (#1) because they exploit different invariants.

**Risk.** If quantile spread is itself biased (e.g., always too narrow at boundaries), gate is miscalibrated. Mitigate by sanity-checking calibration: empirical quantile coverage of [q30, q70] should be ~40% on OOF.

#### IDEA #5 — Cross-Model Disagreement Veto (XMDV)

**Mechanism.** Use NN (T87) and LGB (T75) as orthogonal predictors. Action requires sign agreement:
```
mu = w_NN · pred_NN + w_LGB · pred_LGB
action = sign(mu) iff (sign(pred_NN) == sign(pred_LGB)) AND |mu| > thr
```

**Why it differs from straight ensembling.** Straight ensemble averages disagreements → washed-out small magnitudes. XMDV explicitly **vetoes** when the two models disagree on direction. This is the discrete-decision analog of variance reduction.

**Why it works here.** Cross-corr T87↔T75 = 0.71 → ~30% of samples have meaningful disagreement room. By construction, samples where both agree are more likely to be true positives.

**First principles.** If two models with independent error processes both vote in the same direction, the combined posterior probability of correctness rises super-additively. Specifically, if each model is right with p, sign-veto requires p² agreement, but conditional accuracy P(correct | both agree) > p (under positive base rate of correlated-but-not-identical errors).

**Implementation.** Requires T87 NN integrated into Predictor.py (already on the iter_015 critical path). Adding the sign-veto is a 1-line change.

**Upside.** +0.5 ~ +1.2 LOSO **on top of** the standard T87 ensemble lift. Marginal once T87 is in.

**Risk.** Veto may abstain on real signals where one model is just slow. Use as additional gate (intersection), not replacement.

---

### Angle F — Context-Conditional Decision

#### IDEA #6 — SCG: Spread-Conditioned EV Gate (★ TOP-4)

**Problem.** Microstructure 101: at wide spread, mid moves are mostly noise (mid bounces between bid and ask). At tight spread, real information dominates. Yet our threshold is constant. T86 audit shows `spread_w20`-style features are top-importance — model uses them, but gate doesn't.

**Mechanism.**
```
spread_t = spread1 from current row (last row of window)
thr_up_t = α_up + β_up · spread_t
thr_dn_t = α_dn + β_dn · spread_t
action = +1 if pred > thr_up_t else (-1 if pred < -thr_dn_t else 0)
```
Calibrate `(α, β)` per side via 4-D DE on OOF (still 4 params — same as current robust thresh, no extra overfit).

**Why differs.** Constant thresh treats `spread=2bp` and `spread=20bp` identically. SCG raises the bar in 10× noisier regimes.

**First principles.** The expected fee is roughly `fee · 2 / mid` (≈ 2bp on h=60). The expected noise is proportional to spread (about half-spread per tick from bid-ask bounce). Thus the signal-to-noise ratio at threshold should scale: `thr / spread = constant`. β captures exactly this.

**Implementation.** ~2-3h. Predictor.py only. Re-run kfold5_date DE with 2 extra params (β_up, β_dn).

**Upside.** +0.5 ~ +1.5 LOSO. Multiplicatively additive with BAT (BAT corrects mean drift; SCG corrects per-row noise level).

**Risk.** If spread itself drifts on test (regime shift), needs validation. Use median(spread) over window not just last row.

#### IDEA #7 (★ TOP-3) — IDS: Intraday Schedule Threshold

**Problem.** `date` is zeroed but `time` is preserved (HH:MM:SS). Markets have well-known intraday seasonality:
- 9:40–10:00: opening volatility, more noise
- 11:00–11:20: end of morning session, drift
- 13:10–13:30: post-lunch reopen, low signal
- 14:30–14:50: pre-close, often momentum

Current threshold is constant — completely blind to this.

**Mechanism.** Encode intraday minute (0–200 across both sessions, derived from time). Piecewise-constant threshold:
```
def thr(minute):
  if minute < 10:  return α + δ_open       # 9:40-9:50: penalty
  if minute > 95:  return α + δ_morn_end   # 11:15-11:20
  if 100 < minute < 110: return α + δ_pm_open
  if minute > 190: return α + δ_close
  return α
```
Calibrate the 5 deltas on OOF.

**Why differs.** Constant thresholds waste signal at quiet midday (could lower bar) and chase noise at open/close (should raise bar).

**First principles.** Decades of empirical microstructure literature on intraday U-shapes in volume, volatility, and return autocorrelation. Tang & Xu 2018 etc. — these patterns are real on Chinese A-share LOB.

**Implementation.** ~2h. Read `time` from the **last row** of each window (always present in the 100-tick lookback, even though `date=0`). Compute minute. Apply schedule. Calibrate via OOF DE.

**Upside.** +0.5 ~ +1.5 LOSO. Low overfit risk because only 5 piecewise constants over a domain with strong physical priors.

**Risk.** None of the deltas should be huge — bound them in [-0.0002, +0.0002]. Sanity-check that performance per intraday bin moves the right direction.

**Compliance.** `time` is allowed by spec (`CRITICAL_CONSTRAINTS.md §2`) — the constraint says `date` is zeroed, not `time`. Confirmed.

#### IDEA #8 — Volatility-Regime Switching Policy (VRSP)

**Mechanism.** Compute realized vol over the lookback window:
```
rv = sqrt(sum(diff(midprice[-W:])**2))
regime = 'low' if rv < ρ* else 'high'
thr = thr_low if regime == 'low' else thr_high
```

**Why differs from spread-conditioned (#6).** Spread is instantaneous; rv is integrated over W ticks. They capture different things — spread is *cost* level, rv is *uncertainty* level. Both should be used.

**First principles.** Heston-style: in high-vol regimes the unconditional return distribution is wider → fixed threshold catches noise. In low-vol, fixed threshold is too conservative → miss signals.

**Implementation.** ~2h. Compute rv from window's midprice column; lookup table for thresholds.

**Upside.** +0.5 ~ +1.0 LOSO. May be redundant with SCG (#6) — test both, keep the better.

**Risk.** Overfit ρ* (vol cutoff). Mitigate: use median rv from training as the cutoff, no tuning.

---

### Angle B — Microstructure / Market Making

#### IDEA #9 — OFI-Conviction Multiplier (OCM)

**Problem.** The order flow imbalance (`bsize1 - asize1`) at the moment of decision is a strong contemporaneous signal. The model learned it (it's a top-importance feature), but the *decision* doesn't explicitly multiply by it.

**Mechanism.**
```
OFI = (bsize1 - asize1) / (bsize1 + asize1)        # ∈ [-1, 1]
boost = 1 + λ · OFI · sign(pred)                    # > 1 if OFI agrees with pred
effective_pred = pred · boost
action = standard EV gate on effective_pred
```
λ ∈ [0.1, 0.5] calibrated on OOF.

**Why differs.** OFI in the feature vector is just one of 354 inputs to a tree ensemble, where its marginal effect can be muted. As an explicit multiplier *on the decision*, it gates more sharply.

**First principles.** OFI is the most direct microstructure signal of imminent direction (Cont, Kukanov, Stoikov 2014). It is not a substitute for the model — it is a confirmation.

**Implementation.** ~2h. Predictor.py only.

**Upside.** +0.3 ~ +1.0 LOSO. Likely subsumes some signal already in the model — be honest about marginal contribution.

**Risk.** Could double-count OFI; calibrate λ low first.

#### IDEA #10 — Bid-Ask Bounce Filter (BABF)

**Mechanism.** If the most recent K mid-changes are alternating bounces (sign flips with magnitude ~ half-spread), abstain — we're in a noise regime. Specifically:
```
recent = midprice[-5:].diff().sign()
if abs(sum(recent)) < 2:   # near-zero net direction
  action = 0
```

**Why differs.** EV gate doesn't see "bouncy" patterns; it just sees current pred.

**First principles.** Bid-ask bounce is a structural autocorrelation noise pattern (Roll 1984 spread estimator is built on exactly this). When it dominates, real signal is buried.

**Implementation.** ~1.5h. Predictor.py only. Sanity-check that this doesn't kill too many trades.

**Upside.** +0.2 ~ +0.6 LOSO. Targeted, low magnitude — good for stacking with bigger ideas.

---

### Angle C — Optimal Stopping / Lookback Confirmation

#### IDEA #11 — Lookback-Confirmed Action (LCA)

**Problem.** Current EV gate fires on `pred_t` alone. But `pred_t` is volatile across adjacent ticks. We can't *use* adjacent ticks (statelessness), but **within the lookback window we have 99 historical ticks**. Using only the first 70 of them, we can compute an "implied prediction" for time `t-30` using the same model topology.

**Mechanism.**
```
pred_now = model(window[-100:])
pred_lag = model(window[-100:-30] + zero_padding)   # or shorter window
action = standard gate iff sign(pred_now) == sign(pred_lag)
```

**Why differs.** Two near-overlapping predictions, each independently passing the sign test, has lower false-positive rate than one prediction.

**First principles.** This is intra-window persistence — a stateless surrogate for "the signal has been there for a while." Crucially **no cross-batch state is needed**: the second prediction is computed from a subset of the same window we already have.

**Implementation.** ~3h. Compute model on shorter (70-tick) window OR truncate features to ignore last 30 ticks. Doubles inference cost — verify within 3h budget.

**Upside.** +0.3 ~ +0.8 LOSO.

**Risk.** Compute cost — OK at ~30s extra per fold but should benchmark.

---

### Angle D — Multi-Objective

#### IDEA #12 — CVaR-PnL DE Objective

**Mechanism.** Replace `sum(per_fold_pnl)` objective in DE thresh search with `CVaR_α(per_fold_pnl)` (mean of worst-α folds). This stops fold 3 (active rate 0.28%, drags overall) from dominating noise.

**Why differs.** Sum-PnL rewards optimization that improves any fold. CVaR-PnL rewards optimization that improves *the worst* folds.

**First principles.** Distributional robustness: the true test sym is unknown. We want a threshold that performs robustly across regimes, not one that exploits a single fold's quirk.

**Implementation.** ~2h. Modify `de_thresh.py` objective. Re-tune iter_014 thresholds with new objective.

**Upside.** +0.3 ~ +1.0 LOSO. Already mentioned in PRACTICE-FUNDAMENTAL audit #5 but not deployed.

**Risk.** May reduce sum slightly if best-fold gives way to median-fold consistency — that is a feature, not a bug, for OOD generalization.

---

### Angle G — Policy Learning

#### IDEA #13 — Counterfactual Policy from Logged Predictions (CPL)

**Mechanism.** Train a tiny MLP `π(a|x_decision)` where `x_decision = [pred, sigma_proxy, spread, vol, intraday_minute, OFI]` (≈10 features). Train via importance-weighted PnL:
```
loss = - E_{(x,a*,r) ~ logged} [ (π(a*|x) / μ(a*|x)) · r ]
```
where logged actions come from the current EV gate (uniform-random with small ε for exploration), `r` = realized PnL.

**Why differs from end-to-end PnL training.** SPO+ (T87) inserts decision loss into the regression. CPL takes regression as fixed and learns a *separate* meta-policy on top. This isolates the model error from the policy error, debugging-friendly.

**First principles.** Off-policy evaluation literature (Dudík et al.). The challenge here: our logged policy is deterministic (EV gate), so importance weights blow up. Mitigate via SNIPS estimator + ε-greedy retro-augmentation: assume each logged action had baseline probability `0.6` and others `0.2` — gives bounded weights.

**Implementation.** ~6-8h. Training-time, but tiny model (≈1K params). Substantial overfit risk.

**Upside.** +1 ~ +3 LOSO if it works. Big variance.

**Risk.** Off-policy estimators are notoriously high-variance with deterministic logging. Not a TOP-5 because of risk; valuable to test if everything else plateaus.

---

### Angle H — Fee Structure

#### IDEA #14 — Net-of-Fee Regression Target

**Mechanism.** Replace training target `Δmid_norm` with `Δmid_norm - 2·fee·sign(Δmid_norm)` (i.e., explicitly subtract round-trip fee in the target's direction). The optimal *threshold* on this target becomes ≈ 0, removing one degree of freedom.

**Why differs.** Decouples model from threshold — model directly predicts net PnL. Threshold becomes trivial: `act iff |pred| > 0` (with small slack).

**First principles.** Bayes risk minimization with built-in cost. If you train regression on net-of-fee target with MSE, the conditional mean estimator is exactly the per-decision EV — optimal Bayes action is `sign(pred) iff pred > 0`.

**Implementation.** ~3-4h. Training-time change. Re-run T75-style LGB with new target.

**Upside.** +0.3 ~ +1.0 LOSO. Modest but principled. Reduces threshold-search variance — a robustness win, not raw lift.

**Risk.** Target distribution shape changes (bimodal due to sign-dependent shift) → may complicate gradient boosting. Mitigate via Huber loss.

---

## 3. EXPLICITLY REJECTED IDEAS (with reason)

| Idea | Reason |
|---|---|
| **Per-sym threshold** | Violates §1.3 (test sym ∉ train syms). Only intra-batch surrogates allowed. |
| **TTT / online fine-tune** | User explicitly banned; cross-day/sym mixing in batch breaks self-supervised loss. |
| **Cumulative position tracking** | Violates §1.2 (stateless across batches). |
| **Date-conditional threshold** | `date` is zeroed (§1.1). |
| **Last-trade memory** | Stateless across batches. Even within-batch "last" is meaningless because rows are shuffled. |
| **Soft / fractional sizing** | Output spec is `{0,1,2}` per sample. No fractional positions. |
| **Use sym embedding in policy net** | sym is not allowed as input feature (§1.3). |
| **Adversarial regime detector with retraining** | Retraining at test time = TTT-equivalent. Banned. |
| **Trade frequency throttling across batches** | Stateless, cannot count "recent trades." Within-batch frequency control is OK but largely subsumed by BAT. |
| **Online ensemble weight learning** | Same statelessness issue; covered partially by IDEA #13's offline policy. |
| **Cross-horizon term-structure decision (h5/10/20/40/60 ladder)** | Worth exploring but requires retraining 4 more horizons; deferred until h60 plateau. T77 already showed 1 attempt is unbiased = +28.42 (worse than baseline). Not an obvious win. |

---

## 4. KEY ARGUMENTS & WHY THESE > PAST IDEAS

1. **BAT (#1) and CSSC (#2)** are categorically different from any prior work because they use the **batch as a measurement**. Prior threshold work treated each row independently. The 1024-sample batch is sitting unused. *This is the single biggest unrealized degree of freedom in the project.*
2. **QSA (#4)** uses an asset already on disk (T86 quantile models) but in a **previously unconsidered way** — the spread as a gate, not as another point predictor. The marginal cost is ≈3 hours; the marginal upside +1.0 LOSO. Best ROI in the project.
3. **IDS (#7)** exploits a constraint loophole: `date` is banned but `time` is not. None of the existing thresholding work uses intraday seasonality. There is a strong physical prior (open/close noise) so overfitting risk is bounded.
4. **SCG (#6)** is the canonical "noise-aware threshold." It's surprising no T# experiment has done this explicitly — not even T30/T38 which are 4D DE searches. The 4 parameters could have been (T_up, T_dn, β_up, β_dn) instead of (T_up, T_dn, d_up, d_dn).
5. **HBT (#3)** turns the existing 5-seed disagreement into a real abstain signal. Combined with QSA (which uses quantile-spread instead of seed-spread) we cover both kinds of uncertainty.

The synergy is that **#1, #4, #6, #7 are mutually orthogonal** (intra-batch vs intra-row vs context vs time), so their lifts should approximately add. A "kitchen sink" Predictor combining all four is implementable in 1–1.5 days and could yield +3 to +6 LOSO over iter_014 — i.e., **iter_015 = +39 ~ +42 LOSO**, mapping to **platform PnL +21 ~ +23**.

---

## 5. SUGGESTED ORDER OF EXECUTION (if PM picks 3)

1. **Day 1**: QSA (#4) + IDS (#7). Both Predictor-only, models exist, ~6h total. Validate via leave-2-fold-out on OOF (as in AUDIT-METRIC §2) to confirm OOS lift before submission.
2. **Day 2**: BAT (#1). Calibrate on simulated 1024-batches over OOF. Stacked on top of #4+#7. Lift sanity-checked on multiple batch-shuffle seeds.
3. **Day 3 (if budget remains)**: SCG (#6). Cheap, marginal but additive.
4. **Submit** as iter_016 candidate. Compare LOSO and platform delta vs iter_015 (T87+T86).

The iter_014 → iter_015 gap was +1.81 LOSO via T87 SPO+. The above 4-stack should reasonably aim for **+3 ~ +5 LOSO** atop iter_015, materially closing the capture-rate gap from 3.6% toward 5–6%.
