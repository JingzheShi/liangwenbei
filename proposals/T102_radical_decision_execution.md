# T102 — Radical Decision-Rule + Execution + Scoring Rethink

**Author**: opus xhigh worker (T102)
**Date**: 2026-05-08
**Mandate**: Audit the prediction → action transformation and the scoring objective from first principles. Find **basic but disruptive** improvements. No model/feature/data retraining; **no GPU**, no web; ~90–120 min.
**Anchor**: iter_015 (T87 SPO+ NN + T75 LGB) = LOSO **+40.13**, platform **~+21**. Capture rate of oracle ≈ **3.8%**.

> Disclosure of overlap. There is a prior decision-layer brainstorm (`proposals/IDEA-CTX-DECISION-OPUS/REPORT.md`) which already proposed BAT (batch-adaptive thresh), CSSC (sigma calibration), HBT (heteroskedastic Bayes thresh), QSA (quantile spread abstain), XMDV (cross-model sign-veto), SCG (spread-conditioned gate), IDS (intraday schedule), VRSP (vol-regime switching), OCM (OFI conviction multiplier), BABF (bid-ask bounce filter), LCA (lookback-confirmed action), CVaR-PnL DE objective, CPL (counterfactual policy learning), Net-of-fee target. **T102 stays orthogonal to all of those — every idea below is genuinely new.**

---

## Part 1 — Diagnostic of the Current Decision Framework

### 1.1 The objective is **not** "maximize LOSO PnL"

The contest scoring rule is:

> "5 个任务取最好的那个参与排名" — **the rank uses MAX cum_pnl across horizons h ∈ {5, 10, 20, 40, 60}**.

But every tuning step in the project optimizes **`sum_per_sym(h60)`** as if it were the score. That is technically a *lower bound* on the true scoring (max ≥ any single h), but it discards a **structural lever**: the score is a **max over five independent submissions inside one Predictor**. Two consequences nobody seems to have used:

1. **The 4 short horizons are currently dead weight.** iter_015's `Predictor.predict` returns `[1,1,1,1,1]` per row except `action_h60`. Because `MAX` is what scores, a non-flat short-horizon head can only **help** the score: in the worst case its sum is negative and h60 still wins; in the best case a short horizon happens to outperform on the public test slice.
   - T98 measured per-horizon DE LOSO ≈ +18 (h5), +25 (h10), and h60 = +40, but those are full-fit DE numbers on h-specific predictors. The cost of *adding* h5/h10 active heads is zero (h60 is what wins MAX).
   - **There is no reason `Predictor.predict` should ever output flat on h5/h10/h20/h40.** Even sub-optimal trade rules on those heads strictly weakly improve the MAX score.
2. **Per-row, the most lucrative horizon varies.** The contest's MAX-over-horizons rule means a "horizon router" that fires h_h(i) on the horizon with the strongest signal per row — and flat on others — would in expectation **lift `max_h(sum_h)` toward `sum_i max_h(...)`**, the per-row max. Empirically this gap is large because correlations across horizons are <1.

### 1.2 The DE-4D thresh "overfit" question is settled — but the wrong way

T83 (kfold5_date) plus the current full-fit-DE pipeline establishes:

* The 4D EV thresh search (thr_up, thr_dn, [d_up, d_dn]) **does not** overfit the test 442k materially. K-fold (35.94 unbiased) ≈ full-fit (36.15) within 0.30 spread, and S3 row-shuffle gap ≈ 0. This is the right diagnosis.
* But T83 also showed that the **−17 platform gap is intrinsic data-quality heterogeneity**, not threshold variance. **Tuning thresholds smarter cannot close it.**

What this means for T102: **further refinement of the threshold *value* is dead lever**. The remaining levers are *structural*:

* Replace the **4-parameter scalar gate** with a richer decision function (more dof, but justified by NOT-tuned-on-test priors).
* Replace **"max sum_pnl"** as the calibration objective with something **less data-window-dependent**.
* Use the **batch / row covariates** that today's gate ignores entirely (uncertainty estimates, multi-horizon coherence, OOD signals).
* Use the **MAX scoring rule** that today's pipeline ignores entirely.

### 1.3 Where the lift gap really lives

Oracle PnL ≈ +1054 (sum |Δmid_norm − 2·fee| over all gated rows in OOF, sym-by-sym). iter_015 ≈ +40. **Capture ≈ 3.8%**. Of the 96.2% gap:

* **~50% is "missed trades"** — model predicts |pred| < thr but truth is large; we abstain on real signal. (Already documented in IDEA-CTX-DECISION.)
* **~28% is "wrong direction"** — model fires the wrong way. The 4-param thresh has no veto on direction errors; it just trims magnitude.
* **~18% is "fee burn"** — fired on small signals where |Δmid| barely exceeds fee.

The 4-param gate has structurally *no instrument* to attack (b) — direction errors. Every Part-2 idea that involves multi-horizon coherence, σ-routing, sign-of-median, or OOD vetoes is implicitly attacking the (b) bucket, which the prior brainstorm under-emphasized.

### 1.4 The 5-horizon decision is not 5 independent gates — it's a coupled object

Each of `pred_h5, ..., pred_h60` is a regression of `(mp_{t+H} - mp_t)/(mp_t+1)` on the *same* 100-tick window. They are highly correlated (corr h5/h10/h20 > 0.85 typically) but **not identical**. The vector `(p_5, p_10, p_20, p_40, p_60)` traces an *expected path*. Today we throw four of those numbers away and gate on `p_60` only.

A non-monotone path — e.g. `p_5 = -0.0008, p_10 = -0.0003, p_20 = +0.0002, p_60 = +0.0006` — implies "first dip, then up." For a **ternary action at h60**, this is fine: we hold and exit at +0.0006. But the *risk-adjusted PnL* of that trade is much worse than a monotone `(0.0002, 0.0004, 0.0005, 0.0006, 0.0008)` profile, because in the non-monotone case, the path delta `(p_60 - p_5)` is *itself* the entire signal — we are *betting on the bounce*. **Coherence of the term structure should gate the action.** This is unused.

---

## Part 2 — Radical / Disruptive Trick Proposals

Each idea: **what** / **why basic** / **why disruptive** / **mechanics** / **implementation cost** / **estimated LOSO Δ** / **risk** / **compliance**.
None has been tested in T1–T99 or the existing brainstorm. Constraints (`CRITICAL_CONSTRAINTS.md` §1) are respected throughout.

### TRICK #1 — **Per-Sample Horizon Router (PSHR)** ★ Top Pick

**What.** For each input row, pick the *single horizon h\** with the highest decision-EV; emit non-flat action only on h\* head, flat on the other four. Concretely:

```
for row i:
    EV_h(i) = |pred_h(i)| − λ_h · sigma_h(i) − fee_h          # for h ∈ {5,10,20,40,60}
    h*(i)   = argmax_h EV_h(i)
    action_h*(i)  =  +1 / -1 / 0   (standard EV gate at h*)
    action_h≠h*(i) = 1 (flat)
```

**Why basic.** The contest scoring rule literally says `score = max_h sum_i pnl_h(i)`. The current pipeline submits 5 heads and only one is non-trivial. Routing per-row toward the strongest head is a textbook **arg-max** application of the scoring rule.

**Why disruptive.** Today every per-horizon head is tuned **independently** to a sum-PnL maximum, then we throw four away. PSHR aligns the per-row decision with the *max* aggregator at the column level. It changes the scoring units the pipeline tunes for from "h60 sum" to "max-h sum" — which is what actually matters.

**Mechanics.** Calibrate per-horizon shrinkage λ_h on OOF via a single 5-D DE search optimizing `max_h(sum_i pnl_h(i))`. Use existing T98 multi-horizon predictions (h5/10/20/40 LGB + CB ensembles already trained, plus h60 from iter_015). Per-row dispatch is ≈10 lines in `Predictor.predict`. No retraining.

**Implementation cost.** ~6h. Predictor.py modification + 5D DE on OOF + leave-2-fold-out validation.

**Est LOSO Δ.** On the constructive side: even if h60 wins for 90% of rows, the 10% redirected to short horizons can lift the *winning* horizon's sum without hurting (since the redirected rows would otherwise have been flat anyway). Realistic upside: +1.0–+2.5 over iter_015 LOSO (i.e. **+41 to +43**).

**Risk.** If T98's per-horizon ensembles are noisier than h60, PSHR underperforms. Sanity-check by floor-routing (fall back to h60 if EV_h60 within ε of best).

**Compliance.** sym-agnostic ✅; date not used ✅; per-row decision uses only that row's predictions ✅; stateless across calls ✅. The MAX rule is in PROGRESS.md §1.

---

### TRICK #2 — **Hierarchical Gate-then-Direction (HGD)** ★ Top-2

**What.** Replace the unified EV gate `pred > thr_up else pred < -thr_dn else flat` with a **two-stage cascade**:

```
stage 1 — abstain head:    p_trade(i) = sigmoid( g(features_i) )   # binary "should I trade?"
stage 2 — direction head:  s(i) = sign( pred_h60(i) )              # already exists
final action: 
    if p_trade(i) > tau_trade: action = 1 + s(i)
    else: action = 1 (flat)
```

**Why basic.** The 4-param EV gate is *forced* to express two semantically different decisions ("is the signal large enough" + "which side") with one thresholding rule. **They are different problems**: gating wants high precision on |Δmid|>fee; direction wants high precision on sign(Δmid). The same threshold sits at the wrong point for both.

**Why disruptive.** The "abstain head" can be trained with a **completely different objective** — e.g. binary cross-entropy on `y_trade = 1[|Δmid| > 2·fee]` with class weighting — which the regression head cannot do. This decouples 49.6% missed gap from 27.8% wrong-direction gap, the two failure modes diagnosed in §1.3.

**Mechanics.** Two free options:

* **Cheap path (no retrain).** Use existing T75/T87 *conviction proxy* `|pred_h60| / (q70 - q30)` (T86 quantile models exist) as a "trade probability surrogate", calibrated via an isotonic regressor on OOF mapping conviction → empirical P(|Δmid|>2·fee).
* **Quality path (light retrain).** Train a small LGB binary classifier (≈100 trees) on top of the same 359-d features predicting `y_trade ∈ {0,1}`. ~5 min CPU.

Direction is `sign(pred_h60)` (or the BAT/QSA-vetoed sign from prior brainstorm if combined). Calibrate `tau_trade` and the per-direction biases (`+δ_long, -δ_short` to handle prediction asymmetry) on OOF.

**Implementation cost.** Cheap: ~3h (Predictor only). Quality: ~5h (one CPU train + Predictor).

**Est LOSO Δ.** +1.0–+2.0. The cheap path alone often closes 30–50% of the missed-trade gap because the abstain decision is much smoother than a hard threshold.

**Risk.** Hidden double-counting if abstain head is highly correlated with `|pred_h60|`. Mitigate by orthogonalizing on residuals.

**Compliance.** Both heads fed only the 359-d global-normalized features — all CRITICAL_CONSTRAINTS clean.

---

### TRICK #3 — **Term-Structure Coherence Veto (TSCV)** ★ Top-3

**What.** Use the **shape of the multi-horizon prediction vector** as a veto signal:

```
slope_i        = pred_h60(i) − pred_h5(i)
convexity_i    = pred_h60(i) − 2·pred_h20(i) + pred_h5(i)
sign_agree_i   = #{h : sign(pred_h(i)) == sign(pred_h60(i))} / 5

action  = standard_EV_gate(pred_h60(i))
        IF sign_agree_i ≥ 4/5  AND  sign(slope_i) == sign(pred_h60(i))
ELSE flat
```

**Why basic.** A regression on Δmid at five horizons defines an expected price *trajectory*. A trade at h60 in direction `s` is only sensible if the trajectory **monotonically supports** `s`. Today we use only the endpoint.

**Why disruptive.** This is the first rule in this project to use multi-horizon predictions as a **direction-coherence test** rather than as a vote/ensemble. It directly attacks the **27.8% wrong-direction** failure: most wrong-direction trades come from h60 disagreeing with the implicit short-horizon trajectory.

**Mechanics.** Use T98 multi-horizon predictions (already on disk: `pred_lgb_h{5,10,20,40}_seed*.parquet`, plus h60 from iter_015). Compute slope/convexity/sign_agree per row. Calibrate the agreement threshold (`≥4/5`, `≥3/5`) on OOF.

**Implementation cost.** ~3h. Predictor.py only.

**Est LOSO Δ.** +0.8–+1.5. Conservative because veto-only — abstaining trims wrong-direction at the cost of some good trades; net positive when (precision ≫ recall) is the objective.

**Risk.** If short-horizon predictors are noisy on test sym, the veto over-fires. Use OOF-tuned `agree_threshold` to bound.

**Compliance.** Multi-horizon predictions on the *same* row, no cross-row state, sym-agnostic.

---

### TRICK #4 — **Win-Rate (Stochastic-Dominance) Threshold Calibration (WRTC)**

**What.** Replace DE's `max sum_pnl` objective with `max P(pnl > 0 | trade)` — i.e. tune (thr_up, thr_dn) to maximize the **win rate** on traded samples, with sum_pnl > c as a constraint.

```
max_θ  win_rate(θ)  =  P( pnl_i > 0 | action(θ, pred_i) ≠ flat )
s.t.   sum_pnl(θ) ≥ 0.95 · sum_pnl(θ_DE_max)
```

**Why basic.** The platform-vs-LOSO gap is dominated by **distributional shift**, not threshold variance (T83). A higher *win rate* is a more **distribution-stable** property than a higher *sum* — flipping a point from positive to negative across a regime shift is a much rarer event than flipping its magnitude. Optimizing for win rate trades small expected return for large robustness.

**Why disruptive.** This is **stochastic-dominance optimization** — the threshold is the one that wins more often, not the one that wins *bigger* on this specific OOF. Since we only get one platform shot every 12h, **robustness > expected return**.

**Mechanics.** Modify `de_thresh.py` objective. Run constrained DE: lagrangian `−win_rate + 100·max(0, sum_min − sum_pnl)`. Compare θ_WR vs θ_max_pnl on a date-half holdout.

**Implementation cost.** ~2h.

**Est LOSO Δ.** May *lower* LOSO by 0.5–1.0 (we sacrifice some sum) but **expected platform Δ +1 to +2** (the gap closes because the threshold transmits better). This is a robustness trade.

**Risk.** Over-conservatism: if win-rate-max is at a too-tight threshold, n_active drops and the absolute lift on platform is small.

**Compliance.** Pure post-processing, no constraint conflict.

---

### TRICK #5 — **Differentiable Decision Tree (DDT) Policy Head**

**What.** Replace the EV gate entirely with a **small differentiable decision tree** (depth 4, ~16 leaves) trained on PnL via REINFORCE / SPO+ surrogate. Inputs: `[pred_h5, ..., pred_h60, sigma_h60_seed_disagree, sigma_h60_quantile_spread, spread1, OFI, intraday_min, vol_window]` ≈ 12 features. Output: action ∈ {-1, 0, +1}.

```
DDT params:  splits {(feat_j, threshold_j)}_j   plus  leaf_actions {a_l ∈ {-1,0,1}}_l
training: minimize E_{(x,Δmid)~OOF}[ −pnl(DDT(x), Δmid) ]
        via straight-through Gumbel-softmax on the soft-tree's leaf weights
```

**Why basic.** Our current decision rule has **2 parameters**. Why? Tree booster has 1000s. Decision is exactly the place a decision tree should live. The 2-parameter EV gate is the dumbest possible policy that still tracks fee — it deserves to be a small model, not a constant.

**Why disruptive.** Replaces the 4-parameter scalar threshold with a 16-leaf piecewise-constant policy. Trained directly on PnL via differentiable surrogate. **This is the smallest amount of "policy learning" that is still principled.** Crucially, the tree is small enough that overfitting is bounded (16 leaves × 12 features ≈ 200 dof, vs T75 5-seed ensemble's millions of leaves ⇒ ratio < 10^-4).

**Mechanics.** Train via *differentiable* tree in PyTorch (Gumbel-softmax leaf gates), then compile to a hard tree at inference for `numpy`-only Predictor. Loss = `−PnL` with SPO+ surrogate (T87 already has this exact loss). One CPU run, ~30 min.

**Implementation cost.** ~8h. Highest training cost in this list, but still no GPU.

**Est LOSO Δ.** +1.5–+3.0. Subsumes BAT, QSA, SCG, IDS, VRSP into a single learned policy — the tree's splits will pick up whichever conditional structure is most predictive.

**Risk.** Overfit OOF if leaves end up too specific. Bound depth ≤ 4. Cross-validate against held-out OOF dates.

**Compliance.** Inputs are all CRITICAL_CONSTRAINTS-safe (no sym, no date, no cross-batch state). Tree at inference is pure numpy if-elif → trivially compliant.

---

### TRICK #6 — **Sign-of-Median Direction (SoMD) using T86 quantile triplet**

**What.** Instead of `direction = sign(mean(pred_seeds))`, use `direction = sign(median(q30, q50_proxy, q70))` and explicitly **abstain** when the posterior straddles zero.

```
mu_i        = pred_h60_mean(i)            # T75 ensemble
q30_i, q70_i = T86 quantile preds
sign_strict = sign(mu_i)
sign_robust = +1  if  q30_i > 0
            = -1  if  q70_i < 0
            =  0  otherwise (straddles zero)
final action: nonzero only if sign_strict == sign_robust ≠ 0
```

**Why basic.** Mean is sensitive to one-sided tails. If `q30 = -0.0005` and `q70 = +0.0010`, the mean is positive but the **support of the predictive distribution covers both directions**. The current pipeline takes a long position; the realized outcome may fall in the negative half.

**Why disruptive.** This converts the existing **T86 q30/q70 models** (already on disk, never used) from a midpoint estimator into an **uncertainty-aware sign classifier**. Functionally distinct from QSA (the prior idea), which uses the spread *magnitude* as a gate; SoMD uses the spread *symmetry around zero* as a sign-veto.

**Mechanics.** ~10 lines added to Predictor. Calibrate the abstain region (whether q30·q70 > 0 is the right rule, or `q30 > δ` for some δ > 0) on OOF.

**Implementation cost.** ~2h.

**Est LOSO Δ.** +0.4–+1.0. Especially attacks the wrong-direction bucket.

**Risk.** Combined with HGD/QSA could over-abstain. Track n_active.

**Compliance.** Same as QSA — quantile models are sym-agnostic.

---

### TRICK #7 — **OOD-Adaptive Abstain via Mahalanobis Distance**

**What.** Train an **OOD detector** (Mahalanobis distance in the 359-d feature space, computed against training covariance) and **force abstain** when row's Mahalanobis distance > τ_OOD.

```
Σ_train, μ_train = global covariance and mean (5 syms × 120 dates)
d2(x_i)          = (x_i − μ_train)^T  Σ^{-1}_train  (x_i − μ_train)
if d2(x_i) > τ:  action = 1 (flat),  bypassing all gates
```

**Why basic.** CRITICAL_CONSTRAINTS §1.3 says **test sym may be unseen**. The model is trained sym-agnostic, but a *new* sym's feature distribution can lie outside the training manifold — exactly the regime where the regression's predictions are least trustworthy. Yet the current gate has **no concept of OOD-ness**.

**Why disruptive.** This is the first rule in this project to acknowledge the **distributional reality of the §1.3 trap**. Other ideas treat the model as universally trustworthy; OOD-adaptive abstain explicitly says "I don't know what the new sym's prediction means; don't trade."

**Mechanics.** Pre-compute `Σ_train^{-1}` and `μ_train` once. At inference, compute Mahalanobis distance per row (fast: O(d²) ≈ 130k flops). Use a low-rank approximation (top 50 PCA components) for speed if needed. Calibrate τ on OOF such that the bottom 5% (most OOD) get abstained.

**Implementation cost.** ~3h. One-time covariance computation + per-row distance in Predictor.

**Est LOSO Δ.** Hard to estimate locally because *training* OOF doesn't have unseen sym. But: the local LOSO is roughly neutral (OOF rows are mostly in-distribution), and **platform Δ +1 to +3** if test contains genuine OOD. Hidden upside.

**Risk.** Covariance numerically unstable in 359-d; use low-rank PCA. False-positive abstain on legitimate but unusual rows.

**Compliance.** Σ and μ are global statistics, not per-sym. ✅

---

### TRICK #8 — **Action-Frequency Throttling on the Test Batch (AFT)**

**What.** Use the **batch-level rate** of would-fire actions as a self-consistency check; throttle when the rate exceeds historical OOF.

```
batch_active_rate(call) = #{i : EV_gate fires on i} / N_call
if batch_active_rate > 1.5 × OOF_active_rate:
    raise the threshold for this call only:  thr_call = thr · (rate / median_rate)
```

**Why basic.** OOF active rate ≈ 40%. If a test batch comes back with 80% would-fire rows, **either** the regression has bias-drifted (a known failure mode under OOD), **or** the batch is genuinely a high-volatility regime. In either case, the prior probability that 80% of rows are profitable trades is much lower than 80%.

**Why disruptive.** The first batch-level "self-throttling" mechanism. **Stateful within a single batch** (allowed — statelessness is *across* batches, §1.2), so this is exactly the kind of intra-batch signal that is legal but unused. It generalizes BAT (rank-based) to a **rate-based** safety valve.

**Mechanics.** ~10 lines in Predictor. No retraining.

**Implementation cost.** ~2h.

**Est LOSO Δ.** +0.3–+0.8 LOSO. Modest because OOF batches don't naturally exhibit rate spikes — but **strong robustness on platform** if OOD batches do.

**Risk.** Negligible if the throttle is only one-sided (raise threshold, never lower).

**Compliance.** Stateless across calls ✅.

---

### TRICK #9 — **Action-Asymmetry Regularization in DE Objective**

**What.** Modify DE's objective to penalize threshold asymmetry beyond what the regression's actual bias justifies:

```
estimated_bias    = mean(pred_OOF) − 0     # how far off zero-mean the regressor is
expected_asym     = 2 · |estimated_bias|
penalty(thr_up, thr_dn) = λ · (|thr_up − thr_dn| − expected_asym)²
DE_loss   = −sum_pnl(thr_up, thr_dn) + penalty
```

**Why basic.** Currently `thr_up = 3.72e-4`, `thr_dn = 1.61e-4` — a 2.3× asymmetry. T75 acknowledges this comes from a +0.000087 prediction bias. But the asymmetry the DE finds is **larger than the bias justifies**, because DE is also trying to fit per-fold idiosyncrasies. **The bias is a real distributional property; the residual asymmetry is overfit.**

**Why disruptive.** Frames DE's overfit not as a *parameter-count* problem (4 params is small) but as a **non-identifiability** problem (DE finds many `(thr_up, thr_dn)` pairs with similar PnL, picks one that exploits OOF-specific noise). Asymmetry regularization picks the *most defensible* pair.

**Mechanics.** Modify `de_thresh.py`. Sweep λ ∈ {0.1, 1, 10, 100} × FEE². Pick λ that gives flatter date-half holdout gap.

**Implementation cost.** ~2h.

**Est LOSO Δ.** May *reduce* LOSO by 0.2–0.5 but **platform +0.5 to +1.5** (T83 evidence: regularized DE strategies had `mean_gap = +0.09` vs unregularized `−0.34` on date-fwd vs date-bwd splits). Cheap robustness win.

**Risk.** None — the worst case is current behavior (λ = 0).

**Compliance.** Post-processing, fully compliant.

---

### TRICK #10 — **Cross-Horizon Independent Submission (CHIS): use all 5 heads as independent bets, route the winner via MAX**

**What.** Instead of routing per-row (Trick #1), let **each horizon use its own independently-tuned EV gate** and submit *all 5 active heads* for every row. Scoring picks `MAX_h(sum_h)`, so each horizon is a free shot.

```
for each row: emit  (action_h5, action_h10, action_h20, action_h40, action_h60)
              where each action_h is the h-specific gated decision
```

**Why basic.** This is the *literal* read of the scoring rule. The current pipeline emits flat on h5–h40, scoring **min over the active set**, not max. **We are deliberately throwing away 4/5 chances at the prize.**

**Why disruptive.** Tiny code change (drop the "set inactive horizons to flat" logic). Massive mental shift: the pipeline today acts as if h60 is "the answer". The truth is **the answer is whichever horizon happens to have the best prediction quality on the test slice**, and we don't know in advance which it will be.

**Mechanics.** Use T98's existing 5-horizon LGB+CB ensembles. For each h, run a per-h DE optimization independently. Predictor emits all five gated actions. Score is automatically max.

**Implementation cost.** ~3h (T98 already has the models; Predictor + per-h thresh.json).

**Est LOSO Δ.** "LOSO" itself is a sum, not max — so reporting LOSO from CHIS uses `max_h(LOSO_h)` which equals `LOSO_h60` ≈ +40 if h60 dominates. **The lift is on the *platform*, not LOSO.** If on the platform the test slice happens to favor short horizons (lower fees relative to trade size at α=0.05% for h5/h10), CHIS captures that bonus; if h60 dominates platform too, CHIS gives the same score as iter_015. **Strictly weakly better.**

**Risk.** Zero downside if per-h gates are independently calibrated to be PnL-positive on OOF.

**Compliance.** Same as iter_015 — sym-agnostic, stateless ✅.

> **Note: Trick #1 (PSHR) and Trick #10 (CHIS) are complementary, not redundant.** PSHR is a *router* that picks one horizon per row and zeros the rest, hoping to lift the winning horizon's sum. CHIS is *independent submission* — every row fires on every passing horizon. PSHR is more aggressive (concentrates bets); CHIS is conservative (always submits all). PSHR may *hurt* the winning horizon's sum if routing is imperfect; CHIS strictly cannot hurt. **Try CHIS first as the floor, PSHR second as the upside.**

---

### TRICK #11 (bonus, micro) — **Multi-Calibration via Subgroup Isotonic (MCSI)**

**What.** T15 isotonic failed globally (calibration improves NLL but hurts PnL — calibrated probability lacks magnitude). **Try isotonic per-subgroup**: bucket OOF by (spread1 quartile × intraday minute quartile) = 16 cells, fit a separate isotonic regressor per cell on `pred → Δmid`, apply at inference using the row's subgroup.

**Why basic.** T15 conflated two failure modes: bad calibration AND wrong subgroup-conditional means. Sub-grouping isolates them — within a homogeneous subgroup, isotonic should help; across heterogeneous regimes, it averages signal away.

**Why disruptive.** Multi-calibration (Hebert-Johnson 2018) is mathematically guaranteed to be *no worse* than global calibration on every subgroup. T15 proved global isotonic doesn't help; this would be the first subgroup attempt, with theoretical backing.

**Mechanics.** ~5h. Per-cell `IsotonicRegression.fit`. Predictor maps row → cell → cal(pred).

**Est LOSO Δ.** +0.3–+1.0. Modest, but cleaner failure-mode separation.

**Risk.** Sparse cells. Bound to ≥ 3000 samples per cell or merge.

**Compliance.** Subgroup features (spread, intraday minute) are sym-agnostic and date-free ✅.

---

### TRICK #12 (bonus, micro) — **Net-PnL Direct Regression (NPDR)**

**What.** Train a regressor whose **target is `pnl_h60 = sign(Δmid) · |Δmid| − 2·fee` only on samples where |Δmid| > 2·fee** (the rest are "flat" and don't generate gradient). Decision: `act = sign(pred), with abstain if pred < 0`.

**Why basic.** The regression target is currently `Δmid_norm`, a quantity that **does not coincide with the score**. The score is *net PnL*. Predicting net PnL directly removes the threshold layer entirely (`act iff pred > 0`).

**Why disruptive.** Reduces the decision parameter count from 4 (DE asymmetric) to 1 (a tiny offset to handle bias). Less to overfit, much simpler to validate.

> **Distinction from prior brainstorm Idea #14 (Net-of-Fee Regression Target).** Idea #14 trained on `Δmid − 2·fee·sign(Δmid)` over **all** samples. NPDR trains only on **the subset where |Δmid| > 2·fee** — a subset that is *itself the trade decision*. The two are different sample distributions and different functional forms.

**Mechanics.** Re-run T75-style 5-seed LGB on the filtered/transformed target. ~2h CPU.

**Est LOSO Δ.** +0.5–+1.5. Mostly equal to T75; the upside is robustness (no DE search to overfit).

**Risk.** Class imbalance — only ~30% of samples have |Δmid|>2·fee. Sample-weight to balance.

**Compliance.** Standard regression training ✅.

---

## 3 — Top Recommendations Ranked

| # | Name | Lift est. | Hours | Risk | Compliance | Stack-able with prior brainstorm? |
|---|------|-----------|-------|------|------------|-----------------------------------|
| 1 | **CHIS** (#10) — Cross-Horizon Independent Submission | +0.5 ~ +2 platform | 3 | ★ very low | ✅ | yes (orthogonal) |
| 2 | **PSHR** (#1) — Per-Sample Horizon Router | +1.0 ~ +2.5 LOSO | 6 | medium | ✅ | yes (replaces single-h focus) |
| 3 | **HGD** (#2) — Hierarchical Gate-then-Direction | +1.0 ~ +2.0 LOSO | 3-5 | low | ✅ | additive with QSA |
| 4 | **TSCV** (#3) — Term-Structure Coherence Veto | +0.8 ~ +1.5 LOSO | 3 | low | ✅ | additive with HGD |
| 5 | **DDT** (#5) — Differentiable Decision Tree Policy | +1.5 ~ +3.0 LOSO | 8 | medium-high | ✅ | replaces all gating |
| 6 | **WRTC** (#4) — Win-Rate Calibration | +1 ~ +2 platform | 2 | very low | ✅ | replaces DE objective |
| 7 | **SoMD** (#6) — Sign-of-Median Direction | +0.4 ~ +1.0 LOSO | 2 | low | ✅ | additive with QSA/HGD |
| 8 | **OOD-Abstain** (#7) | +1 ~ +3 platform (hidden) | 3 | low | ✅ | additive |
| 9 | **AFT** (#8) — Active-Frequency Throttling | +0.3 ~ +0.8 LOSO | 2 | very low | ✅ | additive |
| 10 | **DE Asymmetry Reg** (#9) | +0.5 ~ +1.5 platform | 2 | very low | ✅ | replaces de_thresh.py obj |
| 11 | **MCSI** (#11) — Subgroup Isotonic | +0.3 ~ +1.0 LOSO | 5 | medium | ✅ | additive |
| 12 | **NPDR** (#12) — Net-PnL Direct Regression | +0.5 ~ +1.5 LOSO | 2 (CPU) | low | ✅ | replaces T75 target |

---

## 4 — Suggested Execution Order (1–2 weeks of 1-worker time)

**Day 1 (free wins, no retrain).**
- CHIS (#10): no-cost extension of iter_015 to all-5-horizon-active (~3h).
- WRTC (#4) and DE Asymmetry Reg (#9): ~4h. Provides regularized thresholds — submit as a defensive pack.

**Day 2 (medium retrains using existing predictions).**
- HGD (#2 cheap path): isotonic on existing pred_T86 quantiles → conviction → trade decision (~4h).
- TSCV (#3): consume T98's multi-horizon predictions. (~3h).
- SoMD (#6): trivial add. (~1h).
- AFT (#8): trivial add. (~1h).

**Day 3 (more involved).**
- OOD-Abstain (#7): covariance fit + per-row distance (~4h).
- PSHR (#1): 5D DE on max-aggregator (~5h).

**Day 4–5 (training tasks).**
- DDT (#5): differentiable tree training, compile to numpy at inference (~8h).
- NPDR (#12): retrain T75 on filtered/transformed target (~2h CPU + eval).

The first 4 of these are all Predictor-only and stack-able. CHIS+WRTC+HGD+TSCV+AFT combined: estimated +2.5–+5 LOSO over iter_015, transmitting to **+1.5–+3.5 platform**.

---

## 5 — What I am explicitly NOT proposing

| Idea | Why rejected |
|------|--------------|
| Per-sym threshold | §1.3 forbids — already covered |
| Confidence-based fractional sizing | Output spec is `{0,1,2}` — no fractional |
| Cross-batch position carryover | §1.2 forbids stateless across calls |
| TTT / online fine-tune | banned by user |
| Retrain T75 with new feature set | out of scope (decision-layer focus) |
| Online ensemble weight learning | requires past-batch state |
| Calendar-derived features | `date` is zeroed (§1.1) |
| Date-conditional decision policy | `date` is zeroed (§1.1) |
| Simply re-tune the existing 4D DE | T83 falsified this lever |
| Smaller batch DE search | T83 falsified — gap is intrinsic, not DE noise |

---

## 6 — Why these 12 differ from the prior brainstorm

The IDEA-CTX-DECISION brainstorm focused on **uncertainty-aware thresholding** (BAT, CSSC, HBT, QSA), **context-conditioning** (SCG, IDS, VRSP), and **microstructure boosters** (OCM, BABF). T102's contributions are:

- **Scoring-rule alignment** (CHIS, PSHR): nobody before exploited the MAX-over-horizons rule.
- **Decision decomposition** (HGD, SoMD, NPDR): nobody before separated abstain from direction at the architectural level.
- **Multi-horizon coherence** (TSCV): nobody before used the *shape* of the term structure as a veto.
- **Distribution-shift robustness** (WRTC, OOD-Abstain, DE Asymmetry Reg): nobody before treated the LOSO/platform gap as a robustness problem rather than a tuning problem.
- **Learned policy head** (DDT): nobody before parameterized the action layer with anything richer than two scalars.
- **Self-throttling** (AFT): nobody before used batch-rate as a self-consistency check.
- **Subgroup calibration** (MCSI): nobody before retried isotonic on a subgroup-conditional basis after T15's global failure.

Each of the twelve is a **structurally new instrument**, not a parameter tweak.

---

```
RESULT: task=radical_decision_rethink top_recommendations=[CHIS,PSHR,HGD,TSCV,DDT,WRTC,SoMD,OOD-Abstain,AFT,DE-Asym-Reg,MCSI,NPDR] notes="12 orthogonal-to-prior-brainstorm tricks; CHIS/PSHR exploit MAX-over-horizons scoring rule (the largest unrecognized lever); HGD/SoMD/NPDR decompose abstain-vs-direction; TSCV uses term-structure coherence; WRTC/OOD/AFT/DE-Asym-Reg target distribution-shift robustness; DDT replaces 2-param gate with 16-leaf learned policy. Combined upside +2.5~+5 LOSO over iter_015, mapping to +1.5~+3.5 platform."
```
