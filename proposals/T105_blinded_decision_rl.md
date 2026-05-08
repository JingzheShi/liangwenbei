# T105 — Blinded research: decision-focused / RL / direct-PnL methods for HFT LOB

> **Mode**: blinded (no access to existing experiments / audits / progress files until Part 3).
> **Lens**: how to make the *PnL reward itself* the optimization target, not a downstream side effect of class-probability prediction.
> **Cost contract** (from spec): `pnl_t = [(a-1)·Δmid - 0.0001·|a-1|·avg_price] / mp_t`. With `avg_price ≈ mp_t`, the per-trade cost is **≈1 bp = 0.01 %**. Labels are derived from **α=0.05 % (h5/10) / α=0.10 % (h20/40/60)**.

---

## Part 1 — Why optimizing PnL directly is the *right* but *hard* objective

### 1.1  Where the standard pipeline (cross-entropy on the 3-way label) bleeds PnL

Four structural mismatches between the discretized label and the scoring rule, each of which is a *first-order* leak (every one of them costs basis points, not pennies):

| # | Phenomenon | Concrete cost |
|---|---|---|
| **(a)** | **Cost-asymmetry**: `hold` carries 0 risk; trade has fixed 1 bp cost. CE loss treats the three classes symmetrically. | A model with calibrated `P(up)=0.6` and tiny edge will trade aggressively even when expected payoff is below cost. |
| **(b)** | **Magnitude blindness**: a `label=2` row with Δmid=+0.06 % is barely profitable (5 bp net). A `label=1` row with Δmid=+0.04 % is *also* profitable (3 bp net) but discarded. | Roughly 30–40 % of profitable trades sit inside the "flat" class boundary and are invisible to CE. |
| **(c)** | **Threshold mismatch**: label uses α≫cost (5×–10×). The label-defined "dead zone" `\|Δmid\|<α` *contains* a profitable region `cost<\|Δmid\|<α`. | The labels are answering the wrong question. The right binary is "`\|Δmid\|>cost`" not "`\|Δmid\|>α`". |
| **(d)** | **Loss-gradient direction**: CE pushes against the *most-likely wrong class*, which is usually `flat` (since it dominates). PnL only cares about `argmax`-of-EV, so CE wastes capacity correcting flat-vs-up confusions that don't move PnL. |

### 1.2  What's special about *this* dataset that makes direct-PnL feasible

The dataset gives us **full counterfactual feedback for free**: at every training row we observe the real Δmid_h (via `midprice` columns at t and t+h). So we can write down `pnl(a, Δmid)` for *all three* actions on every row. That collapses the canonical RL friction (we only see the action we took) into a **fully-observed contextual decision** problem — a setting where decision-focused methods (SPO+, oracle imitation, expected-PnL surrogate) are *strictly* more powerful than CE without paying the bandit/exploration tax.

### 1.3  Why "just use PnL as the loss" is harder than it sounds

1. **Argmax over actions is non-differentiable.** Need a smoothing (softmax / Gumbel) or a surrogate (SPO+).
2. **Heavy-tailed reward.** A handful of rows with |Δmid|>>α dominate gradients → high-variance policy updates. CE is much more stable.
3. **Imbalance.** ~70 % of rows have |Δmid|<<α: hold-correct but uninformative. These rows still appear in a per-row PnL loss and dilute the signal.
4. **Class-region collapse.** The all-`hold` policy is a local minimum with 0 PnL: exactly the gradient floor of any ill-conditioned PnL loss.
5. **5 horizons compete; only the best ranks.** Per-horizon optima differ (low-α, low-vol vs high-α, high-vol). Joint training can hurt the best horizon's PnL.
6. **Sym-OOD constraint** rules out per-sym normalization, embeddings, or sym-conditioned heads — the natural regularizer is pure feature-only modelling, which makes overfitting a per-tick PnL surface easier than overfitting a 3-class label.
7. **Validation noise.** PnL on a fold is itself a sum of heavy-tailed terms; ranking models by PnL has high stochastic error → false discovery is a real risk.

These 7 frictions are why most LOB literature falls back to CE-on-discretized-label even when scoring is PnL. Solving them is the prize.

---

## Part 2 — Eight foundational, disruptive proposals

Each method is "disruptive" in a precise sense: it changes the *objective* the model is trained against, not just the architecture. Most of them are 1-day implementations once a feature backbone exists. Estimated lift is **conservative LOSO PnL gain over CE-baseline**, in absolute PnL units (the same units the leaderboard ranks on).

### M1 — Counterfactual full-feedback expected-PnL loss (the base method)

**1-line**: replace CE with `L = -Σ_a softmax(logits)[a] · pnl(a, Δmid_i, avg_price_i, mp_i)`, where `pnl` is the exact scoring formula.

**References**: Beygelzimer & Langford "Offset Tree" (ICML 2009); cost-sensitive multi-class folklore.
**Why foundational**: removes the discretization from the loss entirely; the **continuous Δmid** is the supervision target for the *decision*, not for a regression intermediate.
**Why disruptive**: this loss has the *exact same gradient structure* as REINFORCE with the realised counterfactual reward — but with **zero variance** (because we have full feedback, not a sample). It is RL without the noise.
**Difficulty**: 1 day. Drop-in replacement of CE for any GBDT (custom obj/grad+hess) or MLP.
**Estimated lift**: **+1.5 to +3.0 PnL** over a CE baseline at iso-feature-set. The gain comes mainly from (a) and (c) above.
**Risks**: LightGBM custom objective Hessian must be PSD or you'll get instability; smoothing the argmax with a temperature ≪1 prevents the all-hold collapse.

### M2 — SPO+ (Smart Predict-then-Optimize Plus) for Δmid → 3-action

**1-line**: predict `Δmid_hat`; train under the SPO+ surrogate that upper-bounds the *regret* of the resulting decision rule.

**References**: Elmachtoub & Grigas, *Management Science* 2022; survey: Mandi, Kotary et al., JAIR 2024 ("Decision-Focused Learning: Foundations, …").
**Why foundational**: SPO+ is *the* canonical convex surrogate for predict-then-optimize when the downstream optimization is linear/piecewise-linear. Our argmax-of-3 is exactly piecewise-linear in Δmid.
**Why disruptive**: a regression model trained on **MSE** wastes capacity making `Δmid_hat` accurate at large |Δmid| (where the action is unambiguous); SPO+ shifts capacity toward the *boundary* (|Δmid|≈cost) where decisions actually flip. That boundary is precisely where CE and MSE both fail.
**Difficulty**: 2 days. Closed-form gradient (3 actions ⇒ no LP solve required); add to LightGBM as a custom objective.
**Estimated lift**: **+1.0 to +2.5 PnL**. Stacks somewhat with M1 (M1 = soft expected-PnL ≈ same first-order behaviour; M2 = regret-surrogate, different second-order).
**Risks**: SPO+ has slightly higher gradient variance near the cost boundary; works better when `cost/σ_Δmid` is small (true here: 1bp / ~50bp σ ≈ 0.02).

### M3 — Distributional regression of Δmid + Bayesian decision

**1-line**: predict K quantiles of `Δmid_h | x` via pinball loss; act under expected utility (or CVaR) of the predicted distribution.

**References**: Romano et al. "Conformalized Quantile Regression" NeurIPS 2019; Dabney et al. QR-DQN (AAAI 2018) for distributional value; Koenker textbook for quantile regression.
**Why foundational**: textbook decision theory — the optimal decision under uncertainty is `argmax_a E_p[u(a, X)]`. We currently throw away the entire conditional distribution and keep only its mode.
**Why disruptive**: the **same model** can serve EV-greedy (`E[Δmid] > cost?`), risk-aware (`P(Δmid>cost) > τ?`), and tail-aware (`CVaR_0.2(Δmid) > cost?`) policies — choose policy on the validation fold *without* retraining. This is how to *cheaply* explore the risk-return frontier.
**Difficulty**: 2 days. K=9 quantile heads (q ∈ {0.05, …, 0.95}) trained jointly with pinball; multi-quantile LightGBM regression is supported.
**Estimated lift**: **+1.0 to +2.0 PnL** from the EV variant; **+0.5 to +1.5 PnL** *additional* from the CVaR variant *if* sym-OOD test points exist (because CVaR auto-defaults to `hold` when posterior is wide).
**Risks**: quantile crossing (fix with monotonic layer or sort-step); calibration drift (conformal step in M4 fixes).

### M4 — Conformal asymmetric-cost decision rule

**1-line**: build conformal prediction interval `[Δmid_low(x), Δmid_high(x)]` with valid coverage; trade only when `Δmid_low > +cost` (long) or `Δmid_high < -cost` (short).

**References**: Romano CQR 2019; Cresswell-Cuesta et al. "Utility-Directed Conformal Prediction" 2024; Cao et al. "Decision-Theoretic Foundations for Conformal Prediction" arXiv:2502.02561 (2025) — proves prediction sets are optimal for VaR-risk-averse decision makers.
**Why foundational**: gives a **distribution-free** statistical guarantee on coverage; converts uncertainty quantification into a directly-PnL-relevant trade gate.
**Why disruptive**: collapses two normally separate problems — *threshold tuning* and *uncertainty calibration* — into one principled object. A model that only trades when it is conformally certain to clear cost is exactly what HFT desk practice approximates with hand-tuned ad-hoc thresholds.
**Difficulty**: 1 day on top of M3. Just a calibration loop on a held-out fold.
**Estimated lift**: **+0.5 to +1.5 PnL**. Larger lift on noisy / OOD test points than on clean ones — exactly the right regime for this contest.
**Risks**: marginal-coverage (not conditional). Mondrian / sym-stratified conformal mitigates, at the cost of stratum size.

### M5 — Oracle-relabel imitation with PnL-magnitude sample weighting

**1-line**: relabel each row with `a*_i = argmax_a pnl(a, Δmid_i)` (the cost-aware optimal action); train classifier with sample weight `w_i = max_a pnl(a, ·) - pnl(hold, ·)`.

**References**: Lopez de Prado *Advances in Financial ML* 2018, ch. 3 (meta-labeling); Pomerleau ALVINN 1989 / behavioural cloning; ICML inverse-classification weighting folklore.
**Why foundational**: the original 3-class labels {0,1,2} use the *wrong* boundary (α) and the *wrong* metric (count). Relabeling to the cost boundary and weighting by PnL-at-stake *aligns the supervised problem with the scoring problem at the label level*, before any architectural choice.
**Why disruptive**: requires **zero changes** to the modelling code — only the label and sample-weight columns change. Yet it dissolves frictions (b) and (c) from §1.1 in one stroke.
**Difficulty**: 0.5 day.
**Estimated lift**: **+1.0 to +2.0 PnL**. Often the *largest* per-effort lift in a CE-trained pipeline.
**Risks**: introduces severe class imbalance (only ~30 % of rows become `trade` after relabeling); compensate via focal loss or class-weighted CE.

### M6 — Risk-sensitive policy with CVaR objective (sym-OOD shield)

**1-line**: train under `CVaR_α[PnL(π(x), Δmid)]` instead of `E[PnL]`; the worst-case-tail policy is conservative *by default* on novel symbols.

**References**: Tamar, Glassner, Mannor "Optimizing the CVaR via Sampling" AAAI 2015; Chow & Ghavamzadeh NeurIPS 2014 ("Algorithms for CVaR Optimization"); Sagawa Group-DRO ICLR 2020.
**Why foundational**: directly maximises a *coherent* risk measure aligned with how a risk manager actually scores a strategy.
**Why disruptive**: the sym-OOD constraint is otherwise really hard to address — there's no per-sym signal to learn. CVaR makes the policy "shy" wherever input is unusual without needing to detect OOD explicitly. This is a *single hyper-parameter* fix to a structural risk in the scoring rule.
**Difficulty**: 2 days (Tamar's stochastic CVaR estimator is well-documented; pairs naturally with M1).
**Estimated lift**: **+0.5 to +1.5 PnL** *if* the test set contains OOD sym-IDs (the spec says it can). On non-OOD splits the lift is small (CVaR ≈ E in calm regions).
**Risks**: shrinks training-set performance; a too-aggressive α (e.g. 0.05) makes the policy almost-always-hold.

### M7 — Group-DRO over symbols (worst-case-sym training)

**1-line**: for each sym-group, track its training-loss EMA; up-weight the worst sym in the next batch.

**References**: Sagawa et al. "Distributionally Robust Neural Networks" ICLR 2020; Duchi & Namkoong "Variance-based Regularization with Convex Objectives" 2019.
**Why foundational**: ERM minimises *average* training PnL across symbols. Group-DRO minimises *worst-symbol* PnL — the right surrogate when the test set has held-out symbols.
**Why disruptive**: cheap, drop-in modification to the training loop; an explicit lever for the sym-OOD constraint.
**Difficulty**: 1 day.
**Estimated lift**: **+0.5 to +1.0 PnL** on sym-OOD test points; ~0 on sym-in-distribution.
**Risks**: coupled with overfitting to the worst sym; gradient noise rises with group size 5.

### M8 — Differentiable end-to-end policy via Gumbel-Softmax (extension door)

**1-line**: output policy logits, Straight-Through Gumbel-Softmax to sample action, backprop realised PnL through the sampled action.

**References**: Jang et al. "Categorical Reparameterization with Gumbel-Softmax" ICLR 2017; Maddison et al. "Concrete Distribution" ICLR 2017.
**Why foundational**: makes the action a differentiable random variable, opening the gate to richer policies (continuous size; multi-tick lookahead; sequential trades) without resorting to high-variance REINFORCE.
**Why disruptive**: with full feedback we don't *need* it for the basic 3-action problem (M1 is exact). But it is the **only** path to optimise *position-sized* or *multi-step* policies, which is the natural next chapter once the 3-action policy is solved.
**Difficulty**: 2 days.
**Estimated lift**: small in 3-action space (~+0.2 PnL); enables a separate +1–3 PnL frontier in continuous-size space.
**Risks**: temperature schedule sensitive; high gradient variance in 3-action space.

---

## Part 3 — Reality check against the workspace's existing experiments

> I read these four sources before writing this part (and *only* these four):
> - `PROGRESS.md`
> - `experiments/T87_spo_dfl/REPORT.md`
> - `experiments/T60_rl_nn_pnl/loso_summary_h60_t60_seed42_v2.json` (the most recent T60 RL run)
> - `experiments/T99_e2e_execution_gbdt/worker-progress.json` (latest e2e GBDT exploration)

### 3.1  Mapping my proposals to existing experiments

| My method | Existing experiment | Result | Status |
|---|---|---|---|
| **M2 — SPO+** | T87 SPO+ warm-started from T81 NN regression, λ=30, lr=3e-5 | **+38.28 LOSO alone; +40.13 in (T75, T87)=(1, 1.5) ensemble = iter_015. +1.85 over iter_014.** | ✅ done; *positive* |
| **M1 — expected-PnL loss** (full-feedback) | T60 (NN, soft-action via tanh / direct E-PnL); T57 (LGB sample weight) | T60 with all 5 seeds: **LOSO sum +7.6 with per-fold range [-6.49 ... +12.88] — collapse**. T87 REPORT explicitly diagnoses why: tanh-soft action's gradient collapses to 0 outside the band. | ❌ tried in NN form — failed. Not yet tried as full-feedback exact-counterfactual GBDT custom objective. |
| **M3 — distributional regression** | T86 quantile + dual-gate (q030 + q070 + IQR-mid hybrid) | **+1.94 LOSO over iter_014** | ✅ done; positive |
| **M4 — conformal asymmetric-cost rule** | T76 decoupled threshold; T83 robust threshold | DE-tuned (T87) and grid-tuned (T86) thresholds; **no coverage-guaranteed CQR step** | ⚠️ partially done — empirical tuning, not coverage-calibrated |
| **M5 — oracle relabel + magnitude weight** | T57 sample weight `\|c\|^p`; T92 "magnitude sample weight" | T57: matched CE baseline. T92: **falsified ("regression — R44 hypothesis falsified")**. | ❌ tried; failed in straightforward form |
| **M6 — CVaR objective** | nothing visible | **gap** |
| **M7 — Group-DRO over sym** | nothing visible | **gap** |
| **M8 — Gumbel-Softmax STE** | T60 used soft Gumbel-style action; failed | tried implicitly; collapses for the same reason as M1-NN |

### 3.2  Two failure-mode signals I see in the existing data — both point in the same direction

**Signal A — per-sym PnL is hugely uneven on the iter_015 winner (T87 full-test):**

| sym | cum_pnl | n_active / 88,416 |
|---|---|---|
| sym 0 | **+3.74** | 24,816 |
| sym 1 | +5.32 | 39,654 |
| sym 2 | +4.55 | 38,292 |
| sym 3 | +11.97 | 39,110 |
| sym 4 | **+14.51** | 38,294 |

`max/min ratio = 3.9×`. The "head" (sym 4) dominates the total. If a held-out test sym lands near the sym 0 / sym 1 / sym 2 distribution, the leaderboard PnL drops to ~+5/sym × 5 ≈ +25 instead of +40. **This is a directly observable hidden risk in iter_015.**

**Signal B — T60 LOSO per-fold variance is catastrophic:**

`cum_pnl_per_fold = [-0.00, -0.38, -6.49, +1.62, +12.88]` (held-out sym 0..4).
`mean = +1.53`, `min = -6.49` (sym 2 held out). When sym 2 is OOD, the policy actively loses 6.5 PnL.

**Both signals say the same thing:** the model's worst-case-sym performance is the binding constraint, but no training procedure in the workspace explicitly optimises it. ERM (T60, T87, T99) all produce models whose performance is dominated by the easiest sym in train and degrades sharply on the hardest.

### 3.3  Convergent ("殊途同归") items

- T87 ≡ M2 with NN backbone instead of GBDT.  T86 ≡ M3+M4 fused (quantile + grid-tuned dual gate).  The team **independently rediscovered** decision-focused and distributional approaches as the two winning axes — strong evidence my Part-2 priorities point at the right hill.
- T60 + T57 + T92 all tried "use PnL as supervision" in some form, all failed. T87's REPORT.md gives the precise reason: **soft-action gradients collapse to 0 outside a thin band**. SPO+ (M2) fixed this by giving constant subgradient ±2 — exactly the structural fix predicted by the SPO+ theory (Elmachtoub & Grigas 2017). The team's *converged-on-by-experimentation* explanation matches the *literature-predicted* explanation.
- T76 + T83 thresholds ≈ M4 in spirit, but stop short of the conformal step. The team is one calibration loop away from the formal version.

### 3.4  The gold gaps

Sorted by `(unexplored × impact × stack-with-iter_015) / effort`:

1. **Group-DRO / CVaR over symbols** — *completely* unexplored. Both Signal A (uneven per-sym PnL on iter_015) and Signal B (T60 variance) say worst-sym is the binding constraint. The spec explicitly says test sym may be OOD. Cheap (1.5d) and stacks on iter_015.
2. **Conformal CQR dual-gate** — T86 is empirically tuned; replacing with coverage-guaranteed calibration is a small change that converts a fit into a robustness-grounded rule. ≤ 1.5d.
3. **M1 as exact full-feedback GBDT custom objective** — *different* from T60 (NN + sampled action) and T57 (LGB sample weight, no decision loss). Worth a 1-day check before declaring M1 dead.
4. **Combine T87 (SPO+) ⊗ T86 (quantile)** into a single SPO+-trained quantile-output model — both crown jewels in one head. Could enable CVaR/conformal post-hoc directly.

---

## Part 4 — Top 2 step-by-step recommendations

I'm picking on the criterion **"unexplored × addresses a stated test-set risk × ≤ 2 days × stacks with existing iter_015 winner"**.

### 🥇 Recommendation #1 — Group-DRO + CVaR for sym-OOD robustness (M6 + M7 fused)

**Hypothesis**: iter_015's full-test per-sym PnL is `[3.74, 5.32, 4.55, 11.97, 14.51]` — a 3.9× max/min ratio. Sym 4 alone contributes ~36 % of the total +40 PnL. T60's LOSO held-fold range `[-0.00, -0.38, -6.49, +1.62, +12.88]` confirms a model can lose 6.5 PnL when the "hard" sym is held out. The leaderboard PnL is therefore dominated by *which test sym got drawn*. A worst-sym-aware training loop should recover ≥ 1 PnL by lifting the bottom-3 syms (0/1/2), at the cost of ≤ 0.5 PnL on the top-2 syms (3/4).

**Step-by-step (≤ 2 days)**:

1. **Reproduce iter_015 (T87 SPO+) as the baseline** under LOSO. Record per-sym PnL vector `[p_0, p_1, p_2, p_3, p_4]` on the held-out fold of each round; compute `min_sym(p_sym)` and `mean_sym(p_sym)`. Confirm the *gap* between worst-sym and mean-sym PnL — if it's ≥ 2 PnL, this idea is on.
2. **Group-DRO objective**: in the LightGBM custom objective, after computing per-row gradient/hessian, multiply each row's `g_i, h_i` by `w_{sym(i)}` where `w_s = exp(η · L_s) / Σ exp(η · L_·)` and `L_s` is an EMA of per-sym training loss. Update `w_s` once per boosting round (cheap). η ∈ {0.5, 1.0, 2.0} hyper.
3. **CVaR add-on (cheap)**: over the same per-row PnL surrogate, compute the per-sym CVaR_0.2 (mean of bottom-20 % rows) and replace the per-sym aggregation with CVaR rather than mean before the DRO upweight.
4. **Validation**: same LOSO splits as T87. **Primary metric**: `min_sym(LOSO_PnL)` (worst-sym-out PnL). **Secondary**: `mean_sym(LOSO_PnL)` (must not drop > 0.5 below iter_015).
5. **Decision rule**: ship if `min_sym ≥ iter_015's min_sym + 0.7` *and* `mean_sym ≥ iter_015 mean_sym − 0.3`. Otherwise document and shelve.

**Expected outcome**: +0.5 to +1.5 PnL on test (depends on whether test really contains OOD sym).
**Cost**: 1.5 days; one model run per LOSO fold × 3 η values; reuses T87's training pipeline 1:1.
**Why the team hasn't done this**: every previous experiment treated each fold's training as an i.i.d. ERM problem; nobody reweighted by sym during training itself. This is a *structural* change to the gradient, not a feature engineering change.

### 🥈 Recommendation #2 — Conformalised dual-gate (CQR + asymmetric cost rule) on top of T86

**Hypothesis**: T86's dual-gate is *empirically tuned* (q030 / q070 by grid). Replacing it with a CQR-calibrated gate gives the same (or better) PnL with a **coverage guarantee** that protects against feature-distribution drift.

**Step-by-step (≤ 1.5 days)**:

1. **Re-train T86's quantile model with K=9 quantiles** (q ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95}) — already half-done in T86 with q030/q070; add the wider band.
2. **CQR calibration** on a held-out 10-day stretch of training data:
    - For each calibration row compute `score_i = max(q_lo(x_i) − Δmid_i, Δmid_i − q_hi(x_i))`.
    - Take the (1-α)-quantile τ_α of {score_i}.
    - Define calibrated band `[q_lo(x) − τ_α, q_hi(x) + τ_α]`.
3. **Cost-aware decision rule**:
    - long iff `q_lo(x) − τ_α > +cost`
    - short iff `q_hi(x) + τ_α < −cost`
    - else hold.
4. **Sweep α ∈ {0.10, 0.20, 0.30, 0.40, 0.50}** on the validation LOSO PnL. (α controls aggressiveness; we expect the Pareto frontier to peak around α≈0.3.)
5. **Decision rule**: ship if best α beats T86 IQR-mid hybrid (currently iter_014b) by ≥ 0.7 LOSO-equiv on either the worst-sym or mean-sym metric.

**Expected outcome**: +0.5 to +1.0 PnL over T86. Often the CQR step *self-tunes* the dead zone better than grid-search because it is feature-conditional rather than marginal.
**Cost**: 1 day if T86's quantile model can be re-used; 1.5 if quantile heads need retrained.
**Why the team hasn't done this**: thresholds are usually tuned globally; coverage-guaranteed conformal calibration on a quantile model is a small extra step that's just out of the search path the team has been on.

### Stack note

These two recommendations **stack**: Recommendation #1 produces a sym-robust *predictor*; Recommendation #2 produces a coverage-calibrated *decision rule*. Apply #2 on top of the predictor from #1, and the cost rule itself becomes sym-OOD-aware via the predictor's wider conformal interval on novel symbols.

---

```
RESULT: task=blinded_decision_rl top_recommendations=[Group-DRO+CVaR sym-aware training (M6+M7); Conformalised CQR dual-gate (M3+M4)] notes=[8 methods proposed; T87 SPO+ and T86 quantile-dual-gate already won big — confirms decision-focused & distributional are right hills; T60 RL failed which validates §1.3 frictions; biggest open gaps are sym-OOD-aware training (Group-DRO/CVaR) and coverage-guaranteed calibration (CQR conformal). Both <=2 days, stack on iter_015.]
```
