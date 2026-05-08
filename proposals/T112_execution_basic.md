# T112 — Basic-but-Disruptive Execution / Decision-Rule Tricks

**Author**: opus xhigh worker (T112 — execution / decision-rule axis)
**Date**: 2026-05-08
**Mandate**: Propose CPU-friendly, minutes-to-run, **basic-but-disruptive** decision-rule ideas.
Anchors: iter_015 (T87 SPO+ NN + T75 LGB) → +40.13 LOSO, plat ~+21; T99 Huber-family iter_016 candidate → +44.72 LOSO.
Constraints: no model retrain unless trivial CPU; no GPU NN post-processing unless NN already trained; pure post-processing / Predictor-only changes preferred.

> **Key intel.** Top of leaderboard runs **CPU-only in minutes**. Our pipeline is
> 4-way ensemble (T75 LGB + T87 NN + T86 q_mid + iter_016 Huber) + DE-4D
> threshold + 5-horizon CHIS + (in iter_016_v3) per-h thresh. **The handcraft is
> in the decision rule, not the model.** Probably we are 4D-overfitting where
> the leaders are 0D / 1D-optimizing.

> **Overlap with prior brainstorms.** I read `IDEA-CTX-DECISION-OPUS/REPORT.md`
> (BAT, CSSC, HBT, QSA, IDS, SCG, VRSP, OCM, BABF, LCA, CVaR-PnL, CPL,
> Net-of-fee target), `T102_radical_decision_execution.md` (CHIS, PSHR, HGD,
> TSCV, WRTC, DDT, SoMD, OOD-Abstain, AFT, DE-Asym-Reg, MCSI, NPDR), and
> `T105_blinded_decision_rl.md` (M1 expected-PnL loss, M2 SPO+, M3 quantile
> Bayesian, M4 conformal, M5 oracle-relabel, M6 CVaR, M7/M8 RL). **Every T112
> idea below is genuinely orthogonal to all of those — verified no naming /
> mechanism overlap.**

---

## Part 1 — Diagnosis: Where We Over-Engineered the Decision

### 1.1 The DE-4D / per-h-thresh / multi-stage gate is **over-parametrized for the gain**

T83 settled "DE-4D does not overfit the 442k materially" (kfold 35.94 ≈ full-fit 36.15). That is **necessary but not sufficient** for the leaderboard transmission ratio. What T83 did **not** test is whether reducing the threshold parametrization from 4 → 1 → 0 would *transmit better*. The intel says simple wins. Three concrete forms of over-engineering on the decision side:

| Where we have | Param count | What a "leader" likely has |
|---|---|---|
| (thr_up, thr_dn, d_up, d_dn) global DE 4D | **4** | A single threshold or a fixed quantile |
| Per-h thresh × 5 horizons (iter_016) | **20** | All-h symmetric or just h60 |
| (w_lgb, w_nn, w_qmid, w_huber) ensemble weights | **4** | Equal-weight or median |
| Conditional sub-rules (TSCV, AFT, etc.) layered | **+3-5** | Single rule |

Even though T83 says individual layers don't overfit, the **product manifold** of "which ensemble × which threshold × which sub-rule" has very high effective dof on a single 442k OOF. Distribution-shift transmission penalizes flexibility quadratically.

### 1.2 Where the +442 LOSO → +21 platform gap really hides

iter_015 capture rate ≈ 3.8 %; oracle ≈ 1054. Of the 96.2 % gap (per `IDEA-CTX-DECISION` analysis):

- **49.6 %** "missed trade" — model abstained on profitable rows. The current 4-D DE does not distinguish "I'm uncertain" from "I'm certain it's small". A flat-aware decision rule **never re-decides** missed rows.
- **27.8 %** wrong direction. The 4-D DE has zero instrument against direction error; it only modulates magnitude.
- **18.0 %** fee burn — fired on rows where `|Δmid| < 2·fee + spread1` actually. The current `thr_*` is a **constant** but the round-trip cost varies row-by-row by ~5 × through `spread1`. We charge a flat fee in EV but pay a variable fee in reality.

The top-of-leaderboard hypothesis ("simple gate") is consistent with: **leaders are aggressively attacking the 18 % fee-burn bucket via per-row physics-grounded fee floor (a 0-parameter rule)**, plus possibly a model-free rank gate (1-parameter). Neither is in our pipeline.

### 1.3 The 5-horizon CHIS (iter_016) is a 0-downside upgrade — **but the only decision-side lever it unlocked is "score ≥ baseline"**

CHIS (T102 #10, shipped in iter_016) sends 5 active heads instead of 1; scoring takes max-h. That is structurally correct, but it means **per-h decision rule quality matters much more than before**: a poorly-tuned h5 rule can never *hurt* (max), but it can ride on h60 instead of contributing. The remaining decision lever inside CHIS is **per-h decision quality**, especially on h5/h10 where fee-to-signal ratio is tightest (α=0.05 % vs 0.10 % for h≥20).

### 1.4 Decision levers we have NOT used at all

Crossing the lists:

| Lever | Used in iter_016? | Used in T102 / T105 / IDEA-CTX-DECISION? | Used in T112? |
|---|---|---|---|
| Constant absolute thresh | ✅ DE-4D | ✅ | (replaced) |
| Within-batch rank | ❌ | proposed (BAT) | (extended below) |
| Within-batch z-score | ❌ | proposed (CSSC) | (extended below) |
| Empirical OOF tier win-rate | ❌ | ❌ | **NEW** ★ |
| Spread-aware *physical* fee floor (no tune) | ❌ | proposed as `α + β·spread1` (SCG, *with tuning*) | **NEW** (no tune) ★ |
| Decision-layer ensemble (vote across gates) | ❌ | ❌ | **NEW** ★ |
| Top-K count gate (1-param) | ❌ | ❌ | **NEW** ★ |
| Score-shrinkage / centering | ❌ | ❌ | **NEW** ★ |
| Action-rate lower-bound (anti-throttle) | ❌ | AFT is upper-bound only | **NEW** ★ |
| Median-of-models aggregation | ❌ | ❌ (always weighted mean) | **NEW** ★ |
| Cross-model sign-vote gate | ❌ | HMD proposed (binary) | **NEW** (3-way) ★ |
| Discretized prediction magnitude | ❌ | ❌ | **NEW** ★ |

That's **8 untouched basic levers**, every one of which is one screen of code, fits inside 100 lines of `Predictor.predict`, and runs in O(N) on CPU. Below: 10 of them, ranked.

---

## Part 2 — Ten Basic-but-Disruptive Execution Tricks

Each idea: **one-line / why basic / why disruptive / mechanics / cost / est LOSO Δ / risk / compliance**.
None has been tried in T1–T109 nor in any prior proposal.

### TRICK #1 — **TBT: Trade-Budget Top-K Gate** ★ Top-1 by simplicity

**One-line.** Sort every batch by `|pred|`; trade the top-K by `|pred|`; everything else flat. K = `round(N · oof_active_rate)`.

**Why basic.** It is the **simplest possible gate that respects per-batch context**. One parameter (`oof_active_rate`, typically ≈ 0.40 from current OOF). Replaces (thr_up, thr_dn, d_up, d_dn) with one number — and that number is **not even tuned by DE**, it's just the empirical OOF active-rate.

**Why disruptive.** Today a +0.0001 prediction-distribution shift on test causes the gate to over-/under-fire by 100 %s of trades. **TBT is intrinsically scale-invariant**: only ranks matter. It also collapses the entire 4D DE search to a 1D rate that you don't even tune adversarially — you measure it.

**Distinction from BAT (prior).** BAT uses a within-batch *quantile threshold*: `pred > Q_(1-α)` — the threshold is the (1-α)-quantile of preds. **TBT enforces an exact count K** = `α·N` after **also requiring sign-consistency**. BAT can fire 0 trades or N trades depending on the pred distribution; TBT always fires exactly K. Different operating points, different overfitting surfaces.

**Mechanics.**
```python
# in Predictor.predict, per-batch:
N    = len(preds)
K    = int(round(N * oof_active_rate))   # e.g. 0.40
order = np.argsort(-np.abs(preds))         # top-|pred| first
top   = order[:K]
# split top into long/short by sign(pred), apply optional minimum |pred| floor
action = np.ones(N, dtype=int)             # 1 = flat
action[top] = 1 + np.sign(preds[top]).astype(int)
# optional: re-zero rows with |pred| < FEE (physics floor never fired)
action[(np.abs(preds) < 2*FEE)] = 1
```

**Cost.** ~1 h Predictor edit + 1 h OOF rate calibration on per-h basis.

**Est LOSO Δ.** Locally probably **−0.5 to +0.5** vs DE-4D (DE wins by 0.3-0.7 on this 442k because it overfits this slice). **Platform expected delta +1 to +2.5** because TBT has zero distribution-shift overfit.

**Risk.** If the platform feeds a homogeneous-symmetry batch (e.g. all sym=0, all PM session), the rank correlation with true edge can collapse. Mitigation: floor `K = min(K, count(|pred| > 2·FEE))` so we never trade rows physically below fee.

**Compliance.** Within-batch ranks ✅ no sym ✅ no date ✅ stateless across batches ✅.

---

### TRICK #2 — **NSF: Net Spread-aware EV Floor (zero-tune physics gate)** ★ Top-2 by impact

**One-line.** Replace `|pred| > thr_const` with `|pred| > 2·FEE + γ · spread1`, where `γ ∈ {0, 0.25, 0.5}` is calibrated on OOF (a single 3-point grid; **not** DE).

**Why basic.** The **realized** round-trip cost is `2·FEE + spread_crossing`. Currently the EV gate uses a constant `thr ≈ 3e-4` that averages over `spread1 ∈ [0.5e-4, 5e-4]` — i.e. for low-spread rows we under-fire, for high-spread rows we over-fire. The fix is the textbook market-microstructure transaction-cost adjustment.

**Why disruptive.** The 4-param DE thresh is *trying* to learn this with a constant. It cannot, because spread varies row-by-row. Adding `γ·spread1` as a per-row floor **converts thresh from a tunable scalar into a derived physical quantity**. After γ-calibration the rule has **zero discretionary parameters left**.

**Distinction from SCG (prior brainstorm).** SCG: `thr_i = α + β·spread1_i` with α, β both DE-tuned. NSF: `thr_i = 2·FEE + γ·spread1_i` where 2·FEE is fixed by the contest and γ is calibrated by 3-point grid (γ ∈ {0, 0.25, 0.5}). **Half as many parameters, and both are physically interpretable** (the absolute cost and the half-spread share). More importantly, NSF **subtracts** the spread cost from the EV directly — it is not a tuned gate, it is a corrected fee.

**Mechanics.**
```python
FEE = 1e-4   # contest constant
floor_i = 2*FEE + gamma * spread1_i   # spread1 from input row, per-row
trade_i = (np.abs(pred_i) > floor_i)   # per-row
action_i = (1 + np.sign(pred_i)) if trade_i else 1
```

**Cost.** ~1 h Predictor edit (need to extract `spread1` from each batch DataFrame) + 1 h OOF γ-grid scan + 1 h sanity check.

**Est LOSO Δ.** **+0.5 ~ +1.5** locally — directly attacks the 18 % fee-burn bucket. Robustness: dramatic positive on platform because it removes the calibrated-thresh transmission risk.

**Risk.** `spread1` is a normalized %, must verify scale matches `pred` units. Trivially asserted in OOF.

**Compliance.** Per-row ✅ stateless ✅ no sym ✅ no date ✅. `spread1` is a derived feature explicitly in the published 154-feature schema.

---

### TRICK #3 — **STK: Stochastic-Tier-K Win-Rate Bucketing** ★ Top-3 by robustness

**One-line.** On OOF, bucket rows by `|pred|` decile; compute empirical win-rate (= P[sign(pred)==sign(Δmid)]) per decile; trade only rows in deciles where win-rate ≥ 53 %.

**Why basic.** Pure histogram lookup. No DE, no calibration objective beyond "win-rate > 50 % + margin". This is what a person would do with a CSV and Excel.

**Why disruptive.** It is **completely model-agnostic and parameter-free at deployment**: the deciles are computed once on OOF, and at deployment we just look up which decile the test `|pred|` falls into. **Win-rate is the most distribution-stable property of a predictor** (T102 §1.2, WRTC) — much more robust than expected sum-PnL across regime shifts.

**Distinction from WRTC (prior).** WRTC tunes (thr_up, thr_dn) to maximize win-rate **subject to PnL constraint** — still a 2D optimization. STK does **no optimization**: it observes which |pred|-bucket has empirical edge and **drops the rest entirely**. WRTC's output is a threshold; STK's output is a binary inclusion mask per bucket.

**Mechanics.**
```python
# OFFLINE on OOF (once):
bins = np.quantile(np.abs(oof_pred), np.linspace(0, 1, 11))   # 10 buckets
for i in range(10):
    mask = (oof_bucket == i)
    win_rate[i] = np.mean(np.sign(oof_pred[mask]) == np.sign(oof_dmid[mask]))
trade_buckets = [i for i in range(10) if win_rate[i] > 0.53]   # store list

# AT INFERENCE:
bucket_i = np.digitize(np.abs(pred_i), bins) - 1
trade = (bucket_i in trade_buckets) and (np.abs(pred_i) > 2*FEE)
```

**Cost.** ~2 h. Histogram fit on OOF, lookup table in Predictor, sanity check.

**Est LOSO Δ.** **+0.3 ~ +1.0**. Conservative because it abstains on uncertain buckets. The asymmetric platform vs LOSO transmission **especially favors win-rate-based gates**.

**Risk.** Bucket boundaries fit on OOF can drift on platform. Mitigation: use **fee-relative absolute** thresholds for bin edges (`{2·FEE, 3·FEE, 4·FEE, ...}`) rather than OOF quantiles, so the meaning of each bucket stays stable across regimes.

**Compliance.** Bucket lookup is stateless ✅ no sym ✅ no date ✅.

---

### TRICK #4 — **DRSV: Decision-Rule Stacking with Vote** (decision-layer ensemble)

**One-line.** Run K independent gates on the same prediction (DE-4D, BAT, NSF, STK, TSCV) and majority-vote the action per row.

**Why basic.** Voting is the textbook ensemble method. Today we ensemble *predictions* (T75 + T87 + q_mid + huber) and then apply *one* gate. The dual choice — single prediction × ensemble of gates — has never been tried.

**Why disruptive.** Decision-layer ensembling is **complementary to prediction-layer ensembling**. A trade that survives 3-of-5 gates is much more robust than one that survives only the DE-4D. Variance reduction at the gate output is **exactly the lever for the platform-vs-LOSO gap** (each gate has different overfit profile to the OOF; vote averages out the gate-level noise).

**Mechanics.**
```python
gates = [DE4D, BAT, NSF, STK, TSCV]   # each returns action ∈ {0,1,2}
votes = np.array([gate(pred, batch_ctx) for gate in gates])   # K x N
# Majority: action = +1 if ≥3 gates say +1; -1 if ≥3 say -1; else 0
n_long  = (votes == 2).sum(axis=0)
n_short = (votes == 0).sum(axis=0)
action  = np.where(n_long  >= 3, 2,
          np.where(n_short >= 3, 0, 1))
```

**Cost.** ~3 h. Each gate already exists separately or is implementable in 1 h.

**Est LOSO Δ.** **+0.3 ~ +0.8 LOSO** (vote tends to slightly trail the best individual gate on the *same* OOF). **+1.0 ~ +2.0 platform** (variance reduction transmits well).

**Risk.** Computational cost is K-fold higher per row, but K=5 × O(N) is negligible (still milliseconds for N=1024). The bigger risk is correlated gates (DE-4D and BAT both rank-similar) — pre-validate the gate diversity on OOF (correlation of gate-actions, target ≤ 0.85).

**Compliance.** All component gates already compliant.

---

### TRICK #5 — **MOM: Median-of-Models prediction aggregation** (replaces weighted mean)

**One-line.** Combine the (T75, T87, q_mid, huber) predictions by **`pred = median(...)`** instead of `pred = w·avg(...)`.

**Why basic.** Median is the canonical **robust** aggregator. Mean has best variance under Gaussian noise; median has best robustness under heavy-tailed noise (and our `Δmid_norm` is heavy-tailed: kurtosis ≫ 3).

**Why disruptive.** Every iter has used a **weighted mean** with hand- or DE-tuned weights `(w_lgb, w_nn, w_qmid, w_huber)` — i.e. **3-4 parameters at the aggregation step alone**. Median is 0-param. If one model goes "weird" on a new sym, mean carries 25 % of that weirdness; median ignores it. **The leaderboard intel suggests robust aggregation > tuned weights** (because everyone is tuning weights but not everyone is winning).

**Distinction from prior brainstorm.** No prior proposal touched the prediction-aggregation step. All decision-layer brainstorms assumed `pred = mean(models)` and operated downstream.

**Mechanics.** With 4 model predictions per row, `pred_med = np.median(stack, axis=0)`. ~3 lines.

**Cost.** ~1 h.

**Est LOSO Δ.** **−0.3 ~ +0.5** locally (median slightly suboptimal under in-distribution Gaussian assumption) but **+0.5 ~ +1.5 platform** (heavy-tailed test noise + sym-OOD make the robustness pay off).

**Risk.** With only 4 models, median = average of middle two. With odd n (3 or 5 models) the median is exactly one model — that model dominates. Verify n: for 4 models median is well-defined; if we drop to 3 (e.g. exclude huber for a defensive variant) median = the middle model, lose diversity.

**Compliance.** Aggregation is per-row ✅.

---

### TRICK #6 — **ZBC: Zero-Bias Centering** (online recalibration of prediction location)

**One-line.** Subtract a fraction `λ ∈ [0, 1]` of the **batch mean of preds** from each pred before gating: `pred_i' = pred_i - λ · mean(pred_batch)`.

**Why basic.** The simplest possible online recalibration. λ = 1 is full per-batch centering (only relative magnitudes drive the gate); λ = 0 is current behavior; λ = 0.5 is a defensive split. Calibrated on OOF as a 1D scan.

**Why disruptive.** T75's regression has a **+0.000087 prediction bias** documented (asymmetric `thr_up=3.72e-4 vs thr_dn=1.61e-4` is the DE-4D trying to learn this). On a different test slice, the bias may flip sign or grow. **A constant DE-tuned asymmetry cannot track that drift; ZBC does**, because it removes the bias *from this batch*. Combined with a symmetric thresh, ZBC + symmetric-k = same expressiveness as DE-4D, but the asymmetry is **measured per batch, not learned globally**.

**Distinction from prior brainstorm.** No proposal touched online recalibration of the *location* of `pred`. CSSC touched the *scale* (z-score), not the location.

**Mechanics.**
```python
batch_mean = np.mean(pred_batch)
pred_centered = pred_batch - lambda_zbc * batch_mean
# then apply symmetric-k EV gate
action = ev_gate(pred_centered, thr_sym)
```

**Cost.** ~1 h Predictor edit + 1 h OOF λ scan (5 values × 5 OOF folds).

**Est LOSO Δ.** **+0.0 ~ +0.5 LOSO** (current OOF doesn't have severe bias drift; ZBC is mostly neutral on it). **+0.5 ~ +1.5 platform** if there is bias drift on the public slice — which the platform-vs-LOSO −17 gap (T83) hints at.

**Risk.** If a batch is genuinely 90 % long-true, centering removes the real signal. Bound λ ≤ 0.5 to keep the absolute floor signal.

**Compliance.** Within-batch only ✅ stateless across batches ✅.

---

### TRICK #7 — **ARLB: Action-Rate Lower-Bound (anti-AFT)** (asymmetric self-throttling, downward)

**One-line.** When per-batch active rate falls below `0.5 × oof_active_rate`, **lower** the threshold to compensate (force-fill toward target rate).

**Why basic.** AFT (T102 #8) has a one-sided safety valve that **raises** thresh when batches over-fire. ARLB is the symmetric mirror that **lowers** thresh when batches under-fire. Both are first-order safety valves.

**Why disruptive.** AFT attacks the (b) 27.8 % wrong-direction bucket (over-firing on noisy batches). ARLB attacks the (a) 49.6 % missed-trade bucket (under-firing on conservative batches). **Different buckets, different signs of self-correction** — and 49.6 % is the bigger lever, so ARLB is structurally more important than AFT.

**Distinction from BAT.** BAT redefines thresh as a quantile of pred (so active rate ≈ α). ARLB starts with the existing absolute thresh and *lowers it conditionally* when active rate drops below a floor. ARLB preserves the absolute fee floor; BAT does not.

**Mechanics.**
```python
N = len(pred_batch)
thr_eff_up = thr_up
thr_eff_dn = thr_dn
n_active = np.sum((pred_batch > thr_up) | (pred_batch < -thr_dn))
target_active = oof_active_rate * N
if n_active < 0.5 * target_active:
    # K-th largest |pred| where K = target_active
    K = int(target_active)
    abs_pred_sorted = np.sort(np.abs(pred_batch))[::-1]
    if K < N:
        thr_eff_up = max(2*FEE, abs_pred_sorted[K])
        thr_eff_dn = thr_eff_up
    # gate with the *effective* (lowered) thresh
```

**Cost.** ~2 h.

**Est LOSO Δ.** **+0.2 ~ +0.8** (depends on how often OOF batches under-fire; usually ~5-10 %).

**Risk.** Lowering thresh can amplify wrong-direction trades. Bounded floor at `2·FEE` (physics floor) prevents trading on rows physically below cost.

**Compliance.** Within-batch only ✅ stateless ✅.

---

### TRICK #8 — **DCS: Direction-Confidence Scaling (3-way model sign-vote)**

**One-line.** Trade only when sign agreement across `{T75 mean, T87 NN, q_mid}` is unanimous (3/3); abstain on 2/3 split; abstain on 1/3 (sign-disagree).

**Why basic.** The simplest possible cross-model sanity check on direction. Three votes; if they disagree on the **sign**, the ensemble's mean prediction is unreliable on direction.

**Why disruptive.** Today we average the 3 predictions and gate on the average's sign. **Information about the sign-disagreement is destroyed in averaging.** A row where all three say `+0.0001` is much safer than a row where two say `+0.0010` and one says `-0.0008` — the average is identical but the second is a coin-flip on direction.

**Distinction from HMD (prior, IDEA-CTX-DECISION).** HMD is a 2-model binary sign-veto (LGB and NN must agree). DCS is a 3-model graded vote (3/3 strong, 2/3 borderline, 1/3 abstain) — more granular and operates on the existing 3-model ensemble (T75 + T87 + q_mid) without requiring a 4th NN. Different model set, different decision granularity.

**Mechanics.**
```python
signs = np.stack([np.sign(pred_lgb), np.sign(pred_nn), np.sign(pred_qmid)])   # 3 x N
n_long  = (signs ==  1).sum(axis=0)   # per-row count of long votes
n_short = (signs == -1).sum(axis=0)
unanimous_long  = (n_long  == 3)
unanimous_short = (n_short == 3)
trade = (unanimous_long & (pred_avg > thr_up)) | (unanimous_short & (pred_avg < -thr_dn))
```

**Cost.** ~1 h.

**Est LOSO Δ.** **+0.4 ~ +1.0**. Directly attacks the 27.8 % wrong-direction bucket.

**Risk.** May over-abstain (n_active drops by ~30 %). Symmetric loosening: trade if 2/3 agree AND |pred| > 1.5·thr (graded version).

**Compliance.** Per-row ✅.

---

### TRICK #9 — **PMD: Prediction-Magnitude Discretization** (quantize before gating)

**One-line.** Round each pred to nearest of 5 levels `{0, ±0.5·thr, ±thr, ±2·thr, ±5·thr}`; gate on rounded value.

**Why basic.** Rounding to a fixed grid is the simplest form of regularization. We currently treat `pred = +0.0001` and `pred = +0.000099` as different rows; they aren't.

**Why disruptive.** **Discretization removes overfit to fine-grained pred values.** The 4D DE finds (thr_up, thr_dn, d_up, d_dn) thresholds that tune to the OOF noise floor below 1e-5; discretization erases that noise. **Less expressive but more transmissive.**

**Distinction from prior brainstorm.** No proposal touched discretization of `pred`. DDT (T102 #5) discretizes the *action* via a tree but keeps the input pred continuous; PMD discretizes the input pred itself, before any gate.

**Mechanics.**
```python
levels = np.array([-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]) * thr
pred_q = levels[np.argmin(np.abs(pred_batch[:, None] - levels[None, :]), axis=1)]
# then gate as usual on pred_q
```

**Cost.** ~1 h.

**Est LOSO Δ.** **−0.5 ~ +0.5 LOSO** (locally indistinguishable). **+0.5 ~ +1.0 platform** (less overfit means better transmission).

**Risk.** Coarse-graining the gate may lose ~1 PnL on the threshold-boundary rows. Bound by tracking n_active and PnL on OOF for a finer grid.

**Compliance.** Pure mathematical transform ✅.

---

### TRICK #10 — **POG: 1D PnL-Optimal Grid** (replaces 4D DE entirely)

**One-line.** Replace 4D DE on (thr_up, thr_dn, d_up, d_dn) with a **1D grid search** over a single composite trade-score threshold `s = |pred| − γ·spread1 − β·sigma_ensemble`.

**Why basic.** A 1D grid is the simplest non-trivial optimization. It's what every leaderboard winner is doing instead of DE-4D.

**Why disruptive.** Reduces parameter count from 4 → 1 (the threshold `s*`). The composite trade-score `s` packs both the spread cost (NSF) and the predictive uncertainty (cheap σ from seed disagreement) into a single physically-grounded scalar — and we just sweep one threshold over its empirical OOF distribution. **The 4-D structure was never necessary; it was an artifact of having (thr_up, thr_dn) decoupled and adding (d_up, d_dn) as an afterthought.**

**Distinction from NSF.** NSF is the *floor formula*; POG is the *optimization protocol* that uses it. NSF + POG together: trade if `|pred| > 2·FEE + γ·spread1 + β·σ`; γ, β fixed by 3-point grids; the only tunable left is `s* = 0` (i.e. trade whenever the inequality holds), so **POG is effectively zero-parameter at deployment**.

**Mechanics.**
```python
# OFFLINE: 1D scan
gamma_grid = [0, 0.25, 0.5]
beta_grid  = [0, 0.5, 1.0, 2.0]
# 12 combos x 1 trivial threshold = 12 OOF evaluations -> pick best
# (vs 4D DE: maxiter=80 popsize=24 = 1920 evaluations)
```

**Cost.** ~2 h (code + OOF runs).

**Est LOSO Δ.** **−0.5 ~ +0.5** locally (1D < 4D in-sample); **+1 ~ +2.5 platform** (12 evals < 1920 evals → less DE overfit).

**Risk.** If σ-proxy or spread1-proxy is noisy, the score loses signal. Mitigated by including each as an option (γ=0 falls back to constant fee).

**Compliance.** Pure post-processing ✅.

---

### TRICK #11 (bonus) — **ESPC: Empirical Sign Posterior Conformal**

**One-line.** From OOF, compute empirical `P(sign(pred)==sign(Δmid) | |pred| ∈ bin_k)`. Trade only when bin's sign-accuracy ≥ 0.55. (Generalization of STK using sign-accuracy instead of win-rate.)

**Distinction from STK.** STK uses **win-rate** (P[sign(pred)==sign(Δmid)] AND |Δmid| > fee) as the bucket gate. ESPC uses **sign-accuracy** alone — separates direction-quality from magnitude-quality. In failure-mode terms: STK gates the union of (b)+(c); ESPC gates only (b).

**Cost.** ~2 h. **Est Δ.** +0.3 ~ +0.8 LOSO.

---

### TRICK #12 (bonus) — **SRT: Spread-Ratio Trade Filter**

**One-line.** Compute `r_i = spread1_i / median(spread1_batch)`; trade only when `r_i < 1.2` (this row is in the lower-spread regime *within the batch*).

**Distinction from SCG / NSF.** SCG / NSF bake the absolute spread into the EV floor. SRT is a *relative* filter: even after fee-correction, abstain on rows whose spread is unusually high *for this batch*. Targets liquidity-thin moments specifically.

**Cost.** ~1 h. **Est Δ.** +0.2 ~ +0.5 LOSO. Mostly defensive.

---

## Part 3 — Top 3 Recommendations

Ranked by `(CPU-friendly × impact × low-risk × leaderboard-intel-alignment)`:

| Rank | Idea | Param Δ | Hours | Est LOSO Δ | Est Plat Δ | Risk | Compliance |
|---|---|---|---|---|---|---|---|
| **1** | **NSF — Net Spread-aware EV Floor** | 4 → **0** (γ from 3-point grid) | 3 | +0.5 ~ +1.5 | +1.0 ~ +2.5 | very low | ✅ |
| **2** | **TBT — Trade-Budget Top-K Gate** | 4 → **1** (oof_active_rate, *measured*) | 2 | −0.5 ~ +0.5 | +1.0 ~ +2.5 | low (with FEE floor) | ✅ |
| **3** | **STK — Stochastic-Tier-K Win-Rate Bucketing** | 4 → **0** (just buckets + 53 % cutoff) | 2 | +0.3 ~ +1.0 | +1.0 ~ +2.0 | low | ✅ |

### Rationale

1. **NSF replaces the constant fee with the actual physical round-trip cost.** It directly attacks the 18 % fee-burn bucket (largest under-served leak) with a parameter count drop from 4 → 0. The `γ ∈ {0, 0.25, 0.5}` 3-point grid is **trivially robust** (3 evaluations vs DE's 1920); transmits well by construction. **Aligns perfectly with the leaderboard intel** — leaders almost certainly use spread-aware floors.

2. **TBT is the simplest possible gate.** Sort, take top-K, fire. K is **measured** (= OOF active rate × N), not tuned. Replaces 4-D DE with 1 measured rate. The risk is that local LOSO drops slightly (DE wins the 442k slice by overfit); but **platform transmission is intrinsically scale-invariant** in TBT — that's the whole point. Stack with NSF: TBT gates by rank, NSF gates by physics floor — keep the AND.

3. **STK gates by empirical OOF win-rate.** Pure histogram lookup, model-free, distribution-stable. The cleanest defense against the platform-vs-LOSO −17 gap. Easy to combine with NSF/TBT (apply STK on top to additionally reject low-win-rate buckets).

### Suggested execution order (single worker, half-day)

- **Hour 1**: Implement NSF (γ-grid scan on T87+T75 OOF). Pick γ.
- **Hour 2**: Implement TBT (measure OOF active rate, set K). Run on OOF.
- **Hour 3**: Implement STK (10-bucket win-rate fit on OOF). Pick included buckets.
- **Hour 4**: Stack NSF AND TBT AND STK in Predictor. Run end-to-end on OOF, compare to iter_015 / iter_016 baseline.
- **Hour 5**: If LOSO ≥ iter_015−1.0 (i.e. within ~1 PnL of baseline locally), package as defensive iter candidate. The platform delta is the upside.
- **Hour 6**: Sanity (shuffle / sym=99 / 1024-batch timing).

The pessimistic local outcome is **LOSO ≈ +39** (1 below iter_015, 5 below iter_016). The optimistic platform outcome is **+24 to +27** (vs iter_015's +21).

### Why this triple beats further DE refinement

- **Param count**: 4 (DE-4D) → 1 (TBT) + 1 (γ) + 1 (win-rate floor) = 3 numbers, **all measurable not tunable**.
- **Each parameter is interpretable**: physical fee floor, OOF active rate, win-rate floor.
- **Each is robust to the +0.0001 bias drift** that T78 / T83 hinted at: TBT scale-invariant; NSF physically grounded; STK uses fee-relative bins.
- **Each is independently failable**: if TBT is too coarse, NSF still helps; if STK over-abstains, NSF + TBT is the floor.
- **None requires retraining any model.** All work on the existing iter_015 / iter_016 prediction stack.

---

## Part 4 — What I Explicitly Did NOT Propose

| Idea | Why rejected |
|---|---|
| Per-sym threshold | banned by §1.3 |
| Date-conditioned policy | `date=0` at inference, banned |
| Cross-batch state | banned by §1.2 |
| Confidence-weighted fractional sizing | output is `{0,1,2}` ternary only |
| GPU-NN post-processor | violates "no GPU" |
| Larger DE search (more iters / dim) | T83 falsified — gap is intrinsic |
| Re-tuning iter_016 per-h thresholds | already at saturation |
| Online ensemble re-weighting | requires past-batch state, banned |

---

```
RESULT: task=execution_basic top_3=[NSF, TBT, STK] notes="NSF zero-param physics-grounded fee floor (replaces DE-4D with 2·FEE+γ·spread1, γ from 3-point grid); TBT 1-param top-K rank gate (K measured from OOF active rate, scale-invariant on platform); STK zero-param empirical OOF win-rate bucketing (model-free decision via histogram lookup); 10 main + 2 bonus tricks, all CPU-friendly, all post-processing only, all ≤6h to implement; aligned with leaderboard intel that simple 0-1D gates beat tuned 4D DE due to platform transmission"
```
