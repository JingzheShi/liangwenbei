# W-OOD: Distribution Shift Defense for HFT LOB Midprice Prediction
**Author:** Blinded ML/quant researcher (Opus 4.7)
**Date:** 2026-05-08
**Working assumption:** local LOSO PnL is *decorrelated* (or anti-correlated) with platform PnL on date 120+. Any proposal that uses LOSO PnL as its single optimization target is dead on arrival.

---

## 0. Problem reframing

Three observations frame everything below:

1. **"LOSO disagrees with platform" is a falsification result.** It tells us the validator no longer measures the quantity we care about (test-distribution PnL). Adding model capacity, more features, or better in-distribution fit cannot fix this — those just sharpen LOSO further. The bottleneck is **calibration of the validator and the inductive bias**, not predictor expressivity.
2. **The shift dimension we *cannot* see is `date`** — the only info we have is "test ≥ 120 in the unobserved coordinate". sym is partially observed (5 known levels, possibly more); features are fully observed. Therefore any defense must use **feature- and sym-level proxies** to simulate the date shift.
3. **The cost of a wrong prediction is asymmetric and finite.** Action 1 (hold) has 0 PnL and 0 fee. So under shift, *abstention is a free lunch* if we can detect we're out of distribution. The PnL objective itself rewards a "predict only when confident" policy more than it rewards squeezing 1% more accuracy in the bulk.

These observations point to three distinct lever classes, ordered by my confidence in transfer:

| Lever class                            | Mechanism                                                       | Why it should transfer                                                              |
|----------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **A. Reduce shift in feature space**   | Drop / transform non-stationary features; only learn from invariants | If `p_test(x) ≈ p_train(x)` for all `x` we use, covariate shift vanishes by construction |
| **B. Recalibrate the action policy**   | Use conformal / abstention so wrong-but-confident predictions become holds | Conformal coverage degrades *gracefully* under mild shift (quantitative bound)       |
| **C. Force worst-case generalization** | DRO / IRM across sub-environments inside 0–119                   | If the within-train regime variance covers the ≥120 shift, worst-case = test-case  |

The *top-3* picks below combine A + B (which are cheap, low-risk, and have textbook transfer guarantees), and add a single C-lever (which is higher variance but addresses the residual gap A+B leaves on the table).

---

## TOP-3 PICKS — full design

### 🥇 Pick 1 — **Stationary Feature Reformulation + KS-Pruning** (Confidence: **H**)

#### Mechanism
Most LOB features fall into two categories:
- **Non-stationary:** absolute price levels, raw cumulative volumes, time-since-event measured in absolute ticks. Their *marginals* drift over date because volatility regime, average trading volume, and tick-size behaviors all drift.
- **Stationary (or ~ stationary):** **ratios, normalized imbalances, signed shapes, log-returns over rolling windows, dimensionless statistics**. Their marginal is approximately invariant under volume- or volatility-rescaling of the orderbook.

The defense: enforce that every feature fed to the model passes a stationarity test across the within-train date axis. Concretely:

1. Bucket train data into Q=4 date-quartiles within 0–119: B0 = 0–29, B1 = 30–59, B2 = 60–89, B3 = 90–119.
2. For each candidate feature $f$, compute pairwise **two-sample KS divergence** $D_{KS}(B_0 || B_3)$ and $\max_{i<j} D_{KS}(B_i || B_j)$.
3. **Drop** any feature with $D_{KS} > \tau$ (start with $\tau = 0.05$, tune by grid).
4. **Replace** dropped features with their stationary cousin (e.g., raw spread → spread / mid-price; raw bid_volume → bid_volume / (bid_volume + ask_volume); cumulative volume → log-volume z-scored against trailing 100-tick window).
5. Re-train the model on the reformulated feature set.

#### Why it transfers
This is the only lever with a *deterministic* transfer guarantee: if every feature `f_i` in the model has $\|p_{train}(f_i) - p_{test}(f_i)\|_{TV} \le \epsilon_i$, then for a Lipschitz predictor, the test risk is bounded by train risk + $L \cdot \sum \epsilon_i$. We cannot measure $\epsilon_i$ on test, but stationarity *within* train (B0 vs B3) is a strong proxy: if a feature drifts within 120 days of training data, it will drift again into day 120+. Slow Feature Analysis (Wiskott '02) and the practical Avellaneda/Lee literature on quant signals agrees: ratios > absolutes for cross-regime stability.

#### Implementation sketch
- File: a new `features_stationary.py` mirroring `features.py` but with the transforms applied.
- One-time KS audit script `tools/feature_stationarity_audit.py` outputs a CSV with per-feature `ks_max`, `ks_endpoints (B0||B3)`, recommendation (keep/drop/transform).
- Compute cost: KS audit on ~1.5M × 226 features ≈ 3 min CPU; retrain ≈ same as current LGB.

#### LOSO-bypass check (the key innovation)
**Don't use LOSO PnL to pick this.** Instead use:
- **`ks_max_over_features` of the final feature set** (lower = better, target ≤ 0.05 across all kept features).
- **"Late-bucket holdout PnL"**: train only on B0+B1+B2 (date 0–89), eval on B3 (date 90–119). The PnL on B3 is a *temporal* OOD test inside the train set. Models that pass this are predicted to transfer to date 120+; models that pass LOSO but fail B3 are overfit to LOSO.
- Track **B3 PnL / LOSO PnL ratio** — if ratio < 0.7, the LOSO signal is not trustworthy. This ratio is itself the new score function.

#### Risk
- May lose informative signal — some non-stationary features (e.g., regime indicator) genuinely help in-distribution. Mitigate by *always* keeping a "stationarity-controlled" version alongside the raw, never strictly worse than raw.
- KS test on ~2M rows can find spurious differences that don't matter — pair with **conditional MMD on `(x, y)`** as a sanity check.

#### Confidence: **H**
Universal trick in quant ML. Avellaneda & Stoikov use this routinely; the Kearns/Cartea et al. HFT papers all normalize before modeling. Has independently worked in 2023–2025 quant Kaggle competitions (Optiver, Jane Street).

---

### 🥈 Pick 2 — **Robust Cross-Sym Conformal Abstention** (Confidence: **H**)

#### Mechanism
We currently force a 3-way action on every test row. Instead: **predict only when the model is confident enough that the expected PnL exceeds the 0.0001 fee**, otherwise emit action 1 (hold = 0 PnL).

Build the abstention threshold via **conformal prediction calibrated on a sym-leave-one-out scheme**, then take the **worst-case threshold across the 5 folds**, not the average:

1. For each held-out sym $s \in \{0,1,2,3,4\}$, train base model on the remaining 4 syms. On the held-out sym, compute **per-class non-conformity scores** (e.g., $1 - p_{model}(y=k|x)$ for the predicted class $k$).
2. For target coverage $1 - \alpha$, the per-fold conformal threshold is $\hat q_s(\alpha) = $ empirical-$(1-\alpha)$ quantile of non-conformity scores.
3. **Use $\hat q^{robust}(\alpha) = \max_s \hat q_s(\alpha)$** — the most conservative threshold across folds.
4. At inference: emit action $k$ only if $1 - p_{model}(y=k|x) \le \hat q^{robust}(\alpha)$. Otherwise emit hold.
5. Sweep $\alpha$ on **B3 (date 90–119)** PnL, not LOSO PnL, to set final operating point.

The conformal coverage guarantee under exchangeability says: if the test distribution is exchangeable with the calibration distribution, then the abstention rate matches what we set. Under *shift*, the guarantee degrades, but the **adaptive conformal** result (Tibshirani et al. 2019, Gibbs & Candes 2021) says the degradation is bounded by the total-variation distance between calib and test — and worst-case across syms is the tightest bound we can compute without seeing test.

#### Why it transfers
Three reasons:
1. **Coverage transfers under bounded shift** (theorem). We get a quantitative degradation bound from conformal under shift, which simple ensembling does not.
2. **Abstention is a *free* defense.** Even if the model is wrong post-shift, abstaining costs 0. Wrong + confident on a real position costs `|pnl| + 2 × 0.0001`. So we are only losing *opportunity*, not capital — perfectly hedged.
3. **Worst-case across folds** mimics the unobserved date-shift. We do not know how date 120+ relates to date 0–119, but we *do* know how unseen syms relate to seen syms (LOSO). If our threshold survives the worst sym fold, it has a nontrivial chance of surviving the worst date.

#### Implementation sketch
- File: wrap existing predictor in `ConformalAbstainPredictor` (post-hoc, no retraining).
- Compute: precompute `q_robust(α)` table on validation set (3 min). At inference, one comparison per row — negligible.

#### LOSO-bypass check
- Plot **abstention rate vs realized PnL on B3**. The "right" $\alpha$ is the one that maximizes B3 PnL, *not* full-LOSO PnL.
- **Coverage-stability metric**: difference between empirical coverage on B0 vs B3. If $|cov(B_0) - cov(B_3)| < 0.02$, conformal is well-calibrated under within-train shift — predict it will hold under date 120+ shift too.
- Sanity: abstention rate should rise modestly on B3 vs B0; if it spikes (>2x), the calib set is unreliable, lower α.

#### Risk
- If the sym-shift is qualitatively different from the date-shift, worst-case sym threshold may be too lax. Mitigate: also compute B3-vs-rest temporal threshold and take max of (sym-worst, time-worst).
- Abstaining too often erodes profit. Pick $\alpha$ at the knee of the abstention-PnL curve, not the maximum.
- If the existing pipeline already uses conformal: this proposal upgrades it from *single-quantile* to *worst-fold quantile* and adds the temporal B3 calibration.

#### Confidence: **H**
Conformal under distribution shift is a 2019–2024 hot area with rigorous results. Specifically Barber/Candes/Ramdas/Tibshirani "The limits of distribution-free conditional predictive inference" (2020) and Gibbs/Candes "Adaptive conformal" (2021) give us the theory. Applied successfully in Jane Street competition winners (post-hoc calibration was key).

---

### 🥉 Pick 3 — **Group DRO across Sym × Date-Bucket Environments** (Confidence: **M-H**)

#### Mechanism
Replace ERM (minimize average loss) with **GroupDRO** (Sagawa et al. 2020): minimize **worst-case loss** over a partition of train data into $G$ environments. Define environments as **sym × date-quartile** = $5 \times 4 = 20$ groups.

Formally, if $\mathcal{L}_g(\theta)$ is the loss on group $g$:

$$\theta^* = \arg\min_\theta \max_g \mathcal{L}_g(\theta)$$

Implementation via the standard **soft-max group reweighting** with step size $\eta$: each gradient step, increase weight on the worst group, decrease on others.

Why "sym × date-quartile" rather than just "sym" or just "date"? Because:
- **sym alone** has only 5 environments → DRO will overfit to the worst one.
- **date alone** has only 4 buckets → likewise.
- **sym × date-bucket** has 20 environments, enough variance for DRO to find a genuine invariant predictor.

This implicitly assumes that the **structure** of date 120+ shift will look like *some convex combination* of "sym-shift × date-shift" we have within 0–119. If the platform shift is a totally novel direction, DRO will not help — but it will not hurt much either (it converges to a model with comparable in-distribution loss but better tail).

#### Why it transfers
GroupDRO has a generalization bound (Sagawa et al. Thm 1) of the form: test risk ≤ worst-train-group risk + $O(\sqrt{\log G / n_{worst}})$, **regardless of which group the test distribution looks most like**. So if the test is roughly some interpolation of train sym-buckets, GroupDRO bounds its risk.

The nontrivial question is whether the effective "test environment" lies in the convex hull of our 20 train environments. We cannot prove this. But heuristically: if syms 0–4 already span diverse market behaviors and date quartiles span diverse volatility regimes, the convex hull is a reasonable model of "reachable" market regimes.

#### Implementation sketch
- Cleanest impl: LightGBM does not natively support DRO. Two paths:
  1. **Sample-weight version (cheap, less faithful)**: at each boosting round, reweight rows by a running estimate of per-group loss. Implementable as a custom callback.
  2. **PyTorch MLP / TabNet / FT-Transformer with gradient-based DRO** (more faithful). Train for ~15 min on GPU.
- I recommend path **1** first because it's a 50-line patch. If LOSO-bypass metrics improve, escalate to path 2.

#### LOSO-bypass check
The signature DRO metric is **worst-group risk**:

- After training, compute per-(sym × bucket) PnL on a held-out slice. The model that minimizes the *worst* group's PnL is the one that should transfer — not the one with the best average.
- Specifically, compute **`min over groups of group-PnL`**. Pick the model whose `min group PnL` is largest. Empirically, for ERM this minimum is often a strongly negative number for a "latest-quartile sym".
- Track **(min_group_PnL / mean_group_PnL)** ratio. ERM gives ratios around 0.2–0.4; GroupDRO should push to 0.5+. Models with high ratio are predicted to transfer; models with low ratio are not.

#### Risk
- DRO can overfit to one noisy "hardest" group, sacrificing aggregate quality. Mitigate by **smoothed CVaR-DRO** at the 80th-percentile worst group rather than the literal max.
- 20 groups is on the small side for stable DRO; if the worst group has only 50k rows, its loss estimate is noisy. Consider merging adjacent date buckets.
- Gradient boosting + DRO is finicky; the sample-weight hack is approximate.

#### Confidence: **M-H**
Sagawa GroupDRO (2020), CVaR DRO (Levy 2020), and the related risk-extrapolation REx (Krueger 2021) are well-established. Applied to time-series finance specifically, evidence is sparser but emerging (Lopez-Lira 2023; some 2025 NeurIPS workshop papers on regime-robust quant). Pure ERM → GroupDRO swap on tabular data typically buys 1–3% test accuracy under shift, which on PnL terms could be the +12-point gap we are missing.

---

## Honorable Mentions (briefly)

### Pick 4 — **OOD Abstention via Density Score** (Confidence: M-H)
Train a small density estimator (Mahalanobis distance in PCA space, or normalizing flow, or `IsolationForest`) on train features. At test time, score each row's likelihood. Below a threshold → abstain. Pairs orthogonally with Pick 2 (conformal abstention is on `p(y|x)`; density abstention is on `p(x)`). Cheap to add. Risk: density score can be dominated by 1–2 high-variance features; needs whitening first. **Combine with Pick 1**: density score on stationary features only — most informative.

### Pick 5 — **Importance-Weighted ERM with "Recency Discriminator"** (Confidence: M)
Train binary classifier $d(x)$: "is this row from B3 (90–119) or earlier?" Use $w(x) = d(x) / (1 - d(x))$ as importance weight in retraining. Defends if test distribution looks like an extrapolation along the B0→B3 trajectory. Theoretically clean (Sugiyama density-ratio framework), risky if test is *not* a continuation of the within-train trend.

### Pick 6 — **Slice-Ensemble with Held-Out Calibration** (Confidence: M)
Train K=6 models on overlapping date windows (each 30 days wide, stride 15). Aggregate via **uniform average** at inference (no learned weights — learned weights overfit). Each base model is small (LGB with 200 trees). Diversity of regimes seen + averaging reduces variance. The simplest version of "deep ensembles for OOD" applied to time-series.

---

## Cross-cutting recommendation: replace the model-selection objective

The single most important takeaway is **changing what we optimize**.

**Don't pick the model with the best LOSO PnL.** Instead, define a composite **TransferScore**:

$$
\text{TransferScore} = w_1 \cdot \text{B3-PnL} + w_2 \cdot \min_g \text{group-PnL} + w_3 \cdot (-\max_i \text{KS}(f_i; B_0 || B_3)) + w_4 \cdot \text{ConformalCoverageStability}
$$

with `w = [0.4, 0.3, 0.15, 0.15]` as a starting point.

This composite directly penalizes the failure modes that turn LOSO gains into platform losses: in-train temporal weakness, worst-group brittleness, feature non-stationarity, and miscalibrated abstention. Any modeling proposal — including the three above — should be ranked by TransferScore, not by LOSO PnL.

If TransferScore disagrees with LOSO PnL on a given change, **trust TransferScore**. The LOSO-platform decorrelation is empirical evidence that LOSO is biased; TransferScore at least includes terms with theoretical transfer guarantees (KS, conformal coverage).

---

## Recommended sequencing (low-risk → high-risk)

1. **Week 0 (immediate)**: Build the `tools/feature_stationarity_audit.py` and TransferScore harness. No model changes yet. 1 day.
2. **Week 0**: Run audit. Drop / transform top-N most non-stationary features (Pick 1). Retrain existing best model on cleaned features. **First gate**: TransferScore must improve over current best on the same model. 2 days.
3. **Week 1**: Add Robust Cross-Sym Conformal Abstention (Pick 2) on top. Tune α on B3, not LOSO. 1 day.
4. **Week 1**: Submit (1)+(2). This stack alone has the highest probability of closing the gap because both are theoretically grounded and almost-purely-defensive (low downside).
5. **Week 2**: If gap remains, escalate to GroupDRO (Pick 3). This is higher variance but addresses residual non-invariance.
6. **Always**: keep 4/5/6 in the bench as ablation studies; pair them with the top 3.

---

## What I deliberately did *not* propose (and why)

- **Heavy DA techniques (DANN, MMD, CORAL)** — they require a target-domain *feature* sample, which we don't have. Sym-leave-one-out is too small a proxy and the optimization tends to be unstable on tabular data with mixed feature scales.
- **Test-time training (TTT, TENT)** — Predictor is **stateless across calls**. We cannot accumulate test gradients. Even within-call adaptation is doable but yields tiny gains relative to risk of breaking the deterministic predictor contract.
- **Bayesian model averaging in a heavy-handed sense** — boostrapping LGB models gives nearly the same variance reduction as a full Bayesian treatment, at 1% the cost. Use simple ensembles.
- **Self-supervised pretraining** — for tabular HFT data, SSL gains vs supervised are small; not worth the engineering cost given the time budget.
- **Adding more / fancier models (XGBoost-with-MAE, CatBoost-Quantile, etc.)** — the LOSO/platform decorrelation says we are *already overfit to the LOSO objective*. More model variants will sharpen LOSO further while platform stagnates or regresses.

The unifying thesis: **the bottleneck is calibration of evaluation and inductive bias, not modeling capacity.**

---

## Final ranking & confidence summary

| Rank | Strategy                                          | Confidence | Effort | Theoretical guarantee on transfer? |
|-----:|---------------------------------------------------|:----------:|:------:|:-----------------------------------|
| 🥇 1 | Stationary Feature Reformulation + KS-Pruning     | **H**      | Low    | ✅ TV-bound under bounded marginal shift |
| 🥈 2 | Robust Cross-Sym Conformal Abstention             | **H**      | Low    | ✅ Conformal coverage degrades gracefully |
| 🥉 3 | Group DRO over Sym × Date-Bucket Environments     | **M-H**    | Med    | ✅ Worst-group bound (if convex hull holds) |
|    4 | OOD Density Abstention (combine w/ #2)            | M-H        | Low    | Heuristic                              |
|    5 | Importance Weighting via Recency Discriminator    | M          | Low    | ✅ if shift is along B0→B3 trajectory   |
|    6 | Slice-Ensemble with Held-Out Calibration          | M          | Med    | Heuristic / variance-reduction         |

**Bet recommendation**: combine Pick 1 + Pick 2 first; expect them alone to close most of the gap. Hold Pick 3 in reserve for the second submission.

---
