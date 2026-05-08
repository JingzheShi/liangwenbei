# T104 — Blinded Tabular ML / Financial Loss Proposal

> **Mode**: Blinded research worker. Author has NOT read `experiments/`, `audits/`, `research_and_history/`, or any PROGRESS / REPORT files for Parts 1–2. Only `CRITICAL_CONSTRAINTS.md`, `docs/data_schema.md`, and the prompt brief were consulted.
> **Lens**: Tabular ML / financial time series / loss formulation / training tricks. (No microstructure / no DL backbone speculation beyond what tabular DL implies.)
> **Date**: 2026-05-08

---

## Part 1 — Hardest ML Sub‑Problem

The deepest difficulty is NOT "predict mid‑price direction accurately". It is:

> **Reconciling the threshold‑truncated 3‑class label with the continuous‑return PnL metric, while preserving sym‑OOD invariance and respecting an abstain‑capable trading rule.**

This is one knot composed of four mismatches that all defeat naive cross‑entropy:

1. **Loss / metric mismatch.** We train CE (or accuracy) but are scored on cumulative PnL with a 0.01% double‑sided commission. A model with 60% accuracy can lose money if it is *most confident on small moves and least confident on big moves* — exactly the opposite of what PnL wants.
2. **Label noise from threshold truncation.** Labels are step functions of a continuous variable: |Δmid| < α → "flat", else direction. Two adjacent samples whose realized return is α±ε get *opposite labels* despite identical reality. Around the threshold ribbon, label noise is structural and dominates CE gradient. Information about *magnitude* is permanently destroyed in the target.
3. **Abstain is an action, not a class.** "Flat" prediction means "don't trade". The economically optimal "flat" decision is *not* "the model thinks |Δmid| < α"; it is "the cost‑weighted expected reward of trading is < expected reward of abstaining". A model trained with symmetric CE on the 3 classes never sees this asymmetry.
4. **Sym‑OOD invariance.** Test sym IDs 0–4 may map to *unseen* stocks. So per‑sym normalisation, sym embeddings, sym‑specific models, *and* features whose distribution is wildly sym‑specific are all forbidden — but each underlying stock has different micro‑structure (depth, spread, vol). The model must extract sym‑invariant predictive structure while not collapsing onto the average sym.

The combinatorics: a fix that nails one of these (e.g. focal loss for imbalance) typically *worsens* another (focal loss makes the threshold‑ribbon noise hotter). The model class that survives this knot must be (a) regression‑ or quantile‑based on the continuous return so label truncation is bypassed, (b) decision‑aware so the abstain action is endogenous, and (c) explicitly encouraged toward sym‑invariance during training.

Secondary difficulty: **5 horizons must be predicted but only the best horizon counts for ranking** — so per‑horizon optimisation dominates, but multi‑task encoders share regularisation benefit. This is mainly an architecture / training‑schedule question, not the core knot.

---

## Part 2 — Disruptive Methods

I propose 8 methods. Each is "basic" (in the sense that it is a textbook reformulation, not a kitchen‑sink) but disruptive because it ditches some default assumption (CE on labels / point‑estimate prediction / ERM on stacked data).

For each: 1‑line description / source / why basic / why disruptive / build cost / expected lift over a tuned LGB‑CE baseline / risks.

---

### M1. Decision‑Focused Learning with PnL regret loss (SPO+ / direct surrogate)

- **Description.** Replace CE with a loss that is the *regret* of the model's downstream trading action vs the optimal action under the realised return. Either Elmachtoub‑Grigas SPO+ (convex surrogate of SPO) or a custom hinge‑style PnL regret directly: `L = max(0, threshold − margin) + commission * 1{trade}`.
- **Source.** Elmachtoub & Grigas, "Smart Predict, then Optimize", *Mgmt Sci* 2022 (arXiv 1710.08005). Liu, Grigas, "Risk Bounds and Calibration for SPO" (NeurIPS 2021). Mandi 2024 portfolio SPO+ (arXiv 2601.04062 — 14.05% annualised, best Sharpe).
- **Why basic.** Train‑metric / eval‑metric mismatch is the most fundamental fix in the book.
- **Why disruptive.** Most teams here will be optimising CE with `class_weight=balanced` or thresholding logits at inference. SPO+ folds the threshold *and* the abstain *and* the commission into the gradient. Predictions stop trying to be calibrated probabilities; they become *good for the trading rule*, which is what is scored.
- **Implementation difficulty.** Medium. SPO+ subgradient is known and fits inside a custom LightGBM objective (`lgb.train(fobj=spo_plus_grad)`); ~150 LoC. Tunable: commission, abstain margin, directional asymmetry.
- **Estimated lift.** **+5 to +12 LOSO‑equiv units** (rationale: directly aligned to scoring; portfolio paper saw ~30‑40% relative Sharpe lift over CE‑PtO).
- **Risks.** SPO+ surrogate is convex but not as smooth as CE; needs careful early stopping. Subgradient zero in the "agreement zone" can cause stalled trees. Mitigation: add small CE auxiliary term (`λ·CE + (1−λ)·SPO+`).

### M2. Quantile regression with dual‑gate trade rule (Q_lo / Q_hi decision)

- **Description.** Drop classification. Train two regression heads per horizon: τ_lo‑quantile and τ_hi‑quantile of `Δmid_t→t+h`. Trade rule: *long* iff `Q_lo(x) > +α + cost`, *short* iff `Q_hi(x) < −α − cost`, else *flat*. Choose τ_lo, τ_hi (and `cost` margin) on a held‑out fold to maximise PnL.
- **Source.** Koenker & Bassett, "Regression Quantiles" 1978; Bracher et al., "DeepLOB‑QR: Extending Deep Learning Models for LOB to Quantile Regression" (Oxford‑Man 2020). LightGBM has `objective='quantile'` natively.
- **Why basic.** Quantile regression is 1978 statistics. Pinball loss is well‑understood and stable.
- **Why disruptive.** Replaces the entire 3‑class CE pipeline. Naturally encodes "I am confident enough to trade" without ever touching α. The decision rule becomes a 2‑D threshold optimisation on Q_lo / Q_hi instead of a 1‑D softmax tuning. In practice, Q_lo > +α requires the model to think *the worst‑case 30%‑lower estimate is still a winner* — that is exactly the right uncertainty‑aware trade gate.
- **Implementation difficulty.** Low. Two LGB objective='quantile' models per horizon (10 models total for 5 horizons). Inference: 2× cost of one model.
- **Estimated lift.** **+3 to +8 units**. Tighter coupling between predictive uncertainty and trade decision; especially helps when realised vol is heteroscedastic across stocks (which it is on A‑shares).
- **Risks.** Quantile crossings (Q_lo > Q_hi) — need rearrangement (Chernozhukov 2010 monotone rearrangement). Slow training. May overfit threshold pair on small val.

### M3. Robust Huber/Tukey regression on continuous Δmid + decision postprocess

- **Description.** Single regression head per horizon on raw `Δmid_t→t+h` (continuous). Use **Huber** (or **Tukey biweight**) loss to clip outlier spikes (price jumps, halt resumes). Inference: trade if `|pred|` exceeds a tuned `α + margin`. Optionally **asymmetric Huber**: smaller δ on "wrong‑sign large‑magnitude" side to penalise confident wrong direction harder than wrong magnitude same direction.
- **Source.** Huber 1964 (Annals of Math Stat); Beaton & Tukey 1974; Fu & Wang 2021 "Robust regression with asymmetric loss functions" (Stat Methods Med Res). Many Kaggle financial tabular winners use Huber.
- **Why basic.** Switch from classification on truncated label → regression on continuous truth recovers magnitude information that CE permanently destroys.
- **Why disruptive.** Most teams will treat this as a 3‑class problem because the labels are 3‑class; using `(midprice_{t+h} − midprice_t)` directly as the regression target is a small code change with large information gain. Huber/Tukey makes it stable against the heavy left tail of A‑share circuit‑breaker / suspension events.
- **Implementation difficulty.** Trivial. LGB `objective='huber'` exists; Tukey requires custom `fobj` (~30 LoC).
- **Estimated lift.** **+2 to +6 units** standalone; potentially additive with M1/M2.
- **Risks.** Regression on rare large moves: variance explosion if not robustified. Asymmetric Huber needs careful left/right δ.

### M4. Reject‑option / Selective prediction with cost‑sensitive abstain class

- **Description.** Reformulate to 4‑class { down, flat, up, **abstain** } where the abstain action has fixed cost `c_abst`. Optimal Bayes rule: predict abstain whenever `min_y posterior_loss(y) > c_abst`. Train with selective surrogate (e.g. SelectiveNet / Cortes "Boosting with Abstention" / Chow's reject rule with cost matrix).
- **Source.** Chow 1957; Cortes, DeSalvo, Mohri "Boosting with Abstention" NeurIPS 2016; Geifman & El‑Yaniv "SelectiveNet" ICML 2019; Charoenphakdee et al. "Classification with Rejection Based on Cost‑Sensitive Classification" ICML 2021.
- **Why basic.** "Should I act or not act under uncertainty" is the textbook formulation here.
- **Why disruptive.** Replaces the implicit decision rule of "argmax softmax with τ" by an *endogenous* learned abstain that is aware of asymmetric mistake costs. Cleanly maps to "trade vs no‑trade".
- **Implementation difficulty.** Medium. For LGB, easiest path is a 4‑output model with cost‑sensitive surrogate; alternative is to train a separate "trust" head on top of an existing 3‑class model (selective classifier head, Geifman 2019).
- **Estimated lift.** **+3 to +7 units**.
- **Risks.** `c_abst` is a knob — too low → model abstains everywhere → 0 PnL. Needs careful val‑PnL tuning. Selective surrogate losses are convex but unfamiliar; debugging time.

### M5. Self‑supervised DAE pretraining with swap‑noise → frozen features for GBDT

- **Description.** On *all* ~2.4M rows (labels not needed, can include any extra unlabeled session), train a denoising autoencoder. Corruption = **swap noise**: with prob p (e.g. 15%), each value is replaced by a random sample from the same column. Reconstruction loss = MSE per column. Take the bottleneck (e.g. 64 dims) as additional features for the downstream LGB / quantile / SPO models.
- **Source.** Vincent 2010 "Stacked Denoising Autoencoders" (JMLR); Heaton's swap‑noise DAE for Porto Seguro Kaggle 2017 (1st place solution); Yoon "VIME: Self‑/Semi‑Supervised Learning to Tabular Domain" NeurIPS 2020; SubTab (NeurIPS 2021).
- **Why basic.** Self‑supervised pretraining is the dominant pretraining recipe of the last 8 years.
- **Why disruptive.** Almost everyone here is using only the 154 hand‑crafted Kercheval‑Zhang features. The DAE bottleneck captures *latent micro‑structural correlations* (e.g. joint behaviour of OFI + spread + depth) the hand‑crafted features average over. In Porto Seguro it gave the entire competition's lift.
- **Implementation difficulty.** Medium. ~300 LoC for the DAE; 1–2 GPU‑hours to pretrain on the full dataset.
- **Estimated lift.** **+1 to +4 units** (additive).
- **Risks.** Bottleneck quality sensitive to corruption rate; swap‑noise is reportedly weak on some datasets (VIME paper). Mitigation: also try column‑wise mixup (Mixup of pairs of rows after column‑independent swap).

### M6. Group DRO / IRM‑style sym‑invariant training

- **Description.** Treat each sym (or sym × time‑band) as a *group*. Instead of ERM over all rows, minimise the **worst‑group loss** (group‑DRO, Sagawa 2020) or penalise the variance of per‑group risk gradients (IRMv1, Arjovsky 2019). Implementable in LGB by *re‑weighting* rows so each gradient step is equivalent to a worst‑case‑group step (online importance reweighting), or by training one model per held‑out sym and ensembling (LOSO ensemble).
- **Source.** Sagawa, Koh, Hashimoto, Liang "Distributionally Robust Neural Networks for Group Shifts" ICLR 2020; Arjovsky, Bottou, Gulrajani, Lopez‑Paz "Invariant Risk Minimisation" 2019; Soma et al. 2022 (group DRO ergodic bounds).
- **Why basic.** Worst‑case minimax is decades‑old (Wald 1939). Group DRO is "ERM with a re‑weighting on groups".
- **Why disruptive.** The constraint says test sym ⊃ unseen stocks. ERM gives sym‑averaged performance; DRO gives sym‑*worst* performance, which is what generalisation to a new stock looks like. Almost no tabular team uses DRO.
- **Implementation difficulty.** Low‑Medium. For LGB: per‑row weight = (group_loss_share × ema). Wrap in standard training loop. ~80 LoC.
- **Estimated lift.** **+1 to +5 units**, mostly visible in OOD‑sym CV but should also lift held‑out date generalisation.
- **Risks.** DRO is conservative; may underfit easy syms. IRM is famously brittle on real datasets. Mitigation: tune group weighting strength `η` carefully; use group‑DRO before IRM.

### M7. Stochastic weight averaging (SWA) + EMA self‑distillation for label‑truncation noise

- **Description.** Two complementary moves: (a) average model weights / leaf weights over the last K epochs of training (SWA, Izmailov 2018) — for LGB, equivalent to *checkpoint averaging* or final‑K‑round averaging of leaf outputs. (b) **EMA‑teacher self‑distillation**: train student on `α·CE(label) + (1−α)·KL(student || EMA‑teacher.soft)`. Soft EMA labels regularise threshold‑ribbon noise.
- **Source.** Izmailov et al. "Averaging Weights Leads to Wider Optima" UAI 2018; Tarvainen & Valpola "Mean Teachers Are Better Role Models" NeurIPS 2017; Furlanello "Born‑Again Networks" ICML 2018; Caron 2021 DINO.
- **Why basic.** Ensembling and label smoothing are first‑week tricks.
- **Why disruptive.** The label noise here is *structural* (threshold truncation), not random. Soft EMA labels effectively *interpolate across the threshold ribbon*, recovering some of the magnitude information that the labeller threw away. Cheap & reliable.
- **Implementation difficulty.** Low. For NN: native. For LGB: train 2 generations — gen2 sees `0.5·hardlabel + 0.5·gen1_softprob` and uses CE on the soft mix.
- **Estimated lift.** **+1 to +3 units**. Not glamorous, but reliable.
- **Risks.** Two training passes. EMA momentum tuning.

### M8. NGBoost / probabilistic GBDT for decision‑theoretic abstain

- **Description.** Use **NGBoost** (Duan 2020) which outputs a full predictive *distribution* per row (e.g. Normal(μ, σ²) on continuous return). Trade if the lower 1‑sided CVaR exceeds the cost: `μ − k·σ > α + cost` for long, `μ + k·σ < −α − cost` for short, else flat. `k` tuned on val.
- **Source.** Duan, Avati et al. "NGBoost: Natural Gradient Boosting for Probabilistic Prediction" ICML 2020.
- **Why basic.** Just "regression with an explicit variance head".
- **Why disruptive.** Closes the gap that M2 (quantile) only partially closes — gives full distribution, not 2 quantiles. Decision rule is the textbook expected‑utility maximiser, no threshold tuning beyond `k`.
- **Implementation difficulty.** Low. NGBoost is `pip install`. Slower training than LGB (~3–5×).
- **Estimated lift.** **+1 to +3 units** (but may stack with M1 / M2 if used as a diversity component).
- **Risks.** Slower; smaller community than LGB; may not exceed M2 standalone. Best as *ensemble component*, not replacement.

### (Bonus, lower‑priority) — Tabular DL alternatives

If a NN is in budget: **TabM** (Gorishniy ICLR 2025, arXiv 2410.24210) is the current SOTA tabular DL model — parameter‑efficient ensemble of MLPs, beats FT‑Transformer on the GBM benchmark suite. Worth one experiment as a diversity component for an ensemble; not as a replacement for GBDT.

---

## Method Cross‑comparison

| Method | Attacks | Cost | Std lift | Stacks with |
|---|---|---|---|---|
| **M1 SPO+ regret** | loss/metric mismatch | Med | High | M3, M5, M6 |
| **M2 Dual quantile** | label truncation + abstain | Low | Med‑High | M3, M5 |
| **M3 Huber regression** | label truncation | Trivial | Med | M1, M2, M6 |
| **M4 Reject option** | abstain | Med | Med | M1 |
| **M5 DAE pretraining** | feature gap | Med | Low‑Med | every model |
| **M6 Group DRO** | sym OOD | Low‑Med | Low‑Med | every model |
| **M7 SWA / EMA distill** | label noise | Low | Low‑Med | every model |
| **M8 NGBoost** | uncertainty / abstain | Low | Low‑Med | as ensemble |

The cheapest lift bundle: **M3 + M2 + M7** (≈ 1 day, cumulative +5 to +10).
The most aggressive bet: **M1 SPO+** (≈ 2–3 days, +5 to +12 alone).

---

## Part 3 — Gap Analysis vs Current Pipeline

After writing Parts 1–2, I read the four permitted files (PROGRESS.md, T75 REPORT.md, T87 REPORT.md, T99 dir + ev_full_ensemble.log). Trajectory so far:

| Iter | Method | LOSO‑equiv |
|---|---|---|
| iter_012 | 3‑class CE + DE 4D probability gate | +26.44 |
| iter_013 (T75) | LGB **L2 regression** on Δmid_norm + EV gate | +36.23 |
| iter_014 (+T81) | + MLP L2 regression (warm) | +38.28 |
| iter_015 (T87) | + **SPO+ DFL** fine‑tune of the MLP | +40.13 |
| T99 LGB‑Huber alone | **Huber** regression (α=0.001) | +40.79 |
| T99 LGB+CB Huber blend | + CatBoost Huber | +41.56 |
| T99 T87 + LGB‑Huber 1:0.5 | + DFL NN diversity | +41.85 |

Independent diversity component: T95_GRU (corr 0.34 with LGB) is too weak alone (+9.96) but might be useful as a tiny ensemble weight.

### Mapping my Part‑2 proposals onto current state

| ID | Proposal | Status in current pipeline |
|---|---|---|
| **M1** SPO+ DFL regret | ✅ **Done** — T87 (λ_spo=30 settled, 5‑seed warm‑start from T81). +1.85 over iter_014 ensemble; +3.52 mean per‑seed. Validated. |
| **M3** Huber regression | ✅ **Done** — T99 LGB‑Huber α=0.001 5‑seed and CB‑Huber α=0.001 5‑seed. LGB‑Huber alone +40.79; LGB+CB blend +41.56. |
| **M3 asymmetric** Huber | ❌ **Not done** — Tried symmetric only. Predictor errors look biased upward (T75 thr_up=3.72e‑4 vs thr_dn=1.61e‑4); asymmetric δ on the loss side could replace post‑hoc threshold asymmetry. **Cheap, low‑hanging.** |
| **M2** Dual‑quantile + dual gate | ⚠️ **Partial** — `cb_quantile_a0.5` (single median quantile, ≈ MAE) was tried as a single‑variant probe. **No Q_lo + Q_hi dual‑gate decision rule.** This is the largest gap on the *decision/uncertainty axis*. |
| **M4** Reject‑option / selective abstain | ❌ **Not done** — abstain is implemented purely as a post‑hoc EV‑gate threshold; no learned abstain head. |
| **M5** DAE / swap‑noise pretraining | ❌ **Not done** — feature set is "T68 schemeP cache, 359 dims" all hand‑crafted KZ‑style. **Zero self‑supervised features.** |
| **M6** Group DRO / IRM | ❌ **Not done** — sample weighting is "class‑balanced on 3‑class label". No per‑sym worst‑group reweighting. (NB: blue gate is already sym‑agnostic, so the fitness benefit is bounded by date/regime drift, not sym OOD.) |
| **M7** SWA / EMA self‑distill | ❌ **Not done** — 5‑seed averaging done, but no soft‑label self‑distillation generation 2. |
| **M8** NGBoost / probabilistic GBDT | ❌ **Not done.** Quantile head from M2 likely subsumes most of NGBoost's marginal value. |
| Bonus TabM | ❌ **Not done** — only DeepLOB / GRU as DL backbones. |

### Key observations

1. The team **independently arrived at M1 + M3** (SPO+ on regression + Huber on continuous Δmid). This is convergent evidence that Parts 1–2's framing of "loss/metric mismatch" and "label truncation" is correct.
2. **The single biggest unexploited axis is uncertainty‑aware prediction.** All current models output point estimates; the EV gate uses a *deterministic* threshold on that point. There is no head that says "I'm not confident enough to trade *this* row even if its mean is above threshold". M2 (dual quantile) is the most direct attack on this gap and is *unexplored*.
3. **The second largest is the feature axis.** All six current models use the same 359‑dim hand‑crafted feature cache. Five of them have cross‑correlation 0.79–0.93 (T75/T87/T89/T99‑LGB/T99‑CB) — they are very close cousins in feature‑space. Only T95_GRU is genuinely diverse (corr ≈ 0.35) but it's a sequential model on raw windows, which is a different axis. **DAE / swap‑noise (M5) would inject latent features that could either replace some hand‑crafted ones or, more likely, give every existing booster a small lift while increasing ensemble diversity.**
4. **Asymmetric Huber (mini‑M3‑variant) is a 30‑LoC fix with a clean motivation**: the EV gate is already learning the asymmetry post‑hoc (thr_up = 2.3× thr_dn in T75); pushing it into the loss may give a small but free bump.
5. M4 (selective abstain) is intellectually appealing but in this pipeline it would be *replacing* the EV‑gate, which is well‑tuned and works. The expected lift is uncertain and the risk of regression is higher than the upside. Not a top‑2 pick.

### Convergent / "殊途同归" mappings

- M1 SPO+ ≈ T87 exactly (same paper, same surrogate, same warm‑start strategy).
- M3 Huber ≈ T99 exactly.
- T75 EV gate (`pred > 1.25 · 2·FEE`) is structurally a single‑quantile (median) decision — the analytic limit of M2 with τ_lo = τ_hi = 0.5.
- DE 4D thresh in iter_012 was a heuristic ancestor of the M4 cost‑sensitive abstain (different mechanism, same goal).

---

## Part 4 — Top 2 Recommendations

### #1 — **Dual quantile regression with dual‑gate decision rule** (M2)

**Why this wins the priority list.**
- *Pipeline gap*: yes — only median quantile attempted; dual‑gate decision rule never tried.
- *Impact*: **+2 to +6 LOSO‑equiv** standalone, with strong likelihood of additive gain when added as an ensemble component (current ensemble cousins are corr 0.79–0.93; quantile heads should sit at 0.7–0.8, similar to T87's diversity injection).
- *Cost*: 1 day. LightGBM `objective='quantile'` is native. Train 2 models per horizon × 5 seeds = 10 boosters per horizon; reuse the existing 359‑d feature cache, V4 walk‑forward split, and the existing `ev_gate_eval.py` infra (extend the gate from 1‑D to 2‑D threshold tuning).
- *Mechanism*: Trade `long` iff `Q_lo(x) > +α + cost`, `short` iff `Q_hi(x) < −α − cost`, else `flat`. Compared to point regression, this answers a stronger question: *"is even the lower 30%‑quantile of the conditional return distribution still profitable?"* — naturally encoding heteroscedastic uncertainty.
- *Risk* — quantile crossing (Q_lo > Q_hi); fix with monotone rearrangement (Chernozhukov 2010) or by training jointly with a multi‑output head. Threshold pair (τ_lo, τ_hi) needs DE optimisation on val (already plumbed for the EV gate).
- *Concrete plan*:
  1. Add `train_quantile.py` that copies `T99/train_lgb.py` but swaps `objective='quantile' alpha=0.30` and `0.70`. 5 seeds × 2 quantiles = 10 boosters.
  2. Extend `ev_gate_eval.py` to a 4‑threshold sweep (τ_lo_long, τ_hi_short) instead of (thr_up, thr_dn). Reuse DE.
  3. Evaluate **standalone** (Q‑gate) and **ensemble** (mean(point regression) used as a *third* signal, with Q_lo/Q_hi as gating constraints).
  4. Symmetric fallback: τ_lo = τ_hi = 1.25·2·FEE for non‑per‑test thresholding (defensive).

### #2 — **DAE / swap‑noise self‑supervised pretraining of a 64‑d feature bottleneck**, used as additional features for all GBDT and NN heads (M5)

**Why this wins #2.**
- *Pipeline gap*: yes — zero self‑supervised feature learning. All current features hand‑crafted Kercheval‑Zhang style.
- *Impact*: **+1 to +4 LOSO‑equiv** distributed across every head (LGB‑Huber, CB‑Huber, T87 NN, T75 LGB), + likely diversity bump that lifts the ensemble more than the sum of parts.
- *Cost*: 2–3 days. ~300 LoC; 1–2 GPU‑hours pretrain, 1 day to retrain all 6+ heads with extended features.
- *Mechanism*: Train an MLP autoencoder (e.g. 359 → 256 → 128 → **64** → 128 → 256 → 359) on 100‑row contiguous windows from *every* parquet, with **swap noise** corruption (15% of feature values replaced by sampled values from the same column). Reconstruction loss = per‑column MSE (or column‑wise z‑score MSE). Freeze encoder; concatenate the 64‑d bottleneck to the 359‑d schemeP cache → **423‑d** feature for every booster.
- *Risk*: bottleneck quality varies with corruption rate; swap noise reportedly weak on some tabular datasets (VIME paper). Mitigation: (a) tune corruption rate ∈ {0.05, 0.10, 0.15, 0.20}; (b) also try column‑wise mixup as a second corruption family.
- *Concrete plan*:
  1. `pretrain_dae.py`: window‑level DAE on full data (no labels needed). 5 epochs, AdamW.
  2. `extract_dae_features.py`: forward pass over 1200 parquets → write per‑file `dae_64.parquet` cache.
  3. Modify `T99/train_lgb.py` and `T87/train_t87_spo.py` to concatenate dae_64 columns. Re‑train 5‑seed.
  4. Compare: (a) DAE‑only feature set vs (b) schemeP‑only vs (c) concat — concat almost certainly wins.
- *Defensive* — if standalone result is flat, the DAE‑only feature axis still adds **decorrelation** to the ensemble (the bottleneck captures different latent structure than KZ features), and that often pays even when no marginal LOSO lift is visible per‑model.

### Why these and not the others

- **M1 / M3** already done (skip).
- **M3‑asymmetric** is cheaper than #1 but smaller expected lift (≈ +0.3 to +1.0); a *good free bonus* to do alongside, not a top‑2 own.
- **M4** higher risk than upside; abstain is already partially solved by EV gate.
- **M6** sym‑agnostic constraint already enforced architecturally; DRO marginal value bounded.
- **M7** EMA distillation: useful but small (+1 to +3) and lower priority than feature axis or quantile axis.
- **M8** NGBoost: largely subsumed by M2 dual quantile.

---

## Reading list (sources)

- Elmachtoub & Grigas, "Smart Predict, then Optimize", *Mgmt Sci* 2022 (arXiv 1710.08005). [PDF](https://arxiv.org/pdf/1710.08005)
- Mandi et al., "Smart Predict‑then‑Optimize Paradigm for Portfolio Optimization in Real Markets" (arXiv 2601.04062, 2025).
- Liu, Grigas, "Risk Bounds and Calibration for SPO" (NeurIPS 2021).
- Bracher et al., "DeepLOB‑QR: Extending Deep Learning Models for LOB to Quantile Regression" (Oxford‑Man 2020).
- Koenker & Bassett, "Regression Quantiles" *Econometrica* 1978; Chernozhukov et al. "Quantile and Probability Curves Without Crossing" *Econometrica* 2010.
- Huber 1964 (Annals of Math Stat); Beaton & Tukey 1974; Fu & Wang, "Robust regression with asymmetric loss functions", *Stat Methods Med Res* 2021.
- Vincent et al., "Stacked Denoising Autoencoders" *JMLR* 2010; Heaton 2017 Porto Seguro Kaggle DAE write‑up.
- Yoon et al., "VIME: Self‑/Semi‑Supervised Learning to Tabular Domain" *NeurIPS* 2020; SubTab *NeurIPS* 2021.
- Cortes, DeSalvo, Mohri, "Boosting with Abstention" *NeurIPS* 2016; Geifman & El‑Yaniv, "SelectiveNet" *ICML* 2019; Charoenphakdee et al. "Classification with Rejection" *ICML* 2021.
- Sagawa, Koh, Hashimoto, Liang, "Distributionally Robust Neural Networks for Group Shifts" *ICLR* 2020; Arjovsky et al. "Invariant Risk Minimisation" 2019.
- Izmailov et al., "Averaging Weights Leads to Wider Optima" *UAI* 2018; Tarvainen & Valpola "Mean Teachers" *NeurIPS* 2017.
- Duan et al., "NGBoost: Natural Gradient Boosting for Probabilistic Prediction" *ICML* 2020.
- Gorishniy et al., "TabM: Advancing Tabular Deep Learning with Parameter‑Efficient Ensembling" *ICLR* 2025 (arXiv 2410.24210).

---

```
RESULT: task=blinded_ml_tabular top_recommendations=[M2_dual_quantile_dual_gate, M5_DAE_swapnoise_pretraining] notes=Blinded analysis converged on SPO+(M1) and Huber-regression(M3) which the team already shipped (T87+T99). Largest unexploited axes are uncertainty-aware decision (dual quantile gate, M2) and self-supervised features (DAE bottleneck, M5). Cheap bonus: asymmetric Huber. Skip M4/M6/M7/M8 as lower expected payoff.
```

