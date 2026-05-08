# W_paradigm_meta_blinded — Meta-Strategy Paradigm Shift

**Author:** worker (opus xhigh, blinded mode)
**Date:** 2026-05-08
**Time spent:** ~75 min web research + 25 min synthesis
**Status:** complete; cross-check against PROGRESS.md in §4

---

## Problem framing (recap, no PROGRESS read)

- 良文杯 LOB direction prediction; 5 sym × 100-tick window; cumulative-PnL scoring; max-over-5-horizons rank.
- Public LB top team **+40 platform**; we are **+28**, gap **+12**.
- Inventory of tricks already tried: LightGBM Huber, NN ensemble, SPO+ DFL, multi-horizon CHIS, monotone constraints, HYD interactions, lag features, 5-LOSO ensemble.
- Pathology: **local LOSO ↑ but platform ↓ or flat**; transmission ratio (local/platform) **unstable 0.4–0.7**.

**Diagnosis (one line):** classic Kaggle shake-up signature — every "improvement" is being absorbed by the local validation distribution and not generalising to the platform's hidden distribution. Single-paradigm depth-first iteration has stopped paying off; we are local-overfit, not under-engineered. The gap will not be closed by another LightGBM hyperparam tune.

This document is therefore not about another single trick. It is about **changing how we run the project**.

---

## Part 1 — What is the +40 team probably doing? (inferred from competition history)

We have no direct access to the top team. But we can triangulate from analogous public competitions where private-LB winners published writeups, and from the mathematical structure of our scoring.

### 1.1 Three plausible "winning paradigms" from comparable competitions

| Comparable | Winner paradigm | Key data exploit / ML lever |
|---|---|---|
| **Optiver Realized Volatility (Kaggle 2021)** | 1st place (Nearest Neighbors): reverse-engineered the chronological order of shuffled `time_id`s by building a KNN graph on stock features and finding a near-Hamiltonian path; once order was recovered, simple lagged target features dominated. | **Data-side exploit** that none of the standard ML tricks could replicate. |
| **Optiver Trading at the Close (Kaggle 2023)** | 1st (HYD): single-model **robust LightGBM** + LGB×MLP ensemble; cached "last 21 days memory" features per stock; emphasis on generalisation, not depth. Notably **chose robust single model over deep ensemble** to survive shake-up. | Memory features + shake-up-aware model selection. |
| **Jane Street Market Prediction (Kaggle 2021)** | 1st (Yirun): **Supervised Autoencoder + MLP** trained jointly (reconstruction loss + prediction loss); bottleneck denoising → noise-robust representation. | Self-supervised noise-removal layered into the model. |
| **Jane Street RTMDF (Kaggle 2024–25)** | (results post-deadline; gold zone consensus from public discussion) blend of denoising-AE-MLP + boosted trees + heavy temporal augmentation. | Multi-paradigm ensemble. |
| **G-Research Crypto (Kaggle 2021–22)** | Top-3 all LightGBM. Winners explicitly attribute it to **feature engineering, not model**. | Cheap to replicate; we already do this. |
| **Algorithmic Trading Challenge (Kaggle 2011)** | Composition of many heterogeneous models + Random Forest meta-learner + hard feature selection. | Heterogeneous portfolio > single-stack depth. |

### 1.2 What the +40 team is *most likely* doing (ranked by posterior plausibility)

**Most likely (P~40%):** they have **a fundamentally different validation protocol** that mirrors the platform's distribution far better than our 442k-test split — e.g., adversarial-validation-weighted CV, or per-sym holdout that spans multiple market regimes. The +12 gap is not from a special model; it is from the absence of our local-overfit penalty.

**Second most likely (P~25%):** they have **massive multi-seed bagging** (50–100 LightGBM seeds, possibly also CatBoost+XGB) with simple averaging or isotonic calibration — pure variance reduction. This would explain stability without explaining the level lift, unless paired with a slightly better feature set.

**Third (P~15%):** they have **a data-side exploit**: e.g., they figured out that `time` field still leaks intraday seasonality, or they reconstruct cross-sym co-movement that we ruled out. (Lower probability because our `CRITICAL_CONSTRAINTS.md` rules out most exploits.)

**Fourth (P~10%):** they use a **deep architecture (TLOB / MLPLOB / DeepLOB-style) with FI-2010 pretraining**. The 2025 TLOB paper reports SoTA on FI-2010 and Tesla/Intel NASDAQ data; transfer learning from a 100-tick LOB pretrain to our 100-tick task is mechanically aligned.

**Fifth (P~10%):** they have a **PnL-aware end-to-end loss** (Sharpe-loss or differentiable PnL) coupled with a position-sizing head, not a 3-class classifier. That explains why standard CE-tuned models top out below them.

### 1.3 What is *unlikely*

- Unlikely that they have a brand-new feature trick we missed. We have iterated 100+ features. Returns to feature engineering have flattened.
- Unlikely they are using sym-conditioned models (forbidden by §1.3).
- Unlikely they are using date features (forbidden by §1.1).
- Unlikely it is just hyperparameter luck — sustained +12 over many submits suggests a structural advantage.

**Headline take:** the gap is most likely **process, not artefact**. They have a better feedback loop (validation + selection) and possibly bigger compute throughput, *not* a secret feature.

---

## Part 2 — 5–10 meta-paradigm shifts

Each entry includes: 1-line description / source / why disruptive / cost / expected platform delta / risk.

---

### M1. **Adversarial Validation–Weighted CV** (validation rebuild)

- **Description:** Train a binary classifier that distinguishes our training data from a small batch of *recent* data we treat as a platform proxy. Use the classifier's probability that a row "looks like platform" as a sample weight and as a stratification axis. Rebuild the local validation split by selecting train rows whose adversarial p-value is highest — i.e., train rows that look most platform-like form the validation set.
- **Source:** Kaggle adversarial validation canon (zakjost, fastml, KDnuggets); also used in 2023 Optiver Close top-10 pipelines for the Nasdaq close-auction shift problem.
- **Why disruptive:** every "trick" so far has been judged on a local split that the model has implicitly learned to over-fit to. AV reweighting *redirects* the optimisation pressure toward the part of the input space that resembles the unseen platform distribution. Without this, more local LOSO simply means more local overfit.
- **Cost:** 1–2 days. One classifier (LightGBM 200 trees), no architectural change. Apply per-fold sample weights.
- **Expected platform delta:** **+3 to +6**. Closes the transmission-ratio gap rather than moving the local mean — the level lift comes from killing the variance noise that's burying real signals.
- **Risk:** if our train data and platform data are *not* meaningfully distinguishable (AV AUC ≈ 0.5), we get nothing. But AUC > 0.5 is the prior here given the +12 LB gap.

---

### M2. **Massive seed bagging + isotonic calibration** (variance kill)

- **Description:** 50 LightGBM seeds × 5 LOSO folds = 250 models. Average raw class probabilities; fit a **single global isotonic regression** on a held-out slice to map ensemble probs → calibrated probs; then a single fixed threshold. Same for any NN replicas.
- **Source:** NVIDIA Kaggle Grandmaster Playbook (2024) explicitly says 100-seed XGBoost bagging "clearly outperformed single-seed training" in the Optimal Fertilizers comp; HYD's 1st-place note that "single robust models" generalise better is the same idea seen from the other side (don't *over*-bag heterogeneous architectures, but *do* bag many seeds of the same robust architecture).
- **Why disruptive:** removes ~2–4 of variance-driven noise we currently mistake for "iter improvement". DE-search over thresholds + 5 Bayes opt seeds is precisely the wrong direction (over-fits selection); seed-bagging is the right direction (averages it out).
- **Cost:** linear in compute. With GPU LightGBM at ~12s/fit and 250 models, that's < 1 hr per submission. Negligible vs human time.
- **Expected platform delta:** **+2 to +5**, mostly via stability (transmission ratio → 0.7+ from 0.4–0.7).
- **Risk:** none material, except the cost of forgoing per-seed diagnostics. Stop running Optuna on threshold; replace with this.

---

### M3. **Supervised Autoencoder + MLP** (Jane Street paradigm)

- **Description:** Build an MLP whose input is the 100×D LOB window flattened (or 1D-Conv-pooled to a fixed vector). Add a **decoder branch** that reconstructs the input from a bottleneck representation. Total loss = α · CE(label_h) + β · MSE(reconstruction). Bottleneck (~64–128 dim) acts as denoised feature representation. Use as a base in our existing ensemble alongside LightGBM.
- **Source:** Yirun's 1st-place writeup, Jane Street Market Prediction 2021 (Kaggle); Springer Journal of Big Data 2025 paper "Supervised Autoencoder MLP for Financial Time Series Forecasting".
- **Why disruptive:** our current NN(s) learn task-specific features that are very prone to over-fit a single horizon's label noise. SAE forces the network to keep *all* the structure of the input via reconstruction, which acts as an implicit regulariser. Empirically this has been the winning paradigm for noisy financial tabular tasks for 5 years.
- **Cost:** 2 days to implement and tune α/β/bottleneck size. Reuse existing data pipeline.
- **Expected platform delta:** **+2 to +5** as a new ensemble member orthogonal to LightGBM; possibly larger if our current NN is shallow MLP.
- **Risk:** training instability with two losses; needs careful α/β balance. Mitigated by cosine-warmup + small batch.

---

### M4. **TLOB / MLPLOB SoTA architecture, optionally pretrained on FI-2010** (deep arch refresh)

- **Description:** Implement the TLOB (transformer with dual attention: temporal × feature) or MLPLOB (purely MLP) architecture from Berti & Mauro 2025 (arXiv 2502.15757). Both are designed for exactly our setting (100-tick LOB window → 3-class direction). Optionally pretrain on FI-2010 (publicly available NASDAQ Nordic LOB data, 5 stocks × 10 days) before fine-tuning on competition data. DeepLOB literature confirms transfer to unseen stocks works → directly addresses our sym-OOD constraint (§1.3).
- **Source:** TLOB paper arXiv 2502.15757 (Feb 2025); LOBCAST benchmark (Springer AI Review 2024); DeepLOB transfer-learning paper Zhang et al. 2018 (Oxford-Man).
- **Why disruptive:** our NN is a hand-rolled MLP (presumably). TLOB/MLPLOB beat every prior LOB DL model on every horizon on FI-2010, NASDAQ Tesla/Intel, and BTC. The architecture is *purpose-built for our setting* — 100-tick window, multi-horizon, sym-agnostic.
- **Cost:** 3–5 days. TLOB GitHub has reference impl; need to adapt feature dimension and add decoder for our 5 horizons.
- **Expected platform delta:** **+3 to +8** if pretraining works; **+1 to +3** without pretraining (since architecture alone may not beat well-tuned LightGBM on our data scale).
- **Risk:** model-zoo reading often does not reproduce on small new datasets; budget half a week and fail-fast.

---

### M5. **Direct PnL / Sharpe loss + position-sizing head** (objective realignment)

- **Description:** Stop predicting 3-class labels. Predict a continuous **position size** ∈ [-1, 1] per (sample, horizon). Loss = – cumulative PnL on a held-out window, computed exactly the way the platform scores. Optionally add a Sharpe-loss term and a turnover penalty. The platform's "max-over-5-horizons" rule means we should also predict a horizon-selection logit and let it learn which horizon to lean on per-window.
- **Source:** Khubiyev 2025 ("Finance-Grounded Optimization for Algorithmic Trading", arXiv 2509.04541) — finance-grounded losses (Sharpe / PnL / MaxDD + turnover reg) outperform MSE on algorithmic-trading metrics. Modified Sharpe-loss handles scale-invariance issue. Decision-by-Supervised-Learning with deep ensembles (arXiv 2503.13544) pushes the same principle for portfolio.
- **Why disruptive:** every CE-trained model is mis-aligned with the platform metric. SPO+ DFL is a step in this direction but uses surrogate; direct PnL loss is honest end-to-end. Also lets the model learn to **abstain** on flat windows, which is most of label_5/label_10 (76%/65% flat).
- **Cost:** 3–4 days. Need a differentiable PnL trace through the per-sample softmax positions; tractable.
- **Expected platform delta:** **+2 to +6** because it directly optimises what the platform measures, not a proxy.
- **Risk:** Sharpe-loss has known instability (scale invariance); use modified Sharpe + small entropy reg on positions.

---

### M6. **Paradigm portfolio: 4 fundamentally different models, blend on platform feedback** (multi-stack)

- **Description:** Stop iterating one stack. Run **4–5 fundamentally different paradigms in parallel**: (a) LightGBM iter016 baseline, (b) supervised autoencoder + MLP (M3), (c) TLOB/MLPLOB (M4), (d) PnL-loss model (M5), (e) kNN / nearest-neighbour baseline (Optiver-Realized-Vol style). Each team submits to platform individually. Then a **simple linear blend** is fit treating recent platform scores as the only validation set (3-5 platform points → ridge with strong prior). Use a **Hill-Climb ensemble** in the spirit of Caruana 2004: start with the best single model, greedily add and reweight others; stop when platform delta plateaus.
- **Source:** Numerai meta-model contribution (MMC) doctrine — payouts based on **orthogonal** alpha; NVIDIA Grandmaster Playbook 2024 (hill-climb ensemble); Jane Street RTMDF top-tier consensus on multi-paradigm blending; Algorithmic Trading Challenge 2011 winner (composition of many models).
- **Why disruptive:** we have spent many months going deep on one stack. Even a moderately worse heterogeneous member can lift platform if it's orthogonal — diversity beats depth at this point. Also, treating platform as validation rather than "truth" inverts our current discipline (which trains on local LOSO).
- **Cost:** 2 weeks of parallel work, but each stream is independent so dispatch separately. M3 + M4 + M5 above are the streams; this is the meta-orchestration layer.
- **Expected platform delta:** **+5 to +10** — diversity premium is the highest-EV move at this gap size.
- **Risk:** runs hot on submission budget. Need to think carefully about which combos to test; pre-register a 4×4 blend matrix.

---

### M7. **Train on full 0–119 dates with no early stopping ("blend train+val" final-retrain)** (data utilisation)

- **Description:** Standard Kaggle final-submit trick. We currently train on dates 0–79, validate on 80–119 (or some such). For the final platform submission, **retrain on 0–119 with the same number of trees / epochs that the validation-tuned model converged on**. Discards the early-stopping signal but uses 50% more data on a regime that's *closer* to test.
- **Source:** Kaggle Grandmaster Playbook 2024; explicitly used by Optimal Fertilizers winner; AmbrosM 3rd-place "Don't trust the CV scores" writeup (Tabular Playground Nov 2021) — final retrain on full data > respecting the validation split.
- **Why disruptive:** dates 80–119 are the most temporally adjacent to the platform test period. Our current pipeline never trains on them. Late dates likely contain the most relevant regime.
- **Cost:** trivial — one extra fit per model with epochs/n_estimators frozen.
- **Expected platform delta:** **+1 to +3**. Small but free.
- **Risk:** if 80–119 contains a regime shift, we may overfit the new regime; mitigated by cap on epochs.

---

### M8. **Robust ranking / abstention via conformal prediction** (selective trading)

- **Description:** Layer a conformal prediction step over our class probabilities. For each (sample, horizon), compute a non-conformity score using a calibration set; trade only when the prediction set is a **singleton** (high-confidence label). For ambiguous predictions, output 1 (flat). Tune the significance level α to maximise platform PnL (single hyperparameter).
- **Source:** Conformal prediction canon (Vovk; Wikipedia); arXiv 2403.15025 "Robust Conformal Prediction under Distribution Shift" (2024) for the distribution-shift-aware variant; clinical-triage papers for cost-aware deferral.
- **Why disruptive:** our current threshold tuning is a soft version of this, but it's tuned on local LOSO and over-fit. Conformal gives us a **distribution-free coverage guarantee** in expectation, so the "abstain" rate is automatically calibrated to platform conditions. Specifically helps stabilise transmission ratio.
- **Cost:** 2 days. One calibration set, one significance level.
- **Expected platform delta:** **+1 to +3**, mostly via stability.
- **Risk:** under heavy distribution shift, even conformal can lose coverage; pair with M1 for AV-aware calibration.

---

### M9. **Submission-feedback loop: treat the platform as the only validation set** (process change)

- **Description:** Stop searching for the "best local LOSO model" and start running a **directed exploration** over paradigm space, where each platform submit gives one data point. After ~20–30 submits, fit a Bayesian linear model (model paradigm → platform score) and use it to choose the next submit. Pre-commit to which 5 hyperparameters / paradigm choices to vary; never re-tune ones not on the list.
- **Source:** Marcus Hutter's "Competing in a data science contest without reading the data" + Blum & Hardt "The Ladder: A Reliable Leaderboard for ML Competitions" (PMLR 2015) — formalises the leaderboard-as-oracle paradigm with budget bounds. Kaggle Grandmaster lore: top teams submit 5×/day; we do not.
- **Why disruptive:** we treat platform as truth and local as practice. Every team converges on local-overfit unless they invert this. Limited submit budget means we have to be smart, but that's exactly the principled way to use it.
- **Cost:** zero compute; medium project-management cost. Need to track every submit's paradigm and score in a clean spreadsheet.
- **Expected platform delta:** **+2 to +5** indirectly — mostly by *avoiding* the wasted +0 submits we currently make.
- **Risk:** finite submit budget. If we submit randomly we burn it. Use a Thompson-sampling-style policy over paradigms.

---

### M10. **External-data pretraining on FI-2010 / public LOB datasets** (transfer)

- **Description:** Pretrain a 100-tick LOB encoder on FI-2010 (Ntakaris et al. 2018, public, 5 NASDAQ Nordic stocks × 10 days; ~4M rows) using either (a) a self-supervised denoising-AE objective or (b) the FI-2010 native 3-class label (which has the same structure as ours: down/flat/up). Freeze early layers; fine-tune late layers on competition data.
- **Source:** Zhang et al. DeepLOB 2018 (Oxford-Man) explicitly demonstrates transfer across stocks and across markets. TLOB 2025 paper trains on FI-2010 + Tesla/Intel + BTC. Self-supervised denoising is well-established for finance (arXiv 2112.10139, "Denoised Labels for Financial Time-Series via SSL").
- **Why disruptive:** we have ~120 days × 5 stocks. The FI-2010-pretrained encoder has seen orders of magnitude more LOB shapes. Transfer learning has been a 2-3× lift in similar settings.
- **Cost:** 3–4 days; download + retrain. Network restriction note: use academic-mirror download (`bash -c 'source /etc/network_turbo && ...'`).
- **Expected platform delta:** **+2 to +6** if combined with M3 (SAE) or M4 (TLOB).
- **Risk:** FI-2010 stocks are Nordic mid-caps and our 5 stocks are unknown — tick-size / liquidity mismatch. Mitigated by using only the SSL pretrain (no labels), so only invariant structure transfers.

---

### M11. **Data-side exploit hunt: sym-of-known-stock decoding** (creative attack — last-resort)

- **Description:** §1.3 says sym 0–4 may not be the same 5 stocks as training. But what if we can **decode** at inference time which "training stock" the test sym most resembles, then route through a sym-conditioned head? This violates the literal §1.3 if implemented as embedding, but is allowed if implemented as **soft routing**: compute distance in feature space to each training-sym centroid; weight predictions accordingly. Test stock genuinely novel → uniform weights → falls back to global model. This is the analogue of Optiver Realized Vol's Hamiltonian-path trick: a data-level exploit that respects the rules.
- **Source:** Optiver Realized Vol 2021 1st-place writeup (Hamiltonian path on time_id); DeepLOB transfer-learning argument (universal LOB features per asset class).
- **Why disruptive:** §1.3's wording rules out *hard* sym-specific models, but **soft routing is not forbidden**. If 80% of platform test data is one of our 5 training stocks, this trick gives us per-sym precision; if 0%, it gracefully degrades.
- **Cost:** 2 days.
- **Expected platform delta:** **+1 to +5**, depending on platform sym distribution.
- **Risk:** §1.3 contains "禁止 per-sym 的 normalization 统计量"; soft routing on prediction is *not* per-sym normalisation but the line is grey. Get explicit clearance before submitting. **STRICTLY check CRITICAL_CONSTRAINTS.md re-read before implementing.**

---

## Part 3 — Top-3 recommendations

The brief is to pick **paradigm-shift-level** moves, not incremental tricks. The +12 gap will not yield to +1 ideas. Top-3 by expected platform impact and ROI:

### 🥇 #1 — **M6 (paradigm portfolio) + M9 (submission-feedback loop)** as the *meta-process change*

**One-line rationale:** the gap is process, not artefact. Stop running 1 stack deep; run 4 stacks shallow and let the platform tell us which one wins.

**Concrete sequencing (2-week plan):**
1. Day 1: Implement M9 instrumentation — every submit logged with paradigm tag + score + 1-line hypothesis. Pre-register 4 paradigms.
2. Days 2–3: Stand up paradigm A (current LGB iter016) baseline locked. No more changes.
3. Days 4–7: Build paradigm B (M3 = supervised AE+MLP) — 1 worker.
4. Days 4–7: Build paradigm C (M5 = direct PnL loss) — 1 worker, parallel.
5. Days 8–10: Build paradigm D (M4 = TLOB or MLPLOB, no pretrain first) — 1 worker, parallel.
6. Days 11–13: Submit each paradigm individually + 4 simple weighted blends; observe.
7. Day 14: Hill-climb ensemble fit on those ~12 platform points; pick final.

**Expected delta:** **+5 to +10**.

**Why this not M3/M4/M5 individually:** any *single* paradigm is +2 to +5; the **diversity bonus** between heterogeneous paradigms is what closes the +12 gap. Numerai pays for orthogonal alpha for a reason.

---

### 🥈 #2 — **M1 (adversarial-validation-weighted CV) + M2 (50-seed bag) + M7 (final retrain on 0–119)** as the *validation rebuild*

**One-line rationale:** if our local LOSO is over-fit, none of M3/M4/M5 can be trusted. Fix the validation first, in parallel.

**Concrete sequencing (1 week, can run during week 1 of M6):**
1. Day 1: Build adversarial classifier (train vs late-date holdout); compute per-row "platform-likeness" weight.
2. Day 2: Refit baseline with AV weights + AV-stratified validation; compare local-vs-platform transmission ratio (target: > 0.75).
3. Day 3: 50-seed LightGBM bag; replace Optuna threshold search with isotonic on raw probs.
4. Day 4: Final retrain on dates 0–119 with epochs frozen.
5. Days 5–7: Replace **all** our hyperparameter choices with AV-weighted reselection.

**Expected delta:** **+3 to +6** standalone; multiplicative with M6 because better validation makes paradigm portfolio selection *correct*.

**Why this not M9 alone:** M9 is a meta-process, M1+M2+M7 are the technical fixes that make every submit count. They are complementary: M9 says "use platform as oracle"; M1 says "build local proxy that doesn't lie".

---

### 🥉 #3 — **M3 (supervised AE+MLP)** as the *single highest-EV new paradigm*

**One-line rationale:** if we had to pick ONE new model paradigm to add, this is it. Empirically the most robust paradigm for noisy tabular financial data (Jane Street 2021 1st place; Springer JBD 2025), low implementation cost, near-mechanical fit to our setup.

**Concrete sequencing (3–4 days):**
1. Day 1: Flatten our 100×D window to a vector; build encoder (2-layer MLP, 256→128 bottleneck).
2. Day 2: Decoder branch + joint loss (α=1.0 CE + β=0.3 MSE).
3. Day 3: Train 5 LOSO folds × 5 seeds = 25 models; bag.
4. Day 4: Stack as a member of M6 paradigm portfolio.

**Expected delta:** **+2 to +5** standalone; multiplies with M6.

**Why this over M4 (TLOB/MLPLOB):** TLOB has higher upside but higher implementation risk and 5 days budget; SAE+MLP is "boring win" — proven paradigm with cheap implementation. Run M3 first, M4 if M3 succeeds.

---

### Top-3 summary

```
╔════════════════════════════════════════════════════════════════════════╗
║  Top 3 (in execution order):                                           ║
║                                                                        ║
║  1. (Meta-process)  M6 + M9  — paradigm portfolio + submit-feedback    ║
║  2. (Validation)    M1 + M2 + M7 — AV-weighted CV + bag + full retrain ║
║  3. (New paradigm)  M3  — supervised autoencoder + MLP                 ║
║                                                                        ║
║  Combined platform delta: +8 to +18 (vs current +28 → target +36–46)  ║
╚════════════════════════════════════════════════════════════════════════╝
```

**Sequencing note:** #2 should run **first** (1 week). #1 should be the meta-frame for #2 and #3. #3 is the most concrete sub-stream of #1.

---

## Part 4 — Cross-check against PROGRESS.md

### 4.1 What PROGRESS.md confirms about the problem framing

PROGRESS.md is the **initial-state scoping document** (§1–§7), authored before iteration started. It does *not* contain the current iter-016 state, the +28 vs +40 gap, or the list of tried tricks (those live in `experiments/` and `research_and_history/`, which I am forbidden to read). So the cross-check is partial — I can verify framing, constraints, priorities, and the submission-budget reality, but not whether specific tricks have already been attempted.

**Confirmed alignments:**

1. **§6.7 ("每 12h 一次提交——本地 OOF/CV 必须能稳定预估线上分") is the exact problem.** Our brief says transmission ratio is 0.4–0.7 and unstable. That ratio failing is the central pathology PROGRESS already flagged as the top risk. **M1 (adversarial-validation-weighted CV) and M9 (submission-feedback loop) directly target this** — they are not new ideas but the discipline PROGRESS asked for, finally implemented.

2. **§7.D explicitly listed "DeepLOB 后续工作（TransLOB、TabLOB、N-BEATS 等高频文献）" as a candidate path.** My **M4 (TLOB / MLPLOB)** is exactly the 2025 successor of TransLOB — same lineage, same FI-2010 benchmark, dual-attention upgrade. So M4 is *aligned with the original D-priority*, not orthogonal to plan.

3. **§7.B ("写一个本地 PnL 评测器") is presumably already done.** My **M5 (direct PnL / Sharpe loss)** extends it from evaluation to **training objective** — a strict superset.

4. **§6.4 ("初期建议优先打 label_60") is consistent with my M5.** label_60 has α=0.1%, 24% non-flat, best after-fee economics. M5 with horizon-selection logit naturally exploits this; should be biased toward label_60 by the loss landscape.

5. **§6.2 + §6.3 ("测试集可能有训练外股票", "PnL > F0.5 > acc, precision 远比 recall 重要")** — both my M8 (conformal selective prediction = abstain when uncertain) and M11 (soft sym-routing) are precision-first by construction. M11 is the riskier of the two; it sits exactly on the line of §6.2 (do not overfit single sym) and CRITICAL_CONSTRAINTS §1.3 (no per-sym normalisation). Mark it lower priority pending explicit clearance.

### 4.2 ⚠ Critical timing finding (overrides Part 3 priorities)

PROGRESS.md §1 timeline: **初赛提交截止 = 2026-05-11**. Today is **2026-05-08**. **Only 3 days remain**, with 12-h submit cadence (~6 submits left).

This dramatically rebalances the Top-3:

| Rec | Original timeline | Feasible by 5-11? |
|---|---|---|
| M1 (AV-weighted CV) | 1–2 days | ✅ feasible |
| M2 (50-seed bag) | < 1 day | ✅ feasible |
| M7 (full-data retrain) | trivial | ✅ feasible |
| M3 (SAE+MLP) | 3–4 days | ⚠ barely; only if 1 worker dedicated |
| M4 (TLOB/MLPLOB) | 3–5 days | ❌ too late for finals; possible for 决赛答辩 (5-30) defense |
| M6 (paradigm portfolio) | 2 weeks | ❌ too late in full form |
| M9 (submit-feedback) | process | ⚠ only ~6 submits remaining |
| M10 (FI-2010 pretrain) | 3–4 days | ❌ too late |
| M11 (soft sym-routing) | 2 days | ⚠ feasible but constraint-risky |

### 4.3 Revised final recommendation under 3-day budget

```
╔════════════════════════════════════════════════════════════════════╗
║  REVISED Top-3 under 3-day deadline (5-11 submission cutoff)       ║
║                                                                    ║
║  TODAY (5-08, evening) ──────────────────────────────────────────  ║
║    ▸ M1: build adversarial-val classifier (4 hr)                   ║
║    ▸ M2: kick off 50-seed LightGBM bag (background, GPU)           ║
║                                                                    ║
║  5-09 ───────────────────────────────────────────────────────────  ║
║    ▸ M1: refit baseline with AV weights; verify TR > 0.75          ║
║    ▸ M2: complete; isotonic calibration on raw probs               ║
║    ▸ M7: final retrain on dates 0-119, epochs frozen               ║
║    ▸ Submit #1 (M1+M2+M7 stack) — measure platform delta           ║
║                                                                    ║
║  5-10 ───────────────────────────────────────────────────────────  ║
║    ▸ Submit #2 (best of M1+M2+M7 + ablations)                      ║
║    ▸ M3 (SAE+MLP) prototype if time — diagnostic only              ║
║    ▸ M11 (soft sym-routing) IF cleared on constraints              ║
║                                                                    ║
║  5-11 ───────────────────────────────────────────────────────────  ║
║    ▸ Final submit: best paradigm-blend from M1/M2/M7/M3            ║
║    ▸ Public-LB closes; private set evaluated 5-18                  ║
║                                                                    ║
║  FOR 决赛答辩 (5-30, post-public-LB): full M3+M4+M6 portfolio       ║
╚════════════════════════════════════════════════════════════════════╝
```

### 4.4 What changes vs the original Top-3

- **M6 (paradigm portfolio) deferred to defence prep**, not 5-11 push.
- **M9 (submit-feedback loop) collapsed** to 1 disciplined experiment per remaining submit slot.
- **Validation rebuild (M1+M2+M7) promoted** to sole-priority for the final 3 days. Highest ROI in the time budget. The +12 gap is most likely a *validation defect*, and validation defects can collapse in a single submit if the rebuild is correct.
- **M3 (SAE+MLP) becomes a stretch goal**, not a primary lever for 5-11.

### 4.5 Constraint cross-check on every recommendation

| Rec | §1.1 (no date) | §1.2 (no cross-state) | §1.3 (sym-agnostic) | Verdict |
|---|---|---|---|---|
| M1 AV CV | ✓ | ✓ | ✓ | safe |
| M2 seed bag | ✓ | ✓ | ✓ | safe |
| M3 SAE+MLP | ✓ | ✓ | ✓ if no sym embed | safe |
| M4 TLOB/MLPLOB | ✓ | ✓ | ✓ (transformer is sym-blind) | safe |
| M5 PnL loss | ✓ | ✓ | ✓ | safe |
| M6 portfolio | inherits | inherits | inherits | safe (per-stack) |
| M7 full retrain | ✓ | ✓ | ✓ | safe |
| M8 conformal | ✓ | ✓ if calib-set is sym-pooled | ✓ | safe |
| M9 submit-loop | n/a | n/a | n/a | safe (process) |
| M10 FI-2010 pretrain | ✓ | ✓ | ✓ (pretrain is asset-agnostic) | safe |
| M11 soft sym-routing | ✓ | ✓ | ⚠ grey — needs PM clearance | hold |

All recommendations respect the three hard constraints with the single exception of M11 (flagged hold).

### 4.6 One thing PROGRESS.md does NOT have that I would add

PROGRESS.md §7 listed candidate paths but did not include a **"validation health check" milestone**. Given that we are 3 days out and the gap is +12, the very first thing tomorrow (5-09) should be: compute adversarial-validation AUC between training data and the 80-119 holdout, and between training data and our local 442k test split. If both AUCs are far from 0.5, we have proof of distribution shift; if only the first is far, our local test split is misrepresentative; if neither is, the +12 gap is from something else (model capacity / random seed / direct exploit). This 1-hour diagnostic should drive every decision afterward.

---

## Sources

### Primary winning-solution writeups
- [Optiver Trading at the Close — 1st place writeup (HYD)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
- [Optiver Realized Volatility — 1st place (Nearest Neighbors / Hamiltonian path)](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- [Jane Street Market Prediction — 1st place (Yirun, Supervised Autoencoder + MLP)](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- [Jane Street Real-Time Market Data Forecasting (2024–25)](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [G-Research Crypto Forecasting wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- [JPX Tokyo Stock Exchange Prediction analytical models](https://github.com/J-Quants/JPXTokyoStockExchangePrediction)

### LOB deep learning
- [TLOB (arXiv 2502.15757, Feb 2025) — dual-attention transformer SoTA on FI-2010 / Tesla-Intel / BTC](https://arxiv.org/abs/2502.15757)
- [TLOB official repo](https://github.com/LeonardoBerti00/TLOB)
- [DeepLOB (arXiv 1808.03668, Zhang et al. 2018) — transfer learning to unseen instruments](https://arxiv.org/abs/1808.03668)
- [LOBCAST benchmark study (Springer AI Review 2024)](https://link.springer.com/article/10.1007/s10462-024-10715-4)
- [LOB-Based DL Models for Stock Price Trend Prediction (arXiv 2308.01915)](https://arxiv.org/abs/2308.01915)

### Validation / shake-up / ensembling
- [Trust Your CV — Kaggle Handbook (Demir 2022)](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)
- [The Kaggle Grandmasters Playbook (NVIDIA 2024) — multi-seed bagging, hill-climb ensemble](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Adversarial Validation (zakjost)](https://blog.zakjost.com/post/adversarial_validation/)
- [Adversarial cross-validation (varunbpatil 2018)](https://varunbpatil.github.io/2018/11/06/adversarial-cv.html)
- [The Ladder: A Reliable Leaderboard for ML Competitions (Blum & Hardt, PMLR 2015)](http://proceedings.mlr.press/v37/blum15.pdf)
- [Climbing the Kaggle Leaderboard (arXiv 1707.01825)](https://arxiv.org/abs/1707.01825)

### Loss / objective alignment
- [Finance-Grounded Optimization for Algorithmic Trading (arXiv 2509.04541, Khubiyev 2025)](https://arxiv.org/abs/2509.04541)
- [Decision by Supervised Learning with Deep Ensembles (arXiv 2503.13544)](https://arxiv.org/html/2503.13544v3)

### Self-supervised / autoencoder
- [Supervised Autoencoder MLP for Financial Time Series Forecasting (Springer JBD 2025)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01267-7)
- [Denoised Labels for Financial Time-Series via SSL (arXiv 2112.10139)](https://arxiv.org/abs/2112.10139)

### Conformal / robustness
- [Robust Conformal Prediction under Distribution Shift (arXiv 2403.15025)](https://arxiv.org/abs/2403.15025)
- [Conformal prediction (Wikipedia overview)](https://en.wikipedia.org/wiki/Conformal_prediction)

### Numerai / orthogonal alpha
- [Numerai Meta Model Contribution docs](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc)
- [Numerai Signals Alpha & MPC blog](https://blog.numer.ai/signals-alpha-and-mpc/)
