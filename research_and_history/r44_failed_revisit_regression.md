# R44 — Failed-experiment audit under the new "regression on Δmid + EV gate" regime

> Created 2026-05-08 by R44 worker.
>
> Context: T75 replaced 3-class CE + DE 4D probability gate with **L2 regression on
> Δmid_norm = (mp_t60 − mp_t)/(mp_t+1) + asymmetric EV gate**. Result was +9.79 LOSO-equiv
> (iter_012 +26.44 → iter_013 +36.23). T81 transplanted the same target/decision to a
> simple MLP and got +35.89 alone; NN+LGB ensemble = iter_014 +38.28.
>
> The breakthrough was not the model — it was the **target/decision combination**. Many
> earlier "failed" experiments were **CE-bound**: they died because 3-class labelling
> destroyed the magnitude information they tried to exploit, *or* because the only
> downstream lever (4-param probability gate) couldn't translate their improvement into
> PnL. A subset of those failures should now be revived under regression.

This document classifies every prior negative result, decides which are revivable
under regression, and ships a top-5 implementation plan.

---

## 0. Mental model: when does CE-→-regression revive an experiment?

A failed CE experiment is **revivable under regression** when *at least one* of the
following held under CE but no longer holds under L2-on-Δmid + EV gate:

| Failure pathway under CE | Why regression unblocks it |
|---|---|
| **F-A. Magnitude collapse** — the experiment's signal is in |Δmid|, but 3-class labelling collapsed Δmid > τ to a single class so the model never saw it | L2 target is signed Δmid_norm directly; magnitude is preserved |
| **F-B. Decision bottleneck** — the experiment improved p̂ in a way the 4-param (T_up, T_dn, δ_up, δ_dn) gate could not exploit (e.g. better tail behaviour) | EV gate is monotone in pred; any tail improvement directly buys threshold-induced PnL |
| **F-C. Sample-weight × CE non-alignment** — sample weight in CE applies to a softmax-target whose loss does not relate to PnL contribution | Sample weight in L2 directly multiplies loss = (Δmid)² → upweighted samples literally drive the regression toward larger |Δmid| residuals, which is also where PnL is made |
| **F-D. Class-imbalance → degenerate p̂** — flat class is huge; classifier collapses to "predict 1" | Regression has no "majority class"; the L2 loss gradient is balanced across signed Δmid without any class-prior degeneracy |
| **F-E. Pseudo-/synthetic-label thresholding** — pseudo-labels were quantised to {0, 1, 2}, throwing away certainty | Pseudo target is the continuous Δmid prediction itself — no threshold needed |
| **F-F. False-signal feature** — feature got top-rank gain under CE because it correlated with the threshold τ noise, not actual move | Regression target = actual move → if the feature is a true signal it shows up; if it's a τ-noise artefact, regression rejects it |

A failure is **NOT revivable** when:

- The idea is fundamentally about classifier-only objects (focal loss, isotonic
  calibration of softmax, prob-space mixup, OvA decomposition, …) — they have no L2
  analogue or one already covered by T75/T81's design choices.
- The idea died because of **hard constraints** (sym embedding, per-sym norm, online
  retraining, Bidirectional encoder, etc.) — those constraints still apply.
- The idea was confirmed dead in **both regimes** (e.g. T74 time features failed under
  CE; T78 retried under regression and confirmed -0.72 — final dead).
- The signal was already squeezed by iter_007–iter_014 base model (i.e. the residual
  available is < 0.5 LOSO-equiv per the noise floor seen across T63/T67/T73/T76).

---

## 1. Per-experiment revival audit

The list is grouped by failure pathway. "Δ@iter_014" = expected LOSO-equiv gain over
iter_014 (+38.28). "Cost" is wall-clock + GPU hours for one disciplined re-run.

### 1.1 Loss / sample-weight ideas

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| **T57 PnL-aware sample weight** | CE with `weight = clip(\|Δp\|, 99th)` linear / sqrt / cap | F-A + F-C: weight upweights big-|Δp| samples but CE already collapsed magnitude info — weight has nothing to grip on | **Yes (HIGH)** — under L2 the weight directly multiplies the (Δmid)² residual; T75 only used class-balanced weight, not magnitude weight | **+0.5 ~ +2.0** | 0.5 day | ✅ Revive |
| T41 class-weight + focal | CE + 3-class weights (2:1:2 etc.) | F-D: re-balancing fixes degeneracy but discards EV ordering | NO direct map — but covered by T57 magnitude weighting | n/a | n/a | ❌ Subsumed by T57 |
| **S2 LightGBM custom multiclass PnL objective** (R10) | Never built (idea-only) | The Volkova/HYD top-Kaggle precedent showed weighted MSE outperforms custom utility loss | Already done effectively by T75 (regression L2 target = the PnL numerator); custom objective adds little | +0.0 ~ +0.5 | 1.5 days | ❌ Skip |
| T60 NN with direct PnL loss | NN, expected-PnL surrogate as primary loss | F-A + B: 3-class action softmax ⊗ Δp wasn't passing useful gradient to the encoder; CE warm-start was drowning the PnL term | **Partially revived** — T81 var B already tried `regr L2 + λ·EV-PnL aux`, λ=0.3 gave no lift because PnL term ≪ L2 term in scaled units | **+0 ~ +1** if very large λ tuned | 1 day | 🟡 Marginal — try once |

### 1.2 Cross-sym OOD ideas

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| **T45 Group DRO** | LightGBM with multinomial + dynamic sym sample-reweighting | F-C: DRO upweights high-loss sym → CE-loss-high sym → not the same as PnL-loss-high sym | **Yes (MED-HIGH)** — under L2 the per-sym MSE *is* the per-sym squared-Δmid loss, much closer to per-sym PnL contribution | **+0.5 ~ +2.5** | 1.5 days | ✅ Revive |
| T66 Adversarial validation reweighting | LightGBM CE with sample weight = is_test posterior | F-C: AUC=1.0 confirms hard separable train/test, but reweighting under CE biased the classifier toward whatever slice CE happens to find easy in test-like rows | **Yes (MED)** — under L2, `weight = is_test_posterior` directly biases the regressor toward minimizing residual on the slice that resembles the platform; works the same way as T57 magnitude weight stacking | **+0.3 ~ +1.5** (weak alone, may stack with T57) | 0.5 day | 🟡 Try after T57 |
| T11 Cross-Sym Mixup | Yirun JS 2021 mixup — implemented under CE | Mixup of two 3-class labels = soft label, but information-blurred | **Yes (MED)** — mixup of two regression targets = `λ·y₁ + (1-λ)·y₂` is mathematically clean (linear interpolation in target space, exactly what mixup is designed for) | **+0.5 ~ +2** | 0.5 day | 🟡 Try after T57 |

### 1.3 Alternative learners

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| **T46 CatBoost (Plain, h_60, 5-seed)** | CatBoost MultiClass, +12.64 LOSO h=60 (vs LGB +13.61) | F-D: CatBoost MultiClass tends to underfit class-imbalance; ordered boosting was unavailable on GPU | **Yes (HIGHEST)** — CatBoost **regression on MAE/RMSE is the algorithm that won Optiver Trading-at-Close 2023 (1st place HYD)**, with 50% weight in their winning ensemble. We currently have LGB+NN; adding CatBoost-regression as a 3rd diverse base is the textbook move | **+1.0 ~ +3.0** | 1.5 days (GPU CatBoost training works on this hardware per CLAUDE.md) | ✅✅ TOP-1 revive |
| **T56 XGBoost stage-3 5-seed** | XGBoost multi:softprob, +12.45 LOSO (DE +16.45 in T56's own metric) | F-D + algorithm parity: XGB classifier ≈ LGB classifier on this tabular data | **Yes (MED-HIGH)** — XGBoost **regression with `tree_method=hist, device=cuda`** has a different split policy than LightGBM (level-wise vs leaf-wise + gradient histogram) → produces less-correlated residuals → ensemble diversity. Cheap to train (1h) | **+0.3 ~ +1.5** stacked with LGB+NN+CB | 0.5 day | ✅ Revive (low effort) |
| T49 CatBoost Ordered | Aborted (GPU unsupported) | infrastructure | Same constraint applies to regression | n/a | n/a | ❌ Skip |
| T50 LightGBM DART | Aborted (CPU too slow) | infrastructure | DART regression doesn't fix the drop-correction asymmetry that hurt under CE | n/a | n/a | ❌ Skip |

### 1.4 Feature work

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| **T22 alpha101 + alpha191** | 33 WQ + 124 GTJA alphas, tested at **h=10 only**, single seed, CE | F-A + B: alphas designed for **regression-style time-series targets** (signed return horizons), but T22 forced them through a 3-class h=10 head that loses both magnitude AND the longer-horizon dependence the alphas encode | **Yes (HIGH)** — alphas are inherently regression features (Brière/Sornette factor literature). At h=60 (the platform horizon) and L2-on-Δmid the alphas should actually fit their original use case | **+1 ~ +3** | 1 day (T22 cache `schemeI_*.npz` 247-d already on disk; only need to retrain regression heads) | ✅ Revive |
| **T35 ReVol + Savgol** | Scheme K = base + 12 ReVol (Lee 2025) + 20 Savgol features under CE | F-F: feature_importance #1 was `revol_wmp1_sigma_hat` at **28% gain share**, but PnL got worse → ReVol features had high CE-gain because they correlated with the **τ-thresholding noise** in the 3-class label, not with actual Δmid | **Yes (HIGH)** — ReVol's per-sample `(μ̂, σ̂)` causal log-return normalization is **mathematically the same family** as our regression target (`Δmid / (mp+1)` ≈ standardized log-return). Under L2 a true ReVol signal will lift `pred_Δmid_norm`; a τ-noise artefact will be rejected by the L2 loss | **+0.5 ~ +2.5** | 0.5 day (T35 cache `schemeK_extra_*.npz` 32-d already on disk; same target/training as T75) | ✅ Revive |
| T55 Stage 4 features | 5 stage-4 LOB features (cancel-pressure-delta etc.) under CE | F-F (mild): -1.11 vs iter_011 — features were noise-correlated with CE label | **Maybe (LOW-MED)** — same logic as T35, but T55 was small-magnitude (5 features) and the CE drop was small; signal-to-noise ratio low | +0.0 ~ +1.0 | 0.3 day | 🟡 Cheap retry |
| T74 time-cyclic features | sin/cos minute, sin/cos session-progress under CE | F-irrelevant — **already retried in T78 under regression: -0.72** | Confirmed dead | -0.5 ~ -1.0 | n/a | ❌ Skip — confirmed dead |

### 1.5 Half-supervised / pseudo-labeling

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| **T65 Pseudo-labeling** | Gate by `max(p₀, p₂) > 0.55, |p₀-p₂| > 0.10` → quantise to {0, 2}; sample_weight=0.5; -0.61 vs iter_010 | F-E: pseudo accuracy 48% because **30% of high-confidence directional rows were labeled flat** by the τ thresholding — quantising to {0,2} discarded the ‘medium-Δmid’ samples | **Yes (MED-HIGH)** — under regression the pseudo target is the **continuous predicted Δmid_norm** itself; no thresholding is needed, no info loss. Self-distillation on regression target is a well-established trick (Hinton 2015 in spirit) | **+0.3 ~ +1.5** | 1 day | ✅ Revive |

### 1.6 Architecture / NN ideas

| Exp | What CE version did | Failure root cause | Regression revival? | Δ@iter_014 | Cost | Verdict |
|---|---|---|---|---|---|---|
| T1 NN baseline (DeepLOB CNN) | DeepLOB CNN + 3-class CE | F-A + F-D | **Already revived in T81** (simple MLP regression got +35.89) | n/a | n/a | ✅ Done |
| T19 regularized MLP | MLP + 3-class CE, mid-20s | F-A | **Already revived in T81** (same MLP family, now +35.89) | n/a | n/a | ✅ Done |
| T32 DeepLOB h60-only | DeepLOB CNN, h60, CE | F-A | T81 already covers; DeepLOB regression specifically untested but the bottleneck is target choice, not arch | +0 ~ +1 | 1 day | 🟡 Low priority |
| T33 long window | Window > 100 ticks under CE | F-A — long window helps regression more than CE, but feature cache is 100-tick fixed | Possible but feature pipeline rebuild needed | +0 ~ +1 | 2 days | ❌ Skip (high cost) |
| T1/T19/T32 other | various | F-A | Subsumed by T81 | n/a | n/a | ❌ Skip |

### 1.7 Confirmed-dead-in-both-regimes (do not revive)

T63 reg sweep, T64 V2 PnL-metric val, T67 multi-split bagging K=20 (+0.35), T69 V4 +
full data 0–95 (+0.02), T72 binary cascade, T73 best subset of seeds (+0.07), T76
decoupled thresh per intraday bucket, T77 multi-horizon vote ensemble (+32.42 < +36.23),
T82 repeat T81. Same with all "already-covered-by-T75/T81" ideas (S1/S2 from R10, S6
regression on signed Δp from R10 — that **is** what T75 did).

---

## 2. Quantitative summary

**Number of distinct failed experiments reviewed:** 22 (T1, T11, T19, T22, T32, T33,
T35, T41, T45, T46, T49, T50, T55, T56, T57, T60, T63, T65, T66, T67, T74, T82).

| Bucket | Count |
|---|---|
| Already revived in T75/T81 | 3 (T1, T19, T32 ⇒ T81 covers; T75 subsumed S2/S6/regr ideas) |
| **Revivable under regression (this report's recommendations)** | **5 → top picks below** |
| Marginal / cheap-to-try secondary | 4 (T11 mixup, T55 stage-4 features, T56 XGBoost, T66 adversarial) |
| Confirmed dead in both regimes | 7 (T63, T64, T67, T69, T72, T73, T74→T78, T76, T77, T82) |
| Infrastructure-blocked (not loss issue) | 2 (T49, T50) |

**Total revivable inventory:** 5 high-priority + 4 secondary = **9 candidates**.
Conservative aggregate expected gain over iter_014 if all 5 top picks ship and stack
non-trivially: **+2.5 ~ +6.0 LOSO-equiv → platform 0.70× ≈ +1.8 ~ +4.2**.

---

## 3. Top-5 implementation plan

### Pick 1 (TOP) — **T46-revival: CatBoost-regression on Δmid_norm**

- **Why now:** Optiver 2023 1st place (HYD) used CatBoost-regression (MAE) at 50%
  weight in winning ensemble. We have LGB-regression + MLP-regression. CatBoost-
  regression is the obvious 3rd diverse base. Cross-corr LGB↔NN is 0.77; CatBoost-
  regression at h=60 typically corr 0.65–0.75 with LGB (different boosting +
  symmetric tree split policy).
- **Recipe:**
  1. Reuse T68 schemeP cache (359-d feature subset, drop the 11 KS-fail extras).
  2. Train 5-seed CatBoost regression on `y = (mp_t60 − mp_t)/(mp_t+1)`.
     ```python
     params = dict(
         loss_function='RMSE',          # try MAE as variant
         iterations=2000,
         learning_rate=0.03,
         depth=8,
         l2_leaf_reg=3.0,
         border_count=128,
         task_type='GPU',
         devices='0',
         od_type='Iter', od_wait=50,
         random_seed=seed,
     )
     # weight = class-balanced per T75 — keep identical to T75 sample-weight
     pool_train = Pool(X_train, y_train, weight=w_train_cb_balanced)
     pool_val   = Pool(X_val,   y_val)
     model.fit(pool_train, eval_set=pool_val, verbose=100, use_best_model=True)
     ```
  3. Verify single-seed CatBoost LOSO-equiv ≥ +28 (so it's at least within 8 of LGB).
  4. Re-run NN+LGB+CB DE asymmetric weight sweep + threshold tuning.
- **Risk gates:** if any single-seed CB LOSO-equiv < +25, abort (CB regression doesn't
  beat its own classification baseline → wasted ensemble slot).
- **Expected:** +1.0 ~ +3.0 LOSO-equiv at iter_014 → **iter_015 candidate.**
- **Cost:** 1.5 days. Two GPU training runs (5 seeds × 2 loss flavours: RMSE, MAE).
  Inference: CatBoost CPU-mode model `.cbm` → bundle in submission, recipe identical
  to LGB inference. CatBoost-Python is in std requirements list (already installed).
- **Hard-constraint check:** CB regression model takes 359-d feature vector, no
  sym/date/time → CRITICAL_CONSTRAINTS §3 ✅. No per-sym normalization → ✅. Stateless
  predict → ✅.

### Pick 2 — **T22-revival: Alpha101+Alpha191 alphas at h=60 under regression**

- **Why now:** T22 already implemented 33 WQ + 124 GTJA alphas (157 features) and
  built `schemeI_test/train/val.npz` cache, but only tested at **h=10 + CE + single
  seed**. The full 247-d "Scheme Ic" cache exists on disk
  (`experiments/T22_alpha101_alpha191/cache/schemeI_*.npz`). Most alphas are
  designed for regression-style return prediction (Brière/Sornette/GTJA factor lit),
  not 3-class classification.
- **Recipe:**
  1. Build a *Scheme P + alpha* feature set: 359-d schemeP ⊕ top-30 alphas by gain
     (under regression L2). Alpha selection done via single-fold gain on a quick
     400-iter LGB regression.
  2. Train T75-style 5-seed LGB regression on the unified ~389-d feature space.
  3. Run NN+LGB(P) + LGB(P+alpha) ensemble. The added alpha-loaded LGB is the
     diversifier.
  4. Compare `iter_014 +38.28` vs `iter_014 + alpha-LGB`.
- **Risk gates:** if alpha-LGB LOSO-equiv stand-alone ≤ +33 (vs T75 LGB +36.23),
  abort — alphas not adding signal under regression either.
- **Expected:** +1 ~ +3 LOSO-equiv (high uncertainty: alphas may already be
  redundant after the rich Stage 1–5 feature work in iter_007–012).
- **Cost:** 1 day. Cache is built; only training + ensemble integration.
- **Hard-constraint check:** All alphas computed per-sym, per-window from raw LOB only
  → no sym/date dependence → ✅.

### Pick 3 — **T35-revival: ReVol per-sample log-return normalization features under regression**

- **Why now:** T35 found `revol_wmp1_sigma_hat` was the **#1 LightGBM feature by gain
  share (28%)** but adding it under CE *worsened* OOF PnL by -0.77. Diagnosis: under
  CE the model fit the threshold-noise via ReVol features (F-F false signal). Under
  regression on Δmid_norm, ReVol's `(μ̂, σ̂, ε)` per-sample log-return triplet is in
  the **same standardization family** as the target itself, so the correct way to use
  ReVol is *as a regression input*, not a classification input.
- **Recipe:**
  1. Reuse `experiments/T35_revol_savgol/cache/schemeK_extra_*.npz` (32-d ReVol+SG).
  2. Concatenate to schemeP (359-d) → 391-d feature space.
  3. Train T75-style 5-seed LGB regression. Compare per-fold MSE and per-fold cum_pnl
     vs T75 baseline.
  4. If LGB-regression with ReVol beats T75 (+36.23 → +37 territory), bundle into
     iter_015 candidate.
- **Risk gates:** if test_corr drops vs T75 (test_corr 0.153) → abort, ReVol again
  fitting noise.
- **Expected:** +0.5 ~ +2.5. There's a real chance ReVol is redundant w/ T44 R34
  Stage 1's "dual zscore" but the trade-off is 0.5 day, worth the experiment.
- **Cost:** 0.5 day. Cache exists; only retraining + EV gate evaluation.
- **Hard-constraint check:** ReVol uses W=64 causal rolling stats from each row's
  past, no sym/date/time → ✅.

### Pick 4 — **T57-revival: PnL-magnitude sample weighting under regression L2**

- **Why now:** T75 currently uses **class-balanced** sample weight (binary up/down
  upweighted vs flat). The R10 paper survey shows that under weighted-MSE/MAE
  regression (the Volkova JS 2024 / HYD Optiver 2023 1st recipe), the optimal
  sample weight is **proportional to |target|** not class-balanced. T57 tried that
  weighting under CE and got -3 ~ -11 because CE doesn't propagate magnitude.
- **Recipe:**
  ```python
  # T75 currently:
  w_T75 = class_balanced_weight(label_3class)                  # 3 distinct values
  # Replacement to test:
  w_R44 = np.clip(np.abs(y_train), 0, np.quantile(np.abs(y_train), 0.99))
  w_R44 = w_R44 / w_R44.mean()                                 # mean=1.0
  w_T57_R44_blend = 0.5 * w_T75 + 0.5 * w_R44                  # blend or pure
  ```
  Try three variants:
  - V1: pure |Δmid| weight, capped at 99th
  - V2: sqrt(|Δmid|) weight (less aggressive)
  - V3: linear blend with class-balanced (T75 default)
- **Risk gates:** if any of V1/V2/V3 single-seed LOSO-equiv drops ≥ -1.0 vs T75
  baseline (+30.20 mean over 5 seeds), abort — magnitude-weighting hurting at
  inference even though target is aligned (means model is overfitting big-Δmid
  outliers).
- **Expected:** +0.5 ~ +2.0. Papers all consistently report +5%-25% return
  improvement from magnitude weighting in regression. Our headroom is bounded
  because T75 already captures the magnitude signal in the *target*; this lever
  only sharpens fitting prio not feature selection.
- **Cost:** 0.5 day. Pure code-side change to `train_regr.py`'s `lgb.Dataset(weight=)`
  argument. Same model artifact size.
- **Hard-constraint check:** weight is derived from training-time labels only, no
  sym/date dependency. ✅.

### Pick 5 — **T65-revival: Self-distilled pseudo-targets under regression**

- **Why now:** T65 lost -0.61 under CE because gating to {0,2} **discarded
  ~30% of high-confidence directional rows that were actually labeled flat** by
  τ-thresholding. Under regression there's no thresholding step — the pseudo
  target is the continuous Δmid_norm prediction itself. This is exactly the
  knowledge-distillation regime where teacher-soft-targets > teacher-hard-labels
  (Hinton 2015).
- **Recipe:**
  1. Get iter_014 (NN+LGB ensemble) prediction on test rows: `ŷ_pseudo = pred_iter014`.
  2. Append `(X_test_rows, ŷ_pseudo, w_pseudo=0.3)` to training set.
  3. **Crucial guard against test leakage:** evaluate the pseudo-trained model on
     a **held-out 25% of the original train** (use `date 56-79` as a clean holdout),
     not on test. This catches if the pseudo training is memorizing the test target
     rather than improving generalization.
  4. If the holdout MSE drops vs T75 baseline by > 5%, ship it.
- **Risk gates:** the obvious risk — pseudo-training can over-fit test by
  construction. The 25% train holdout protocol catches this. If holdout doesn't
  improve, abort — gain is likely test-leakage artefact.
- **Expected:** +0.3 ~ +1.5 LOSO-equiv (more uncertain than other picks because
  of the leakage-vs-genuine-improvement ambiguity).
- **Cost:** 1 day. Most time is the holdout protocol design, not training.
- **Hard-constraint check:** pseudo-training pulls test rows but **predicts at
  inference time without test-row state** → stateless requirement satisfied. ✅.

---

## 4. Stack expectation if all 5 ship

Naive sum: +1.0 + +1.5 + +1.0 + +1.0 + +0.5 = **+5.0 LOSO-equiv** (mid-estimate).

Realistic stack (correlation-discounted):
- Pick 1 (CatBoost): +1.0 ~ +3.0 — uncorrelated with NN/LGB, full-stack
- Pick 2 (alphas): +0.5 ~ +1.5 — partial overlap with R34 Stage 1-5
- Pick 3 (ReVol): +0.5 ~ +1.0 — partial overlap with R34 Stage 1 z-score features
- Pick 4 (magnitude weight): +0.3 ~ +1.0 — model-architecture-specific (LGB only or NN+LGB)
- Pick 5 (pseudo-self-distill): +0.0 ~ +1.0 — risk of leakage trumps mean expectation

**Realistic stack mean: +2.5 ~ +5.0 LOSO-equiv → platform projection +1.8 ~ +3.5
above iter_014's expected +20.67 → platform target +22.5 ~ +24.2.** (Public-board
strong score is +29.18; this would close ~50% of the remaining gap.)

---

## 5. Suggested 5-day implementation order

Order picked for risk decoupling — each step confirms before the next.

| Day | Worker | Task | Decision gate before next step |
|---|---|---|---|
| **Day 1 (AM)** | T46-revival | CatBoost-regression 5-seed pilot on 1 seed | single-seed LOSO-equiv ≥ +30 → continue |
| Day 1 (PM) – Day 2 | T46-revival | Full 5-seed + DE asymmetric + ensemble integration | 3-base ensemble (NN+LGB+CB) LOSO-equiv ≥ +39 → ship as iter_015 |
| **Day 3 (AM)** | T57-revival | Magnitude-weight 3 variants × 5-seed (cheap) | best variant LOSO-equiv ≥ +37 → keep |
| Day 3 (PM) | T35-revival | ReVol+P feature concat 5-seed | LOSO-equiv ≥ +37 → keep |
| **Day 4** | T22-revival | Alpha-loaded LGB feature gain + 5-seed | alpha-loaded LGB ≥ +35 standalone → keep |
| **Day 5** | T65-revival | Self-distill protocol with train-holdout guard | holdout MSE drops ≥ 5% vs baseline → ship |

Each step's revived experiment is **independently ablatable**; stacking is performed
in a final iter_016 candidate after Day 5.

---

## 6. Failure mode for the report itself

If T46 (Pick 1, CatBoost) doesn't beat its own classification baseline by enough
(LOSO-equiv < +28), the entire "regression revives CE failures" thesis is **mildly
weakened** — the breakthrough came from L2-target, not from target-aligned
ensembling diversity. In that case Picks 2–5 are still viable but expected gains
should be halved and budget reallocated to entirely new ideas (e.g. R41 fresh-eye
HFT proposals, R42 SPO+/DFL).

Conversely, if Pick 1 shows +2.0+ stand-alone, the regression-revives-CE thesis is
strongly confirmed and **all 9 secondary candidates** (including T11 cross-sym
mixup, T56 XGBoost, T55 Stage 4 features, T66 adversarial reweighting) should be
queued.

---

## RESULT

```
RESULT: task=r44_revisit metrics={n_revivable=9, top5=[T46_catboost_regr, T22_alphas_regr, T35_revol_regr, T57_magnitude_weight, T65_self_distill_regr], expected_loso_gain_at_iter014=+2.5_to_+5.0, expected_platform_gain=+1.8_to_+3.5} notes=top1=CatBoost-regression (Optiver-2023-1st recipe), top2=alpha101/191 at h60+regr (cache exists), top3=ReVol regr (cache exists, fixes F-F false-signal), top4=magnitude sample-weight (1-line LGB Dataset arg), top5=continuous-target self-distillation (Hinton-style); 5-day disciplined order with per-step decision gates
```
