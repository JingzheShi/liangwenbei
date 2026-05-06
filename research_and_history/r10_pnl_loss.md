# r10 — PnL-aware / utility-aware loss functions for 良文杯

> **Created**: 2026-05-06 (R10 research worker)
> **Goal**: Find loss / training-objective changes that directly optimise the leaderboard
> metric (cumulative PnL with 0.0001 fee per side) instead of cross-entropy.
> **Current strongest baseline**: iter_002 LightGBM Scheme C multi-horizon, h_10 LOSO sum
> **+21.86** (5-fold), trained with CE + class_weight balanced + post-hoc threshold gate
> (T=0.55, δ=0.10).

## 0. Background — why CE is wrong

Our scoring rule (`submission/RULES.md` §7) reduces to:

```
pnl_t  ≈  (label_pred_t − 1) · Δp_t  −  2·fee · 1[label_pred_t ≠ 1]
        with Δp_t = midprice_{t+n} − midprice_t,   fee = 1e-4
```

(modulo a `1/(midprice_t+1)` ≈ 1 normaliser; midprice is "涨跌幅" so
midprice_t+1 ∈ [0.9, 1.1] for normal stocks.)

Cross-entropy loss treats every error symmetrically:

* misclassifying a "Δp ≈ 3·fee" sample (would have made +1e-4) costs the same as
  misclassifying a "Δp ≈ 30·fee" sample (would have made +28e-4)
* misclassifying flat→up (cost = 2·fee = 2e-4) costs the same as
  misclassifying down→up (cost = 2·fee + 2|Δp|, can be 5e-3+)

So CE is **mis-aligned** with the leaderboard. Every paper / Kaggle winner that
optimises a financial metric finds a way to rewire this. The schemes below are
ranked by ease of dropping into our LightGBM Scheme C pipeline.

---

## 1. Schemes (10 concrete proposals)

### S1. Sample-weighted CE by |Δp|  ⭐ EASIEST

**Math**: keep CE; pass `sample_weight_t = clip(|Δp_t|, 0, q99(|Δp|))` to `lgb.Dataset`.

**Where**: LightGBM `Dataset(..., weight=w)` — single line of code change.

**Source**:
* arxiv 2502.17493 ("A Novel Loss Function for Daily Stock Trading", 2025) —
  return-weighted CE: `loss = CE · |r_cap|`. CNN went from
  33.73% annual return (CE) → **61.73%** with this trick. Sharpe 0.79 → 1.18.
* Kaggle Jane Street 2021 / 2024 — almost every top solution uses
  `weight = ln(1 + w_competition)` or similar; importance is signal-magnitude proxy.

**Implementation complexity**: 1/10. We already have `midprice_diff_h` materialised
in the cache; just pass it as `sample_weight`.

**Expected gain**: +3 ~ +6 LOSO sum (h_10). High-confidence small win. Composes
with everything we already have.

**Trade-off**: model focuses on tail-volatility samples — risk: forgets the
"flat" majority and over-trades on sym=2 (the low-vol blue-chip).
Mitigation: cap weight at 99th percentile.

---

### S2. LightGBM custom multiclass objective: expected-PnL maximisation  ⭐ HIGHEST EV

**Math** (per sample t with realised Δp_t, fee f):

Define class rewards
* `r_0(t) = -Δp_t - 2·f`     (predicting down)
* `r_1(t) = 0`               (predicting flat)
* `r_2(t) = +Δp_t - 2·f`    (predicting up)

Let `p = softmax(z)`. Expected PnL per sample:
```
E[pnl | t] = Σ_k p_k · r_k
```

Loss = `−E[pnl | t]` (minimise negative expected PnL).

Closed-form gradient w.r.t. logit `z_k`:
```
∂L/∂z_k = −p_k · (r_k − Σ_j p_j · r_j)
        = −p_k · (r_k − E[r])
```

Hessian (diagonal-only approximation that LightGBM accepts):
```
∂²L/∂z_k² = −p_k(1−p_k) · (r_k − E[r]) + p_k · Σ_j p_j(1[j=k]−p_k)·r_j
```

A safer Newton-style approximation that's universally used in custom GBDT objectives:
```
hess_k ≈ |grad_k| · (1 − |grad_k|)   # bounded > 0
```

**Where**: `lgb.train(..., fobj=our_obj, feval=pnl_eval)` — drop-in replacement
for `objective='multiclass'`. Need to reshape preds (N·K,) ↔ (N, K) in F-order
(LightGBM convention).

**Source**:
* Hippocampus's Garden (Custom Objective for LightGBM,
  https://hippocampus-garden.com/lgbm_custom/) — exact API & gradient/hessian pattern.
* Max Halford (https://maxhalford.github.io/blog/lightgbm-focal-loss/) —
  binary template that we generalise to multiclass.
* `LightGBM-with-Focal-Loss/lucamassaron` (Kaggle) — multiclass template
  showing the (N,K) reshape with F-order: `y_pred.reshape(N, K)`.
* arxiv 2509.04541 (Khubiyev 2025, "Finance-Grounded Optimization") — exactly
  this `PnLLoss = -α·r` framing, but for regression. Best LSTM+MDDLoss got
  Sharpe 1.76 vs Sharpe **−0.46** for MSE baseline. Confirms PnL loss > MSE.

**Implementation complexity**: 4/10. Need careful gradient shape handling +
init_score from CE pre-training (LightGBM custom obj starts at 0 logits → softmax
is uniform → gradient blows up). Solution: warm-start with **20 rounds of CE**,
then switch to PnL objective.

**Expected gain**: +6 ~ +15 LOSO sum (h_10). Biggest single lever. Still composes
with threshold gating.

**Risk**:
* Gradient is non-convex in feature space → `min_data_in_leaf` must be raised
  (e.g. 500+) to avoid overfitting on noisy single-sample gradients
* No regularisation → model collapses to "always predict label_1" if fee is
  too aggressive. Mitigation: subtract entropy regulariser:
  `L = −E[pnl] − β·H(p)` with β ∈ [0.001, 0.01]

---

### S3. Cost-sensitive CE with full cost matrix C[true][pred]

**Math**: every sample t has a 3×3 cost matrix C(t) computed from realised Δp:

| true \ pred | 0 (down) | 1 (flat) | 2 (up) |
|---|---|---|---|
| 0 (down)   | 0          | +Δp_t (foregone)  | -Δp_t·2 + 2f (loss) |
| 1 (flat)   | -2f        | 0                  | -2f                  |
| 2 (up)     | -Δp_t·2 + 2f | -Δp_t (foregone) | 0                    |

(treat foregone gains as "regret" = 0 for the conservative predictor; treat
flat-true rows symmetrically.)

Then per-sample loss:
```
L_t = Σ_k p_k · |C[y_t][k]|·sign(C[y_t][k])    # signed cost
    = -Σ_k p_k · reward_k   # equivalent to S2 if we drop foregone gains
```

In fact **S3 reduces to S2 when foregone gains are ignored** because
"true=0, pred=2" cost is exactly the same as the negative of `r_2 = +Δp_t − 2f`
when Δp_t < 0. So S3 is mathematically very close to S2; it's a different
*framing* (Bayes risk vs utility maximisation).

**Source**:
* CSBoost / MetaCost — Sun et al., "Cost-sensitive boosting for classification of imbalanced data" (Pattern Recognition 2007)
* López de Prado, *Advances in Financial Machine Learning* (Ch 3 — Labeling, Ch 12 — Bet sizing)

**Implementation complexity**: 5/10. Same as S2 but with explicit per-sample
cost matrices passed via a custom callback; some bookkeeping to vectorise.

**Expected gain**: same as S2 (+6 ~ +15) since they're equivalent under our fee
structure; pick S2 unless you want regret-style framing.

---

### S4. Differentiable expected-PnL loss (NN; DeepLOB / MLPLOB head)

**Math**: identical to S2 but in PyTorch, using softmax cross with reward vector:

```python
# logits: (B, 3); deltas: (B,) realised Δp; fee = 1e-4
rewards = torch.stack([-deltas - 2*fee, torch.zeros_like(deltas), +deltas - 2*fee], dim=1)
p = F.softmax(logits, dim=1)
loss = -(p * rewards).sum(dim=1).mean()
```

**Where**: any NN we train on the LOB tensor (DeepLOB, MLPLOB, BiN-CTABL).
Replaces `nn.CrossEntropyLoss(weight=class_weight)`.

**Source**:
* arxiv 2509.04541 (Khubiyev 2025) — exact framing for regression: `PnLLoss = -α·r`
* Yirun's 1st place Jane Street 2021 writeup — "plug the utility score function
  directly as loss function (multiplied by -1 to maximize)". Yirun's bottleneck-
  shared SAE+MLP was trained this way.
* Moody-Saffell 1998 ("Reinforcement Learning for Trading", NIPS) — reward-driven
  differentiable training of trading systems with differential Sharpe ratio.

**Implementation complexity**: 3/10. PyTorch is straightforward.

**Expected gain**: +5 ~ +12 LOSO sum if NN is competitive. But our NNs haven't
beat LightGBM yet — so this alone won't win.

**Critical caveat**: in a 3-class softmax with this loss, the model can collapse
to always-flat (all p_1 = 1) because `r_1 = 0` is the safe action when fees
dominate. **Add entropy regulariser** `+ β·(p · log p).sum()` with β ≈ 0.01.

---

### S5. Differentiable Sharpe loss (NN; portfolio-style)

**Math**:
```
SharpeLoss = -E_t[pnl_t] / sqrt(Var_t[pnl_t] + ε)
```
where `pnl_t` is computed from softmax probs as in S4.

Compute over each minibatch (or each "session" if you have time grouping).

**Source**:
* Moody & Saffell 1998 — differential Sharpe for online trading
* Zhang/Zohren/Roberts 2020 — "Deep Learning for Portfolio Optimization", Sharpe loss
* arxiv 2509.04541 — "SharpeLoss = E(pnl) / (Var(pnl) + ε)"

**Implementation complexity**: 4/10.

**Expected gain**: similar to S4, but Khubiyev showed Sharpe-trained models often
get LOWER raw cum_pnl than PnL-trained ones (max-PnL: 18.7%; Sharpe: lower).
For a leaderboard that scores **cum_pnl directly**, prefer S4.

**Verdict for us**: **NOT recommended** — wrong objective for our metric.
Sharpe penalises variance which we don't care about; we just want max sum.

---

### S6. Profit-aware regression on signed Δp (LightGBM regressor)

**Math**: train `LGBMRegressor` to predict `Δp_t` directly with Huber/MAE loss,
then at inference apply threshold:

```
pred_label = 2  if pred_Δp > +τ_up
           = 0  if pred_Δp < -τ_dn
           = 1  otherwise
```

Tune τ_up, τ_dn on LOSO val (search grid τ ∈ [1·fee, 5·fee]).

**Source**:
* Optiver Realized Volatility 1st place — regression on log-return per minute
* HYD Optiver Trading at Close 1st place — MAE regression on closing price moves,
  CatBoost(0.5) + GRU(0.3) + Transformer(0.2). MAE 5.3070.
* Macrosynergy "How to measure the quality of a trading signal" — argues
  regression on signed return is better than classification when signal magnitude
  varies ("strategy taking medium-term positions of varying size based on risk
  premium estimates calls for a signal that is linearly correlated with subsequent returns")

**Implementation complexity**: 3/10. Just retrain Scheme C as regressor; same
features.

**Expected gain**: +3 ~ +8. Theoretically equivalent to S2 in the limit of
perfect threshold tuning, but in practice 3-class CE→argmax loses the
magnitude information that regression preserves.

**Risk**: Δp distribution is heavy-tailed; need Huber/quantile loss not MSE.

**Trick**: if you train regressor + use it as a *gating* signal alongside the
classifier (i.e. "trade only when both classifier and regressor agree"), this
becomes a poor-man's meta-labelling.

---

### S7. Meta-labelling: primary direction + secondary "should I trade?"

**Math** (López de Prado, Adv. Fin. ML, Ch 3):

* **Primary** M1 = our existing iter_002 Scheme C LightGBM →
  outputs (p_0, p_1, p_2) and direction y₁ ∈ {0, 1, 2}.
* **Secondary** M2 = NEW binary classifier: features = (X, p_from_M1, y_pred_from_M1),
  target = `1 if pnl(y_pred_from_M1, t) > 0 else 0`.
  Trained ONLY on samples where y₁ ∈ {0, 2} (primary said "trade").
* **Decision rule** at inference:
  * y_final = y₁ if M2 says "take it" (prob > τ); else y_final = 1 (skip).

This **explicitly learns when our existing model is right vs wrong**, gating
out OOD-fail cases.

**Source**:
* López de Prado, *Advances in Financial Machine Learning*, Chapter 3 §3.5
  ("Meta-Labeling")
* Hudson & Thames blog, "Does Meta-Labeling Add to Signal Efficacy?" — out-of-sample
  accuracy 17%→63% for mean-reversion, 48%→55% for trend-following
* Wikipedia "Meta-Labeling" (López de Prado coined this in 2017 at Guggenheim)

**Implementation complexity**: 5/10. Needs 2-stage training pipeline + LOSO
evaluation discipline so we don't leak the primary's training labels into M2.

**Expected gain**: **+10 ~ +25 LOSO sum** (most cited improvement in the literature
for transaction-cost-bound settings). Could be the biggest win in this list IF
the primary already has decent precision.

**Why it works for us**: our threshold gating (T=0.55, δ=0.10) is essentially a
*hand-coded* meta-labeller. M2 should be a strict generalisation that learns
which features are predictive of "trade succeeds vs fee-eats-it".

---

### S8. Probability calibration (Platt / isotonic) + Expected-Value gating

**Math**:
1. Train a model with CE (any model — e.g. existing iter_002).
2. Fit isotonic regression on LOSO validation: `p̂_calibrated = iso(p_raw)` per class.
3. At inference compute **expected PnL per action**:
   ```
   EV(action=2 | x) = p̂_2(x) · E[Δp | up] − 2·fee
   EV(action=0 | x) = p̂_0(x) · |E[Δp | down]| − 2·fee
   EV(action=1 | x) = 0
   ```
   where `E[Δp | up]` is estimated from training data (≈ +0.0006 for h_10).
4. **Take action with highest EV; only if EV > 0** else action 1.

**Source**:
* scikit-learn 1.16 Probability Calibration docs
* Platt 1999, "Probabilistic outputs for SVMs"
* Niculescu-Mizil & Caruana 2005 ("Predicting Good Probabilities With Supervised Learning")
* López de Prado Ch 12 — "Bet Sizing" (CDF-based sizing from probabilities)

**Implementation complexity**: 2/10. `from sklearn.isotonic import IsotonicRegression`
+ a numpy line for EV.

**Expected gain**: +2 ~ +6. Smallest of the quick wins, but **stacks with all
other schemes**. Almost free.

**Note**: our current threshold gate `T=0.55, δ=0.10` is a piecewise-constant
calibration; isotonic is its smooth nonparametric upgrade and provably ≥ Platt
when n_cal > 1000 (we have plenty).

---

### S9. Asymmetric focal loss with cost-matrix scaling (LightGBM multiclass)

**Math**: combine focal loss `(1−p_t)^γ · CE` with class-cost scaling:

```
L = −Σ_k 1[y=k] · w_k · (1 − p_k)^γ · log p_k
where w_k = expected fee-loss-magnitude when true class is k
```

For us:
```
w_0 ∝ E[|Δp| | y=0] + 2·fee   (down samples are expensive to miss)
w_1 = 1·fee                    (flat samples are cheap to miss)
w_2 ∝ E[|Δp| | y=2] + 2·fee
```

**Source**:
* arxiv 1908.01672 — "Imbalance-XGBoost: leveraging weighted and focal losses"
* Lin et al. 2017 (RetinaNet) — original focal loss
* Max Halford / jrzaurin LightGBM-with-Focal-Loss — concrete code
* Luca Massaron Kaggle multiclass focal LightGBM kernel — `(N, K)` reshape pattern

**Implementation complexity**: 4/10. Plug-and-play extension of S2 if you already
have multiclass custom obj infrastructure.

**Expected gain**: +2 ~ +5 over plain S2. Useful when class imbalance is severe
(our flat:up:down ≈ 0.74:0.13:0.13 for h_10).

**Trade-off**: yet another hyperparameter (γ ∈ {1, 2, 5}). Train γ-grid.

---

### S10. Continuous bet-sizing head (regression on softmax → position)

**Math**: predict a *continuous position* `pos ∈ [-1, 1]` instead of 3-class:

```
pos_t = p_2(x_t) − p_0(x_t)   (or  =  p_2 − p_0 if max(p_2, p_0) > τ else 0)
pnl_t = pos_t · Δp_t − 2·fee · |pos_t|
```

Train via expected PnL maximisation (S2-style) **but with continuous output**.

**Source**:
* López de Prado Ch 10 — "Bet Sizing" (Sigmoid Optimal Position Sizing)
* Moody-Saffell — recurrent reinforcement learning with continuous position
* HYD Optiver Trading-at-the-Close — predicts continuous price-move proxy not class

**Implementation complexity**: 6/10 (NN-only — LightGBM doesn't natively output
continuous positions; would need a wrapper).

**Expected gain**: +5 ~ +10 IF the platform accepts non-integer labels. Our
platform requires `label ∈ {0, 1, 2}` — so we'd round position to discrete.
**Effective floor**: same as S4 with threshold post-proc.

**Verdict**: skip unless the platform's Predictor signature changes.

---

## 2. Comparison table

| ID | Scheme | Where | Complexity | Expected gain | Composability | Risk |
|---|---|---|---|---|---|---|
| **S1** | weight CE by \|Δp\| | LGBM Dataset weight | 1/10 | +3 ~ +6 | ⭐⭐⭐ all | low |
| **S2** | custom multiclass PnL obj | LGBM fobj | 4/10 | +6 ~ +15 | ⭐⭐⭐ all | medium (collapse) |
| S3 | cost matrix CE | LGBM fobj | 5/10 | ~S2 | ⭐⭐ | medium |
| S4 | differentiable PnL (NN) | torch loss | 3/10 | +5 ~ +12 | ⭐⭐ NN-only | medium (collapse) |
| S5 | differentiable Sharpe | torch loss | 4/10 | unclear | ⭐ | wrong objective |
| S6 | regression on Δp + threshold | LGBMRegressor | 3/10 | +3 ~ +8 | ⭐⭐ alt model | low |
| **S7** | meta-labelling | 2-stage LGBM | 5/10 | +10 ~ +25 | ⭐⭐⭐ on top | low |
| **S8** | isotonic + EV gating | post-proc | 2/10 | +2 ~ +6 | ⭐⭐⭐ all | very low |
| S9 | focal + cost weights | LGBM fobj | 4/10 | +2 ~ +5 over S2 | ⭐⭐ | medium |
| S10 | continuous bet | NN regression | 6/10 | +5 ~ +10 | ⭐ | platform compat |

---

## 3. **Specific recommendation: if I can only try 2 things, I try these**

### 🥇 #1: **S2 (custom LightGBM multiclass PnL objective)** + entropy regulariser

Why: biggest single lever, drops into existing Scheme C training, closed-form
gradient/hessian, mathematically aligned with the leaderboard. Worst case:
falls back to current iter_002 if entropy collapses (gating still salvages it).

**Concrete plan (write code tomorrow)**:

```python
# train_scheme_c_pnl.py — replacement for the CE training step
def pnl_multiclass_obj(z_flat, dataset, K=3, fee=1e-4, beta_entropy=0.01):
    """LightGBM custom objective: maximise expected PnL with entropy reg.
    z_flat: (N*K,) raw logits, F-ordered
    dataset.label: (N,) class labels {0,1,2}
    dataset.weight: (N,) optional per-sample weight
    REQUIRES: dataset.set_init_score(...) called externally with realised Δp_t
              or attach Δp_t as the *weight* (we hijack weight to carry Δp).
    """
    N = len(dataset.label)
    z = z_flat.reshape(N, K, order='F')
    # softmax (numerically stable)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    delta = dataset.delta  # custom attribute — see hack below
    rewards = np.stack([-delta - 2*fee, np.zeros_like(delta), +delta - 2*fee], axis=1)
    expected_r = (p * rewards).sum(axis=1, keepdims=True)
    # gradient: ∂(-E[r])/∂z_k = -p_k(r_k - E[r])
    grad = -p * (rewards - expected_r)
    # entropy reg: H(p) = -Σ p log p; ∂(-βH)/∂z_k = β * p_k * (log p_k + H_norm)
    # simplification: use diagonal Newton hessian = p_k(1-p_k) * |grad|
    grad += beta_entropy * p * (np.log(p + 1e-12) - (p * np.log(p + 1e-12)).sum(axis=1, keepdims=True))
    hess = (p * (1 - p)) * np.abs(rewards - expected_r) + 1e-6
    return grad.reshape(-1, order='F'), hess.reshape(-1, order='F')

# Hack to carry Δp through LightGBM:
#   Use LightGBM's `Dataset(..., init_score=ce_warmstart_logits)` for warmstart,
#   stash Δp on dataset object via `dataset.delta = midprice_diff_h_array`.
# Or: pass Δp as a column in X during fobj via closure.

def pnl_eval(z_flat, dataset, K=3, fee=1e-4):
    N = len(dataset.label)
    z = z_flat.reshape(N, K, order='F')
    pred = z.argmax(axis=1)
    delta = dataset.delta
    pnl = np.where(pred == 2, delta - 2*fee,
          np.where(pred == 0, -delta - 2*fee, 0)).sum()
    return 'cum_pnl', pnl, True   # higher_is_better

# warm-start: 50 rounds CE → snapshot logits → switch
booster = lgb.train({'objective': 'multiclass', 'num_class': 3,
                     'learning_rate': 0.05}, train_data, num_boost_round=50)
init_score = booster.predict(X_train, raw_score=True)  # (N, 3)
train_data2 = lgb.Dataset(X_train, label=y_train, init_score=init_score.flatten('F'))
train_data2.delta = delta_train
booster2 = lgb.train({'learning_rate': 0.02}, train_data2,
                     fobj=pnl_multiclass_obj, feval=pnl_eval, num_boost_round=2000,
                     valid_sets=[val_data2], early_stopping_rounds=50)
```

**Expected outcome**: LOSO sum h_10 +21.86 → **+30 ~ +40** (with existing
threshold gate kept). New iter ≥ iter_002.

### 🥈 #2: **S1 (sample-weighted CE by |Δp|)** as a **2-day quick win** parallel track

If S2 takes longer, ship S1 first as a low-risk improvement and combine with
existing threshold gating.

```python
# in load.py / build_cache.py
for h in horizons:
    df[f'sample_w_{h}'] = np.clip(df[f'midprice_diff_{h}'].abs(), 0, df[f'midprice_diff_{h}'].abs().quantile(0.99))

# in train_scheme_c.py
train_data = lgb.Dataset(X_train, label=y_train_h_10, weight=sample_w_train_h_10)
```

That's literally 6 lines of code. Expected: **LOSO sum +21.86 → +25 ~ +28**.

### Bonus stack: S2 + S8 (isotonic calibration + EV gating)

After S2 trains, recalibrate softmax probs with isotonic on LOSO val, replace
hard threshold gate with `EV > 0` gate. Free +2 ~ +4 on top.

---

## 4. **What we've ruled out and why**

* **S5 (differentiable Sharpe)** — wrong objective. Leaderboard is `Σ pnl`, not
  `Σ pnl / σ(pnl)`. Sharpe-trained models systematically underperform raw-cum-pnl
  on raw-cum-pnl benchmarks (Khubiyev 2025 Table 3).
* **S10 (continuous position)** — platform requires discrete labels {0, 1, 2}.
  Best we can do is round; no benefit over S2.
* **DeepLOB / NN PnL loss without FT-Transformer** — we already concluded NN
  hasn't beaten LGBM on our data; doing it via S4 doesn't change that.

## 5. **Hard constraints that all schemes respect**

* No `sym` features, no `date` features (CRITICAL_CONSTRAINTS.md §1)
* Per-sample reward `Δp_t` is computed at TRAINING time only, from
  `midprice_diff_{h}` already cached. **Predictor.predict** never sees Δp.
* Predictor still emits 0/1/2 → no platform contract change.

---

## 6. Citation sources (linked in detail in `r10_papers/`)

1. arxiv 2509.04541 — Khubiyev, "Finance-Grounded Optimization For Algorithmic
   Trading", 2025 — `r10_papers/finance_grounded_khubiyev_2025.md`
2. arxiv 2507.19639 — "Directly Learning Stock Trading Strategies Through Profit
   Guided Loss Functions", 2025 — `r10_papers/stockloss_2025.md`
3. arxiv 2502.17493 — "A Novel Loss Function for Daily Stock Trading", 2025 —
   `r10_papers/return_weighted_ce_2502_2025.md`
4. Yirun Jane Street 2021 1st place — `r10_papers/yirun_jane_street_2021.md`
5. Volkova Jane Street 2024 1st place — `r10_papers/volkova_jane_street_2024.md`
6. HYD Optiver Trading-at-the-Close 2023 — `r10_papers/hyd_optiver_2023.md`
7. López de Prado *Adv. Fin. ML* (meta-labelling, bet sizing) —
   `r10_papers/lopez_de_prado_meta_labeling.md`
8. Moody-Saffell 1998 — `r10_papers/moody_saffell_1998.md`
9. Hippocampus's Garden / Max Halford LightGBM custom objective tutorials —
   `r10_papers/lightgbm_custom_objective_howto.md`

---

**Last updated**: 2026-05-06 by R10 worker.
