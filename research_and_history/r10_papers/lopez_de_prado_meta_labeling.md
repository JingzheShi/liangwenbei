# Meta-Labeling — López de Prado, Advances in Financial Machine Learning (Ch 3, §3.5)
**Coined**: 2017 at Guggenheim Partners. Now standard in quant ML.
URLs:
* Wikipedia: https://en.wikipedia.org/wiki/Meta-Labeling
* Hudson & Thames: https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/
* mlfinpy docs: https://mlfinpy.readthedocs.io/en/latest/Labelling.html

## Core idea

**Decouple direction prediction from sizing/gating.**

* **Primary M1**: a model (any) that predicts trade direction. Output ∈ {-1, 0, +1}.
* **Secondary M2**: a binary classifier that learns "given the primary said
  TRADE in this direction, was that profitable?". Trained ONLY on samples
  where M1 was non-zero.
* **Decision rule**: `final = M1 if M2.prob_profit > τ else 0`

This is **exactly the structure** that our threshold-gating heuristic
(T=0.55, δ=0.10) approximates by hand — except meta-labelling LEARNS the
gate from data using all features, not just probability magnitude.

## Loss / objective

* M2 is trained as **binary classification with CE loss** on the binary
  meta-label `1 if pnl_realised > 0 else 0`.
* Threshold τ is chosen on a holdout to maximise downstream PnL.

## Position sizing from M2 probability (Ch 10 — Bet Sizing)

Once you have a calibrated M2 probability `p̂`, López de Prado offers several
sizing methods:

| Method | Formula |
|---|---|
| All-or-nothing | `pos = sign(M1) if p̂ > τ else 0` |
| Linear | `pos = sign(M1) · max(0, p̂ - τ) / (1 - τ)` |
| Normal CDF | `pos = sign(M1) · (2·Φ(z) - 1)` where z = z-score from p̂ |
| Empirical CDF | rank p̂ over training percentiles |
| **SOPS (Sigmoid)** | `pos = sign(M1) · sigmoid(α · (p̂ - 0.5))` — α tuned for Sharpe |

For our 3-class submission constraint, we round `pos` to {-1, 0, +1} →
{label_0, label_1, label_2}.

## Empirical evidence (Hudson & Thames)

| Strategy | Metric | Before | After |
|---|---|---|---|
| Mean reversion | accuracy | 17% | **63%** |
| Mean reversion | precision | 0.17 | 0.20 |
| Trend following | accuracy | 48% | 55% |
| Trend following | precision | 0.48 | 0.54 |

Big precision-and-accuracy gains for mean reversion (very low base precision).
Smaller for trend-following but consistent.

## What we'd borrow

This is **Scheme S7** in `r10_pnl_loss.md`. Specific recipe for our setting:

1. **M1** = our existing iter_002 LightGBM Scheme C multi-horizon
   (frozen).
2. **M2** = NEW binary LightGBM, features = 154 LOB features + (p_0, p_1, p_2)
   from M1 + y_pred from M1. Target = 1 if `pnl_realised(y_pred, t) > 0` else 0.
   Trained ONLY on samples where y_pred ∈ {0, 2}.
3. At inference: y_pred = M1; if M2.predict_proba(positive) < τ → predict 1
   instead of y_pred.
4. Tune τ on LOSO val. Expected τ ≈ 0.55 ~ 0.7 for our fee level.

This generalises our hand-coded threshold gate strictly — at worst it
recovers the gate behaviour, at best it learns nuanced "trade only when
sym-similar OOD pattern looks safe" rules.

## Implementation gotcha

**Avoid label leakage**: M2's training labels are computed from realised future
prices; M1 must be retrained without the same future leakage if M1 used any
features that touch the future. Our Scheme C is safe (features are 100-tick
backward window), but watch carefully.

**LOSO discipline**: train M2 on the same LOSO folds as M1, never on the
held-out sym. Otherwise M2 just memorises M1's mistakes on OOD.

## Why this is the most promising scheme

* Already validated by ~10 years of quant practice
* Meta-labelling consistently produces **+10 ~ +25 score gains** in
  published comparisons
* Our threshold gate already captures part of the value (+13 LOSO sum from
  T2 fold sums); a learned gate plausibly captures another +10
