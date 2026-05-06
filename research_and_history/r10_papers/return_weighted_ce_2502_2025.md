# A Novel Loss Function for Deep Learning Based Daily Stock Trading System
**arxiv 2502.17493 (Feb 2025)**
URL: https://arxiv.org/abs/2502.17493 / html: https://arxiv.org/html/2502.17493v2

## Abstract (extracted)

Introduces a **return-weighted cross-entropy loss** for daily stock direction
prediction. CNN with technical indicators. 2019–2024 backtest:
**61.73% annual return, Sharpe 1.18.**

## The loss (verbatim)

```
loss(y_true, y_pred) = CE(y_true, y_pred) · |r_cap|
```

where `r_cap = clip(daily_return, -50%, +50%)`.

**Motivation**: regular CE treats every misclassification equally. A "Strong
Sell" mislabel (true return -5%) costs the same as a "Hold" mislabel (true
return 0.5%). Return-weighted CE forces the model to focus on **big movers**.

## Architecture

* CNN with 1-D temporal convolutions (28 technical features)
* Sector embedding (12 × 28 = 336 params)
* 53,280 trainable params (CNN), 82,596 with attention head
* Mixture-of-experts at the end

## Results (2019–2024)

| Loss | Annual return | Sharpe | Max DD |
|---|---|---|---|
| **CE (baseline)** | 33.73% | 0.79 | -58.37% |
| MSE | 53.87% | 1.00 | -47.50% |
| **return-weighted CE** | **61.73%** | **1.18** | **-30.99%** |
| Attention | 31.33% | 0.97 | -46.38% |

Return-weighted CE doubled annual return and halved max drawdown vs plain CE.

## What we'd borrow — directly applicable as our S1

This **is exactly our S1**: weight CE by absolute realised return.

For us:
```python
sample_weight_t = clip(|midprice_diff_h_t|, 0, q99)
lgb.Dataset(X, label=y, weight=sample_weight_t)
```

Single-line change. No gradient/hessian rewriting needed. Composes with
existing `class_weight` and threshold gate.

**Caveat**: the paper applies to daily horizon and binary direction. Our
problem is multiclass (0/1/2) and tick-level (h ∈ 5..60). The intuition still
holds: large |Δp_t| samples are where the trade margin matters most.

## What we'd skip

* Sector embedding — we don't have sector info (and `sym` is forbidden, see
  CRITICAL_CONSTRAINTS).
* Mixture-of-experts head — overkill for our setup.
* CNN architecture — orthogonal; we use LightGBM.

## Implementation tip

The paper caps return at ±50% to prevent gradient explosion. For us the
analogue is q99(|Δp|) — without capping, a single 0.5% mid-move sample
(z>5σ) can dominate. Inspect the |Δp_t| distribution before applying.
