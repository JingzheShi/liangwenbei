# HYD's Solution (1st place) — Optiver Trading at the Close 2023
**Kaggle competition**
URLs:
* Writeup: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
* X/Twitter: https://x.com/hydantess1993/status/1773982572537581799

## Architecture & ensemble

* **CatBoost (0.5) + GRU (0.3) + Transformer (0.2)** — weighted average
* CatBoost is the dominant single weight (Volkova's pattern again — boosting
  is the workhorse in tabular finance ML)
* **300 features** total

## Loss function

* **Mean Absolute Error (MAE)** regression on the closing-auction price target
* No custom utility loss
* Best public single-model MAE = 5.3070

## Feature engineering (the "magic features")

Triplet imbalances, pairwise price imbalances, synthetic index features, etc.:

```python
size_imbalance = bid_size / ask_size
liquidity_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
market_urgency = (ask_price - bid_price) * liquidity_imbalance
```

These are **sym-agnostic** (no per-stock normalisation) and **time-local**
(no global state). Compatible with our hard constraints.

## Training tricks

* **Online learning every 12 days, 5 cycles** during inference
* Post-processing of predictions (clipping, mean-reverting toward 0)

## What we'd borrow

1. **Imbalance-style features**: We already have most of these (see
   `feature_ideas.md` and Scheme C feature list). Confirms they're in the
   global SOTA.

2. **MAE > MSE** in heavy-tailed financial regression: relevant for our
   scheme **S6 (regression on Δp + threshold)** — use Huber/MAE not MSE.

3. **CatBoost vs LightGBM** parity: confirmed CatBoost (0.5 weight) doesn't
   blow LightGBM (0.0 weight in their stack) out — they didn't even include
   LightGBM. But we already have LightGBM Scheme C working. The data is more
   important than CatBoost-vs-LightGBM choice.

## What we'd skip

* Online learning — platform doesn't allow.
* Transformer head — we've shown LightGBM dominates at our data size.
* GRU — could revisit but Scheme C beat our last GRU baseline.

## Critical observation

This is the **second** 1st-place solution (after Volkova) that uses
**plain regression MSE/MAE with sample weighting** rather than a custom
utility loss. This tells us the **value of S1 (weighted CE) is high relative
to S2 (custom PnL obj)** as a starting point — the bigger competitions chose
the simpler path. We should ship S1 first, then iterate on S2.
