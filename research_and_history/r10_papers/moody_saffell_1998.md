# Reinforcement Learning for Trading — Moody & Saffell, NIPS 1998
**Original paper**: http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf
**Extended journal version (2001)**: "Learning to trade via direct reinforcement", IEEE TNN

## Summary

Foundational paper on **direct reinforcement learning (DR)** for trading.
Introduces the **differential Sharpe ratio** as an online-trainable
performance objective.

## Key contributions

1. **Recurrent Reinforcement Learning (RRL)**: trades a sequence of positions
   `F_t ∈ {-1, 0, +1}` (or continuous in [-1, 1]) by maximising cumulative
   risk-adjusted return.

2. **Reward function**:
   ```
   R_t = F_{t-1} · (P_t - P_{t-1}) - δ · |F_t - F_{t-1}|
   ```
   Differentiable in F_t (which is itself a smooth function of features).
   Includes transaction cost δ. **This is exactly our pnl formula** with
   F_t being a soft position derived from softmax probabilities.

3. **Differential Sharpe ratio (online)**:
   ```
   D_t = (B_{t-1}·ΔA_t - 0.5·A_{t-1}·ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}
   ```
   where A_t, B_t are EWMA of returns and squared returns.
   Maximising D_t is equivalent to online Sharpe maximisation.

4. **Empirical**: RRL trading the S&P 500 monthly 1970–1994 outperformed both
   buy-and-hold and Q-learning approaches.

## What we'd borrow

* The **reward shape** R_t is exactly our PnL formula. This is the universal
  framing that justifies S2/S4.
* **Smooth-position output**: F_t = tanh(...) — but our submission needs
  discrete labels, so we'd post-process.
* **Differential Sharpe** for online updates — N/A for our setting (no
  online inference allowed).

## What we'd skip

* Full RRL (recurrent state across time-steps) — violates our hard constraint
  (Predictor must be stateless across `predict()` calls).
* Sharpe-as-loss — wrong objective for our raw-cum-pnl leaderboard
  (see `r10_pnl_loss.md` S5).

## Critical insight relevant to S2

> "Maximizing the differential Sharpe ratio yields more consistent results
> than maximizing profits, and **both methods outperform a trading system
> based on forecasts that minimize MSE**."

i.e. ANY reward-based objective beats MSE/CE for this kind of metric, even
when the metric isn't exactly Sharpe. This is empirical support for "stop
training on CE, start training on something pnl-shaped".
