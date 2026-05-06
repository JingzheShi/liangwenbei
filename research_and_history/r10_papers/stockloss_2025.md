# Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions
**arxiv 2507.19639 (July 2025)**
URL: https://arxiv.org/abs/2507.19639 / html: https://arxiv.org/html/2507.19639v1

## Abstract (extracted)

Proposes **four novel loss functions** for end-to-end stock allocation
(buy/short/hold across N stocks). Tested on 50 S&P 500 stocks with Crossformer
backbone. Best loss (StockLoss-L2) returned 51.42% / 51.04% / 48.62% in three
test years vs PPO/DDPG max ~41% / 2.81% / 41.58%. **Directly maximising profit
beat RL methods consistently.**

## The four loss functions (verbatim from the html)

For N stocks, output `Ô = (V̂_1, …, V̂_N+1)` where V̂_{N+1} is "hold" weight,
sign(O_i) = ±1 buys/shorts.

### Loss I — StockLoss (raw profit)
```
ℒ = -[Σ_i V̂_i · (Ret_{i,t+1} - Ret_{i,t}) · sign(O_i) + 𝕀(e→1)·H(V)]
```

### Loss II — StockLoss-Max (max-normalised)
```
ℒ = 1 - Σ_i V̂_i · ((Ret_{i,t+1} - Ret_{i,t}) / max_j (Ret_{j,t+1} - Ret_{j,t})) · sign(O_i) - 𝕀(e→1)·H(V)
```

### Loss III — StockLoss-L2 (squared, best performer)
```
ℒ = 1 - sqrt[ Σ_i V̂_i · ((Ret_{i,t+1} - Ret_{i,t}) / max_j (...))² + 𝕀(e→1)·H(V)² ]
```

### Loss IV — StockLoss-Norm (normalised by total magnitude)
```
ℒ = 1 - Σ_i |O_i|·(Ret_{i,t+1} - Ret_{i,t})·sign(O_i) / Σ_j |O_j|·(Ret_{j,t+1} - Ret_{j,t}) - 𝕀(e→1)·H(V)
```

`𝕀(e→1)·H(V)` is an entropy-regularising bonus only in the early epochs
(decays over training).

## Architecture

Output layer = `N + 1` neurons. tanh activation (so V̂ ∈ [-1, 1]).
Backbone: Crossformer / DeformTime / DLinear (transformer family).
Best: Crossformer + StockLoss-L2.

## Results (Table 3)

| Model | 2023 | 2022 | 2021 |
|---|---|---|---|
| Crossformer + StockLoss-L2 | **51.42%** | **51.04%** | **48.62%** |
| DeformTime + StockLoss-L2 | 51.75% | 53.18% | 39.23% |
| DLinear + StockLoss-L2 | 39.11% | 53.91% | 50.94% |
| PPO (best RL) | 27.31% | 2.81% | 41.58% |
| Buy & Hold | 27.77% | -20.64% | 36.05% |

## What we'd borrow

1. **The decaying entropy regulariser** `𝕀(e→1)·H(V)` is the same trick we need
   in our S2/S4 to prevent collapse to "always flat". Schedule β_entropy from
   0.05 in epoch 1 down to 0.0 by epoch 10.

2. **L2-normalisation framing (StockLoss-L2)** turns "raw expected return" into
   a bounded loss in [0, 1] which trains more stably. Could adapt for our
   3-class softmax: `L = sqrt(Σ_k p_k²·r_k²)` instead of `Σ_k p_k·r_k`. Worth
   trying as a stability variant of S2.

3. **Multi-stock joint training is irrelevant for us** (we predict per-tick
   single-instrument). Single-stock setting reduces to S4/S2.

## What we'd skip

* Crossformer architecture — overkill for our 100×D LOB tensor and we already
  have results showing MLPLOB/LightGBM dominate transformers here.
* Tanh-bounded continuous output — our submission requires {0,1,2}. See
  `r10_pnl_loss.md` S10 for why we abandon continuous bet sizing.
