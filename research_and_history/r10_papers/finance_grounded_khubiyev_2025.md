# Finance-Grounded Optimization For Algorithmic Trading
**Khubiyev et al. — arxiv 2509.04541 (Sept 2025, 17 pages, 6 fig, 5 tab)**
URL: https://arxiv.org/abs/2509.04541 / html: https://arxiv.org/html/2509.04541

## Abstract (extracted)

Introduces "financially grounded loss functions derived from key quantitative
finance metrics: Sharpe ratio, Profit-and-Loss (PnL), Maximum Drawdown." Adds
"turnover regularization" to constrain position turnover. Shows finance-grounded
losses **outperform MSE** for return-prediction tasks when judged by trading
metrics (Sharpe, PnL, MDD).

## Key formulas (extracted directly)

**SharpeLoss (differentiable)**:
```
SharpeLoss = E(pnl) / (Var(pnl) + ε)
```

**PnLLoss**:
```
PnLLoss = -α · r
```
α: position size, r: realised return.

**MDDLoss** (drawdown):
```
MDDLoss = -min_t DD_t       where DD_t = C_t - max_{u≤t} C_u
```

**RiskAdjLoss (composite)**:
```
RiskAdjLoss = -E(pnl) + λ·DrawDown + γ·(α-r)²
```

## Results (Table 3, top performers)

| Model | Sharpe | MaxDD | PnL (%) |
|---|---|---|---|
| LSTM **MSELoss** baseline | **−0.46** | — | — |
| LSTM LogMDDLoss | 1.76 | -0.064 | 16.17 |
| LSTM MDDLoss | 1.69 | -0.067 | 15.87 |
| LSTM ModSharpeAbsLoss+ClassicalTvr | 1.56 | -0.145 | 18.72 |

**Key takeaway**: replacing MSE with a finance-grounded loss flipped Sharpe
from −0.46 to +1.76 on the same architecture. This is the strongest empirical
evidence we have that loss-function alignment > model architecture.

## What we'd borrow

1. **`PnLLoss = −α·r` is exactly our S4** for the NN classifier head — replace
   `nn.CrossEntropyLoss` with reward-weighted softmax expectation. Drop-in.

2. **Position-from-prediction**:
   * For regression heads: `α_t = sign(prediction_t)·magnitude_t`
   * For us (classification): `α_t = (label_pred_t − 1)`, exactly the structure
     in our `pnl_single` formula.

3. **Turnover regulariser**: irrelevant for us (no transaction cost across t).

4. **MDDLoss**: irrelevant — leaderboard scores `Σ pnl`, not drawdown.

## What we'd skip

* Sharpe-only training: their own table shows SharpeLoss-trained models often
  have lower raw PnL than PnLLoss-trained ones. Our metric is raw PnL.

## Implementation hint

The paper trains LSTM end-to-end. For us this would map to Scheme C MLP/CNN
heads (S4 in `r10_pnl_loss.md`). For LightGBM the same idea (S2) needs the
custom multiclass objective with closed-form gradient/hessian.
