---
name: Hybrid VAR + FNN for OFI Prediction (Rahman & Upadhye 2024)
description: 两阶段：VAR 学线性，FNN 学 VAR residual 中的非线性。BTC/ETH 上 R²=0.997。可移植为 Ridge backbone + LightGBM-on-residual。
type: project
arxiv: 2411.08382
date: 2024-11
authors: Abdul Rahman, Neelesh Upadhye (IIT Madras)
priority: ★★★ — Linear backbone + GBDT-on-residual 思路
---

# Hybrid VAR + FNN for OFI

## TL;DR

经典 econometrics + 神经网络的简单组合：
1. **VAR(p=2)** 拟合 OFI 的线性 vec-autoregression
2. **FNN(32-64-2)** 学 VAR 的 residual

在 Binance BTC/ETH 上 R² = 0.997 / 0.983，trading accuracy 96.4-98.2 %。

## 1. 模型公式

### 1.1 Stage A: VAR(p) on Order Flow Imbalance

OFI_t = (B_t, S_t)（buy/sell volume），p-lag VAR：

$$\mathbf{OFI}_t = c + \sum_{i=1}^p A_i \mathbf{OFI}_{t-i} + \boldsymbol{\eta}_t$$

OLS 拟合 → 得 $\hat{\mathbf{OFI}}_t^{VAR}$。

### 1.2 Stage B: FNN on residuals

$$\boldsymbol{\eta}_t = \mathbf{OFI}_t - \hat{\mathbf{OFI}}_t^{VAR}$$

FNN(input=η_{t-1, t-2, ...}, hidden=[32, 64], output=2) → $\hat{\boldsymbol{\eta}}_t$。

### 1.3 Final prediction

$$\hat{\mathbf{OFI}}_t = \hat{\mathbf{OFI}}_t^{VAR} + \hat{\boldsymbol{\eta}}_t$$

## 2. 我们的项目落地

我们任务不是预测 OFI 而是预测 mid_price 方向（3 分类）。但**两阶段思想可移植**：

### 2.1 Variant A: Ridge / OLS + LightGBM on residual（连续目标版）

如果有连续 mid-return 训练目标（在 build label 阶段保留 raw Δp）：

```python
# Stage 1 — Ridge on a small set of linear features
linear_feats = [bid_ask_imbalance, OFI_ema_5s, OFI_ema_30s,
                vol_change_log, mid_velocity_3s, ...]  # 5-10 维
ridge = Ridge(alpha=1.0).fit(X_train[:, linear_feats], y_train_continuous)
y_pred_lin = ridge.predict(X_val[:, linear_feats])

# Stage 2 — LightGBM on residual
residual = y_train_continuous - ridge.predict(X_train[:, linear_feats])
lgb_residual = lgb.train({'objective':'regression', ...},
                          lgb.Dataset(X_train, residual))

# Final
y_pred_final = ridge.predict(X_val) + lgb_residual.predict(X_val)
# Then convert to {down, flat, up} via R13 z-gate
```

### 2.2 Variant B: Stacking 形式（与 R12 互通）

Stage 1 = LightGBM small（弱模型）；Stage 2 = LightGBM large 学 residual。但 paper 的精髓是 Stage 1 必须 **线性可解释**，否则 stacking 退化为 R12 的 "two big models".

### 2.3 Stateless 兼容性

✅ 完全兼容。Ridge 模型权重固定；LightGBM 推理 stateless；residual 是 sample-wise 函数。

## 3. 期望增益

| 形式 | 预期 |
|---|---|
| Ridge backbone + LightGBM residual（10-feat 线性 backbone） | + 1 ~ + 3 |
| OLS-on-OFI lag-2 + LightGBM residual | + 0.5 ~ + 2 |

主要 lever：**降低 LightGBM 学习线性结构的方差**（在 LOSO 跨股票时尤其有用，因为线性 OFI lag 跨股票更 invariant）。

## 4. 风险

- 如果 OFI 的线性结构本来就被 LightGBM 第一棵树拟合好了 → Stage 1 没有边际贡献
- 选错线性 features 会让 Stage 2 学不到东西

## 5. 与既有工作的关系

| 工作 | 异同 |
|---|---|
| R12 stacking | R12 用 Ridge meta over LightGBM OOF；本文反过来：linear → NN/GBDT；不同方向 |
| R20 alpha101/158 | R20 是 feature side；本文是 model side |
| Cont & Stoikov OFI 2014 | OFI 经典定义来源（已 R20 覆盖）；本文是 OFI 预测 |

## 6. 来源

- arxiv: https://arxiv.org/abs/2411.08382
- v1 html: https://arxiv.org/html/2411.08382
