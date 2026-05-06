---
name: Adaptive Market Intelligence — Volatility-Sensitive MoE for Stock Forecasting (2025)
description: MoE 架构按 volatility regime 路由；volatile asset MSE -33%、stable -28%。可移植为 GBDT 双 model + soft gate
type: project
arxiv: 2508.02686
date: 2025-08
priority: ★★★ — 中等优先级；对应 §0 Top 4 Volatility-Conditional MoE
---

# Adaptive Market Intelligence MoE

## TL;DR

把 MoE 架构 specialize 成 **per-volatility-regime expert**，让模型对高波动 / 低波动 sample 用不同权重。Volatile asset MSE -33 %，stable -28 %。

## 1. 关键 idea

NN 内嵌的 MoE 路由（gating network）对 volatility 敏感。我们 GBDT 没有 MoE，但可**双 model + soft gate** 模拟。

## 2. 我们的落地（GBDT 版）

### 2.1 双 model 训练

```python
# 训练时
realized_vol = df['mid_return_30s_std']    # sample-wise feature

low_vol_mask  = realized_vol <= np.percentile(realized_vol, 70)
high_vol_mask = ~low_vol_mask

lgb_low  = train_lgb(X[low_vol_mask],  y[low_vol_mask])
lgb_high = train_lgb(X[high_vol_mask], y[high_vol_mask])
```

### 2.2 推理时 soft gate（stateless）

```python
# 推理时
for sample in test_samples:
    rv = sample['mid_return_30s_std']
    p70 = REFERENCE_P70  # 训练集统计的常数，存权重里
    sigma = REFERENCE_P70_SCALE
    w_high = sigmoid((rv - p70) / sigma)

    p_low  = lgb_low.predict(sample)
    p_high = lgb_high.predict(sample)
    p_final = w_high * p_high + (1 - w_high) * p_low
```

`REFERENCE_P70`、`sigma` 是从训练集算的固定数 → ✅ stateless。

### 2.3 注意

- `realized_vol` 必须是 sample 内部计算的 **过去** vol，不是全局
- 不能用未来；用 `np.std(mid_return[t-30:t])`
- 也可 fold 进多 horizon stacking：每个 horizon 一对 (lgb_low, lgb_high)

## 3. 期望增益

+ 1 ~ + 4。主要 lever：
- 高 vol sample 信号-噪声比低 → 单独训练让 leaf split 更细分
- 低 vol sample 移动小 → 损失函数权重不同

## 4. 与 R12 ensemble 的区别

R12 ensemble：多 model **平均** （或 stacking 学权）
本方案：多 model **switch by sample feature**

可叠加：(lgb_low_seed1 + lgb_low_seed2) gated with (lgb_high_seed1 + lgb_high_seed2)。

## 5. 风险

- p70 阈值如果选错 → 两 model 中一个 sample 极少 → 过拟合
- vol distribution 在 cross-sym 间 shift → 阈值需 sym-agnostic 选择（比如全局 p70）

## 6. 来源

- arxiv: https://arxiv.org/abs/2508.02686
