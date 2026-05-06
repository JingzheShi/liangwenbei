---
name: ReVol — Return-Volatility Normalization (Lee 2025)
description: Per-sample log-return / volatility normalization to mitigate cross-stock and temporal distribution shift; tested in US/CN/UK/KR markets, IC +0.03, Sharpe +0.7
type: project
arxiv: 2508.20108v2
date: 2025-08
authors: Hyunwoo Lee, Jihyeong Jeon, Jaemin Hong, U Kang (SNU)
priority: ★★★★★ — Top 1 推荐尝试
---

# ReVol — Return-Volatility Normalization

## TL;DR

为每个 sample 估计 (μ̂, σ̂) 后，把 raw log-return 转换为 ε（GBM-启发 closed form），让所有 sample（不同股票、不同时间窗）的 ε 分布一致。**与 RevIN/Dish-TS 区别**：不仅对齐 mean/var，还吸收 skew/kurt（高阶矩）的部分差异。

**关键属性：**
- 纯 sample-wise 函数 → **stateless 兼容**（用过去 window 估 μ̂/σ̂，不跨调用）
- 与 backbone 无关：LSTM / GRU / Transformer / GBDT 都受益
- LOSO-style cross-market 测试 IC + 0.03、SR + 0.7

## 1. 三模块详解

### Module A: Return-Volatility Normalization (RVN)

输入：raw OHLC 价格 $S_t = (S_t^o, S_t^c, S_t^h, S_t^l)$。

闭收价 normalize（GBM-Itô 形式）：

$$\varepsilon_t^c = \frac{1}{\hat\sigma}\left[\log\frac{S_t^c}{S_{t-1}^c} - \left(\hat\mu - \frac{1}{2}\hat\sigma^2\right)\right]$$

开高低同理（intraday 时刻 τ ∈ (0,1) 调整）：

$$\varepsilon_t^o = \frac{1}{\sqrt{\tau}\,\hat\sigma}\bigl[\log\frac{S_t^o}{S_{t-1}^c} - (\hat\mu - \tfrac12\hat\sigma^2)\tau\bigr]$$

→ ε_t 在 GBM 假设下应 ~ 𝒩(0,1)。

### Module B: Return-Volatility Estimator (RVE)

用 attention-LSTM 在过去 window 上学**带权**的 (μ̂, σ̂)：

$$\hat\mu = \sum_{t=1}^{T} \alpha_t \cdot \log\frac{S_t^c}{S_{t-1}^c}, \quad \hat\sigma^2 = \sum_{t=1}^{T} \alpha_t \cdot (\log\frac{S_t^c}{S_{t-1}^c} - \hat\mu)^2$$

其中 α_t = softmax(MLP(LSTM_hidden_t))。这套 **down-weight** 异常高波动 / 跳空，比简单滑动均值更鲁棒。

### Module C: Return-Volatility Denormalization (RVD)

预测出 ε̂_{T+1} → 反变换：

$$\hat S_{T+1}^c = S_T^c \cdot e^{\hat\mu + \tfrac12 \hat\sigma^2} \cdot e^{\hat\sigma \cdot \hat\varepsilon_{T+1}^c}$$

## 2. 我们的落地形式

### 2.1 适配硬约束（无 cross-call state）

我们的 sample 是固定历史窗口。**只用 sample 内部** 估 μ̂, σ̂：

```python
def revol_normalize_features(price_window, vol_window, lookback=64):
    """
    price_window: shape [T] log-mid-price
    vol_window:   shape [T] log-volume

    Returns: eps_p [T-1], eps_v [T-1]
    Stateless: depends only on the input window, no cross-sample state.
    """
    log_ret = np.diff(price_window)           # [T-1]
    mu_hat = log_ret[-lookback:].mean()
    sigma_hat = log_ret[-lookback:].std() + 1e-6
    eps_p = (log_ret - mu_hat + 0.5 * sigma_hat**2) / sigma_hat

    log_vol_change = np.diff(vol_window)
    vmu = log_vol_change[-lookback:].mean()
    vsig = log_vol_change[-lookback:].std() + 1e-6
    eps_v = (log_vol_change - vmu) / vsig

    return eps_p, eps_v
```

### 2.2 Drop-in 到 LightGBM pipeline

替换 build_features.py 中的 `mid_return`, `volume_return` 字段为 `eps_p`, `eps_v`：

- 旧：`mid_return = log(mid[i] / mid[i-1])`
- 新：`mid_return_eps = (log(mid[i]/mid[i-1]) - μ̂) / σ̂`，其中 μ̂, σ̂ 在该 sample 过去 64 步上估

### 2.3 注意事项

- **不要** 用 attention-LSTM 估 μ̂/σ̂（cost too high；rolling mean/std 已经足够）
- σ̂ 加 EWMA 半衰期 t_half = 32 → 更平滑、更短延迟
- 验证：训练前对一份 sample，确认 ε_p 接近 𝒩(0,1)（QQ-plot）

## 3. 与我们既有路线的关系

| 与 R10 / R11 / R12 / R13 的关系 | 说明 |
|---|---|
| R10 (PnL loss) | **互补**。ε 是 input 变换，loss 是 output 端；ε 让 logit 更稳，PnL loss 让 logit 对齐 leaderboard |
| R11 (cross-stock OOD) | **直接强化**。ReVol 是 P11 R11 中 "global normalization" 类的更高阶版本；可与 Group DRO 叠加 |
| R12 (ensemble) | **正交**。ReVol 影响 input；ensemble 影响 model |
| R13 (calibration) | **正交**。calib 影响 output |
| aug_a (cross-sym data aug) | **正交并互补**。aug_a 增加样本多样性；ReVol 让混入的 cross-sym 样本更同分布 |

## 4. 预期增益（基于 paper + 本项目特性）

| 预测 | 推理 |
|---|---|
| LOSO h_10 + 3 ~ + 8 | Paper 在 cross-market（更难）任务 IC + 0.03；我们 LOSO h_10 IC ~0.6，即 2-3 % relative，绝对 + 3 ~ + 8 PnL 合理 |
| LOSO h_60 + 1 ~ + 4 | Long-horizon noise dominates；ε 提升幅度小 |
| Local val（in-distribution）+ 0 ~ + 2 | In-distribution 时受益最小；ReVol 主要帮 OOD |

## 5. 风险与失败模式

- **σ̂ → 0 退化**：极静止区间会让 ε 爆炸；务必加 σ̂ + ε_floor (1e-6) 并 clip ε ∈ [-10, +10]
- **μ̂ 估错方向**：lookback 太短时 μ̂ 受 noise 主导；推荐 lookback ≥ 32
- **与现有 raw return feature 共存**：建议 **替换**（不是 append），否则 LightGBM 会学到 raw vs eps 的差，等于偷偷重建 μ̂

## 6. 实施 checklist

- [ ] 在 `src/features/` 新增 `revol.py`
- [ ] 单元测试：synthetic GBM 路径 → ε 应近似 𝒩(0,1)
- [ ] integration test：full 5-fold CV，对比 raw vs ε
- [ ] LOSO test：5-seed × 5-LOSO，记录 LOSO h_10 / h_60 cum_pnl 差异
- [ ] WandB log: project="aug-a-r31-revol"

## 7. 来源

- arxiv: https://arxiv.org/abs/2508.20108
- v2 html: https://arxiv.org/html/2508.20108
- 启发文献：RevIN (ICLR 2022), Dish-TS (2302.14829)
