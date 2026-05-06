# Hawkes Processes for Limit Order Book Modelling

## 关键参考

- Bacry, Mastromatteo, Muzy (2015) "Hawkes processes in finance" — 综述
- Lu, Abergel (2018) "High dimensional Hawkes processes for LOB" — 高维 Hawkes
- Morariu-Patrichi & Pakkanen (2021) "State-dependent Hawkes processes and LOB" — Tandfonline
- Bortoluzzo et al. (2024) "Deep Hawkes process for high-frequency market making" — Springer
- arXiv 2503.14814 (2025) "Hawkes processes in high-frequency trading" — 综述

## 核心思想

1. **Poisson 假设不成立**：LOB 订单到达呈现强 self-excitation（一个 buy order 触发更多 buy orders）和 cross-excitation（buy 触发 cancel ask 等）。
2. **Hawkes 强度**：
```
λ_i(t) = μ_i + Σ_j ∫_{-∞}^t φ_{ij}(t - s) dN_j(s)
```
其中 φ_{ij} 是 j 类事件对 i 类事件未来到达率的影响 kernel（常用 exponential decay）。
3. 多类型 marked Hawkes 同时建模 6 类事件：limit buy/sell, market buy/sell, cancel buy/sell。
4. State-dependent Hawkes：基础 intensity μ_i 还依赖当前 LOB 状态（imbalance, spread, depth）。

## 提出的特征 / 应用

### Hawkes-derived 特征（可作为模型输入）
- **Intensity λ_i(t)**：用 EWMA 近似 Hawkes 强度：
```
λ_i^{EW}(t) = α · 1_{event_i at t} + (1-α) · λ_i^{EW}(t-1)
```
不同 α（对应不同 decay scale）给出多尺度 intensity。
- **Cross-excitation ratio**：`λ_buy(t) / (λ_buy(t) + λ_sell(t))` —— Hawkes 版的"order arrival imbalance"。
- **Branching ratio** n = ‖φ‖：度量市场内生性。n→1 时市场不稳定。

## 对我们比赛的可借鉴点

- 我们 schema 已有 `*_intst`（"到达强度"）和 `*_acc`（"强度变化率"），但**没有显式的 cross-excitation**。可以构造：
  - `mb_intst / ma_intst` —— market buy vs market sell 强度比
  - `(lb + cb_inv) / (la + ca_inv)` —— 净 buy 压力（limit buy 和 cancel ask 都是看涨信号）
- **Multi-scale EWMA**：用 α ∈ {0.05, 0.1, 0.3, 0.5} 对每个 intensity 做指数加权平均，让网络看到不同时间尺度。
- 不需要正经拟合 Hawkes（昂贵），用 EWMA 近似就够了。

## 我的评分

- Hawkes EWMA proxies：实现成本 low
- 完整 Hawkes 拟合：成本 high，性价比低
- 优先级：proxies **medium-high**（特别是 cross-excitation 比例）
