---
name: Order Book Filtration & Directional Signal Extraction (Anantha 2025)
description: 三套实时 stateless 噪声 filter（lifetime / modification-count / modification-time）显著改善 OBI/OFI 与未来收益的相关性
type: project
arxiv: 2507.22712
date: 2025-07
authors: Aditya Nittur Anantha, Shashi Jain, Prithwish Maiti (IISc / SigmaQuant)
priority: ★★★★★ — Top 2 推荐尝试
---

# Order Book Filtration

## TL;DR

**问题**：高频 LOB 中大量"假报价"（fleeting / spoofing / quote stuffing）会让 OFI、OBI 噪声极重。
**方案**：用三套**可观测、实时、stateless** 的 filter 把这些假报价从 OFI/OBI 计算中剔除。
**结果**：filtered OBI 与未来收益的 Pearson r 显著提升；regime-classification 提升；Hawkes 因果激励有限改善。
**关键 takeaway**：**trade-based OBI（基于真实成交方向）的因果对齐优于任何 LOB-derived 信号**。

## 1. 三套 Filter 详解

### Filter 1: Lifetime ℱ_T

```
若 order.lifetime_ms < τ_T:
    drop from OFI/OBI calculation
```

**经验阈值** τ_T ∈ [50, 200] ms（论文推荐先用 100 ms）。
**目标**：过滤毫秒级 cancel 的 fleeting orders（无真实执行意图）。

### Filter 2: Modification Count ℱ_M

```
若 order.modification_count > k_M:
    drop
```

**经验** k_M ∈ [3, 10]。
**目标**：过滤反复改单的 quote-stuffing 行为。

### Filter 3: Modification Time ℱ_M̃

```
若 last-modification timestamp 与 cancel timestamp 间隔 < τ_M̃:
    drop
```

**目标**：过滤"先改后撤"的 spoofing pattern（特意把 quote 改到激进价位再瞬撤）。

## 2. 我们项目的可行性分析

### 2.1 数据字段需求

| Filter | 需要字段 | 我们数据有? |
|---|---|---|
| Lifetime | order_id 全生命周期（add → cancel）| ❓ 视 raw 数据决定 |
| Mod-count | per-order modification 计数 | ❓ |
| Mod-time | 改单时间戳 | ❓ |

**第一步：验证数据**。先在 `notebooks/` 跑一个 EDA，确认 raw 数据是否有 message-level event 流（add / modify / cancel / trade）还是仅有 snapshot 5×（bid/ask × price/vol）×T。

### 2.2 如果只有 snapshot 数据（最坏情况）

仍可用**近似版**：

| 近似 feature | 公式 | 反映啥 |
|---|---|---|
| `bbo_update_count_per_window` | best-bid 或 best-ask 在 t-30s..t 内变化次数 | 高 = 噪声 / 报价撤换密集 |
| `quote_stability` | sum(`bid[1]_price unchanged at adjacent t`) / window_len | 越高越稳 |
| `top_level_volume_volatility` | std(`bid[1]_vol`) / mean(`bid[1]_vol`) | 高 = 非真实流动性 |
| `aggressive_modification_proxy` | (bid[1]_price 跳变次数) / (bid[1]_vol 跳变次数) | 价跳变多但 vol 不变 → 改单未成交 |

→ 当某 sample 的 `bbo_update_count` > p90 时，**给 OFI 一个降权系数 / 或在该窗口内重新算 OFI 时仅用 stable update 的样本**。

### 2.3 如果有 message data（最佳情况）

**直接照 paper 实现**：

```python
def filtered_ofi(messages, lifetime_threshold_ms=100):
    """
    messages: list of (event_type, order_id, ts, price, size)
    event_type ∈ {'A','M','C','T'}
    Returns: filtered OFI (one value per snapshot)
    """
    order_lifetimes = compute_lifetimes(messages)  # {order_id: lifetime_ms}
    valid_orders = {oid for oid, lt in order_lifetimes.items() if lt >= lifetime_threshold_ms}

    filtered_msgs = [m for m in messages if m.order_id in valid_orders]
    return compute_ofi(filtered_msgs)
```

## 3. 期望 LOSO 增益

| 路径 | 预期 |
|---|---|
| 有 message data + filter 全部启用 | + 3 ~ + 5（来自 paper：filtered OBI 信号-收益 Pearson r 提升 30-50 %） |
| 仅 snapshot 近似 | + 1 ~ + 3（信号清洁度有限提升） |
| **trade-based OBI** = aggressor-side flow（如果数据中有 buy-initiated / sell-initiated 标记）| + 2 ~ + 4 强烈推荐（paper 显式说优于 LOB-derived） |

## 4. 实施 roadmap（≤ 2 天）

- **Day 1 上午**：EDA — 看数据是否有 message stream / lifetime 字段
- **Day 1 下午**：实现 trade-based OBI（如果有 trade direction / aggressor flag）
- **Day 2 上午**：实现 filtered OFI（lifetime filter）或近似版
- **Day 2 下午**：5-seed CV + LOSO，对比 raw OFI vs filtered OFI

## 5. 与我们 setup 的兼容性

- ✅ **stateless**：filter 是 sample-wise pure function，无跨 predict 调用 state
- ✅ **sym-agnostic**：filter 不引用 sym ID
- ✅ 不需 date：纯基于 microstructure event
- ✅ 与 ReVol（P17）正交：ReVol 改 input scale，filter 改 input quality

## 6. 风险

- 若 lifetime threshold 太严：drop 掉真实 limit order，OFI 反而 underestimate
- 若数据是 5-level snapshot only（无 message）：filter 退化为粗略代理，gain 缩水到 + 1 ~ + 2

## 7. 来源

- arxiv: https://arxiv.org/abs/2507.22712
- 同作者前作（OFI-Hawkes，与 P15 同篇）：arxiv 2408.03594
