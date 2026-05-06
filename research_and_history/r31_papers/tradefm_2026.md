---
name: TradeFM — Generative Foundation Model for Trade-flow (J.P. Morgan AI 2026)
description: 524M-param Transformer pretrained on Level-3 trade messages of 9K+ equities; introduces "scale-invariant features" + "universal tokenization" — strong endorsement of sym-agnostic feature design
type: project
arxiv: 2602.23784
date: 2026-02
authors: Maxime Kawawa-Beaudan, Srijan Sood, Kassiani Papasotiriou, Daniel Borrajo, Manuela Veloso (J.P. Morgan AI Research)
priority: ★★★ — Idea-borrow only, 不可直接复现
---

# TradeFM

## TL;DR

**第一个 Level-3 trade-flow 上的 generative foundation model**，524 M 参数，9 K+ equities。最有学习价值的 **不是模型本身**（524 M 我们跑不动），而是它的 **"scale-invariant features + universal tokenization"** 哲学——**强烈对应** 我们的 sym-agnostic 硬约束。

## 1. 关键论点

### 1.1 Scale-Invariant Features

**Definition**：一组**不依赖资产特异性 calibration** 的市场微观结构表征。

举例（论文未列详尽，但可推断）：
- `relative_spread = (ask - bid) / mid`（绝对价差不可比，相对价差可比）
- `volume_ratio = vol_at_level / total_book_vol`（绝对 vol 不可比，比例可比）
- `time_since_last_trade / mean_inter_trade_time`（时间归一化）
- `OFI / sqrt(total_volume_in_window)`（vol-scaled OFI）
- `microprice - mid` divided by `tick_size_in_mid_units`

**思想**：**所有特征构造完后必须是无量纲、跨股票同分布的随机变量**。

### 1.2 Universal Tokenization

把 (event_type, level_relative_to_mid, log_size_bucket, time_bucket) 四元组离散化为统一 token，每个股票/市场都用同一个 vocabulary。

**意义**：cross-asset zero-shot transfer 不需要 per-asset embedding。

## 2. 实证结果

| Metric | TradeFM | Compound Hawkes baseline |
|---|---|---|
| 分布距离（pretraining domain） | 1× | 2-3 × 高（worse） |
| Stylized facts (heavy tails, vol cluster, no return autocorr) | ✅ reproduced | partial |
| **APAC OOD zero-shot** | ✅ moderate degradation | failed |

## 3. 我们的项目可借用的具体 idea

虽然不可复现 524M 模型，**以下 1-3 条可直接落地**：

### 3.1 Audit 当前 features 的 scale-invariance

把每个 LightGBM input feature 在 cross-sym 维度上做以下检查：

```python
# 5 个 sym 的同名 feature 应该 stat 同分布
for feat in features:
    for s in [0,1,2,3,4]:
        plot histogram of feat where sym==s
    # 用 KS test 两两对比；若 KS p-value < 0.01 → 该 feature scale-dependent
```

→ 任何分布显著不同的 feature 都需要 normalize / scale。

### 3.2 引入 spread/volume 比例化

把所有绝对量（spread, volume, OFI）替换或追加比例化版本：

| 原 feature | 比例化版本 |
|---|---|
| `spread = ask - bid` | `rel_spread = (ask - bid) / mid` |
| `bid_vol_l1` | `bid_vol_l1 / total_book_vol` |
| `OFI` | `OFI / sqrt(total_window_volume)` |

### 3.3 时间相对化（去除 wallclock 依赖）

把 `time_since_open` 等绝对时间字段替换为相对时间间隔（与 last trade、last OFI peak 等的相对距离）。

## 4. 与我们硬约束的高度对齐

| 我们的硬约束 | TradeFM 设计 |
|---|---|
| sym-agnostic（禁 sym embedding）| Universal tokenization across 9 K equities，无 per-asset emb |
| stateless predict | TradeFM 自身有 state，但 features 设计独立于 calibration |
| date 被置 0 | TradeFM 用相对时间 token |

→ **TradeFM 是我们硬约束的"超大规模合理化证明"**。如果 J.P. Morgan AI Research 在 9 K 股票上达到 zero-shot OOD，我们在 5 sym 训练上做同类 invariance 设计应该收益更大。

## 5. 不可借用的部分

- ❌ 524 M Transformer：compute 不可行，且 stateless 推理（每次 predict）不能用大模型
- ❌ Pretrain on 9 K equities：我们只有 5 sym
- ❌ Decoder-only autoregressive head：不适合分类任务

## 6. 建议执行

- **小项目（1 day）**：scale-invariance audit（§3.1）
- **小项目（1 day）**：把 5-10 条绝对量 feature 比例化（§3.2）
- 不上 TradeFM 模型本身

## 7. 来源

- arxiv: https://arxiv.org/abs/2602.23784
- HuggingFace: https://huggingface.co/papers/2602.23784
- ICLR 2026 submission
