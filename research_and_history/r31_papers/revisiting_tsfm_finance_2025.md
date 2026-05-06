---
name: Re(Visiting) Time Series Foundation Models in Finance (Rahimikia 2025)
description: 第一个 TSFM 在金融大样本（2B 收益、94 国、34 年）上的全面实证；off-the-shelf zero-shot 严重失败，from-scratch finance pretraining 才能成功
type: project
arxiv: 2511.18578
date: 2025-11
authors: Eghbal Rahimikia, Hao Ni, Weiguan Wang
priority: ★★★ — Negative result, 验证我们 GBDT 路线合理
---

# Re(Visiting) TSFMs in Finance — 关键 negative result

## TL;DR

**第一篇** 系统评估时序 foundation model 在 **金融领域** 表现的论文。结论 **支持我们 GBDT 主路线**，并 **反对** 任何 "拿 Chronos / TimesFM 直接套" 的尝试。

## 1. 实验规模

- **2 billion** daily excess return 观测
- **94 个国家**
- **34 年** 历史
- 三种 setting：(a) zero-shot pre-trained → (b) fine-tune on financial data → (c) **from-scratch financial pretraining**
- 对比 baseline：Linear / Ensemble / NN / **CatBoost** / XGBoost

## 2. 关键发现

### 2.1 Zero-shot 失败

| Model | OOS R² (%) |
|---|---|
| TimesFM 500 M | **−2.80 %** |
| Chronos large | **−1.37 %** |
| **CatBoost** | **−0.10 %** |

**通用 TSFM 比简单 CatBoost 还差**。

### 2.2 Fine-tune 也救不回来

> "most fine-tuned TSFM performance deteriorates, except for Chronos (large)"

**Fine-tune 经济价值依然不足**（portfolio construction Sharpe 不达标）。

### 2.3 From-scratch finance pretrain 是唯一出路

| Setting | Chronos-small R² | 年化 return | Sharpe |
|---|---|---|---|
| Zero-shot | −77.07 % | — | — |
| Pretrain from scratch on financial data | −3.18 % | — | — |
| + global data + synthetic aug + JKP factors | — | **41.89 %** | **5.42** |
| + Hyperparam optim | — | — | (matches SOTA ensemble) |

## 3. 推论 / 建议

### 3.1 直接验证我们 GBDT 主路线

> CatBoost OOS R² = −0.10 % > TimesFM −2.80 %

**LightGBM / CatBoost 是金融 short-horizon 的合理 default**，不要因为 "transformer / foundation model 时髦" 而切。

### 3.2 不要试通用开源 TSFM zero-shot

跳过：Chronos-zero-shot、TimesFM-zero-shot、Moirai-zero-shot。

### 3.3 如果必须做 NN，**from-scratch** + finance data

- 可考虑：在我们 5 sym × full history 上 from-scratch pretrain 一个 small Transformer (10-50 M)
- 不考虑：fine-tune Chronos / TimesFM

### 3.4 增强信号：global data + synthetic aug + JKP factors

- **JKP factors** = 经典金融因子（Jensen Kelly Pedersen），已在 R20 (alpha101 / alpha158) 部分覆盖
- **synthetic aug** = aug_a 思想验证

## 4. 我们的执行清单（基于 P3 结论）

- [x] 继续 LightGBM 主路线（已验证合理）
- [x] R20 alpha101/158 features 已加（JKP-style）
- [x] aug_a (cross-sym 数据增强) 已部署（synthetic-like）
- [ ] **明确放弃** generic TSFM zero-shot 实验
- [ ] 如果未来上 NN baseline → 选 MLPLOB（P9，简单）而非 generic TSFM

## 5. 局限

- 论文研究的是 **daily excess return**，不是 **sub-second LOB**。
- LOB 粒度上 TSFM 是否可行 **尚未严测**（但常理外推：粒度更细、噪声更高、generic TSFM zero-shot 应该更糟）。

## 6. 来源

- arxiv: https://arxiv.org/abs/2511.18578
- v1 html: https://arxiv.org/html/2511.18578v1
