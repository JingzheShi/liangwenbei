---
name: LOB-Bench — Benchmarking Generative AI for LOB (Nagy 2025)
description: 第一个 LOB 生成模型的分布性 benchmark。LOBS5 (S5 SSM) 全面赢；所有 generative 模型在长 horizon 都"derail"
type: project
arxiv: 2502.09172v1
date: 2025-02
authors: Peer Nagy, Sascha Frey, Kang Li, Bidipta Sarkar, Svitlana Vyetrenko, Stefan Zohren, Anisoara Calinescu, Jakob Foerster
priority: ★★ — 信号：长 horizon 难，避免在 h_60 寄予过高期望
---

# LOB-Bench

## TL;DR

**首个 distributional LOB generative benchmark**，测了 5 个生成模型（包括 35M S5 状态空间模型 LOBS5、170M RWKV、cGAN、parametric）在 GOOG/INTC 2022-2023 上的分布距离、市场冲击曲线、判别器对抗。

**核心结论**：
1. **LOBS5（S5 SSM-based）全面赢**
2. **所有生成模型 horizon 加长后剧烈漂移**（"model derailment"）
3. 即使统计对齐很好，**判别器仍 ROC ≈ 0.83 区分真假** → 微观结构细节难复制

## 1. 评估维度

- **Unconditional**：spread, OFI, volumes, inter-arrival
- **Conditional**：(time of day, vol level)
- **Market impact response**：response function of price to order
- **Discriminator**：classifier real vs synthetic ROC

## 2. 模型对比

| Model | Params | 优势 | 弱点 |
|---|---|---|---|
| **LOBS5** | 35 M (S5 SSM) | 长程依赖、impact curve 真实 | small-tick stocks 边际优 |
| RWKV-4/6 | 170 M | linear 推理 | 分布距离次优 |
| cGAN (Coletta) | — | 模式丰富 | 难训练、impact curve 偏 |
| Parametric (Cont) | — | 可解释 | 不能复制非线性 |

## 3. **关键警告：Derailment**

> "The further out the prediction horizon, the worse is the model performance."

**具体**：
- 在 ~ 几十至几百 message-step 内，error compounding 让 generative 模型偏离真实分布
- 即使每步 KL 距离小，复合后整段轨迹严重偏

## 4. 我们项目的启示

### 4.1 Multi-horizon h_60 比 h_10 更难

我们的 h_10 vs h_60 :
- h_10 ~ 10 ticks ≈ 100 ms ~ 几秒（高 SNR 区）
- h_60 ~ 60 ticks ≈ 几秒至几十秒（noise 主导）

**意味**：
- h_10 增益预算 > h_60 增益预算
- 凡是 R31 推荐的 lever（ReVol / Filter / SG / MoE / VAR-residual）都 **优先 ablate h_10**，再看 h_60 是否同向
- h_60 上的预测 noise 主要来自 derailment，不是模型不够大

### 4.2 我们任务是 directional prediction，不是生成

LOB-Bench 是 generative benchmark，**与我们任务不同方向**。但 derailment 同理：在长 horizon 上**模型预测的不确定性增长比 horizon 快**。

### 4.3 不要做 cross-horizon imitation

> 不要用 h_10 model output 去 simulate h_60 → 就是 derailment 现象的具体落地

## 5. 与我们既有 setup 比较

| 我们的做法 | LOB-Bench 验证 |
|---|---|
| h_10 + h_60 双 horizon 各训独立 model | ✅ 正确（避免单 model 长 rollout） |
| Multi-horizon stacking（meta over h_10/h_60 OOF）| ✅ R12 已规划 |
| h_60 上预算更少 | ✅ 与 derailment 一致 |

## 6. 不要做的事

- ❌ 用 generative LOB model（LOBS5 / RWKV / cGAN）来给我们 directional 信号
- ❌ 在 h_60 上寄予 +20+ PnL 增益预期（derailment 限制）
- ❌ 做 horizon-extrapolation（用 h_10 forecast 多步推到 h_60）

## 7. 来源

- arxiv: https://arxiv.org/abs/2502.09172
- v1 html: https://arxiv.org/html/2502.09172v1
- code: https://github.com/peernagy/lob_bench
