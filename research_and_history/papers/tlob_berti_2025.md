# TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with LOB Data

- **Authors**: Leonardo Berti, Gjergji Kasneci 等
- **arXiv**: 2502.15757 (Feb 2025, 多次更新)
- **Links**: https://arxiv.org/abs/2502.15757 ; code: https://github.com/LeonardoBerti00/TLOB

## 核心贡献

1. 提出 **dual attention** Transformer：Temporal self-attention（沿时间）+ Spatial self-attention（沿 LOB 深度方向）+ MLPLOB feed-forward。两路 attention 显式 model 时序和空间结构。
2. 同时提出 **MLPLOB**——一个纯 MLP 模型，feature-mixing MLP（按 row）+ temporal-mixing MLP（按 column），用 GeLU + LayerNorm。
3. **关键观察**：MLPLOB 在短 horizon 上能 match 甚至超过 TLOB，**说明复杂架构未必必需**，数据/表征更重要。
4. FI-2010 上 TLOB 平均 +3.7 F1，Tesla +1.3，Intel +7.7。但作者也指出**预测能力随时间下降**：Intel 从 2012 的 66.87 降到 2015 的 60.19（市场变得更高效）。
5. 加入 transaction cost 后，分类精度的提升不一定转化为盈利——这点和我们的 PnL-based 评分高度相关。

## 提出的特征 / 输入表示

`4L` 维 LOB record（L=10 levels，4 = ask price/volume + bid price/volume）。**Bilinear normalization** 处理非平稳与量级差异。

## 对我们比赛的可借鉴点

- **MLPLOB 是性价比之王**：实现简单、训练快，参数远少于 Transformer。**先跑 MLPLOB 再上 Transformer**。
- **dual attention** 的思路可以借鉴：spatial attention 让模型在不同 LOB level 之间分配权重，比简单 conv 更灵活。
- **预测能力随时间衰减**这件事提醒我们：单纯靠 feature 不够，要持续 retrain，且要小心**look-ahead bias**。
- **Bilinear / 5-day rolling z-score** normalization 是处理非平稳的标准做法，要做。

## 我的评分

- MLPLOB 实现成本：low；TLOB 实现成本：medium
- 优先级：**high**（MLPLOB 是非常强的简单 baseline，必跑）
