# Attention-Based Reading, Highlighting, and Forecasting of the LOB

- **arXiv**: 2409.02277 (Sep 2024)
- **Links**: https://arxiv.org/abs/2409.02277

## 核心贡献

1. 把 **Spacetimeformer**（一种联合时空注意力 transformer）改造用在 LOB 上，做"全 LOB 结构"预测，不只是 mid-price 方向。
2. 引入 **compound multivariate embedding**：对每个属性（bid/ask, price/volume, level, stock）独立 embedding 再合成，参数比单 variable embedding 少很多。
3. 用 **Performer linear attention** 降复杂度到 O(N)。
4. **Percent-change normalization + min-max scaling** 处理非平稳。

## 输入特征

LOBSTER 数据，5 levels × (bid price, bid vol, ask price, ask vol) = 20 features per tick + Time2Vec embeddings。

## 对我们比赛的可借鉴点

- **Compound multivariate embedding** 是个好思路：我们的 154 维特征不应该当成 154 个独立维度，而是结构化的（按 level / 按 type）。如果用 Transformer，对 (level_id, side, type) 做独立 embedding 再 sum / concat 比一股脑投影更高效。
- **Percent-change normalization** 即先取 `pct_change`（相对变化）再 min-max scale——比 z-score 在非平稳数据上有时更稳。值得作为 normalization 的 ablation 实验。
- **Time2Vec** 时间编码（sinusoidal + linear 组合）可以替代 raw 时间戳——对 AM/PM session 区分有用。

## 我的评分

- 实现成本：medium-high
- 优先级：**medium**（embedding 思路值得借鉴；但完整 Spacetimeformer 实现门槛高）
