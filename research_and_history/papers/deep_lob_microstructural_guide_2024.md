# Deep Limit Order Book Forecasting: A Microstructural Guide

- **Authors**: Antonio Briola 等（UCL group）
- **arXiv**: 2403.09267 (Mar 2024)
- **Links**: https://arxiv.org/html/2403.09267v1

## 核心贡献

1. 用 DeepLOB 在 15 只 NASDAQ 股票上做大规模实证，研究**微观结构属性如何决定深度模型的可预测性**。
2. 把股票按 tick size 分类：small-tick (⟨σ⟩ ≳ 3θ), medium-tick, large-tick (⟨σ⟩ ≲ 1.5θ)，θ = $0.01。
3. **关键发现**：large-tick 股票的可预测性显著高于 small-tick：H10 时 MCC = 0.29 vs 0.11；H50 时 0.36 vs 0.04。原因是 large-tick 股票 spread 紧、depth 集中、book 同质，方向信号更清晰。
4. 提出 **5-day rolling z-score normalization**（特征级），有效应对非平稳，比静态训练集 normalize 好。
5. 类别不均衡处理：对每类采样 5000 样本来 balance（"random and balanced sampling"）；short-tick 在长 horizon 上不均衡更严重。
6. 强调"分类 metric 不能反映实际交易质量"，提议 **transaction-accuracy** metric——这正是我们这次比赛 PnL 评分的核心思想。

## 提出的特征 / 关键观察

- LOB 三个微结构属性决定可预测性：(1) bid-ask spread, (2) liquidity at best quotes, (3) actual LOB depth Ξ（10 个 level 的价差结构）。
- AAPL（medium-tick）反常地表现极强，作者归因为**极高的成交频率**——成交频率本身可以作为特征。

## 对我们比赛的可借鉴点

- **强烈推荐 5-day rolling z-score 归一化**作为预处理 baseline。我们 120 天 × 5 sym，可以按 sym × 5-day 滚窗。
- **类别 balanced sampling**：我们的 label=1（平）占多数，应该按 class 采样而不是直接 cross-entropy。
- **按 sym 分组训练 / 评估**：5 只 sym 可能 tick size / 流动性差异大，要么加 sym embedding，要么单独建模。
- **transaction-accuracy 思路**：我们已经有 PnL 评测器，可以直接训练时用 PnL-aware loss。
- **成交频率（活跃度）作为特征**：每 tick 的 trade count、order_intst 总量等是预测能力的代理。

## 我的评分

- 实现成本：low（主要是预处理）
- 优先级：**high**（5-day rolling z-score 是必做的预处理改进）
