# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

- **Authors**: Zihao Zhang, Stefan Zohren, Stephen Roberts (Oxford)
- **arXiv**: 1808.03668 (v1 2018, v3 2020)
- **Venue**: IEEE Transactions on Signal Processing 67(11), 2019
- **Links**: https://arxiv.org/abs/1808.03668 ; code: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## 核心贡献（3-5 句话）

1. 提出第一个端到端、零手工特征的 LOB 中价方向预测网络，输入是 `100×40` 张量（100 个 LOB snapshot × 40 个 raw 量价值），输出是 3 类（涨/平/跌）。
2. 架构 = **Conv stack** → **Inception module**（多尺度时序滤波）→ **LSTM** → softmax。Conv 阶段被设计为先合 price/volume，再合 ask/bid，再合多档。
3. 在 FI-2010 公开 benchmark 和一年伦敦交易所的真实数据上都达到 SOTA，且模型可以在没见过的股票上做迁移（universal features）。
4. 标签 = `mean(p[t+1..t+k]) / mean(p[t-k+1..t]) - 1`，用 ±α 阈值切三类（典型 α≈2e-5）。
5. 这套 100×40 tensor 输入和"先 price-vol、再 ask-bid、再 levels"的 conv 顺序成为后续 LOB 深度学习模型的标准 baseline。

## 提出的特征 / 输入表示

输入是 raw LOB 的 40 维向量（每 tick）：
```
x_t = [p_ask^1, v_ask^1, p_bid^1, v_bid^1, ..., p_ask^10, v_ask^10, p_bid^10, v_bid^10]
```
没有手工 feature。Z-score 归一化用过去若干天的滚动均值方差（5 天滚动 z-score 是后人推荐的）。

## 实验设置

- **FI-2010**：5 只 Helsinki Nasdaq Nordic 股票 × 10 天，~395k events（每 10 LOB updates 取一帧），horizons k ∈ {10, 20, 30, 50, 100} events。
- **LSE**：1 年伦敦交易所，约 50 亿条事件，跨股票 transfer 测试。
- **Metric**：Accuracy / Precision / Recall / F1（balanced macro），3 分类。

## 对我们比赛的可借鉴点

- **直接可拿来当 baseline**：我们的输入已经是 10 档量价 + 衍生，可以裁剪到 40 维原始量价喂 DeepLOB；这是 mmpc_demo 的思路。
- **Conv 顺序**：先把 price/volume 合到一个 channel，再合 ask/bid，再合 levels。如果我们做 CNN，这个顺序很关键。
- **Label 阈值 α**：DeepLOB 的 label 用未来 k tick 的均价 / 过去 k tick 均价 - 1，再用 ±α 切。我们的比赛标签已经给了，但**评测器 sanity check 显示阈值后处理（pred=1 但 logit 偏离很远翻转）有空间**，可以参考 DeepLOB 的 α 选择思路。
- **Universal features**：跨股票训练再迁移在 LSE 数据上奏效——我们有 5 只 sym，应该一个模型 + sym embedding，不要每只单独训。

## 我的评分

- 实现成本：medium（已有开源 PyTorch 实现）
- 优先级：**high**（baseline 必跑 + 是后续 trans 系列的对比基线）
