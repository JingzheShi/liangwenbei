# An Efficient Deep Learning Model to Predict Stock Price Movement Based on Limit Order Book

- **Authors**: 中国研究组（A 股 LOB 应用）
- **arXiv**: 2505.22678 (May 2025)
- **Links**: https://arxiv.org/abs/2505.22678 ; https://arxiv.org/html/2505.22678v1

## 核心贡献

1. **核心 insight**：LOB 数据天然有 **bid/ask 对称性**——bid 侧和 ask 侧是镜像结构。当前主流模型把它们当做无关 feature concat 起来处理，浪费了这个先验。
2. 提出 **Siamese (双胞胎) 架构**：两个 encoder 分别处理 bid 侧和 ask 侧，**参数共享**——这样 bid_extractor(x_bid) 和 ask_extractor(x_ask) 学到的是"同一种特征提取规则"。
3. 把 Siamese 应用到 5 个 baseline：MLP, LSTM, MLP-LSTM, CNN-LSTM, LSTM-MHA。在 75% 以上场景中 Siamese 版本胜出（仅 MLP 例外）。
4. 在 LSTM-MHA + OFI 特征 + 10-tick horizon 上：Siamese 排第 1 (0.615)，baseline 排第 4 (0.198)——巨大差距。

## 实验设置

- **数据**：14 只 A 股军工类股票，2021 年 1-5 月。**对中国 A 股 LOB 直接验证！**
- **窗口**：50 ticks (~150 秒)
- **Horizon**：10、20、50 ticks
- **Backtest**：rolling-window，1 周 valid + 5 周 train + 1 周 test

## 关键架构

```
ask 侧特征 ──→ encoder_ask  ───┐
                              ├─→ concat / sum → classifier head → 3-class softmax
bid 侧特征 ──→ encoder_bid  ───┘
              ↑
              parameter sharing（encoder_bid 和 encoder_ask 是同一个网络）
```

## 对我们比赛的可借鉴点

**这是中国 A 股 LOB 上少见的有效论文，直接 transferable**：

1. **bid/ask 对称性 prior 几乎免费可加**：把网络 forward 一次改两次（共享权重），参数量不增加，训练成本约 +30%（两次 forward 一次 backward）。
2. **可叠加在任意 baseline 之上**：DeepLOB → Siamese-DeepLOB；MLPLOB → Siamese-MLPLOB。**应该作为 ablation 默认开**。
3. **A 股环境验证有效** — 比 FI-2010、NASDAQ 数据上的结论更直接。
4. **OFI 特征 + LSTM-MHA + Siamese** 是一个 evidence-based combo，可以照搬。

## 注意事项 / 局限

- 该研究只用 14 只军工股，行业偏向严重；我们的 5 只匿名 A 股是否同分布未知。
- Siamese 假设 ask/bid encoder 应该用相同 weight；但理论上 ask 侧（卖压）和 bid 侧（买压）信号不对称（牛熊不对称）。可以考虑 partial sharing。

## 我的评分

- 实现成本：low（只是改 forward pass）
- 优先级：**high**（A 股专属验证 + 易加 + 几乎免费的归纳偏置）
