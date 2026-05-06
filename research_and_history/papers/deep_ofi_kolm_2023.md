# Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the LOB

- **Authors**: Petter N. Kolm, Jeremy Turiel, Nicholas Westray
- **Venue**: Mathematical Finance 33(4), 2023
- **Links**: https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413 ; SSRN 3900141

## 核心贡献

1. 在 115 只 NASDAQ 股票上系统比较"训练于 raw LOB" vs "训练于 OFI" 的深度模型。
2. **关键结论**：用 multi-level OFI（MLOFI）作为输入，模型在所有 horizon 上都显著超过用 raw LOB 的同样模型。
3. OFI 比 raw LOB 信息更"干净"——价格变化主要由 OFI 驱动，所以让模型学一个"压缩好的"输入有显著优势。
4. 多层 OFI 之间高度相关，建议用 **PCA** 或 stack 后让网络自己学线性组合。
5. 跨 horizon 一致性：OFI 在 1-tick 到 100-tick 上都比 LOB 强。

## OFI 公式（Cont, Kukanov, Stoikov 2014）

每个 LOB level n 的 OFI：

```
e_n^t = I(p_b^n,t ≥ p_b^n,t-1) · q_b^n,t  -  I(p_b^n,t ≤ p_b^n,t-1) · q_b^n,t-1
      - I(p_a^n,t ≤ p_a^n,t-1) · q_a^n,t  +  I(p_a^n,t ≥ p_a^n,t-1) · q_a^n,t-1
```

直观解释：
- bid 价上涨 / ask 价下跌 → +size（买压）
- bid 价下跌 / ask 价上涨 → -size（卖压）
- 价不变 → ±size 差

L1 OFI 是 best level 的 OFI，**MLOFI** 是把每个 level 的 OFI stack 成向量。

## 对我们比赛的可借鉴点

- **必须把 OFI 作为特征**：我们已有 `bid_diff*`, `ask_diff*`, `bsize_rate*`, `asize_rate*` 等，可以直接拼出 multi-level OFI。
- **多 horizon 训练**：DeepOFI 是多 horizon 共训，我们的赛题正好要预测 5/10/20/40/60 ticks，这是天然的 multi-task 设置。
- **PCA 降维**：10 个 level 的 OFI 高度相关，可以先 PCA 到 3-5 个主成分作为高维特征。
- 把 LOB 原始量价 + OFI **同时**喂给模型（或 stack 在 channel 维），让模型选择。

## 我的评分

- 实现成本：low（向量化计算）
- 优先级：**high（特征 top 1）**
