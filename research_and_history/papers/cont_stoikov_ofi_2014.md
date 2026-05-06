# The Price Impact of Order Book Events (OFI)

- **Authors**: Rama Cont, Arseniy Kukanov, Sasha Stoikov
- **Venue**: Journal of Financial Econometrics 12(1), 2014, pp. 47–88
- **arXiv**: 1011.6402 (2010 preprint)
- **Links**: https://arxiv.org/abs/1011.6402 ; SSRN 1712822

## 核心贡献

1. 在 NYSE TAQ 50 只股票的 trade+quote 数据上，发现 **mid-price 变化主要由 Order Flow Imbalance (OFI) 线性驱动**，斜率 ~ 1/depth。
2. 把 LOB 事件（limit order, market order, cancel）压缩成单一的 OFI 标量后，price impact 模型变得简洁、可迁移、跨 stock 跨日稳定。
3. 这个结果对 intraday seasonality 鲁棒，跨时间尺度（10s 到 10min）都成立。

## 关键公式

最佳档（L1）OFI：
```
e_n = I(P_b^n ≥ P_b^{n-1}) · q_b^n  -  I(P_b^n ≤ P_b^{n-1}) · q_b^{n-1}
    - I(P_a^n ≤ P_a^{n-1}) · q_a^n  +  I(P_a^n ≥ P_a^{n-1}) · q_a^{n-1}
```

聚合到时间区间 [t_0, t_N]：OFI = Σ e_n。

线性 price impact 模型：
```
ΔP = β · OFI + ε,   β ≈ c / depth
```

## 后续扩展（Multi-level OFI / MLOFI, Xu Cont 2018; Kolm Turiel Westray 2023）

把 OFI 推广到 LOB 第 1 到 第 M 档每档单独算，得到 M 维向量。MLOFI 对 mid-price 的预测能力**显著优于** L1 OFI。

## 对我们比赛的可借鉴点

- **核心特征 #1**：L1 OFI + 多档 OFI 聚合。我们的 schema 已经接近——`bid_diff*`, `ask_diff*`, `bsize_rate*`, `asize_rate*` 拼起来就是。
- **跨 horizon 聚合**：OFI 在不同时间窗（5/10/30/60 ticks）的滚窗 sum，是预测对应 horizon 价格变化的强 signal。
- 比赛只能用过去 ≤100 ticks，所以滚窗最长到 100。
- 虽然 schema 里有六类订单（lb/la/mb/ma/cb/ca）的 intst，但 **OFI 是更 net 的"压差"信号**，不要漏。

## 我的评分

- 实现成本：low
- 优先级：**top 1 必加**
