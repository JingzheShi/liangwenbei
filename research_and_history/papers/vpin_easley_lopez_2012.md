# Flow Toxicity and Liquidity in a High-frequency World (VPIN)

- **Authors**: David Easley, Marcos López de Prado, Maureen O'Hara
- **Year**: 2011 (working paper) / 2012 (Review of Financial Studies)
- **Links**: https://www.quantresearch.org/VPIN.pdf ; SSRN 1695041

## 核心贡献

1. 提出 **VPIN (Volume-Synchronized Probability of Informed Trading)**——PIN 模型在 volume clock 上的高频化。
2. 用 volume bucket（每个 bucket 固定成交量 V）替代 calendar time，这样 sampling 频率随市场活跃度自适应。
3. 在每个 volume bucket 上用 bulk-volume classification 估算 buy 量 V_B 和 sell 量 V_S，VPIN = E[|V_B - V_S|] / V。
4. 实证：在 2010 flash crash 前数小时和数天，VPIN 显著上升——VPIN 是 **toxicity / informed trading** 的实时代理。
5. VPIN 高 → market maker 损失风险大，预示 illiquidity / 反向风险。

## 关键公式（简化）

对每个固定体积 V 的 bucket：
```
V_B = Σ trades 中归为买方主导的体积
V_S = V - V_B
order_imbalance = |V_B - V_S| / V

VPIN = mean over last n buckets of order_imbalance
```

Buy/Sell 分类常用 **bulk volume classification** (BVC)：
```
%Buy = Z((P_t - P_{t-1}) / σ_ΔP)   (Z = standard normal CDF)
V_B = V · %Buy
```

## 对我们比赛的可借鉴点

- 用过去 100 ticks 的总成交量构建 **volume bucket**，对每个 tick 算"当前所在 bucket 的 VPIN-like 信号"。
- 我们没有 trade direction，但可以用 BVC：用 mid-price 变化方向 / 标准化变化 来分配 buy% 和 sell%。
- 也可以用更简单的代理：`(mb_intst - ma_intst) / (mb_intst + ma_intst)`——用六类订单的 market buy / market ask 直接。
- VPIN 是**"市场不健康"的代理**，在 PnL 优化里可以作为 gating feature：VPIN 高时降低 confidence。

## 我的评分

- 实现成本：medium（需要 volume clock 和 BVC 实现）
- 优先级：**medium**（不是 top 5，但作为 secondary 特征 / regime detector 有价值）
