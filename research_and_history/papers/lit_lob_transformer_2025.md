# LiT: Limit Order Book Transformer

- **Year**: 2025 (Frontiers in AI)
- **Links**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full

## 核心贡献

1. 用 **structured patching** 把 LOB 输入切成 (Ph, Pw) 的 patch，避免 conv layer，直接 Transformer + LSTM。
2. 输入 Level-2 LOB 数据：20 个 levels × (price + volume) × ask/bid = 80 features per tick。
3. **Z-score normalize price 和 volume 分别做**，避免量级差。
4. 在 Binance crypto 数据上比 DeepLOB / TransLOB 略好（66.4% F1 vs 66.2% on 300-1000ms）。

## 关键架构

```
LOB tensor (T × F)
  → 2-channel (price, volume)
  → patch projection (Ph × Pw)
  → +positional embedding
  → Transformer blocks (multi-head self-attention)
  → LSTM
  → softmax (3-class)
```

## 对我们比赛的可借鉴点

- **structured patching** 在我们这个 100 tick × 154 feat 的 setting 下值得试：把 (10 ticks × 14 feat) 当一个 patch，T=100 → 10 个 time patch，跨 patch 用 attention。这在 Vision Transformer 思路上很自然。
- **price 和 volume 分别 normalize**——量级真的差很多，分开做更稳。
- 对 crypto 微秒级 horizon 有效，对我们 3s/tick 的中频也应该 transferable。

## 我的评分

- 实现成本：medium
- 优先级：**medium**（如果 MLPLOB / DeepLOB baseline 卡瓶颈，可以试 patching transformer）
