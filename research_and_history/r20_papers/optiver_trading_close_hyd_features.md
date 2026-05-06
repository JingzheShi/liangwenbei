# Optiver Trading at Close 1st (HYD) — Complete Feature Engineering

**Source**: kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution + fan2goa1's blog分析

## Context

2023-12 Optiver Kaggle 比赛，500 stocks，预测 closing auction last 10 min wap movement。**1st place HYD** 用了 LightGBM + 极致特征工程。即使我们 setup 是 LOB tick（不是 auction），其 **pairwise/triplet imbalance + 衍生 ratio 系列**完全可移植。

## 完整 feature 列表

### V1: Baseline (~25 features)
```
volume          = ask_size + bid_size
mid_price       = (ask_price + bid_price) / 2
liquidity_imb   = (bid_size - ask_size) / (bid_size + ask_size)
matched_imb     = (imbalance_size - matched_size) / (matched_size + imbalance_size)
size_imbalance  = bid_size / ask_size
```

### Pairwise imb（关键 trick #1）

对 prices = [ref_price, far_price, near_price, ask, bid, wap] 的所有 C(6,2)=15 pair：
```
{p1}_{p2}_imb = (p1 - p2) / (p1 + p2)
```

→ 15 个 features (HYD 报告 importance 排前 30%)

### Triplet imb（关键 trick #2）

对 4-price 集合 [ask, bid, wap, ref_price] 取所有 3-tuple，再对每个 tuple 排序后算：

```
triplet = (max_val - mid_val) / (mid_val - min_val)
```

→ C(4,3) = 4 个 triplet features。同样的对 [matched_size, bid_size, ask_size, imb_size] 算 4 个 → 共 8 triplet features.

**为什么 work**：triplet 编码 3 个数的"几何配比"——max-mid 拉开 vs mid-min 拉开；> 1 = 上偏，< 1 = 下偏，抓住 book 形状几何。

### V2: Imbalance + momentum 衍生

```
weighted_wap        = stock_weights × wap
wap_momentum        = weighted_wap.pct_change(periods=6)        # ⚠️ 跨 batch state，我们改成 within-window
imbalance_momentum  = imbalance_size.diff(1) / matched_size
price_spread        = ask - bid
spread_intensity    = price_spread.diff()
price_pressure      = imbalance_size × (ask - bid)
market_urgency      = price_spread × liquidity_imb
depth_pressure      = (ask_size - bid_size) × (far_price - near_price)
spread_depth_ratio  = (ask - bid) / (bid_size + ask_size)
mid_price_movement  = sign(mid_price.diff(periods=5))
micro_price         = (bid × ask_size + ask × bid_size) / (bid_size + ask_size)
relative_spread     = (ask - bid) / wap
```

### Statistical Aggregations（descriptive）

对所有 prices 和 sizes 算 mean, std, skew, kurt → 增广 30+ features。

### V3: Lag & Rolling features

对 [matched_size, imbalance_size, ref_price, imb_buy_sell_flag] × windows {1, 2, 3, 5, 10}：
- shift(W)
- ret(W) = pct_change

对 [ask, bid, ask_size, bid_size, wap, near_price, far_price] × windows {1, 2, 3, 5, 10}:
- diff(W)

→ ~70 个 lag/rolling features

### Global stock-level features
- global_median_size, global_std_size, global_ptp_size
- global_median_price, global_std_price, global_ptp_price

→ 6 个（在我们 setup 是 within-window stats）

### Time features
```
dow      = date_id % 5      # ⛔ 我们没 date
dom      = date_id % 20     # ⛔
seconds  = sec_in_bucket % 60
minute   = sec_in_bucket // 60
```

## 关键 takeaway 对我们

### ✅ 完全可移植（已 mark 在 r20_factor_library §H）

- **15 pairwise imb** (H1) — trivial 实现，15 features 一次到位
- **8 triplet imb** (H2) — trivial
- **micro_price, market_urgency, price_pressure, depth_pressure** (E12, H3-H5) — 每个 1 line code
- **wap_momentum, imbalance_momentum** (H4, H6) — pct_change 改成 within-window pct
- **mid_price_movement (sign-only)** (H8)
- **spread_intensity (diff)** (H7)
- **stat aggregates** mean/std/skew/kurt of price/size series — 30+ features (H9)

### ⚠️ 需要改造

- **stock_weights** — Optiver 给了 weights map，**我们没有**；改成 within-window mean(volume)
- **dow/dom** — date 不可用
- **groupby stock pct_change** — 不能跨 batch；改成 within-window difference

### ⛔ 不可用

- 跨 stock features (global stats are over a 5-stock universe in their setup, not us)
- closing auction-specific features (matched_size, imbalance_size — 我们 LOB 没这些)

## 模型部分（HYD）

- **LightGBM 主模型** + 极少 NN
- Online learning：用 every-day re-training（**评测限制：我们不能 online retrain**）
- post-processing minimal — 主要靠 feature engineering

## 启发

1. **Pairwise + Triplet 是 cheap diversity**：HYD 加了 ~25 个这种 features 后 LB 跳 1%（在 Optiver leaderboard 是 huge gain）
2. **Sign + magnitude 分开 encode**：`sign(diff)` 和 `abs(diff)` / `diff_squared` 都加 → tree-based 模型更易切
3. **WMP + spread + size 的 cross product** 已 cover OFI 大部分信号；我们的 multi-level OFI 是更精细版本

## 我们的实施

按 §10 short list rank #5-#6（pairwise/triplet）和 #30（H4 H6）加进去。**预期 trivial 实现成本 + ~+2 ~ +5 ROI**（HYD 在 Optiver 上的相对增益）。
