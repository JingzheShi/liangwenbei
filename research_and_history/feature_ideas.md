# 候选特征清单（含公式 + 来源 + 优先级）

> **比赛约束**：每个 t 的预测只能用过去 ≤100 ticks（3s × 100 = 300s）。已有 154 维特征，下面是**额外可加**的候选。
>
> **优先级判定**：以"PnL 提升预期 × 实现成本"打分。high = 必加；medium = 见时间加；low = 学术兴趣。
>
> **符号约定**：
> - `b1, a1` = best bid / ask price；`bs1, as1` = best bid / ask size
> - `bk, ak` = level-k bid/ask price；`bsk, ask` = level-k size
> - `mid = (a1 + b1) / 2`
> - 时间 t 的窗口长度 W ticks (W ∈ {5, 10, 20, 50, 100})

---

## 一、微观结构核心特征（top priority）

### 1. Order Flow Imbalance (OFI) — Cont, Kukanov, Stoikov (2014)
- **公式 (单档 L1)**:
  ```
  e_t = I(b1_t ≥ b1_{t-1}) · bs1_t  - I(b1_t ≤ b1_{t-1}) · bs1_{t-1}
       - I(a1_t ≤ a1_{t-1}) · as1_t + I(a1_t ≥ a1_{t-1}) · as1_{t-1}
  ```
- **W-window 聚合**: `OFI_W = Σ_{s=t-W+1..t} e_s`
- **来源**: arxiv:1011.6402 / JFE 2014
- **适用模型**: 全（CNN/Transformer/GBDT）
- **实现成本**: low
- **优先级**: ⭐⭐⭐ high

### 2. Multi-Level OFI (MLOFI) — Kolm, Turiel, Westray (2023)
- **公式**: 对 level k=1..10 分别算 `e_k(t)`，得到 10 维向量 `MLOFI_t`。再做 W-window sum → 10 维特征。
- **可选 PCA 降维到 3-5 维**避免共线。
- **来源**: Math. Finance 33(4), 2023
- **优先级**: ⭐⭐⭐ high

### 3. Weighted Mid-Price (WMP) — Stoikov 标准
- **公式**:
  ```
  WMP1 = a1·bs1/(bs1+as1) + b1·as1/(bs1+as1)
  ```
  注意：bid 用 ask size 加权，ask 用 bid size 加权（imbalance 推动方向）。
- **来源**: Stoikov 2018; Optiver Vol 比赛官方 starter
- **优先级**: ⭐⭐⭐ high

### 4. Multi-level WMP
- **公式**: 对 k=1..10 算 `WMP_k`，得到 10 维。
- **WMP balance**: `WMP1 - WMP2`（two-level WMP 差，Optiver vol 比赛主特征之一）
- **优先级**: ⭐⭐ high

### 5. Micro-price (Stoikov 2018)
- **简化版（也叫 Stoikov micro-price）**:
  ```
  M = mid + (S/2) · (2I - 1) · ρ   ， S=spread, I=bs/(bs+as), ρ≈相关性参数 0.5
  ```
- **完整版**: 迭代 G_k(I, S) 估计长期 mid-price 期望（需要从历史数据拟合 G）。
- **来源**: SSRN 2970694
- **实现成本**: low（简化版）/ medium（完整）
- **优先级**: ⭐⭐ high

### 6. Order Book Imbalance (OBI) — 标准
- **公式**: `I = bs / (bs + as)`，多档版本 `I_k = bsk / (bsk + ask)`
- **总量版**: `(totalbsize - totalasize) / (totalbsize + totalasize)` （schema 已有 imbalance — 检查是否就是这个）
- **优先级**: ⭐⭐ high（如果 schema 的 imbalance 已经是这个，则跳过；否则补全多档）

---

## 二、Volume / Activity 特征

### 7. Realized volatility (rolling)
- **公式**:
  ```
  r_t = log(WAP_t / WAP_{t-1})
  RV_W(t) = sqrt(Σ_{s=t-W+1..t} r_s²)
  ```
- **多窗**: W ∈ {5, 10, 20, 50, 100}
- **来源**: Optiver Vol 比赛
- **优先级**: ⭐⭐⭐ high（**直接的波动率代理；过去波动是未来波动和方向幅度的强 predictor**）

### 8. Bipower Variation
- **公式**:
  ```
  BV_W = (π/2) · Σ |r_s|·|r_{s-1}|
  ```
  比 RV 对 jump 更鲁棒。
- **来源**: Barndorff-Nielsen & Shephard 2004
- **优先级**: medium

### 9. Volume-Synchronized Probability of Informed Trading (VPIN)
- **简化代理**:
  ```
  VPIN_W = mean(|mb_intst - ma_intst| / (mb_intst + ma_intst)) over past W ticks
  ```
- **完整版**: 用 volume bucket + BVC，见 vpin_easley_lopez_2012.md
- **来源**: Easley-Lopez de Prado-O'Hara 2012
- **优先级**: medium（regime indicator）

### 10. Trade Flow Imbalance (TFI)
- **简化**: 用六类 intensity 直接构造 net buy pressure：
  ```
  TFI = (mb_intst + lb_intst + ca_intst) - (ma_intst + la_intst + cb_intst)
  ```
- **来源**: 通用微观结构
- **优先级**: ⭐⭐ high

---

## 三、Spread / Depth 衍生特征

### 11. Relative spread
- **公式**: `(a1 - b1) / mid` — schema 已有 spread1，但相对 spread 没有
- **优先级**: medium

### 12. Spread momentum
- **公式**: `spread1_t - spread1_{t-W}`
- **W ∈ {5, 10, 50}**
- **来源**: Optiver Trading at Close
- **优先级**: medium

### 13. Effective vs quoted spread proxy
- **公式**: 用 trade-side 估计：`2 · |last_trade_price - mid| / mid` (没有 trade price 时跳过)
- **优先级**: low（缺数据）

### 14. Book slope (price vs cumulative volume)
- **公式（bid 侧）**:
  ```
  slope_bid = mean over k=2..10 of (bk - b1) / (Σ_{j=1..k} bsj)
  ```
- **类似 ask 侧**，再取平均得整体 slope。
- **来源**: Næs & Skjeltorp (2006), frds.io
- **优先级**: ⭐⭐ high（depth quality 的强度量）

### 15. Liquidity-weighted depth at threshold
- **公式**: Σ size 直到 cumulative 等于固定 K（比如 K=avgvol），返回那个 level 的价格 — book "fill" 价。
- **优先级**: medium

### 16. Pairwise price imbalance
- **公式**: 对所有可用价格 pair (p_i, p_j) 算 `(p_i - p_j) / (p_i + p_j)`
  - 比如 `(b1 - b10)/(b1+b10)`, `(a1 - a10)/(a1+a10)`
  - 也可 cross side: `(a1 - b1)/(a1 + b1)`
- **来源**: Optiver Trading at Close 1st
- **优先级**: ⭐⭐ high

### 17. Triplet price imbalance
- **公式**: 对 3 个价格 sort 出 max/mid/min：
  ```
  triplet = (max - mid) / (mid - min)  （加 epsilon 防 0）
  ```
- **常见 triplet**: (b1, mid, a1), (b1, b5, b10), (a1, a5, a10), (avgbid, mid, avgask)
- **来源**: Optiver Trading at Close 1st (HYD)
- **优先级**: ⭐⭐ high（HYD 1st 解决方案的关键 trick）

---

## 四、价格 momentum / 趋势特征

### 18. Multi-scale log returns
- **公式**: `log(mid_t / mid_{t-W})` for W ∈ {1, 5, 10, 20, 50, 100}
- **优先级**: ⭐⭐⭐ high（必加）

### 19. Hull Moving Average (HMA)
- **公式**:
  ```
  WMA_n(x) = Σ_{i=0..n-1} (n-i) · x_{t-i} / Σ(n-i)
  HMA_n = WMA(2·WMA_{n/2}(x) - WMA_n(x), period=√n)
  ```
- **窗口**: n ∈ {10, 20, 50}
- **来源**: G-Research Crypto Forecasting writeup（最重要单一特征）
- **优先级**: ⭐⭐ high

### 20. EWMA / EWMV (exponentially weighted)
- **公式**: 
  ```
  EWMA(x, α) = α·x_t + (1-α)·EWMA(x, α)_{t-1}
  EWMV(x, α) = α·(x_t - EWMA)² + (1-α)·EWMV_{t-1}
  ```
- **多 α**: {0.05, 0.1, 0.3, 0.5, 0.7} 对应 half-life 13/7/2/1/0.7 ticks
- **来源**: 通用 / Hawkes proxy（参见 hawkes_lob_2024.md）
- **优先级**: ⭐⭐ high

### 21. Z-score of recent return
- **公式**: `(r_t - mean_W(r)) / std_W(r)` — relative move strength
- **优先级**: medium

### 22. Fibonacci-window rolling stats
- **窗口**: W ∈ {5, 13, 34, 89}（受 100-tick 限制取小）
- **统计**: mean, std, skew, kurt, max-min range
- **来源**: G-Research Crypto Forecasting
- **优先级**: medium

### 23. Tick rule / Lee-Ready proxy
- **公式**: `tick_sign_t = sign(price_t - price_{t-1})`，三态：+1, 0, -1
- **聚合**: `tick_rule_imbalance_W = Σ tick_sign / W`
- **来源**: Lee & Ready 1991
- **优先级**: medium

### 24. Volume-weighted price change
- **公式**: `(close_t - open_t) · log(1 + volume_delta)` — schema 已有 volume_delta
- **优先级**: medium

---

## 五、Hawkes / 订单到达特征

### 25. Multi-scale arrival intensity
- **公式**: 对 schema 已有 6 类 *_intst 做 EWMA：
  ```
  λ_i^EW(α) = α · 1_{event_i happens at t} + (1-α) · λ_i^EW(α)_{t-1}
  ```
- **本来 schema 已有 *_intst 是单一尺度；加 multi-α → 6 × |α| 维**
- **来源**: Hawkes literature
- **优先级**: ⭐⭐ high

### 26. Cross-excitation ratios
- **公式**:
  - `mb_intst / (mb_intst + ma_intst + 1e-9)` — market buy 占比
  - `(lb_intst + ca_intst) / (la_intst + cb_intst + 1e-9)` — net 看涨 vs 看跌（含 cancel 信号）
  - `cb_intst / lb_intst` — cancel-to-limit ratio (looking-aggressive on bid)
- **来源**: Hawkes / market microstructure
- **优先级**: ⭐⭐ high

### 27. Branching ratio proxy
- **公式**: `Σ recent intst / mean intst` — 度量市场内生性
- **优先级**: low

---

## 六、Order book 形状特征

### 28. Quote intensity / book convexity
- **公式 (bid 侧)**: `(b1 - b5) / (b1 - b10)` 比例 — 越接近 1，前 5 档集中；越小，深度均匀。
- **类似 ask 侧**
- **优先级**: medium

### 29. Liquidity ratio at level k
- **公式**: `bsk / Σ bsj` — 每档占比
- **stack 10 维 each side → 20 维**
- **优先级**: low（共线性高）

### 30. Spread crossing rate
- **公式**: `count_W(spread_t < spread_{t-1}) / W` — spread 缩窄频率
- **优先级**: medium

---

## 七、跨 sym 截面特征

### 31. Cross-sectional rank of feature
- **公式**: 在每个 t，把 5 只 sym 在某 feature 上 rank → 1..5（normalized to 0-1）
- **可 rank 的 feature**: log_return_W, OFI_W, RV_W, imbalance
- **来源**: Two Sigma / 通用 quant
- **优先级**: ⭐⭐ high（5 只 sym 可能有共动 / 反向）

### 32. Cross-sectional z-score
- **公式**: `(x_sym - mean over syms at time t) / std`
- **优先级**: medium

### 33. Sym-pair price ratio
- **公式**: `mid_sym0 / mid_sym1` 之类 — 长期共动则 ratio 是 mean-reverting
- **变量**: 5×5 - 5 = 20 个 ratio（不算自己 vs 自己）
- **优先级**: low（5 sym 都匿名，pair 关系可能不稳）

### 34. Sym-id embedding
- **方法**: 训练时给每个 sym 一个 learnable embedding (16 维)，concat 到 feature。
- **来源**: 普遍做法，Optiver Trading at Close 也用 stock weights 做类似事
- **优先级**: ⭐⭐ high（只要建模允许，必做）

---

## 八、时间 / session 特征

### 35. Time-in-session features
- **公式**:
  - `tick_idx / 2001` — session 内进度（0-1）
  - `is_AM` — AM/PM session 区分
  - `time_to_close` — 离 session 结束多远
- **来源**: Optiver Trading at Close
- **优先级**: ⭐⭐ high（开盘和尾盘 dynamics 差很大）

### 36. Sin/cos cyclical encoding
- **公式**: `sin(2π · tick_idx / 2001)`, `cos(2π · tick_idx / 2001)`
- **优先级**: medium

### 37. Time2Vec (Kazemi 2019)
- **公式**: `t2v(t) = [ω_0·t + φ_0, sin(ω_1·t + φ_1), ..., sin(ω_K·t + φ_K)]`，ω 学习
- **来源**: Spacetimeformer-LOB, Kazemi et al. 2019
- **优先级**: low（先用简单 sin/cos）

---

## 九、Cross-stat / interaction 特征

### 38. Spread × imbalance ("market urgency")
- **公式**: `spread1 · ((bs1 - as1) / (bs1 + as1))`
- **来源**: Optiver Trading at Close 1st
- **优先级**: ⭐⭐ high

### 39. Imbalance momentum
- **公式**: `imbalance_t - imbalance_{t-W}` for W ∈ {1, 5, 20}
- **来源**: Trading at Close 1st
- **优先级**: ⭐⭐ high

### 40. Price pressure
- **公式**: `(bs1 - as1) · spread1`
- **来源**: Trading at Close 1st
- **优先级**: medium

### 41. Depth pressure
- **公式**: `(as1 - bs1) · (a10 - b10)` — best vs deep 的力差
- **来源**: Trading at Close 1st
- **优先级**: medium

### 42. Spread-depth ratio
- **公式**: `spread1 / (bs1 + as1)`
- **来源**: Trading at Close 1st
- **优先级**: medium

---

## 十、频域 / 信号处理特征

### 43. FFT magnitudes of recent prices
- **方法**: 对过去 100 ticks 的 mid-price 做 FFT，取前 K 个幅值（去除 DC + 最低频）作为特征。
- **K = 5-10**
- **来源**: 时序 ML 通用，最近也在 LOB 上有人试
- **优先级**: low（解释性差，CV 表现不一定稳）

### 44. Wavelet decomposition coefficients
- **方法**: Daubechies-4 or Haar wavelet 分解 mid-price，取每个 scale 的 energy。
- **优先级**: low

---

## 十一、Realized 微结构 statistic

### 45. Roll's effective spread estimate
- **公式**: `2 · sqrt(-cov(Δp_t, Δp_{t+1}))` — 没有 trade direction 也能估 spread
- **来源**: Roll 1984
- **优先级**: low（schema 已有 spread）

### 46. Kyle's lambda (price impact coefficient)
- **公式**: 回归 `Δmid = λ · signed_volume` 在过去 W ticks，取 λ 作为 feature
- **优先级**: medium（度量市场冲击力）

### 47. Amihud illiquidity proxy
- **公式**: `|r_t| / volume_delta_t` — 单位成交量造成的价格变化
- **来源**: Amihud 2002
- **优先级**: medium

---

## 十二、Label-aware（小心 leakage）

### 48. Past-label majority
- **方法**: 用过去 W ticks 同 sym 的 **historical** label_h 的众数 / 比例作为 feature。**严格用 t-W 之前**。
- **优先级**: low（leakage 风险高）

### 49. Past-label transition probability
- **方法**: 用历史数据估计 P(label_h_t | label_{h-1}_t, label_5_t)，作为先验。
- **优先级**: low

---

## 🎯 「如果只能加 5 个特征，我会加哪 5 个」

排序基于：(a) 学术证据强度，(b) Optiver/Jane Street 实战验证，(c) 我们 schema 缺口，(d) 实现成本。

| Rank | 特征 | 一句话理由 |
|---|---|---|
| **1** | **Multi-level OFI (10 维 + W-rolling sum)** | Kolm 2023 直接证明 OFI > raw LOB；schema 没显式有；low cost |
| **2** | **Weighted Mid-Price (WMP) + WMP balance (WMP1-WMP2)** | Stoikov 标准；Optiver Vol 比赛核心；schema 完全没有；low cost |
| **3** | **Realized Volatility @ multi-window (W∈{5,10,20,50})** | 过去波动是未来 |Δmid| 最强 predictor；直接预测 PnL 上限 |
| **4** | **Multi-scale EWMA of intensities (6 × 4 windows = 24 维)** | Hawkes proxy；schema 只有单尺度 *_intst；low cost；预测短期到达爆发 |
| **5** | **Time-in-session + sym embedding** | AM/PM 与开盘尾盘 dynamics 差异大；sym 异质要让模型知道 |

如果 5 之后还有第 6 名候选：
- 6. **Triplet imbalance over (b1, mid, a1) 等组合**（Trading at Close 1st 杀手锏）
- 7. **Cross-sectional rank of OFI/RV**（5 sym 共动信息）
- 8. **Hull Moving Average (HMA)**（G-Research 单一最强）
- 9. **Book slope (per side)**
- 10. **Spread × imbalance interaction**

---

## 实现注意事项

1. **向量化**：所有 W-rolling 用 `pd.Series.rolling(W)` 或 `numpy stride_tricks`；**禁止 Python for 循环**。
2. **NaN 处理**：开头 W ticks 内 rolling stats 会 NaN，要么 forward-fill，要么 mask 掉这些样本不参与训练。
3. **泄漏防护**：所有 feature 必须是 `t` 时刻**已知**的；任何用到 t+1 的都是 leakage。
4. **Normalization**：上面 raw 特征要做 5-day rolling z-score（per-sym）；ratio 类不用归一化或简单 min-max。
5. **Cross-sym features**: 对于跨 sym rank/zscore，要确保所有 5 个 sym 在该 t 都有有效数据。
6. **Cost-vs-gain**: 先全部生成→训练 LightGBM 看 feature importance → 砍掉低重要性 → 再上深度模型。

---

## 来源汇总

主要来源：
- **OFI / MLOFI**: arxiv:1011.6402, mafi.12413 (Kolm 2023)
- **Micro-price / WMP**: SSRN 2970694 (Stoikov 2018)
- **DeepLOB schema**: arxiv:1808.03668
- **Kercheval-Zhang 144 schema**: Quantitative Finance 15(8), 2015
- **VPIN**: SSRN 1695041 (Easley-Lopez de Prado-O'Hara 2012)
- **Optiver Vol writeup**: kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970
- **Optiver Trading Close 1st**: kaggle.com/c/optiver-trading-at-the-close/writeups/hyd-1st-place-solution + fan2goa1 blog
- **Jane Street 1st (Yirun)**: kaggle.com/c/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s
- **G-Research Crypto**: kaggle.com/c/g-research-crypto-forecasting
- **Microstructural guide**: arxiv:2403.09267
- **TLOB / MLPLOB**: arxiv:2502.15757
- **LiT**: Frontiers AI 2025
- **Hawkes**: arxiv:2503.14814 (综述)
- **HMA**: G-Research community writeups
- **Lee-Ready**: Journal of Finance 1991
- **Time2Vec**: Kazemi et al. 2019 NeurIPS
