# R20 — 因子库深挖：100+ 个可落地因子（Alpha101 + Qlib Alpha158 + AFML + 微观结构 + Kaggle top）

> **核心交付物**。把 4 大公开因子库 + 8 篇微观结构 paper 中**所有可落地到我们 100-tick LOB setup** 的因子枚举出来，每条带公式、来源、适配度评估、复杂度、优先级。
>
> **任务边界**：本文档只做研究，不写训练代码。每条因子的公式都已提炼到"能直接翻译成 numpy 向量化"的层级。
>
> **范围**：除 T3 已实施的 72 个因子外，新增 **130+ 候选因子**（去重后实际可加 ~80）。

---

## ⚠️ 三条评测硬约束（每条因子都要过这个 filter）

1. **`date` 评测置 0** → 不能依赖日期
2. **测试点顺序被打乱** → Predictor stateless；每次 100-tick window 独立计算
3. **sym 0-4 但可能含训练外股票** → **sym-agnostic**，**不能用 cross-sectional 截面**（评测 batch shuffle，多 sym 不保证同时刻共存）

→ **Alpha101 中所有 `rank(x)`（cross-sectional rank）和 `IndClass.X`（行业中性化）的版本都不能直接用**。但很多可以**改造为单股票 ts_rank** 版本（替换 `rank` → `ts_rank` over W=50-100 ticks）。下文对每条 Alpha 单独评估。

---

## 🎯 总览（按类别）

| 类别 | 数量 | 高优先级 |
|---|---|---|
| A. 价量动量 / 反转 (Alpha101 + Qlib + 经典) | 32 | 12 |
| B. 流动性 / 微观结构 (Roll, CS, OFI variants, Kyle, Amihud, Hasbrouck) | 18 | 9 |
| C. 波动率 / 跳跃 (RV variants, BV, Garman-Klass, Parkinson, jump test) | 15 | 6 |
| D. 订单流 / Toxicity (VPIN, BVC, Hawkes, cancel rate) | 12 | 6 |
| E. 跨档 / 形状 (book slope, depth, level skew, convexity) | 14 | 7 |
| F. 衍生 / 比例 / Kbar (Qlib KMID 家族 + RSI 家族) | 18 | 6 |
| G. 高频特殊 (tick rule, microstructural noise, signature plot) | 11 | 4 |
| H. 价位差 / Pairwise / Triplet (Optiver Trading at Close 套路) | 14 | 8 |
| **合计** | **134** | **58** |

**Top-30 short list 在末尾 §10**。

---

## A. 价量动量 / 反转

### A1. Multi-scale log return（已 T3 部分实现，扩展窗口）
- 公式：`r_W(t) = log(mid_t / mid_{t-W})`，W ∈ {1, 3, 5, 10, 20, 50, 100}
- **新增价位**：用 WMP 替代 mid → `r_wmp_W`；用 mid_geom = sqrt(b1*a1) → `r_geom_W`
- 来源：Alpha101 helper `returns`；Qlib `ROC`
- 适配：✅ 完美 sym-agnostic + stateless
- 复杂度：low
- 优先级：⭐⭐⭐ high

### A2. Z-score of price relative to recent mean
- 公式：`(mid_t - sma_W) / std_W`，W ∈ {10, 20, 50}；可对 mid / WMP / vwap 各做一份
- 来源：通用，Qlib RANK/QTL 类似
- 复杂度：low
- 优先级：⭐⭐ high

### A3. Hull MA & WMA & EMA & DEMA & TEMA（多窗口三件套）
- 公式：
  ```
  WMA_n(x) = Σ(n-i)·x_{t-i} / Σ(n-i)，i=0..n-1
  HMA_n = WMA( 2·WMA_{n/2}(x) - WMA_n(x), period=√n )
  EMA_α(x) = α·x + (1-α)·EMA_α_{t-1}（多 α）
  DEMA = 2·EMA - EMA(EMA)
  TEMA = 3·EMA - 3·EMA(EMA) + EMA(EMA(EMA))
  ```
  W ∈ {5, 10, 20, 50}，α 取 {0.05, 0.1, 0.3, 0.5}
- 来源：G-Research Crypto winners; technical analysis classics
- 复杂度：low
- 优先级：⭐⭐ high (HMA 单一最强)

### A4. RSI / SMI （Wilder's RSI 变种）
- 公式：
  ```
  gain_W = sum(max(Δp, 0) over past W ticks)
  loss_W = sum(max(-Δp, 0) over past W ticks)
  RSI_W = 100 · gain_W / (gain_W + loss_W + ε)
  ```
- W ∈ {7, 14, 21, 50}
- 来源：J. Welles Wilder 1978; Qlib SUMP/SUMN/SUMD（同公式）
- 适配：✅
- 复杂度：low
- 优先级：⭐⭐ high

### A5. Stochastic K%D% (Williams %R)
- 公式：
  ```
  K = (close - min_W(low)) / (max_W(high) - min_W(low) + ε) · 100
  D = sma(K, 3)
  Williams_R = -100 · (max_W(high) - close) / (max_W(high) - min_W(low) + ε)
  ```
- 等价于 Qlib `RSV`
- 复杂度：low
- 优先级：⭐ medium

### A6. Aroon Oscillator
- 公式：
  ```
  AroonUp = (W - argmax_idx_in_window(high)) / W · 100
  AroonDn = (W - argmin_idx_in_window(low))  / W · 100
  AroonOsc = AroonUp - AroonDn
  ```
- 等价于 Qlib `IMAX/IMIN/IMXD`
- W ∈ {14, 25, 50}
- 复杂度：low
- 优先级：⭐ medium

### A7. CCI (Commodity Channel Index)
- 公式：
  ```
  TP = (high + low + close) / 3
  CCI = (TP - sma_W(TP)) / (0.015 · mean_dev_W(TP))
  其中 mean_dev_W(TP) = mean(|TP - sma_W(TP)|, W)
  ```
- W ∈ {10, 20, 30}
- 来源：Lambert 1980
- 复杂度：low
- 优先级：⭐ medium

### A8. ADX / DMI（趋势强度）
- 公式：
  ```
  +DM = max(high_t - high_{t-1}, 0) if (high_t - high_{t-1}) > (low_{t-1} - low_t) else 0
  -DM = max(low_{t-1} - low_t, 0)   if (low_{t-1} - low_t) > (high_t - high_{t-1}) else 0
  TR = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
  +DI_W = 100 · ema(+DM, W) / ema(TR, W)
  -DI_W = 100 · ema(-DM, W) / ema(TR, W)
  DX = 100 · |+DI - -DI| / (+DI + -DI)
  ADX_W = ema(DX, W)
  ```
- W ∈ {14, 28}
- 复杂度：medium
- 优先级：⭐ medium

### A9. MACD (multi-α)
- 公式：
  ```
  MACD = EMA_12 - EMA_26
  signal = EMA_9(MACD)
  hist = MACD - signal
  ```
- 多组 (fast, slow, sig)：(12,26,9), (5,20,9), (3,10,3)
- 复杂度：low
- 优先级：⭐ medium

### A10. ROC (Rate of Change)
- 公式：`ROC_W = (close_t / close_{t-W} - 1) · 100`
- 等价于 Qlib `ROC`
- W ∈ {5, 10, 20, 50}
- 复杂度：low
- 优先级：⭐⭐ high (与 A1 等价但保留作为 Qlib baseline 名)

### A11. Alpha101 reversal/momentum 单股票可改造子集

> 把 Alpha101 中**仅依赖单股票 OHLCV** 的 alpha 提取出来，用 ts_rank 替代 rank（when applicable）。

| 编号 | 改造后公式 | 复杂度 | 价值 |
|---|---|---|---|
| **Alpha#1** | `ts_rank( argmax( signedpower(((ret<0)?std(ret,20):close), 2), 5 ), W=50)` — 0.5 | medium | low（cross-section 强相关） |
| **Alpha#5** | `ts_rank(open - sma(vwap,10), W) * (-abs(ts_rank(close-vwap, W)))` | low | medium |
| **Alpha#6** | `-corr(open, volume, 10)` — **完全单股票** | low | high |
| **Alpha#7** | `(adv20 < volume) ? (-ts_rank(|delta(close,7)|, 60) * sign(delta(close,7))) : -1` | medium | medium |
| **Alpha#9** | conditional reversal: `if(min(Δclose_5)>0) ? Δclose : (max(Δclose_5)<0 ? Δclose : -Δclose)` | low | high |
| **Alpha#11** | `(ts_rank(max(vwap-close,3), W) + ts_rank(min(vwap-close,3), W)) * ts_rank(Δvolume, W)` | low | medium |
| **Alpha#12** | `sign(Δvolume) · (-Δclose)` — **完全单股票，超 trivial** | low | high |
| **Alpha#23** | `(sma(high,20) < high) ? -Δhigh_2 : 0` | low | medium |
| **Alpha#24** | 长期均价偏离触发反转：`if Δsma100/delay(close,100) <= -0.05 : (-(close - min(close,100))) else (-Δclose_3)` （**W=100 刚好我们窗口上限**） | medium | medium |
| **Alpha#28** | `scale(corr(adv20, low, 5) + (high+low)/2 - close)` — 用 sma(volume,20) 替 adv20 | low | medium |
| **Alpha#32** | `scale((sma(close,7) - close)) + 20·scale(corr(vwap, delay(close,5), 230))` — **230 太长**，改 W=50 | low | low |
| **Alpha#41** | `sqrt(high·low) - vwap` — **完全单股票，超 trivial** | low | ⭐ high |
| **Alpha#42** | `ts_rank(vwap-close, W) / ts_rank(vwap+close, W)` | low | medium |
| **Alpha#46** | 三重区间反转：参见原文，纯单股票 if/else，价格 5/10/20 day delta 比较 | low | medium |
| **Alpha#49** | 同 Alpha#46 但阈值 -0.1 | low | low |
| **Alpha#51** | 同 Alpha#46 但阈值 -0.05 | low | low |
| **Alpha#53** | `-Δ((close-low - high+close)/(close-low), 9)` | low | medium |
| **Alpha#54** | `-(low-close) · open^5 / ((low-high) · close^5)` | low | medium |
| **Alpha#101** | `(close - open) / (high - low + 0.001)` — **超 trivial 单股票** | low | ⭐ high |

**汇总建议**：纯单股票可直接落地的 alpha 共 ~12 条；其余需要 rank → ts_rank 改造（约 25 条）。**优先 #6, #9, #12, #41, #46, #54, #101**（trivial 实现，已被验证）。

### A12. ZigZag-like break detector（单股票）
- 公式：`break_signal = 1 if close_t > max(high_{t-W..t-1}) else (-1 if close_t < min(low_{t-W..t-1}) else 0)`
- W ∈ {10, 20, 50}
- 复杂度：low
- 优先级：⭐ medium

### A13. Bollinger Band position
- 公式：
  ```
  BB_mid = sma_W(close)
  BB_up  = BB_mid + 2·std_W(close)
  BB_dn  = BB_mid - 2·std_W(close)
  BB_pos = (close - BB_dn) / (BB_up - BB_dn + ε)
  BB_width = (BB_up - BB_dn) / BB_mid
  ```
- W ∈ {10, 20, 50}
- 复杂度：low
- 优先级：⭐⭐ high (BB_width 是 vol regime indicator)

### A14. KAMA (Kaufman Adaptive MA)
- 公式：
  ```
  ER = |close_t - close_{t-W}| / sum(|Δclose|, W)  (efficiency ratio)
  SC = (ER · (2/(2+1) - 2/(30+1)) + 2/(30+1))²
  KAMA_t = KAMA_{t-1} + SC · (close_t - KAMA_{t-1})
  ```
- W ∈ {10, 20}
- 复杂度：low
- 优先级：⭐ medium

### A15. Vortex Indicator
- 公式：
  ```
  +VM = |high_t - low_{t-1}|
  -VM = |low_t - high_{t-1}|
  VI+_W = sum(+VM, W) / sum(TR, W)
  VI-_W = sum(-VM, W) / sum(TR, W)
  ```
- W ∈ {14, 21}
- 复杂度：low
- 优先级：⭐ low

### A16. PPO (Percentage Price Oscillator)
- 公式：`PPO = (EMA_12 - EMA_26) / EMA_26 · 100`
- 复杂度：low
- 优先级：⭐ medium

### A17. TRIX
- 公式：`TRIX_W = pct_change(EMA_W(EMA_W(EMA_W(close))))`
- W ∈ {15, 30}
- 复杂度：low
- 优先级：⭐ low

### A18. Ulcer Index
- 公式：
  ```
  drawdown_t = (close_t - max(close, W)) / max(close, W)
  UI_W = sqrt(mean(drawdown², W))
  ```
- W ∈ {14, 50}
- 复杂度：low
- 优先级：⭐ low

### A19. Coppock Curve
- 公式：`CC = WMA_{10}(ROC_{14}(close) + ROC_{11}(close))`
- 复杂度：low
- 优先级：⭐ low

### A20. CMO (Chande Momentum Oscillator)
- 公式：`CMO_W = 100 · (sum_up - sum_dn) / (sum_up + sum_dn)`
- 等价于 Qlib `SUMD`
- W ∈ {9, 14, 28}
- 复杂度：low
- 优先级：⭐ low

### A21. Linear regression slope of close (Qlib BETA)
- 公式：`Slope_W(close) / close_t` — 时间 t 上对过去 W 个 close 做 OLS 取斜率
- W ∈ {5, 10, 20, 50}
- 来源：Qlib `BETA`
- 复杂度：low
- 优先级：⭐⭐ high

### A22. Linear regression R² (Qlib RSQR)
- 公式：`R²` of OLS(close ~ time index, window=W)
- W ∈ {5, 10, 20, 50}
- 来源：Qlib `RSQR`
- 复杂度：low
- 优先级：⭐ medium

### A23. Linear regression residual (Qlib RESI)
- 公式：`(close_t - predicted_close_t) / close_t`，predicted 来自 W-window OLS
- W ∈ {5, 10, 20}
- 来源：Qlib `RESI`
- 复杂度：low
- 优先级：⭐ medium

### A24. Quantile features (Qlib QTLU/QTLD)
- 公式：`QTLU_W = quantile(close, 0.8, W) / close_t; QTLD_W = quantile(close, 0.2, W) / close_t`
- W ∈ {5, 10, 20, 50}
- 复杂度：low
- 优先级：⭐ medium

### A25. RANK feature (Qlib)
- 公式：`Rank_W(close)` — current close 在过去 W 个 close 中的 percentile rank
- W ∈ {5, 10, 20, 50}
- 这是 ts_rank，**单股票合法**（不是 cross-section）
- 复杂度：low
- 优先级：⭐⭐ high

### A26. CNTP/CNTN/CNTD (up/down day count, Qlib)
- 公式：
  ```
  CNTP_W = mean(close > delay(close, 1), W)
  CNTN_W = mean(close < delay(close, 1), W)
  CNTD_W = CNTP_W - CNTN_W
  ```
- W ∈ {5, 10, 20, 50}
- 复杂度：low
- 优先级：⭐ medium

### A27. CORR / CORD (price-volume correlation, Qlib)
- 公式：
  ```
  CORR_W = corr(close, log(volume + 1), W)
  CORD_W = corr(close/delay(close,1), log(volume/delay(volume,1) + 1), W)
  ```
- W ∈ {5, 10, 20, 50}
- 来源：Qlib `CORR/CORD`
- 复杂度：low
- 优先级：⭐⭐ high (price-volume divergence is classic alpha)

### A28. WVMA (volume-weighted price change vol, Qlib)
- 公式：
  ```
  ratio_t = |close_t / close_{t-1} - 1| · volume_t
  WVMA_W = std(ratio, W) / (mean(ratio, W) + ε)
  ```
- W ∈ {5, 10, 20, 50}
- 来源：Qlib `WVMA`
- 复杂度：low
- 优先级：⭐⭐ high

### A29. CMF (Chaikin Money Flow)
- 公式：
  ```
  MFM = ((close - low) - (high - close)) / (high - low + ε)
  MFV = MFM · volume
  CMF_W = sum(MFV, W) / sum(volume, W)
  ```
- W ∈ {10, 20}
- 复杂度：low
- 优先级：⭐ medium

### A30. OBV (On-Balance Volume)
- 公式：`OBV_t = OBV_{t-1} + sign(Δclose_t) · volume_t`；用 W-window 增量 `OBV_W = OBV_t - OBV_{t-W}`
- 复杂度：low
- 优先级：⭐ medium

### A31. Force Index
- 公式：`FI_W = ema_W(volume · Δclose)`
- W ∈ {2, 13, 50}
- 复杂度：low
- 优先级：⭐ low

### A32. Mass Index
- 公式：
  ```
  range = high - low
  ratio = ema_9(range) / ema_9(ema_9(range))
  MI_W = sum(ratio, W)
  ```
- W ∈ {25}
- 复杂度：low
- 优先级：⭐ low

---

## B. 流动性 / 微观结构 (Spread, Depth, OFI variants)

### B1. Roll's Effective Spread Estimator
- 公式：`s_W = 2 · sqrt( max( -cov(Δp_t, Δp_{t-1}), 0 ) )`，over past W ticks
- 来源：Roll 1984 Journal of Finance — `r20_papers/roll_1984.md`
- W ∈ {30, 50, 100}
- 适配：✅ 单股票，stateless
- 复杂度：low
- 优先级：⭐⭐ high (与 schema 已有 spread 互补 — 后者是 quoted spread，Roll 估的是 effective spread)

### B2. Roll's Impact = Roll spread × dollar volume
- 公式：`RollImpact_W = sqrt( max( -cov(Δlog_p_t, Δlog_p_{t-1}), 0 ) ) · sum(volume · price, W)^β`，β 通常 1
- 来源：AFML Ch 19 §19.3
- 复杂度：low
- 优先级：⭐ medium

### B3. Corwin-Schultz HL Spread Estimator
- 公式（关键三参数）：
  ```
  γ = log(max(H_{t,t+1}) / min(L_{t,t+1}))²       # 2-day high-low log ratio²
  β = log(H_t/L_t)² + log(H_{t+1}/L_{t+1})²       # sum of 1-day high-low log²
  α = (sqrt(2β) - sqrt(β)) / (3 - 2·sqrt(2)) - sqrt(γ / (3 - 2·sqrt(2)))
  Spread_HL = 2·(exp(α) - 1) / (1 + exp(α))
  ```
- 在 tick 频率：用 100-tick 切两半（或 20-tick × 2）作为 "two-day" 替代
- 来源：Corwin & Schultz 2012 JOF — `r20_papers/corwin_schultz_2012.md`
- 复杂度：low
- 优先级：⭐⭐ high (用 high/low/close 估 spread + 隐含 vol)

### B4. Becker-Parkinson Volatility (Corwin-Schultz vol component)
- 公式：
  ```
  k1 = 4·ln(2);  k2 = sqrt(8/π)
  σ_HL = (sqrt(β/2) - sqrt(β)) / (k2·(3 - 2·sqrt(2))) + sqrt(γ / (k2²·(3 - 2·sqrt(2))))
  ```
- 来源：Bekker 1990 + Parkinson 1980 + AFML Ch 19
- 复杂度：low
- 优先级：⭐ medium (跟 RV 互补)

### B5. Kyle's Lambda (price impact regression)
- 公式：在过去 W 窗口内：
  ```
  signed_dollar_vol_t = sign(Δmid_t) · |volume_delta_t · mid_t|^0.5  (或 plain volume)
  λ̂ = OLS slope of:  Δmid_t ~ signed_dollar_vol_t
  ```
- 输出：rolling λ̂_W (W ∈ {30, 50, 100})；同时输出 rolling R²
- 来源：Kyle 1985 Econometrica + frds.io implementation guide
- 适配：✅ 单股票
- 复杂度：medium (rolling OLS — 用 numpy.lib.stride_tricks + closed-form)
- 优先级：⭐⭐ high (price impact 是 LOB 学位毕业必备)

### B6. Amihud Illiquidity
- 公式：`Amihud_W = mean( |r_t| / (volume_delta_t · mid_t + ε), W )`
- W ∈ {30, 50, 100}
- 来源：Amihud 2002 J. Financial Markets
- 复杂度：low
- 优先级：⭐⭐ high (illiquidity 是 PnL 反向 predictor)

### B7. Hasbrouck λ (price impact via root-vol regression)
- 公式：
  ```
  λ_H = OLS( Δlog(p_t) ~  signed_root_vol_t )
  signed_root_vol_t = sign(Δp_t) · sqrt( volume_t · p_t )
  ```
- 来源：Hasbrouck 1991 + AFML Ch 19 §19.4
- 复杂度：medium
- 优先级：⭐ medium (与 Kyle λ 强相关，但 root 形式对 outlier 更鲁棒)

### B8. Multi-Level OFI with deeper L (extend T3)
- 现有 T3 = 10 维 × 3 windows = 30
- 公式（已 T3 实现，扩展）：
  ```
  e_k(t) = I(b_k_t ≥ b_k_{t-1}) · bs_k_t  - I(b_k_t ≤ b_k_{t-1}) · bs_k_{t-1}
         - I(a_k_t ≤ a_k_{t-1}) · as_k_t + I(a_k_t ≥ a_k_{t-1}) · as_k_{t-1}
  OFI_k_W = sum(e_k, W)
  ```
- **新增建议**：
  - **Sum across levels**: `OFI_total_W = sum_{k=1..10} OFI_k_W`
  - **Decayed weights**: `OFI_decayed_W = sum(α^(k-1) · OFI_k_W, k=1..10), α=0.7`
  - **Far/near contrast**: `OFI_near = sum(OFI_k, k=1..3); OFI_far = sum(OFI_k, k=8..10); OFI_diff = OFI_near - OFI_far`
- 来源：Cont-Kukanov-Stoikov 2014; Kolm-Turiel-Westray 2023
- 复杂度：low
- 优先级：⭐⭐ high

### B9. Volume Synchronized OFI（VSOFI）
- 公式：把过去 V 个单位的 dollar volume 视为一个 bucket，bucket 内 sum OFI；类似 VPIN volume bucket
- 来源：Easley-Lopez de Prado-O'Hara 2012 改造
- 复杂度：medium
- 优先级：⭐ medium

### B10. ARIMA-OFI residual
- 公式：拟合 OFI_t ~ AR(p) over W=50；用残差 ε_t 作为"surprise OFI"
- 来源：Hendershott-Jones-Menkveld 2011 OFI implementation
- 复杂度：medium
- 优先级：⭐ low

### B11. Generalized OFI（GOFI）— Cao Hansch Wang 2008
- 公式：在 OFI 基础上 + cancellation 信号：
  ```
  e_k_GOFI(t) = OFI_classic(t) + I(b_k_t = b_k_{t-1}) · (bs_k_t - bs_k_{t-1})
              - I(a_k_t = a_k_{t-1}) · (as_k_t - as_k_{t-1})
  ```
- 即"价格不变时 size 变化也算入信号"
- 来源：Cao-Hansch-Wang 2008; arxiv 2112.02947
- 适配：✅
- 复杂度：low
- 优先级：⭐⭐ high (capture passive size adjustments — schema 已有 size 但没编进 OFI)

### B12. Top-of-book quote slope
- 公式：`slope_top = (a1 - b1) / (as1 + bs1)` — quote 厚度归一化的 spread
- 复杂度：low
- 优先级：⭐ medium

### B13. Total quoted size
- 公式：`Q_total = sum(bs_k + as_k, k=1..10)`；其 W-MA、W-std、log；以及 ratio `Q_total / sma_W(Q_total)`
- 复杂度：low
- 优先级：⭐⭐ high (quote depth proxy)

### B14. Order book Z-score (per side)
- 公式：`(bs_k - sma_W(bs_k)) / std_W(bs_k)` for k=1..10 — 单档 size 偏离
- W=50
- 复杂度：low
- 优先级：⭐ medium

### B15. Quote update intensity
- 公式：`UpdateRate_W = mean( I(bid1_t ≠ bid1_{t-1} OR ask1_t ≠ ask1_{t-1}), W )` — 比例 of ticks with quote change
- W ∈ {10, 50}
- 复杂度：low
- 优先级：⭐⭐ high (high frequency activity proxy)

### B16. Mid price change rate
- 公式：`MidChangeRate_W = mean( I(mid_t ≠ mid_{t-1}), W )` — 比例 mid moves
- 复杂度：low
- 优先级：⭐ medium

### B17. Bid-Ask Spread Volatility
- 公式：`std(spread1, W)` for W ∈ {10, 30, 50}
- 复杂度：low
- 优先级：⭐ medium

### B18. Imbalance Persistence (autocorr)
- 公式：`acorr_lag1_W = corr( imbalance_t, imbalance_{t-1}, W )`
- W=50
- 复杂度：low
- 优先级：⭐ medium

---

## C. 波动率 / 跳跃

### C1. Realized Volatility multi-window (T3 已实现，扩展窗口)
- 公式：`RV_W = sqrt( sum( (log(WAP_t/WAP_{t-1}))², W ) )`
- 已 T3：W ∈ {5, 10, 20, 50}
- **新增**：W=100, W=200（截断到现有窗口）；用不同基价：mid, WMP, vwap proxy
- 复杂度：low
- 优先级：⭐⭐ high (扩展 T3)

### C2. Bipower Variation (jump-robust RV)
- 公式：`BV_W = (π/2) · sum(|r_t|·|r_{t-1}|, W)`
- W ∈ {10, 30, 50}
- 来源：Barndorff-Nielsen & Shephard 2004 J. Financial Econ.
- 复杂度：low
- 优先级：⭐⭐ high

### C3. Realized Quarticity
- 公式：`RQ_W = (W/3) · sum(r_t^4, W)`
- 用于 jump test (z-stat)
- 来源：BNS 2004
- 复杂度：low
- 优先级：⭐ medium

### C4. BNS Jump Test Statistic
- 公式：
  ```
  J_W = (RV_W - BV_W) / sqrt( ((π²/4) + π - 5) · max(BV_W², RQ_W) / W )
  ```
- 大值 → recent jumps occurred
- 来源：Barndorff-Nielsen-Shephard 2006 JFE
- 复杂度：low
- 优先级：⭐ medium (jump regime indicator)

### C5. Jump Variation (truncated diff)
- 公式：
  ```
  JumpVar_W = sum( I( |r_t| > 4·sqrt(BV_W/W) ) · r_t² , W )
  ```
- 4 是常用 threshold
- 来源：Mancini 2009; LdP AFML
- 复杂度：low
- 优先级：⭐ medium

### C6. Realized Skewness / Kurtosis
- 公式：
  ```
  RSkew_W = (sqrt(W) · sum(r_t³, W)) / RV_W^1.5
  RKurt_W = (W · sum(r_t^4, W)) / RV_W²
  ```
- 来源：Amaya-Christoffersen-Jacobs-Vasquez 2015 JFE
- 复杂度：low
- 优先级：⭐⭐ high (RSkew 是 short-term return predictor — Amaya 2015)

### C7. Realized Semi-variance (downside / upside)
- 公式：
  ```
  RV_pos_W = sum( r_t² · I(r_t > 0), W )
  RV_neg_W = sum( r_t² · I(r_t < 0), W )
  RVS_signed_W = (RV_pos_W - RV_neg_W) / RV_W
  ```
- 来源：Barndorff-Nielsen-Kinnebrock-Shephard 2010
- 复杂度：low
- 优先级：⭐⭐ high (signed vol asymmetry — directional info)

### C8. Garman-Klass Volatility (HL-aware)
- 公式：`GK_W = sqrt( (1/W) · sum( 0.5·log(H/L)² - (2·log2 - 1)·log(C/O)² , W ) )`
- 来源：Garman & Klass 1980 J. Business
- 复杂度：low
- 优先级：⭐ medium

### C9. Parkinson Volatility (HL-only)
- 公式：`Park_W = sqrt( (1/(4·log2)) · mean( log(H/L)², W ) )`
- 来源：Parkinson 1980 J. Business
- 复杂度：low
- 优先级：⭐ medium

### C10. Rogers-Satchell Volatility (drift-free HL)
- 公式：`RS_W = sqrt( mean( log(H/C)·log(H/O) + log(L/C)·log(L/O), W ) )`
- 来源：Rogers & Satchell 1991
- 复杂度：low
- 优先级：⭐ medium

### C11. Yang-Zhang Volatility (most efficient HL+OC)
- 公式：
  ```
  σ_oc² = mean( log(O_t/C_{t-1})², W )       # overnight (我们 setup 没有，用 log(O_t/C_{t-W})²)
  σ_co² = mean( log(C_t/O_t)², W )            # close-to-open intraday
  σ_rs² = RogersSatchell_W²
  k = 0.34 / (1.34 + (W+1)/(W-1))
  YZ_W² = σ_oc² + k·σ_co² + (1-k)·σ_rs²
  ```
- 来源：Yang & Zhang 2000
- 复杂度：medium
- 优先级：⭐ medium

### C12. Realized Range
- 公式：`RR_W = (1/(4·log2)) · sum( log(H_t/L_t)², W )`
- 来源：Christensen & Podolskij 2007
- 复杂度：low
- 优先级：⭐ low

### C13. EWMA Volatility (RiskMetrics-style)
- 公式：`σ²_t = λ·σ²_{t-1} + (1-λ)·r_t²`，λ ∈ {0.94, 0.97, 0.99}
- 来源：J.P. Morgan RiskMetrics 1996
- 复杂度：low
- 优先级：⭐⭐ high (EWMA vol 是 GARCH 廉价代理)

### C14. GARCH-like normalized return
- 公式：`r_normed_t = r_t / EWMA_vol_t`
- 复杂度：low
- 优先级：⭐ medium (减小 fat tail，提高 LightGBM 表现)

### C15. Volatility-of-Volatility
- 公式：`vovol_W = std(RV_5_t over past W ticks, W=20)`
- 复杂度：low
- 优先级：⭐ low

---

## D. 订单流 / Toxicity

### D1. VPIN (full BVC implementation)
- 公式：
  ```
  Step 1: 时间 bar 内 — 在过去 100 ticks 选取 W ticks 一组（如 W=10 → 10 个 bar）
  Step 2: 每个 bar 计算 ΔP_t（bar 内 last - first close）和 V_t（bar 内 volume_delta sum）
  Step 3: 估计 σ_ΔP（全样本 std of ΔP）
  Step 4: BVC: Buy_t = V_t · Φ(ΔP_t / σ_ΔP); Sell_t = V_t · (1 - Φ(ΔP_t / σ_ΔP))
  Step 5: 把 bar 按 volume cum 装到 buckets（每 bucket 体积固定 VBS）
  Step 6: VPIN = mean( |sum_Buy - sum_Sell| / VBS , over n buckets )
  ```
- **关键参数**：bar size、bucket size VBS、sample length n
- 在 100-tick 窗口下：bar=5 ticks → 20 bars；buckets 数取 5；sample length 5
- 来源：Easley-Lopez de Prado-O'Hara 2012 RFS — `r20_papers/vpin_easley_2012.md`
- 适配：✅ 单股票，stateless（在 100-tick window 内重算）
- 复杂度：medium
- 优先级：⭐⭐ high (toxicity proxy；可用作 Bayes EV 的 confidence gate)

### D2. Lee-Ready Tick Rule (signed volume)
- 公式：
  ```
  sign_t = +1 if close_t > close_{t-1}
         = -1 if close_t < close_{t-1}
         = sign_{t-1} if close_t = close_{t-1}     # tick test, 用上次 sign
  signed_vol_t = sign_t · volume_delta_t
  TickRuleImb_W = sum(signed_vol_t, W) / sum(volume_delta_t, W)
  ```
- 来源：Lee-Ready 1991 JOF
- 适配：✅
- 复杂度：low
- 优先级：⭐⭐ high

### D3. Easy Tick Imbalance (3-state)
- 公式：`tickimb_W = mean( sign(Δclose), W )`
- W ∈ {5, 20, 50}
- 复杂度：low
- 优先级：⭐⭐ high

### D4. EWMA-OFI / EWMA-TFI
- 公式：把 OFI_t 或 TFI_t 做 EWMA：`OFI_EW(α) = α·OFI_t + (1-α)·OFI_EW_{t-1}`
- α ∈ {0.05, 0.1, 0.3, 0.5}
- 复杂度：low
- 优先级：⭐⭐ high (EWMA + OFI 双 lever；T3 已 EWMA intensities 但没 EWMA OFI)

### D5. Hawkes Cross-Excitation Ratios（已 feature_ideas.md 提及，扩展）
- 公式（基于已有 6 类 *_intst）：
  ```
  net_buy = mb + lb - cb
  net_sell = ma + la - ca
  imb_intst = (net_buy - net_sell) / (net_buy + net_sell + ε)
  
  # Cancellation pressure
  cancel_ratio_buy = cb / (lb + ε)
  cancel_ratio_sell = ca / (la + ε)
  
  # Aggressive vs passive
  aggressive_buy_ratio = mb / (mb + lb + ε)
  aggressive_sell_ratio = ma / (ma + la + ε)
  ```
- 复杂度：low
- 优先级：⭐⭐ high

### D6. Branching Ratio Proxy (Hawkes 内生性)
- 公式：`BR_t = sum(intst_t over past W) / mean(intst over longer window)`
- W=10, longer=50
- 来源：Hardiman-Bercot-Bouchaud 2013
- 复杂度：low
- 优先级：⭐ low

### D7. Cancellation rate (per side)
- 公式：`CancelRate_W = mean(ca, W) / (mean(la + ma, W) + ε)`，bid 侧同理 with cb / (lb + mb)
- 复杂度：low
- 优先级：⭐⭐ high

### D8. Signed Cancellation Imbalance
- 公式：`(cb - ca) / (cb + ca + ε)`，W-window EWMA
- 复杂度：low
- 优先级：⭐ medium

### D9. Trade Flow Imbalance Multi-window (扩展 feature_ideas #10)
- 公式：T3 没做 multi-window EWMA TFI；用 α ∈ {0.05, 0.1, 0.3, 0.5}
- 复杂度：low
- 优先级：⭐⭐ high

### D10. Order Arrival Rate (Poisson estimate)
- 公式：`λ̂_W = mean( I(any event happens), W )` for each event type
- 复杂度：low
- 优先级：⭐ medium

### D11. Toxicity Score (heuristic)
- 公式：`tox = VPIN · spread_volatility · |OFI|` — 高 toxicity = 高 VPIN + spread 大波动 + 大 OFI
- 复杂度：low
- 优先级：⭐ medium (composite gate)

### D12. Aggressor share in volume
- 公式：`aggressor_share_W = sum(mb + ma, W) / sum(mb + ma + lb + la, W)` — market vs limit ratio
- 复杂度：low
- 优先级：⭐⭐ high

---

## E. 跨档 / 形状 (Book Slope, Depth, Skew)

### E1. Book Slope (bid / ask side, multi-method)
- 公式 1（mean of price-vs-cumvol slope, feature_ideas #14）：
  ```
  slope_bid = mean over k=2..10 of (b_k - b_1) / (sum_{j=1..k} bs_j + ε)
  slope_ask = mean over k=2..10 of (a_k - a_1) / (sum_{j=1..k} as_j + ε)
  ```
- 公式 2（OLS slope）：对 (cumvol_k, price_k) for k=1..10 做 OLS，取 slope
- 来源：Næs & Skjeltorp 2006
- 复杂度：low
- 优先级：⭐⭐ high

### E2. Cumulative depth at threshold K
- 公式：找最小 k 使 sum(bs_j, j=1..k) ≥ K，返回 b_k 作为"fill price"；K 取 全局 median(volume)
- 复杂度：medium (循环找 cum)
- 优先级：⭐ medium

### E3. Order Book Convexity (curvature)
- 公式：`convex_bid = (b_1 - 2·b_5 + b_10)`；类似 ask 侧；衡量 book 形状的二阶导
- 复杂度：low
- 优先级：⭐ medium

### E4. Level concentration ratio
- 公式：`bid_concentration = sum(bs_1..bs_3) / sum(bs_1..bs_10)` — top-3 占比；ask 侧同理
- 复杂度：low
- 优先级：⭐⭐ high

### E5. Depth skew (bid vs ask asymmetry)
- 公式：`depth_skew = (sum(bs_1..bs_10) - sum(as_1..as_10)) / (sum_total + ε)`
- 复杂度：low
- 优先级：⭐⭐ high

### E6. Level-k spread vs L1 spread
- 公式：`spread_k_ratio = (a_k - b_k) / (a_1 - b_1 + ε)` for k ∈ {3, 5, 10}
- 复杂度：low
- 优先级：⭐ medium

### E7. Multi-level WMP（扩展 T3）
- T3 已有 11 维 WMP；可以加：
  - **WMP3-WMP5 ratio**: `WMP_3 / WMP_5` 衡量 multi-level momentum
  - **WMP std across levels**: `std(WMP_1, WMP_2, ..., WMP_10)` — 价格分层一致性
- 复杂度：low
- 优先级：⭐ medium

### E8. Stoikov Micro-price (full)
- 公式（简化版，实战常用）：
  ```
  M = mid + (S/2) · (2I - 1) · ρ
  S = a1 - b1; I = bs1 / (bs1 + as1); ρ ≈ 0.5（参数，常 fix or per-window estimate）
  ```
- 完整版需历史拟合 G_k(I, S) 表，对我们 stateless 不适用
- 来源：Stoikov 2018 SSRN
- 复杂度：low
- 优先级：⭐⭐ high (与已有 WMP 互补)

### E9. Imbalance Slope across levels
- 公式：`(bs_k - as_k) / (bs_k + as_k + ε)` for k=1..10 → 10 维特征 + 各档 W-EWMA
- 复杂度：low
- 优先级：⭐⭐ high (multi-level imbalance — schema 没显式有)

### E10. Quote depth ratio (bid vs ask within level)
- 公式：`bs_k / (as_k + ε)` for k=1..10 + 取 log
- 复杂度：low
- 优先级：⭐ medium

### E11. Volume-weighted bid/ask price (within depth)
- 公式：`vwbp_K = sum(b_k · bs_k, k=1..K) / sum(bs_k, k=1..K)` — top-K 加权 bid 价
- K ∈ {3, 5, 10}；ask 侧同理
- 复杂度：low
- 优先级：⭐⭐ high

### E12. Book pressure index (Optiver Trading at Close family)
- 公式（feature_ideas #40-41）：
  - `price_pressure = (bs1 - as1) · spread1`
  - `depth_pressure = (as1 - bs1) · (a10 - b10)`
- 来源：HYD 1st place writeup
- 复杂度：low
- 优先级：⭐⭐ high

### E13. Spread depth ratio
- 公式：`spread1 / (bs1 + as1 + ε)` （feature_ideas #42）
- 来源：HYD 1st
- 复杂度：low
- 优先级：⭐ medium

### E14. Ladder shape descriptors (kurtosis / skew of size dist)
- 公式：对 [bs_1, bs_2, ..., bs_10] 计算 skew 和 kurt；ask 侧同理
- 复杂度：low
- 优先级：⭐ low

---

## F. Kbar 衍生 / RSI 家族 (Qlib KMID 完整移植)

### F1. KMID — 实体相对长度
- 公式：`KMID = (close - open) / open`
- 来源：Qlib Alpha158
- 复杂度：trivial
- 优先级：⭐⭐ high

### F2. KLEN — 全 range 长度
- 公式：`KLEN = (high - low) / open`
- 复杂度：trivial
- 优先级：⭐⭐ high

### F3. KMID2 — 实体在 range 中占比
- 公式：`KMID2 = (close - open) / (high - low + ε)`
- 复杂度：trivial
- 优先级：⭐⭐ high

### F4. KUP — 上影线长度
- 公式：`KUP = (high - max(open, close)) / open`
- 复杂度：trivial
- 优先级：⭐⭐ high

### F5. KUP2
- 公式：`KUP2 = (high - max(open, close)) / (high - low + ε)`
- 复杂度：trivial
- 优先级：⭐ medium

### F6. KLOW — 下影线长度
- 公式：`KLOW = (min(open, close) - low) / open`
- 复杂度：trivial
- 优先级：⭐⭐ high

### F7. KLOW2
- 公式：`KLOW2 = (min(open, close) - low) / (high - low + ε)`
- 复杂度：trivial
- 优先级：⭐ medium

### F8. KSFT — 实体偏移（中线 vs HL midpoint）
- 公式：`KSFT = (2·close - high - low) / open`
- 复杂度：trivial
- 优先级：⭐⭐ high

### F9. KSFT2
- 公式：`KSFT2 = (2·close - high - low) / (high - low + ε)`
- 复杂度：trivial
- 优先级：⭐ medium

### F10. SUMP（Qlib RSI numerator）
- 公式：`SUMP_W = sum(max(close - delay(close,1), 0), W) / (sum(|Δclose|, W) + ε)`
- W ∈ {5, 10, 20, 50}
- 复杂度：low
- 优先级：⭐⭐ high

### F11. SUMN（Qlib RSI complement）
- 公式：`SUMN_W = sum(max(delay(close,1) - close, 0), W) / (sum(|Δclose|, W) + ε)` = 1 - SUMP_W
- 复杂度：low
- 优先级：⭐ medium (与 SUMP 互补)

### F12. SUMD（Qlib up-down diff ratio = CMO/100）
- 公式：`SUMD_W = (gain_W - loss_W) / (gain_W + loss_W + ε)`
- 复杂度：low
- 优先级：⭐⭐ high

### F13. VSUMP / VSUMN / VSUMD (Volume RSI variants, Qlib)
- 公式：把 F10-F12 中 close 换成 volume_delta
- 复杂度：low
- 优先级：⭐ medium

### F14. VMA / VSTD (Volume MA/STD, Qlib)
- 公式：`VMA_W = sma(volume, W) / volume_t; VSTD_W = std(volume, W) / volume_t`
- 复杂度：low
- 优先级：⭐ medium

### F15. Open-Close strength
- 公式：`OC_str = sign(close - open) · |close - open| / (high - low + ε)`
- 复杂度：trivial
- 优先级：⭐ medium

### F16. Body-vs-shadow ratio
- 公式：`body_shadow = |close - open| / (KUP + KLOW + ε)`
- 复杂度：low
- 优先级：⭐ medium

### F17. Doji indicator
- 公式：`doji = I(|close - open| / (high - low + ε) < 0.1)` — 实体小于 10% range
- 复杂度：trivial
- 优先级：⭐ low

### F18. Hammer / Shooting-star (binary patterns)
- 公式：
  ```
  hammer = I(KLOW > 2·|close - open| AND KUP < 0.3·|close - open|)
  shooting = I(KUP > 2·|close - open| AND KLOW < 0.3·|close - open|)
  ```
- 复杂度：low
- 优先级：⭐ low

---

## G. 高频特殊 (Tick rule, Microstructural noise, Signature plot)

### G1. Tick imbalance over volume buckets
- 公式：把 100 ticks 切成 5 个 volume bucket（每 bucket 累计 volume = total/5），每 bucket 内算 sum(sign(Δp))，输出 5 个值
- 来源：AFML Ch 19; LdP volume bars
- 复杂度：medium
- 优先级：⭐ medium

### G2. Microstructural noise ratio
- 公式：`noise_ratio = (RV_W at high freq) / (RV_W at low freq subsampled)` — 远 > 1 → 高 noise
- 复杂度：low
- 优先级：⭐ medium

### G3. Signature plot statistics
- 公式：对不同 sample lag q ∈ {1, 2, 5, 10}，计算 RV_q = sum(r_q²) / (W/q)；signature = std(RV_q across q)
- 来源：Andersen-Bollerslev 1998
- 复杂度：medium
- 优先级：⭐ low

### G4. First-order autocorr of returns（Roll's input）
- 公式：`acf1_W = corr( r_t, r_{t-1}, W )`
- 负 → bid-ask bounce 主导（高 spread）；正 → momentum
- 复杂度：low
- 优先级：⭐⭐ high

### G5. Tick reversal probability
- 公式：`p_rev_W = mean( I(sign(Δp_t) ≠ sign(Δp_{t-1})), W )`
- 复杂度：low
- 优先级：⭐ medium

### G6. Run length (current trend duration)
- 公式：`runlen_t = max k s.t. sign(Δp_{t}) = sign(Δp_{t-1}) = ... = sign(Δp_{t-k+1})`
- 复杂度：medium (loop)
- 优先级：⭐ low

### G7. Volume burst indicator
- 公式：`burst_W = volume_t / sma(volume, W)` — 当下 volume 是过去均值多少倍
- W ∈ {10, 20, 50}
- 复杂度：low
- 优先级：⭐⭐ high

### G8. Time-since-last large trade
- 公式：`tsince_t = min k s.t. volume_{t-k} > 90th percentile(volume, W)`
- W=50, normalized by W
- 复杂度：medium
- 优先级：⭐ medium

### G9. Subsampled return at multiple lags
- 公式：`r_lag_q = mid_t - mid_{t-q}` for q ∈ {1, 3, 5, 10, 30}；标准化 by RV
- 复杂度：low
- 优先级：⭐ medium

### G10. Reversal-vs-continuation conditional return
- 公式：`return_after_up = mean( r_{t+1} | r_t > 0, past W )`；类似 down — **会 leak 未来**，需用 t-W 之前训练 prior
- ⚠️ 风险：可能需要重新设计为 stateless
- 复杂度：medium
- 优先级：⭐ low

### G11. Quote stuffing detector
- 公式：`stuffing_W = sum( (cb + ca + lb + la) / (mb + ma + ε), W )` — cancels + limits 远多于 markets
- 复杂度：low
- 优先级：⭐ low

---

## H. Pairwise / Triplet / Cross-feature (Optiver Trading at Close 套路)

> HYD 1st place 关键 trick：**对所有可用价格 pair 算归一化差**，对 triplet 算几何关系。已有 feature_ideas #16-17 提及；这里给完整可落地清单。

### H1. Pairwise price imbalance (all pairs)
- 公式：`{p_i}_{p_j}_imb = (p_i - p_j) / (p_i + p_j + ε)`
- **完整 pair 集合**（在我们 schema 下）：
  - `(b1, a1)`, `(b1, mid)`, `(a1, mid)`, `(b1, WMP)`, `(a1, WMP)`
  - `(WMP, mid)`, `(WMP, vwap_proxy)`, `(open, close)`, `(open, mid)`, `(close, mid)`
  - `(b1, b5)`, `(b1, b10)`, `(a1, a5)`, `(a1, a10)`, `(b5, b10)`, `(a5, a10)`
- → 16 个 pair imb features
- 来源：HYD 1st Optiver Trading at Close
- 适配：✅
- 复杂度：trivial
- 优先级：⭐⭐⭐ high (HYD 关键 lever)

### H2. Triplet price imbalance
- 公式：`triplet = (max(p1,p2,p3) - mid(p1,p2,p3)) / (mid - min + ε)`
- **常用 triplet**：
  - `(b1, mid, a1)` — 经典 spread shape
  - `(b1, b5, b10)` — bid 侧深度形状
  - `(a1, a5, a10)` — ask 侧深度形状
  - `(b1, WMP, a1)` — WMP 在 spread 中的位置
  - `(b5, mid, a5)` — 中档对称性
- → 5 个 triplet 特征
- 来源：HYD 1st (核心 trick)
- 复杂度：low
- 优先级：⭐⭐⭐ high

### H3. Spread × imbalance ("market urgency")
- 公式：`urgency = spread1 · ((bs1 - as1) / (bs1 + as1 + ε))`
- 来源：HYD 1st
- 复杂度：trivial
- 优先级：⭐⭐ high

### H4. Imbalance momentum
- 公式：`imb_mom_W = imbalance_t - imbalance_{t-W}`，W ∈ {1, 5, 20}
- 来源：HYD 1st (feature_ideas #39)
- 复杂度：trivial
- 优先级：⭐⭐ high

### H5. Price pressure × depth pressure 组合
- 公式：见 E12 + 它们的 ratio
- 复杂度：trivial
- 优先级：⭐ medium

### H6. WAP momentum (HYD)
- 公式：`wap_mom_W = WMP_t / WMP_{t-W} - 1`，W ∈ {3, 6, 10}
- 来源：HYD 1st (`weighted_wap.pct_change(periods=6)`)
- 复杂度：trivial
- 优先级：⭐⭐ high

### H7. Spread intensity
- 公式：`spread_intst_W = std(spread1, W)` 加 `Δspread_t`
- 来源：HYD 1st (`price_spread.diff()`)
- 复杂度：trivial
- 优先级：⭐ medium

### H8. Mid price movement direction (sign-only)
- 公式：`mid_dir = sign(mid_t - mid_{t-W})`
- W ∈ {1, 5, 10}
- 来源：HYD 1st
- 复杂度：trivial
- 优先级：⭐ medium

### H9. Statistical aggregates over price/size series
- 公式：对 [b1..b10, a1..a10, bs1..bs10, as1..as10] 算 mean/std/skew/kurt
- → 16 个 stats × 4 series = 64 features
- 来源：HYD 1st (cross-stock aggregates → 我们改 within-window)
- 复杂度：low
- 优先级：⭐ medium (descriptive stats，但容易共线)

### H10. Price spread squared
- 公式：`spread_sq = spread1²`；同理 `(a_k - b_k)²` for k=1..10
- 复杂度：trivial
- 优先级：⭐ low

### H11. Liquidity imbalance (Optiver default)
- 公式：`liq_imb = (bs1 - as1) / (bs1 + as1 + ε)` ，等同 schema 已有 imbalance（核对）
- 复杂度：trivial
- 优先级：⭐ medium (确认 schema 是否已含)

### H12. Mid-WMP gap
- 公式：`gap = WMP_1 - mid` 或 `(WMP_1 - mid) / spread1`
- 复杂度：trivial
- 优先级：⭐⭐ high (price formation direction)

### H13. Volume-weighted spread
- 公式：`vw_spread = (a1·bs1 - b1·as1) / (bs1 + as1 + ε)` — 强 imbalance 拉宽 effective spread
- 复杂度：trivial
- 优先级：⭐ medium

### H14. Microprice gap to mid
- 公式：`micro_gap = micro_price - mid`，micro_price 见 E8
- 复杂度：trivial
- 优先级：⭐⭐ high

---

## 🚫 评估时**故意排除**的因子

> 这些虽然在公开库里很常见，但因评测硬约束**绝对不能用**。

| 因子 / 类别 | 排除原因 |
|---|---|
| Alpha101 中所有 `IndClass.X` 行业中性 (#48, 56, 58-59, 63, 67, 69-70, 76, 79-80, 82, 87, 89-91, 93, 97, 100) | 无行业数据，且会 cross-section |
| Alpha101 中所有 `rank()` 未改造的 cross-section | sym shuffle 后 rank 失效 |
| 任何 cross-sym z-score / rank（feature_ideas #31-33） | 评测 batch shuffle，不保证 5 sym 同时刻共存 |
| `cap`-需要的 alpha (#56) | 没市值数据 |
| Per-sym normalization | 违反硬约束 #3 |
| `delay(., 240)` / `sum(., 250)` 长窗口 | 超过 100-tick 窗口 |
| Online-updated GARCH / Kalman filter | stateless 限制 |
| Alpha101 `signedpower(x, a)` with non-trivial `a` | 数值不稳，常需 a∈[0.5, 5]，标度敏感 |
| Volume-bar bucketing across batches | 跨 batch state 不允许 |

---

## 📊 §10. Top-30 Short List — 如果只能加 30 个最高 ROI 因子

> 排序基于：(a) 学术 / Kaggle 实证强度；(b) 与 T3 已有 72 个 features 的互补性；(c) 实现成本；(d) sym-agnostic + stateless 完美适配。
>
> **强烈建议按此顺序实施**。每加 5-10 个就重训 LightGBM（GPU），对比 LOSO h_10 ensemble。

| Rank | 因子 | 类别 | 一句话理由 | 实现成本 |
|---|---|---|---|---|
| **1** | **Stoikov micro-price (E8)** | E | WMP 之外的另一种 mid，HFT 标准 | ⏱️ 0.5h |
| **2** | **Roll's effective spread (B1)** | B | quoted spread 已有，effective spread 互补；low cost | ⏱️ 1h |
| **3** | **Realized Skewness + signed RV (C6, C7)** | C | Amaya 2015 证明 RSkew 是 short-term return predictor | ⏱️ 1h |
| **4** | **Bipower Variation + jump indicator (C2, C4)** | C | jump-robust vol；regime split | ⏱️ 1h |
| **5** | **Pairwise price imbalances (H1, ~16 个)** | H | HYD 1st place 杀手锏；trivial | ⏱️ 0.5h |
| **6** | **Triplet price imbalances (H2, ~5 个)** | H | HYD 1st place 杀手锏；catches book shape | ⏱️ 0.5h |
| **7** | **Multi-level imbalance slope (E9, 10 维)** | E | schema 没显式 multi-level OBI；OFI 互补 | ⏱️ 0.5h |
| **8** | **EWMA-OFI multi-α (D4, 4 维)** | D | T3 EWMA on intensities work了，OFI 也应 EWMA | ⏱️ 1h |
| **9** | **Generalized OFI / GOFI (B11)** | B | passive size adjust 信号，OFI extend | ⏱️ 1h |
| **10** | **Tick rule signed volume (D2, multi-W)** | D | Lee-Ready 经典；TFI 强信号 | ⏱️ 0.5h |
| **11** | **Kyle's λ (B5, rolling W=50)** | B | price impact 标尺；和 RV 不同 channel | ⏱️ 2h |
| **12** | **Amihud illiquidity (B6)** | B | illiquidity 是 PnL 反向 predictor | ⏱️ 0.5h |
| **13** | **Book slope per side (E1)** | E | depth quality；feature_ideas #14 | ⏱️ 1h |
| **14** | **Top-3 vs total depth concentration (E4)** | E | 流动性集中 vs 分散；trivial | ⏱️ 0.3h |
| **15** | **Depth skew bid/ask asymmetry (E5)** | E | size-side 失衡，OBI 之外的角度 | ⏱️ 0.3h |
| **16** | **Volume-weighted bid/ask price top-K (E11)** | E | depth-aware mid alternative | ⏱️ 0.5h |
| **17** | **VPIN with BVC (D1)** | D | toxicity score；做 Bayes EV gating 输入 | ⏱️ 3h |
| **18** | **Cancellation rate per side (D7)** | D | T3 EWMA(ca) work，加 ratio 形 | ⏱️ 0.3h |
| **19** | **Cross-excitation ratios (D5, 4-6 个)** | D | passive vs aggressive intensity ratio | ⏱️ 0.5h |
| **20** | **HMA + DEMA + TEMA (A3)** | A | G-Research single best；smoothed momentum | ⏱️ 1h |
| **21** | **EWMA volatility λ ∈ {0.94, 0.97, 0.99} (C13)** | C | GARCH 廉价代理 | ⏱️ 0.5h |
| **22** | **GARCH-normalized return (C14)** | C | de-fat-tail 帮助 LightGBM | ⏱️ 0.3h |
| **23** | **Quote update rate (B15)** | B | high-freq activity 强 indicator | ⏱️ 0.3h |
| **24** | **Volume burst (G7)** | G | abnormal volume regime | ⏱️ 0.3h |
| **25** | **First-order return autocorr (G4)** | G | bid-ask bounce vs momentum 区分；Roll 输入 | ⏱️ 0.3h |
| **26** | **Qlib KMID + KMID2 + KSFT family (F1, F3, F8)** | F | Kbar 衍生，Qlib Alpha158 经典 | ⏱️ 0.5h |
| **27** | **Qlib SUMP/SUMN/SUMD = RSI 家族 (F10-12)** | F | RSI 多窗口 | ⏱️ 0.5h |
| **28** | **Qlib WVMA volume-weighted vol (A28)** | A | volume-aware vol metric | ⏱️ 0.5h |
| **29** | **Aggressor share in volume (D12)** | D | aggressive vs passive trade ratio | ⏱️ 0.3h |
| **30** | **Imbalance momentum + WAP momentum (H4, H6)** | H | HYD pct_change tricks | ⏱️ 0.3h |

**总实施时间估计**：~20 小时（一个研究迭代周期）

**预期增益**：以 T3 基线（iter_002 LOSO h_10 = +21.86 → iter_003 = +22.22）为参考，本 list 的 30 个新因子有望把 LightGBM CV PnL 再提升 **+5 ~ +12**。但需配合 R10/R13 的 PnL loss + Bayes EV 后处理才能 fully realize。

---

## 📋 §11. Roadmap：30 因子 × 3 周实施序列

| 周 | 加入因子 | 目标 |
|---|---|---|
| **W1** | Top 1-10（HYD pairwise/triplet + 价位 + Roll/Amihud + multi-level imb + EWMA OFI + GOFI + tick rule） | 量价微观结构互补层 |
| **W2** | Top 11-20（Kyle/VPIN/depth/HMA/EWMA-vol） | 流动性 + 高频 toxicity 层 |
| **W3** | Top 21-30（Qlib Kbar/RSI/WVMA + autocorr + volume burst + momentum diff） | Kbar/RSI 完整 + 高频特征收尾 |

每周末跑 LOSO h_10 5-seed Scheme C ensemble，对比 baseline + 增量贡献，剔除负贡献因子（用 LightGBM `feature_importance(split)` < 1% 作为 cull threshold）。

---

## 📚 来源汇总

### Alpha 101
- **WorldQuant/Kakushadze 2015** "101 Formulaic Alphas" arxiv:1601.00991
- yli188/WorldQuant_alpha101_code (GitHub) — Python implementation reference

### Qlib Alpha158/360
- microsoft/qlib `qlib/contrib/data/loader.py` — 完整源码已读

### AFML（Lopez de Prado）
- AFML Ch 18 "Microstructural Features"（Roll, Kyle, Amihud, Hasbrouck, VPIN）
- AFML Ch 19 §19.1-19.5 — 对应 r20 §B 的实现公式

### 微观结构 Paper
- **Roll 1984** JOF "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- **Kyle 1985** Econometrica "Continuous Auctions and Insider Trading"
- **Amihud 2002** J. Financial Markets "Illiquidity and stock returns"
- **Hasbrouck 1991** JOF "Measuring the Information Content of Stock Trades"
- **Easley-Lopez de Prado-O'Hara 2012** RFS "Flow Toxicity and Liquidity in a HFT World" (VPIN)
- **Easley-Kiefer-O'Hara-Paperman 1996** RFS "Liquidity, Information, and Infrequently Traded Stocks" (PIN)
- **Cont-Kukanov-Stoikov 2014** JFE "The Price Impact of Order Book Events"
- **Kolm-Turiel-Westray 2023** Math. Finance "Deep OFI Multi-level"
- **Stoikov 2018** SSRN "The Micro-Price"
- **Barndorff-Nielsen-Shephard 2004** JFE "Power and Bipower Variation"
- **Amaya-Christoffersen-Jacobs-Vasquez 2015** JFE "Realized Skewness"
- **Corwin-Schultz 2012** JOF "High-Low Spread Estimator"
- **Garman-Klass 1980 / Parkinson 1980 / Yang-Zhang 2000** — HL/OC vol estimators

### Kaggle
- **Optiver Trading at Close 1st (HYD)** — pairwise/triplet imb tricks 完整列表
- **Optiver Realized Vol 7th** — github.com/michaelpoluektov/orvp
- **Jane Street 1st (Yirun)** — utility loss + 4-fold day-iso CV
- **G-Research Crypto winners** — HMA single strongest

### 综述 / Modern
- **arxiv:2403.09267** "Deep LOB Forecasting: A Microstructural Guide" — 2024 综述
- **arxiv:2502.17417** "Event-Based LOB Simulation under Neural Hawkes" — 2025
- **Springer 2024** "Deep Hawkes process for HF market making"

---

## 📌 实施提示与陷阱

1. **NaN handling**：所有 rolling 在窗口前 W ticks NaN，统一 forward-fill；EWMA 用 first-tick 初始化为 0
2. **Numerical stability**：所有除法加 ε=1e-9；log 加 1e-9
3. **Feature scaling**：5-day rolling z-score → trade off：tick 频率 5 day = 800 ticks，对我们仅 100-tick window 不可用；改用**全局训练集 z-score 参数**（fit 一次，infer 时复用）
4. **Co-linearity check**：训练 LightGBM 时检查 feature importance；删除 importance < 1% 且 与已 high-importance feature corr > 0.9 的
5. **Pipeline 调整**：先全部生成 → LightGBM importance → 砍尾 30% → 上 NN（如 MLPLOB）
6. **窗口设计**：默认 W ∈ {5, 10, 20, 50}；若 100-tick 不够需要扩展，不要在 Predictor 里 cache 跨 batch — 重新在 100 ticks 内重算
7. **VPIN 特殊**：bucket size 在 100-tick 内只能 5-10 个 buckets；sample length 也只能 5；这种"短 VPIN"和原 paper 50-bucket 版本性质不同，把它当 short-term toxicity proxy
8. **Kyle λ rolling OLS**：用 numpy 的 closed-form `λ = cov(x,y)/var(x)`，不要用 scipy.linregress 在循环里（慢 10x）

---

## 🔍 与已有 R-doc 的关系

- **`feature_ideas.md`**（49 条）：本文档是它的**深化扩展**。本文新增 ~85 条，去重后实际 net 新加因子 ~70。
- **`competition_insights.md`**：Kaggle 比赛侧 trick；本文的 §H 完整化了它的 pairwise/triplet 部分。
- **`r10_pnl_loss.md` / `r13_position_calibration.md`**：本文是**特征侧**，他们是**loss + 后处理**。本 list 实施后，配合 R10 S2 (PnL custom obj) + R13 D (z-gate) 才是完整 stack。
- **`r12_ensemble_stacking.md`**：本 list 的因子可作为不同 base model 的 input 子集 → 增强 ensemble diversity（方案 1, 2 都受益）。

---

**完成于 2026-05-06，R20 worker，11+ WebSearch + 8+ WebFetch 抓取，公式全部从一手 paper / 源码提炼。**
