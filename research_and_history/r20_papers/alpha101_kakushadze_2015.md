# Kakushadze 2015 — 101 Formulaic Alphas

**arxiv:1601.00991** (December 2015) — Zura Kakushadze (Quantigic Solutions LLC)

## Core claim

WorldQuant 在生产环境用的 **101 个 explicit formulaic alpha**——不是玩具，80+ 在 active production。每条 alpha 都是显式 close-form 公式，输入只有 OHLCV + vwap + adv（average daily dollar volume）+ industry classification。

## 关键统计（Table 1）

| 指标 | 中位数 | 均值 |
|---|---|---|
| 年化 Sharpe | 2.224 | 2.265 |
| Daily turnover | 0.4752 | 0.5456 |
| Holding period (1/T) | 2.104 days | 2.391 days |
| Pair-wise correlation | 14.3% | 15.9% |

→ 平均 holding period **0.6-6.4 天**，所以这些是 mid-frequency；**不是 tick-frequency**。在我们 setup 下需要把日窗口缩到 tick 窗口（W=20 → W=20 ticks）。

## Helper functions（Appendix A.1-A.2）

```
abs(x), log(x), sign(x)               # 标准
rank(x)                               # cross-sectional rank — ⛔ 不能直接用
delay(x, d)                           # x at t-d
correlation(x, y, d)                  # rolling Pearson over d
covariance(x, y, d)                   # rolling cov
delta(x, d)                           # x_t - x_{t-d}
signedpower(x, a)                     # sign(x) · |x|^a
decay_linear(x, d)                    # WMA: weights d, d-1, ..., 1
indneutralize(x, IndClass.X)          # ⛔ 行业中性化，无行业数据不可用
ts_{O}(x, d)                          # 时序 O over past d (O ∈ {min, max, argmax, argmin, rank, sum, product, stddev})
scale(x, a=1)                         # rescale s.t. sum(|x|) = a
```

## Inputs（Appendix A.2）

- `returns` = daily close-to-close returns
- `open, close, high, low, volume`
- `vwap` = daily volume-weighted avg price — 我们 schema 没有，可用 `(open+high+low+close)/4` 近似
- `cap` = market cap — ⛔ 不可用（schema 没）
- `adv{d}` = avg daily dollar volume past d days — 我们用 `sma(volume_delta · mid, d)` 替代
- `IndClass` — ⛔ 不可用

## 适配到我们 setup 的 filter

| 类别 | 数量（估） | 状态 |
|---|---|---|
| 完全单股票（无 rank, 无 IndClass） | ~12 | ✅ 直接用：#6, #9, #12, #23, #28, #32, #41, #46, #49, #51, #53, #54, #101 |
| 用 cross-sectional rank（可改 ts_rank） | ~50 | ⚠️ 可改造，但 ts_rank 与 cross rank 性质不同 |
| 用 IndClass 行业中性化 | ~19 | ⛔ 不可用：#48, 56, 58-59, 63, 67, 69-70, 76, 79-80, 82, 87, 89-91, 93, 97, 100 |
| 需要 cap | 1 (#56) | ⛔ |
| 需要 240/250 day window | 几个 | ⚠️ 改 W=50-100 或不用 |

## 高 ROI 的"完全单股票" alpha（节选 5 条）

- **Alpha#6**: `-corr(open, volume, 10)` — 价量负相关 → reversion
- **Alpha#9**: 三重条件 reversal — `if(min(Δclose,5)>0): Δclose; elif(max(Δclose,5)<0): Δclose; else: -Δclose`
- **Alpha#12**: `sign(Δvolume,1) · (-Δclose,1)` — 量增价跌反转
- **Alpha#41**: `sqrt(high·low) - vwap` — geometric mid 与 vwap gap
- **Alpha#101**: `(close - open) / (high - low + 0.001)` — 日内 K 线方向强度（同 Qlib KMID2）

## 与 Qlib Alpha158 的关系

Qlib KMID family（KMID/KLEN/KMID2/KUP/KLOW/KSFT 等 9 个）几乎覆盖了 Alpha101 中所有"日内 K 线形态"alpha。所以两库在低层有重叠。Alpha101 的独特价值在 **价-量交互（correlation 类）+ 多 time scale rolling rank**。

## 对我们的启发

1. **Cross-sectional rank ≠ Time-series rank**：Alpha101 的 `rank()` 设计基于跨股票排序，改成 `ts_rank()` 后语义变化（变成"过去 W 时刻的相对位置"）。一些 alpha 可能失效，但**很多保留 directional 价值**。
2. **Holding period 适配**：原 alpha 0.6-6.4 day，对应 daily horizon。我们标签是 5/10/60 tick → 短 100x，所以原 alpha 中 "mean reversion" 类（短窗）应该比 "momentum" 类（长窗）更适配。
3. **Performance R∝V^X with X≈0.76**：alpha return 与 volatility 强相关（0.76 power scaling）。→ 高 vol 时 alpha 更强；用我们的 RV 特征做 alpha gating 可能 work。

## 主要引用

- Avellaneda-Lee 2010, Quant. Finance — stat arb
- Jegadeesh-Titman 1993 JOF — momentum
- Pastor-Stambaugh 2003 JPE — liquidity factor
- Kakushadze-Tulchinsky 2015 — performance v turnover
- Tulchinsky et al. 2015 — "Finding Alphas" book

## 笔记

完整 101 个 alpha 公式已在 `r20_factor_library.md §A11` 表格中分类标注。重点筛选了 ~12 个完全单股票可用的，加 ~50 个可 ts_rank 改造的。
