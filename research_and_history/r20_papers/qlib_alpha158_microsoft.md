# Microsoft Qlib — Alpha158 / Alpha360 因子库

**Source**: github.com/microsoft/qlib `qlib/contrib/data/loader.py` (full source code reviewed)

## Alpha158 = Kbar (9) + Price (4 raw) + Rolling (multiple ops × 5 windows)

### Kbar 9 个（hard-coded）—— K 线形态衍生

| 名 | 公式 | 含义 |
|---|---|---|
| KMID | `(close-open)/open` | 实体相对长度（带方向） |
| KLEN | `(high-low)/open` | 总 range |
| KMID2 | `(close-open)/(high-low+ε)` | 实体在 range 中的占比（HYD style） |
| KUP | `(high - max(open,close))/open` | 上影线 |
| KUP2 | `(high - max(open,close))/(high-low+ε)` | 上影占比 |
| KLOW | `(min(open,close) - low)/open` | 下影 |
| KLOW2 | `(min(open,close) - low)/(high-low+ε)` | 下影占比 |
| KSFT | `(2·close - high - low)/open` | 实体偏移 |
| KSFT2 | `(2·close - high - low)/(high-low+ε)` | 实体偏移占 range |

### Price 4 个 (window=0)

```
OPEN0 = open / close
HIGH0 = high / close
LOW0  = low / close
VWAP0 = vwap / close   # 我们没 vwap，可用 (O+H+L+C)/4 替代
```

### Rolling 19 类 × 5 windows = 95 个

Default windows = {5, 10, 20, 30, 60} → 在 tick 频率改为 {5, 10, 20, 50, 100}。

| 算子 | 公式（标准化后） |
|---|---|
| ROC{d} | `Ref(close,d) / close` — past d-day return inverted |
| MA{d} | `Mean(close, d) / close` — rolling avg / current |
| STD{d} | `Std(close, d) / close` — vol normalized |
| BETA{d} | `Slope(close, d) / close` — OLS slope of close ~ time |
| RSQR{d} | `Rsquare(close, d)` — OLS R² |
| RESI{d} | `Resi(close, d) / close` — OLS residual |
| MAX{d} | `Max(high, d) / close` |
| MIN{d} | `Min(low, d) / close` |
| QTLU{d} | `Quantile(close, d, 0.8) / close` — 80% quantile |
| QTLD{d} | `Quantile(close, d, 0.2) / close` |
| RANK{d} | `Rank(close, d)` — current close 在过去 d 中的 percentile |
| RSV{d} | `(close - Min(low,d)) / (Max(high,d) - Min(low,d) + ε)` — Stochastic K |
| IMAX{d} | `IdxMax(high, d) / d` — Aroon up |
| IMIN{d} | `IdxMin(low, d) / d` — Aroon down |
| IMXD{d} | `(IdxMax(high,d) - IdxMin(low,d)) / d` |
| CORR{d} | `Corr(close, log(volume+1), d)` — 价量相关 |
| CORD{d} | `Corr(close/Ref(close,1), log(volume/Ref(volume,1)+1), d)` — 收益-量变率相关 |
| CNTP{d} | `Mean(close > Ref(close,1), d)` — % up days |
| CNTN{d} | `Mean(close < Ref(close,1), d)` |
| CNTD{d} | `CNTP - CNTN` |
| SUMP{d} | `Sum(max(Δclose,0), d) / Sum(|Δclose|, d)` — = RSI/100 |
| SUMN{d} | `Sum(max(-Δclose,0), d) / Sum(|Δclose|, d)` |
| SUMD{d} | `(SUMP - SUMN)·Sum / Sum(|Δclose|)` — CMO/100 |
| VMA{d} | `Mean(volume, d) / volume` |
| VSTD{d} | `Std(volume, d) / volume` |
| WVMA{d} | `Std(\|Δclose/close\|·volume, d) / Mean(\|Δclose/close\|·volume, d)` — vol-of-vol-weighted |
| VSUMP{d} | `Sum(max(Δvolume,0), d) / Sum(\|Δvolume\|, d)` |
| VSUMN/VSUMD | 同 SUMN/SUMD on volume |

总计：~95 rolling features。

## Alpha360 = 60-tick raw history × 6 fields

```
For i in 59..0:
    CLOSE_i = Ref(close, i) / close
    OPEN_i  = Ref(open, i) / close
    HIGH_i  = Ref(high, i) / close
    LOW_i   = Ref(low, i) / close
    VWAP_i  = Ref(vwap, i) / close
    VOLUME_i = Ref(volume, i) / (volume + ε)
```

→ 360 维原始历史快照。CLOSE0 = 1（自身）。这是 "raw history" 路线，给 deep model（GRU/Transformer）当输入。

**对我们**：60-tick window 完全在 100-tick 内可用，但 360 维输入数太多容易共线，**不推荐直接 dump**；只做 deep model 的 raw input baseline 时考虑。

## 适配评估

| 模块 | 评分 | 备注 |
|---|---|---|
| Kbar (9) | ✅✅✅ | 直接用，trivial 实现 |
| Price (4) | ✅ | 简单 ratio |
| ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN/QTLU/QTLD/RANK/RSV | ✅✅ | 都是单股票时序，全可用 |
| IMAX/IMIN/IMXD | ✅ | Aroon 类 |
| CORR/CORD | ✅✅ | 价量交互（高价值） |
| CNTP/CNTN/CNTD | ✅ | up/down day count |
| SUMP/SUMN/SUMD = RSI 家族 | ✅✅✅ | 经典 |
| VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD | ✅✅ | 量侧版本 |

**全部 158 个**都是 sym-agnostic + stateless（评测过）。**最高 ROI = Kbar 9 + RSI 家族 + WVMA + CORR**。

## 与 T3 已有的差异

T3 有 EWMA intensities, MLOFI, WMP, RV, time encoding。**Qlib 全是价量 K 线类**，与 T3 几乎不重叠。
- KMID family (Kbar) **完全没在 T3**
- RSI / SUMP / WVMA **没在 T3**
- BETA / RSQR / RESI （rolling OLS 类）**没在 T3**
- CORR / CORD **没在 T3**

→ Qlib Alpha158 移植是**最大正交补充**。

## 来源链接

- 源码：https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- 笔记：Alpha158 在 Microsoft Research 的多篇 paper（如 RNN+RankIC）中是 baseline；社区主跑 LightGBM 上 IC 0.05-0.08 量级。

## 实施备注

1. Qlib 的 `Slope`/`Rsquare`/`Resi` 是 closed-form OLS（用矩阵解析解，不要循环）；公式：
   ```
   X = arange(W); X̄ = (W-1)/2; Var(X) = (W²-1)/12
   Slope = sum((X - X̄)·(close - mean(close, W))) / (W · Var(X))
   ```
2. `IdxMax`/`IdxMin` 用 `numpy.argmax/argmin` over rolling window，但 Aroon 是 `(W - argmax_idx_from_oldest) / W`，注意起点
3. 对我们 100-tick：用 `windows = [5, 10, 20, 50, 100]` 取代 default `[5, 10, 20, 30, 60]`
4. `WVMA` 是非常独特的指标 — vol-of-(vol·return)，不要简化成 `std(volume)`

## 推荐先实施 30 个

10 × {ROC, MA, STD} = 30 + Kbar 9 + RSI(SUMP/SUMN/SUMD) × 4 windows = 12 + WVMA × 4 + CORR/CORD × 4 = 8 → 共 ~60 个。
但与 30-shortlist 重叠的优先：**KMID/KMID2/KSFT (3) + SUMP/SUMD × 4 windows (8) + WVMA × 4 (4) + CORR × 4 (4) = 19 个新 features**。
