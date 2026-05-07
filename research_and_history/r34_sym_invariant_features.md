# R34 — Sym-Invariant Feature Catalog (穷举)

> Date: 2026-05-07
> Author: R34 worker (research-only)
> Goal: 列出**所有可能的 sym-agnostic 特征构造方法**，特别是修 fold 2 (sym=2 held) confidently-wrong overfit。
>
> ⚠️ 三条硬约束（CRITICAL_CONSTRAINTS.md 抄一遍）：
> 1. `date` 评测置 0 — 不能当 feature
> 2. Predictor 不能跨 `predict()` 调用维护 state
> 3. **sym ID 训练后不能在 inference 用** — 任何 per-sym normalization / per-sym 模型 / sym embedding 都禁止
>
> **所有下文公式都满足**：(a) 在 100-tick window 内可计算；(b) 不依赖 sym ID / date；(c) Predictor 看到任何 100×D 切片都能算（stateless）。

---

## ⚠️ 当前状态回顾

- **226-d Scheme C** = 154 raw last-tick + 30 MLOFI + 11 WMP + 4 RV + 24 EWMA + 3 time（飞地，T3 已 cache）
- **iter_006 LOSO h_60 = +13.61**（已饱和）
- **Fold 2 bug**：sym 2 (蓝筹 ETF, spread=2.7bp, σ=2bp, amount=151k 元/tick) 是其他 4 sym 的 OOD outlier。模型在 fold 2 上 acc=0.279 vs naive baseline 0.608 — **confidently wrong**。
- **D1 (window-zscore 308d) 单跑** LOSO h_60 +13.91（最早期），再没继续打磨。
- **Aug_a [0.80, 1.20]** 部分缓解 spread* 依赖，但不够。

**Root cause**：226-d 中**有相当一部分特征本身是 scale-coupled**（spread1=23bp 在 sym 3 上是 1，在 sym 2 上是 0.05，根本不在同一分布），aug_a 砍掉了 FI 但**raw feature 仍然在分布上偏移**。LightGBM 的 split 阈值是从训练集学的；当 sym 2 的 spread 分布完全 off-distribution 时，旧的 split 阈值给出错误信号。

**唯一的修法**：让特征在**最终送进模型前已经是 cross-sym 同分布**——即所有特征都是**比例/形状/局部归一化**而非绝对值。

---

## 核心理论：为什么"局部归一化"是必经之路

### 数学论证

设 sym $i$ 的 spread 服从分布 $S_i \sim D_i$（不同 sym 的 $D_i$ 不同，spread 范围 2.7-23.4 bp，4-9 倍 scale 差）。
当我们用 raw `spread1` 当 feature：

$$
\Pr[\text{spread1}=s \mid \text{sym}=i] = D_i(s)
$$

模型学到的 split (e.g. `spread1 > 5`) 对 sym 0/1/4 (spread 5-8 bp) 是有意义的中位数；对 sym 2 (spread 2.7 bp) 几乎永远 false；对 sym 3 (spread 23 bp) 几乎永远 true。**模型的 logical structure 在不同 sym 上塌陷为常数**。

但若我们把 `spread1` 替换为 `spread1 / median(spread1, W=100)`（局部归一化），分布变为：

$$
\Pr[\text{spread1}/\text{median} \approx 1 \mid \text{sym}=i] \approx \text{constant for all } i
$$

**即所有 sym 在归一化后的特征分布近乎一致**，模型的 split 在所有 sym 上都有意义。

### Kyle-Obizhaeva 微结构不变假设（2016 Econometrica）

Kyle & Obizhaeva 2016 证明：在**业务时间**（business time）下，跨股票的 risk transfer 分布**不变**。具体地，把交易活动 $W = P \cdot V \cdot \sigma$（dollar-volatility 复合），则：
- bet rate $\gamma \propto W^{2/3}$
- bet size $|Q| \propto W^{1/3}$
- Risk per bet $\propto W^{0}$ ← **不变量**

**对我们的启示**：用 $W^{1/3}$ 标准化任何 size 类特征（volume、bsize_k、order arrival rate 等），结果应该跨股票同分布。我们已有 amount_delta（=P·V）和 RV_50（≈σ），可以构造 $W^{1/3}$ 当归一化分母。

来源：Kyle-Obizhaeva 2016 Econometrica "Market Microstructure Invariance: Empirical Hypotheses"。

---

## A. Adaptive Normalization（让特征跨 sym 一致）

> **核心思想**：把每个 raw feature 替换为它在过去 W=100 ticks 窗口内的**局部相对位置**。所有局部归一化都是 sym-agnostic（窗口内统计完全独立于 sym ID）。

### A1. Window z-score (T7 D1 已实现，但仅基线，需扩展 multi-W)

```python
def window_zscore(x: np.ndarray, W: int) -> np.ndarray:
    # x: (T,) ; returns z (T,) with first W-1 = NaN
    sw = np.lib.stride_tricks.sliding_window_view(x, W)  # (T-W+1, W)
    mu = sw.mean(-1); sd = sw.std(-1) + 1e-8
    z = (x[W-1:] - mu) / sd
    return np.concatenate([np.full(W-1, np.nan), z])
```

- **窗口 W ∈ {20, 50, 100}**（multi-horizon adaptive：短窗口跟瞬时震荡，长窗口跟整段水平）
- **应用对象**（约 30-50 raw 列）：bid1, ask1, bid_mean, ask_mean, bsize1..bsize10, asize1..asize10, totalbsize, totalasize, spread1..spread10, midprice, volume_delta, amount_delta, all 6 *_intst, all 6 *_acc, imbalance, cumspread
- **sym-agnostic**：✅ 完全（window 统计是 sym-aware 的，但只在 100-tick window 内，不跨 sym borrow stats）
- **对 fold 2 的预期帮助**：sym 2 的 spread1 raw value ≈ 2.7bp 但归一化后 z 值与其他 sym 同分布

### A2. Window quantile rank (median-robust)

```python
def window_qrank(x: np.ndarray, W: int) -> np.ndarray:
    # Returns rank in [0,1] of x[t] among x[t-W+1..t]
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    rank = (sw <= sw[:, -1:]).sum(axis=-1).astype(np.float32) / W
    return np.concatenate([np.full(W-1, np.nan), rank])
```

- **比 z-score 更 robust**：异常值不影响 rank
- W ∈ {50, 100}
- 对 spread1 / vol burst 类异常值场景**强烈建议**（sym 3 spread 偶发 jump，z-score 会被拉飞，rank 不会）
- **sym-agnostic**：✅

### A3. Min-max normalization within window

```python
def window_minmax(x: np.ndarray, W: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    mn = sw.min(-1); mx = sw.max(-1)
    out = (x[W-1:] - mn) / (mx - mn + 1e-8)
    return np.concatenate([np.full(W-1, np.nan), out])
```

- 输出 [0, 1]，比 z-score 对 outlier 更敏感（不一定坏）
- W=100
- 与 A2 互补但更轻量
- **sym-agnostic**：✅

### A4. Robust scaler (median / IQR within window)

```python
def window_robust(x: np.ndarray, W: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    med = np.median(sw, axis=-1)
    q75 = np.quantile(sw, 0.75, axis=-1)
    q25 = np.quantile(sw, 0.25, axis=-1)
    iqr = q75 - q25 + 1e-8
    return np.concatenate([np.full(W-1, np.nan), (x[W-1:] - med) / iqr])
```

- 比 z-score robust（不被 fat-tail 主导）
- 比 A2 保留单调缩放（rank 损失了量级信息）
- W=100
- **sym-agnostic**：✅

### A5. Window log-mean ratio

```python
def window_log_ratio(x: np.ndarray, W: int) -> np.ndarray:
    # log(current / window_mean)
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    return np.concatenate([np.full(W-1, np.nan), 
                           np.log((x[W-1:] + 1e-8) / (sw.mean(-1) + 1e-8))])
```

- 对乘性变化鲁棒（spread × 2 → log_ratio = +0.69，不论 baseline）
- 对 size 类（amount_delta, totalbsize）尤其适合
- W=50
- **sym-agnostic**：✅

### A6. Causal Box-Cox / Yeo-Johnson with rolling λ

> ⚠️ **不推荐**：rolling λ 估计在 100-tick 内统计量不稳，易过拟合 noise。改用 fixed λ=0.5（即 `sign·sqrt(|x|)`）做轻量 power-transform。

```python
power_root_signed = np.sign(x) * np.sqrt(np.abs(x))  # 等价 Yeo-Johnson with λ=0.5
```

- 对 fat-tail（amount_delta, OFI）有效
- 不需 window 估计 λ
- **sym-agnostic**：✅
- 优先级：⭐ low（aug_a 已经强制模型对 scale 不敏感，再做 sqrt 边际收益小）

### A7. Causal exponentially weighted moving (EWM) z-score

```python
def ewm_zscore(x: np.ndarray, alpha: float) -> np.ndarray:
    # 替代 window z-score，无窗口边界问题，所有 tick 都有值
    mu = pd.Series(x).ewm(alpha=alpha, adjust=False).mean().values
    sd = pd.Series((x - mu)**2).ewm(alpha=alpha, adjust=False).mean().values ** 0.5 + 1e-8
    return (x - mu) / sd
```

- α ∈ {0.05, 0.1, 0.3}（半衰期 ~14 / 7 / 2 ticks）
- 与 A1 互补：A1 是 hard window，A7 是 soft window
- **sym-agnostic**：✅

### A8. Window CDF position (类似 A2 但插值)

```python
def window_cdf(x: np.ndarray, W: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    sorted_w = np.sort(sw, axis=-1)
    pos = np.searchsorted(sorted_w, x[W-1:][:, None], side='right').squeeze() / W  # 需 vectorize
    return ...  # 实际 pandas/numpy 实现见 scipy.stats.rankdata over rolling
```

- 输出 [0, 1] CDF 位置
- 与 A2 数学上等价，但 numpy 实现稍贵
- **sym-agnostic**：✅
- 优先级：⭐ low（A2 已覆盖）

### A9. **Cross-window Dual-z-score (本提案的核心创新)**

```python
def dual_zscore(x: np.ndarray, W_short: int=20, W_long: int=100) -> np.ndarray:
    # short z = 当下相对短期均值
    # long z = 当下相对长期均值
    # 两者差 = 短期 vs 长期 偏离程度
    z_s = window_zscore(x, W_short)
    z_l = window_zscore(x, W_long)
    return z_s - z_l  # "瞬时偏离 minus 长期偏离" — sym-agnostic
```

- **每个 raw 列 → 1 维 dual_z**
- 这是模型最容易消化的格式：tells "当下相对自己最近的偏离程度"，跨 sym 同分布
- W_short=20, W_long=100
- **sym-agnostic**：✅
- **预期对 fold 2 修复**：sym 2 突发 spread 扩大时 dual_z 变高，与 sym 3 平时高 spread 但稳定时 dual_z=0 区分开

### A10. Sigmoid-of-z (squash extreme values)

```python
sig_z = 1 / (1 + np.exp(-window_zscore(x, 100)))  # 输出 [0, 1]
```

- 把 z-score 压到 (0, 1)，对 LightGBM split 更平滑
- **sym-agnostic**：✅
- 优先级：⭐ low（z-score 直接给 LightGBM 通常已够）

---

## B. 微结构不变量（economic-meaningful sym-agnostic）

> **核心思想**：经济学定义本身就是无量纲的指标，不依赖 sym 的 scale。

### B1. Roll's effective spread estimator

```python
def roll_spread(delta_p: np.ndarray, W: int) -> np.ndarray:
    # delta_p = mid diff series (in normalized price units)
    sw = np.lib.stride_tricks.sliding_window_view(delta_p, W)
    sw_lag = np.lib.stride_tricks.sliding_window_view(delta_p[:-1], W)  # offset by 1
    # cov(delta_p_t, delta_p_{t-1}) within window
    cov = ((sw - sw.mean(-1, keepdims=True)) * 
           (sw_lag - sw_lag.mean(-1, keepdims=True))).mean(-1)
    return 2 * np.sqrt(np.maximum(-cov, 0))
```

- 公式：`s_W = 2·sqrt(max(-cov(Δp_t, Δp_{t-1}), 0))`
- W ∈ {30, 50, 100}
- **sym-agnostic**：✅（cov 本身有量纲，但跟 quoted spread 配 ratio `roll_spread / spread1` 是无量纲）
- 与 schema 已有 quoted spread 互补：roll = effective spread (实际成交折扣)
- 来源：Roll 1984 JOF
- **预期对 fold 2**：sym 2 的 quoted spread 极小但可能有非零 effective spread；这能区分"窄 quote 但 noisy" vs "窄 quote 真无成本"

### B2. Kyle's λ (price impact regression)

```python
def kyle_lambda(delta_mid: np.ndarray, signed_dvol: np.ndarray, W: int) -> np.ndarray:
    # signed_dvol = sign(delta_mid) * sqrt(|amount_delta|)
    # rolling OLS slope of delta_mid ~ signed_dvol
    sw_y = np.lib.stride_tricks.sliding_window_view(delta_mid, W)
    sw_x = np.lib.stride_tricks.sliding_window_view(signed_dvol, W)
    cov = ((sw_x - sw_x.mean(-1, keepdims=True)) * 
           (sw_y - sw_y.mean(-1, keepdims=True))).mean(-1)
    var = sw_x.var(-1) + 1e-8
    return cov / var
```

- W=50, W=100
- **sym-agnostic**：✅（slope 是无量纲比 mid_diff/dvol，跨股票同 unit）
- 来源：Kyle 1985 Econometrica
- 与 spread* 完全独立 channel

### B3. Amihud illiquidity ratio

```python
amihud_W = pd.Series(np.abs(delta_log_mid) / (np.abs(amount_delta) + 1e-8)).rolling(W).mean()
```

- W ∈ {30, 50, 100}
- **sym-agnostic**：✅（return / dollar volume，纯无量纲ratio）
- 来源：Amihud 2002

### B4. Microstructure invariance bet size proxy (Kyle-Obizhaeva)

```python
# W = trading activity = amount_delta · sigma_mid_per_window
# bet size proxy = amount_delta_t / W^(1/3)
sigma_W = pd.Series(np.diff(np.log(midprice + 1))).rolling(W).std() + 1e-8
W_activity = (np.abs(amount_delta) * sigma_W) ** (1/3) + 1e-8  # invariance scaling
inv_bet_size = amount_delta / W_activity  # 跨股票同分布的 "bet size in business time"
```

- **理论保证 sym-agnostic**（这是 Kyle-Obizhaeva 2016 的核心 prediction）
- 来源：Kyle-Obizhaeva 2016 Econometrica；Andrew Lo 同思路
- 实验上从未试过，**最可能直接修 fold 2**

### B5. Probability of informed trading (PIN) — short-version

> ⚠️ 标准 PIN 需要 daily 数据 + EM 估计，不适合 100-tick stateless setup。**改造**：

```python
# 用 100-tick 内 buy/sell intensity 不平衡度做近似
buy_intst = pd.Series(mb_intst + lb_intst).rolling(W).sum()
sell_intst = pd.Series(ma_intst + la_intst).rolling(W).sum()
pin_proxy = np.abs(buy_intst - sell_intst) / (buy_intst + sell_intst + 1e-8)
```

- 高 PIN_proxy = 单方向 dominate，可能有 informed flow
- **sym-agnostic**：✅
- 来源：Easley-Lopez de Prado-O'Hara 2012

### B6. VPIN (BVC version)

```python
# bar-level: 把 100 ticks 切成 W=10 bar，每 bar 5 ticks
# Buy_t = V_t · Φ(ΔP_t / σ_ΔP)
# VPIN = mean( |sum_Buy - sum_Sell| / VBS )
```

- 公式细节见 r20_factor_library.md §D1
- **sym-agnostic**：✅（输出是 toxicity 概率 ∈ [0, 1]）
- 来源：Easley-Lopez de Prado-O'Hara 2012 RFS
- 与 B5 互补：B5 是 raw imbalance，VPIN 是 volume-bucketed normalized

### B7. Lambda-spread invariance (合成)

```python
# Kyle λ × spread = 价格 elasticity 系数
lambda_x_spread = kyle_lambda_W * spread1
```

- 跨股票应近似 const（高 λ + 窄 spread 产生 invariant impact cost）
- **sym-agnostic**：✅
- 来源：Hasbrouck 1991 + Kyle-Obizhaeva

### B8. Hasbrouck "trade impact"

```python
# r_t / sqrt(volume·price)
hasbrouck_impact = delta_log_mid / np.sqrt(np.abs(amount_delta) + 1e-8)
hasbrouck_impact_W = pd.Series(hasbrouck_impact).rolling(W).mean()
```

- W ∈ {30, 50}
- **sym-agnostic**：✅（无量纲）
- 来源：Hasbrouck 1991 JOF

### B9. Roll-based variance ratio

```python
# Roll spread² / σ²(Δmid_1) — 衡量 bid-ask bounce 占比
roll_share = (2*np.sqrt(np.maximum(-cov, 0)))**2 / (delta_mid_var + 1e-8)
```

- 范围 [0, 1]，无量纲
- 高值 → 噪声主导（sym 3 在短 horizon），低值 → trending（sym 2 在长 horizon）
- **sym-agnostic**：✅
- 来源：Roll 1984 + variance ratio test

---

## C. 高阶矩 / 形状特征

> **核心思想**：mean/std 已有；moment of order ≥ 3 + ACF + range 等是分布形状描述符，**无量纲**或**自带归一化**。

### C1. Realized Skewness within window

```python
def rolling_skew(r: np.ndarray, W: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(r, W)
    mu = sw.mean(-1, keepdims=True)
    s = sw.std(-1) + 1e-8
    skew = ((sw - mu)**3).mean(-1) / s**3
    return skew  # already pad to length T
```

- 输入 r = log return per tick；W ∈ {20, 50, 100}
- **sym-agnostic**：✅（skewness 是 third moment / σ³，无量纲）
- 来源：Amaya-Christoffersen-Jacobs-Vasquez 2015 JFE — 证明 RSkew 是 short-term return predictor

### C2. Realized Kurtosis within window

```python
kurt_W = ((sw - mu)**4).mean(-1) / s**4 - 3
```

- W ∈ {20, 50, 100}
- 高 kurtosis = 厚尾（高频 jump pre-event）
- **sym-agnostic**：✅

### C3. ACF coefficients (multi-lag, within window)

```python
def rolling_acf(r: np.ndarray, lag: int, W: int) -> np.ndarray:
    # corr(r_t, r_{t-lag}) within trailing W
    sw_y = np.lib.stride_tricks.sliding_window_view(r[lag:], W)
    sw_x = np.lib.stride_tricks.sliding_window_view(r[:-lag], W)
    cov = ((sw_x - sw_x.mean(-1, keepdims=True)) * 
           (sw_y - sw_y.mean(-1, keepdims=True))).mean(-1)
    return cov / (sw_x.std(-1) * sw_y.std(-1) + 1e-8)
```

- lag ∈ {1, 5, 10}, W=50
- ACF1 < 0 → bid-ask bounce 主导；ACF1 > 0 → momentum
- ACF5/ACF10 → mid-range structure
- **sym-agnostic**：✅
- 来源：Roll 1984 (ACF1 是 spread estimator 输入)

### C4. Range-based vol (Parkinson)

```python
park_W = np.sqrt(np.log(high/low)**2 / (4*np.log(2)))  # per-tick
park_W_avg = park_W.rolling(W).mean()  # smoothed
```

- W ∈ {20, 50}
- **sym-agnostic**：✅（log(H/L)² 是 ratio）

### C5. Garman-Klass volatility

```python
gk_W = sqrt( (1/W) * sum( 0.5*log(H/L)**2 - (2*log2-1)*log(C/O)**2 ) )
```

- W ∈ {20, 50}
- 比 Parkinson 多用 OC 信息
- **sym-agnostic**：✅

### C6. Rogers-Satchell volatility

```python
rs_W = sqrt( mean(log(H/C)·log(H/O) + log(L/C)·log(L/O), W) )
```

- 对 drift 鲁棒（drift 不影响 RS）
- **sym-agnostic**：✅

### C7. Bipower variation (jump-robust RV)

```python
bv_W = (np.pi/2) * pd.Series(np.abs(r) * np.abs(r.shift(1))).rolling(W).sum()
```

- W ∈ {20, 50, 100}
- **sym-agnostic**：✅（与 RV 同量纲，但跳跃-robust）
- 来源：Barndorff-Nielsen-Shephard 2004 JFE

### C8. Jump indicator (BNS test)

```python
rv_W = (r**2).rolling(W).sum()
J_W = rv_W - bv_W  # jump variation component
J_share = J_W / (rv_W + 1e-8)  # 无量纲 [0, 1]
```

- **sym-agnostic**：✅
- 高 J_share = 最近有跳跃
- 来源：BNS 2006 JFE

### C9. Realized semi-variance (signed vol asymmetry)

```python
rv_pos_W = pd.Series(r**2 * (r > 0)).rolling(W).sum()
rv_neg_W = pd.Series(r**2 * (r < 0)).rolling(W).sum()
rvs_signed_W = (rv_pos_W - rv_neg_W) / (rv_pos_W + rv_neg_W + 1e-8)  # 无量纲 [-1, 1]
```

- W ∈ {20, 50, 100}
- 无量纲 directional vol
- 来源：Barndorff-Nielsen-Kinnebrock-Shephard 2010
- **sym-agnostic**：✅
- ⭐⭐⭐ **极强期望，T3 没做，与 RV 完全互补**

### C10. Hurst exponent estimator (rescaled range)

```python
def hurst_RS(r: np.ndarray, W: int=100) -> float:
    # 复杂度高，建议简化为 lag-2 / lag-1 var ratio
    sw = np.lib.stride_tricks.sliding_window_view(r, W)
    var1 = sw.var(-1)
    sw2 = (r[:-1] + r[1:]).reshape(...)  # 2-tick aggregated
    var2 = sw2.var(-1)
    h = 0.5 * np.log(var2 / (2*var1)) / np.log(2)  # 简化估计
    return h  # H≈0.5 random walk; H>0.5 trending; H<0.5 mean-reverting
```

- **sym-agnostic**：✅
- 计算稍贵，可能 marginal

---

## D. 时间结构（sym-agnostic time embedding）

> **核心约束**：`date` 评测置 0 → 不能用 date。**`time` 字段保留**（"HH:MM:SS"）→ 可推 intraday seasonality。

### D1. Intraday seasonality encoded as cyclic features

```python
# time -> minutes_since_session_start (0..99 for am, 0..99 for pm)
# T3 已有 time_minutes_since_session_start, time_session_progress, time_is_pm
# 新增：cyclic encoding
sin_progress = np.sin(2*np.pi * time_session_progress)
cos_progress = np.cos(2*np.pi * time_session_progress)
```

- 让 LightGBM/NN 不会 split 不连续
- **sym-agnostic**：✅

### D2. Time-since-last-large-move

```python
abs_r = np.abs(r)
threshold = abs_r.rolling(100).quantile(0.9)  # adaptive top 10%
event_at_t = abs_r > threshold
# tsince = number of ticks since last True; capped at W
tsince = ...  # custom rolling
```

- W=100
- **sym-agnostic**：✅（threshold 是 within-window quantile）
- 实现需 numba 或 numpy stride tricks

### D3. Distance-to-session-boundary

```python
dist_to_close = 99 - time_minutes_since_session_start  # T3 已隐含
dist_to_open = time_minutes_since_session_start
log_dist_to_close = np.log1p(dist_to_close)
```

- 提交时改 100-tick 内的 minute（time field 留存）
- **sym-agnostic**：✅
- T3 部分已实现

### D4. Position-of-current-tick within window

```python
# 在每个 100-tick window 内，当下 tick 是第几个？(对所有样本=99)
# 但**用 mid 在 window 中的 rank**：mid 在过去 100-tick 中的位置
# 这就是 A2（quantile rank of mid）。
```

合并到 A2，不重复列出。

### D5. Tick-since-last-event (per event type)

```python
# 对 mb_intst / ma_intst / cb_intst / ca_intst 各算 time-since-last-nonzero
# 6 个特征
```

- **sym-agnostic**：✅
- 信号意义：流量到达点的时间 sparsity

### D6. Quote-update intensity (multi-W)

```python
quote_changed = (bid1.shift(1) != bid1) | (ask1.shift(1) != ask1)
update_rate_W = quote_changed.rolling(W).mean()
```

- W ∈ {10, 50}
- **sym-agnostic**：✅
- 来源：Hasbrouck 高频 activity proxy
- 与 schema 已有 *_intst 互补

### D7. Mid-price-change rate

```python
mid_changed = (midprice.diff() != 0).astype(int)
mid_change_rate_W = mid_changed.rolling(W).mean()
```

- **sym-agnostic**：✅

---

## E. Order Flow Imbalance 高阶版

> **核心**：T3 已有 30 维 MLOFI（10 levels × 3 windows）。检查是否还有 untapped 角度。

### E1. Multi-level OFI sum / decay weighted

```python
ofi_total_W = sum_{k=1..10} mlofi_W_lvl_k
ofi_decayed_W = sum_{k=1..10} (0.7**(k-1)) * mlofi_W_lvl_k  # near levels weighted more
ofi_far_W = sum_{k=8..10} mlofi_W_lvl_k
ofi_near_W = sum_{k=1..3} mlofi_W_lvl_k
ofi_near_far_diff = ofi_near_W - ofi_far_W
```

- 5 个新特征 × 3 windows = 15 维（或精简到 5 维 W=20）
- **sym-agnostic**：✅
- T3 30 维 + 这 15 维互补

### E2. Volume-weighted OFI per level

```python
# 现有 MLOFI 是 size-summed; 加 dollar-weighted 版
mlofi_dollar_k_W = sum(e_k(t) · price_k_t, W)
```

- 但其量纲是 dollar，需归一化 → `mlofi_dollar_k / mean_amount_W`
- **sym-agnostic**：✅（归一化后）
- 与 T3 MLOFI 互补

### E3. EWMA-OFI multi-α

```python
ofi_ew_alpha_lvl_k = pd.Series(e_k_t).ewm(alpha=α).mean()
```

- α ∈ {0.05, 0.1, 0.3, 0.5}
- T3 EWMA 用在 *_intst 但**没用在 OFI**
- **sym-agnostic**：✅
- 6 columns × 4α = 24 维
- 来源：r20 §D4

### E4. GOFI (generalized OFI with passive size adjustments)

```python
e_k_GOFI = MLOFI_classic + I(b_k_t == b_k_{t-1}) * (bs_k_t - bs_k_{t-1})
                          - I(a_k_t == a_k_{t-1}) * (as_k_t - as_k_{t-1})
```

- 即"价格不变时 size 也算入"
- 来源：Cao-Hansch-Wang 2008
- **sym-agnostic**：✅
- 与 T3 MLOFI 信号源不同

### E5. Trade flow imbalance (TFI = signed volume)

```python
sign_t = np.sign(midprice.diff())  # tick rule
tfi_W = (sign_t * np.abs(amount_delta)).rolling(W).sum() / np.abs(amount_delta).rolling(W).sum()
```

- W ∈ {20, 50}
- **sym-agnostic**：✅（ratio 形式无量纲）
- 来源：Lee-Ready 1991

### E6. Aggressor share

```python
aggressor_share_W = ((mb_intst + ma_intst).rolling(W).sum() / 
                     (mb_intst + ma_intst + lb_intst + la_intst).rolling(W).sum())
```

- **sym-agnostic**：✅
- W ∈ {20, 50, 100}

### E7. Cancellation pressure (per side)

```python
cancel_buy_share_W = cb_intst.rolling(W).sum() / (lb_intst + mb_intst + cb_intst).rolling(W).sum()
cancel_sell_share_W = ca_intst.rolling(W).sum() / (la_intst + ma_intst + ca_intst).rolling(W).sum()
cancel_imb = cancel_buy_share_W - cancel_sell_share_W
```

- **sym-agnostic**：✅
- T9 F4 的扩展（T9 没做 W 滚动，只用 last-tick）

### E8. ARIMA-OFI residual (surprise OFI)

```python
# 拟合 OFI_t = ρ*OFI_{t-1} + ε_t over W=50
# 提取 ε_t 作为"surprise OFI"
```

- 实现复杂；可改用：`ofi_t - ofi_t.rolling(50).mean()`
- **sym-agnostic**：✅
- 与 raw OFI 互补

---

## F. Cross-tick / cross-window 比例（无量纲）

> **核心**：用 ratio 抹掉 sym scale。比 A 系列更针对性。

### F1. Vol ratio across scales

```python
rv_5_over_60 = rv_5 / (rv_60 + 1e-8)
rv_20_over_50 = rv_20 / (rv_50 + 1e-8)
rv_50_over_100 = rv_50 / (rv_100 + 1e-8)
```

- 3 维
- 高 rv_short/long → 短期波动 spike (event)
- **sym-agnostic**：✅
- T3 有 rv_5/10/20/50 但没显式 ratio

### F2. Volume burst ratio

```python
vol_burst_W = volume_delta_t / (volume_delta.rolling(W).mean() + 1e-8)
amt_burst_W = amount_delta_t / (amount_delta.rolling(W).mean() + 1e-8)
```

- W ∈ {20, 50, 100}
- **sym-agnostic**：✅
- 来源：r20 §G7

### F3. Spread ratio (multi-W)

```python
spread_ratio_5 = spread1_t / spread1.rolling(5).mean()
spread_ratio_50 = spread1_t / spread1.rolling(50).mean()
spread_ratio_5_50 = spread_ratio_5 / spread_ratio_50  # tier 2
```

- **sym-agnostic**：✅
- W ∈ {5, 20, 50}

### F4. Depth ratio

```python
depth_concentration_top3 = (bsize1+bsize2+bsize3) / (totalbsize + 1e-8)
depth_concentration_ask3 = (asize1+asize2+asize3) / (totalasize + 1e-8)
```

- **sym-agnostic**：✅
- 来源：r20 §E4

### F5. Trade size distribution shape (within window)

```python
trade_size_W = np.abs(amount_delta).rolling(W)
median_size_W = trade_size_W.quantile(0.5)
mean_size_W = trade_size_W.mean()
trade_size_skew = (mean_size_W - median_size_W) / (mean_size_W + 1e-8)  # 无量纲
trade_size_q90_q50 = trade_size_W.quantile(0.9) / median_size_W  # 无量纲 fat-tail
```

- **sym-agnostic**：✅

### F6. Multi-level depth shape (无量纲)

```python
# 分布 shape，不是绝对量
bid_size_skew_levels = skew([bsize1, bsize2, ..., bsize10])
ask_size_skew_levels = skew([asize1, asize2, ..., asize10])
bid_size_kurt_levels = kurt([bsize1..bsize10])
```

- 每个 tick 跨 10 levels 的 distribution shape
- **sym-agnostic**：✅
- 来源：r20 §E14（之前 priority low，但放在 sym-invariance 视角下重要）

### F7. Spread vs depth ratio

```python
spread_per_depth = spread1 / (totalbsize + totalasize + 1e-8)
```

- **sym-agnostic**：（量纲 spread/size，跨股票仍可能 scale 不同）
- 应配 window normalize → A1(spread_per_depth)
- 优先级：⭐ medium

### F8. Multi-level mid prices ratio

```python
midprice_2_over_1 = midprice2 / midprice1  # ≈ 1, deviation 是 cross-level info
midprice_5_over_1 = midprice5 / midprice1
midprice_10_over_1 = midprice10 / midprice1
```

- **sym-agnostic**：✅（所有股票这些 ratio 都接近 1）
- 与 raw midprice_2..10 互补

### F9. Imbalance momentum (HYD)

```python
imb_mom_W = imbalance_t - imbalance.shift(W)
```

- W ∈ {1, 5, 20}
- **sym-agnostic**：✅（imbalance 已是 ratio）

### F10. WMP momentum (HYD)

```python
wmp_mom_W = wmp_lvl1 / wmp_lvl1.shift(W) - 1
```

- W ∈ {3, 6, 10}
- **sym-agnostic**：✅

---

## G. Encoding 创新

> **核心**：把 raw 列变成新表达，让 LightGBM 更易消化。

### G1. Quantile-bucket encoding (within-window 1-99 percentile)

```python
def window_q_bucket(x: np.ndarray, W: int=100, n_buckets: int=10) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    qs = np.linspace(0, 1, n_buckets+1)[1:-1]
    bounds = np.quantile(sw, qs, axis=-1).T  # (T-W+1, n_buckets-1)
    bucket = (x[W-1:][:, None] > bounds).sum(axis=-1)  # int 0..n_buckets-1
    return bucket  # 离散化的 rank
```

- 把 z-score 离散化为整数 0..9
- LightGBM 对小整数 split 极快
- **sym-agnostic**：✅
- 与 A2 类似但更轻量
- 优先级：⭐ medium（A2 已是 continuous rank）

### G2. Causal target encoding（用过去样本的 label）

> ⚠️ **不能用**！违反 CRITICAL_CONSTRAINTS：Predictor 单次调用只看 100×D 切片，无 label 历史。
> （即使训练时可用，inference 时 label 不可见，会导致 train-infer mismatch）。
> **跳过**。

### G3. Frequency encoding

```python
# 对 binary indicator 列 *_ind / cancel events / market-aggressor events
# 算 100-tick window 内 occurrence frequency
freq_mb_ind = mb_ind.rolling(100).mean()  # 0-1
```

- T3 部分 EWMA 已有；但 hard window mean 是另一信号
- **sym-agnostic**：✅
- 与 EWMA-intst 互补

### G4. Feature interaction encoding (pairwise / triplet)

```python
# 已存在 imbalance · spread = "urgency" — 来自 HYD
urgency = imbalance * spread1
spread_x_voldelta = spread1 * volume_delta
```

- 已在 r20 §H3
- **sym-agnostic**：（取决于 base feature；建议先 normalize 再 multiply）
- 优先级：⭐ medium

### G5. Differential encoding

```python
# 把 raw column 替换成 "raw - rolling_mean"（残差）
spread1_resid = spread1 - spread1.rolling(50).mean()
```

- 是 A1 z-score 的简化版（去除分母）
- 与 A1 互补：A1 给 σ-units，G5 给原 unit 的残差
- **sym-agnostic**：✅

---

## H. 残差 / detrend

> **核心**：对每个 raw 列减去局部 trend，残差是 high-pass filtered 信号。

### H1. Within-window linear regression detrend

```python
def detrend_linear(x: np.ndarray, W: int) -> np.ndarray:
    # 对 [0, 1, ..., W-1] 做 OLS, 取残差 at last position
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    t_idx = np.arange(W)
    t_centered = t_idx - t_idx.mean()
    cov = ((sw - sw.mean(-1, keepdims=True)) * t_centered).mean(-1)
    var = (t_centered**2).mean()
    slope = cov / var
    pred_last = sw.mean(-1) + slope * (W - 1 - t_idx.mean())
    resid = x[W-1:] - pred_last
    return resid  # 残差 at last tick
```

- W ∈ {20, 50}
- **sym-agnostic**：✅
- 应用：detrend midprice，取 residual = "deviation from local linear trend"

### H2. Linear regression slope (Qlib BETA)

```python
slope_W = (cov(close, t_idx) / var(t_idx)).rolling(W)
slope_norm_W = slope_W / close_t  # 无量纲
```

- W ∈ {5, 10, 20, 50}
- **sym-agnostic**：✅（slope/level 无量纲）
- 来源：Qlib BETA

### H3. Linear regression R² (Qlib RSQR)

```python
r2_W = corr(close, t_idx, W) ** 2
```

- W ∈ {5, 10, 20, 50}
- 高 R² → trending；低 → choppy
- **sym-agnostic**：✅

### H4. EWMA-residual (high-pass)

```python
ewma_long = pd.Series(midprice).ewm(alpha=0.05).mean()
midprice_innovation = midprice - ewma_long
midprice_innovation_norm = midprice_innovation / ewma_long  # 无量纲
```

- **sym-agnostic**：✅（normalized 后）
- 高频 deviation signal

### H5. ARIMA-style residual (one-step-ahead)

```python
# AR(1) prediction:
ar1_pred = c + phi * x.shift(1)  # phi from rolling estimation
ar1_resid = x - ar1_pred
```

- 复杂度 medium（需 rolling regression）
- **sym-agnostic**：✅
- 优先级：⭐ low（H1 / H4 已涵盖大部分价值）

### H6. Innovation series (Kalman-like, simple)

```python
# 简化 Kalman: 把 raw signal 与 EWMA prediction 做差
innov_t = x_t - ewma_t.shift(1)  # 用上一步预测
```

- **sym-agnostic**：✅
- 与 H4 类似但 1-step lag

---

## I. 跨 sym 一致性诊断（不算特征，但应该做）

> 加完所有特征后做的健康检查：

```python
# 对每个特征 f，检查跨 sym 分布相似度
ks_stat, p_value = ks_2samp(f[sym==0], f[sym==2])
# p_value > 0.05 → 分布不可区分 → 特征是真正 sym-agnostic
```

理想情况：top-15 特征都通过 KS test（p > 0.05）。

---

## J. 总结：所有候选特征数量

| 方向 | 候选数 | 已在 226-d? | 高优先级 |
|---|---|---|---|
| A. Adaptive normalization | ~120 (40 cols × 3 W) | 仅 D1 试过 308d，且只 W=100 | A1, A2, A4, A9 |
| B. 微结构不变量 | 9 | 无 | B1, B2, B3, B4 |
| C. 高阶矩 / 形状 | 10 | 部分 (RV in T3) | C1, C7, C9 |
| D. 时间结构 | 7 | 部分 (T3 time encoding) | D1, D6 |
| E. OFI 高阶 | 8 | 部分 (T3 MLOFI 30维) | E3, E4, E6, E7 |
| F. Cross-tick ratio | 10 | 部分 (T3 wmp_balance) | F1, F2, F4, F5 |
| G. Encoding | 5 | 无 | G1, G3 |
| H. 残差 / detrend | 6 | 无 | H1, H2, H4 |
| **合计** | **~175** | **30/175 ≈ 17%** | **15** |

**关键 takeaway**：
1. 226-d 中 **scale-coupled** raw features（spread*, *_mean, bid/ask price 10 档等）是 fold 2 OOD 的主因。
2. 解药是 **A 系列（local normalization） + B 系列（economic invariants） + F 系列（无量纲 ratios）**——这三类一起占 ~85% 的候选特征。
3. **C9 (signed RV) 和 B4 (Kyle-Obizhaeva invariance)** 是可能的"杀手锏"，T3/T7/T9 都没试过。
4. **D1 (Window-zscore 308d)** 早期 LOSO h_60 +13.91 的成功暗示 A 系列扩展到 multi-W 应有效，但需配合 aug_a。

---

## K. 与现有 226-d 重复关系（精确表）

| 现有 226-d 项 | 与 R34 哪些重复 | 仍要加吗？ |
|---|---|---|
| 154 raw last-tick (spread*, *_mean, bid/ask) | A 系列对它们做 z-score → 互补 | A 系列是替换 / 补充，加 |
| 30 MLOFI (3 windows × 10 levels) | E1/E2/E3 互补（不重复） | E 系列加 |
| 11 WMP | E2 互补，F8/F10 互补 | 加 |
| 4 RV (W=5/10/20/50) | C7 BV 互补；C9 signed RV 互补；F1 ratio 互补 | C9, F1 加 |
| 24 EWMA intensities | E3/E4 EWMA-OFI 是新方向（T3 只 EWMA on intensities） | E3, E4 加 |
| 3 time | D1 sin/cos cyclic 互补 | D1 加 |

→ R34 提议的所有 top-15 与 226-d **不重复**。

---

## L. 实施优先级建议

### 立即 (一个 worker × 2-4h)
1. **A1 + A9** (Window-zscore multi-W + dual-z) 应用到 30-50 个 scale-coupled raw 列 → ~120 维
2. **C9 (signed RV)** 加 4 W = 4 维
3. **B4 (Kyle-Obizhaeva invariance bet size)** = 1 维

预期：226 → ~350 维；LOSO h_60 测试单 fold（fold 2）看 acc 是否回升 > 0.5。

### 一周内 (2-3 worker)
4. **B1, B2, B3** (Roll, Kyle's λ, Amihud) = ~9 维
5. **E3, E4** (EWMA-OFI, GOFI) = ~30 维
6. **F1, F2** (vol burst, vol ratio) = ~6 维
7. **H1, H4** (detrend / EWMA-residual) = ~10 维

### 之后
8. ablation 砍冗余；保留贡献 > 1% gain 的；约目标 280-300 维稳态。
9. 全部加上 aug_a (per-feature [0.85, 1.15]，已知 work) 训。

---

## 参考

- Roll 1984, JOF
- Kyle 1985, Econometrica
- Amihud 2002, J. Financial Markets
- Hasbrouck 1991, JOF
- Easley-Lopez de Prado-O'Hara 2012, RFS (VPIN)
- Cao-Hansch-Wang 2008 (GOFI)
- **Kyle-Obizhaeva 2016, Econometrica** "Market Microstructure Invariance: Empirical Hypotheses" — γ ∝ W^(2/3), Q ∝ W^(1/3)
- Barndorff-Nielsen-Shephard 2004 (BV) / 2010 (signed RV)
- Amaya-Christoffersen-Jacobs-Vasquez 2015, JFE (RSkew)
- Lee-Ready 1991, JOF (tick rule)
- HYD 1st place Optiver Trading at Close (pairwise/triplet imb)
- T7 build_features.py — D1 Scheme reference
- T9 build_features.py — F1-F4 reference
- r30_data_characteristics.md — 5 sym microstructure 画像
- r20_factor_library.md §A-H — 完整因子库
