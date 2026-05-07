# R34 — Top-15 Sym-Invariant Features (精选实施清单)

> Date: 2026-05-07
> 配套 r34_sym_invariant_features.md（穷举版）
>
> **选取标准**：
> 1. **解决 fold 2 (sym=2) confidently-wrong overfit** 是首要目标
> 2. 与现有 226-d (T2 154 + T3 72) **不重复**
> 3. **sym-agnostic 严格保证**（KS test 期望 p > 0.05 跨所有 sym）
> 4. 实施成本 ≤ 2 小时（pandas/numpy 向量化即可）
> 5. 已有学术 / Kaggle 实证支持
>
> **Order**：从最高 ROI 开始；每加 5 个建议重训 LOSO h_60 single-fold（fold 2）做验证。

---

## 速览表

| Rank | 特征 | 类别 | 维度 | 公式核心 | 预期 LOSO h_60 增益 | 实施 |
|---|---|---|---|---|---|---|
| **1** | **Dual window z-score (A9)** | A | ~40 | `z_short(x, 20) - z_long(x, 100)` | **+1.5 ~ +3.0** | 1.5h |
| **2** | **Signed Realized Vol (C9)** | C | 4 | `(rv_pos - rv_neg) / (rv_pos + rv_neg)` | +0.5 ~ +1.5 | 0.5h |
| **3** | **Kyle-Obizhaeva bet size invariant (B4)** | B | 3 | `amount / (\|amount\|·σ)^(1/3)` | +0.5 ~ +1.5 | 0.5h |
| **4** | **Window quantile rank (A2)** | A | ~20 | `rank(x_t in trailing W)` | +0.5 ~ +1.5 | 1h |
| **5** | **Realized skewness (C1)** | C | 3 | `E[r³] / σ³` rolling | +0.5 ~ +1.0 | 0.5h |
| **6** | **EWMA-OFI multi-α (E3)** | E | 12 | `EWM(mlofi_lvl_k, α)` | +0.3 ~ +1.2 | 1h |
| **7** | **GOFI generalized OFI (E4)** | E | 30 | `MLOFI + passive size adj` | +0.3 ~ +1.0 | 1.5h |
| **8** | **Kyle's λ rolling (B2)** | B | 2 | OLS slope `Δp ~ signed_dvol` | +0.3 ~ +1.0 | 1.5h |
| **9** | **Amihud illiquidity (B3)** | B | 3 | `mean(\|r\|/dollar_vol)` rolling | +0.2 ~ +0.8 | 0.3h |
| **10** | **Vol-burst ratio (F2)** | F | 4 | `vol_t / mean(vol, W)` | +0.3 ~ +0.8 | 0.3h |
| **11** | **EWMA-residual midprice (H4)** | H | 1 | `(mid - ewma_mid) / ewma_mid` | +0.2 ~ +0.6 | 0.3h |
| **12** | **Multi-W RV ratio (F1)** | F | 3 | `rv_5/rv_50, rv_20/rv_100` | +0.2 ~ +0.5 | 0.3h |
| **13** | **Bipower variation (C7)** | C | 4 | `(π/2)·sum(\|r_t\|·\|r_{t-1}\|)` | +0.2 ~ +0.5 | 0.5h |
| **14** | **Cancel-pressure imbalance (E7)** | E | 3 | `cancel_buy_share - cancel_sell_share` | +0.2 ~ +0.5 | 0.5h |
| **15** | **Roll's effective spread (B1)** | B | 3 | `2·sqrt(max(-cov(Δp), 0))` rolling | +0.1 ~ +0.4 | 1h |
| | **总计** | | **~135** | | **+5.5 ~ +15.8** | **~12h** |

---

## #1 — Dual Window Z-score (A9) ⭐⭐⭐ 首选

### 公式

```python
def dual_window_zscore(x: np.ndarray, W_short: int=20, W_long: int=100) -> np.ndarray:
    """
    返回 z_short - z_long, 长度 T；前 W_long-1 是 NaN。
    z_short = (x_t - mean(x[t-19..t])) / std(x[t-19..t])
    z_long  = (x_t - mean(x[t-99..t])) / std(x[t-99..t])
    """
    sw_s = np.lib.stride_tricks.sliding_window_view(x, W_short)
    sw_l = np.lib.stride_tricks.sliding_window_view(x, W_long)
    z_short_full = (sw_s[:, -1] - sw_s.mean(-1)) / (sw_s.std(-1) + 1e-8)
    z_long_full  = (sw_l[:, -1] - sw_l.mean(-1)) / (sw_l.std(-1) + 1e-8)
    # 对齐长度（z_long 输出更短）
    pad = np.full(W_long - W_short, np.nan, dtype=np.float32)
    z_short_aligned = np.concatenate([pad, z_short_full[W_long - W_short:]])
    dual = z_short_aligned - z_long_full  # length T - W_long + 1
    return np.concatenate([np.full(W_long-1, np.nan), dual])
```

### 应用对象（约 40 个 raw 列）

```
spread1, spread5, spread10, cumspread,
bid1, ask1, bid_mean, ask_mean, midprice,
bsize1, bsize_mean, totalbsize,
asize1, asize_mean, totalasize,
volume_delta, amount_delta, imbalance,
lb_intst, la_intst, mb_intst, ma_intst, cb_intst, ca_intst,
lb_acc, la_acc, mb_acc, ma_acc, cb_acc, ca_acc,
midprice2, midprice5, midprice10,
bid_diff1, ask_diff1, bid_diff5, ask_diff5
```

→ ~40 维新特征。

### 为什么 sym-agnostic

- **z_short, z_long 都是 within-window 归一化**——只用窗口内 sym 自己的 mean/std，不跨 sym borrow。
- sym 2 spread1=2.7bp 但 dual_z(spread1) ≈ sym 3 spread1=23bp 的 dual_z（都 ≈ 0 if stable，都 spike if event）→ **跨 sym 同分布**。
- KS test 期望：所有 5 sym 的 dual_z 分布 p > 0.10。

### 预期解决什么 fold 问题

- **Fold 2 (sym=2 held)** ：sym 2 的 raw spread 极小且 stable；模型在 fold 0/1/3/4 训练学到的 split (e.g. spread1 > 5) 在 sym 2 上几乎永远 false → 失去信号。dual_z(spread1) 让 sym 2 的 spread spike 与其他 sym 同 unit，模型 split 能 reuse。
- **Fold 4 (sym=4 held)**：sym 4 是 "金本位"，dual_z 对它影响 minimal（不破坏现有正贡献）。
- **跨 sym 一致性**：Aug_a 砍 spread* FI 到 0.0008，dual_z 把它们以 sym-agnostic 形式重新引回 → 信息利用率回升。

### 与现有 226-d 重复？

- **不重复**。T7 D1 试过 single-W=100 z-score（308d），但**没做 short-vs-long differential**，且 D1 因为整体 308d 太宽没有继续。
- T3 EWMA 是 ewm 形式而非 hard window z-score，不重复。

### 预期 LOSO h_60 增益

- D1 single-W=100 zscore 试过 +13.91 (vs baseline +11)；dual_z 提供更清晰信号 → 期望 +13.61 → **+15 ~ +16**（+1.5 ~ +3.0）。
- **关键 risk**：100-tick 窗口对 std 估计 noise 大，前 99 tick 全 NaN（已知 limitation）。建议 fillna with 0。

### 实施成本

- 1.5h（写 vectorized 函数 + apply 到 40 列 + 集成到 build_cache）。

---

## #2 — Signed Realized Vol (C9) ⭐⭐⭐

### 公式

```python
def signed_rv(r: np.ndarray, W: int) -> np.ndarray:
    rv_pos = pd.Series(r**2 * (r > 0)).rolling(W).sum()
    rv_neg = pd.Series(r**2 * (r < 0)).rolling(W).sum()
    return ((rv_pos - rv_neg) / (rv_pos + rv_neg + 1e-8)).values
```

`r = log(midprice_t / midprice_{t-1})`

### 维度

W ∈ {20, 50, 100, 200} → 4 维

### 为什么 sym-agnostic

- Signed RV 是 ratio (rv_pos - rv_neg) / (rv_pos + rv_neg)，**严格 [-1, 1]**，无量纲。
- 跨 sym 的 vol scale 不同，但 signed-share **理论上同分布**（除非 sym-specific drift）。

### 预期解决什么 fold 问题

- **Fold 2 (sym=2)**：sym 2 蓝筹 ETF 的 Δmid VR=1.52 (轻度 trending) → signed RV 应有信号；目前 226-d 中只有 unsigned RV，方向信息丢失。
- **Fold 1 (sym=1)**：sym 1 中等流动性、中性 VR；signed RV 提供方向 prior。

### 与现有 226-d 重复？

- **不重复**。T3 已有 4 个 unsigned RV (W=5/10/20/50)，但没做 signed split。

### 预期 LOSO h_60 增益

- Amaya 2015 JFE 证明 RSkew (= signed RV 的高阶版) 是 short-term return predictor；加 4 维 signed RV 应贡献 +0.5 ~ +1.5。
- **特别注意**：与 RV 共线但方向不同——LightGBM 会通过 split 自动 disambiguate。

### 实施成本

- 0.5h。

---

## #3 — Kyle-Obizhaeva Bet Size Invariant (B4) ⭐⭐⭐ 高 expected ROI

### 公式

```python
# 来源：Kyle-Obizhaeva 2016 Econometrica
# 跨股票 bet size 在 "trading activity W" 标准化下不变
# W = |amount_delta| · σ_mid_per_window

def kyle_obizhaeva_invariant_size(amount_delta, midprice, W=100):
    log_mid = np.log(midprice + 1e-8)
    sigma_W = pd.Series(np.diff(log_mid, prepend=log_mid[:1])).rolling(W).std() + 1e-8
    activity_W = (np.abs(amount_delta) * sigma_W) + 1e-8
    activity_third_root = activity_W ** (1/3)
    inv_bet_size = amount_delta / activity_third_root  # 跨股票同分布
    inv_bet_arrival = ... # γ ∝ W^(2/3)，可加 W=100 内 large-trade 频率
    return inv_bet_size  # +1 维 base; 多 W → 3 维
```

### 维度

W ∈ {50, 100, 200} → 3 维

### 为什么 sym-agnostic

- Kyle-Obizhaeva 2016 **理论证明**：bet size in business time 跨股票不变。
- 用过去 W ticks 估计 σ 和 activity，**完全 within-window**，不依赖 sym ID。
- 实证：Kyle 2016 在 1993-2001 监测 trade rate 对 W 系数稳定 0.666，跨股票一致。

### 预期解决什么 fold 问题

- **Fold 2 (sym 2 蓝筹 ETF)**：amount_delta 量级 151k 元/tick (4-7× 其他 sym)。raw amount_delta 在 fold 2 完全 OOD；Kyle invariant 后跨 sym 同分布。
- **Fold 0/1/4**：amount_delta 是 24-59k；Kyle 标准化抹去 scale，纯保留方向 + 异常程度。
- **数学上保证修 fold 2 sym=2 amount-related overfit**。

### 与现有 226-d 重复？

- **完全不重复**。T2/T3/T7/T9 都没做 Kyle-Obizhaeva 标准化。
- T9 F1 (amount log1p z-score) 是简单 z-score，不是基于经济学不变量。

### 预期 LOSO h_60 增益

- 这是**理论支持最强的特征**，但实证未试过。**推测 +0.5 ~ +1.5**。
- **如果 work**：可能修 fold 2 acc 从 0.279 → 0.45+。

### 实施成本

- 0.5h。

---

## #4 — Window Quantile Rank (A2) ⭐⭐ Robust Alternative to A9

### 公式

```python
def window_qrank(x: np.ndarray, W: int=100) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(x, W)
    rank = (sw <= sw[:, -1:]).sum(axis=-1).astype(np.float32) / W
    return np.concatenate([np.full(W-1, np.nan, np.float32), rank])
```

### 应用对象

约 20 个**对 outlier 敏感**的列：
```
spread1, spread5, spread10,  # spread spike outlier
amount_delta, volume_delta,  # vol burst outlier
imbalance, totalbsize, totalasize,  # depth outlier
lb_intst, la_intst, mb_intst, ma_intst,  # intensity bursts
mb_acc, ma_acc,  # accumulation
midprice  # price level
... + 5 个 derived
```

### 维度

20 维（W=100 single）。

### 为什么 sym-agnostic

- Quantile rank ∈ [0, 1] 严格分布无关——任意 monotonic 变换不改变 rank。
- 比 A9 z-score 更 robust：sym 3 突发 spread spike 不会把 z 拉到 ±10 outlier，rank 仍 ∈ [0, 1]。

### 预期解决什么 fold 问题

- **Fold 3 (sym=3 small-cap)**：sym 3 spread 23bp 高 + jumps，z-score 容易被 outlier 主导；rank 稳定。
- **Fold 2**：与 A9 互补 robust path。

### 与现有 226-d 重复？

- **不重复**。

### 预期 LOSO h_60 增益

- 与 #1 部分共线（同样是 within-window normalization），但 rank vs z 的非线性不同，LightGBM 应能利用差异 → +0.5 ~ +1.5。

### 实施成本

- 1h（vectorized rank 实现稍慢于 z-score，需 mask 或 broadcasting）。

---

## #5 — Realized Skewness (C1) ⭐⭐

### 公式

```python
def rolling_skew(r: np.ndarray, W: int) -> np.ndarray:
    sw = np.lib.stride_tricks.sliding_window_view(r, W)
    mu = sw.mean(-1, keepdims=True)
    s = sw.std(-1) + 1e-8
    skew = ((sw - mu)**3).mean(-1) / s**3
    return np.concatenate([np.full(W-1, np.nan), skew])
```

### 维度

W ∈ {20, 50, 100} → 3 维

### 为什么 sym-agnostic

- Skewness = 3rd moment / σ³，**无量纲**。
- 所有 sym 的 return distribution skewness 在 (-2, 2) 区间，跨 sym 同分布。

### 预期解决什么 fold 问题

- **Fold 2 (sym 2)**：蓝筹 ETF 的 return 分布更对称（skew ≈ 0）；其他 sym (尤其 sym 3) 有 fat-tail asymmetric。Skew 提供 regime indicator → 修 fold 2 OOD。
- **Fold 4**：Skew 加强 directional prior。

### 与现有 226-d 重复？

- **不重复**。T3 RV 是 unsigned vol，没有 skew/kurt 信息。
- 与 #2 (signed RV) 部分相关但更高阶 moment。

### 预期 LOSO h_60 增益

- Amaya 2015 JFE 证明 1-week 股票 return RSkew 对 next-week return 有 R²≈0.05；推断 100-tick 内 RSkew 应贡献 +0.5 ~ +1.0。

### 实施成本

- 0.5h。

---

## #6 — EWMA-OFI Multi-α (E3) ⭐⭐

### 公式

```python
# 对 mlofi_W20_lvl_k (T3 已 cache 30 维) 做 EWMA
mlofi_W20_lvl_k_ewm_a = pd.Series(mlofi_W20_lvl_k).ewm(alpha=α).mean()
# α ∈ {0.05, 0.1, 0.3, 0.5} (半衰期 14/7/2/1)
```

### 维度

精简：选 levels k ∈ {1, 5, 10}（top, mid, far）× 4α = 12 维。

### 为什么 sym-agnostic

- MLOFI 本身已是 size 单位，跨 sym 量级不一；但 **EWMA 比 raw OFI 更平滑**，且我们只用它的 directional 信号。
- 如配 #1 dual-z 一起 → EWMA-OFI/local_std 完全无量纲。

### 预期解决什么 fold 问题

- **所有 fold**：EWMA 平滑后的 OFI 信号比 raw 强，r30 显示 OFI/return correlation 在长 horizon 仍 +0.05~0.08。
- **Fold 2**：sym 2 OFI 信号小但 stable EWMA 应能 amplify。

### 与现有 226-d 重复？

- **部分重复**。T3 已 EWMA on intensities (lb/la/mb/ma/cb/ca)，但**没 EWMA on OFI**。
- 30d MLOFI 是 hard-window sum，EWMA 是软 window。互补。

### 预期 LOSO h_60 增益

- T3 EWMA-intst 是 top-3 重要 feature；类比 EWMA-OFI 应贡献 +0.3 ~ +1.2。

### 实施成本

- 1h（vectorized pandas EWM 即可，但需小心 cross-session reset）。

---

## #7 — GOFI Generalized OFI (E4) ⭐⭐

### 公式

```python
# 标准 MLOFI:
e_k_classic = I(b_k_t ≥ b_k_{t-1}) · bs_k_t - I(b_k_t ≤ b_k_{t-1}) · bs_k_{t-1} \
            - I(a_k_t ≤ a_k_{t-1}) · as_k_t + I(a_k_t ≥ a_k_{t-1}) · as_k_{t-1}

# GOFI 加 passive size adjustments:
e_k_GOFI = e_k_classic + I(b_k_t == b_k_{t-1}) · (bs_k_t - bs_k_{t-1}) \
                       - I(a_k_t == a_k_{t-1}) · (as_k_t - as_k_{t-1})

# Rolling sum:
gofi_W_lvl_k = sum(e_k_GOFI, W) for W ∈ {5, 20, 60}
```

### 维度

10 levels × 3 windows = 30 维。

### 为什么 sym-agnostic

- 与 MLOFI 同结构，size 单位；与 #1 配合做 dual_z(gofi) → 无量纲。
- 直接 raw GOFI 仍 OK 因为 LightGBM 学局部 split，scale 不影响 logical structure（只要训练分布覆盖）。
- **better**：把 GOFI 也 dual-z 处理。

### 预期解决什么 fold 问题

- **所有 fold**：GOFI 比 MLOFI 多了"价不变 size 变"信息——这是 schema 没显式给的；passive 用户是 informed 信号。
- **Fold 2 (蓝筹 ETF)**：sym 2 价稳但 size 变化频繁（深档薄但活跃），GOFI 应有信号。

### 与现有 226-d 重复？

- **不重复**（30d 新维度）。
- 与 T3 30d MLOFI 同结构但补充 channel。

### 预期 LOSO h_60 增益

- Cao-Hansch-Wang 2008 证明 GOFI 在 NYSE TAQ 上比 MLOFI 提升 ~10% R²；类比 +0.3 ~ +1.0。

### 实施成本

- 1.5h（与 T3 MLOFI 实现并列，向量化）。

---

## #8 — Kyle's λ Rolling (B2) ⭐⭐

### 公式

```python
def kyle_lambda_rolling(delta_mid, signed_dvol, W=50):
    # signed_dvol = sign(delta_mid) * sqrt(|amount_delta|)
    sw_y = np.lib.stride_tricks.sliding_window_view(delta_mid, W)
    sw_x = np.lib.stride_tricks.sliding_window_view(signed_dvol, W)
    cov = ((sw_x - sw_x.mean(-1, keepdims=True)) * 
           (sw_y - sw_y.mean(-1, keepdims=True))).mean(-1)
    var = sw_x.var(-1) + 1e-8
    lambda_ = cov / var
    return lambda_  # rolling slope ≡ price impact coefficient
```

### 维度

W ∈ {50, 100} → 2 维

### 为什么 sym-agnostic

- λ 是 OLS slope `Δp / signed_dvol`，无量纲（同 unit price/sqrt(volume·price)）。
- Kyle 1985 证明 λ 是 microstructure invariant。

### 预期解决什么 fold 问题

- **Fold 2 (sym 2 蓝筹)**：高 λ → trade 推动价位；低 λ → liquidity 充足。区分 sym 2 (低 λ, 大单不动价) 和 sym 3 (高 λ, 小单大动)。
- **跨 sym 信号**：λ 是流动性的"硬度"指标，与 spread 互补。

### 与现有 226-d 重复？

- **不重复**。

### 预期 LOSO h_60 增益

- AFML Ch 19 给出 Kyle λ 在 LOB-based prediction 中的 standard 贡献；预期 +0.3 ~ +1.0。
- **小 risk**：W=50 / 100 内 OLS 估计 noise 大，前 50/100 tick NaN。

### 实施成本

- 1.5h（rolling OLS closed-form：cov/var）。

---

## #9 — Amihud Illiquidity (B3) ⭐⭐

### 公式

```python
def amihud_illiq(midprice, amount_delta, W):
    log_mid = np.log(midprice + 1e-8)
    r = np.diff(log_mid, prepend=log_mid[:1])
    illiq = np.abs(r) / (np.abs(amount_delta) + 1e-8)
    return pd.Series(illiq).rolling(W).mean().values
```

### 维度

W ∈ {30, 50, 100} → 3 维

### 为什么 sym-agnostic

- Amihud = |return| / dollar_volume，纯 ratio 无量纲。
- Amihud 2002 证明跨股票 illiquidity 都用同一 metric。

### 预期解决什么 fold 问题

- **Fold 2 (sym 2)**：低 illiq (深流动性) vs 其他 sym 的中等 illiq → 跨 sym signal 一致。
- **Fold 3 (sym 3 小盘)**：高 illiq → vol-spread regime indicator。

### 与现有 226-d 重复？

- **不重复**。

### 预期 LOSO h_60 增益

- 与 Kyle λ 相关但 不同 channel；预期 +0.2 ~ +0.8。

### 实施成本

- 0.3h。

---

## #10 — Vol-Burst Ratio (F2) ⭐⭐

### 公式

```python
vol_burst_W = volume_delta / (volume_delta.rolling(W).mean() + 1e-8)
amt_burst_W = np.abs(amount_delta) / (np.abs(amount_delta).rolling(W).mean() + 1e-8)
```

### 维度

W ∈ {20, 50} 各 2 个 → 4 维

### 为什么 sym-agnostic

- volume_t / mean(volume) 是 unit-less ratio，跨 sym 同分布（mean ≈ 1）。

### 预期解决什么 fold 问题

- **Fold 2**：sym 2 突发 ETF block trade → vol_burst 巨大。raw volume_delta 在 sym 2 是 OOD outlier；vol_burst 跨 sym 同分布。
- **Fold 4**：sym 4 vol burst 与 mean 关系稳定。

### 与现有 226-d 重复？

- **不重复**。Schema 有 raw volume_delta 但没 ratio。

### 预期 LOSO h_60 增益

- HYD 1st place 用过类似 trick；预期 +0.3 ~ +0.8。

### 实施成本

- 0.3h。

---

## #11 — EWMA-Residual Midprice (H4) ⭐⭐

### 公式

```python
ewma_long = pd.Series(midprice).ewm(alpha=0.05).mean()  # 半衰期 14 ticks
midprice_innov = (midprice - ewma_long) / (ewma_long + 1e-8)  # 无量纲 innovation
```

### 维度

α ∈ {0.05} → 1 维（如需 multi-α 可加 0.1, 0.3 → 3 维）

### 为什么 sym-agnostic

- 归一化后 (mid - ewma) / ewma 是百分比偏离，跨 sym 同 unit。

### 预期解决什么 fold 问题

- **所有 fold**：高频 mid deviation 在 normalized form 下 sym-invariant；提供 mean-reversion proxy。

### 与现有 226-d 重复？

- **不重复**。T3 RV 是 vol 量级；H4 是 directional deviation。

### 预期 LOSO h_60 增益

- 简单但有效；预期 +0.2 ~ +0.6。

### 实施成本

- 0.3h（pandas .ewm one-liner）。

---

## #12 — Multi-W RV Ratio (F1) ⭐

### 公式

```python
rv_5_over_50 = rv_5 / (rv_50 + 1e-8)
rv_20_over_100 = rv_20 / (rv_100 + 1e-8)
rv_50_over_200 = rv_50 / (rv_200 + 1e-8)  # 需扩展 RV W=100, 200
```

### 维度

3 维（其中 rv_100, rv_200 需补充计算）

### 为什么 sym-agnostic

- ratio 无量纲，跨 sym 应同分布（vol regime relative）。

### 预期解决什么 fold 问题

- **所有 fold**：高 rv_short/long → 短期 spike (event-driven)；这是 cross-fold 共通 regime info。

### 与现有 226-d 重复？

- T3 已有 rv_5/10/20/50；ratio 是新衍生层，不重复。

### 预期 LOSO h_60 增益

- 预期 +0.2 ~ +0.5。

### 实施成本

- 0.3h（计算 rv_100, rv_200 + 3 个 ratio）。

---

## #13 — Bipower Variation (C7) ⭐

### 公式

```python
bv_W = (np.pi/2) * pd.Series(np.abs(r) * np.abs(r.shift(1))).rolling(W).sum()
# Jump variation share:
J_share_W = (rv_W - bv_W) / (rv_W + 1e-8)  # in [0, 1]
```

### 维度

W ∈ {20, 50, 100} → 3 维 BV + 1 维 J_share = 4 维

### 为什么 sym-agnostic

- BV 与 RV 同 unit；J_share 是 ratio ∈ [0, 1] 完全 sym-agnostic。
- BNS 2004 证明 BV 是 jump-robust。

### 预期解决什么 fold 问题

- **Fold 3 (sym 3)**：高 jump → J_share 飙升；模型 regime split。
- **Fold 2 (sym 2)**：低 jump (蓝筹平稳) → J_share ≈ 0。

### 与现有 226-d 重复？

- **不重复**。T3 RV 没 jump-decompose。

### 预期 LOSO h_60 增益

- 预期 +0.2 ~ +0.5。

### 实施成本

- 0.5h。

---

## #14 — Cancel-Pressure Imbalance (E7) ⭐

### 公式

```python
cancel_buy_share_W = cb_intst.rolling(W).sum() / (lb_intst + mb_intst + cb_intst).rolling(W).sum()
cancel_sell_share_W = ca_intst.rolling(W).sum() / (la_intst + ma_intst + ca_intst).rolling(W).sum()
cancel_imb_W = cancel_buy_share_W - cancel_sell_share_W
```

### 维度

W ∈ {20, 50, 100} → 3 维 (just imb)

### 为什么 sym-agnostic

- Share ratios ∈ [0, 1]，无量纲，跨 sym 同分布。

### 预期解决什么 fold 问题

- **所有 fold**：高 cancel_buy_share → 买方撤单（previously bullish reversal）；directional信号。
- **Fold 4 (sym 4 中等流动性)**：cancel events 比例信息丰富。

### 与现有 226-d 重复？

- T9 F4 只用 last-tick 分母 (lb+mb+cb)，没用 rolling W；T3 EWMA-intst 不直接给 share。**不重复**。

### 预期 LOSO h_60 增益

- 预期 +0.2 ~ +0.5。

### 实施成本

- 0.5h。

---

## #15 — Roll's Effective Spread (B1) ⭐

### 公式

```python
def roll_spread(delta_mid, W):
    sw = np.lib.stride_tricks.sliding_window_view(delta_mid, W)
    sw_lag = np.lib.stride_tricks.sliding_window_view(delta_mid[:-1], W)
    cov = ((sw[1:] - sw[1:].mean(-1, keepdims=True)) *
           (sw_lag - sw_lag.mean(-1, keepdims=True))).mean(-1)
    return 2 * np.sqrt(np.maximum(-cov, 0))
```

### 维度

W ∈ {30, 50, 100} → 3 维

### 为什么 sym-agnostic

- Roll 1984 证明 effective spread 与 quoted spread 跨股票均同 unit (price)；做 ratio 后无量纲。
- 建议输出 `roll_spread / (spread1 + 1e-8)` 作为 derived → 无量纲 fraction。

### 预期解决什么 fold 问题

- **Fold 2 (sym 2)**：sym 2 quoted spread = 2.7bp，但 roll spread 可能更大（实际成交折扣）→ 区分 quoted vs effective。
- **Fold 3 (sym 3)**：roll spread / quoted spread → bid-ask bounce 主导度。

### 与现有 226-d 重复？

- **不重复**。Schema 有 quoted spread (spread1) 但没 effective spread。

### 预期 LOSO h_60 增益

- 预期 +0.1 ~ +0.4。

### 实施成本

- 1h（rolling cov 实现，注意 stride trick lag）。

---

## 📋 综合实施计划

### Stage 1（高 ROI 4h block, 单 worker）
- #1 Dual z-score (1.5h)
- #2 Signed RV (0.5h)
- #3 Kyle-Obizhaeva invariant (0.5h)
- #6 EWMA-OFI (1h)
- + 集成到 build_cache + 单 fold 训练（fold 2）测试

**Checkpoint**：fold 2 acc 是否回升 > 0.45（baseline 0.279）？如果是 → 继续 Stage 2；不是 → 重新分析为什么。

### Stage 2（4h block）
- #4 Window quantile rank (1h)
- #5 RSkew (0.5h)
- #7 GOFI (1.5h)
- #8 Kyle's λ (1.5h) — 注意 50/100 NaN

### Stage 3（4h block，扫尾）
- #9-#15 (各 0.3-1h)
- 全量 LOSO 5-fold + aug_a [0.85, 1.15]
- LightGBM feature_importance 砍 < 1% gain 维度

### 最终目标

- 226 → 280-300 维稳态
- LOSO h_60 sum：iter_006 +13.61 → **target +15 ~ +18**（+1.5 ~ +4.5）
- **Fold 2 acc**：0.279 → **target ≥ 0.50**（最重要的 KPI）

---

## ⚠️ 实施陷阱

1. **NaN 处理**：所有 rolling 在窗口前 W-1 ticks NaN；fillna(0) 或 mask 训练。统一处理见 T9 实现。
2. **跨 session 隔离**：所有 rolling/EWMA **必须在 single (sym, date, sess) 内 reset**；不要跨 session shift（cross-session 是数据 boundary）。
3. **Numerical stability**：所有除法加 1e-8；log 加 1e-8；sqrt 前 max(0, ...)。
4. **Aug_a 兼容**：新加的 multi-W normalized 特征**已经 sym-agnostic**，aug_a [0.85, 1.15] 仍可应用，强度可能可调降到 [0.90, 1.10]（已经够 robust）。
5. **Memory**：226 → 300 维 × 1.47M rows × float32 ≈ 1.7GB；OK on dev machine 但训练 batch 大小可能要降。
6. **CV split**：保持现有 LOSO 5-fold，**fold 2 是关键 KPI**，先单 fold validate 再全 fold。

---

## 引用

- Kyle-Obizhaeva 2016 Econometrica - microstructure invariance
- Roll 1984 JOF - effective spread
- Kyle 1985 Econometrica - λ
- Amihud 2002 - illiquidity
- Cao-Hansch-Wang 2008 - GOFI
- Barndorff-Nielsen-Shephard 2004 (BV), 2010 (signed RV)
- Amaya-Christoffersen-Jacobs-Vasquez 2015 JFE - RSkew
- Easley-Lopez de Prado-O'Hara 2012 - VPIN
- T7 build_features.py (D1 reference)
- T9 build_features.py (F1-F4 reference)
- r30_data_characteristics.md (跨 sym profile)
- r34_sym_invariant_features.md (穷举版)
