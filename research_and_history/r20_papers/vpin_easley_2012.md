# Easley, Lopez de Prado, O'Hara 2012 — VPIN (Flow Toxicity)

**Easley, D., M. Lopez de Prado, M. O'Hara. 2012. "Flow Toxicity and Liquidity in a High-Frequency World."** *Review of Financial Studies* 25(5): 1457-1493.

## Core idea

把 PIN（probability of informed trading）从 daily 频率推广到 HFT。**关键创新**：用 **volume-time** 替代 clock-time + **bulk volume classification (BVC)** 替代 Lee-Ready 逐笔分类。

## VPIN 计算（5 步）

### Step 1: Time bars
原始 tick-by-tick 数据 → 1-min time bar，每 bar 内：
- `V_τ` = bar 内总 volume
- `ΔP_τ` = bar 内 last - first close

### Step 2: 估计 σ(ΔP)
用全样本（或 rolling 长 window）的 std of ΔP_τ across all time bars。

### Step 3: BVC（bulk volume classification）

不分类单笔交易，而是把 bar 内 volume 按概率分成 buy / sell：

$$
V_\tau^{Buy} = V_\tau \cdot Z\left( \frac{\Delta P_\tau}{\sigma_{\Delta P}} \right)
$$

$$
V_\tau^{Sell} = V_\tau \cdot \left[ 1 - Z\left( \frac{\Delta P_\tau}{\sigma_{\Delta P}} \right) \right]
$$

`Z(·)` = standard normal CDF。直觉：bar 内价格涨多 → buy 比例高（因为 informed buyers 推价）。

### Step 4: Volume buckets
把 time bar 按 cumulative volume 分到 **fixed-size buckets**（每个 bucket 体积 = VBS）。当 bar 不刚好填满，溢出体积给下一个 bucket。
- VBS = 平均日 volume / 50（典型 daily VPIN with 50 buckets）
- 或 VBS = ADV / 5（弱 toxicity sense）

### Step 5: VPIN
对最近 n 个 buckets：

$$
\text{VPIN} = \frac{ \sum_{\tau=1}^{n} | V_\tau^{Sell} - V_\tau^{Buy} | }{ n \cdot \text{VBS} }
$$

VPIN 范围 [0, 1]，约 0.2-0.5；高 → toxicity 高 → market makers 容易亏损。

## 实战参数（paper Table 6, TEF stock 2009）

| Spec | mean | median | std |
|---|---|---|---|
| VPIN 1-50-50 (1-min bar, 50 buckets, n=50) | 0.227 | 0.221 | 0.049 |
| VPIN 5-50-50 | 0.397 | 0.396 | 0.041 |

## 在我们 100-tick setup 的简化

100-tick = ~5 min。原 paper 50-bucket 不可用。我们的 short-VPIN：
- bar = 5 ticks → 20 bars per window
- bucket size VBS = 20 ticks 的 mean(volume_delta) × 4 bars
- 每 100 ticks 内填 ~5 buckets
- n = 5 (sample length = bucket count，**daily VPIN 近似**)

注意：这种 short-VPIN 与 paper 50-bucket 版本性质不同——paper 50-bucket 有 dampening；short-version 噪声大但能 capture local toxicity 突变。

## 实施代码（Python，per-window）

```python
def short_vpin(close, volume, W=100, bar=5):
    # close[-W:], volume[-W:]
    n_bars = W // bar
    bar_close = close[-W:].reshape(n_bars, bar)[:, -1]   # last close per bar
    bar_first = close[-W:].reshape(n_bars, bar)[:, 0]
    bar_vol   = volume[-W:].reshape(n_bars, bar).sum(1)
    
    dP = bar_close - bar_first
    sigma_dP = dP.std() + 1e-9
    z = stats.norm.cdf(dP / sigma_dP)
    
    buy  = bar_vol * z
    sell = bar_vol * (1 - z)
    
    # bucket: simple equal-bar version (not equal-vol bucket)
    vbs = bar_vol.sum() / 5
    cum_v = bar_vol.cumsum()
    bucket_id = (cum_v / vbs).astype(int)
    
    vpin_per_bucket = []
    for b in range(bucket_id.max() + 1):
        mask = bucket_id == b
        if mask.sum() < 1: continue
        vpin_per_bucket.append(abs(buy[mask].sum() - sell[mask].sum()) / vbs)
    return np.mean(vpin_per_bucket)
```

## 重要结论 (paper §6)

1. VPIN 在 Flash Crash 前 1 hour 高出 baseline 2 std → flash crash early warning
2. VPIN 50-bucket 与 PIN ρ=0.93 (Spanish stocks) — VPIN ≈ PIN proxy
3. **bucket size 比 sample length 重要得多**：少 bucket 更接近 PIN 概念；多 bucket 更接近 short-term toxicity

## 我们的优先级

⭐⭐ high — **作为 R10/R13 Bayes EV gating 的 confidence 输入**：
- VPIN 高 → 高 toxicity → 模型预测信心降低 → 提高 trade threshold T
- VPIN 低 → 平静市场 → 降低 T，多接 trade

但 standalone feature 增益可能不大（与 OFI 重叠），建议**作为 calibration / position sizing 的一个 conditioning variable**。

## 限制与争议

- **Andersen-Bondarenko 2011** 提出 VPIN 与 short-run vol 关系不强，质疑预测力
- **Easley et al. 2012d** 反驳：bucket size 选错才会出现这个问题
- 学术界尚未形成共识；**我们应该 empirically test**，看在 LOSO PnL 上是否有用

预期增益：standalone +0.3 ~ +0.8；作为 R13 gating 输入 +1 ~ +3。
