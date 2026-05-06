# Corwin-Schultz 2012 — High-Low Spread Estimator

**Corwin, S. A., and P. H. Schultz. 2012. "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices."** *Journal of Finance* 67(2): 719-760.

## Core idea

只用 daily high (H) 和 low (L)，不需要 tick 数据，估计 bid-ask spread + true volatility。

**两个直觉**：
1. 高价大概率是 ask 一侧成交（buyer 推价），低价大概率是 bid 一侧 → spread 拉宽 high-low ratio
2. 单日 vs 双日 high-low ratio 对比 → 能分离出 spread vs volatility 的贡献（vol 与 时间成比例，spread 不与）

## Formulas（pre-implementable）

$$
\gamma = \left( \ln \frac{H_{t,t+1}^0}{L_{t,t+1}^0} \right)^2
$$

$$
\beta = \left( \ln \frac{H_t^0}{L_t^0} \right)^2 + \left( \ln \frac{H_{t+1}^0}{L_{t+1}^0} \right)^2
$$

$$
\alpha = \frac{\sqrt{2 \beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}
$$

$$
\hat{S}_{HL} = \frac{2(e^\alpha - 1)}{1 + e^\alpha}
$$

其中：
- `H_t^0` = day t 的 high；`L_t^0` = day t 的 low
- `H_{t,t+1}^0 = max(H_t^0, H_{t+1}^0)`；`L_{t,t+1}^0 = min(L_t^0, L_{t+1}^0)`

## Becker-Parkinson Volatility（同方法的副产物）

$$
\sigma_{HL} = \frac{ \sqrt{\beta/2} - \sqrt{\beta} }{ k_2 \cdot (3 - 2\sqrt{2}) } + \sqrt{ \frac{\gamma}{ k_2^2 (3 - 2\sqrt{2}) } }
$$

with `k_1 = 4·ln 2 ≈ 2.7726`, `k_2 = sqrt(8/π) ≈ 1.5958`.

## R 实现（直接抄）

```r
HLSpreadEstimator <- function(highs, lows) {
    # highs, lows are length-2 vectors: [day_t, day_{t+1}]
    beta  <- (log(highs[1]/lows[1]))^2 + (log(highs[2]/lows[2]))^2
    H     <- max(highs); L <- min(lows)
    gamma <- (log(H/L))^2
    alpha <- (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    s     <- (2*(exp(alpha) - 1)) / (1 + exp(alpha))
    return(s)
}
```

## 适配到 tick 频率

原方法是 daily H/L over **2 个连续日**。在 tick 频率，把 100-tick 窗口分成两个 50-tick "sub-day"：
- `H_t = max(high)` over ticks 1..50
- `L_t = min(low)` over ticks 1..50
- `H_{t+1} = max(high)` over ticks 51..100
- `L_{t+1} = min(low)` over ticks 51..100

然后代入公式。

## 实战注意

1. **Spread 经常为负** → 这是公式数值不稳的体现。Bernt Ødegaard 课件给了 3 个 fixes：
   - 把负值 setting 0
   - 把负值 setting NaN
   - **对 overnight 调整**：如果 day t+1 的 H/L 与 day t close 不重叠，把 day t+1 的 H/L 整体平移
2. 我们 setup 没 overnight，不需要 fix #3
3. 在 100-tick 内可能波动率太低 → spread 估计噪声大，**只做 multi-window 平均**：W ∈ {30, 50, 100} 各算一份再 mean

## 与 Roll 的关系

| | Roll (1984) | Corwin-Schultz (2012) |
|---|---|---|
| 输入 | Δp 的 cov | H, L only |
| 假设 | bid-ask bounce 主导 cov | high = buy, low = sell |
| 当 cov > 0 / spread 为负 | 失效 | 仍可强制 0 |
| 还能估 vol？ | 否 | 是（Becker-Parkinson 副产物） |

**两者互补**，建议都加。在 100-tick 频率，**Corwin-Schultz 通常比 Roll 更稳**（因为 Roll 在短 window 下 cov 噪声大）。

## 后续文献

- **Abdi-Ranaldo 2017** RFS — CHL estimator (close + high + low) 改进
- **Bernhardt-Ahn 2018** — corporate bonds 的实战 implementation 注意

## 我们的实施

```python
def cs_spread(high, low, W=50):
    n = len(high)
    half = W // 2
    h_t  = high[-W:-half].max(); l_t = low[-W:-half].min()
    h_t1 = high[-half:].max();   l_t1 = low[-half:].min()
    H, L = max(h_t, h_t1), min(l_t, l_t1)
    
    beta  = np.log(h_t/l_t)**2 + np.log(h_t1/l_t1)**2
    gamma = np.log(H/L)**2
    
    den = 3 - 2*np.sqrt(2)
    alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return max(s, 0)  # impose positivity
```

预期增益：**+0.3 ~ +1.0**（短期 vol regime + spread 互补 channel）。
