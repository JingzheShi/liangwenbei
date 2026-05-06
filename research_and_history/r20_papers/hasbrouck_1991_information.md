# Hasbrouck 1991 — VAR Information Content of Trades

**Hasbrouck, J. 1991. "Measuring the Information Content of Stock Trades."** *Journal of Finance* 46(1): 179-207.

## Core innovation

把 trades 和 quote revisions 一起用 **Vector Autoregression (VAR)** 建模：

$$
\begin{pmatrix} r_t \\ x_t \end{pmatrix} = \sum_{i=1}^{p} A_i \begin{pmatrix} r_{t-i} \\ x_{t-i} \end{pmatrix} + \begin{pmatrix} \epsilon_{1,t} \\ \epsilon_{2,t} \end{pmatrix}
$$

- `r_t` = mid-quote return
- `x_t` = signed trade indicator (or signed volume)
- `ε` 是 VAR shocks（通过 Cholesky 分解恢复 structural shocks）

**Information content of a trade** = ultimate cumulative effect of trade innovation `ε_2` on price:

$$
\text{Inf}(x) = \sum_{k=0}^{\infty} \psi_k(x \to r)
$$

`ψ_k` = impulse response function (k step out)。

## 简化代理：Hasbrouck Lambda

AFML §19.4 给的简化版（避免 VAR 估计成本）：

$$
\hat{\lambda}_H = \text{OLS slope of: } \Delta \log p_t \sim S_t
$$

with `S_t = sign(Δp_t) · sqrt(volume_t · price_t)`.

→ 与 Kyle λ 几乎一样，只是 returns 用 log 形式。

## 关键 empirical finding (Hasbrouck 1991, NYSE)

1. **价格冲击 protracted lag**：trade 的全部价格效果不在当下完成，要 5-10 mins 后才"全部价格化"
2. **Concave size response**：单笔 100k 股的冲击 < 100 笔 1k 股的冲击之和（spread widening + queue depletion）
3. **Spread widening after large trades**：大 trade 后 spread 变宽 (adverse selection signal)
4. **Information asymmetry highest in small caps**：小盘股 trade information content 最高

## 我们 setup 适配

✅ **简化 λ 版本可用**（rolling W=50）
- 完全单股票
- stateless（每个 100-tick window 内 fit）
- 与 Kyle λ corr ~0.95，**不要两个都加**，二选一

⚠️ **完整 VAR 版本**：太复杂（要 solve Cholesky + impulse response），且需要 stationary，**不推荐在 100-tick 内做**。

## 选择 Kyle vs Hasbrouck

| 对比 | Kyle λ | Hasbrouck simplified λ |
|---|---|---|
| Return 形式 | linear `Δp` | log `Δlog p` |
| Volume 形式 | √v | √v |
| Robustness to scale | 差（受 price level 影响） | 好（log return 稳定） |
| 经典文献 | Kyle 1985 Econometrica | Hasbrouck 1991 JOF |

**推荐 Hasbrouck 简化版**：log return 在不同股票间更可比（sym 0-4 价格 level 不同）。

## 与 Cont-Kukanov-Stoikov OFI 关系

OFI（cont 2014）证明 net order flow 与 price change 近乎线性。Hasbrouck VAR 是 OFI 的"动态 lag" 版本——**OFI 是 Hasbrouck VAR 的 lag-0 component**。

→ 已有 OFI W=60 + 我们加 Hasbrouck λ 短窗 = 提供"impact decay rate" 隐含信息。

## 优先级

⭐ medium — 与 Kyle λ 强重合，建议**择一实施**。如果选 Hasbrouck，rolling OLS 用 log return 即可。

## 后续 paper

- **Hasbrouck 2009** — Trading costs from daily data (GMM Roll model with overnight)
- **Foster-Viswanathan 1990** — Multi-period extension of Kyle
- **Glosten-Harris 1988** — Decompose spread into informational + transient

## 我们的实施推荐

如果 Kyle λ 已加，Hasbrouck **不必额外加**；反之亦然。两者 ROI 接近。
