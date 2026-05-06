# Barndorff-Nielsen & Shephard 2004/2006 — Bipower Variation & Jump Test

**Barndorff-Nielsen, O. E., and N. Shephard.**
- 2004. "Power and bipower variation with stochastic volatility and jumps." *Journal of Financial Econometrics* 2(1): 1-37.
- 2006. "Econometrics of testing for jumps in financial economics using bipower variation." *Journal of Financial Econometrics* 4(1): 1-30.

## 核心 idea

Realized Volatility（RV）会被 **jumps（突发跳价）混入**——这是问题，因为：
1. Jumps 不是连续 vol 信号
2. 把 jumps 当 vol 会过预测下期 vol

**BV (bipower variation)** 是 RV 的 jump-robust 替代：用相邻 |r| 乘积，jumps（孤立大值）被相邻小值"稀释"掉。

## Formulas

### Bipower Variation
$$
\text{BV}_W = \frac{\pi}{2} \sum_{t=2}^{W} |r_t| \cdot |r_{t-1}|
$$

为何 π/2？因为 `E[|r|·|r|] = (2/π)·σ²` for Gaussian r → 校正系数 π/2 使 BV 在无 jump 下 → ∫σ²。

### Realized Quarticity
$$
\text{RQ}_W = \frac{W}{3} \sum_{t=1}^{W} r_t^4
$$

### Jump Test Statistic
$$
J_W = \frac{ \text{RV}_W - \text{BV}_W }{ \sqrt{ \left( \frac{\pi^2}{4} + \pi - 5 \right) \cdot \frac{\max(\text{RQ}_W, \text{BV}_W^2)}{W} } }
$$

H_0: 无 jump → J_W ~ N(0, 1)
- |J_W| > 1.96 → 在 5% level 拒绝 → recent jumps occurred

## 实战 features

1. **BV** 单独作 vol feature（与 RV 互补）
2. **RV - BV** = "continuous vol - jump-free vol" diff = jump component proxy
3. **J statistic** 作为 binary jump regime indicator（分 |J| > 2 / not）
4. **JumpVar (Mancini 2009 truncated)**:
   $$
   \text{JumpVar}_W = \sum_{t} \mathbf{1}\{ |r_t| > c \cdot \sqrt{\text{BV}_W / W} \} \cdot r_t^2
   $$
   c = 4 是常用 threshold。

## 在我们 100-tick 频率

- W ∈ {30, 50, 100}
- jumps 在 tick 频率指"单 tick 突变 > 5σ"——可能由信息事件、bid-ask 跳档、撤单激增触发
- **价值**：同时 expose model "smooth vol" (BV) 和 "jump intensity" (RV - BV)，让 model 自己学 jump regime

```python
def bv_jump(r, W=50):
    r_abs = np.abs(r[-W:])
    bv = (np.pi/2) * (r_abs[1:] * r_abs[:-1]).sum()
    rv = (r[-W:]**2).sum()
    rq = (W/3) * (r[-W:]**4).sum()
    j = (rv - bv) / np.sqrt((np.pi**2/4 + np.pi - 5) * max(rq, bv**2) / W + 1e-9)
    return rv, bv, rv - bv, j
```

→ 4 个 features × {30, 50, 100} = 12 个

## 后续 paper

- **Andersen-Bollerslev-Diebold 2007** RFS — HAR-RV-J model（用 jump separately predict vol）
- **Mancini 2009** — truncated jump variation
- **Lee-Mykland 2008** RFS — jump test with shrinking window

## 与 R10/R13 协同

- R10 PnL loss: jump regime 下 outcomes 大→ sample weight `|Δp|` 自然给 jump tick 高 weight
- R13 Bayes EV: J_W > 2 时（recent jump regime）调高 abstain threshold

## 优先级

⭐⭐ high - **BV 和 J 是 RV 之外最有学术 backing 的 vol features**。预期 +1 ~ +3 增益（特别在 high-vol regime）。

## 与 RSkew/Semi-vol 关系

- BV/J 度量 "vol 中 jump 占比"
- RSkew (Amaya 2015) 度量 "vol 的方向偏度"
- Semi-vol (BNS 2010) 度量 "正/负 vol 分量"

三者**互补，建议都加**。在 LightGBM 看 importance 决定哪个保留。
