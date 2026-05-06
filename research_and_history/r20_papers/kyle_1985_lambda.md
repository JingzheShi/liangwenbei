# Kyle 1985 — Lambda (Price Impact)

**Kyle, A. S. 1985. "Continuous Auctions and Insider Trading."** *Econometrica* 53(6): 1315-1335.

## Core idea

设市场含三类参与者：(1) informed trader（独占 private signal）；(2) noise traders；(3) market maker。MM 看 net order flow 推断 fair price。Kyle 证明：

$$
\Delta p_t = \lambda \cdot \text{net order flow}_t + \epsilon_t
$$

`λ` 是市场流动性的核心参数：高 λ = 同样 order 推动价格更多 = 流动性差。

## 实战估计公式（最常用版本）

$$
r_{t} = \lambda \cdot S_t + \epsilon_t
$$

- `r_t` = log return over period t
- `S_t = Σ sign(v_k) · sqrt(|v_k|)` — signed root-volume in period t
- 用 OLS 回归得到 λ̂

**为什么用 sqrt(volume) 不是 raw volume？**——Kyle (1985) 假设 flow 与价格 linear；实证 (Hasbrouck 1991) 发现 √volume 拟合更好（concave price impact）。

## 我们的实施（rolling W=50 ticks）

```python
def kyle_lambda(close, volume, W=50):
    r = np.diff(np.log(close[-(W+1):]))                           # W returns
    sign = np.sign(np.diff(close[-(W+1):]))                        # tick rule
    S = sign * np.sqrt(np.abs(volume[-W:]))
    
    # OLS slope (closed form)
    X = S - S.mean()
    Y = r - r.mean()
    lam = (X*Y).sum() / ((X*X).sum() + 1e-9)
    return lam * 1e6  # rescale for numerical stability
```

frds.io 的实现就是上面这版（multiply by 1e6 for human-readable scale）。

## 为什么作为 feature

1. **Liquidity proxy**: 高 λ → illiquid → 大订单冲击大
2. **Adverse selection proxy**: λ 高暗示 informed 多 → 后续价格 mean-revert 风险大
3. **与 RV 不同 channel**: RV 度量已实现波动；λ 度量边际成交对价格的弹性

## 适配评估

- ✅ 单股票
- ✅ stateless（rolling W=50 内重算）
- ✅ sym-agnostic
- ⚠️ 需要 rolling OLS — closed-form 实现 ~50 lines numpy

## 与 Amihud illiquidity 的关系

Amihud (2002) 是 Kyle 的廉价代理：
- Amihud = mean(|r| / volume) — 不需要 sign
- Kyle λ = OLS coef — 需要 signed volume

Amihud 简单但少 directional info；Kyle 完整但要算 sign。**两个都加**：常 corr ~0.7，但残差仍有信号。

## 后续文献

- **Hasbrouck 1991** — VAR 模型扩展（同时 model trade signs and quote revisions）
- **Brennan-Subrahmanyam 1996** — Kyle λ 在 cross-section 解释 expected return
- **Barbon-Buraschi 2020** — High-frequency Kyle λ 在 SPX 上 90%+ R²

## 我们的优先级

⭐⭐ high — Kyle λ 是 microstructure 标尺，比简单 imbalance 高 1 阶。和 R12 ensemble 结合好（不同 base model 用 RV vs λ 作主特征 → diversity）。

预期增益：standalone +1 ~ +3；与 RV combine 后 +0.5 ~ +1.5。

## 实施风险

1. **多重共线** with TFI/OFI（同源信号）→ 加之前看 corr matrix
2. **rolling OLS 慢** if 用 scipy.linregress in loop；必须用 closed-form numpy
3. **Outlier 敏感** — 单 tick 极端 volume 会 dominate λ̂；考虑 winsorize at 99%

## 与已实施 features 关系

T3 已有：mb_intst, ma_intst (market buy/sell intensity)。Kyle λ 是这些的"边际价格响应"，更高阶。
