# Amihud 2002 — Illiquidity Ratio

**Amihud, Y. 2002. "Illiquidity and stock returns: cross-section and time-series effects."** *Journal of Financial Markets* 5(1): 31-56.

## Core formula

$$
\text{ILLIQ}_W = \frac{1}{W} \sum_{t \in W} \frac{|r_t|}{v_t}
$$

- `r_t` = return at t (log or simple)
- `v_t` = dollar volume at t = price × shares traded
- W = aggregation window

直觉：每单位 dollar volume 造成多少 |return|；高 = illiquid。

## 在我们 setup

```python
def amihud(close, volume, W=50):
    r = np.diff(np.log(close[-(W+1):]))         # W returns
    dv = volume[-W:] * close[-W:]               # dollar volume
    return np.mean(np.abs(r) / (dv + 1e-9))
```

W ∈ {30, 50, 100} 三档。

## 为什么 work

1. **Liquidity premium**: illiquid stocks 长期超额 return → 有 vol 但被 illiquidity 拖累
2. **Cross-section evidence**: Amihud 1963-1997 美股月度数据：ILLIQ 高的 stock 月 return 高 1.5%（risk premium）
3. **Time-series evidence**: 同 stock 内 high ILLIQ 月 → 下月 return 高

我们用 tick 频率：高 ILLIQ tick → 后续短期价格更大概率 mean-revert（low liquidity 易被 noise 推动）。

## 与 Kyle λ 关系

| | Kyle λ | Amihud ILLIQ |
|---|---|---|
| 输入 | signed sqrt-volume | unsigned dollar volume |
| 估计方法 | OLS | mean ratio |
| 计算成本 | medium | trivial |
| Directional info | 是 | 否 |
| Robustness | OLS outlier 敏感 | mean 平稳 |
| 经典对比 | Brennan-Subrahmanyam 1996 | Amihud-Mendelson 1986 |

**实战**：先加 Amihud（trivial）；如果 Amihud feature_importance 高，再加 Kyle λ 看是否互补。

## 实施 trick

1. **极小 volume 导致 ratio 爆炸**：加 ε=1e-9，并 winsorize at 99%
2. **零 volume tick 跳过**：用 mask
3. 用 `mid * volume_delta` 作 dollar volume；schema 已有 `amount_delta` 直接用更准确

## 与 R10/R13 协同

- R10 PnL loss S2 weights samples by `|Δp| / fee` → high Amihud tick = high |Δp| 风险高
- R13 Bayes EV gating → Amihud 高的 tick 提高 confidence threshold

## 优先级

⭐⭐ high — 实施成本最低（trivial），与 RV/Kyle 互补 channel。

## 后续文献

- **Acharya-Pedersen 2005** RFS — liquidity-adjusted CAPM with Amihud
- **Brennan et al. 2013** JFE — high-freq version of Amihud
- **Lou-Shu 2017** RFS — turnover-based liquidity

## 我们的优先级位

短 list 第 12 名。预期增益：+0.5 ~ +1.5（与 Kyle 不强重叠时）。
