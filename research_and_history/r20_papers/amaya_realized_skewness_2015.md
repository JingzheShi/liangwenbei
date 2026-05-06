# Amaya, Christoffersen, Jacobs, Vasquez 2015 — Realized Skewness

**Amaya, D., P. Christoffersen, K. Jacobs, A. Vasquez. 2015. "Does Realized Skewness Predict the Cross-Section of Equity Returns?"** *Journal of Financial Economics* 118(1): 135-167.

## Core finding

用 5-min intraday returns 计算 daily realized skewness（RSkew）：

$$
\text{RSkew} = \frac{ \sqrt{N} \cdot \sum_{i=1}^{N} r_i^3 }{ \text{RV}^{3/2} }
$$

发现：**low RSkew 的 stocks 下周 return 显著高于 high RSkew**（spread ~24 bps/week）。

## 直觉

- 高 RSkew = recent returns 偏斜向正 → 可能"已经涨过头"或处在"乐观情绪"top → mean-revert
- 低 RSkew = 近期负偏（左尾事件多）→ 投资者过度厌恶 → 反弹

## 也包括 Realized Kurtosis

$$
\text{RKurt} = \frac{ N \cdot \sum_{i=1}^{N} r_i^4 }{ \text{RV}^2 }
$$

- 高 RKurt = recent fat tails → tail risk 高 → return 反向（risk premium 体现）

## 在我们 setup

✅ 完全适配 — 单股票，stateless，window 内可算。

```python
def rskew_rkurt(r, W=50):
    rv = (r[-W:]**2).sum()
    rskew = (np.sqrt(W) * (r[-W:]**3).sum()) / (rv**1.5 + 1e-9)
    rkurt = (W * (r[-W:]**4).sum()) / (rv**2 + 1e-9)
    return rskew, rkurt
```

W ∈ {20, 50, 100}

## 与 Realized Semi-variance 关系

Amaya 用 ALL returns 算 skew；BNS 2010 提出 **realized semi-variance** 拆开正/负 returns：

$$
\text{RV}^+ = \sum_{r_t > 0} r_t^2 ; \quad \text{RV}^- = \sum_{r_t < 0} r_t^2
$$

**Signed RV ratio**:
$$
\text{SRV} = \frac{ \text{RV}^+ - \text{RV}^- }{ \text{RV} }
$$

→ 正 SRV = upside vol dominant；负 SRV = downside vol dominant。

**两者互补**：
- RSkew 度量 third moment shape
- SRV 度量 sign 相对贡献
- 加全 = 4 个 vol asymmetry features

## 优先级

⭐⭐ high — 学术验证强，trivial 实现，与 RV/BV 不重叠。

## 后续 paper

- **Bali-Hu-Murray 2019** — 用 RSkew 当 cross-section 选股 signal
- **Patton-Sheppard 2015** — Good/Bad volatility（HAR-RV-Sν+/Sν-）
- **Conrad-Dittmar-Ghysels 2013** RFS — 期权 implied skew vs realized skew

## 我们的实施序列

加在短 list rank #3 — 与 BV/Jump 同时加（C2-C7 一起）；总 4 windows × 4 features = 16 个 vol asymmetry features。

预期增益：+1.5 ~ +3（vol asymmetry channel + jump regime channel 双重作用）。
