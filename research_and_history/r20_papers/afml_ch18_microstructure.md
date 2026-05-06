# Lopez de Prado AFML Ch 18-19 — Microstructural Features

**López de Prado, M. 2018. "Advances in Financial Machine Learning."** Wiley.
- Ch 18: "Entropy Features"
- Ch 19: "Microstructural Features"
- Ch 20: "Multi-Process Feature Engineering"

## Ch 18: Entropy Features

### Shannon Entropy of returns
$$
H = -\sum_i p_i \log p_i
$$
其中 `p_i` 是过去 W ticks 内 return 落到 bin i 的频率（K=10 bins）。
- 高 H = returns 分布均匀 (high disorder)
- 低 H = returns 集中在 few bins (predictability higher?)

### Plug-in entropy estimator (LZ Lempel-Ziv approximation)

For binary tick-rule sequence `[+1, -1, +1, ...]`:

$$
H \approx \frac{n}{\sum_{j} l_j}
$$

`l_j` = length of j-th non-redundant substring in LZ parsing。

→ Tick rule entropy = "how predictable is the next sign". 低 entropy → predictable sequences → momentum regime.

## Ch 19: Microstructural Features (核心三组)

### §19.1-19.2 Roll-style spread (见 r20_papers/roll_1984.md)

```python
def roll_measure(close, W=100):
    dlog = np.diff(np.log(close[-(W+1):]))
    cov = np.cov(dlog[:-1], dlog[1:])[0, 1]
    return 2 * np.sqrt(max(-cov, 0))

def roll_impact(close, dollar_volume, W=100):
    roll = roll_measure(close, W)
    return roll * np.sum(dollar_volume[-W:])
```

### §19.3 Corwin-Schultz HL spread (见 r20_papers/corwin_schultz_2012.md)

提供 spread + Becker-Parkinson vol。

### §19.4 Kyle / Amihud / Hasbrouck Lambdas

完整形式（trade-based）：

```python
def kyle_lambda(returns, dollar_volume, signs):
    # OLS: returns = lambda * (signs * sqrt(dollar_volume))
    S = signs * np.sqrt(np.abs(dollar_volume))
    return np.cov(returns, S)[0, 1] / np.var(S)

def amihud_lambda(returns, dollar_volume):
    return np.mean(np.abs(returns) / (dollar_volume + 1e-9))

def hasbrouck_lambda(log_returns, dollar_volume, signs):
    # Same as Kyle but log returns
    S = signs * np.sqrt(np.abs(dollar_volume))
    return np.cov(log_returns, S)[0, 1] / np.var(S)
```

### §19.5 VPIN (见 r20_papers/vpin_easley_2012.md)

完整 BVC + buckets + sample length 实现。

## Ch 20: Multi-Process Feature Engineering

主要讨论用 `mpUtils` 并行 feature 计算（Python 多进程）→ 我们 sklearn/numpy 向量化已够，**不需要这章的 multiprocessing 框架**。

但 Ch 20 提到的 **PCA 降维 50→5** 在我们 setup 也适用 — 当 130+ 因子加完后，跑 PCA 看是否 redundant。

## Ch 5-6 (relevant): Volume / Tick / Run / Imbalance Bars

- **Tick bars**: 每 N tick 一个 bar
- **Volume bars**: 每 V volume 一个 bar
- **Dollar bars**: 每 D dollar volume 一个 bar
- **Imbalance bars**: 每 |Σsign(Δp)·v| > θ 一个 bar
- **Run bars**: 每相同 sign run 长度 > T 一个 bar

→ 我们 setup 是 **time bars (3s)**，但**可以在 100-tick window 内用上 volume / tick / imbalance bar 重新切片**做 feature aggregation：
```
e.g. group 100 ticks into 5 volume buckets, each bucket: mean OFI, std OFI
```

## Ch 17: Detecting Structural Breaks (CUSUM)

**CUSUM filter**:
$$
S_t^+ = \max(0, S_{t-1}^+ + r_t - h^+)
$$

`h^+` = pre-set threshold。当 S 超 h → break detected。
- 在我们 setup：作为 regime change indicator features（CUSUM stat 本身）

## 适配评估

| 章节 | 类别 | 我们用度 |
|---|---|---|
| Ch 18 entropy | feature | ⭐ low（计算成本 vs 收益不明） |
| Ch 19 spread (Roll, CS) | feature | ⭐⭐⭐ high — 已 nominate |
| Ch 19 lambdas (Kyle, Amihud, Hasbrouck) | feature | ⭐⭐ high — 已 nominate |
| Ch 19 VPIN | feature + gating | ⭐⭐ high — 已 nominate |
| Ch 20 PCA / multi-proc | infra | ⭐ low (我们 numpy 已快) |
| Ch 5-6 volume/tick bars | aggregation method | ⭐ medium (在 100-tick 内重新切片) |
| Ch 17 CUSUM regime | regime feature | ⭐ medium |
| Ch 3 triple barrier label | label | ⭐ medium (已是 label，不修改) |

## 我们已 cover 的 AFML 章节

- Ch 3 三分位标签 → 已用 (R10 paper notes)
- Ch 7 PurgedKFold → R12 已用
- Ch 10 bet sizing → R13 Scheme D 已 propose
- Ch 17 CUSUM → 这次 R20 mention（可加 as feature G11 区域）
- **Ch 19 microstructure → R20 重点**

## 实施提示

1. mlfinlab 库在 GitHub 上是 stub（pass 实现）；需要自己照 Ch 19 公式 reimplement
2. AFML PDF 在 Google Scholar 大量上载，公式可直接找到
3. **VPIN 需要 std normal CDF**：用 `scipy.stats.norm.cdf` 或 numpy `0.5 * (1 + erf(x / sqrt(2)))`

## 优先级总结

R20 short list 中：
- Roll spread (B1) — Ch 19 §19.1
- Corwin-Schultz (B3) — Ch 19 §19.3
- Kyle λ (B5) — Ch 19 §19.4
- Amihud (B6) — Ch 19 §19.4
- VPIN (D1) — Ch 19 §19.5
- BV / Jump (C2-C5) — Ch 19 §19.6（与 BNS 2004 同源）

→ 6/30 个 short list 直接 from AFML Ch 19。

**结论**：AFML Ch 19 是 R20 工作的"骨架来源"。读完这章，就 cover 了 60% 微观结构 feature 设计空间。
