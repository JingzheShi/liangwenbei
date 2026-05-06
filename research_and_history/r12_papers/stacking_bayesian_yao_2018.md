# Using Stacking to Average Bayesian Predictive Distributions

**作者**：Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman
**期刊/年**：Bayesian Analysis 2018
**PDF**：[Columbia stat archive](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf)

---

## 核心论点

在 **M-open** setting（即真实 data-generating process 不在候选模型集合里）中，
**Stacking > Bayesian Model Averaging (BMA)**。

我们金融 LOB 任务就是典型 M-open——所有 LightGBM/XGBoost/NN 都不可能是真实市场过程。

## BMA 的问题

BMA 给每个模型 k 一个权重：

$$w_k^\mathrm{BMA} \propto p(\mathcal{D}|M_k) \cdot p(M_k)$$

理论上这是"对模型不确定度做 marginalize"。但在 M-open 下：
1. 渐近上 BMA 把所有 weight 给**最接近真实**的那个候选 → 退化为单模型选择。
2. 对 prior 选择超敏感，工程不稳。
3. 不优化预测目标，只优化 marginal likelihood → 与 test prediction 错位。

## Stacking 的修正

直接优化 **leave-one-out (LOO) predictive distribution** 的 score：

$$\max_{w \geq 0,\, \sum w_k = 1} \frac{1}{n}\sum_{i=1}^n \log\Big(\sum_k w_k\, p_k(y_i | y_{-i})\Big)$$

可以理解为 "stacking weights = 让 mixture 的 LOO log-likelihood 最大的权重"。
约束 w ≥ 0 且 sum=1（simplex），用 PSIS-LOO 提供 $p_k(y_i|y_{-i})$ 的一致估计避免重复拟合。

## 与传统 stacking 的关系

经典 stacking (Wolpert 1992, Breiman 1996) 是**点估计** stacking：base 输出点预测，meta 学一个线性组合。
本论文 (Yao et al. 2018) 是**分布 stacking**：base 输出整个 predictive distribution，meta 学一个 mixture weight。

对我们 3 类分类：
- 经典 stacking → meta 输入 K × 3 概率，输出 3 类 logit。
- 分布 stacking → meta 输出 mixture weights w_k，最终概率 = sum_k w_k p_k(class)。

后者**等价于**约束 meta-learner 的 weight 是 simplex（非负且和为 1）—— **物理意义清晰、防过拟合**。

## 对我们的实践影响

### 1. Meta-learner 优先选 simplex-constrained linear

```python
from scipy.optimize import minimize

def neg_loo_log_score(w, base_probs, y_true):
    # base_probs: shape (N, K, C)
    mix = (w[None, :, None] * base_probs).sum(axis=1)  # (N, C)
    # log p(y_true)
    return -np.log(mix[np.arange(len(y_true)), y_true]).mean()

w0 = np.ones(K) / K
res = minimize(
    neg_loo_log_score, w0, args=(oof_probs, y),
    constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
    bounds=[(0, 1)] * K,
)
```

这正是我们**方案 10**（Nelder-Mead on OOF）的理论基础——但目标改成 PnL 而不是 log-likelihood。

### 2. Pseudo-BMA weights 作为正则

如果 OOF 数据少（5 LOSO fold 只 5 sym），simplex stacking 容易过拟合。Yao et al. 推荐 **bootstrapped Pseudo-BMA**（用 Bayesian bootstrap 算 weight 的不确定性 + Dirichlet 平滑）。
可以在 OOF 上 bootstrap 100 次，每次 stacking 得 weight，最后取 median weight。

### 3. 与 XStacking (2025) 一致

最近 XStacking 论文（ScienceDirect 2025）明确 cite 这篇，强调"非负权重 + 限制 meta-learner 复杂度"是金融 tabular 上 stacking 不退化的关键。

## 失败模式（需避免）

1. **K 太大、OOF 数据少**：5 LOSO sym × 240k samples，base K > 8 容易过拟合。
2. **Base 高度相关**：weight 优化 ill-conditioned，bootstrap 一下 weights 抖得很厉害。
3. **OOF 不真正 leakage-free**：本论文假设 LOO 估计是 unbiased；我们用 LOSO 时必须确保 base 训 fold 不含 holdout sym。

## 关键 takeaway

把我们的 **方案 2 (stacking with Ridge meta)** 升级为：
- meta = simplex-constrained mixture（非负、和为 1） **>** 普通 Ridge
- 目标 function = OOF cum_pnl（or weighted average of OOF accuracy 和 OOF PnL） **>** 单纯 cross-entropy
- bootstrap weights 取 median，提高鲁棒性
