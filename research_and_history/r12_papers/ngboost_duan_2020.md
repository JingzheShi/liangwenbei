# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

**作者**：Tony Duan, Anand Avati, Daisy Yi Ding, Sanjay Basu, Andrew Ng, Alejandro Schuler
**会议/年**：ICML 2020
**arXiv**：[1910.03225](https://arxiv.org/abs/1910.03225)
**仓库**：[stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost)

---

## 核心 idea

让 gradient boosting 直接预测**概率分布**而不是单点。

LightGBM/XGBoost 默认输出 ŷ（点估计）；NGBoost 输出**分布参数** θ（如 Normal 的 (μ, σ²)、Categorical 的概率向量），boosting 在分布参数空间上做。

## 关键技术贡献

### 1. Multiparameter boosting

每一步在所有分布参数上同时学一个 base learner（默认 decision tree）：
- 对 Normal: 同时学 μ 和 log σ
- 对 Categorical: 学 K-1 个 logit

### 2. Natural gradient

普通 gradient ∂L/∂θ 在不同 parametrization 下方向不一致（如 σ vs log σ）。
NGBoost 用 Fisher information matrix 修正：

$$\tilde\nabla_\theta L = F(\theta)^{-1} \nabla_\theta L$$

→ 与 parametrization 无关，等价于在 KL 度量下做 steepest descent。

### 3. Proper scoring rule

支持任何 proper scoring rule 当 loss：
- **Negative Log-Likelihood (NLL)**：最常用，对应 MLE
- **CRPS (Continuous Ranked Probability Score)**：对长尾更稳

## 性能（regression benchmark）

| 数据集 | LightGBM RMSE | NGBoost RMSE | NGBoost gives σ? |
|---|---|---|---|
| Boston Housing | 3.10 | 2.94 | ✅ |
| Energy | 0.49 | 0.51 | ✅ |
| Yacht | 0.66 | 0.50 | ✅ |
| Wine | 0.62 | 0.62 | ✅ |

→ point prediction RMSE **大致与 LightGBM 持平**（偶尔输 5-10%），但**多送一份 σ**（uncertainty）。

## Tradeoff vs LightGBM

| 维度 | LightGBM | NGBoost |
|---|---|---|
| 单点准确率 | ⭐⭐⭐ | ⭐⭐ |
| 不确定度 | ❌ 没有 | ✅ 全分布 |
| 训练速度 | 极快（GPU/CPU） | 慢 3-10×（纯 Python multiparam tree） |
| 实现成熟度 | 工业级 | 学术 prototype |
| 大规模数据 | ✅ | ⚠️ 慢，谨慎用于 > 1M 样本 |

## 对我们的启示

### 1. 不确定度可作为 threshold gating 的第二维度

T4 已验证：`max(p_0, p_2) > 0.5 且 > p_1 + 0.15` 阈值 → LOSO -22 → +6.45。
NGBoost 给我们一个新维度：**predictive entropy** 或 **logit posterior std**。
新阈值规则：`prob_max > 0.5 且 entropy < τ_h`，过滤"高均值但高不确定"的不可信预测。

### 2. Stacking 时作为 base 提供 calibration 信号

把 NGBoost 的 (μ, σ) 作为 meta feature → meta-learner 学到 "高 σ 时应当 fallback to flat (label=1)"。

### 3. 警惕速度

154 维特征 × 1M+ 训练样本 × 3 类 NGBoost 单 fold 训练可能 > 4 小时。建议先在 240k 样本子集验证 NGBoost 单模型 LOSO ≥ +10 再考虑用。

### 4. 也可换更轻量的 uncertainty estimator

如果 NGBoost 太慢，备选：
- **MC Dropout** on MLPLOB (Gal & Ghahramani 2016)：fast，但只对 NN 有效。
- **Deep ensemble std**（M 个独立 model 的 prediction std）：免费副产物。
- **Quantile regression LightGBM**（`objective='quantile', alpha=0.1/0.5/0.9`）→ 用 90%-10% quantile 当不确定度代理。**这是最便宜的路径**。

## 关键 takeaway

NGBoost 适合作为 **stacking 的一个 base + 提供 uncertainty 给 meta-learner**，**不太适合**作为唯一主模型（速度+精度都不胜 LightGBM）。
