# Group DRO — Distributionally Robust Neural Networks for Group Shifts

> Sagawa, Koh, Hashimoto, Liang. ICLR 2020 (arXiv:1911.08731)
> [Paper](https://arxiv.org/abs/1911.08731) | [OpenReview](https://openreview.net/forum?id=ryxGuJrFvS) | [Code](https://github.com/kohpangwei/group_DRO)

## 一句话

**Min-max objective**：找到模型参数 θ 使得**最差 group 的 expected loss** 尽可能小（不是 mean loss）。

## 公式

`min_θ max_g E_{(x,y) ∼ P_g}[ℓ(x,y;θ)]`

实现上用 **online algorithm**：维护 group weights q_g，每个 step
- 算 per-group losses `R_g(θ)`
- 更新 `q_g ← q_g · exp(η_q · R_g)` (然后归一化)
- 模型梯度 `∇_θ Σ_g q_g · R_g(θ)`

## 关键发现（论文 main contribution）

> "Regularization is important for worst-group generalization in the overparameterized regime, even if it is not needed for average generalization."

- 标准 ERM 在 overparameterized NN 上 average accuracy 很高，但 worst-group accuracy 很差（10-40 percentage points worse）
- 加 Group DRO + 强 L2 / early stopping → worst-group accuracy 提升 10-40 个百分点
- **没有强正则的 Group DRO 不 work**——这一点很重要

## 我们怎么用

**LightGBM 路线**（自定义 training loop）：
```python
import lightgbm as lgb
import numpy as np

K = 200  # 每 K rounds 重算 group weight
model = None
group_weights = np.ones(5) / 5

for stage in range(num_stages):
    # 给每个样本打 sample_weight：基础类权重 × group_weight[sym]
    sample_weight = base_class_weight * group_weights[sym_array]
    train_set = lgb.Dataset(X, y, weight=sample_weight)
    
    # 接着训 K rounds
    model = lgb.train(params, train_set, num_boost_round=K, init_model=model)
    
    # 算 per-sym training loss
    pred = model.predict(X)
    losses = np.array([compute_loss(y[sym==g], pred[sym==g]) for g in range(5)])
    
    # 更新 group weights
    group_weights = softmax(losses / temperature)
```

**NN 路线**：直接用 [github.com/kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)

## 与我们 setup 的契合度

- ✅ 我们已有"sym 当 group" 完美映射
- ✅ Already validated 跨 sym variance 大（sym=2 outlier）
- ⚠️ 强正则要求 → LightGBM 要加大 `lambda_l2`、降 `num_leaves`、严格 `early_stopping`
- ⚠️ **5 个 group 太少**——Group DRO 在 group 数量大时统计稳定。5 个 group 时，最差的那个会主导整个 training。**对策**：用 softmax temp τ ≥ 2，平滑成"加权 ERM" 而非 hard min-max

## 预期增益

LOSO sum +3 ~ +8（保守估计；我们 sym=2 brittleness 是首要修复目标，Group DRO 对 sym=2 加权 → 模型更稳）

## 关键风险

**Worst-case 优化倾向于过拟合 outlier group**。我们的 sym=2（蓝筹）量级与其他差 4-7×，Group DRO 可能让 sym=2 主导 → 反而拖累 sym=0/1/3/4。**必须**配合 LOSO CV 监控每个 fold。

## 理论 bounds

> "10-40 percentage point improvements on worst-group accuracy" — Sagawa 2020 在 Waterbirds / CelebA / MultiNLI

我们任务是回归/3-class classification，目标指标是 cum_pnl 而非 accuracy，但**worst-group loss 改善 → worst-sym 的 PnL 改善**这个传导关系应该成立。

## 与 IRM 对比

| | IRM | Group DRO |
|---|---|---|
| Theory guarantee | linear setting only, needs envs > non-invariant dim | always min-max valid |
| 我们的 5 个 sym | 不够（154-d features） | 工作 |
| Implementation | 复杂（Hessian penalty） | 简单（reweighting） |
| Empirical robustness | 不稳定（CTB-2024 paper） | 在 WILDS 上稳赢 ERM |

→ **Group DRO 是 IRM 的实用替代品**

## Sources
- [Paper arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- [OpenReview](https://openreview.net/forum?id=ryxGuJrFvS)
- [Slides ICLR 2020](https://cs.stanford.edu/~ssagawa/assets/slides/groupDRO_ICLR2020.pdf)
- [Group DRO Code](https://github.com/kohpangwei/group_DRO)
- [Group-Level Distributional Uncertainty Extension 2025](https://arxiv.org/html/2509.08942v1)
