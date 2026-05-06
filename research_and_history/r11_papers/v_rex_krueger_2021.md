# V-REx — Out-of-Distribution Generalization via Risk Extrapolation

> Krueger, Caballero, Jacobsen, Zhang, Binas, Zhang, Le Priol, Courville. ICML 2021 (arXiv:2003.00688)
> [Paper](https://arxiv.org/abs/2003.00688) | [PMLR v139](https://proceedings.mlr.press/v139/krueger21a.html)

## 一句话

在 ERM loss 上加一项**跨 environment risk variance penalty**：让模型在所有 envs 上的 loss 一样大（而不是平均小）。

## 核心 objective

**V-REx**:
```
L_VREx = β · Var_e(R_e(θ)) + (1/N) Σ_e R_e(θ)
```
其中 R_e(θ) 是 env e 上的 expected risk。
- β = 0 → ERM
- β → ∞ → 强迫所有 env 的 risk 相等

**MM-REx (extrapolative variant)**:
```
L_MMREx = max over αₑ ≥ -λ, Σαₑ = 1: Σ αₑ R_e(θ)
```
等价于在"放大版"的 environment 集合上做 robust optimization (允许 α 取负值即"反向外推" envs)。

## 关键 insight

> "Variation across training domains is **representative** of variation at test time, but shifts at test time may be more **extreme** in magnitude."

→ 通过 reduce 训练时的 risk variation，模型自动 robust 于"更极端"的 test-time shift。

> "V-REx is exactly equivalent to a mean squared error penalty term."

→ 极简实现。在我们的训练 loop 里只需：
```python
losses_per_env = [loss_fn(...)[sym==g].mean() for g in range(5)]
ermloss = mean(losses_per_env)
penalty = var(losses_per_env)
total_loss = ermloss + beta * penalty
```

## vs IRM

> "By appropriately trading-off robustness to causally induced distributional shifts and covariate shift, REx is able to **outperform alternative methods such as Invariant Risk Minimization** in situations where these types of shift co-occur."

- IRM 在涉及 covariate shift（不仅是 label shift）的 setting 下崩
- V-REx 不崩
- ICLR 2021 / ICML 2021 比较实验里 V-REx > IRM

## 我们能用吗

- ✅ NN 路线：直接加 variance penalty 项
- ❌ LightGBM 路线：没有 batch loss / per-env loss 的概念。可以**近似**通过修改 sample_weight（让 high-loss env 的 weight 升高）→ 这等价于 Group DRO 的 soft 版本

## 推荐 β 值

论文里 β ∈ [1, 100]，太大会破坏 ERM 学习。建议从 β=10 开始 search。

> "Increasing the V-REx penalty (β) leads to a flatter risk plane and more consistent performance across domains, as the model learns to ignore color in favor of shape-based invariant prediction."

## 时序 / 金融实验

- 原 paper 主要在 ColoredMNIST（toy）和 PunyMNIST 上做
- 没有金融数据实验
- Wild-Time benchmark (Yao 2022) 把 V-REx 列为 baseline

## 我们任务的预测

- 优势：相比 IRM，V-REx 对"5 个 env 太少 + 154-d features"的 setting 更鲁棒
- 劣势：penalty term 仍可能让 model 在所有 sym 上都 underfit
- 预期增益：未知；NN 路线先小规模 tune β 再决定

## Sources
- [Paper arXiv:2003.00688](https://arxiv.org/abs/2003.00688)
- [ICML 2021 PMLR](https://proceedings.mlr.press/v139/krueger21a.html)
- [PDF](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)
- [Appendices](http://proceedings.mlr.press/v139/krueger21a/krueger21a-supp.pdf)
- [PaperTalk](https://papertalk.org/papertalks/32095)
