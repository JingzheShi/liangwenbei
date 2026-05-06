# R11 — Cross-Stock / OOD Robustness Methods (调研报告)

> Date: 2026-05-06
> Author: R11 worker
> Goal: 给比赛"评测时 sym 0-4 可能含训练外股票"提一组**具体、可落地**的 OOD 鲁棒性方法。
> 当前 baseline: iter_002 (T5b Scheme C, multi-horizon) — LOSO sum +21.86，跨 sym gap 仍 ~10-30 分。
>
> ⚠️ 三条硬约束（来自 `CRITICAL_CONSTRAINTS.md`）必须遵守：
> 1. `date` 评测时被置 0 — 不能当 feature
> 2. Predictor 不能跨 `predict()` 调用维护 state（顺序被打乱）
> 3. **sym ID 训练后不能在 inference 用**（评测可能含训练外 sym）
>    → 任何"训练时用 sym 当 group / domain / env label"的方法**只能影响 backbone 学到的表示**，不能让 inference 路径依赖 sym

---

## TL;DR — 如果只能上 2 个方案

**🥇 第 1 优先：Group DRO via dynamic sample reweighting (LightGBM-compatible)**
- 把 sym 当 group，训练时按 worst-group loss 动态重加权样本
- 直接套到 LightGBM `sample_weight` 参数，每 K boosting rounds 重算一次 group loss → reweight
- 已有 `class_weight='balanced'` 是 per-class，本方案是 per-(class, sym)
- 预期 LOSO sum +3~+8（主修 sym=2 / sym=0 brittleness）
- 实现复杂度：1 天（custom training loop wrapping `lgb.train`）

**🥈 第 2 优先：Cross-sym Mixup（在原始 100×D 切片上）**
- 训练时按概率 p=0.3 把同一 batch 内**不同 sym** 的两个窗口做 mixup（α=0.4 beta dist）
- Label 也 mixup（soft label → 用 cross-entropy with soft target）
- Jane Street 1st 验证有效（"fills empty space, soften overfitting"）
- 对 LightGBM 也能做：把 mixup 后的 (x, y) 当合成训练样本喂进去
- 预期 LOSO sum +1~+4
- 实现复杂度：0.5 天（数据 loader 改造）

**为什么是这 2 个**：都不破坏 sym-agnostic 推理路径（训练时用 sym，inference 时丢），都能直接套到现有 LightGBM 流水线，都已有金融数据上的实证支持。两个方案**互补**——一个改 loss landscape，一个改 data manifold——可以叠加。

---

## 方法清单（12 条，按"现实可行度"排序）

### A. Group DRO (Distributionally Robust Optimization, Sagawa et al. 2019)

- **核心机制**：min-max 优化，找让 worst-group loss 最小的模型。每个 epoch 重算 per-group loss，给损失最大的 group 加权重。
- **公式**：`L = max_g E_{(x,y)∈g}[ℓ(x,y;θ)]` （min over θ, max over groups g）
- **我们怎么用**：`group = sym ∈ {0,1,2,3,4}`。
  - **NN 路线**：直接套官方实现 ([github.com/kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO))，加 L2 weight decay (强正则是 worst-group accuracy 的关键)。
  - **LightGBM 路线**：每 K=200 rounds 暂停训练，eval per-sym training loss，按 softmax(loss_g / τ) 更新 `sample_weight`，继续训。
- **可用性**：✅ NN ✅ LightGBM
- **来源**：[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)；[OpenReview](https://openreview.net/forum?id=ryxGuJrFvS)；[github.com/kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)
- **金融数据 evidence**：被 [Group DRO under group-level uncertainty (2509.08942)](https://arxiv.org/html/2509.08942v1) 列为高 stakes 金融场景的标准工具
- **实现复杂度**：低（LightGBM 路径）/ 中（NN 路径）
- **预期增益**：LOSO sum +3~+8
- **风险**：只有 5 个 group，且 sym=2 outlier 太极端 → 模型可能被 sym=2 拽偏。**对策**：用 group loss 的 softmax 软加权（τ 大一点），不要硬选 worst。

---

### B. Cross-Sym Mixup（数据增强）

- **核心机制**：训练时随机抽 2 个不同 sym 的样本 (x_i, y_i)、(x_j, y_j)，做凸组合：`x' = λx_i + (1-λ)x_j, y' = λy_i + (1-λ)y_j`，λ ~ Beta(α, α)。
- **为什么修跨 sym gap**：synthesizing "sym 之间的中间分布"，迫使模型学到 sym-invariant decision boundary。
- **我们怎么用**：
  - 在 build_cache 后，训练时 dataloader 里以 p=0.3 触发 mixup，确保 i, j 来自不同 sym
  - α=0.4 (Jane Street 用 ~0.4)
  - LightGBM 路径：先生成 N_aug = 0.5 × N_train 条 mixup 样本，concat 进 Dataset
- **可用性**：✅ NN ✅ LightGBM (合成数据预生成)
- **来源**：[Zhang 2018 Mixup arXiv:1710.09412](https://arxiv.org/abs/1710.09412)；[Jane Street 1st place writeup discussion 224348](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348)；[Embarrassingly Simple MixUp for Time-series (arXiv:2304.04271)](https://arxiv.org/abs/2304.04271)；[Improving Time Series Forecasting with Mixup (Amazon Science)](https://www.amazon.science/publications/improving-time-series-forecasting-with-mixup-data-augmentation)
- **金融数据 evidence**：Jane Street 1st place 用 mixup（"fills empty space, softens overfitting"）；ResearchGate "Data augmentation for stock return prediction" 显示 mixup 增强 LightGBM stock prediction 全部 case 都有 accuracy 提升
- **实现复杂度**：低（dataloader 改造，半天）
- **预期增益**：LOSO sum +1~+4
- **风险**：100-tick 窗口内的时序 coherence 可能被 mixup 打散 → 试试只 mix 静态 features，时序部分原样保留

---

### C. V-REx (Risk Extrapolation, Krueger 2021)

- **核心机制**：在 ERM loss 上加一项**跨 environment risk variance penalty**：`L = mean(R_e) + β · Var(R_e)`，其中 R_e 是 env e 的 training risk。
- **vs IRM**：V-REx **更稳定**（IRM 在涉及 covariate shift 的场景下崩，V-REx 不崩，是 ICLR 2021 比较里的胜出方）；equivalent 到 MSE penalty term；β 越大，risk plane 越平 → 越偏 invariant。
- **我们怎么用**：每个 batch 算 per-sym loss → 加 variance penalty → 总 loss back-prop。
- **可用性**：✅ NN ❌ LightGBM（没有 batch loss 概念，只能近似）
- **来源**：[arXiv:2003.00688](https://arxiv.org/abs/2003.00688)；[ICML 2021 PMLR](https://proceedings.mlr.press/v139/krueger21a.html)
- **金融数据 evidence**：原 paper 没有金融实验；但 Wild-Time benchmark ([arXiv:2211.14238](https://arxiv.org/pdf/2211.14238)) 跨 method 比较里 V-REx 是 IRM 类方法的稳健替代
- **实现复杂度**：低（一个 reduce + variance op）
- **预期增益**：未知（非 LightGBM 主路线，备用）

---

### D. IRM (Invariant Risk Minimization, Arjovsky 2019)

- **核心机制**：找一组 representation Φ(x) 使得**同一个 classifier w 对所有 environment 都最优**。Penalty `||∇_w R_e||²` 强迫 invariance。
- **致命限制**：[OpenReview "The Risks of IRM"](https://openreview.net/forum?id=BbNIbVPJ-42) 证明：linear setting 下 IRM 只有当 **#envs > dim(non-invariant features)** 才能恢复 robust predictor。**我们只有 5 envs (sym)，特征 154 维**，远不满足条件 → IRM 大概率退化到 ERM 或更糟。
- **建议**：❌ 不优先做。如果一定试，先 PCA 降维再 IRM。
- **来源**：[arXiv:1907.02893](https://arxiv.org/abs/1907.02893)；[The Risks of IRM (arXiv:2010.05761)](https://arxiv.org/abs/2010.05761)
- **可用性**：✅ NN ❌ LightGBM
- **实现复杂度**：中

---

### E. DANN (Domain-Adversarial Neural Network, Ganin 2016)

- **核心机制**：feature extractor + label predictor + domain classifier (sym classifier) 三头。Gradient Reversal Layer 反向传播 domain loss → backbone 学 sym-invariant 表示。
- **我们怎么用**：
  - feature extractor: T5b Scheme C 的 MLPLOB / DeepLOB backbone
  - label predictor: 三分类 (down/flat/up)
  - domain classifier: sym 五分类
  - 训练时三头一起训；inference **只用 label predictor**（domain classifier 直接丢）
  - λ 按 schedule: `λ(p) = 2/(1+exp(-10p)) - 1`（p = training progress）
- **可用性**：✅ NN ❌ LightGBM
- **来源**：[arXiv:1505.07818 (DANN)](https://arxiv.org/abs/1505.07818)；[JMLR 2016](https://jmlr.org/papers/volume17/15-239/15-239.pdf)；[github.com/fungtion/DANN](https://github.com/fungtion/DANN)；[Springer 2023 "DANN for DG: when it works and how to improve"](https://link.springer.com/article/10.1007/s10994-023-06324-x)
- **金融 evidence**：DANN 在 ts forecasting 的 domain generalization 领域被 Deng 2024 ([Domain Generalization in Time Series Forecasting](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/deng-2024-domain.pdf)) 列为 baseline；有效但不是最强
- **实现复杂度**：中（NN 改架构）
- **预期增益**：未知
- **风险**：**只有 5 个 domain class**，domain classifier 太容易 → 反向梯度信号噪声大。可以试 **subsetting**：在每个 mini-batch 里只用 4 个 sym (LOSO 风格) 强化"留一外推"信号。

---

### F. CORAL / Deep CORAL (Sun & Saenko 2016)

- **核心机制**：对齐 source/target 的二阶矩（covariance）：`L_CORAL = (1/4d²)·||Cov(Φ_s) - Cov(Φ_t)||_F²`
- **我们怎么用**：
  - 在每个 mini-batch 里，对 sym i 的样本和 sym j 的样本各算 feature covariance，取 pairwise CORAL loss 加进总 loss
  - inference 不用任何 sym 信息
- **可用性**：✅ NN ❌ LightGBM
- **来源**：[arXiv:1607.01719 Deep CORAL](https://arxiv.org/abs/1607.01719)；[arXiv:1612.01939 CORAL UDA](https://arxiv.org/abs/1612.01939)；[github.com/SSARCandy/DeepCORAL](https://github.com/SSARCandy/DeepCORAL)
- **金融 evidence**：[TransCORALNet (supply chain credit cold-start)](https://arxiv.org/abs/2310.) 用 CORAL loss 处理 domain shift；信用风险论文也用 weighted CORAL (Springer 2020 "Robust and high-order CORAL UDA")
- **实现复杂度**：低（一个 cov loss）
- **预期增益**：未知

---

### G. Test-Time Adaptation — Within-Window-Only

- **核心约束**：**Tent 原版不能直接用**。Tent 跨 batch 累计 BN running stats，违反我们硬约束 §2 (no cross-call state)。但在**当前 100×D 窗口内**重算 normalization stats 是 OK 的（已被 T7 实现，window z-score）。
- **新发现 (2026)**: [Test-Time Adaptation for Non-stationary Time Series: Synthetic Regime Shifts to Financial Markets (arXiv:2602.00073)](https://arxiv.org/abs/2602.00073) 在 SPY/QQQ/EUR/USD 上明确报告：
  - "On synthetic gradual drift, normalization-based TTA improves forecasting error"
  - "In financial markets a simple batch-normalization statistics update is a robust default"
  - **"more aggressive norm-only adaptation can even hurt"** ← 警告：不要做激进 TTA
- **我们怎么用**：T7 已经做了 window z-score 这一步，等于已经吃了大部分 TTA 的肉。无新增建议除非搞 NN 路线（那时把 BN 的 batch stats 用当前 100-tick 窗口内的 stats 替代）。
- **可用性**：✅ NN（BN replacement）；T7 已经在 LightGBM 上做了等价的 window z-score
- **来源**：[Tent (arXiv:2006.10726)](https://arxiv.org/abs/2006.10726)；[Test-Time Adaptation for Non-stationary TS (arXiv:2602.00073)](https://arxiv.org/abs/2602.00073)
- **预期增益**：T7 已吃；NN 路线时再考虑

---

### H. Stochastic Weight Averaging (SWA, Izmailov 2018)

- **核心机制**：训练后期把多个 epoch 的 SGD 权重取平均，→ 更平的 minimum → 更好的 OOD 泛化。
- **OOD evidence**：["SWA exhibits greater robustness against distribution shift" — Izmailov 2018 / SWA Revisited 2022](https://arxiv.org/abs/1803.05407)
- **我们怎么用**：NN 训练时启用 PyTorch `torch.optim.swa_utils.AveragedModel`，最后 25% epochs 做权重 SWA。
- **可用性**：✅ NN ❌ LightGBM
- **来源**：[arXiv:1803.05407](https://arxiv.org/abs/1803.05407)；[PyTorch SWA blog](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)；[SWA Revisited (arXiv:2201.00519)](https://arxiv.org/abs/2201.00519)
- **实现复杂度**：低（PyTorch one-liner）
- **预期增益**：NN 路线可加；LightGBM 等价物 = ensemble multi-seed (我们 T6/T11 在做)

---

### I. Bootstrap LOSO Ensemble（已在做的延伸）

- **核心机制**：训练 K=5 个模型，每个 hold-out 一个 sym，inference 时取**简单平均/中位数**。这是 implicit IRM——每个模型被迫不依赖 hold-out 的 sym 特性。
- **我们怎么用**：
  - 我们 T4 / T5b 都跑 LOSO 5-fold，已经有 5 个 model 在手 → 直接 ensemble inference
  - **新建议**：每个 fold 内再训 N=3 seed → 总共 15 个模型，inference 取 mean prob 再 threshold
  - 与 T11 (Scheme C multi-seed) 互补
- **可用性**：✅ NN ✅ LightGBM
- **来源**：implicit IRM by ensembling — 见 [DG via ensemble (Cha 2021 SWAD)](https://arxiv.org/abs/2102.08604)；["Group Distributionally Robust ML under group level uncertainty"](https://arxiv.org/html/2509.08942v1)
- **实现复杂度**：低（已有 LOSO checkpoints）
- **预期增益**：LOSO sum +1~+3（vs single fold）

---

### J. Hash-Trick Stock Embedding + Random Mask（处理"训练外 sym"的方法）

- **核心机制**：把 sym ID hash 到一个固定 bucket（比如 `bucket = hash(sym) % 16`），训练时再随机 mask 50% 样本的 bucket id（设为特殊"unknown"token）→ 模型必须学会"没有 sym 也能预测"。
- **为什么不违反硬约束**：bucket=`hash(sym)%16` 对训练外的 sym=ID 不会 IndexError（只要 bucket size ≥ unique sym 数）。但**直接禁止用**——因为训练外 sym 的 hash bucket 与训练内 sym 的 bucket 重合时会走它**邻居 sym 的 embedding** → 仍然可能 mismatch。
- **结论**：**不推荐**。除非平台明确"训练外 sym 与训练 sym 行为相似"，否则风险不可控。CRITICAL_CONSTRAINTS.md §1 #3 推荐"完全 sym-agnostic"是最安全方案。
- **可用性**：仅 NN
- **来源**：[Hash Embeddings (arXiv:1709.03933)](https://arxiv.org/abs/1709.03933)；[Stock2Vec (arXiv:2010.01197)](https://arxiv.org/abs/2010.01197)
- **预期增益**：风险大于收益，跳过

---

### K. Quantile / Huber / Focal Loss（鲁棒 loss family）

- **核心机制**：替换 cross-entropy / MSE 为对 outlier 不敏感的 loss。
  - **Huber loss**：` |x| < δ ? 0.5x² : δ(|x| - 0.5δ)` — 离群点线性损失而非二次
  - **Quantile loss (pinball)**：`max(τ(y-ŷ), (τ-1)(y-ŷ))` — 学 quantile 不学 mean
  - **Focal loss**：`-(1-p_t)^γ log(p_t)` — focus on hard examples
- **跨 sym 应用**：sym=2 是 outlier sym（量级 4-7× 其他），对它做 robust loss → 不让 sym=2 主导梯度。
- **我们怎么用**：
  - LightGBM 直接支持 `objective='quantile'` (回归) / `objective='huber'`
  - 多分类用 focal loss → ([github.com/jrzaurin/LightGBM-with-Focal-Loss](https://github.com/jrzaurin/LightGBM-with-Focal-Loss))
- **可用性**：✅ NN ✅ LightGBM
- **来源**：[Lin 2018 Focal Loss](https://arxiv.org/abs/1708.02002)；[LightGBM with Focal Loss (Medium)](https://medium.com/data-science/lightgbm-with-the-focal-loss-for-imbalanced-datasets-9836a9ae00ca)；["False awareness stock market prediction by LightGBM with focal loss" (ACM RACS 2022)](https://dl.acm.org/doi/10.1145/3538641.3561502)
- **金融 evidence**：focal-loss LightGBM 在 stock prediction 任务上"outperformed other ML systems regarding predictive accuracy, profitability, and risk-controlling performance" (ACM RACS 2022)
- **实现复杂度**：低（custom objective）
- **预期增益**：LOSO sum +1~+3

---

### L. Per-Stock Standardization @ Train + Sym-Agnostic @ Inference (Optiver HYD trick — 改造版)

- **背景**：Optiver Trading at the Close 1st place (HYD) 用 **per-stock_id standardization** "to slightly improve MAE"。但这违反我们硬约束 #3（评测时无法 per-sym 标准化）。
- **改造方案**：训练时**先用 per-sym z-score 学一遍模型**，再做 distillation 到一个"全局 z-score 输入"的学生模型——学生模型推理时不用 sym。本质是用 per-sym 模型的 soft label 当 teacher signal。
- **可用性**：✅ NN ✅ LightGBM (LightGBM 用 prob 当 soft target → MSE on logit 即可)
- **来源**：[Optiver Trading at Close 1st place (HYD)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)；[Medium "Gauging the Market" Optiver writeup](https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0)
- **实现复杂度**：中（两阶段 training）
- **预期增益**：未知（推断有 +1~+3，但量级不确定）
- **风险**：teacher 用了 sym → 如果 teacher overfit 到训练 sym，蒸馏出的学生也是有限的

---

## 不推荐的方法（明确反对）

| 方法 | 为什么不推荐 |
|---|---|
| Per-sym embedding (`nn.Embedding(5, ...)`) | 直接违反硬约束 #3，训练外 sym 报错或拿到错的 embedding |
| Per-sym normalization at inference | 违反硬约束 #3 |
| Hash-trick stock embedding (J) | bucket collision 时仍会走邻居 sym 的 embedding，风险不可控 |
| 原版 Tent (跨 batch 累计 BN running stats) | 违反硬约束 #2，Predictor 不能维护跨 call 状态 |
| Cross-sectional features 跨 sym 同时刻 | 评测 batch 不一定包含多个 sym，且顺序被打乱 |
| Domain-specific online retraining (Optiver 12-day retrain trick) | 评测时没有 retraining 接口 |

---

## 关于"评测 sym 与训练 sym 同 ID 不同股票"这一极端情况

- 平台明文："sym 范围 0-4（**可能**有不来自训练集 5 只股票的数据）"
- **最坏情况**：评测的 sym=0 对应的股票与我们训练的 sym=0 完全不是一只 → 任何依赖 sym ID 的方法都死
- **唯一安全的姿态**：**模型对 sym ID 不知情**——这是我们已经在做的，本调研所有方法都遵守
- **但训练时仍可用 sym 当 group label**——只要它**只影响 loss / regularization / data sampling**，**不进 inference 路径**

---

## 与已有 trick 的兼容性矩阵

|  | T7 window z-score | T5b multi-horizon Scheme C | Threshold gating | Multi-seed ensemble |
|---|---|---|---|---|
| Group DRO (A) | ✅ | ✅ | ✅ | ✅ |
| Cross-sym Mixup (B) | ✅ | ✅ | ✅ | ✅ |
| V-REx (C) | ✅（NN）| 仅 NN 路线 | ✅ | ✅ |
| DANN (E) | ✅（NN）| 仅 NN 路线 | ✅ | ✅ |
| CORAL (F) | ✅（NN）| 仅 NN 路线 | ✅ | ✅ |
| TTA (G) | ✅（已含）| ✅ | ✅ | ✅ |
| SWA (H) | ✅（NN）| 仅 NN 路线 | ✅ | ✅（已是 ensemble）|
| LOSO Bootstrap (I) | ✅ | ✅ | ✅ | ✅ |
| Robust Loss (K) | ✅ | ✅ | ✅ | ✅ |

---

## 落地建议（明天 / 这周可做）

### 明天（1 个 worker，半天）
**T12-A: Cross-Sym Mixup augmented LightGBM 训练**
- Step 1: 在 T5b Scheme C 训练数据基础上，生成 N_aug = 0.5 × N_train 条 cross-sym mixup 合成样本（α=0.4，确保 sym_i ≠ sym_j）
- Step 2: 把 mixup 样本 concat 进训练集，重训 LightGBM Scheme C
- Step 3: LOSO 5-fold 评估 → 对比 iter_002 +21.86 baseline

### 本周（1-2 个 worker，2-3 天）
**T12-B: Group DRO via dynamic sample reweighting**
- Step 1: wrap `lgb.train` with custom training loop，每 K=200 rounds 算 per-sym training loss
- Step 2: 按 `sample_weight_g = sample_weight_g · exp(η · loss_g)`（η=0.05）
- Step 3: LOSO 5-fold 评估

**T12-C (备用)**：robust loss switch
- 把 Scheme C 的多分类 loss 换成 focal loss (γ=2)，看是否对 sym=2 hard examples 有帮助

### 后续（如果 NN 路线启动）
- DANN with sym discriminator + GRL
- V-REx penalty (β=10)
- Deep CORAL pairwise sym alignment
- SWA 后期 epoch 权重平均

---

## 参考文献完整列表

### Distribution Shift / OOD Generalization
- Arjovsky et al. 2019. "Invariant Risk Minimization". [arXiv:1907.02893](https://arxiv.org/abs/1907.02893)
- Kamath et al. 2020. "The Risks of Invariant Risk Minimization". [arXiv:2010.05761](https://arxiv.org/abs/2010.05761) / [OpenReview](https://openreview.net/forum?id=BbNIbVPJ-42)
- Krueger et al. 2021. "Out-of-Distribution Generalization via Risk Extrapolation (REx)". [arXiv:2003.00688](https://arxiv.org/abs/2003.00688) / [PMLR v139](https://proceedings.mlr.press/v139/krueger21a.html)
- Sagawa et al. 2020. "Distributionally Robust Neural Networks for Group Shifts (Group DRO)". [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) / [github](https://github.com/kohpangwei/group_DRO)
- Group DRO under Group-Level Distributional Uncertainty 2025. [arXiv:2509.08942](https://arxiv.org/html/2509.08942v1)

### Domain Adaptation
- Ganin & Lempitsky 2016. "Domain-Adversarial Training of Neural Networks (DANN)". [arXiv:1505.07818](https://arxiv.org/abs/1505.07818) / [JMLR 17](https://jmlr.org/papers/volume17/15-239/15-239.pdf)
- Sun & Saenko 2016. "Deep CORAL". [arXiv:1607.01719](https://arxiv.org/abs/1607.01719)
- Sun et al. 2016. "Correlation Alignment for UDA". [arXiv:1612.01939](https://arxiv.org/abs/1612.01939)
- DANN for DG (Springer Mach Learn 2023). [link](https://link.springer.com/article/10.1007/s10994-023-06324-x)

### Test-Time Adaptation
- Wang et al. 2021. "TENT: Fully Test-time Adaptation by Entropy Minimization". [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)
- "Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets". [arXiv:2602.00073](https://arxiv.org/abs/2602.00073)

### Mixup / Augmentation
- Zhang et al. 2018. "mixup: Beyond ERM". [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- "Embarrassingly Simple MixUp for Time-series". [arXiv:2304.04271](https://arxiv.org/abs/2304.04271)
- Amazon Science. "Improving Time Series Forecasting with Mixup". [link](https://www.amazon.science/publications/improving-time-series-forecasting-with-mixup-data-augmentation)
- Data augmentation for stock return prediction (ResearchGate). [link](https://www.researchgate.net/publication/365908543)

### Time-Series DG Benchmarks
- Wild-Time. [arXiv:2211.14238](https://arxiv.org/pdf/2211.14238)
- Domain Generalization in Time Series Forecasting (Deng 2024). [paper](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/deng-2024-domain.pdf) / [ACM TKDD](https://dl.acm.org/doi/10.1145/3643035)
- Latent Temporal Generalization for TS Forecasting. [arXiv:2412.11171](https://arxiv.org/html/2412.11171v1)

### Robust Loss
- Lin et al. 2018. "Focal Loss". [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- LightGBM Focal Loss implementation. [github jrzaurin](https://github.com/jrzaurin/LightGBM-with-Focal-Loss)
- "False awareness stock market prediction by LightGBM with focal loss". [ACM RACS 2022](https://dl.acm.org/doi/10.1145/3538641.3561502)

### Weight Averaging
- Izmailov et al. 2018. "SWA: Averaging Weights Leads to Wider Optima". [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- Cha et al. 2021. "SWAD: Domain Generalization by Seeking Flat Minima". [arXiv:2102.08604](https://arxiv.org/abs/2102.08604)

### Kaggle Writeups (Cross-Stock Generalization in Practice)
- Yirun. Jane Street 1st place writeup. [Kaggle](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) / [Numerai forum recap](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)
- Jane Street competition discussion 224348 (mixup + 4-fold day-isolation CV). [Kaggle](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348)
- HYD. Optiver Trading at the Close 1st place. [Kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
- "Gauging the Market" Optiver writeup (per-stock_id standardization). [Medium](https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0)
- Stock2Vec. [arXiv:2010.01197](https://arxiv.org/pdf/2010.01197)

### LOB-specific
- An Efficient Deep Learning Model for LOB (Siamese parameter sharing). [arXiv:2505.22678](https://arxiv.org/abs/2505.22678)
- Deep LOB Forecasting: A microstructural guide. [arXiv:2403.09267](https://arxiv.org/html/2403.09267v1)
- Representation Learning of LOB benchmark. [arXiv:2505.02139](https://arxiv.org/abs/2505.02139)
