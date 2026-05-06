# R12 — Ensemble & Stacking 调研（tabular 高频金融）

> 目的：在 iter_002 (Scheme C multi-horizon, LOSO h_10 +21.86) 之上找下一档增益。
> T6b（5-seed Scheme A 简单平均）→ -2.24，告诉我们"low-diversity ensemble 无效"；
> T10（feature-ensemble Scheme F = z-score+T3）→ 输给 iter_002，告诉我们"加噪声特征当 base 也无效"；
> T11（5 seed × 3 horizon × 强制 hyperparam diversity Scheme C）正在跑——这是当前最有希望的路。
>
> 本文整理 12 条 ensemble / stacking 方案，每条都对照我们硬约束和已知失败给出可行性判断。

---

## 0. 评测硬约束 vs ensemble 设计

| 约束 | 后果 |
|---|---|
| `date=0`、Predictor 跨调用无 state | ❌ online learning / sliding-window retrain / 跨样本累积统计的 ensemble 全部不可用 |
| 测试点顺序被打乱 | ❌ snapshot 间状态共享、跨样本一致 RNG seed 这些 trick 不能用 |
| sym 0–4 但可能含训练外股票 | LOSO 必须是首要验证；任何 ensemble 必须在 LOSO 下评估 |
| 推理时延 / submission zip 大小 | NN ensemble 可放 5–10 个；GBDT 可放 5–20 个；模型总量 < 100MB |

→ 适合我们的 ensemble 形式：**离线训好 N 个 model，把 N 个权重 + N 份 model 一起打包到 submission，predict() 里做加权平均**。

---

## 1. 12 条候选方案

每条按以下字段：**实现细节 / 来源 / 复杂度 / 适配度 / 期望增益**。
"期望增益" 单位 = LOSO sum (5 fold, h_10) 相对 iter_002 (+21.86) 的 Δ。

---

### 方案 1 — Heterogeneous algo ensemble (LightGBM + XGBoost + CatBoost)

**实现**：用同一 Scheme C 特征 + 同一 LOSO split + 同一 label，分别训 LightGBM、XGBoost、CatBoost（默认或轻调超参）。inference 时三者概率均值（或 logit 均值再 softmax）。

**关键**：三者的"正则、split criterion、目标编码、对 outlier 的反应"都不同 → 天然 diversity。在 LOB 高频任务上 CatBoost 的 ordered boosting 对短窗口特别稳，LightGBM leaf-wise 学得快但容易过，XGBoost 介于中间。

**来源**：Stephen Echessa "Stacking XGB+LGBM+CatBoost"（Medium）；Optiver Trading@Close 1st (HYD) 提到"CatBoost 与 LightGBM 互补"；XStacking 论文（ScienceDirect 2025）证实了三者堆叠比单模型平均高 3–6%。

**复杂度**：⭐⭐（低）。Scheme C 已有 LightGBM 路线，加 XGBoost / CatBoost 是抄一遍 wrapper；CatBoost CPU 慢但单 fold 几十分钟内能跑完。

**适配度**：✅ 完全合规（无 state、无 sym embedding）。比 T6b 的"单算法多 seed"diversity 大一个数量级——T6b 失败的核心就是 base 太相关。

**期望增益**：**+2 ~ +6**（保守）。如果三个算法的 OOF pearson 相关 < 0.85，期望加 stacking 能再 +2。

---

### 方案 2 — Stacking with linear meta-learner (LOSO-OOF)

**实现**：
1. L0 base：4–6 个模型（如 [LGBM-Scheme C, LGBM-Scheme A, XGB-Scheme C, CatBoost, MLPLOB, NGBoost]）。
2. 用 LOSO 5 fold 收集每个 base 的 OOF 概率 → shape (N_train, K_models × 3_classes)。
3. 训一个轻 meta-learner（Ridge / LogisticRegression / 单层 small LGBM with `num_leaves=15` `lr=0.01`）。
4. 推理：base 模型用全数据重训（or 取 LOSO 五个的均值）→ meta-learner 接 base output。

**关键防 leakage**：每个 fold 的 base 预测必须来自 **没见过该 fold sym 的模型**（即 LOSO 的 holdout sym）；这与"每个 sample 的 OOF prediction 必须来自没训练过它的 model"等价。

**来源**：Wolpert 1992 stacking；Lopez de Prado 《Advances in Financial Machine Learning》Ch.7（Purged K-Fold + Stacking）；scikit-learn StackingRegressor/Classifier；XStacking (2025)。Yao et al. 2018《Using stacking to average Bayesian predictive distributions》证明在 M-open（真模型不在候选里）下 stacking > BMA。

**复杂度**：⭐⭐⭐（中）。LOSO-OOF 收集 + meta-learner 训练流程要写一次；但完成后可以重复使用。

**适配度**：✅ 完美——LOSO-OOF 本来就是我们 T4/T5 的验证协议，只是把"评估"改成"把 OOF 当 meta feature"。Predictor 里 base 模型 + meta 层都是无 state。

**期望增益**：**+3 ~ +8**。Kaggle Optiver Vol 1st、Jane Street 1st 都用过；金融 tabular 上 stacking 通常比简单平均多 1.5–3 个百分点。

---

### 方案 3 — Hyperparameter-diverse bag (LightGBM 10 random configs)

**实现**：随机采 10 套 LightGBM 超参（`num_leaves ∈ [31, 127, 255]`、`lr ∈ [0.01, 0.05]`、`feature_fraction ∈ [0.5, 1.0]`、`bagging_fraction ∈ [0.6, 1.0]`、`bagging_freq=5`、`min_data_in_leaf ∈ [50, 500]`、`reg_alpha/lambda ∈ [0, 5]`），每套 + 不同 seed 训 → 取所有 OOF 后用 LOSO score 选 top-K（K=5），最终平均。

**关键**：T11 已经走这条路（5 seed × 3 horizon），但只覆盖 horizon 维度，没采 `feature_fraction/bagging_fraction/num_leaves`——这两个才是 LightGBM 多样性的主要来源。LightGBM 文档明确说 "use feature sub-sampling and bagging to increase unpredictability and reduce overfitting"。

**来源**：LightGBM Parameters-Tuning 文档；Kaggle ensembling guide (mlwave)；Optiver 2023 1st 提到 "large random search across LGBM configs is the cheapest diversity"。

**复杂度**：⭐⭐（低）。1 个脚本跑 10 个 config，每个 5 fold，约 10× 当前训练时间。

**适配度**：✅ 完全合规。是 T11 的 superset / 一般化版本。

**期望增益**：**+1 ~ +4**。比单 seed 平均强、但比 cross-algo (方案 1) 弱——因为 base 仍是同一算法。

---

### 方案 4 — Snapshot Ensemble（NN 路线）

**实现**：训单个 MLPLOB / DeepLOB（或 BiN-CTABL），learning rate 用 **cosine annealing with warm restart**：每 cycle 从 lr_max 降到 lr_min，cycle 结束 save 一个 snapshot。M=5 cycles → 5 个模型，inference 时 logit 均值。

**核心论文**：Huang et al. 2017 *Snapshot Ensembles: Train 1, get M for free* (ICLR 2017, arXiv:1704.00109)。Cosine LR schedule  
$\eta(t) = \frac{\eta_0}{2}\left(\cos\left(\frac{\pi \cdot \mathrm{mod}(t-1, T/M)}{T/M}\right)+1\right)$

CIFAR-10/100 上比单模型基线低 0.5–1.5% error，**训练 cost 等于单模型**。

**复杂度**：⭐⭐（低）。PyTorch `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` 一行；每 cycle 末 `torch.save(model.state_dict())`。

**适配度**：✅ 合规。M 个 snapshot 各自前向 + 平均 → predict() 内 stateless。

**期望增益**：**+1 ~ +3**（在 NN 路线上）。但前提是**我们先把单 NN 跑过 GBDT**，否则 snapshot 一组 NN 也打不过 LightGBM single。当前我们 NN 路线还没出 LOSO > +10 的成绩，snapshot 的边际价值不如先做 GBDT diversity。

---

### 方案 5 — SWA / Stochastic Weight Averaging（NN 路线）

**实现**：训 NN 用常规 schedule 到收敛，然后切到 **constant lr** 或 **cyclical lr**，每个 epoch (or cycle) 把当前权重做 running mean → SWA model。BatchNorm 层在最后用 SWA weights forward 一遍 train data 重算 running stats（一次 epoch 即可）。

**核心论文**：Izmailov et al. 2018 *Averaging Weights Leads to Wider Optima and Better Generalization* (UAI 2018, arXiv:1803.05407)。SWA finds **flatter minima**, 等价于 single-pass FGE。CIFAR/ImageNet 上 0.5–1.5% test acc gain，PyTorch 已经把它放进 `torch.optim.swa_utils`。

**复杂度**：⭐⭐（低）。PyTorch 内置：`AveragedModel`、`SWALR`、`update_bn`。

**适配度**：✅ 合规——SWA 只是把 N 个 checkpoint 平均成 1 个 weight，inference 与单模型一致。

**期望增益**：**+0.5 ~ +2**（NN 路线）。优点：**inference 成本不变**（只 1 个模型而不是 M 个）。可叠加 snapshot ensemble。

---

### 方案 6 — Multi-horizon stacking（横向 horizon 互补）

**实现**：对 h_10 任务做 stacking，但 base prediction 包括 **多个 horizon** 的概率：
```
meta_features = [
  p_LGBM_h5(up,flat,down),
  p_LGBM_h10(up,flat,down),
  p_LGBM_h20(up,flat,down),
  p_LGBM_h40(up,flat,down),
  p_LGBM_h60(up,flat,down),
]   # 共 15 维
meta_target = label_h10
```
meta-learner = Ridge 或 LightGBM（小）。

**为什么**：iter_002 已经显示 multi-horizon shared backbone 互相帮忙；stacking 是把这种 cross-horizon 信号显式当成 feature 给 meta，理论上比"shared backbone 内化"更稳健。h_5 概率提供 short-window 趋势，h_60 概率提供长趋势，h_10 任务用它们做 calibration。

**来源**：Briola 2023 LOB benchmark + Zhang 2021 multi-horizon DeepLOB-Attention；Konrad B. *Multi-layer Stack Ensembles for Time Series Forecasting* (Substack)；Multi-horizon LOB forecasting (arXiv:2105.10430)。

**复杂度**：⭐⭐（低）。Scheme C 已经训了多 horizon，本来就有这些预测；只需把它们 OOF 收集 + 训 meta。

**适配度**：✅ 合规。**且 scheme C iter_002 已经包含所有原料**，几乎零边际开销。

**期望增益**：**+1 ~ +4**。把 multi-horizon 信号从 implicit (shared backbone) 变 explicit (stacking feature)，至少不亏。

---

### 方案 7 — Trimmed / middle-60% averaging（Jane Street 1st trick）

**实现**：训 N≥5 个 base 模型（多 seed × 多 horizon × 多算法），inference 时对每个 (sample, class) 收集 N 份概率，**砍掉最高 20% 和最低 20%（或中位数 ±1 个），剩下取均值**。

**为什么**：单点 outlier（某个 seed 在某个 sample 上学歪）会被 trim 掉。比简单 mean 鲁棒，比 median 信息保留多。

**来源**：Yirun Zhang Jane Street 2020 1st place writeup："blending by concatenating models in a bag then taking the middle 60%'s average"。M. Kim Optiver 也提过类似 trick。

**复杂度**：⭐（极低）。`np.percentile(preds, [20, 80], axis=0)` clip 后 mean。

**适配度**：✅ 合规。**直接套在 T11 输出上即可**（如果 T11 的 simple mean 也不 work，trimmed mean 是第一个尝试）。

**期望增益**：**+0.5 ~ +2** 相对 simple mean。重要的是它**修正失败 seed 的污染**，对应 T6b 的失败模式。

---

### 方案 8 — Rank averaging（calibration-free 跨算法 blend）

**实现**：对每个测试点 t、每个类别 c，用每个 base 模型给出的概率 → 在 N 个测试点的 batch 内做 rank（小→大），rank 归一化到 [0,1]，然后对 K 个模型的 rank 取平均。最终 argmax 用 rank。

**为什么**：LightGBM softmax 概率 vs MLPLOB softmax 概率**量级差异大**，LGBM 容易出 0.95 但 NN 收敛到 ~0.6。直接 mean 会让"信号强但分布尖"的 LGBM dominate；rank average 把每个模型校准到同一概率空间。

**但**：阈值 post-processor (T4 keep) 依赖**绝对概率值**（max(p_0, p_2) > 0.5），rank 之后阈值需要重新 tune。

**来源**：MLWave Kaggle ensembling guide；KDnuggets *Using ensembles in Kaggle competitions Part 3*。多个 Kaggle 比赛冠军证明 cross-arch blend 时 rank avg > simple avg。

**复杂度**：⭐⭐（低）。需要 batch processing → 但**评测时打乱测试点**！每次 predict 调用是单个 sample，没法做 cross-sample rank。

**适配度**：⚠️ **部分合规**——只能在 Predictor 内部维护一个**当前训练集分布的 quantile lookup table**（提前算好）做 per-prob rank，而不是 cross-test-sample rank。这是简化版 isotonic-style calibration。

**期望增益**：**+0 ~ +2**。优先级低于 stacking，因为 stacking 自然解决 calibration mismatch。

---

### 方案 9 — Probabilistic ensemble (NGBoost) + uncertainty-gated thresholding

**实现**：把当前 LightGBM-Scheme C base 改成 NGBoost（base learner = decision tree, distribution = Categorical for classification with Dirichlet）→ 输出每类概率的 **均值 + 不确定度**。在 threshold gate 阶段用 `prob - k*std > τ` 而非 `prob > τ`，过滤"高均值但高方差"的不可信预测。

**核心论文**：Duan et al. 2020 *NGBoost: Natural Gradient Boosting for Probabilistic Prediction* (ICML 2020, arXiv:1910.03225)。natural gradient 修正普通 gradient 在分布参数空间的"reparametrization 偏差"。 

**为什么对我们有价值**：T4 已经验证"高置信度阈值 gating"是当前最大 lever（LOSO -22.10 → +6.45）。NGBoost 的 epistemic uncertainty 给阈值多一个维度——同样 prob=0.55，confident NGBoost 出手、uncertain NGBoost 不出手。

**复杂度**：⭐⭐⭐（中）。`pip install ngboost`，但 NGBoost CPU 训练显著慢于 LightGBM；3 类分类需要自定义 distribution（Categorical/Dirichlet）。

**适配度**：✅ 合规。NGBoost 是 stateless tree。

**期望增益**：**+2 ~ +5**（如果 NGBoost 单模型本身能接近 LightGBM；它可能落后 1–3 个百分点，要靠不确定度做后处理补回来）。

---

### 方案 10 — Cross-architecture blend with weight-optimized on OOF PnL

**实现**：N=5–8 个 base（[LGBM Scheme C, LGBM Scheme E, XGBoost, CatBoost, NGBoost, MLPLOB, BiN-CTABL]），收集 LOSO-OOF probs。用 scipy.optimize.minimize（**Nelder-Mead**）直接优化 ensemble weight w∈[0,1]^N（约束 sum=1，softmax parameterize）使得 weighted-blend 的 OOF **cum_pnl** 最大化。

**关键**：损失函数不是 cross-entropy，而是真正的评测指标（PnL with fee + threshold）。这就跳过了"代理 metric → 真实 metric 不一致"的问题。

**来源**：Kaggle ensembling guide §Weighted average via OOF；KazAnova Amazon Employee Access 比赛；Optiver Vol 1st Yakov 用过；Stacking Bayesian PSIS-LOO weighting (Yao 2018) 类似。

**复杂度**：⭐⭐⭐（中）。需要写 OOF PnL evaluator，但**我们已有**（T4 用过）；scipy 优化 5–8 维很快（< 1 分钟）。

**适配度**：✅ 合规。最终只是一组 weight + N 个 model。

**期望增益**：**+1 ~ +3** 相对均匀加权。注意**过拟合 OOF PnL 风险**——8 维上 LOSO 5 fold 容易把噪声当信号；应加 weight 上的 L2 正则，或限制 w_i ≥ 0。

---

### 方案 11 — Bagging-on-features Scheme C variants（防 noise feature 污染）

**实现**：基于 Scheme C 完整特征集（设 D 维），训 K=8 个 LightGBM，每个用 `feature_fraction=0.6` 和不同 seed → 每个 model 看到 ~0.6D 个特征。inference 时取 K 个 logit 均值。

**为什么**：T10 的失败教训是"加 z-score 给原 base 模型当 fixed feature 反而 dilute"。但**让多个 base 各自随机选一半特征**就不一样——每个 base 在它的子空间是最优的，组合后**等价于一个隐式的 boosting + bagging**。LightGBM 的 `feature_fraction` 是 column-bagging 内置实现。

**来源**：Breiman 2001 RF；LightGBM docs；scikit-learn `BaggingRegressor`。Optiver Vol 1st 用过类似的"feature subsampling × seed 多次训"。

**复杂度**：⭐（极低）。LightGBM 一行 `feature_fraction=0.6, feature_fraction_seed=k`。

**适配度**：✅ 合规。

**期望增益**：**+0.5 ~ +2**。比方案 3 (hyperparam-diverse) 多样性来源更结构化（特征维度），但天花板差不多。

---

### 方案 12 — Bayesian Hierarchical Stacking (sym-aware, 但 sym-agnostic friendly)

**实现**：Yao et al. 2022 *Bayesian Hierarchical Stacking*（arXiv:2101.08954）。每个 base 模型在每个 sym 上有不同的最优权重，但 hierarchically 共享一个超先验。**关键 trick**：把 sym 当 group covariate 但**不让 weight 直接用 sym ID**——用 sym 的 distribution-derived stats（amount_delta z-score、spread normalized 等）作为 group 协变量。这样推理时遇到训练外 sym，模型按它的特征分布"插值"权重。

**来源**：Yao, Pirš, Vehtari, Gelman 2022 *Bayesian Hierarchical Stacking: Some Models Are (Somewhere) Useful*; mc-stan loo package。

**为什么**：T8 已经诊断出 sym=2 distribution outlier；如果某个 base 模型对 outlier sym 表现差但对 typical sym 好，hierarchical stacking 会自动给 outlier sym 降权。

**复杂度**：⭐⭐⭐⭐（高）。需要 PyMC/Stan/numpyro，全 MCMC 拟合 weight posterior；但 base 已固定，meta 层可以小（≤ 200 维参数）。

**适配度**：⚠️ **要小心**——**如果 group covariate 含 sym ID 直接做 embedding 就违反 sym-agnostic 约束**。我们必须只用"sym 的 distribution-level 协变量"（z-score 类）。

**期望增益**：**+1 ~ +5**。理论上限高（特别是 sym=2 这种 outlier），但实现复杂度高 → 优先级低。

---

## 2. 失败方案排除（已知不适合我们）

| 方案 | 为什么排除 |
|---|---|
| Online learning ensemble (Algonomy / VW) | 评测无 state、无法 online update |
| Sliding-window retrain at inference | 同上，无 state；且每次 predict 是单点 |
| Sym-specific ensemble (per-sym model) | ❌ sym-agnostic 约束硬禁止 |
| 5 个独立 seed Scheme A simple mean | T6b 已验证失败（-2.24 vs single） |
| Z-score features + T3 base 直接 mean | T10 已验证失败 |
| Boosting-of-boosters（用 LightGBM 残差再训 LightGBM） | 已经在 base 里做了 boosting，外层 boosting 容易过拟合 |
| Stacking 用 sym ID 做 meta feature | ❌ 违反 sym-agnostic |
| Snapshot ensemble 之间共享 hidden state | ❌ 违反 stateless predictor |

---

## 3. 关于 T6b 失败的复盘（指导本次推荐）

T6b: 5-seed Scheme A simple mean → -2.24 vs single seed. **失败核心**：

1. **base 太相关**：同算法 + 同数据 + 同特征 + 同超参，只换 seed → OOF pearson > 0.97。Ensemble 增益 ≈ √(1-ρ²)/√N，ρ=0.97 时几乎为 0。
2. **fee 放大噪声**：相关 base 的小扰动加起来变成"出手次数微涨"，但 fee 是常数 → 净负。

**修复路径**（按多样性来源排序）：
- ✅ 不同算法（方案 1）→ 强 diversity
- ✅ 不同特征子空间（方案 11）→ 中 diversity
- ✅ 不同 horizon stacking（方案 6）→ 跨任务 diversity
- ✅ 不同超参（方案 3，T11）→ 弱 diversity
- ❌ 只换 seed → 几乎没 diversity

---

## 4. 如果只能试 3 个方案，我会选这 3 个

### 🥇 方案 2 — Stacking with Ridge meta-learner over LOSO-OOF

**理由**：iter_002 + T11 输出的 OOF 都已经现成，写一个 meta-learner 几乎零额外训练。Ridge / LightGBM-small meta 上手即用。最大 upside（+3 ~ +8），最小 downside（如果 meta 学不到东西，回退到 simple mean）。

**实施 checklist**：
1. 整合 [iter_002 LGBM Scheme C, T11 五个 hyperparam-diverse models, MLPLOB if available] 的 LOSO-OOF 概率（K base × 3 class = 3K-d）。
2. 在 LOSO 5 fold 上训 Ridge（α=1.0~10.0 sweep）或 LogisticRegression（C=0.5~5.0），meta target = h_10 label 三分类。
3. 推理时 base 用各自 5 fold 的均值或全数据重训版本 → meta 套上去。
4. 配合现有 threshold post-processor（T=0.5, δ=0.15）。

### 🥈 方案 1 — Heterogeneous algo: LightGBM + XGBoost + CatBoost ensemble

**理由**：T6b 已验证"同算法多 seed"无效；最便宜 diversity 来源就是"换算法"。CatBoost 的 ordered boosting 对短 horizon 噪声特征鲁棒，XGBoost reg 路线和 LightGBM 不同 → 三者组合 OOF 相关 < 0.85（经验值）。直接 simple mean 就有 +2~+6 增益，再做 stacking 上叠（方案 2）。

**实施 checklist**：
1. 复用 Scheme C 特征集 + LOSO split。
2. 各算法默认参数 + 1 轮 quick tune（CatBoost `iterations=2000, lr=0.03, depth=6`；XGBoost `n_estimators=1500, lr=0.03, max_depth=6, subsample=0.8`）。
3. simple mean → LOSO score → 比 iter_002 高即继续做 stacking；否则查 OOF 相关性。

### 🥉 方案 6 — Multi-horizon stacking with Ridge meta

**理由**：Scheme C iter_002 **已经训了所有 horizon**，OOF 现成；这条路是**纯免费**的——只需要重新组织已有预测做一次 meta-learner 拟合。如果有效，可作为方案 2 的特例 / 起步版本。

**实施 checklist**：
1. 从 iter_002 收集 5 horizon × 3 class = 15 维 OOF。
2. Ridge meta target = h_10 label。
3. 5 fold LOSO 评估，看 meta 是否学出"h_5 上涨概率 + h_60 下跌 → h_10 平"这种 cross-horizon 校正。

---

## 5. 综合实施顺序建议（一周内）

1. **Day 1（已有 T11 数据）**：方案 6 multi-horizon stacking（zero training cost，看 baseline）。
2. **Day 2–3**：方案 1 跑 XGBoost + CatBoost on Scheme C → simple mean → 2-algo / 3-algo blend LOSO。
3. **Day 4**：方案 2 stacking 整合所有上面的 base + Ridge meta；同时跑方案 7 trimmed mean 作 sanity baseline。
4. **Day 5（如果方案 2 增益 > +3）**：方案 10 Nelder-Mead 在 OOF PnL 上重 tune 权重，可能再 +1。
5. **Day 6+（保留）**：方案 3 (hyperparam-diverse bag) 和方案 11 (feature bagging) 是 T11 的扩展，留作后备。

---

## 6. 关键参考文献（详见 `r12_papers/`）

- Huang et al. 2017 *Snapshot Ensembles* (ICLR 2017) — `r12_papers/snapshot_ensembles_huang_2017.md`
- Izmailov et al. 2018 *Stochastic Weight Averaging* (UAI 2018) — `r12_papers/swa_izmailov_2018.md`
- Duan et al. 2020 *NGBoost* (ICML 2020) — `r12_papers/ngboost_duan_2020.md`
- Yao et al. 2018 *Stacking to Average Bayesian Predictive Distributions* (Bayesian Analysis) — `r12_papers/stacking_bayesian_yao_2018.md`
- Lopez de Prado 2018 *Advances in Financial ML* Ch.7 (Purged K-Fold for stacking) — `r12_papers/purged_kfold_lopez_de_prado.md`
- Kaggle Optiver Trading@Close 1st (HYD); Jane Street 1st (Yirun) — `competition_insights.md` & `competition_insights_models.md`
