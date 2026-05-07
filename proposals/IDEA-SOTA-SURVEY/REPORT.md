# IDEA-SOTA-SURVEY — 2026-05-07 调研报告

> **Worker 类型**: SOTA 调研型（无训练）
> **调研期间**: 2026-05-07
> **基于已有**: R31 (2026-05-06) / R33 (2026-05-07)
> **本报告追新**: R31/R33 截止后的新论文、新比赛方案、新 negative results
> **禁止**: iTransformer / PatchTST / TimesNet / TimesFM 类纯 TSF 方向

---

## 0. Executive Summary

| 结论 | 来源 |
|---|---|
| **CatBoost h60 5-seed（T46）with DE thresh = 12.64 LOSO**，与 iter_006 LightGBM (13.61) 相差 0.97，可直接 ensemble | T46 实验结果 |
| **DART h60（T50）seed42 raw argmax = 2.12**，仍在跑 5-seed；若 DE thresh 后 ≥ 12，则 LGB+CB+DART 三 GBDT ensemble 有望 14+ | T50 实验状态 |
| **GMADL（方向感知损失）**是 R31/R33 完全遗漏的关键方向，CatBoost 原生支持 custom_obj，预期 +1~+3 | arxiv 2412.18405 |
| **VWAP-to-mid deviation + Roll effective spread**：跨 3 篇顶级实证论文的顶-SHAP 特征，226-d 中不存在 | 2602.00776, 2403.09267, 2505.02139 |
| **LOBench 全局 z-score 归一化**（所有价格列共用 μ/σ，所有量列共用 μ/σ）修复 LOB 约束，可能比 T7/T44 的 per-feature z-score 更好 | 2505.02139 |
| **T44 R34 features、T45 Group DRO、T47 multi-horizon stacking** 均不超过 iter_006——三条路已被 negative result 确认 | T44/T45/T47 结果 |
| **T48 cross-sym pooling** inference 不兼容（需要跨 sample 实时统计），死路 | T48 结果 |

---

## 1. 新发现 Paper/Winner 列表（R31/R33 截止后新增）

> 标注 ⭐ = 在本项目有直接可用角度；每篇含引用、年份、数据集、关键 metric

### 1.1 LOBench — China A-Share LOB Representation Learning Benchmark ⭐⭐⭐

- **arxiv:** 2505.02139（2025-05-04）
- **Authors:** Muyao Zhong, Yushi Lin, Peng Yang
- **数据:** 中国 A 股深交所 5 只股票（平安银行、万科、贵州茅台、格力电器、广州香雪制药），2019年全年，3秒采样，10档 LOB，与本项目**完全同格式**
- **关键 metric:** SimLOB (representation learning) 在 fine-tune 场景（100 batch）超越所有 end-to-end 模型
- **核心发现（可直接应用）:**
  1. **Feature-wise z-score 会破坏 LOB 结构约束**（打乱价格档位单调性）——我们 T7/T44 的 per-feature z-score 可能是一个 bug
  2. **正确做法：Global z-score**：所有 bid/ask 价格列用同一 μ/σ 归一，所有 bid/ask quantity 列用同一 μ/σ 归一 → 保持 bid1 > bid2 > bid3 约束
  3. OFI 输入在 75% 情况下优于 raw LOB
  4. 数据集、代码全部公开：https://github.com/financial-simulation-lab/LOBench

### 1.2 GMADL — Generalized Mean Absolute Directional Loss ⭐⭐⭐

- **arxiv:** 2412.18405（2024-12-24）
- **Authors:** Jakub Michańków, Paweł Sakowski, Robert Ślepaczuk（华沙大学）
- **数据:** 高频算法交易资产（未公开具体 ticker），Transformer/LSTM/RNN 三种架构
- **关键 metric:** 所有架构 + GMADL 均优于 MSE-type 损失
- **核心（可直接应用）:**
  ```
  ℓᵢ = −(sigmoid(a · R_true · R_pred) − 0.5) · |R_true|^b,  a,b > 0
  ```
  - 当预测方向正确（R_true × R_pred > 0）：奖励，且大幅移动奖励更多
  - 当预测方向错误（R_true × R_pred < 0）：惩罚
  - 参数 a 控制对方向误差的敏感度，b 控制对幅度的权重
  - **对我们 3-class 的映射**：label=2→R_pred=+1，label=0→R_pred=-1，label=1→R_pred=0；|R_true| = |midprice_{t+60} - midprice_t| / midprice_t
  - LightGBM custom_obj + CatBoost custom_obj 均可实现（需要提供一阶/二阶梯度）

### 1.3 Explainable Patterns in Crypto Microstructure ⭐⭐

- **arxiv:** 2602.00776（2026-02）
- **Authors:** Bartosz Bieganowski, Robert Ślepaczuk（华沙大学）
- **数据:** Binance Futures 5 只 perp（BTC/LTC/ETC/ENJ/ROSE），1s 频率，2022-01 ~ 2025-10
- **模型:** CatBoost + GMADL 目标函数 + Bayesian (TPE/Optuna) 超参搜索 + rolling CV with purge window
- **SHAP 分析（重要！）:**
  - 三类特征跨所有 5 只资产保持稳定 SHAP 主导：
    1. **Order flow imbalance（OFI L1）** — 最强预测因子，单调效应但极端值有边际递减
    2. **Bid-ask spread** — 与预测能力负相关（spread 高 → adverse selection 风险高）
    3. **VWAP-to-mid deviation**（= buy/sell VWAP 偏离中间价的比率）⭐ — 非对称短期压力
  - 特征排名跨资产保持一致 → **scale-invariant microstructure** 成立
- **对我们的启示:**
  - VWAP-to-mid deviation **完全不在我们 226-d 中**，实现成本极低（1行公式）
  - 这个方向 LOBench + DeepLOB microstructural guide + crypto 论文 3 篇共同验证

### 1.4 Robust-GBDT — Nonconvex Robust Focal Loss for GBDT ⭐⭐

- **出处:** arXiv 2310.05067，2025 发表于 KAIS（Knowledge and Information Systems）期刊
- **Authors:** Zheng Chen et al.
- **数据:** 40 个 imbalanced 数据集（binary/multiclass/multilabel）
- **关键 metric:**
  - Binary：平均 +1.33%，最大 +8.15%（噪声+不平衡组合场景）
  - **Multiclass：平均 +0.51%，最大 +10.72%**
- **核心:** 证明了 GBDT 的 Newton step 仅需局部凸性，不需全局凸——允许使用非凸 robust focal loss
- **Robust Focal Loss 公式:**
  ```python
  # 比标准 Focal Loss 加了 outlier-robust component:
  # 在低置信度（γ大）下适当减少 hard-negative 的惩罚
  ```
- **可用 Python 包:** `pip install gbdtCBL` → https://github.com/Luojiaqimath/ClassbalancedLoss4GBDT
  - 支持 XGBoost/LightGBM/SketchBoost，三类损失：WCE/FL/ASL/ACE/AWE
- **对我们的启示:** 当前 T41 (class_weight_focal) 在跑，但用的是标准 focal loss。Robust Focal Loss 在 **noise + imbalance** 场景收益更显著，我们 label_60 本来就是 noisy（长 horizon label quality 差）

### 1.5 Improving GBDT on Imbalanced Datasets ⭐⭐

- **arxiv:** 2407.14381（2024-07）
- **Authors:** 中科院团队
- **数据:** 40 个不同数据集（binary 15 + multiclass 15 + multilabel 10）
- **关键发现（multiclass 直接相关）:**
  - WCE（weighted cross-entropy）在 **绝大多数场景最稳定**
  - Focal Loss 在某些超参设置下**低于** WCE baseline
  - ASL（Asymmetric Loss）在 multilabel 最强，multiclass 次之
  - multiclass 改进幅度 0.02% ~ 5.45%（binary 改进更大）
  - **关键实现**：`gbdtCBL` 包（同上）
- **对我们的启示:**
  - T41 (class_weight_focal) 在测试 WCE/Focal；本论文建议**先试 WCE**，Focal 需 careful tuning
  - 在我们 3-class 任务（label=1 占 60-80%）中，WCE 等价于对 label=0/2 加权 1.5-4x

### 1.6 Deep LOB Forecasting: Microstructural Guide（期刊版 2025）⭐

- **出处:** Quantitative Finance 期刊，July 2025（online）；arxiv 2403.09267
- **Authors:** Briola, Bartolucci et al.（LSE）
- **数据:** NASDAQ 股票，Level 2 LOB，多 ticker 多时间段
- **关键新发现（期刊版 vs arxiv 扩展了）:**
  1. **5-day rolling z-score**（= 2001 tick，3s/tick）显著优于 global normalization 和 within-window normalization；**这是我们还没有试过的 window 长度**
  2. **Tick size 分级** 决定预测能力：
     - Large-tick stocks（spread < 1.5× tick size）MCC = 0.29 @h=1 → 最可预测
     - Small-tick stocks（spread > 3× tick size）MCC = 0.11 → 难预测
     - 我们 5 只 sym 的 spread 特性（r30 已记录）可用此框架判断哪个 fold 预期好
  3. **LOBFrame** 开源框架，统一 preprocessing/modeling/evaluation

### 1.7 Understanding Temporal Shift in Tabular Data (TabReD) ⭐

- **arxiv:** 2502.20260（2025-02）
- **Authors:** Hao-Run Cai, Han-Jia Ye
- **数据:** TabReD benchmark（8个 real-world temporal tabular 数据集，包含 insurance/housing/ETA）
- **关键发现（与我们 LOSO OOD 强相关）:**
  1. **减少 training lag**（即让 validation set 在时间上紧邻 test set）比 random split 好 ~2.18%（std 从 154% → 16.7%）
  2. **GBDT（XGB/LGB/CatBoost）比 deep tabular 在时间漂移下更稳定**
  3. **Fourier 时间嵌入**（周期+趋势）可以帮助深度模型恢复时间漂移鲁棒性
  4. 实用 tip：validation 集的时序位移程度应与 test 集对齐（不是越近越好，而是"同等位移"）
- **对我们的启示:**
  - 我们 LOSO CV 设计（5 sym 轮流作为 test）不是 temporal split，但 **在每个 fold 内的 train/val 划分** 应用"training lag 最小化"原则
  - 当前 T34 CV v2 可能已在做类似的事，但值得确认

### 1.8 Stochastic Price Dynamics in Response to OFI ⭐

- **arxiv:** 2505.17388（2025-05）
- **Authors:** Chen Hu, Kouxiao Zhang（国联期货）
- **数据:** CSI 300 指数期货（中国 A 股期货）
- **关键发现:**
  1. **OFI 的 OU 过程均值回复速度（θ）** 是有效特征：快回复 → 噪声主导；慢回复 → 趋势信号
  2. OFI 系数在 5s 以上区间**持续 > 0.5**，说明 3s/tick 的我们需要**多 lag 的 OFI 而不只是 last-tick**
  3. **Horizon-matching 策略**：OFI window 的长度应与 label horizon 匹配（h60 label 用 60-tick OFI window）
  4. 与 Trade Imbalance、Kyle's λ 组合 > 单独 OFI
- **对我们的启示:**
  - T3 的 30-d MLOFI 用了 3 个固定 window，但没有"horizon-matched OFI"
  - 新特征：`OFI_W60`（60-tick OFI）对 h60 label 有理论动机

### 1.9 Covariate-Dependent Stacking (CDST) ⭐⭐

- **arxiv:** 2408.09755（2024-08，2025-09 更新）
- **核心:** 传统 stacking 的权重 (α_1, ..., α_K) 是固定标量；CDST 让权重成为协变量的函数：`α_k(x) = g_k(x)`（通过 local polynomial / kernel smoother 或小 NN 学习）
- **适用场景:** 数据生成机制在特征空间不同区域（或不同时间段）变化时，CDST > 固定 stacking
- **关键 metric:** 在 land price prediction（空间异质性）上 CDST > fixed stacking 显著
- **对我们的启示:**
  - 我们的 5-seed ensemble 用固定权重平均；如果 LGB 在高 volatility 区间好、CatBoost 在低 volatility 好（或反之），CDST 可提取这个差异
  - 实现：计算每个 sample 的 `realized_vol_20` + `spread_ratio`，作为 gate 网络输入，输出 (α_LGB, α_CB, α_DART)（softmax）
  - **完全 stateless**：gate 只看 100-tick window 内的统计量

### 1.10 LiT: Limit Order Book Transformer ⭐（NN 候选）

- **出处:** Frontiers in Artificial Intelligence，2025-10-13；King's College London
- **Authors:** Xiao, Ventre, Wang, Li, Huan, Liu
- **数据:** 多个 LOB 数据集，多 horizon
- **核心:** Structured patches → Transformer → LSTM（patch + attention + recurrence 三合一）
- **关键 metric:** 超越 DeepLOB/MLPLOB/TLOB 在多个 dataset 和 horizon
- **对我们的启示:**
  - 如果 NN 路线重新开启（T32 LOSO = -16.97），LiT 是最新的 SOTA baseline
  - 相比 MLPLOB（推荐 by R31 P9），LiT 更强但更复杂
  - **当前 NN 多次失败根因是 sym-specific 信号泄漏**，换架构不解决根因，优先级低

### 1.11 Jane Street 2024 1st Place 方案补充 ⭐⭐

- **来源:** GitHub `scaomath/kaggle-jane-street` 部分还原
- **架构:** 5 PyTorch + 3 Embedding + 3 AE + 1 TF-ResidualMLP = **12 个 base 模型**
- **关键 tricks（R33 未记录）:**
  1. **Hierarchical trimmed-mean ensemble:** 取 12 个模型输出的**中间 60% 平均**（淘汰最高/最低各 20%），在"busy days"用 50%
  2. **Utility function regularizer:** 每 10 epochs 额外用比赛的评估函数（零均值加权 R²）fine-tune 1 个 mini-step（lr=1e-3 × 0.1）
  3. **De-noised target:** 去除 feature covariance matrix 特征值后的 target（类似 PCA 去噪）
  4. **Adaptive threshold:** 用 `feature_64` 的平均梯度（arcsin scaling）决定 which-model-to-use 和 confidence level
- **对我们的直接借用:**
  - **Trimmed-mean ensemble** 取代 simple average：对 5 seeds 取 trim-20% mean（中间 60%）≈ 去掉最极端 seed
  - **Utility regularizer** = 我们 R10 S2 的 PnL-aware loss 思路（但是"每 N steps 插入一次"而不是始终使用）
  - De-noised target 需要协方差矩阵去噪，**实现复杂且 stateless 有疑问，跳过**

### 1.12 Efficient LOB DL on Chinese A-Share ⭐

- **arxiv:** 2505.22678（2025-05，中科院）
- **数据:** 中国 A 股 14 只国防行业股票（2021，3s/tick，10档，与本项目同格式）
- **关键发现:**
  - **Siamese 结构（bid/ask 对称 weight sharing）在 75% 场景优于非对称架构**
  - **OFI 输入在 75% 场景优于 raw LOB**（与 R31 P11 Siamese LOB 结论完全一致）
  - LSTM-MHA (attention) + Siamese + OFI 是该 benchmark 最优组合
- **对我们的启示:**
  - 与 R31 P11 一起强化了"OFI 信号 > raw LOB"的结论
  - 我们 226-d 中 OFI（MLOFI 30-d + EWMA intensities 24-d）已是主力，方向正确

### 1.13 Online HFT RBFNN with Feature Clustering ⭐（特征选择视角）

- **arxiv:** 2412.16160（2024-12，爱丁堡会议 2023 延迟发表）
- **数据:** 美国 mega-cap 20 只 NASDAQ/NYSE 股票，2022 tick 级 L1 数据
- **核心:** k-means 聚类自动确定 feature centroids，**MDI（Mean Decrease Impurity）** feature importance 在 36/60 场景最优
- **对我们的启示:**
  - 每 ~10 个 trading event 就有一次 regime change（clustered regime）
  - 我们 T33（long window）/ T26（domain randomization）等已在尝试 regime-aware 思路
  - **MDI feature importance 结合 regime-change 检测**可能指导特征剪枝

### 1.14 Stochastic Price Dynamics OFI（CSI 300 期货）⭐

- （见 1.8，已完整）

### 1.15 Global Normalization for LOB（LOBench 的推论）⭐⭐

- **来源:** 2505.02139 §3.1 + 图 3
- **核心:** 标准 feature-wise z-score 导致：(a) 价格档位单调性被打破；(b) best bid/ask 和远端档位的 z-score 偏离过大（因 best bid 方差本来就是最大的）
- **正确方式:**
  ```python
  # 不分列，统一归一化价格 & 量
  price_cols = [bid1..10, ask1..10, midprice1..10, spread1..10, ...]
  vol_cols = [bsize1..10, asize1..10, totalbsize, totalasize, ...]
  mu_p, sd_p = all_price_values.mean(), all_price_values.std()
  mu_v, sd_v = all_vol_values.mean(), all_vol_values.std()
  df[price_cols] = (df[price_cols] - mu_p) / sd_p  # 所有价格列同一标准化
  df[vol_cols] = (df[vol_cols] - mu_v) / sd_v       # 所有量列同一标准化
  ```
- **与 T7/T44 的区别:**
  - T7: per-feature window z-score（每列独立均值和方差）
  - T44: multi-W per-feature z-score（同上，但多窗口）
  - **LOBench**: 全局（across all price columns）统一归一化
- **对 fold 2 (ETF) 的预期效果:** fold 2 的 spread 虽然是其他 sym 的 1/10，但 mid price 在同一量级 → 全局归一化后 spread 信息保留（spread 相对于 mid 的比例）；per-feature 归一化反而把 fold 2 的 spread z-score 拉到和其他 sym 同一尺度（错误）

---

## 2. 现有实验 Negative Results 更新（R31/R33 之后新增）

| 实验 | 方案 | LOSO sum | 结论 |
|---|---|---|---|
| **T44** | R34 adaptive normalization (A1+A9+C9+B4 features, 350-d) | -4.20 (raw argmax sum, seed42) | ❌ fold 2 仍然 -8.67；per-feature adaptive norm 不能解决跨 sym scale 差异的根本问题 |
| **T45** | Group DRO (组内 DRO 重加权，修 fold 2) | 8.66 (DE thresh, vs iter_006 13.61) | ❌ DRO 过度修正 fold 2（-8.89→-2.80）但拉低其他 fold；总分下降 5 分 |
| **T47** | Multi-horizon stacking h30/60/120/240 | 12.36 (stacking+DE thresh) | ❌ 4 horizon 等权 stacking 低于 iter_006 (13.61)；但 h60 单独模型与 iter_006 接近 |
| **T48** | Cross-sym pooling features | 推理不兼容 | ❌ 跨 sym 实时统计（mean/std/rank）需要跨 batch 历史，违反约束 2 |
| **T35** | ReVol + SG (Scheme K) | 5.53 (single seed+thresh) | ⚠️ ReVol 排名正确 (R31 P17 预测兑现)；SG 对 LightGBM 边际贡献几乎为 0（NN 论文 claim 在 LightGBM 不成立） |
| **T46** | CatBoost h60 5-seed Plain | 12.64 (DE thresh) | ✅ 接近 iter_006；ensemble 候选 |

**关键教训:**
1. **Per-feature adaptive normalization 不能修复跨 sym scale 不一致**——必须是 global normalization（所有价格列共用 μ/σ）
2. **Multi-horizon stacking 增加了模型数量但引入了 h30/h120/h240 的噪声**——可以只加 h60 precision 最高的 horizon 的辅助目标，而不是 4 头等权
3. **Group DRO 在我们任务中副作用大于正作用**——5 只 sym 太少，DRO 的 group 估计很不稳

---

## 3. Top 5 Still-Untried-and-Promising Methods

### 🥇 Method 1: GMADL Custom Loss for LightGBM/CatBoost

**来源:** arxiv 2412.18405（GMADL）+ 2602.00776（Crypto CatBoost 验证）

**核心思路:**
```python
# LightGBM custom objective:
def gmadl_obj(y_pred, dataset):
    y_true = dataset.get_label()     # 0/1/2
    R_true = y_true - 1.0            # -1 / 0 / +1
    R_pred = y_pred[:, 2] - y_pred[:, 0]  # softmax 输出的 logit 差
    
    a, b = 2.0, 0.5  # 超参，a: direction sensitivity, b: magnitude weighting
    
    # ℓ = -(sigmoid(a * R_true * R_pred) - 0.5) * |R_true|^b
    sig = 1 / (1 + np.exp(-a * R_true * R_pred))
    loss = -(sig - 0.5) * np.abs(R_true) ** b
    
    # 一阶梯度 (对 y_pred 求导)
    grad = -a * (sig * (1 - sig)) * R_true * np.abs(R_true) ** b
    # 二阶 Hessian (对角近似)
    hess = a**2 * (sig * (1 - sig)) * (1 - 2 * sig) * R_true**2 * np.abs(R_true) ** b
    # 返回每个样本的 (grad, hess)
    return grad.ravel(), np.abs(hess).ravel() + 1e-8  # hess 必须正
```

**关键优势:**
- **对 label=1（平）天然冷淡**：R_true=0 → loss=0，不惩罚 "平" 的误判
- **对高 PnL 移动惩罚更重**：|R_true|^b 加权，大移动方向错误 → 惩罚更大
- **完全 stateless**：只依赖 (y_pred, y_true) per sample

**Minimum Viable Experiment:**
```bash
# 新实验 T53：GMADL obj on iter_006 架构
# 1. 实现 gmadl_obj + metric 函数
# 2. 单 seed=42，5-fold LOSO h60，Scheme C 226-d
# 3. 超参扫描 a ∈ {1, 2, 4}, b ∈ {0.5, 1.0}（6组，估计 6h）
# 4. 若 seed42 LOSO sum > 13.61 → 跑 5-seed
# 预计 LOSO Δ: +1 ~ +3
```

**兼容性分析:**
- ✅ sym-agnostic（loss 不看 sym ID）
- ✅ stateless（per sample 计算）
- ✅ 与 aug_a 完全兼容
- ✅ 与 DE thresh 兼容（thresh 在 softmax prob 上操作，不依赖 loss 函数）
- ⚠️ 3-class → multiclass GMADL 需要仔细设计梯度（上述是 1D 近似；准确版需要 one-vs-rest 展开）
- 🔧 **实施建议:** 先用 LightGBM 的 fobj 接口，用 OVR（3个 binary）实现后加权合并

**预期 LOSO 提升:** +1 ~ +3（参考 Crypto CatBoost 论文中的效果）

---

### 🥈 Method 2: Triple GBDT Ensemble (LGB + CatBoost + DART) with Trimmed Mean

**来源:** T46 (12.64) + T50 (pending) + Jane Street 1st place "trimmed-mean ensemble"

**核心思路:** 我们有 3 种不同的 GBDT 算法，每种 5 seeds = 15 模型：
- LightGBM GBDT（iter_006 backbone, 5 seeds, LOSO 13.61）
- CatBoost Plain（T46, 5 seeds, LOSO 12.64 with DE thresh）
- LightGBM DART（T50, 5 seeds, pending）

取 15 模型的 softmax 输出 **trimmed mean**（去掉最高/最低各 20% = 去掉 3 个，取中间 9 个）后再做 DE thresh optimization。

**Minimum Viable Experiment:**
```python
# 1. 收集 T46 (CB), T50 (DART) 的 parquet 预测文件
# 2. 与 iter_006 (LGB) 的 parquet 预测文件对齐
# 3. 逐样本 trimmed mean（trim 比例 = 2/15 ≈ 13%，取中间 11/15 或 9/15）
# 4. DE thresh 4D 重优化（用 T43/de_thresh.py 改写）
# 5. 预计运行时间 < 1h（全是后处理）
```

**预期 LOSO 提升:** +0.5 ~ +2（从 13.61 → 14.1 ~ 15.6）

---

### 🥉 Method 3: VWAP-to-Mid Deviation + Roll Effective Spread（新特征对）

**来源:** 2602.00776（Crypto CatBoost SHAP top-3）+ 2403.09267（microstructural guide）+ R34 B1

**核心思路:** 两个新 feature 均为 scale-invariant，在所有 LOB 论文中反复出现为 top SHAP，但**不在我们 226-d 中**。

**Feature 1: VWAP-to-Mid Deviation**
```python
# Buy-side VWAP 偏离中间价 (ratio，无量纲)
# 用订单流数据近似：weighted avg of trade prices vs mid
# 我们有 mb_intst（market buy intensity）和 volume/amount，可近似为：
vwap_buy_proxy = (mb_intst * close) / (mb_intst + 1e-8)  # 近似买方 VWAP
vwap_sell_proxy = (ma_intst * close) / (ma_intst + 1e-8)  # 近似卖方 VWAP
vwap_buy_dev_W = (vwap_buy_proxy.rolling(W).mean() - midprice) / (spread1 + 1e-8)
vwap_sell_dev_W = (vwap_sell_proxy.rolling(W).mean() - midprice) / (spread1 + 1e-8)
# W ∈ {10, 30, 60} → 6 维
```

**Feature 2: Roll Effective Spread**（来自 R34 B1，但未在 T44 中实测）
```python
def roll_spread(delta_mid, W=50):
    # 真实有效价差估计 = 2 * sqrt(max(-cov(Δmid_t, Δmid_{t-1}), 0))
    # 与 quoted spread (bid1-ask1) 的比值 = effective-to-quoted ratio
    sw_y = sliding_window_view(delta_mid[1:], W)
    sw_x = sliding_window_view(delta_mid[:-1], W)
    cov = ((sw_x - sw_x.mean(-1, keepdims=True)) * (sw_y - sw_y.mean(-1, keepdims=True))).mean(-1)
    eff_spread = 2 * np.sqrt(np.maximum(-cov, 0))
    return eff_spread / (spread1.values[W:] + 1e-8)  # 无量纲 ratio, sym-agnostic ✅
```

**兼容性分析:**
- ✅ 完全 sym-agnostic（VWAP deviation = ratio；Roll = ratio）
- ✅ stateless（causal window 计算）
- ✅ 与 aug_a 兼容
- 🔧 VWAP 精确需要 tick-level trade price，我们只有 amount_delta / volume_delta 近似，精度有限

**Minimum Viable Experiment:**
```bash
# 新实验 T54：VWAP-to-mid + Roll spread ratio 作为新特征
# 1. 在 build_features.py 加 8-10 维新特征（6 VWAP-dev + 3 Roll spread-ratio）
# 2. cache 重建（增量模式：加到 Scheme C 226-d 后面 → 235-d）
# 3. 单 seed=42 LOSO h60 测试
# 4. 若 LOSO sum > 13.2 → 跑 5-seed 全量
# 预计运行时间: 特征构建 2h + 训练 2h = 4h
```

**预期 LOSO 提升:** +0.5 ~ +2

---

### 🏅 Method 4: Covariate-Dependent Stacking（CDST）

**来源:** arxiv 2408.09755（2024-08，updated 2025-09）

**核心思路:** 当前 iter_006 ensemble 用固定权重（5 seeds 等权）。CDST 让每个 sample 的模型权重是观测协变量的函数：

```python
# 推理时（全 stateless）：
# x: 100-tick 窗口
realized_vol = np.std(log_returns_in_window) * np.sqrt(60)  # 年化 vol proxy
spread_ratio = current_spread / mean_spread_in_window       # spread 水平

# Gate network（训练时用 val set 上 5-seed predictions 学）：
gate_input = np.array([realized_vol, spread_ratio, time_of_day])  # 3-d
gate_weights = softmax(gate_net(gate_input))  # (n_models,)，可学习

# 加权组合：
y_pred = sum(w_k * pred_k for k, w_k in zip(preds, gate_weights))
```

**实现方案（轻量版）:**
```python
# 不用神经网络，用分箱版 CDST：
vol_bucket = pd.qcut(realized_vol, q=3, labels=['low', 'mid', 'high'])
weights_by_bucket = {
    'low':  [alpha_lgb_low, alpha_cb_low, alpha_dart_low],   # 从 val 上优化
    'mid':  [alpha_lgb_mid, alpha_cb_mid, alpha_dart_mid],
    'high': [alpha_lgb_high, alpha_cb_high, alpha_dart_high],
}
# 每个 bucket 单独做 DE thresh → 3 × 4D = 12D 联合优化
```

**兼容性分析:**
- ✅ sym-agnostic（gate 只看 window-level summary stats）
- ✅ stateless（gate input 从 100-tick window 计算）
- ✅ 与 DE thresh 可串联
- ⚠️ gate 过拟合风险：3个 bucket × 12-D DE = 多一层 optimization，需要在 hold-out fold 上验证

**预期 LOSO 提升:** +0.5 ~ +2（若 LGB 和 CB/DART 在不同 vol regime 下互补）

---

### 🏅 Method 5: LOBench Global Normalization（修 fold 2 的正确做法）

**来源:** 2505.02139 §3.1（全局 z-score 保留 LOB 约束，与本项目 setup 完全对齐）

**核心思路:** 见 §1.15 详细说明。Summary：

```python
# 当前 T7/T44 的错误（per-feature）：
df['bid1'] = (df['bid1'] - df['bid1'].mean()) / df['bid1'].std()
df['bid2'] = (df['bid2'] - df['bid2'].mean()) / df['bid2'].std()
# 问题：bid1 mean ≠ bid2 mean → 归一化后价格单调性 bid1 > bid2 被破坏

# LOBench 正确做法（global）：
price_cols = ['bid1','bid2',...,'bid10','ask1',...,'ask10','midprice','spread1',...,'spread10']
vol_cols = ['bsize1',...,'bsize10','asize1',...,'asize10','totalbsize','totalasize']
all_prices = df[price_cols].values.ravel()
all_vols = df[vol_cols].values.ravel()
mu_p, sd_p = all_prices.mean(), all_prices.std() + 1e-8
mu_v, sd_v = all_vols.mean(), all_vols.std() + 1e-8
df[price_cols] = (df[price_cols] - mu_p) / sd_p
df[vol_cols] = (df[vol_cols] - mu_v) / sd_v
```

**实现策略（causal version）:**
```python
# 为保持 stateless 且 causal，统计量从 train 集计算，固定后用于 val/test：
# （不是 per-sample，而是全局 train 统计量）
train_price_vals = train_df[price_cols].values.ravel()
GLOBAL_MU_P = train_price_vals.mean()
GLOBAL_SD_P = train_price_vals.std() + 1e-8
# 在 Predictor 里 hardcode 这个统计量 → stateless ✅
```

**兼容性分析:**
- ✅ stateless（推理时只用预计算的 global μ/σ）
- ✅ sym-agnostic（不分 sym 计算统计量）
- ⚠️ 与 aug_a 可能交互（aug_a 本身是 per-sample scale，global norm 之后 aug 的语义不变）
- 🔧 注意：global normalization 之后 spread 信号变弱（因 spread1 << bid1 in absolute）；考虑分两级：price 内部 global，vol 内部 global，spread 单独 global

**Minimum Viable Experiment:**
```bash
# 新实验 T55：Global LOB Normalization
# 1. 修改 build_features.py 的 normalization 为 global（train-level 统计量）
# 2. 重建 cache（完整 1200 parquet）
# 3. 单 seed=42 LOSO h60 + DE thresh
# 4. 重点观察 fold 2 (sym=2) 的 per-fold pnl（从 -8.67 能否 > -3）
# 预计运行时间: 3h
```

**预期 LOSO 提升:** 主要是修 fold 2 OOD；若 fold 2 从 -8.67 → -3，LOSO sum 可能 +5 ~ +10（但有不确定性）

---

## 4. Top 3 已验证 Anti-Patterns（本项目尚未运行前的错误方向，现已实证）

### 🚫 Anti-Pattern 1: Per-Feature Adaptive Normalization 不能修复跨 Sym OOD（T44 实证）

**原假设（来自 R34）:** window z-score + dual z-score + Kyle-Obizhaeva invariance 特征能让跨 sym 特征分布一致。
**实验结果（T44）:** seed42 LOSO sum = -4.20，fold 2 仍为 -8.67（vs iter_006 fold 2 = +1.48）。
**根因:** per-feature z-score 仅让每个列的均值=0，方差=1；但当 sym 2 的 spread 值域只有 sym 3 的 1/9 时，z-score 把不同 sym 的同一特征列对齐到 [0, 1] 后，模型看到的仍然是"同一 bucket 内的不同绝对水平"——tree split 阈值依然是 sym-特定的。
**正确方向:** Global normalization（所有 price 列共用 μ/σ）而不是 per-feature normalization（LOBench 结论）。

**节省时间的建议:** 不要再试 A 系列的变体（A3/A4/A5/A6/A10 等），它们都是 per-feature 的变种，根因相同。

---

### 🚫 Anti-Pattern 2: Multi-Horizon Equal-Weight Stacking 降低 Precision（T47 实证）

**原假设（来自 R33 Volkova/Yirun）:** 多 horizon（h30/60/120/240）辅助目标通过 stacking 可以提升 h60 性能。
**实验结果（T47）:** stacking+DE = 12.36，低于 iter_006 单 h60 (13.61)。
**根因:**
1. h30 和 h120/h240 的噪声引入：h30 label 更容易（α=0.1%, 平更少），但预测 h30 和 h60 的最优特征可能不同，stacking 没学到有益的互补
2. linear stacking（12维 → 3类）在 5 fold LOSO 下 capacity 有限，容易 underfit
3. DE thresh 4D 在 12-input 空间可能局部最优
**正确的 multi-horizon 思路:** 不是 equal-weight stacking，而是 "把其他 horizon 的预测作为特征（meta-feature）"，并让 h60 主模型自己学如何使用这些 meta-feature。

---

### 🚫 Anti-Pattern 3: Group DRO 在小 Group 数场景副作用大（T45 实证）

**原假设（来自 R32）:** Group DRO 能够修复 fold 2 (sym=2) 的 OOD overfit，通过对 worst-group loss 加强关注。
**实验结果（T45）:** LOSO DE thresh = 8.66（vs iter_006 13.61），虽然 fold 2 从 -8.89 改善到 -2.80，但 fold 1/3/4 均下降。
**根因:**
1. 只有 5 个 group（sym 0-4），DRO 的 worst-group re-weighting 方差极大
2. DRO 在 fold 2 high-cost 样本上过度 up-weight，导致模型 learns sym-2 specific features，反而让其他 fold 的 sym-agnostic 信号变弱
3. n_dro_updates 只有 3-4 次（K=50 触发），更新频率太低
**正确方向:** 修 fold 2 的根本路径是 **feature normalization（Global z-score，Method 5）** 而不是 loss 重加权。

---

## 5. 对比表：已做 vs 未做（关键候选）

| 方法 | 状态 | 实验 ID | LOSO 结果 | 推荐下一步 |
|---|---|---|---|---|
| LGB 5-seed h60 DE thresh | ✅ Done | iter_006 | **13.61（SOTA）** | 保持基线 |
| CatBoost 5-seed h60 | ✅ Done | T46 | 12.64 (DE) | 等 T50 DART 做 ensemble |
| DART 5-seed h60 | 🔄 Running (1/5 seed) | T50 | 待定 | 完成后做 triple ensemble |
| class_weight/focal | 🔄 Running | T41 | 待定 | 完成后与 GMADL 比较 |
| OvA binary ensemble | 🔄 Running | T42 | 待定 | 完成后评估 |
| CatBoost Ordered h60 | 🔄 Running | T49 | 待定 | CPU，等结果 |
| R34 stage 2 features | 🔄 Running | T51 | 待定 | 完成后看是否改善 fold 2 |
| Cross-sym mixup | 🔄 Running | T52 | 待定 | 评估 mixup 有效性 |
| **GMADL custom loss** | ❌ Not tried | T53 (提议) | 预计 +1~+3 | ⭐ **高优先级** |
| **VWAP-to-mid + Roll spread** | ❌ Not tried | T54 (提议) | 预计 +0.5~+2 | ⭐ **中高优先级** |
| **Global LOB normalization** | ❌ Not tried | T55 (提议) | 预计 +5~+10（fold 2 修复） | ⭐ **高优先级**（若 T44 根因诊断正确）|
| **Triple GBDT trimmed-mean** | ❌ Not tried（等 T50）| T56 (提议) | 预计 +0.5~+2 | 等 T46+T50 完成后做 |
| **CDST ensemble** | ❌ Not tried | T57 (提议) | 预计 +0.5~+2 | 等 T46+T50 完成后做 |
| **WCE/focal gbdtCBL** | 🔄 在 T41 | — | 待定 | 与 GMADL 比较 |
| Group DRO | ✅ Done | T45 | 8.66 ❌ | 不再尝试此路线 |
| Multi-horizon stacking | ✅ Done | T47 | 12.36 ❌ | 不再尝试等权 stacking |
| R34 per-feature norm | ✅ Done | T44 | -4.20 ❌ | 改为 global norm（T55）|
| Cross-sym pooling | ✅ Done | T48 | 推理不兼容 ❌ | 架构层面 banned |
| Online learning | — | — | 违反约束 2 ❌ | 永久 banned |
| Sym embedding | — | — | 违反约束 3 ❌ | 永久 banned |

---

## 6. 竞赛 Stacking 设计（OOD 场景的 Cross-Validated Stacker 设计）

> 针对任务问题 5：Kaggle 顶级方案在 OOD 比赛里的 stacking 设计

基于 R33 + 本轮新调研（CDST + Jane Street top solutions）：

### 6.1 在 OOD 比赛中 stacking 的正确姿势

**原则 1: Leave-One-Domain-Out Stacking（而不是 K-fold）**
```python
# 我们的 LOSO 框架本身就是 "leave-one-domain-out"：
# fold k = leave-out sym k，这是 stacking 的正确 CV 设计
# 正确：base model 在 fold k 上用 sym ≠ k 训练，对 sym=k 的 test 预测
# 错误：base model 在全量训练，然后对 test 做 stacking（leakage）
# 我们目前做法正确 ✅
```

**原则 2: Meta-feature 不能包含 domain-specific stats**
```python
# 正确：在 OOF 上用 stacker，输入 = {pred_lgb, pred_cb, pred_dart}（3-d logit）
# 错误：在 OOF 上加 per-sym 统计量作为 meta-feature（domain-leakage）
```

**原则 3: Stacker 应是极简的（避免 meta-overfitting）**
```python
# 选项 A: 线性 (Ridge/Logistic) stacker → 等价于 constrained ensemble weights
# 选项 B: CDST (covariate-dependent) → 允许 per-vol-bucket 不同权重
# 不推荐: XGBoost/LGB stacker → 在 5 fold LOSO 上容易过拟合 meta-features
```

**原则 4: OOD 场景下 trimmed mean 比 simple mean 稳**
```python
# Jane Street 1st place: 取中间 60% 的模型输出，去掉极端 20%
# 在我们 5-seed ensemble 中：取 seed=42,1,7,13 的 4 个（去掉 seed=100 的最极端一个）
# 更 robust 的版本：15 个模型（5-seed × 3-algo），取中间 9 个
```

### 6.2 LOSO OOD Stacker 的最优实践（结合 CDST）

```python
# Step 1: 收集所有 base model 的 OOF predictions
# {pred_lgb_s42, pred_lgb_s1, ..., pred_cb_s42, ..., pred_dart_s42, ...}
# 每个 = (n_test, 3) softmax 输出

# Step 2: 对每个 sample 计算 gate input
gate_feats = {
    'realized_vol_20': window_std(log_ret, 20),       # 短期 vol
    'spread_ratio': spread1 / spread1.rolling(50).mean(),  # spread burst
    'time_progress': session_progress,               # 盘口时间进度
}

# Step 3: 学 CDST gate（用 val set）
# 分 vol_bucket (low/mid/high) 学 3 套权重 (alpha_lgb, alpha_cb, alpha_dart)
# 各套权重用 DE 优化（联合 4D thresh + 3D weight = 7D）

# Step 4: 推理时
alpha = cdst_gate(gate_feats)  # lookup by vol_bucket
y_pred = sum(alpha_k * pred_k for k, pred_k in zip(models, preds))
label = thresh_decision(y_pred, T_up, T_dn, d_up, d_dn)
```

---

## 7. 调研来源 & 链接

| 论文/资源 | URL | 状态 |
|---|---|---|
| LOBench (2505.02139) | https://arxiv.org/abs/2505.02139 | ✅ 已读 |
| GMADL (2412.18405) | https://arxiv.org/abs/2412.18405 | ✅ 已读 |
| Crypto Explainable (2602.00776) | https://arxiv.org/html/2602.00776v1 | ✅ 已读 |
| Robust-GBDT (2310.05067/KAIS 2025) | https://arxiv.org/abs/2310.05067 | ✅ 已读 |
| GBDT Imbalanced Loss (2407.14381) | https://arxiv.org/html/2407.14381v1 | ✅ 已读 |
| Deep LOB Guide (QF journal 2025) | https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911 | ✅ 已读 |
| Temporal Shift Tabular (2502.20260) | https://arxiv.org/html/2502.20260v1 | ✅ 已读 |
| Stochastic OFI (2505.17388) | https://arxiv.org/html/2505.17388v1 | ✅ 已读 |
| CDST Stacking (2408.09755) | https://arxiv.org/html/2408.09755 | ✅ 已读 |
| LiT LOB Transformer (Frontiers 2025) | https://www.frontiersin.org/articles/10.3389/frai.2025.1616485 | ✅ 已读 |
| Efficient LOB DL (2505.22678) | https://arxiv.org/html/2505.22678v1 | ✅ 已读 |
| Jane Street 1st (scaomath GitHub) | https://github.com/scaomath/kaggle-jane-street | ✅ 已读 |
| gbdtCBL package | https://github.com/Luojiaqimath/ClassbalancedLoss4GBDT | 📦 可直接安装 |
| LOBench code | https://github.com/financial-simulation-lab/LOBench | 📦 可参考 |

---

## 8. 调研统计

- **WebSearch 次数:** 18
- **WebFetch 次数:** 12
- **新 Paper 数量:** 15 篇（在 R31/R33 31篇 基础上净增）
- **新 Winner Tricks:** 4（Jane Street 1st trimmed-mean / utility reg / de-noised target / adaptive thresh）
- **实验更新:** 查阅了 T41~T52 所有当前进行中/完成的实验状态
- **2024-2026 论文比例:** 100%（本轮追新，2024H2-2026H1）
