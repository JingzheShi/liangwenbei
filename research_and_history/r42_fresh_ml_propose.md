# R42 — Fresh ML Literature Propose（loss / label / decision / model 创新）

> 作者：fresh-eye worker（不知道 iter_* / r3*.md / experiments 内容）
> 日期：2026-05-07
> 任务：从 ML 通用 SOTA（2020-2025）中抽取**对本题适用**的 disruptive ideas，标注是否已做。
> 信息源：仅 `docs/data_schema.md` + `submission/RULES.md` + `submission/ENV_NOTES.md` + `CRITICAL_CONSTRAINTS.md`，最后才看 SUBMISSION_LOG 速览表对照。
>
> 题目摘要：5 sym × 100-tick × ~154-D LOB → 三分类（涨/平/跌），评 5 horizon 取 max(cum_pnl)；
> fee=0.0001 双边；test 集 OOD（time + 含训练外 sym）；硬约束：no date / no sym embedding / no cross-call state。

---

## 0. 提示给后续 PM

- **本文不做 ranking 替代下面 r3*.md，只补"通用 ML 而非金融"的 fresh angle**。
- 凡涉及 sym categorical / state / date 的方案——**直接 PASS**（违反 CRITICAL_CONSTRAINTS）。
- 模型 ≤ 2GB，FP32，CPU 评测 16 核；3h 单轮上限 → 复杂 NN 必须 throughput 测过。
- 已知 SUBMISSION_LOG（速览）：iter_000 → iter_013，最强为 iter_013 +19.23（regression on Δmid + EV gate）。

---

## 1. Idea 池（带 SOTA 锚点 + 实施估算 + ✅/❌/⚠️ 已做标记）

> ✅ 表 SUBMISSION_LOG 速览表里出现过本质等价做法
> ❌ 表完全没看到等价做法
> ⚠️ 表速览表里出现过相关词但角度可能不同

---

### A. Loss / 训练目标创新

#### **1. Profit-Aware BCE / pairwise PnL ranking loss（PA-BCE, LambdaRank-style）** ❌

- **Source**: "Learn to Rank Risky Investors" (arXiv 2509.16616, 2025) + "Finance-Grounded Optimization for Algorithmic Trading" (arXiv 2509.04541, 2025)。LambdaRank 启发，把**两个样本之间的 P&L gap** 作为 pairwise weighting 加入 loss。
- **本题 fit**：现状 `regression on Δmid + EV gate` 等价于"先预测，再外接决策"。若直接把"我们最关心的 trades 排在最前"嵌入 loss，可减少 LOSO→Platform 的 -17 gap。
- **实施**：在 LightGBM 用 `lambdarank` objective + custom gain（gain = realized PnL of trade）。一行替换 + 把 dataset 按 (sym, date) 做 query group。预算：~0.5 day。
- **冲突检查**：不依赖 sym ID，只需要 query 切分，✓ 安全。

#### **2. SPO+ / 决策聚焦学习（Decision-Focused Learning, DFL）** ❌

- **Source**: Elmachtoub & Grigas, "Smart Predict-then-Optimize", arXiv 1710.08005 (Mgmt Sci 2022)；NeurIPS 2024 "Decision-Focused Learning with Directional Gradients"；JAIR survey 2024。
- **本题 fit**：iter_013 已经用了 EV gate，相当于**手工 SPO**。把 EV gate 写成可微 surrogate（soft-relu / Gumbel-softmax over {long, flat, short}），用 `−PnL` 作 backward，就能让模型 jointly 学 prediction + decision threshold。
- **实施**：在 PyTorch 上写 `pnl_surrogate(p_up, p_down, Δmid_pred, fee)` → 用 Adam 训。需要 LightGBM → small NN switch，但保留 GBDT 残差也行（"GBDT pred + small head trained with SPO+"）。预算：1.5 day。
- **冲突检查**：纯模型层面，不依赖 sym/date/state，✓ 安全。

#### **3. Pinball / quantile regression on Δmid（替代 L2 regression）** ❌

- **Source**: "Pinball boosting of regression quantiles" (Comp Stat & Data Anal 2024)；OpenReview "Beyond Pinball Loss" (ICLR 2024)。
- **本题 fit**：iter_013 用 L2 regress Δmid。换 pinball（α=0.5 等同 L1，α=0.1/0.9 估 quantile）能直接给"上 90% 分位数 / 下 10% 分位数"——比 EV 估计更适合 fee-aware gating（"只在置信高时出手"）。
- **实施**：LightGBM 直接支持 `objective='quantile', alpha=0.9` 和 `0.1`，跑 2 个模型 → 决策规则 `if q_low > 2·fee: long; if q_high < -2·fee: short`。预算：0.5 day。
- **冲突检查**：✓ 安全。

#### **4. PnL-aware composite loss（Sharpe + Max Drawdown）** ⚠️

- **Source**: "Finance-Grounded Optimization" (arXiv 2509.04541)：`PnLLoss + λ_RA RiskAdjLoss + λ_MDD MDDLoss`。
- **本题 fit**：cum_pnl 是评分，但 LOSO→Platform gap -17 提示模型在某 fold 集中赌，其它 fold 反弹——加 Sharpe 项可降低 fold-variance；MDD 项可避开"几个超大输 trade 拉低总 PnL"。
- **实施**：mini-batch 内算 `Sharpe = mean(pnl) / std(pnl)` 加进 loss；MDD 用 `min(cumsum) - max(cumsum)`。需可微 NN 实现。预算：1 day。
- **⚠️ 备注**：速览表 "T57_pnl_aware_loss/" 实验目录存在但尚未填进 LOG → 部分覆盖，但 Sharpe + MDD composite 看似未试。建议先确认 T57 范围。

#### **5. 软标签：基于 |Δmid|/α 的 regression-classification 混合** ❌

- **Source**: "label smoothing variants"（focal/CAMixup/RankMixup 思想）+ ordinal regression。
- **本题 fit**：当前 hard label = sign(Δmid > α)；但样本"刚好越过 α 0.001%" vs "远超 α 0.5%"信息差极大。把 label 设 `y = clip((Δmid − α)/(2α), -1, 1)` 软标签 + Smooth-L1，可让模型学到 magnitude，而不止 sign。
- **实施**：LightGBM regression 直接吃，决策时 `if y_pred > τ: long, ...`。这其实是 iter_013 的近亲 — 但 iter_013 用的是 raw Δmid 而非 normalized clip-soft。这里关键差异：clip 和 normalization 让 5 horizon 同尺度，可单模型 multi-horizon。预算：0.5 day。
- **冲突检查**：✓ 安全。

#### **6. Cost-sensitive class weighting based on horizon-specific α/fee ratio** ⚠️

- **Source**: 经典 cost-sensitive learning + 2025 bankruptcy ensemble (Springer s44163-026-01266-4)。
- **本题 fit**：每 horizon 的 α 不同（h5/h10=0.05%, h20/h40/h60=0.1%）但 fee 都是 0.01%。意味着 h5 信号-噪声比最差（α/fee=5），h60 最好（α/fee=10）。把 class weight = f(α, fee) 反映"激活成本"。
- **实施**：LightGBM `sample_weight` 按 `(|Δmid| − α) / fee` clip-positive 加权。预算：0.3 day。
- **⚠️ 备注**：速览表 T41 "class weight 全失败"——但那是上权 0/2 类，这里是按"超过 fee 的距离"加权样本，不同。可能值得再试。

---

### B. Label engineering

#### **7. Multi-horizon target via ordinal regression（horizon as continuous covariate）** ❌

- **Source**: ordinal regression literature；DLinear/iTransformer multi-horizon。
- **本题 fit**：5 horizon 模型独立训练浪费 4× compute。把 horizon h ∈ {5,10,20,40,60} 作为**输入特征**（log-h normalized），让一个 backbone 学全部 horizon。然后 inference 时 5× forward 给 5 个 h。预算：1 day（要重写 dataloader）。
- **冲突检查**：horizon 是 metadata，不是 sym/date/state，✓ 安全。

#### **8. Self-supervised contrastive pre-training（TF-C / SoftCLT）** ❌

- **Source**: TF-C (NeurIPS 2022, mims-harvard)；SoftCLT (ICLR 2024)；Drift-Resilient TabPFN (2024)。
- **本题 fit**：训练数据 1200 sessions，每个 session 几千 ticks → 几 M unlabeled samples 可拿。Time-Frequency Consistency 把同 LOB 切片的 time-domain 与 freq-domain (FFT) embedding 拉近，对 sym-OOD 极有用（freq pattern 比 time pattern 跨 sym 稳定）。Pre-train 一个 encoder → fine-tune 三分类头。
- **实施**：写一个小 conv encoder 在所有 sessions 跑 contrastive；fine-tune 阶段冻结/解冻。预算：3 days（NN 工程）。
- **风险**：速览表已 NN 失败（DeepLOB / iTransformer / PatchTST / TimesNet 全负）→ 但失败可能因为**没有 SSL 预训练**，直接 supervised 数据太少。SSL 是没试过的角度。

#### **9. Manifold Mixup（hidden-layer mixup, NOT input mixup）** ❌

- **Source**: ICML 2019 Manifold Mixup；2024 综述 arXiv 2409.05202；CAMixup / RankMixup 校准版。
- **本题 fit**：速览表 T52 "Cross-sym mixup fold 2 改善有限"——但那应该是 input-level mixup。**Manifold Mixup 在 NN 中间层混**，对 sym-OOD 极强（迫使 hidden manifold 平坦化）。
- **实施**：仅适用 NN 模型；如 idea #8 SSL 跑通后再叠 Manifold Mixup。预算：0.5 day（基于 NN）。

---

### C. Decision rule / Inference 创新

#### **10. Conformal prediction (split / online adaptive)** ❌

- **Source**: "Tutorial on Distribution-Free UQ" (Sage 2025)；CPTC (OpenReview 2024)；NixtlaConformal financial。
- **本题 fit**：iter_013 EV gate 用的是 raw L2 prediction → 没有"我有多自信"的概念。Split conformal 给每个预测一个 finite-sample 覆盖区间 `[ŷ − q̂_α, ŷ + q̂_α]`，决策规则改成 `if lower bound > 2·fee: long; if upper bound < -2·fee: short`，**比硬阈值更稳健**。
- **实施**：在 LOSO held-out fold 校准 nonconformity scores → q̂_α；inference 直接 `pred ± q̂`。预算：0.5 day。
- **冲突检查**：✓ 安全（q̂ 从训练数据估，evaluator 时无 state）。
- **特别注意**：CPTC for change points 适合本题（5 sessions × 5 sym 是天然 change-point）。

#### **11. TabPFN v2 / TabPFN-2.5（in-context tabular foundation model）** ❌

- **Source**: TabPFN v2 (Nature 2025)；TabPFN-2.5 model report (priorlabs.ai 2025) — 50k samples × 2k features，超过 GBDT。
- **本题 fit**：100×D 切片 flatten 后 ~15k features；TabPFN-2.5 支持 50k/2k → 取关键 200 features 后可直接喂。它是 in-context model，**无需训练**，直接 forward 给样本 → 输出 class probs。对 sym-OOD 极友好（meta-learning over 1M synthetic datasets）。
- **实施**：先 feature select → 100 tick 池化为 single row（mean/last/std/quantiles 50 维 × 4 = 200）→ TabPFN-2.5 直接 in-context predict。预算：1 day。
- **风险**：模型权重 + Python deps 可能 > 2GB → 提交前确认。
- **冲突检查**：TabPFN 不接收 sym/date 直接处理 LOB，✓ 安全。

#### **12. Drift-Resilient TabPFN（temporal distribution shift）** ❌

- **Source**: arXiv "Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts" (2024-2025, 编号 397198124)。
- **本题 fit**：题目天然有"date 0..119 但 OOD test 是新 date"——drift-resilient 版 TabPFN 内置 temporal drift handling。可作为 ensemble 一员补 GBDT。
- **实施**：同上 #11，但用 drift-resilient 变体。预算：1 day。

#### **13. NGBoost / XGBoostLSS / PGBM（probabilistic GBDT）** ❌

- **Source**: NGBoost (Stanford); "From point to probabilistic gradient boosting" (Eur Actuarial J 2025)。
- **本题 fit**：直接输出 (μ, σ) 两个量，决策规则 `if μ - k·σ > 2·fee: long`（k 是用 LOSO tune 的 gating 参数）。比 quantile regression 更系统。XGBoostLSS 在该综述里被评为最快的 probabilistic GBDT。
- **实施**：`pip install ngboost xgboostlss`，跑一遍。预算：0.5 day。
- **冲突检查**：✓ 安全。

#### **14. Test-Time Adaptation（TTA）via confidence maximization** ⚠️

- **Source**: arXiv 2106.14999 "Test-Time Adaptation by Confidence Maximization"。
- **本题 fit**：评测时 100-tick 窗口可计算"自信度 = max(softmax)"，若低于阈值则**临时 batch-norm refit** 或 entropy-min 1 step。但平台**禁止 cross-batch state**——所以 TTA 必须在每 100-tick 内独立完成。这意味着每个 batch 内部做 BN-affine reparam 是合法的，但 update 不能跨 batch persist。
- **实施**：用 GroupNorm/LayerNorm 替代 BatchNorm，无 running stat，本身就是"per-call adaptive"。预算：0.3 day（如果用 NN）。
- **⚠️ 警告**：CRITICAL_CONSTRAINTS §1.2 禁止跨 batch state——TTA 实现要小心 BN running mean。LayerNorm 可，BatchNorm 不可。

#### **15. Deceptive Risk Minimization (DRM) for OOD** ❌

- **Source**: deceptive-risk.github.io 2025（OOD 通过欺骗 distribution-shift detector 提升泛化）。
- **本题 fit**：sym + date OOD shift 显著（iter_013 LOSO→Platform gap -17）。DRM 用对抗性扰动让模型对 distribution-shift detector "看不出"训练 vs 测试，迫使学习真正的 invariant features。
- **实施**：训练阶段加 critic 网络 + adversarial loss。需 NN 框架。预算：2 day。

---

### D. 模型类（已做大类外的 fresh angle）

#### **16. TLOB / LiT — LOB-specific dual-attention transformer (2025)** ❌

- **Source**: TLOB (arXiv 2025, GitHub LeonardoBerti00)；LiT (Frontiers AI 2025)。
- **本题 fit**：LiT 在 ultra-short-term LOB forecasting 拿 F1=58.99 / Acc=59.03 优于 DeepLOB；TLOB 用 dual attention（time-attention + level-attention 分离）。比 DeepLOB / iTransformer 更**LOB-aware**。
- **实施**：fork TLOB 仓库 → 适配本题 154-D 输入（vs 原仓库 40-D LOB-only）→ supervised fine-tune。**关键不同 vs 速览表 NN 失败**：TLOB 是为 LOB 三分类专门设计，不是通用 TS forecaster。预算：3 day。
- **风险**：模型 + torch 可能慢 → 测 throughput < 3h。
- **冲突检查**：模型不接收 sym/date，✓ 安全。

#### **17. Multi-horizon Knowledge Distillation（TimeDistill style）** ❌

- **Source**: arXiv 2502.15016 "TimeDistill"。
- **本题 fit**：现状每 horizon 各自训一个 LightGBM。TimeDistill 思路：**先训一个大 multi-task NN/GBDT 当 teacher，蒸馏到 5 个 small student**（每 student 专攻一个 h）；蒸馏目标 = teacher 的 soft probs + ground truth 的 hard label。
- **实施**：teacher = iter_013 regression model；student = same arch但小一号 + KL on regression target。预算：1 day。
- **冲突检查**：✓ 安全。

---

### E. 多 horizon strategy

#### **18. Per-sample horizon selection（gate over 5 horizons）** ❌

- **Source**: mixture-of-experts / horizon-selection 经典。
- **本题 fit**：当前 ensemble = 单 horizon 提交。但事实上每个 sample 不同 horizon 期望 PnL 不同。训一个**轻量 gate** `g(x) = argmax_{h} E[PnL_h | x]`，inference 时只在 best horizon 出手，其它 4 个 horizon = 1（不出手，0 PnL 0 fee）。
- **实施**：先离线在 OOF 上算每 sample × 每 horizon 的 PnL → train gate（多分类）。inference 时按 gate 输出选 horizon。预算：1 day。
- **冲突检查**：✓ 安全；评分公式取 max(cum_pnl) 5 horizon → gate 不会损失评分。

#### **19. Multi-output regression with per-horizon EV gate（统一模型）** ⚠️

- **Source**: multi-task learning baseline。
- **本题 fit**：单 LightGBM 输出 5 个 Δmid（不同 horizon）→ 5 个独立 EV gate。比每 horizon 独立训省 5×。
- **实施**：LightGBM 不原生支持多输出 regression，但可在 PyTorch / XGBoost-multi 实现。预算：1 day。
- **⚠️**：速览表已有 multi-horizon (iter_002) 但用的是分类；多 horizon **regression** 单模型未见明确。

---

### F. 数据 / 增广

#### **20. Diffusion-based synthetic LOB augmentation（sym OOD 防御）** ❌

- **Source**: Tandfonline 2025 "Generation of synthetic financial time series by diffusion models"；Diffolio (Sci.Direct 2026)。
- **本题 fit**：训练数据只有 5 sym → OOD test 含训练外 sym。用 DDPM 在已有 5 sym 上拟合 LOB 切片分布，sample 出 sym-agnostic 合成数据扩训练集，希望提升 sym-OOD 泛化。
- **实施**：训 1D-DDPM on (100, 154) tensor → sample 100k 合成切片 → label 用 GBDT teacher 标 → 加进训练集。预算：5 days（很大投入）。
- **风险**：合成数据质量是赌博；mid/long-term high-investment idea，不优先。

#### **21. Curriculum learning by horizon difficulty** ❌

- **Source**: 经典 curriculum learning。
- **本题 fit**：先在最易 horizon (label_60，p_flat 最低 53%，信号 SNR 最强) 训 → 再 fine-tune 到 label_5/10。或先用 |Δmid| 大的样本（"明显的"）训，再加 |Δmid| 接近 α 的样本。
- **实施**：训练 loop 改 sample weight schedule。预算：0.5 day。
- **冲突检查**：✓ 安全。

---

## 2. 与 SUBMISSION_LOG（速览表）对照矩阵

| # | Idea | 已做？ | 速览表锚点 |
|---|------|------|----------|
| 1  | Profit-Aware BCE / Pairwise PnL ranking loss | ❌ | 无 |
| 2  | SPO+ / DFL（gradient through EV gate）       | ❌ | iter_013 用 EV gate 但是手工，非 e2e |
| 3  | Pinball / Quantile regression on Δmid       | ❌ | iter_013 用 L2 |
| 4  | PnL-aware composite (Sharpe + MDD)          | ⚠️ | T57_pnl_aware_loss 实验目录存在，未填回 LOG |
| 5  | Δmid-soft label (clip-normalized)           | ❌ | iter_013 raw Δmid |
| 6  | Cost-sensitive sample weight by α/fee gap   | ⚠️ | T41 上权 0/2 类失败，但角度不同 |
| 7  | Horizon-as-covariate ordinal regression     | ❌ | 无 |
| 8  | TF-C / SoftCLT self-supervised pre-training | ❌ | NN 直接 supervised 失败，无 SSL |
| 9  | Manifold Mixup（hidden-layer）              | ❌ | T52 cross-sym **input** mixup，不同 |
| 10 | Conformal prediction (split / online)       | ❌ | 无（速览表只有 Platt/Isotonic/Temp 失败） |
| 11 | TabPFN v2 / TabPFN-2.5                      | ❌ | 无 |
| 12 | Drift-Resilient TabPFN                      | ❌ | 无 |
| 13 | NGBoost / XGBoostLSS / PGBM                 | ❌ | 无 |
| 14 | Test-Time Adaptation (LayerNorm-only)       | ⚠️ | 速览表无 TTA，需查约束 |
| 15 | Deceptive Risk Minimization (DRM)           | ❌ | 无 |
| 16 | TLOB / LiT (LOB-specific transformer 2025)  | ❌ | DeepLOB/iTrans/PatchTST 失败但角度不同 |
| 17 | Multi-horizon KD (TimeDistill)              | ❌ | T47 multi-horizon stacking +12.36，但 stacking ≠ KD |
| 18 | Per-sample horizon-selection gate           | ❌ | 无 |
| 19 | Multi-output regression statt 5 sep models  | ⚠️ | iter_002 是 multi-horizon classification，非 regression |
| 20 | Diffusion-based synthetic LOB augmentation  | ❌ | 无 |
| 21 | Curriculum learning by horizon difficulty   | ❌ | 无 |

总计：**21 ideas**，**❌ 17 个未做** / **⚠️ 4 个相关但角度不同** / **✅ 0 个完全等价**。

---

## 3. Top 5 强烈推荐（未做、低风险、高 lift）

### 🥇 #2 — SPO+ / Decision-Focused Learning

- **Why top**：iter_013 已经 +19.23（手工 EV gate）；DFL 把 EV gate **可微化** 让 prediction + decision e2e 训练，理论上 strict ≥ 现状。
- **预算**：1.5 day
- **First step**：写 `pnl_surrogate(p_up, p_down, Δmid_pred, fee, label_true)` 函数（Gumbel-softmax over {long/flat/short}）→ 替换 LightGBM L2 obj 为 custom obj：`grad/hess of −E[PnL]`。
- **预期 lift**：+2~5 PnL（vs +19.23 baseline），主要来自缩小 LOSO→Platform gap (-17)。
- **风险**：custom LightGBM obj 实现繁琐；NN-based 更直接但要重训。

### 🥈 #10 — Conformal Prediction（split / online adaptive）

- **Why top**：iter_013 LOSO→Platform gap -17 是**置信度估计错位**的典型；conformal 给 finite-sample 覆盖保证。**Platt/Isotonic/Temp 失败是 calibrate softmax，但 conformal 是 calibrate prediction interval** — 不一回事。
- **预算**：0.5 day
- **First step**：在 LOSO 5 fold 留出 fold k 当校准集，算 nonconformity score `s_i = |y_i − ŷ_i|`，q̂_{0.9} 当 90% interval halfwidth → 决策 `if pred − q̂ > 2·fee: long; if pred + q̂ < −2·fee: short`。
- **预期 lift**：+3~6 PnL，主要来自更保守的 trade selection。
- **风险**：极小，可作 plug-in 不破坏 iter_013 模型。

### 🥉 #11 — TabPFN-2.5（in-context tabular foundation）

- **Why top**：sym-OOD 是本题最大痛点；TabPFN-2.5 在 1M synthetic datasets meta-learn → "新 sym 也是又一个新 dataset"，inherently OOD-robust。
- **预算**：1 day
- **First step**：feature engineering 把 100×D 池化为 single row（mean/last/std/q25/q75 各 25 features × 8 = 200）→ pip install tabpfn-2.5 → forward。
- **预期 lift**：+0~10 PnL（高 variance 但可能突破）；尤其与 LightGBM ensemble 后 diversity benefit。
- **风险**：模型大小（可能 > 2GB），dependencies 复杂；submission_pipeline 必须验证。

### #1 — Profit-Aware BCE / LambdaRank PnL ranking

- **Why top**：把 cum_pnl ranking 直接放进 loss，最 aligned 评分目标。
- **预算**：0.5 day
- **First step**：LightGBM `objective='lambdarank'`，query group = (sym, date)，gain = realized PnL of that trade。
- **预期 lift**：+1~4 PnL；可与 DFL 互补（lambdarank 是 ordinal aware，DFL 是 cardinal aware）。
- **风险**：需要正确定义 query group；评测时不出现"组"概念，但训练时分组学到的 ranking 仍 transfer。

### #18 — Per-sample horizon-selection gate

- **Why top**：评分公式 = max(cum_pnl over 5 horizons) → 没有 penalty 对于"在 4 horizon 不出手"；理论上 oracle gate 可达 max_h(perfect) 上限 ≈ +85 单 session。当前每 horizon 独立 → 浪费很多机会。
- **预算**：1 day
- **First step**：用 iter_013 OOF 算每样本每 horizon 的 PnL → train 5-class gate（哪个 horizon 收益最大）。
- **预期 lift**：+3~8 PnL，尤其对 long horizon。
- **风险**：gate 本身可能 OOD-sensitive；可保守：gate 只控"是否出手"二分类，不控 horizon-id。

---

## 4. 实施优先级（Day 0 → Day 7）

```
Day 0:    write pnl_surrogate (idea #2 first-cut, prototype on 1 sym)
Day 0.5:  conformal interval on iter_013 (idea #10) → quick submit candidate
Day 1-2:  TabPFN-2.5 prototype (idea #11), pool features, single-sym test
Day 2-3:  LambdaRank PnL (idea #1), define query group, LOSO test
Day 3-4:  multi-horizon KD (#17) + horizon-gate (#18) on shared backbone
Day 5-6:  Pinball quantile (#3) + composite Sharpe loss (#4 confirm vs T57)
Day 7:    SPO+ DFL full e2e (idea #2 finalized) — biggest investment
```

---

## 5. 显式 PASS 列表（对本题不适用 / 已知失败）

- **CatBoost Ordered + GPU + MultiClass**: 速览表 negatives 明确（库不支持，CPU 太慢）
- **DeepLOB / iTransformer / PatchTST / TimesNet (raw supervised)**: 速览表 negatives 已失败 → 但可被 SSL 预训练 (#8) 重新激活
- **Platt / Isotonic / Temperature scaling on softmax**: 速览表 T40 全部摧毁信号 → conformal (#10) 不一回事
- **alpha101 + alpha191**: 速览表 T22 collinear → r41 改进可能不同方向
- **sym embedding / per-sym normalization / sym-specific models**: CRITICAL_CONSTRAINTS §1.3 硬禁
- **date-derived features**: CRITICAL_CONSTRAINTS §1.1 硬禁
- **cross-batch hidden state in Predictor**: CRITICAL_CONSTRAINTS §1.2 硬禁

---

## 6. 来源（关键 paper / repo）

- Smart Predict-then-Optimize: https://arxiv.org/abs/1710.08005
- DFL Foundations 2024: https://arxiv.org/html/2307.13565v4
- Profit-Guided Loss 2025: https://arxiv.org/abs/2507.19639
- Finance-Grounded Optimization 2025: https://arxiv.org/pdf/2509.04541
- LambdaRank PA-BCE: https://arxiv.org/pdf/2509.16616
- Pinball boosting 2024: https://www.sciencedirect.com/science/article/pii/S0167947324001117
- Conformal time series tutorial 2025: https://journals.sagepub.com/doi/10.1177/25152459251380452
- CPTC change-point conformal: https://openreview.net/forum?id=HgLaVgCpCl
- TabPFN v2 / 2.5: https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report
- Drift-Resilient TabPFN: https://www.researchgate.net/publication/397198124
- TLOB transformer 2025: https://github.com/LeonardoBerti00/TLOB
- LiT transformer 2025: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- TF-C self-supervised: https://github.com/mims-harvard/TFC-pretraining
- SoftCLT ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/file/ccc48eade8845cbc0b44384e8c49889a-Paper-Conference.pdf
- Manifold Mixup survey 2024: https://arxiv.org/html/2409.05202v1
- TimeDistill 2025: https://arxiv.org/html/2502.15016v1
- NGBoost / XGBoostLSS / PGBM survey 2025: https://link.springer.com/article/10.1007/s13385-025-00428-5
- TS-OOG survey 2025: https://arxiv.org/html/2503.13868v3
- Deceptive Risk Minimization: https://deceptive-risk.github.io/

---

## 7. RESULT

```
RESULT: task=r42_fresh_ml metrics={n_ideas=21, n_undone=17, top5=[#2 SPO+/DFL, #10 Conformal, #11 TabPFN-2.5, #1 LambdaRank PA-BCE, #18 horizon-gate]} notes=focus on SPO+/conformal/TabPFN to close LOSO→Platform -17 gap; SSL+TLOB to revive NN angle; lambdarank/horizon-gate as quick quick wins
```
