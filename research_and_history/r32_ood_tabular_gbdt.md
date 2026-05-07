# R32 — 2024-2026 OOD Generalization for Tabular GBDT in Finance / HFT

> Date: 2026-05-07 (R32 worker, opus-xhigh)
> Goal: 在 R10/R11/R12/R13/R20/R30/R31 之上**只追新**调研 — 重点是 **GBDT-specific OOD generalization**（不是 NN OOD），以及 **stock-level OOD** + **microstructure invariants**（sym-agnostic 在我们 setup 是硬约束）。
> Working hypothesis: iter_006 LOSO h_60 +13.61 已饱和（T39 marginal precision < break-even），standard 调阈 / 集成 / 校准都失败（T37/T38/T40），**必须从 OOD-overfit 根源攻**。
>
> ⚠️ 三条硬约束（CRITICAL_CONSTRAINTS.md）：
> 1. `date` 在评测时被置 0 — 不能当 feature
> 2. 测试点顺序被打乱 — Predictor 不能跨 `predict()` 维护 state
> 3. `sym` 0-4 但**可能含训练外股票** — 模型必须 sym-agnostic
>
> 任何方案如**违反 §3 (sym-agnostic at inference)** 直接淘汰；**§2 (stateless)** 几乎所有 LightGBM 方案天然满足。

---

## 0. TL;DR — Top-5 Picks（详细版见 `r32_top10_ideas.md`）

| # | 方案 | 类型 | 期望 LOSO h_60 | 工时 | sym-agnostic OK ? |
|---|---|---|---|---|---|
| 🥇 | **DART boosting + heavier L2** (LightGBM) | Training | +1 ~ +4 | 2 h | ✅ |
| 🥈 | **Group DRO via dynamic sample weight** (sym=group) | Training | +2 ~ +6 | 6 h | ✅（仅训练用 sym）|
| 🥉 | **Stable Feature Boosting (SFB) w/ sym envs** | Training | +1 ~ +5 | 1 day | ✅（同上） |
| 4 | **Integrated multi-level OFI (Cont-Cucuringu PCA)** | Feature | +1 ~ +3 | 4 h | ✅ |
| 5 | **VPIN (volume-clock toxicity) feature** | Feature | +0.5 ~ +2 | 4 h | ✅ |

**核心论点：饱和不是模型容量，而是 OOD 泛化失败。** 4/5 fold 模型 acc 低于 "全猜 1" baseline 表明 LightGBM 把 train sym 的 spurious correlation 学进去了。OOD literature 2024-2026 主要给两类武器：
- **Training-time invariance**（DRO / IRM / SFB / Mixup）：让模型只学 sym-stable 的关系
- **Sym-invariant features**（universal LOB features / scale-invariant ratios / VPIN / integrated OFI）：让"sym-stable 的关系"先 explicit 暴露给模型

我们应**两管齐下**。

---

## 1. 已读但 R31 未涵盖的核心范式

R31 已 list 31 paper（其中 P17 ReVol、P16 Order Book Filtration、P7 TradeFM、P14 Hybrid VAR+FNN、P10 HLOB、P11 Siamese LOB、P19 TimeAlign、P21 Return-Weighted Loss、P26 Multi-Class Calibration 与 OOD 间接相关）。
本次 R32 **追新 + 深入**：

### 1.1 GBDT-specific OOD methods（R31 缺失）

#### [N1] Stable Feature Boosting (SFB) — Eastwood et al. NeurIPS 2023 ⭐⭐
- **arxiv:** [2307.09933](https://arxiv.org/abs/2307.09933)（NeurIPS 2023 poster — R31 漏掉）
- **核心:**
  1. 把 features 分成 stable（cross-domain 和 label 关系不变）和 unstable / spurious（关系变）两组
  2. 训练 stable head 输出 pseudo-label
  3. 用 pseudo-label "校准" unstable head 在 test domain 的输出
  4. 在 (stable ⊥ unstable) ∣ y 假设下可证 asymptotically optimal
- **与 GBDT 的桥梁:** 可两阶段实现：
  - Stage 1 — train LightGBM-A on **only sym-invariant features** (queue imbalance, spread/tick ratio, return-vol normalized OFI 等) → 输出 ŷ_stable
  - Stage 2 — train LightGBM-B on full features，target = (y − α·ŷ_stable)，α 在 val 上调
  - Inference: y_pred = ŷ_stable + ŷ_residual_B
- **与我们关系:** 直接对应 "想用 sym-spurious feature 但不被它 OOD 拖死"。
- **风险:** 需要先找出"sym-stable" feature 子集（人工或基于 per-sym IC variance 自动）。
- **预期增益:** +1 ~ +5

#### [N2] LightMIRM — Light Meta-learned IRM with LightGBM (ICDE 2023 → 2024 应用)
- [LightMIRM PDF](https://blacksingular.github.io/papers/icde23-LightMIRM.pdf)；[ResearchGate](https://www.researchgate.net/publication/372666675_LightMIRM)
- **核心:** Loan default prediction 上把 IRM 与 LightGBM 嫁接 — LightGBM 做 feature extractor，对 IRM penalty 做"loss replaying"近似。
- **关键技巧:** IRM 的 penalty 项 `‖∇_w R_e(w)‖²|_{w=1}` 在 LightGBM 没有 batch 概念，所以本文用：
  - 把训练分 K 个 environment（per-sym group）
  - 每若干 boosting rounds 暂停，重算 per-env loss → softmax 反向加权 sample → 继续训
  - = 效果类似 Group DRO，但加了 invariance penalty 形式
- **与我们关系:** 直接可移植，LightGBM 路径，sym=env。
- **预期增益:** +1 ~ +3

#### [N3] DART Boosting — Vinayak & Gilad-Bachrach 2015（标准方法 R31 没专列）
- LightGBM 早就支持 `boosting_type='dart'`
- **机制:** 每轮训练时随机 drop 一部分前序树（drop_rate ~ 0.1），新树要 compensate dropped 树之和。这破坏 GBDT 的 "over-specialization"（晚期树只对少数样本起作用 — 我们 best_iter ~120 << 600 + 4/5 fold acc 低于 "全猜 1" 正是 over-specialization 信号）。
- **为什么 DART 治 OOD overfit:** 让晚期树承担更多分布重心，强制模型不依赖单棵或少数树的"个例修补"。
- **与 R31 P9 / P10 / etc 关系:** 完全 orthogonal — 是训练 hyperparameter 改动，10-line 配置变更
- **预期增益:** +1 ~ +4（非常 cheap）

#### [N4] Domain Randomization for Tabular Finance（综述 2024-2025）
- [A Survey of Data Augmentation in DG (Springer 2025)](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- [Domain Generalization Through DA Survey (MDPI 2025)](https://www.mdpi.com/2227-7390/13/5/824)
- **核心:** 对 tabular finance 数据，常用 DR 增强:
  1. **Feature-level Gaussian noise injection**（σ 与 feature std 成比例，p=0.5 触发）
  2. **Random feature dropout** (Cutout-style，p=0.1 把单 feature 设为 mean)
  3. **Mixup at sample level**（已在 R11 B 节）
  4. **Synthetic sample generation via diffusion** （太重，跳过）
- **与我们关系:** 1+2 是 30 行 Python；先生成 N_aug 条扰动样本拼到训练集即可。已知 GBDT 能用，但要小心 noise σ。
- **预期增益:** +0.5 ~ +2

---

### 1.2 Universal / Scale-Invariant Features（R31 已涉及但未深入）

#### [N5] Sirignano & Cont 2019 "Universal features of price formation" ⭐
- [arXiv 1803.06917](https://arxiv.org/abs/1803.06917) / Quantitative Finance 2019
- **核心:** 在 billions of US equity quotes 上训练 deep LSTM，**pooled 训练（不分 stock）的 universal model 显著优于 stock-specific 模型**，且**对训练外股票仍准**。
- **关键 insight:** 价格 formation 在 LOB 状态层面是 stationary universal 的 — feature → label 的关系**跨股票、跨时间、跨训练外股票一致**。
- **对我们的启示:**
  - 我们 GBDT 已是 pooled，方向正确
  - 但 LOB feature 必须以 **scale-invariant** 形式输入：`(price - mid) / spread` 而不是 raw price，`vol / mean_vol_in_window` 而不是 raw vol
  - **如果原始 feature 已是绝对量级**（e.g., raw price * 100），LightGBM split 阈值会绑死在某 sym 的 price scale 上 → OOD sym 上 split 失效。**这正是我们 sym=2 brittleness 的物理根源之一**
- **行动:** 审计现有 226 features，检查每条是否 scale-invariant；不是的改造（除以 mid 或 spread 或 EMA-vol）
- **预期增益:** +0.5 ~ +3（如果当前 feature 已大部分归一化，则下限低）

#### [N6] TradeFM — JPMorgan AI Research 2026 ⭐
- [arXiv 2602.23784](https://arxiv.org/abs/2602.23784)（R31 P7 已收录，本节深入"scale-invariant features"细节）
- **关键设计 [仍需读 PDF 核实]:**
  1. **Universal tokenization**: 把 (event_type, level_offset_to_mid, normalized_size) 三元组离散化为 token，跨 9K+ stock 共享词表
  2. **Scale-invariant features 的具体构造（推测，待核实）:**
    - `log(price_t / price_{t-1})` 而不是 `price_t - price_{t-1}`
    - `size_t / VWAP_window_size` 而不是 raw size
    - `level_idx` (0=best, 1=second, ...) 而不是绝对价格
    - `time_to_last_event` 用 quantile rank 而不是绝对秒数
- **与 GBDT:** 这些是普适 feature engineering 原则，可全部并入 LightGBM，不需要 524M Transformer
- **预期增益:** +0 ~ +2（与 N5 重叠）

#### [N7] Cont-Cucuringu-Zhang Integrated Multi-Level OFI 2023-2024 ⭐
- [Quantitative Finance 2023](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159) / [SSRN 3993561](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3993561)
- **核心定理:** 把多 LOB level 的 OFI 用 PCA 投影到一个 1D scalar（**integrated OFI**, iOFI）后：
  - 比 best-level OFI 解释力强很多
  - 多资产模型在加 cross-impact 项后**不再有额外信息** — 即 iOFI 是 cross-stock universal 的
- **公式（两步法）:**
  1. 在每个 LOB level k ∈ {1,...,K}，定义 OFI_k = ∑_{updates} sign(side) × Δqueue_k
  2. 取 ∑_k w_k × OFI_k，其中 w 是 PCA 第一主成分（在所有 stock 上联合 fit）
- **stateless?** ✅ — 100-tick 窗口内可纯函数计算 OFI_k 然后线性投影
- **与我们关系:** 我们已经有 multi-level OFI features，可能没**显式 PCA 投影**。加一条 iOFI 是 trivial。
- **预期增益:** +1 ~ +3

#### [N8] Microstructure Invariance — Kyle & Obizhaeva 2016, applied 2024 ⭐
- [Microstructure invariance in U.S. stock market trades (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1386418116303123)
- [Easley et al. 2024-2026 crypto extension](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- **核心定理（Kyle-Obizhaeva 2016 MMI hypothesis）:** 跨 asset / 跨 time，**在 business time 而不是 clock time、用 bet 而不是 trade、用 bet volatility 而不是 return volatility** 衡量后，risk transfer 和 transaction cost 的分布 invariant。
- **2024 实证:** Hou et al. (J Futures Markets 2024) 在 futures 上验证；Easley et al. 2024-2026 在 BTC/ETH 上验证 — VPIN 在 Toxic state 切换前 30-120s 系统性 spike，**跨币种共享**
- **与我们关系:** 这是 N5/N6/N7 的理论支撑 — 它解释了**为什么** scale-invariant feature works。具体可借的工具：
  1. **Bet definition:** 用最近 N=20 ticks 内的 net signed volume（buy − sell）做"bet"代理
  2. **Business time:** 用累计 volume 划分 bin，而不是 100-tick clock window 划分
  3. **VPIN:** 见 N9
- **预期增益:** 间接（启发其他 features）

#### [N9] VPIN — Volume-Synchronized PIN (Easley/López de Prado/O'Hara) ⭐
- 经典 [VPIN paper PDF](https://www.quantresearch.org/VPIN.pdf)；[2024-2026 实证证据](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)（crypto）
- **核心:** 把 trade 流按等量 volume 分桶（典型 50 桶），每桶内估 buy% / sell%，VPIN = E[|B − S| / V]。**直接代理 informed-trader 占比**。
- **stateless 实现:** 100-tick 窗口内就能近似 — 累计 abs trade-size 到 50 等额 bin（或 V/N bin），每 bin 用 tick rule（uptick=buy, downtick=sell, equal=split）打 sign，最后取 |sum| / V。
- **跨股普适性:** 经典 paper 在多 asset 验证；2024-2026 crypto 论文验证 spike-precedes-toxic-state 现象在 BTC/ETH 共享 — **典型 cross-asset universal toxicity feature**。
- **与我们关系:** 当前 226 features 里有 OFI / OBI / queue_imbalance，**未必有 VPIN**。加一条 VPIN_50bucket 即可。
- **预期增益:** +0.5 ~ +2

#### [N10] Tick-Size & Spread Aware Features — Briola et al. 2024
- [LOBFrame 综述 + benchmark study](https://link.springer.com/article/10.1007/s10462-024-10715-4)（Briola, Vidal-Tomás, Wang, et al. 2024）
- **核心实证:** Small-tick stocks（spread 经常 > 1 tick）和 large-tick stocks（spread ≈ 1 tick）的 LOB structure 性质完全不同 — 单一 model fit 跨 tick-size group 表现 brittle。
- **3 类 tick-aware feature 建议:**
  1. **`spread / tick_size`** ratio (≥ 1)
  2. **`queue_at_best / total_queue_in_top_5_levels`**
  3. **`mid_move_per_event_per_tick`** = mid 累计变化幅度 / events / tick
- **与我们关系:** 5 syms 在测评中可能 tick size 各异，是 sym-OOD 的**物理根因之一**。Add 3-5 explicit tick-invariant features 让 LightGBM split 不绑 tick scale。
- **预期增益:** +0.5 ~ +2

---

### 1.3 Loss / Optimization Robustness for OOD（R10 已涵盖 PnL loss，本节扩 robustness）

#### [N11] ε-Insensitive Pinball Loss — 2024 paper
- [Pinball boosting of regression quantiles (CSDA 2024)](https://www.sciencedirect.com/science/article/pii/S0167947324001117)
- [Robust Quantile Huber Loss (arXiv 2401.02325)](https://arxiv.org/html/2401.02325v2)
- **核心:** Pinball loss 在 GBDT 上有两个问题：
  1. **No curvature in [-ε, ε]** → tree split 找不到 informative cut
  2. **Heavy tail outliers** unbounded influence
- **解决:**
  - **ε-insensitive tube** 让 |residual| < ε 部分 loss = 0（增加 sparsity）
  - **Rescaled Huberized Pinball** 在 inner 区域 quadratic（增加 curvature）
- **与我们关系:** 当前我们做 multi-class classification，但 h_60 本质是 continuous → 可改 quantile regression（10-quantile 或 50-quantile head），用 RHPL 训练；最后用 quantile cuts 转 directional 信号（例如 q90 - q10 > threshold 取 long）。
- **预期增益:** +0 ~ +3（本质是 reframing 任务，风险中等）

#### [N12] Hybrid Linear Backbone + GBDT-on-Residual（R31 P14 重申，未实施）
- 已在 R31 但重要性被低估，这里加 OOD 视角:
- **OOD 益处:** Linear (Ridge / OLS) 在 5 个 OFI feature 上学 cross-stock 通用线性效应；LightGBM 仅学非线性 residual。OOD 风险主要在非线性部分被分摊到只有 ~30% 总信号，brittleness 显著降低。
- **预期增益:** +1 ~ +3

---

### 1.4 Test-Time / Inference-Time（受 stateless 约束极严，多数不适用）

> ❌ **完全不适用**: TENT / SHOT / online TTA — 所有依赖 batch statistics 累计。
> ❌ **不适用**: Conformal prediction across syms — 需要 "calibration set" 预选，但平台 shuffle 让我们没法保证 cal/val 分隔。
> ⚠️ **部分适用**: Conformal prediction with quantile regression LightGBM — 如果 val set 在训练后做 split-conformal，coverage guarantee 在评测 shuffle 下仍成立。但这只给 prediction interval，不直接提升 PnL。
> ✅ **唯一有意义的**: **per-window stateless normalization** — 每次 predict(x) 内对 100-tick window 自身做归一化（不跨 window 累计），相当于把 OOD 部分 burden 转移到 feature engineering。

---

### 1.5 Group Robustness — Group DRO 2024 进展

#### [N13] Group DRO — 2024 进展
- 原 [Sagawa 2019 (arXiv 1911.08731)](https://arxiv.org/abs/1911.08731)（R11 A 节）
- 2024 新进展：[Group DRO under group-level uncertainty (2509.08942)](https://arxiv.org/html/2509.08942v1)
- **改进点:**
  - 软加权 vs 硬 worst-group：在 5 个 sym 且 sym=2 outlier 极端时，硬 worst 会被 sym=2 拉爆
  - 推荐 **τ-sharpened softmax**: w_g = softmax(loss_g / τ), τ=0.5 → 比纯 ERM 更倾向 worst，比 worst-only 不极端
- **LightGBM 实现:** 每 K=200 boosting rounds 计算 per-sym training loss → reweight `sample_weight`
- **R11 提到但 NOT 实施。强烈建议做。**
- **预期增益:** +2 ~ +6

#### [N14] V-REx — 2024 follow-up
- 原 [Krueger 2021 (arXiv 2003.00688)](https://arxiv.org/abs/2003.00688)
- LightGBM **不直接支持** batch-level variance penalty（无 batch 概念），但可近似:
  - 每 K rounds 算 per-sym loss vector R = [R_0, R_1, ..., R_4]
  - 加一项 β·var(R) 到 objective 上 — 通过 sample_weight 间接：`weight_g ← weight_g × (1 + β·(R_g − mean(R))²)`
- **预期增益:** +1 ~ +3（与 N13 替代关系，不叠加）

---

### 1.6 Cross-Sym Mixup（R11 B 节，未实施）— 2024 重述

#### [N15] Cross-Sym Mixup — 2024 evidence
- [Embarrassingly Simple MixUp for Time-series (arXiv 2304.04271)](https://arxiv.org/abs/2304.04271)（R11 引用）
- 2024 GBDT 上有效证据:
  - "Data augmentation for stock return prediction" (ResearchGate 2024) — **mixup 在所有 case 都增 LightGBM stock prediction accuracy**
  - JaneStreet 1st place "fills empty space"
- **与我们关系:** R11 已设计未实施，这次明确推。
- **变体（结合 R32 视角）:**
  - 仅 mix **不同 sym 的 sample**（forces sym-invariant boundary）
  - λ ~ Beta(0.4, 0.4)
  - **Soft target** 处理：原 multi-class hard label 改 soft label，需 LightGBM custom objective
- **预期增益:** +1 ~ +4

---

### 1.7 Asymmetric Generalization Insight（重要新观察）

#### [N16] Asymmetric Generalization — Orderbook Feature Learning 2025/2026
- [arXiv 2510.12685](https://arxiv.org/abs/2510.12685)（intraday electricity but 普适）
- **关键发现:**
  - **Train on liquid → test on illiquid: works**
  - **Train on illiquid → test on liquid: fails badly** (AQL 6.56 → 80.15, 12x degradation)
- **对我们的启示:**
  - 看 sym 0/1/3/4 的 fold-out vs sym=2 fold-out 有无类似 asymmetry
  - 如果 sym=2 是"最 illiquid 或最 atypical"，则**训练时上调 sym=2 的样本权重 2-3x** 可能让模型从更难的环境出发，转移到其他 sym 时不掉。**这与 Group DRO 直接互补**。
- **预期增益:** 间接（指导 sample reweighting 策略）

---

### 1.8 已 R31 列出但因新视角值得重读

R31 的 P3 (Re(Visiting) TSFM) — generic foundation model zero-shot 在金融上不靠谱；**支持我们 GBDT 主路线，避免再尝试 Chronos/TimesFM**。
R31 的 P9 (TLOB / MLPLOB) — NN 是 alternative，不是 OOD 解药。
R31 的 P17 (ReVol) — T35 已实施 Scheme K 失败 LOSO -0.77，可能是 ε 域估计 noise 太大；**ReVol 思想本质是 N5/N6/N8 在 NN 中的版本，N5/N6 在 GBDT 上的纯 feature 实现是更便宜替代**。

---

## 2. 值得收藏但 OOD 优先级低

| 类别 | Paper | 不优先原因 |
|---|---|---|
| Foundation model | Kronos / FinCast / LOBERT (R31 P1/P2/P4) | Wrong granularity (daily) 或 self-supervised pretraining 太重 |
| Cross-sectional ranking | Multi-Factor Quant Trading 2025 (arxiv 2507.07107) | 需要 cross-sym view at inference, shuffle 约束下不易做 |
| GraphMamba / SAMBA | R31 P28 | bidirectional 不能 stateless inference |
| Asset embeddings | R32 N15 ref | 违反 sym-agnostic（学到的 embedding 用 sym ID 索引）|
| Diffusion / GAN data aug | NVF-DPGAN 2024 | 实施重，对 dirichlet shift 增益不确定 |

---

## 3. 行动建议（优先级排序见 `r32_top10_ideas.md`）

**第一阶段（cheap, 1-2 day）:**
1. DART boosting + 强 L2 (N3) — 配置改动
2. 加 5-8 条 sym-invariant features：integrated OFI (N7) + VPIN (N9) + tick-aware (N10) + log/ratio scale-invariant 改造 (N5/N6) — 纯 feature engineering
3. Cross-Sym Mixup augmentation (N15) — 数据合成

**第二阶段（medium, 3-5 day）:**
4. Group DRO LightGBM 动态 sample_weight (N13)
5. Stable Feature Boosting 2-stage 训练 (N1)
6. ε-Insensitive Pinball quantile head (N11)

**第三阶段（heavy, 1-2 week, 风险高）:**
7. LightMIRM-style IRM penalty + LightGBM (N2)
8. Linear backbone + LightGBM residual (N12)
9. V-REx variance penalty (N14)
10. Asymmetric reweighting based on sym difficulty (N16)

---

## 4. References & Sources

### NeurIPS / ICLR / ICML / Top Venues 2024-2025

- **SFB**: Eastwood et al., "Spuriosity Didn't Kill the Classifier" — [arXiv 2307.09933](https://arxiv.org/abs/2307.09933)（NeurIPS 2023）
- **Better by Default (boosted trees)**: NeurIPS 2024 — [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)
- **Group DRO**: Sagawa et al. 2019 — [arXiv 1911.08731](https://arxiv.org/abs/1911.08731)；2024 follow-up [Group DRO under uncertainty](https://arxiv.org/html/2509.08942v1)
- **V-REx**: Krueger 2021 — [arXiv 2003.00688](https://arxiv.org/abs/2003.00688)
- **LightMIRM**: ICDE 2023 — [PDF](https://blacksingular.github.io/papers/icde23-LightMIRM.pdf)
- **DART**: Vinayak & Gilad-Bachrach 2015；LightGBM doc — [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

### Microstructure / Universal Features

- **Sirignano & Cont 2019**: "Universal features of price formation" — [arXiv 1803.06917](https://arxiv.org/abs/1803.06917)
- **Cont, Cucuringu, Zhang 2023**: "Cross-impact of order flow imbalance" — [Tandfonline](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159) / [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3993561)
- **Kyle & Obizhaeva 2016**: "Microstructure invariance in U.S. stock market trades" — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1386418116303123)
- **Easley et al. 2024-2026**: "Microstructure and market dynamics in crypto" — [SSRN-4814346](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- **VPIN**: Easley/López de Prado/O'Hara — [quantresearch.org/VPIN.pdf](https://www.quantresearch.org/VPIN.pdf)
- **Hou et al. 2024**: "Futures trading costs and market microstructure invariance" — [J Futures Markets](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22496)
- **Briola et al. 2024**: "LOB-based deep learning models — benchmark" — [Springer AI Review](https://link.springer.com/article/10.1007/s10462-024-10715-4)

### Cross-stock / OOD Empirical (2024-2025)

- **Asymmetric Generalization**: arXiv 2510.12685 — [HTML](https://arxiv.org/html/2510.12685v2)
- **Better Inputs LOB (Wang 2025)**: arXiv 2506.05764 — [HTML](https://arxiv.org/html/2506.05764v2)（R31 P-Wang）
- **Robust Meta-Learning Zero-Shot Finance**: arXiv 2504.09664 — [HTML](https://arxiv.org/html/2504.09664)
- **Re(Visiting) TSFM**: arXiv 2511.18578 — generic TSFM zero-shot fails (R31 P3)

### Robust loss / regularization

- **Pinball boosting CSDA 2024**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167947324001117)
- **Robust Quantile Huber 2024**: [arXiv 2401.02325](https://arxiv.org/html/2401.02325v2)
- **Pinball-Huber boosted ELM 2024**: [Springer Applied Intelligence](https://link.springer.com/article/10.1007/s10489-024-05651-3)

### Data augmentation / Domain randomization 2024-2025

- **Embarrassingly Simple MixUp for TS**: [arXiv 2304.04271](https://arxiv.org/abs/2304.04271)
- **DG via Data Aug Survey 2025**: [Springer](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- **DG via Data Aug Survey MDPI 2025**: [MDPI](https://www.mdpi.com/2227-7390/13/5/824)

### HFT competitions 2024-2025

- **Optiver Trading-at-Close (2023-2024)**: 1st place hyd writeup — [Kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)；nimashahbazi LSTM/CNN — [GitHub](https://github.com/nimashahbazi/optiver-trading-close)
- **Jane Street RTM Forecasting (2024-2025)**: [Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)；2nd place Patrick Yam — [YouTube walkthrough](https://www.youtube.com/watch?v=lfzzPZZyzjE)；mlcontests 2025 — [State of ML Competitions](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

### Conformal / uncertainty (2024-2025)

- **Conformal Quantile Regression LightGBM**: [Kaggle](https://www.kaggle.com/code/syerramilli/lgbm-conformal-reg-part-2-prediction-intervals)
- **DISTRIBUTION-FREE INFERENCE FOR LIGHTGBM**: [arXiv 2507.06921](https://arxiv.org/pdf/2507.06921)

### Tabular / GBDT 2024-2025 representative

- **Better by Default (NeurIPS 2024)**: [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)
- **Tree-hybrid MLPs (T-MLP) 2024**: [arXiv 2407.09790](https://arxiv.org/html/2407.09790v1)
- **GBDT label noise robustness 2024**: [arXiv 2409.08647](https://arxiv.org/html/2409.08647v2)
- **Assets Forecasting LightGBM features 2024**: [arXiv 2501.07580](https://arxiv.org/abs/2501.07580)
- **Orderbook Feature Learning Asymmetric Generalization 2025/2026**: [arXiv 2510.12685](https://arxiv.org/abs/2510.12685)

### LOB representation learning 2024-2025

- **LOBench representation learning 2025**: [arXiv 2505.02139](https://arxiv.org/abs/2505.02139)
- **Deep Limit Order Book Forecasting 2024**: [arXiv 2403.09267](https://arxiv.org/html/2403.09267v1)（R31 引用）
- **Contrastive Asset Embeddings 2024**: [arXiv 2407.18645](https://arxiv.org/html/2407.18645v1)（违反 sym-agnostic, 仅 idea 借鉴）

### LightGBM specific docs

- **DART boosting in LightGBM**: [Parameters doc](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- **Sample weight / class weight**: [GitHub issue #6807](https://github.com/microsoft/LightGBM/issues/6807)

---

## 5. 总结：在我们 setup 下 OOD 的根本因果链

```
             [硬约束: sym 0-4 可能含训练外股票]
                            │
                            ▼
       [模型必须 sym-agnostic 在 inference]
                            │
                            ▼
   [问题: GBDT 容易学 sym-spurious correlation]
            │                │              │
            ▼                ▼              ▼
[Feature 不 scale-       [Loss 没有        [训练数据
 invariant → tree         sym-stability     里 sym 分布
 split 绑死 sym          约束              不平衡]
 specific scale]          (普通 ERM)]
            │                │              │
        [N5/N6/N7/        [N1/N2/        [N13/N14/
         N8/N9/N10]        N3/N4]          N15/N16]
       Universal          Training-      Group/Sample
       Features           level OOD      reweight
                          methods
```

**两条腿走路最稳:** Universal feature engineering（最便宜，可叠加）+ Training-level OOD method（次便宜，可叠加）。
**避免:** 单纯调阈值 / 校准 / 集成（T37/T38/T40 已证明饱和），单纯加 size / depth / horizon (T22 alpha101/191 / T33 long-window 失败)。
