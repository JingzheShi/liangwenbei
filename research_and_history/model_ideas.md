# 候选模型架构 + 训练策略清单

> 良文杯高频 LOB 中价方向预测任务专用清单。
> 每条 idea 含来源、复杂度、显存/推理估计、推荐优先级。
> 用户硬约束：**不推荐通用 ts forecasting 模型**（iTransformer / PatchTST / TimesNet / Informer / Autoformer / FEDformer），见末尾 §6。

任务背景速览（决定优先级）：
- 输入：100 tick × 154 feature；输出：3 类 × 5 horizon
- 训练：5 sym × 120 day，~240 万样本
- 评测：CPU 或 GPU ≤3h，模型 ≤2GB FP32
- 评分：cumulative PnL（手续费 0.01% 双边）
- baseline：DeepLOB CNN

---

## A. 模型架构候选

### A1. DeepLOB（Zhang 2018）
- **核心**：CNN(price-vol → ask-bid → levels) + Inception + LSTM + softmax，输入 100×40 raw LOB tensor
- **来源**：arxiv 1808.03668；https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- **复杂度**：medium；显存 ~2GB（batch 64）；推理 ~3ms/sample on GPU
- **场景**：LOB raw tensor baseline；mmpc_demo 已实现可复用
- **优先级**：**HIGH**（必跑，作为 NN 的 reference baseline）

### A2. DeepLOB-Attention / DeepLOB-Seq2Seq（Zhang 2021）
- **核心**：DeepLOB encoder + LSTM decoder（autoregressive 输出 multi-horizon）+ attention
- **来源**：arxiv 2105.10430
- **复杂度**：medium-high
- **场景**：**5 个 horizon 共享 backbone** —— 完美对应我们 5-head 任务
- **优先级**：**HIGH**（multi-horizon 任务首选 NN 架构）

### A3. BiN-CTABL（Tran 2018-2020）
- **核心**：Bilinear projection（沿时间 + 沿特征联合）+ Bilinear Normalization + Temporal Attention
- **来源**：arxiv 1712.00975 (TABL) / 2003.00598 (BiN)
- **复杂度**：medium；参数极少（k 量级）
- **场景**：**LOB benchmark 2023 中最稳的 NN**，跨数据集鲁棒度第 1
- **优先级**：**HIGH**（性价比之王，必跑）

### A4. MLPLOB（Berti 2025，TLOB 论文中的 baseline）
- **核心**：纯 MLP，feature-mixing MLP + temporal-mixing MLP（类似 MLP-Mixer），GeLU+LayerNorm
- **来源**：arxiv 2502.15757
- **复杂度**：low；训练快、参数少
- **场景**：在 FI-2010 上短 horizon 与 TLOB 接近，**是简单 NN 的强 baseline**
- **优先级**：**HIGH**（先于 Transformer 跑）

### A5. TLOB（Berti 2025）
- **核心**：dual attention（temporal + spatial），4+4 attention layer
- **来源**：arxiv 2502.15757；https://github.com/LeonardoBerti00/TLOB
- **复杂度**：high；显存 4-8GB
- **场景**：长 horizon（h=50/100）SOTA，对我们 label_60 可能有用
- **优先级**：**MEDIUM**（如果 MLPLOB 上限不够再试）

### A6. TransLOB（Wallbridge 2020）
- **核心**：dilated CNN + 2-layer Transformer encoder
- **来源**：arxiv 2003.00130
- **复杂度**：medium
- **场景**：FI-2010 强但跨数据集崩（87→59 F1）
- **优先级**：**LOW**（被 BiN-CTABL / TLOB 取代）

### A7. LiT（2025，Frontiers in AI）
- **核心**：structured patching + Transformer + LSTM，**抛弃 CNN**
- **来源**：https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- **复杂度**：medium
- **场景**：Binance crypto 上略胜 DeepLOB / TransLOB
- **优先级**：**MEDIUM**（patching 思路值得借鉴）

### A8. Siamese DeepLOB / MLPLOB / LSTM-MHA（2025）
- **核心**：bid 侧 / ask 侧 共享参数 encoder，利用 LOB 对称性 prior
- **来源**：arxiv 2505.22678
- **复杂度**：low（在已有 NN 上加 forward 一次）
- **场景**：**A 股专属验证**，在 14 只军工股上 75% 场景胜出
- **优先级**：**HIGH**（A 股证据 + 几乎免费的 inductive bias）

### A9. LightGBM (multi-class)
- **核心**：gradient boosting trees，每 horizon 独立 / multi-output
- **来源**：standard library
- **复杂度**：low；训练 30-60 min on CPU，推理 ms 级
- **场景**：**任何 LOB tabular 任务的默认 baseline**
- **优先级**：**HIGH**（必跑，第一个出 leaderboard 数字）

### A10. CatBoost (multi-class)
- **核心**：ordered boosting + per-feature cat encoding，金融数据更稳
- **来源**：https://catboost.ai
- **复杂度**：low
- **场景**：Optiver 2023 1st place 用它（权重 0.5）
- **优先级**：**HIGH**（必跑；金融场景 CatBoost 常 > LightGBM）

### A11. XGBoost (multi-class with custom focal loss)
- **核心**：与 LGBM 类似，custom objective 支持 focal/asymmetric loss
- **来源**：standard library
- **复杂度**：low
- **场景**：作为 ensemble 的第三个 GBDT，或不均衡 focal loss 训练时
- **优先级**：**MEDIUM**（与 LGBM/CatBoost 三选二即可）

### A12. ResNet-Tab / FT-Transformer
- **核心**：Yandex 2021 提出的 tabular DL，ResMLP 块 + feature tokenizer + Transformer
- **来源**：arxiv 2106.11959；https://github.com/yandex-research/rtdl-revisiting-models
- **复杂度**：medium；NN 参数 1-5M
- **场景**：把 100 tick 窗口 flatten + rolling stats → 几百维 tabular vector，跑 FT-Transformer
- **优先级**：**MEDIUM**（GBDT ensemble 的 NN 队友候选）

### A13. TabNet
- **核心**：sparse attentive feature selection，AAAI 2021
- **来源**：arxiv 1908.07442；https://github.com/dreamquark-ai/tabnet
- **复杂度**：medium，调参敏感
- **场景**：154 维 feature 上做 sparse selection，可作为 feature importance 工具
- **优先级**：**LOW**（被 FT-Transformer 取代；后续 benchmark 显示不及 GBDT）

### A14. Hybrid: NN encoder + GBDT classifier (Stacking)
- **核心**：DeepLOB / MLPLOB encoder 输出 latent → LGBM/CatBoost 做 final 分类
- **来源**：Optiver 2023 winner ensemble 思路；Jane Street 2020 (AE-MLP+xgb) Top1
- **复杂度**：medium；需要先训 NN 再训 GBDT
- **场景**：**两类模型互补的最直接做法**
- **优先级**：**HIGH**（如果 GBDT 和 NN 各自达到天花板，下一步必然是 stacking）

### A15. Conv1D + GRU + Transformer ensemble
- **核心**：Optiver 2023 1st 用的组合：CatBoost (0.5) + GRU (0.3) + Transformer (0.2)
- **来源**：Kaggle hyd writeup
- **复杂度**：medium-high
- **场景**：**直接 transferable 到我们任务**
- **优先级**：**HIGH**（被 Kaggle 实战验证的胜出 recipe）

### A16. Mamba / S-Mamba（2024）
- **核心**：Selective State-Space Model，线性复杂度 O(N)
- **来源**：arxiv 2405.16440 / 2403.11144
- **复杂度**：high（实现门槛）
- **场景**：未在 LOB 上有强 evidence；**实验性质**
- **优先级**：**LOW**（先把基础 GBDT/NN 调好再说）

### A17. AutoEncoder pretrain + MLP head
- **核心**：先 unsupervised AE 学嵌入，再 supervised MLP fine-tune
- **来源**：Jane Street 2020 1st place
- **复杂度**：medium
- **场景**：我们的 154 feature 是 explicit semantic 的，AE 收益较小
- **优先级**：**LOW**（不如直接 supervised；但作为正则化技巧可以试）

---

## B. 训练策略候选

### B1. Class weight (sklearn balanced)
- **核心**：按 class frequency 倒数加权
- **来源**：sklearn doc
- **复杂度**：trivial
- **场景**：label=1（平）占多数时立即缓解不均衡
- **优先级**：**HIGH**（一行代码，必加）

### B2. Focal loss (Lin 2017)
- **核心**：γ 调节 hard example 权重
- **来源**：arxiv 1708.02002
- **复杂度**：low
- **场景**：当 class_weight 不够、想强化对 hard sample 学习
- **优先级**：**HIGH**（NN 必试；LightGBM 也支持 custom focal）

### B3. Label smoothing (ε ≈ 0.1)
- **核心**：把 one-hot label 软化为 (1-ε) + ε/(K-1)
- **来源**：Szegedy 2016
- **复杂度**：trivial
- **场景**：高频金融 noise 大，硬 label 容易 overfit
- **优先级**：**HIGH**（与 focal loss 二选一或并用）

### B4. PnL-aware loss / Utility regularizer
- **核心**：在 cross-entropy 之外加 differentiable PnL 项；E[ pred_prob × signed_return - cost ]
- **来源**：Jane Street 2020/2024 winner; arxiv 2509.04541 (Finance-grounded optimization)
- **复杂度**：medium
- **场景**：**直接对齐评分目标**，比 PnL 上限关键
- **优先级**：**HIGH**（PnL 评分比赛的最大 lever）

### B5. Multi-task loss with uncertainty weighting (Kendall 2018)
- **核心**：5 horizon 各自 task，用 learned σ_i 自动平衡 loss 权重
- **来源**：arxiv 1705.07115
- **复杂度**：low
- **场景**：5 个 horizon 共享 backbone 时
- **优先级**：**MEDIUM**（先用等权 + 短 horizon 加权 0.7、长 horizon 加权 1.3 也够）

### B6. GradNorm（多任务自适应权重）
- **核心**：根据梯度范数自动平衡多任务 loss
- **来源**：arxiv 1711.02257
- **复杂度**：medium
- **场景**：5 个 horizon 训练不平衡时
- **优先级**：**MEDIUM**（uncertainty weighting 的替代）

### B7. AdamW + cosine schedule + warmup
- **核心**：lr 1e-3 → cosine decay，warmup 5% steps
- **来源**：std practice
- **复杂度**：low
- **场景**：所有 NN 训练
- **优先级**：**HIGH**

### B8. SWA (Stochastic Weight Averaging)
- **核心**：训练后期对 model weights 做平均
- **来源**：Izmailov 2018, arxiv 1803.05407
- **复杂度**：low
- **场景**：抑制 NN 过拟合（高频金融 noise 大）
- **优先级**：**MEDIUM**

### B9. Mixup / Cutmix（时序版）
- **核心**：随机加权混合两个样本
- **来源**：arxiv 1710.09412 / 1905.04899
- **复杂度**：low
- **场景**：金融时序 augmentation 慎用——可能破坏 micro-structure
- **优先级**：**LOW**

### B10. Random masking / dropout on features
- **核心**：训练时随机 mask 5-15% 输入 feature
- **来源**：常见技巧
- **复杂度**：trivial
- **场景**：154 维有冗余，masking 提升鲁棒性
- **优先级**：**MEDIUM**（NN 试一下）

### B11. Adversarial training (FGSM 微扰)
- **核心**：对输入加小扰动让模型鲁棒
- **来源**：Goodfellow 2014
- **复杂度**：medium
- **场景**：金融数据 noise 大，可能有用
- **优先级**：**LOW**

### B12. Online learning / fine-tune on tail
- **核心**：用最近 N 天数据 fine-tune（参考 Optiver 2023 1st place 12-day retrain）
- **来源**：Kaggle hyd writeup
- **复杂度**：medium
- **场景**：**我们 120 天数据，可以模拟"前 90 天 train + last 30 天 fine-tune"**
- **优先级**：**HIGH**（金融时序漂移强，必试）

### B13. 5-day rolling z-score normalization
- **核心**：用过去 5 天滚动 mean/std 归一化（按 sym × feature）
- **来源**：DeepLOB Microstructural Guide 2024 强烈推荐
- **复杂度**：low
- **场景**：处理非平稳 + 跨股票量级差异
- **优先级**：**HIGH**（必做的 preprocessing 改进）

### B14. Per-symbol normalization vs global
- **核心**：每只股票单独 normalize / 全局 normalize / sym embedding
- **来源**：DeepLOB universal feature 思路 + Briola 2023 跨股票崩盘
- **复杂度**：low
- **场景**：5 sym + 测试外股票泛化要求
- **优先级**：**HIGH**（决定跨 sym 泛化能力）

### B15. Sym embedding（把 sym 当 categorical feature）
- **核心**：embedding(sym) 加入输入或 conditioning
- **来源**：DeepLOB universal model 思路
- **复杂度**：low
- **场景**：5 sym 训练 → 测试可能含未见 sym
- **优先级**：**MEDIUM**（vs 完全 mask sym，需要 ablation）

---

## C. Ensemble & 后处理候选

### C1. Multi-seed averaging
- **核心**：同一架构跑 5-10 个 seed，平均预测概率
- **复杂度**：trivial（多训几次）
- **场景**：金融数据 variance 大，单 seed 不可靠
- **优先级**：**HIGH**（最简单的提分手段）

### C2. Stacking with LightGBM meta-learner
- **核心**：把 N 个模型的输出 prob 作为 meta-feature 喂 LGBM
- **来源**：Briola 2023 METALOB；Jane Street 2020 1st AE-MLP+xgb
- **复杂度**：medium
- **场景**：多个异构模型时
- **优先级**：**HIGH**

### C3. Snapshot ensemble (cyclic LR)
- **核心**：训练中存多个 checkpoint，平均
- **来源**：arxiv 1704.00109
- **复杂度**：low
- **场景**：训练预算紧时
- **优先级**：**MEDIUM**

### C4. Weighted average by validation PnL
- **核心**：找加权平均使 valid PnL 最大化
- **来源**：std Kaggle practice
- **复杂度**：trivial（grid search 权重）
- **场景**：3-5 模型 ensemble
- **优先级**：**HIGH**

### C5. Threshold tuning by PnL（不是 argmax）
- **核心**：分类时不用 argmax，而是按 PnL 在 valid 上 grid search 阈值
- **来源**：直接受比赛 PnL scoring 启发
- **复杂度**：low
- **场景**：**这个比赛要求 0/1/2 输出，所以阈值后处理是 critical**
- **优先级**：**HIGH**（PnL 评分比赛的最大 lever）

### C6. Calibration (Platt / Isotonic regression)
- **核心**：用 valid 集上的可靠图 (reliability diagram) 矫正 prob
- **来源**：std practice
- **复杂度**：low
- **场景**：NN softmax 通常 overconfident，校准后阈值更准
- **优先级**：**HIGH**

### C7. Asymmetric class threshold
- **核心**：对涨/跌/平用不同阈值（涨预测难度可能不对称）
- **来源**：std practice
- **复杂度**：low
- **场景**：手续费阈值决定 0/1 翻转的边际成本
- **优先级**：**HIGH**

### C8. Trade size / position sizing post-processing
- **核心**：连续 confidence → discrete output 的 mapping 用 PnL 优化
- **来源**：Optiver 2023 1st post-processing
- **复杂度**：medium
- **场景**：阈值 + size 二选一
- **优先级**：**MEDIUM**

---

## D. 跨 sym 泛化 / 训练外股票 候选

### D1. Domain-invariant features
- **核心**：用 OFI (Order Flow Imbalance), normalized spread 这类 sym-agnostic 特征
- **来源**：Cont & Stoikov 2014; R-feature worker 已经在做
- **优先级**：**HIGH**

### D2. Sym embedding（discriminative）vs mask
- **核心**：embedding(sym) 还是丢掉 sym？
- **来源**：DeepLOB universal model
- **优先级**：**HIGH**（必须 ablation）

### D3. Per-sym z-score + global model
- **核心**：每只股票自己 normalize，但模型权重 share
- **优先级**：**HIGH**

### D4. Domain Adversarial Training (DANN)
- **核心**：用 gradient reversal 让 encoder 输出不能区分 sym
- **来源**：arxiv 1505.07818
- **优先级**：**LOW**（高级技巧，先把基础做好）

### D5. Test-time adaptation
- **核心**：在测试期用 unsupervised 信号微调 BatchNorm 等
- **来源**：std domain adaptation
- **优先级**：**LOW**（评测限制下不易实施）

---

## E. 类别不均衡专项

### E1. SMOTE / random oversampling for minority class
- **来源**：std practice
- **优先级**：**LOW**（在金融时序里破坏时间结构）

### E2. Random undersampling majority class
- **优先级**：**LOW**（丢弃信息）

### E3. Class-balanced sampler in DataLoader
- **核心**：batch 里强制各 class 均衡
- **来源**：std practice
- **优先级**：**MEDIUM**（NN 试试）

### E4. Cost-sensitive learning (LightGBM scale_pos_weight)
- **核心**：直接给 LightGBM 不同 class 不同 weight
- **来源**：LightGBM doc
- **优先级**：**HIGH**

---

## F. 总精选清单：如果只能跑 3 个模型架构

按"覆盖面 + Kaggle 已验证胜出 + 最小复杂度"的原则：

### 🥇 #1: CatBoost / LightGBM ensemble + heavy feature engineering
- **GBDT 是 LOB tabular 任务的不会输的 baseline**。Optiver 2023 1st place 50% 权重在 CatBoost。
- 第一周必须出 PnL 数字。
- 训练快，可大量 ablation；推理快，符合 CPU 评测预算。
- **同时跑 LightGBM + CatBoost + XGBoost 3 个 GBDT 同模型族 ensemble，再 average。**

### 🥈 #2: DeepLOB-Attention（multi-horizon shared backbone）
- **专门为 5 horizon 多任务设计的架构**。
- 在 LSE 真实数据上验证过 universal model 跨股票迁移。
- 提供 NN 多样性（与 GBDT 互补）。
- **加 Siamese parameter sharing（A 股专属 prior）+ 5-day rolling z-score。**

### 🥉 #3: BiN-CTABL（attention bilinear，LOB benchmark 最稳 SOTA）
- **Briola 2023 benchmark 中跨数据集鲁棒度第 1（99.7%）**。
- 参数极少，训练快，泛化稳定。
- 与 DeepLOB 形成 2 个不同 inductive bias 的 NN，ensemble 多样性最大化。

### 加分（如果时间充裕）
- Stage 4: Hybrid Stacking — [3 GBDT + DeepLOB-Att + BiN-CTABL] → LightGBM meta-learner。
- Stage 5: PnL-aware threshold tuning + isotonic calibration（这一步通常单独提升 PnL 5-15%）。

---

## ⛔ §6 不推荐的方向（用户硬约束）

明确**不**作为模型候选的论文（即使学术热度高）：
- **iTransformer**（arxiv 2310.06625）—— 维度倒置 ts forecasting，玩具数据上 SOTA
- **PatchTST**（arxiv 2211.14730）—— patch 化时序 Transformer
- **TimesNet**（ICLR 2023）—— 时序周期分解
- **Informer**（AAAI 2021）—— sparse attention 长序列预测
- **Autoformer**（NeurIPS 2021）—— auto-correlation 替代 attention
- **FEDformer**（ICML 2022）—— 频域 Transformer
- 其他通用 ts forecasting / 仅在 ETT/Weather/Electricity 等 toy benchmark 上 SOTA 的工作

**理由**（用户提供）：这些方法在工业级带噪、低 SNR、有交易成本的金融数据上根本打不过 GBDT + 精心特征工程，只是在 ETT/Weather 这类玩具数据上刷指标。

如果调研中确实碰到，仅作为相邻文献提一句"用户明确不看好"，不展开、不写专题、不进入 short list。

---

## 引用清单

- DeepLOB (Zhang 2018): https://arxiv.org/abs/1808.03668
- DeepLOB-Attention (Zhang 2021): https://arxiv.org/abs/2105.10430
- TransLOB (Wallbridge 2020): https://arxiv.org/abs/2003.00130
- TLOB / MLPLOB (Berti 2025): https://arxiv.org/abs/2502.15757
- TABL / CTABL / BiN-CTABL (Tran 2017+): https://arxiv.org/abs/1712.00975 ; https://arxiv.org/abs/2003.00598
- LiT (2025): https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- LOB Benchmark (Briola 2023): https://arxiv.org/abs/2308.01915
- Deep LOB Microstructural Guide (Lucchese 2024): https://arxiv.org/abs/2403.09267
- Siamese LOB (2025): https://arxiv.org/abs/2505.22678
- Spacetimeformer LOB (2024): https://arxiv.org/abs/2409.02277
- TabNet (Arik 2019): https://arxiv.org/abs/1908.07442
- FT-Transformer (Gorishniy 2021): https://arxiv.org/abs/2106.11959
- Tabular DL is Not All You Need (Shwartz-Ziv 2021): https://arxiv.org/abs/2106.03253
- Focal Loss (Lin 2017): https://arxiv.org/abs/1708.02002
- Multi-task uncertainty weighting (Kendall 2018): https://arxiv.org/abs/1705.07115
- SWA (Izmailov 2018): https://arxiv.org/abs/1803.05407
- Finance-Grounded Optimization (Khubiyev 2025): https://arxiv.org/abs/2509.04541
- Optiver 2023 1st place: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- Jane Street 2020 1st place: https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1
- Optiver 2021 1st place: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
