# 量化/HFT 实战 NN 架构调研

> 任务：良文杯 LOB 中间价方向预测，h=60，5 sym，359-d 特征
> 调研日期：2026-05-30
> 当前最佳 baseline：MLP 4-layer 平均 ~34.4 test_pnl（5 seed）

---

## 0. 排除清单

### 0A. 用户明确拒绝（通用 TS forecasting，实测无效）

| 架构 | 原因 |
|---|---|
| TimesNet | 日频周期建模，LOB tick 不适用 |
| iTransformer | 维度倒置 Transformer，玩具数据 SOTA，金融无效 |
| PatchTST | 长周期 patch，tick 频率下 patch 退化 |
| N-BEATS / N-HiTS | 可解释 TS，无金融归纳偏置 |
| ModernTCN | 实测 avg test_pnl ≈ 0.77（全5 seed），完全失败 |
| TimeKAN | 实测 avg test_pnl ≈ 0.05，完全失败 |
| TimeMixer | 实测 avg test_pnl ≈ 31.4，勉强（接近 MLP 但没有超过） |

### 0B. 已在 nn_arch_ablation 实测（5 seed 平均）

| 架构 | avg test_pnl | vs MLP baseline | 结论 |
|---|---|---|---|
| **MLP** (256/128/64) | **34.44** | baseline | ✅ 主线 |
| MLP_deep (256/256/128/128/64) | 34.17 | −0.3 | ≈ baseline，深度无益 |
| MLP_wide (512/256/128) | 33.00 | −1.4 | 略差 |
| FT-Transformer (d_token=16, depth=2) | 24.63 | −9.8 | ❌ 差 |
| AutoInt (d_token=12, depth=2) | 28.86* | −5.6 | ❌ 差（s1 OOM） |
| GatedMLP | 27.56 | −6.9 | ❌ 差 |
| MLP-Mixer | ~30.2 (2 seeds) | −4.2 | ❌ 差 |
| ResMLP | 26.72 | −7.7 | ❌ 差 |

### 0C. 正在测试（cross_sym_nn worker 41c7e76c，截至调研时）

| 架构 | 状态 | 当前 partial 结果 |
|---|---|---|
| CrossConcatMLP (5×359 flatten) | s0,s1 done | avg ~5.5（很差，参数过多） |
| SymAttentionModel (sym_emb + attention) | s0,s1 done | avg ~22（s0=18.35, s1=26.57） |
| HybridCrossAttn (per-sym MLP + cross-attn) | running | — |

---

## §1 方向一：经典量化 RNN/GRU + Attention 系列

### DeepLOB（Zhang, Zohren & Roberts 2018/2019）

- **出处**：IEEE Trans. Signal Processing 67(11); [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- **数学原理**：
  - 输入：(B, T, 2L) raw LOB tensor（T=100 tick, L=10 level）
  - CNN (Inception modules, kernel 1×2) → LSTM 64 hidden → softmax 3-class
  - 本质：局部价格/量特征提取（CNN）+ 序列记忆（LSTM）
- **为何适合本任务**：比赛主办方 baseline，在 FI-2010 上验证过。但我们已有 359-d 工程特征，DeepLOB 适合**原始 LOB 输入路线**（若想绕开特征工程）
- **参数量预估**：~200k
- **训练成本**：~2-4 min/seed/fold on GPU（原始 LOB 输入）
- **风险/未知**：(1) 需要原始 100-tick LOB tensor，而当前 pipeline 已提取 359-d 特征；(2) 历史记录 T32 DeepLOB h_60 = -16.97（失败）
- **优先级**：⭐（已测过等效版，再测价值低）

---

### DeepLOB-Attention / Multi-Horizon Extension（Zhang et al. 2021）

- **出处**：[arXiv:2105.10430](https://arxiv.org/abs/2105.10430)
- **数学原理**：
  - DeepLOB encoder → LSTM decoder（autoregressive multi-horizon）
  - 加入 temporal attention：$\alpha_t = \text{softmax}(e^\top h_t / \sqrt{d})$
  - 5 个 horizon head 共享 backbone
- **为何适合本任务**：5 horizon 多任务共享 backbone 是 Jane Street 1st/Volkova 8th 的共识
- **参数量预估**：~300-500k（加 decoder + attention）
- **训练成本**：~4-6 min/seed on GPU
- **风险/未知**：仍需原始序列输入；attention on 5 heads 的实现细节复杂
- **优先级**：⭐⭐（若走原始序列路线，是首选多 horizon 架构）

---

### GRU + Cross-sym Context（Volkova/hyd 启发）

- **出处**：Volkova 2024 Jane Street 8th [GitHub](https://github.com/evgeniavolkova/kagglejanestreet); hyd 2023 Optiver 1st
- **数学原理**：
  - 每个 sym 独立 GRU（隐藏 dim = 64）encode 100-tick 序列
  - Cross-sym pooling：$\mu_{cross} = \frac{1}{5}\sum_s h_s$，$\sigma_{cross}$ 同理
  - concat $[h_s, \mu_{cross}, \sigma_{cross}]$ → 线性头输出预测
- **为何适合本任务**：Volkova 8th 明确表明"sym embedding 不工作"但 cross-sym pooling 有效；和我们 5-sym 批处理设置完美契合
- **参数量预估**：~150-200k（每 sym 小 GRU + 简单 head）
- **训练成本**：~3-5 min/seed（需重构数据为序列格式）
- **风险/未知**：需要访问原始 tick 序列而非预处理的 359-d 特征；实现成本偏高
- **优先级**：⭐⭐⭐（方向正确，但需要原始序列数据）

---

### Hawkes 过程 NN（Bacry et al. 2015 + Lu & Abergel 2018）

- **出处**：[arXiv:1502.04592](https://arxiv.org/abs/1502.04592); Tandf 2018
- **数学原理**：$\lambda(t) = \mu + \int_{-\infty}^{t} \phi(t-s) dN(s)$，用 NN 参数化衰减核 $\phi$
- **为何不适合本任务**：(1) stateless Predictor 约束——Hawkes 需要积分历史事件流；(2) R_Hawkes 实验已证明跨 sym 不稳；(3) 实现复杂
- **优先级**：❌（硬约束冲突，不建议）

---

## §2 方向二：跨股 / Cross-sectional Attention（本 workdir 主线）

### Set-Transformer / ISAB（Lee et al. 2019）⭐⭐⭐⭐

- **出处**：ICML 2019; [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)
- **数学原理**：
  ```
  MAB(X, Y) = LayerNorm(H + rFF(H))  where  H = LayerNorm(X + Att(X, Y, Y))
  SAB(X) = MAB(X, X)
  ISAB(X) = MAB(X, MAB(I, X))  # I ∈ R^{m×d} learned inducing points
  PMA(X) = MAB(S, rFF(X))       # S ∈ R^{k×d} seed vectors
  ```
  - 关键：ISAB 用 m 个 inducing points 做"中间人"，复杂度从 O(n²) 降到 O(mn)
  - 完全排列不变（permutation invariant）→ 天然 sym-agnostic
- **为何适合本任务**：
  - 5 sym 作为集合（set）输入，没有预设的 sym 顺序依赖
  - inducing points 学到 "市场状态的抽象概念"，每个 sym 通过 inducing points 间接交互
  - 不用 sym embedding → 当所有 5 sym 在训练/测试都存在时有最佳性能
- **参数量预估**：~250-400k（d_model=128, m=8 inducing points, depth=2）
- **训练成本**：~3-5 min/seed on GPU
- **风险/未知**：(1) ISAB 的 inducing points 在只有 5 个元素时可能不必要（SAB 就够）；(2) 5 个元素的 attention 信息量可能有限
- **优先级**：⭐⭐⭐⭐（Task TOP-1 候选；排列不变性是最优雅的设计）

---

### DeepSets（Zaheer et al. 2017）

- **出处**：NeurIPS 2017; [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- **数学原理**：
  - $f(S) = \rho(\sum_{s \in S} \phi(s))$
  - 对 5 sym：$\phi: \mathbb{R}^{359} \to \mathbb{R}^{d}$（共享 MLP），$\rho: \mathbb{R}^{d} \to \mathbb{R}$
  - 增强版：$[h_s, \text{mean}(h), \text{max}(h), \text{std}(h)] \to \text{head}(s)$
- **为何适合本任务**：最简单的排列不变架构；可作为 Set-Transformer 的轻量替代
- **参数量预估**：~100-150k
- **训练成本**：~2 min/seed
- **风险/未知**：sum/mean 聚合丢失 sym 间的精细交互信息
- **优先级**：⭐⭐⭐（计算高效的强 baseline 跨 sym 架构）

---

### Graph Attention Network（跨资产）（Feng et al. 2019 SR-GNN）

- **出处**：IJCAI 2019 "Temporal Relational Ranking for Stock Prediction"; [论文](https://www.ijcai.org/proceedings/2019/0922.pdf)
- **数学原理**：
  - 构建 5-sym 全连接图，边权 $e_{ij} = a(W h_i, W h_j)$
  - GAT 消息传递：$h_i' = \sigma(\sum_j \alpha_{ij} W h_j)$
  - 聚合后接预测头
- **为何适合本任务**：股价之间的相关性/因果关系图；5 sym 的小图使得 GAT 计算很轻
- **参数量预估**：~150-250k
- **训练成本**：~3 min/seed
- **风险/未知**：(1) 图结构固定（全连接）vs 动态图学习；(2) 5 sym 信息量有限
- **优先级**：⭐⭐⭐（与 Set-Transformer 互补验证，GAT 更明确地学边权）

---

### HATS（Kim et al. 2019）

- **出处**：AAAI 2019 "Hats: A Hierarchical Graph Attention Network for Stock Prediction"
- **数学原理**：
  - 层次图（sector → individual stock → prediction）
  - Relation attention module + Momentum attention module
  - $a_{ij} = \text{softmax}(\text{MLP}([h_i, h_j, r_{ij}]))$，$r_{ij}$ 是预定义关系特征
- **为何适合本任务**：不太适合——HATS 需要外部关系数据（行业分类/相关系数矩阵）；5 sym 无行业分类
- **优先级**：⭐（外部关系数据缺失，适配成本高）

---

### Cross-sectional Transformer（hyd Optiver 2023 变体）

- **出处**：hyd 1st Optiver 2023 writeup; 无独立论文
- **数学原理**：
  - 每个时间步（这里：每个 sample group）把所有 N sym 作为 sequence
  - Transformer 编码跨 sym 关系，每个 sym token 输出预测
  - 等价于 Transformer 在 sym 维度做 attention（不是 time 维度）
- **为何适合本任务**：Optiver 2023 1st place 的 Transformer 子模块（weight=0.2）
- **参数量预估**：~100-200k（5 sym as tokens, d=64-128）
- **训练成本**：~2-3 min/seed
- **风险/未知**：5 tokens 的 attention 方差大；hyd 实现中有 stock_id cat feature 违反我们约束
- **优先级**：⭐⭐⭐（本质上就是 SymAttentionModel 去掉 sym_emb，当前 worker 已在测试）

---

## §3 方向三：HFT / LOB-specific 架构

### TLOB / MLPLOB（Berti et al. 2025）

- **出处**：arXiv:2502.15757; [GitHub](https://github.com/LeonardoBerti00/TLOB)
- **数学原理**：
  - MLPLOB：feature-mixing MLP + temporal-mixing MLP（MLP-Mixer 变体），LayerNorm + GELU
  - TLOB：dual attention（spatial over LOB levels × temporal over ticks）
  - $\text{TLOB} = \text{Temporal-Attn} \otimes \text{Spatial-Attn}$
- **为何适合本任务**：MLPLOB 在 FI-2010 上接近 SOTA，"简单 MLP 在恰当训练下可达 SOTA" 验证了我们 flat-feature 路线
- **参数量预估**：MLPLOB ~50-200k; TLOB ~500k-1M
- **训练成本**：MLPLOB ~2 min/seed; TLOB ~10 min/seed
- **风险/未知**：TLOB 的 dual attention 需要 raw LOB tensor (100×40)；我们的 359-d 已 flat
- **优先级**：⭐⭐⭐（MLPLOB 已隐含在 MLP ablation 中；TLOB 适配有成本）

---

### TabNet（Arik & Pfister 2021）

- **出处**：AAAI 2021; [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)
- **数学原理**：
  - Sequential attention 选择特征 steps：$h_i = f_i(\mathbf{M}_i \cdot \mathbf{x})$
  - $\mathbf{M}_i = \text{sparsemax}(\mathbf{P}_i \cdot \mathbf{h}_{i-1})$ (sparse feature selection)
  - 每一步聚合被选特征的 logit
- **为何适合本任务**：359 维特征中有很多冗余，sparse selection 可能 focus 关键 features；可解释性好
- **参数量预估**：~300-500k（N_steps=6, N_a=16, N_d=16）
- **训练成本**：~5-10 min/seed（TabNet 比 MLP 慢）
- **风险/未知**：(1) dreamquark-ai 实现复杂；(2) 在 r35 调研时认为优势不明显；(3) 稀疏 attention 可能对低 SNR 金融数据不稳
- **优先级**：⭐⭐（中等优先级；sparse feature attention 概念对 LOB 有理论价值）

---

### WaveNet-LOB / 多尺度 Dilated ConvNet（Tsantekidis 2018 / Mukherjee 2022）

- **出处**：Tsantekidis et al. 2018 (AIAI); Mukherjee et al. 2022 (Applied Soft Computing)
- **数学原理**：
  - Dilated Conv 1D，dilations = [1, 2, 4, 8, 16]
  - 接受 tick 序列输入，指数增长感受野
  - $y[t] = \sum_k w_k \cdot x[t - d \cdot k]$ (causal, dilated)
- **为何适合本任务**：多尺度时间建模；比 LSTM 并行效率高
- **参数量预估**：~300-800k（5 dilation levels × 32-64 channels）
- **训练成本**：~4-6 min/seed on GPU
- **风险/未知**：(1) 需要原始 tick 序列；(2) 与 ModernTCN 思想相近（已失败）；(3) 金融序列 SNR 低，长程依赖可能是噪声
- **优先级**：⭐⭐（LOB 专门设计的实测少；ModernTCN 失败是警示）

---

### LOB Transformer（Briola et al. 2024 arXiv:2403.09267）

- **出处**：Quantitative Finance (July 2025); [arXiv:2403.09267](https://arxiv.org/abs/2403.09267)
- **数学原理**：
  - 针对 LOB 的特化 Transformer，将 bid/ask 两侧作为独立 token 流
  - Cross-attention between bid tokens and ask tokens
  - 结论：5-day rolling z-score 归一化对 LOB 至关重要
- **为何适合本任务**：我们已采纳其 5-day rolling z-score 建议（window-z 实现）；架构本身的 bid/ask cross-attention 有 LOB 物理意义
- **参数量预估**：~1-2M
- **训练成本**：~10-20 min/seed
- **风险/未知**：参数量大；原始 LOB tensor 输入；未开源代码
- **优先级**：⭐⭐（Briola 2024 主要贡献是 normalization 建议，架构本身复杂）

---

### BiN-CTABL / TABL（Tran et al. 2018/2020）

- **出处**：arXiv:1712.00975 (TABL) / arXiv:2003.00598 (BiN-CTABL)
- **数学原理**：
  - Bilinear projection：$H = (W_T X W_F^\top)$ 同时沿时间 T 和特征 F 轴
  - Bilinear Normalization：每维独立 batch norm
  - Temporal Attention：$a = \text{softmax}(q^\top H)$，提取最重要时刻
- **为何适合本任务**：LOB benchmark 2023 中跨数据集最稳的 NN；双轴 projection 对 tick×feature 输入高效
- **参数量预估**：~50-100k（参数极少）
- **训练成本**：~2 min/seed（轻量）
- **风险/未知**：(1) 需要 100×F 的输入格式（raw ticks），不是 359-d flat features；(2) 原 paper 实验只在 FI-2010，泛化性未知
- **优先级**：⭐⭐⭐（轻量高效，但需要原始 tick 输入路线）

---

### Siamese LOB（2025 A 股验证）

- **出处**：arXiv:2505.22678 (papers/model_siamese_lob_2025.md)
- **数学原理**：
  - Bid 侧和 Ask 侧使用共享权重的 encoder：$h_{bid} = f(X_{bid})$, $h_{ask} = f(X_{ask})$
  - 利用 LOB 的 bid-ask 对称性 inductive bias
  - $\text{pred} = g([h_{bid}, h_{ask}])$
- **为何适合本任务**：A 股专属验证（14 只军工股，75% 场景胜出）；我们数据是 A 股 LOB；与现有 mirror 增强一致
- **参数量预估**：~200-400k（取决于 encoder 大小）
- **训练成本**：~3-5 min/seed
- **风险/未知**：仅 14 只军工股验证，样本量小；我们的 mirror aug 已隐含了类似的对称 inductive bias
- **优先级**：⭐⭐（A 股证据有价值；但 mirror aug 已覆盖核心 idea）

---

## §4 方向四：量化业界实战 NN（Kaggle SOTA）

### Jane Street 2020 1st: Supervised AutoEncoder-MLP（Yirun Zhang）

- **出处**：Kaggle writeup; [GitHub](https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1)
- **数学原理**：
  ```
  Input → GaussianNoise(σ=0.035) → Dense(512) → BN → Swish
                                           ↓
           ├─ Decoder (MSE loss, reconstruction)
           ├─ Aux-head (BCE, reconstruct binary targets)
           └─ concat(latent, raw_input) → MLP head (main BCE)
  ```
  - 3 个 loss 同时反传：reconstruction + aux_BCE + main_BCE
  - Latent concat raw input 是关键："去噪后特征 + 原始特征" 双流
- **为何适合本任务**：(1) 多任务（5 horizon aux targets）框架直接可用；(2) 359-d 特征 → AE 去噪后仍是 tabular，无需 raw sequence；(3) latent + raw concat 的双流设计显著增加特征多样性
- **参数量预估**：~500k-1M（encoder 512→128, decoder 128→359, MLP 488→256→1）
- **训练成本**：~5-8 min/seed on GPU
- **风险/未知**：(1) GaussianNoise 在训练有益，推理时无噪声——需保持训练/推理一致；(2) Reconstruction loss 可能冲突 regression 目标
- **优先级**：⭐⭐⭐（适配性好；无需原始序列；多任务框架直接可落地）

---

### Optiver 2023 1st: CatBoost + GRU + Transformer 双轴集成（hyd）

- **出处**：Kaggle 1st place; [docswell writeup](https://www.docswell.com/s/8980249862/K6YQ3E-2024-05-23-200638)
- **核心 NN 组件**：
  - GRU (0.3 weight)：per-stock, 55 timesteps，intra-stock time-axis
  - Transformer (0.2 weight)：per-timestep, all 200 stocks，cross-stock axis
- **为何适合本任务**：双轴思路（time-axis GRU + sym-axis Transformer）是最强实战 evidence
- **适配后形态**：在 5 sym 上等价于 HybridCrossAttn（当前 worker 已测）；但 200 stocks → 5 sym 大幅缩减
- **优先级**：⭐⭐（核心思路已在 HybridCrossAttn 实验中验证）

---

### Jane Street 2024 8th: GRU Ensemble（Volkova）

- **出处**：[GitHub solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md)
- **核心 NN**：
  - 3-layer GRU + cross-sym market average features（per-time mean/std of 16 key features）
  - 4 aux target GRU models → linear 混合
  - 明确放弃了 sym embedding（不工作）
- **为何适合本任务**：sym-agnostic 路线的最强 Kaggle evidence（8th in 3757 teams）
- **适配后形态**：GRU over (B, 100, F) tick序列 + 5-sym mean/std augmented features
- **优先级**：⭐⭐⭐（需要 raw tick 序列，但方向最有 evidence）

---

### Numerai SOTA NN（2024-2025 Discourse 最佳方案）

- **出处**：Numerai Forum; [https://forum.numer.ai/](https://forum.numer.ai/)
- **数学原理**：
  - MLP + auxiliary era-level targets（对我们是 date-level，但 date=0 评测时死）
  - Era-boosting ensemble（多 seed + 多 target 加权）
  - Target engineering：short/medium/long 多 horizon 的线性组合
- **为何适合本任务**：Numerai 数据也是 tabular + symbol-agnostic + PnL 评分；方法论直接迁移
- **优先级**：⭐⭐（era-level 特征不可用；但多 horizon target engineering 可参考）

---

## §5 方向五：Feature-level Attention 和 Novel Tabular NN

### Feature Group Self-Attention（归纳偏置变种）

- **出处**：无单一 paper；受 TLOB spatial attention + Yandex FT-Transformer 启发
- **数学原理**：
  - 将 359 维特征分成 ~10 语义组（OFI 30d, WMP 11d, VPIN 4d, jump 8d, price/size 100d, ...）
  - 每组学一个 token embedding（dim=16）
  - 跨组 Transformer attention：组 token 互相 attend
  - 每组的贡献以 attention-weighted 方式合并
- **为何适合本任务**：比 FT-Transformer（每个 feature 一个 token）更节省计算；分组 leverages 领域知识（OFI 组内高相关，跨组关系需要 attention）
- **参数量预估**：~100-200k（10 groups × d=64 tokens + 2 attn layers）
- **训练成本**：~3-5 min/seed
- **风险/未知**：分组需要手动设计（domain expertise）；FT-Transformer（每 feature 一 token）已测失败，分组可能重蹈覆辙
- **优先级**：⭐⭐（理论有趣，但 FT-T 失败是负面信号）

---

### TabM（Gorishniy et al. 2025）

- **出处**：ICLR 2025; [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- **数学原理**：
  - k 个 mini-MLP（每个 ~50-200 params）分别处理输入
  - 高效参数共享：$y = \frac{1}{k}\sum_i f_i(x)$ with shared but distinct linear layers
  - "参数高效集成"：一个模型内部集成 k=32 个 mini 模型
- **为何适合本任务**：解决 MLP 过拟合的新方法；比 bagging 更高效；参数在 k 个 mini-MLP 间共享
- **参数量预估**：~100-300k（k=16, hidden=[128, 64]）
- **训练成本**：~4-6 min/seed（比单 MLP 慢 k×，但比 k 个独立 MLP 快）
- **风险/未知**：2025 论文，金融实测少；是否优于 multi-seed ensemble 未知
- **优先级**：⭐⭐（新方法，值得一试；但多 seed ensemble 已有效）

---

### MoE（Mixture of Experts）for LOB Regimes

- **出处**：概念来自 Shazeer 2017 MoE; 金融应用：Volatility MoE 2025
- **数学原理**：
  - K 个专家 MLP + gating network：$y = \sum_k g_k(x) \cdot E_k(x)$
  - 每个专家专注一种"市场状态"（趋势、震荡、高波动）
  - Gating 根据输入 x 软选择
- **为何适合本任务**：LOB 数据有明显 regime（大盘涨/跌/震荡）；专家可能各自捕获不同 regime
- **参数量预估**：~300-600k（K=4 experts, d=128 each）
- **训练成本**：~4-6 min/seed
- **风险/未知**：MoE load balancing 困难；gating collapse 常见；regime 标注不稳定
- **优先级**：⭐⭐（理论有吸引力，但工程挑战大）

---

## 优先级排序总表

| 排名 | 架构 | 预期提升 | 实现成本 | 核心理由 | 优先级 |
|---|---|---|---|---|---|
| 1 | **Set-Transformer (ISAB)** | +1~+5 vs sym_attn | 低 | 排列不变，最优雅的 cross-sym 设计；无 sym embedding | ⭐⭐⭐⭐ |
| 2 | **DeepSets Enhanced** | +0~+3 vs sym_attn | 极低 | 最简单的跨 sym baseline，快速验证 cross-sym 是否有益 | ⭐⭐⭐ |
| 3 | **SAE-MLP（监督自编码）** | +1~+4 vs MLP | 中 | Jane Street 2020 winner 架构；不需要 raw sequence；359-d 直接用 | ⭐⭐⭐ |
| 4 | **GRU + Cross-sym Pool** | +2~+6 vs MLP | 高（需原始序列） | Volkova 8th evidence 最强；但需要 raw tick 数据 | ⭐⭐⭐ |
| 5 | **FeatureGroup Attention** | +0~+2 vs MLP | 中 | 分组 domain knowledge + attention；FT-T 失败是警告 | ⭐⭐ |
| 6 | BiN-CTABL | +1~+3 | 高（需 raw sequence） | LOB benchmark 稳健，但需要序列格式 | ⭐⭐ |
| 7 | GAT (5-node graph) | +0~+2 vs sym_attn | 低 | 明确学边权；与 Set-T 互补 | ⭐⭐⭐ |
| 8 | TabM | +0~+2 vs MLP | 中 | 内置集成，新方法 | ⭐⭐ |
| 9 | DeepLOB + Attention | +0~+2 | 高 | host baseline 改进，需要原始序列 | ⭐⭐ |
| 10 | Siamese LOB | +0~+2 | 中 | A 股证据；但 mirror aug 已涵盖核心 idea | ⭐⭐ |
| 11 | TabNet | +0~+1 | 中 | 稀疏选特征；已在 r35 判定优势不明显 | ⭐ |
| 12 | TLOB dual attention | +0~+1 | 高 | 需要 raw LOB tensor；复杂 | ⭐ |
| 13 | WaveNet-LOB | +0~+1 | 高 | ModernTCN 失败是警告 | ⭐ |
| 14 | MoE for regimes | +0~+2 | 高 | 工程挑战大，gating collapse 风险 | ⭐ |
| 15 | Hawkes-NN | 不可行 | — | stateless 约束冲突 | ❌ |

**注意事项**：所有 cross-sym 架构（排名 1, 2, 4, 7）的当前 evidence 偏弱（SymAttentionModel ~22 vs MLP ~34）。Set-Transformer 是理论上最优雅的，但 5 sym 的 attention 信息量有限是客观挑战。SAE-MLP（排名 3）是不依赖 cross-sym 的独立方向，风险更低。

---

## 附：实测数字汇总（用于校准预期）

```
nn_arch_ablation 5-seed avg test_pnl:
  MLP baseline:    34.44  ← 目标是超过这个
  MLP_deep:        34.17
  MLP_wide:        33.00
  TimeMixer:       31.44
  AutoInt:         28.86 (4 seeds)
  GatedMLP:        27.56
  MLP-Mixer:       30.18 (2 seeds)
  ResMLP:          26.72
  FT-Transformer:  24.63
  ModernTCN:        0.77  ← 完全失败
  TimeKAN:          0.05  ← 完全失败

cross_sym_nn partial (2 seeds each):
  CrossConcatMLP:    ~5.5  ← 非常差（参数过多）
  SymAttentionModel: ~22.5 ← 明显低于 MLP baseline
  HybridCrossAttn:   running...
```

---

*调研 worker 执行时间：2026-05-30*
*基于 REFERENCES.md（130+ 条）+ research_and_history/ 调研笔记 + nn_arch_ablation 实测数据*
