# R33: 提升 Recall 而保持 Precision 的 Trick 列表（≥10 个）

> 目标：把公榜 leader 的 5x recall（多交易）招数挖出来，**全部投影到我们的硬约束**：226-d Scheme C / LightGBM / sym-agnostic（C1 date=0、C2 test order shuffle、C3 sym OOD）。
>
> 调研日期：2026-05-07。

## 0. 总体诊断

我们目前 LightGBM 单模 + iter_006 LOSO h_60 = +13.61，公榜估算 +11.4。比公榜较好 team 的 7x PnL 差距来源：

| 来源 | 我们 vs leader | 影响 PnL |
|---|---|---|
| Accuracy | 0.330 vs 0.330 | × 1.0 |
| Recall | ≈0.20 vs 0.302 | **× 1.5** |
| F0.5 | 0.235 vs 0.324 | × 1.38 |
| Per-trade PnL | 0.000117 vs 0.000167 | **× 1.43** |
| 公榜 cum_pnl | 4.2 vs 29.18 | **× 7.0** |

**recall 1.5x + per-trade 1.43x = 2.14x**，仍然差 ~3.3x ——说明 leader 不是简单 lower threshold；他有 **「同 acc 下更多正例 + 选出来的正例平均更强」** 的双重优势。这只能来自 **更好的特征/模型**，不是单纯调阈值。

下面 16 个 trick 按 **「能不能解决 recall 双重优势」** 排序，前 5 个最关键。

---

## Trick 1 — 多 horizon Aux Target（multitask）

**方法**：除了主 target h=60 之外，再设置 h=30 / h=120 / h=240 的 aux target，共 4 个 head 共享 226-d input；每个 head 独立训练 LightGBM，再把 4 个模型的 prediction 用 stacking ridge 融合。

**哪个比赛/谁用**：
- Jane Street 2024 (Volkova 8th)：4 个 responder 同时训练，**+0.001 LB**（占总 +0.011 的 9%）
- Jane Street 2020 (Yirun 1st)：5 个 action target multitask
- JPX Tokyo (Shoki Sakai 1st)：noise-reduced rolling avg target

**适用我们 226-d Scheme C / LightGBM / sym-agnostic**：✅完全适用。LightGBM 不能直接 multitask，但**等价于训 N 个独立模型再 stacking**——预测时把它们当 4 个 feature 喂进顶层模型。

**预期 LOSO h_60 提升**：+0.5 ~ +1.5（这是 Volkova 测出来的 +0.001 R² × 我们 LOSO 的尺度比例）

**为什么帮 recall**：长 horizon target（h_240）噪声小、信号稳，会推高边缘正例的 prob → 推到 threshold 之上 → recall ↑。短 horizon（h_30）反映微观 alpha，过滤掉低 per-trade 信号 → precision 不掉。

**实施成本**：低。已有 horizon=60 框架，加 3 个 horizon labeling + 训练循环。1 day。

---

## Trick 2 — Cross-sym Pooling Features（market average）

**方法**：每个样本在 (t_local) 这个时间锚点附近 ± 1 bar 内，对 sym ∈ {0,1,2,3,4} 中**所有可见 sym** 的同 feature 算 mean / std / rank percentile / range。新增约 16 高相关 feature × 4 stat = 64 d。

**哪个比赛/谁用**：
- Jane Street 2024 (Volkova 8th)：「averages per date_id and time_id」**作为最大单点 feature engineering 涨分项**
- Optiver 2023 (hyd 1st)：cross-stock mean ratio + percentile rank（200 stocks 池化）
- 共同结论：跨 stock 池化是 sym-agnostic 模型的**信息补偿手段**

**适用**：✅ 完美适用。**正好绕开 C3（sym embedding 禁用）**——不直接使用 sym index，而是「同 t 下其他 sym 的统计量」。
- ⚠️ 注意：C1 让 date 不可信，所以 pooling 锚点必须用 **t_local 而不是 (date, t)**
- ⚠️ C2 让 t_local 不连续，但 ±1 bar 邻域可以用「同 (sym, t_local) 的 mean」近似

**预期 LOSO 提升**：+0.3 ~ +0.8

**为什么帮 recall**：当某只 sym 处于市场普涨/普跌环境时，per-bar return 期望偏离 0，原 226-d 单 sym 特征看不出，cross-sym mean 作为「市场状态信号」让模型更敢下注 → recall ↑

**实施成本**：低。0.5 day。

---

## Trick 3 — KNN-Target Retrieval（nyanp's Nearest Neighbors）

**方法**：训练时把所有 train 样本的 226-d feature + target 存进 FAISS index（IVFPQ 或 HNSW）；推理时每个测试点查最近 K=5 train 样本，把它们的 target 平均/中位数作为 1–2 个新 feature。

**哪个比赛/谁用**：
- Optiver Realized Vol 2021 (nyanp 1st)：**这是 nyanp 拿冠军的核心 trick**
- Optiver Realized Vol 2021 (orvp 7th)：同样用法

**适用**：✅ 完美适用且对症 C2。
- C2「测试点 order shuffle」 ↔ Optiver 2021「time_id shuffle」高度同构
- nyanp 用 NN 来替代不能用的 lag feature
- 我们 226 d 量级（1.47M train × 226 d）适合 FAISS HNSW，inference latency ≤ 1ms/query

**预期 LOSO 提升**：+0.3 ~ +1.0（高方差，依赖 226-d 空间是否有局部结构）

**为什么帮 recall**：当当前样本 226-d 跟某个历史强信号样本接近时，KNN target 直接把那个 target 作为 prior，pushing prob 上去 → 边缘正例被捞起来 → recall ↑。当 KNN 都不强时（没有相似历史），prior 是 0，不假阳 → precision 保持。

**实施成本**：中。需要 FAISS 集成、Predictor 内 query 测试。1 day。**注意**：pickle 提交包大小可能受 FAISS index 限制——如果太大，用 PQ 压缩或只保留 top-quantile 的 train 样本。

---

## Trick 4 — Cross-bar GRU 序列模型（sym-agnostic）

**方法**：每个 sym 内取连续 60 bar 序列，用一个轻量 GRU（hidden=64）做 sequence-to-1 预测；GRU 共享参数（不分 sym），所以 sym-agnostic。把 GRU prediction 作为新 feature 跟 LightGBM stacking。

**哪个比赛/谁用**：
- Jane Street 2024 (Volkova 8th)：3-layer GRU per day sequence；**单模 LB 0.0105**
- Optiver 2023 (hyd 1st)：GRU per stock 55-step
- JS 2024 codefluence：LSTM/GRU 在他的 NN+GBDT 合奏里

**适用**：✅ 适用。
- 训练时 sym 信息**只通过「同 sym 内连续序列」隐式输入**，参数不分 sym → sym-agnostic 安全
- ⚠️ 风险：GRU 一定要做严格的 LO2SO 验证；hidden state 不能跨 episode 残留
- ⚠️ inference 时 GRU 预测需要有 60-bar 的 context，**违反 C2「Predictor 不能维护跨调用 state」**——除非把整条 sym 序列在每次调用时**重新 forward**（O(60 × 64²)，CPU 16 核 < 5ms/sample，可行）

**预期 LOSO 提升**：+0.5 ~ +2.0

**为什么帮 recall**：序列模型把 226-d 「flat」 feature 看不到的 short-term momentum / mean-reversion 信号挖出来 → 在不增加 bias 的前提下补充信号 → 边缘 case 被推过 threshold。

**实施成本**：高。需要新 NN pipe + 推理时 batch reconstruct sym 序列。3 days。

---

## Trick 5 — Triplet Imbalance Features

**方法**：从 226 d 里挑选 30–50 组 3-列组合（如 bid/mid/ask price，或 size_1/2/3）；row-wise 算 `(max - mid) / (mid - min)`，每组生成 1 个 feature → 30–50 个新 feature。

**哪个比赛/谁用**：
- Optiver 2023 多个 top 方案（包括 hyd 1st）。
- 公式 `(max-mid)/(mid-min)` 是 Optiver 2023 公开 baseline 的 "Triplet imbalance"，被多个 top 方案 fold-in。

**适用**：✅ 完全适用。我们 226 d 里大量 LOB 价/量字段，三元组合很多。

**预期 LOSO 提升**：+0.2 ~ +0.5

**为什么帮 recall**：三元 ratio 捕捉非线性「中间点偏向」信号，原线性 LightGBM 单 split 只能近似；这给 ensemble tree 一个高信噪 split 候选 → marginal positives 被识别 → recall ↑。

**实施成本**：低。numba 向量化 < 30 min 跑完 1.47M 行。0.5 day。

---

## Trick 6 — Hull MA + Fibonacci 多窗口

**方法**：在 16 个最强 feature 上叠加 5 个 Fibonacci 窗口的 Hull Moving Average：window ∈ {5, 13, 34, 89, 233} (intraday bars)。
- HMA(n) = WMA(2·WMA(n/2) − WMA(n))，n=window；比 EMA 响应更快但平滑性更好

**哪个比赛/谁用**：
- G-Research Crypto Forecasting：top 方案 "hull MA + Fibonacci windows" 是公开最重要 feature 类型

**适用**：✅ 适用。Hull MA 是 EMA 的改进，单 sym 内计算，sym-agnostic。
- ⚠️ 跨 episode 不计算（重置）

**预期 LOSO 提升**：+0.1 ~ +0.4

**为什么帮 recall**：更精细的 multi-scale momentum 让模型对「趋势刚开始」的样本敢下注，而不是等到趋势已经成熟（precision 老 case）。

**实施成本**：低。pandas 向量化。0.3 day。

---

## Trick 7 — Lightweight Supervised Autoencoder（SAE）Latent

**方法**：4-layer NN（256 → 128 → 32 latent → 128 → 226 reconstruct，外加 32 → 1 supervised target head）。训练完取 32-d latent + raw 226 d concat 喂 LightGBM。

**哪个比赛/谁用**：
- Jane Street 2020 (Yirun 1st)：**整个比赛的核心架构**

**适用**：✅ 适用。NN 不分 sym → sym-agnostic 自动满足。

**预期 LOSO 提升**：+0.3 ~ +1.0

**为什么帮 recall**：denoising 把 noise 消掉的 226 d → LightGBM 看到「干净版」特征 + 「有监督压缩版」特征，对噪声大的 marginal cases 更稳定 → 信号一致时敢下注。

**实施成本**：中。需要 NN training pipeline + checkpoint + inference。2 days。

---

## Trick 8 — Gaussian Noise Input Augmentation

**方法**：训练 LightGBM 之前，将训练集 input 复制 K=2 份，第二份每个 feature 加 N(0, 0.03·σ_feature) 高斯噪声；标签不变。等价于在 input 上做 dropout 鼓励模型学习 robust pattern。

**哪个比赛/谁用**：
- Jane Street 2020 (Yirun 1st)：σ=0.035 写在 supervised AE 的 GaussianNoise layer
- 普遍 NN 训练技巧；对 GBDT 也有效，等价于 jittering augmentation

**适用**：✅ 适用。

**预期 LOSO 提升**：+0.1 ~ +0.3

**为什么帮 recall**：robust feature 让 marginal cases（feature 值靠近 split point）更稳定地推过阈值；noisy training 让模型不依赖某单一精确值，更普遍地识别正例 pattern。

**实施成本**：低。0.5 day。

---

## Trick 9 — Multi-Threshold Ensemble (per-horizon classifier)

**方法**：训练 3 个 LightGBM binary classifier，target 分别为 `return_h60 > q90`, `> q70`, `> q50`（即 top-10%, top-30%, top-50% positives）。inference 时 3 个 prob 加权平均当 final score。

**哪个比赛/谁用**：
- Jane Street 2020：5 个 action target = 5 个不同 cutoff 的 binary
- Optiver 2023 多个方案：multi-target ensemble

**适用**：✅ 适用。

**预期 LOSO 提升**：+0.2 ~ +0.6

**为什么帮 recall**：单 cutoff 的 BCE 模型更关注极端正例；多 cutoff 让中间强度的正例也被建模 → 边缘正例 prob 不再压到 0 → recall ↑。

**实施成本**：低。1 day（要重新跑 3 轮训练）。

---

## Trick 10 — Per-trade PnL-aware Loss（focal / weighted）

**方法**：把 binary loss 换成 weighted-CE，权重 = `|future_return|` 或 `|future_return|^0.5`。让模型把注意力放在 PnL 大的样本上，而不是只看分类对错。

**哪个比赛/谁用**：
- Jane Street 2020 比赛 metric 本身就是 weighted-utility 加权
- Jane Street 2024 metric = weighted-zero-mean R²
- 共识：当 metric 是 PnL，loss 必须是 PnL-aware 的，否则 leakage

**适用**：✅ 适用。我们当前 cross-entropy 没考虑 PnL 大小。

**预期 LOSO 提升**：+0.3 ~ +1.0（直接 align metric 和 loss）

**为什么帮 per-trade alpha**（更大）：模型不再追求 accuracy，而是追求「猜对的那部分 PnL 加起来最大」 → 自然偏向 PnL 大的预测 → per-trade 1.43x gap 直接缩小。同时高 PnL 样本通常更明显，recall 也会涨。

**实施成本**：低。LightGBM 支持 sample_weight。0.5 day。

---

## Trick 11 — Cross-sym Rank Percentile

**方法**：对每个 sample，对 16 个高相关 feature，算「在同 t_local 的 5 个 sym 里这个 feature 排第几 percentile」。新增 16 d。

**哪个比赛/谁用**：
- Optiver 2023 (hyd 1st)：cross-stock percentile rank 是 magic features 之一

**适用**：✅ 适用。**与 Trick 2 互补**（Trick 2 给 mean/std，这里给 rank）。

**预期 LOSO 提升**：+0.2 ~ +0.4

**为什么帮 recall**：rank-based feature 是 unit-free 的，跨 OOD sym 也稳定（Trick 2 mean 在 OOD sym 上可能有偏移）→ 测试集如果含 unseen sym，rank percentile 仍然有效。

**实施成本**：低。0.3 day。

---

## Trick 12 — Time-Bucket Cutoff Features

**方法**：把 t_local 切成 3–5 段（如 0–120 / 120–360 / 360–720 / 720–960），在每段内做 separate 统计（mean / std / rank）。新增约 12 d。

**哪个比赛/谁用**：
- Optiver 2023 (hyd 1st)：300 s / 480 s / 540 s 三段切
- 直觉：不同时段的 LOB 行为很不一样（开盘、盘中、收盘）

**适用**：✅ 适用。t_local 不被 mask，可信。

**预期 LOSO 提升**：+0.1 ~ +0.4

**为什么帮 recall**：模型可以学「在某些时段更敢下注，在某些时段更保守」 → 在 high-signal 时段更多 trade → recall ↑。

**实施成本**：低。0.3 day。

---

## Trick 13 — Stacking with NN Residual

**方法**：先训 LightGBM 出 prediction p_lgb；residual r = y - p_lgb；再训一个 4-layer MLP 拟合 r = f(226 d features)。final = p_lgb + 0.3 × p_nn。

**哪个比赛/谁用**：
- Optiver 2023 (hyd 1st)：CatBoost(0.5) + GRU(0.3) + Transformer(0.2)
- Jane Street 2024 (codefluence 13th)：LightGBM + MLP

**适用**：✅ 适用。

**预期 LOSO 提升**：+0.3 ~ +0.8

**为什么帮 recall**：NN 能学 LightGBM 难学的非线性（连续 + 高阶）→ residual 中包含 LightGBM 漏掉的边缘 case 信号 → recall ↑。

**实施成本**：中。1.5 days。

---

## Trick 14 — Pseudo-labeling from Unlabeled Test Set（半监督）

**方法**：first pass 训 LightGBM；second pass 把 high-confidence test prediction（top 5%）当 pseudo-label 加进训练集再训。

**哪个比赛/谁用**：
- Jane Street 2020 多个 top 5 方案
- 普遍 Kaggle CV-based 比赛技巧

**适用**：⚠️半适用。我们 evaluation 是平台 oracle，test set 不能离线访问；但**可以在 LOSO 内做**——用训练集做 K-fold，hold-out fold 当「pseudo-test」做 pseudo-labeling。

**预期 LOSO 提升**：+0.1 ~ +0.4（数据规模够大时收益小）

**实施成本**：中。1 day。**优先级 P3**（数据已经 1.47M 行，pseudo-label 收益边际）。

---

## Trick 15 — Calibrated Probability + Cost-aware Threshold

**方法**：对 LightGBM 输出做 isotonic regression 校准，再在校准后概率上设阈值；阈值优化用 PnL-aware grid search（不是 acc / F1）。

**哪个比赛/谁用**：
- Jane Street 2020 顶层 post-processing
- Optiver 2023 (hyd 1st)：「post-processing contributed significantly」

**适用**：✅ 适用。我们 T34 / T40 已经做过 calibration v2，但**阈值优化没用 PnL-aware grid search**——值得复盘。

**预期 LOSO 提升**：+0.05 ~ +0.3（已部分实现）

**为什么帮 recall**：校准让边际概率更可信 → 阈值调到对应 recall 0.30 时，误判率不爆炸 → 安全降阈 → recall ↑。

**实施成本**：极低。0.2 day。

---

## Trick 16 — Multi-seed Ensemble + Snapshot Ensemble

**方法**：同 hyperparameter 跑 5 个 seed（每个 200 epochs，但 cosine LR 重启 5 次产 5 个 snapshot）→ 总共 25 个模型 simple average。

**哪个比赛/谁用**：
- Jane Street 2024 (Volkova)：3 seeds × 2 GRU = 6 模型，**+0.001 LB**
- 普遍 Kaggle 通用技巧

**适用**：✅ 适用。LightGBM seed 重启便宜。

**预期 LOSO 提升**：+0.1 ~ +0.4

**为什么帮 recall**：variance reduction → 边缘 case prob 更稳 → 不会被噪声推到 threshold 下面 → recall ↑。

**实施成本**：低。训练时间 × 5。1 day（多 seed 可并行）。

---

## 总览：trick 优先级矩阵

| # | Trick | 预计 +LOSO h60 | 实施成本 | 优先级 |
|---|---|---|---|---|
| 1 | Multi-horizon aux target | +0.5 ~ +1.5 | 1d | 🥇 P0 |
| 2 | Cross-sym pooling | +0.3 ~ +0.8 | 0.5d | 🥇 P0 |
| 3 | KNN-target retrieval | +0.3 ~ +1.0 | 1d | 🥇 P0 |
| 5 | Triplet imbalance | +0.2 ~ +0.5 | 0.5d | 🥇 P0 |
| 10 | PnL-aware loss | +0.3 ~ +1.0 | 0.5d | 🥇 P0 |
| 11 | Cross-sym rank percentile | +0.2 ~ +0.4 | 0.3d | 🥈 P1 |
| 9 | Multi-threshold ensemble | +0.2 ~ +0.6 | 1d | 🥈 P1 |
| 12 | Time-bucket cutoffs | +0.1 ~ +0.4 | 0.3d | 🥈 P1 |
| 13 | Stacking NN residual | +0.3 ~ +0.8 | 1.5d | 🥈 P1 |
| 6 | Hull MA Fibonacci | +0.1 ~ +0.4 | 0.3d | 🥈 P1 |
| 16 | Multi-seed ensemble | +0.1 ~ +0.4 | 1d | 🥈 P1 |
| 4 | Cross-bar GRU sequence | +0.5 ~ +2.0 | 3d | 🥉 P2 |
| 7 | Lightweight SAE latent | +0.3 ~ +1.0 | 2d | 🥉 P2 |
| 8 | Gaussian noise aug | +0.1 ~ +0.3 | 0.5d | 🥉 P2 |
| 15 | PnL-aware threshold | +0.05 ~ +0.3 | 0.2d | 🥉 P2（部分已做） |
| 14 | Pseudo-labeling | +0.1 ~ +0.4 | 1d | 🥉 P3 |

## iter_007 推荐组合

**Bundle A（保守，1.5 day，+1.5 ~ +3.5 LOSO）**：Trick 1 + 2 + 5 + 10 + 11 + 12
- 全是低成本、sym-agnostic 安全的 feature engineering + loss 调整
- 不动模型架构

**Bundle B（激进，3 day，+2.5 ~ +5.5 LOSO）**：Bundle A + Trick 3 + 13
- 加 KNN retrieval + NN residual stacking
- 风险更高但天花板也高

**Bundle C（重型，6 day，+3.5 ~ +8.0 LOSO）**：Bundle B + Trick 4
- 全套 + GRU 序列模型
- 接近 Volkova 8th place 的复杂度

**建议**：先 Bundle A（高确定性 +2 LOSO 已能让我们从 +13.61 升到 +15.5+），观察后再决定是否上 Bundle B / C。

---

## 一条没写但重要的 Meta Trick

### Trick 0 — 不要试图一次拼装所有 trick

**why**：iter_006 → iter_007 的目标不是 doubling pnl，而是稳健地拉个 +1 ~ +2 LOSO h_60。Volkova solution.md 里每个招数都是独立 +0.001 R² 测出来的。如果我们一次叠 5 个改动跑 LOSO，发现退步 0.5 时，没法定位到底是哪个有害。

**how**：每个 trick **独立 ablation 一次**，用 fixed seed=42 LO2SO（fast 单 fold 即可，不用全 LOSO），筛掉退步的，再叠加。Volkova 比赛里花了 3 个月就是这么逐项 +0.001 拼起来的。

---

## 来源
所有 trick 都在 `r33_kaggle_hft_winners.md` 中给出原始来源链接，本文不重复。
