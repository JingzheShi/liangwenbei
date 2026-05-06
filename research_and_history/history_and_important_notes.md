# 良文杯 — 研究记录与 trick 库

> 这是我们的"实验记忆"。每次试一个新 idea 都来这里登记：来源 → 实现 → 在本地 test set 上的效果。
> 不要在 PROGRESS.md 里记这些；PROGRESS.md 是项目状态，这里是知识沉淀。

---

## 0. 文件组织

- **`../docs/data_schema.md` ← 字段权威说明（官方版）；任何 EDA / 训练代码与本文不一致以此为准**
- `history_and_important_notes.md` — 本文件（索引 + tracker）
- `papers/` — 重要论文摘要（每篇一个 .md）；**R-model worker 用 `model_*.md` 前缀**
- `feature_ideas.md` — 候选特征清单（含来源 + 公式，R-feature worker）
- `model_ideas.md` — 候选模型架构 + 训练 trick 清单（R-model worker，**核心交付物**）
- `competition_insights.md` — Kaggle/比赛社区的 trick（R-feature worker，特征侧）
- `competition_insights_models.md` — Kaggle 比赛 winner 模型选型 + 训练 trick + 后处理（R-model worker，模型侧）
- `gbdt_vs_nn_for_lob.md` — GBDT vs NN 系统对比专题（R-model worker，决定建模路线）
- `tracker.md` — 实验 tracker（一行一个尝试，含 PnL/acc/状态）

---

## 1. 已验证的事实（来自 Worker A 的 PnL 评测器 sanity check）

测试集：`snapshot_sym0_date0_am.parquet`（1842 个有效 t）

| Strategy | label_5 | label_10 | label_20 | label_40 | **label_60** |
|---|---|---|---|---|---|
| 完美预测（PnL 上限近似） | 0.568 | 0.967 | 1.216 | 2.065 | **2.743** |
| 全预测平 | 0 | 0 | 0 | 0 | 0 |
| 全预测涨 | -0.328 | -0.296 | -0.264 | -0.170 | -0.062 |
| 全预测跌 | -0.409 | -0.441 | -0.473 | -0.567 | -0.675 |
| 随机三分类 | -0.272 | -0.261 | -0.250 | -0.267 | -0.368 |

**结论**：
1. label_60 PnL 上限最高 — horizon 越长 PnL 余地越大
2. 随机出手必亏（每 session 损 0.25–0.37）
3. **单次预测的 PnL 上限**（label_60）≈ 0.27%，所以预测 precision 必须 > "手续费阈值 / 平均移动幅度" 才有正期望
4. 完美预测 ≠ 理论 PnL 上限：当真实 |Δmidprice| 略大于 2× 手续费但被 α 阈值规整为 label=1 时，预测涨/跌反而比预测平赚得多 — **这是一个潜在的 trick：自定义阈值后处理**

---

## 2. 验证过的 baseline 模型表现

（待填写。Worker C 应该会跑 mmpc_demo baseline）

---

## 3. Trick 列表（持续更新）

格式：每条 trick 一行
```
[YYYY-MM-DD] [类别(feature/model/loss/post-proc/...)] [trick 名] - 来源 - 状态(idea/wip/tried/dropped/keep) - 效果(label_60 cum_pnl 等)
```

| 日期 | 类别 | trick 名 | 来源 | 状态 | 效果 |
|---|---|---|---|---|---|
| 2026-05-06 | post-proc | 阈值后处理（pred=1 但 model logit 偏离很远时翻转） | 评测器 sanity check 副产品 | idea | — |

---

## 4. 待研究方向

### 已研究完成（2026-05-06，研究 worker）

- [x] **LOB 高频预测的 SOTA 论文综述** [研究完成] — 见 `papers/`，覆盖 DeepLOB (2018) → TransLOB (2020) → TLOB/MLPLOB (2025) → LiT (2025) → Spacetimeformer-LOB (2024) → Microstructural guide (2024)。**重点结论**：MLPLOB（纯 MLP）在短 horizon 上能超过 transformer，先跑它。
- [x] **Kaggle Optiver Realized Volatility 冠军方案** [研究完成] — 1st place 用 **kNN over time_ids + tick-size 反推真实价格**重排时序，再 LightGBM/MLP ensemble。核心特征 = WAP + log return + realized volatility per minute。详见 `competition_insights.md` §1。
- [x] **Kaggle Optiver Trading at Close 冠军方案 (HYD)** [研究完成] — "Feature is all you need"。三大杀手锏：**triplet imbalance**, **pairwise price imbalance**, **synthetic index features**。LightGBM + MLP ensemble。详见 `competition_insights.md` §2。
- [x] **Kaggle Jane Street 1st (Yirun) 方案** [研究完成] — **Supervised Autoencoder + MLP**，bottleneck features 共享，5 个 resp head 共训。3-fold time-grouped CV with **10-day embargo**。多架构 ensemble + middle-60% averaging。详见 `competition_insights.md` §3。
- [x] **G-Research Crypto Forecasting trick** [研究完成] — Hull Moving Average (HMA) 是单个最强特征；**Fibonacci-window lags** = {55, 210, 340, 890, 3750}；regime gating（up/down/stable 分别建模）。
- [x] **Order-flow imbalance / OFI** [研究完成] — Cont-Kukanov-Stoikov 2014 公式可直接从我们 schema 拼出。Kolm 2023 证明 **MLOFI > raw LOB**。详见 `papers/cont_stoikov_ofi_2014.md` 与 `papers/deep_ofi_kolm_2023.md`。
- [x] **VPIN** [研究完成] — 简化代理 = `|mb-ma|/(mb+ma)` rolling mean；regime indicator 价值。详见 `papers/vpin_easley_lopez_2012.md`。
- [x] **Micro-price / Weighted Mid-Price** [研究完成] — Stoikov 2018，WMP 必加：`a·bs/(bs+as) + b·as/(bs+as)`。详见 `papers/microprice_stoikov_2018.md`。
- [x] **Hawkes / 订单到达建模** [研究完成] — 不需要正经拟合 Hawkes，用 multi-α EWMA 近似 + cross-excitation ratio 即可。详见 `papers/hawkes_lob_2024.md`。
- [x] **Kercheval-Zhang 144 维特征 schema** [研究完成] — 我们已有 154 维 schema 几乎完全覆盖了 Kercheval-Zhang 的 basic + time-insensitive + time-sensitive 三类。详见 `papers/kercheval_zhang_svm_2015.md`。
- [x] **类别不均衡处理** [研究完成] — **balanced sampling**（每类等量采样，见 microstructural guide）；focal loss 是替代；标签平滑可帮忙。

### 仍待 worker 研究

- [ ] **LightGBM/XGBoost vs 深度模型在我们数据上的实测**（理论上 GBDT 在 tabular feat 上很强；实证 by Worker C）
- [ ] **多任务（5 horizon head）loss 加权**：均匀 vs 按 PnL upper bound 加权（label_60 上限最高）— 实验问题
- [ ] **PnL-aware loss 设计**：直接优化 cumulative PnL；可参考 portfolio-aware loss（Zhang 2020）— 待实验
- [ ] **Threshold 后处理**：评测器 sanity check 已发现潜力（见 §1 结论 4）— 待实验
- [ ] **Ensemble 策略**：multi-arch (CNN + MLPLOB + Transformer + LightGBM) middle-60% averaging（Jane Street 1st 的 trick）— 待实验
- [ ] **Cross-sym feature interaction**：5 sym 共动 / 反向有多强 — 需要 EDA
- [ ] **Causal mask + Time2Vec**：在 transformer 路线时再考虑

### R-model worker 调研结论（2026-05-06，模型 + 训练策略侧）

**详细见 `model_ideas.md`、`competition_insights_models.md`、`gbdt_vs_nn_for_lob.md`。**

关键 takeaways：

1. **GBDT (CatBoost / LightGBM) 是几乎所有显式 LOB tabular 比赛的不会输的 baseline**——Kaggle Optiver 2023 1st 用 CatBoost(0.5)+GRU(0.3)+Transformer(0.2)，CatBoost 是单权重最重的部分。
2. **NN 作为 ensemble 队友 (~20-50% 权重)**——单 NN 罕见冠军，但 ensemble 多样性必需。
3. **BiN-CTABL > TransLOB > DeepLOB** 在跨数据集鲁棒性上（Briola 2023 LOB benchmark 99.7% robustness）。
4. **MLPLOB（纯 MLP）短 horizon 几乎与 TLOB 平**——简单优先。
5. **bid-ask Siamese parameter sharing 在 A 股 LOB 上 75% 场景胜出**（arxiv 2505.22678）——几乎免费的 inductive bias，强烈推荐。
6. **5-day rolling z-score normalization** 是处理 LOB 非平稳的标配（Lucchese 2024）。
7. **PnL-aware loss + threshold tuning + calibration** 是 PnL 评分比赛的最大 lever，必须独立优化。
8. **Online learning / fine-tune on tail** 对长时序漂移有效（Optiver 2023 1st 12-day retrain 5 次）。
9. **Purged K-Fold CV with gap (1-2 day)** 是金融 ML 标配。
10. **用户硬约束**：iTransformer / PatchTST / TimesNet / Informer / Autoformer / FEDformer 等通用 ts forecasting 模型**不在候选清单**——金融数据上打不过 GBDT。

### Short list：如果只能跑 3 个模型（来自 R-model worker）

1. **CatBoost / LightGBM ensemble + 重特征工程**（GBDT 主战场，Optiver 2023 1st 配方）
2. **DeepLOB-Attention multi-horizon shared backbone + Siamese + 5-day rolling z-score**
3. **BiN-CTABL**（LOB benchmark 最稳 SOTA NN）
- 加分：[3 GBDT + 2 NN] → LightGBM meta-learner stacking + threshold-by-PnL 后处理。

详细决策见 `model_ideas.md` §F。

---

## 5. Research worker 产出索引（2026-05-06）

- `papers/deeplob_zhang_2018.md` — DeepLOB baseline
- `papers/translob_wallbridge_2020.md` — TransLOB
- `papers/tlob_berti_2025.md` — TLOB / MLPLOB（强烈推荐先跑 MLPLOB）
- `papers/lit_lob_transformer_2025.md` — LiT (patches + transformer)
- `papers/spacetimeformer_lob_2024.md` — compound multivariate embedding
- `papers/deep_lob_microstructural_guide_2024.md` — 5-day rolling z-score + balanced sampling
- `papers/cont_stoikov_ofi_2014.md` — OFI 经典
- `papers/deep_ofi_kolm_2023.md` — MLOFI > raw LOB
- `papers/microprice_stoikov_2018.md` — WMP & micro-price
- `papers/vpin_easley_lopez_2012.md` — VPIN toxicity
- `papers/kercheval_zhang_svm_2015.md` — 144-feat schema 出处（与我们 schema 对应）
- `papers/hawkes_lob_2024.md` — Hawkes EWMA proxy
- `competition_insights.md` — Optiver Vol / Trading at Close / Jane Street / G-Research / Two Sigma 5 个比赛冠军方案要点（特征侧）
- `feature_ideas.md` — **49 条候选特征 + top 5 short list**（R-feature worker 核心交付物）

## 6. R-model worker 产出索引（2026-05-06）

### 模型论文摘要（前缀 `model_`）

- `papers/model_deeplob_zhang_2018.md` — DeepLOB CNN baseline
- `papers/model_translob_wallbridge_2020.md` — TransLOB
- `papers/model_lob_benchmark_briola_2023.md` — LOB benchmark study (15 模型对比，BiN-CTABL 最稳)
- `papers/model_multihorizon_deeplob_zhang_2021.md` — DeepLOB-Attention/Seq2Seq for multi-horizon
- `papers/model_siamese_lob_2025.md` — bid-ask 对称性 Siamese 架构（A 股验证）
- `papers/model_tabnet_arik_2019.md` — TabNet 表格 attention
- `papers/model_ft_transformer_gorishniy_2021.md` — FT-Transformer 表格 DL
- `papers/model_tabular_dl_not_all_2021.md` — Shwartz-Ziv："DL 不一定赢" reality check
- `papers/model_tabl_ctabl_tran_2018.md` — TABL/CTABL/BiN-CTABL bilinear+attention

### 综合分析文档

- `model_ideas.md` — **40+ 条模型架构 + 训练 trick + 后处理 ideas，含 short list（核心交付物）**
- `competition_insights_models.md` — Kaggle 4 大比赛 winner 模型选型 + 训练 + 后处理
- `gbdt_vs_nn_for_lob.md` — GBDT vs NN 系统对比（决定建模路线优先级）
