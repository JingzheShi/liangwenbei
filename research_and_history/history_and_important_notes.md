# 良文杯 — 研究记录与 trick 库

> 这是我们的"实验记忆"。每次试一个新 idea 都来这里登记：来源 → 实现 → 在本地 test set 上的效果。
> 不要在 PROGRESS.md 里记这些；PROGRESS.md 是项目状态，这里是知识沉淀。

## ⚠️ 评测硬约束（写代码前必读 → [`../CRITICAL_CONSTRAINTS.md`](../CRITICAL_CONSTRAINTS.md)）

**任何 trick 设计、特征工程、模型架构必须遵守这 3 条红线，否则提交报错或拿 0 分**：

1. **`date` 字段评测时被置 0**——禁止把 date 当 feature 或推理时段
2. **评测程序打乱测试点顺序**——Predictor 必须每次 predict 调用完全独立，不能维护跨调用 state（hidden state / cache buffer / sym→映射）
3. **sym 0-4 但可能含训练外股票**——模型必须 **sym-agnostic**：
   - 禁 `nn.Embedding(num_sym, ...)` 用 sym ID
   - 禁把 sym 加到 GBDT/树模型 feature
   - 禁 per-sym 模型集合 / per-sym normalization
   - 全局 normalization、不依赖 sym 的架构
4. **可以用 `time`**（保留实际时间戳），但需要在 config.feature 列里显式 list 才会被送入 DataFrame；推荐在 build cache 阶段把 time-derived 列预算好

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

## 1.9 跨 sym OOD 鲁棒性方法调研（2026-05-06，R11 worker）

**核心交付物**：[`r11_ood_cross_stock.md`](r11_ood_cross_stock.md) + 6 篇 paper summary in `r11_papers/`

**TL;DR — Top 2 方案**（如果只能上 2 个）：

| 优先级 | 方法 | 复杂度 | 预期 LOSO 增益 | 来源 |
|---|---|---|---|---|
| 🥇 #1 | **Group DRO via dynamic sample reweighting** (LightGBM-compatible) | 1 天 | +3 ~ +8 | Sagawa 2020 |
| 🥈 #2 | **Cross-Sym Mixup** (training data augmentation) | 0.5 天 | +1 ~ +4 | Yirun 2021, Zhang 2018 |

两个方案**互补**且都不破坏 sym-agnostic 推理路径。

**调研覆盖的 12 条方法**（详见 r11 报告）：

A. Group DRO — ✅ LightGBM 友好；最大 lever
B. Cross-sym Mixup — ✅ LightGBM + NN；Jane Street 1st 验证
C. V-REx — NN only；比 IRM 稳
D. IRM — ❌ 5 个 env 太少（需要 envs > non-invariant dim）
E. DANN — NN only；TS DG 中是 weak baseline
F. CORAL — NN only；moment alignment
G. TTA / Tent — 原版违反硬约束 #2；within-window 等价物 = T7 已做
H. SWA — NN only；OOD generalization 自带改善
I. LOSO Bootstrap Ensemble — 我们 T11 在做扩展
J. Hash-trick stock embedding — ❌ 风险大于收益
K. Quantile / Huber / Focal Loss — ✅ LightGBM 友好
L. Per-sym standardization @ train + distill 到 sym-agnostic — 中等复杂度

**Negative findings (不要做)**：

| 不推荐方法 | 为什么 |
|---|---|
| Per-sym embedding | 违反硬约束 #3 |
| Per-sym normalization at inference | 违反硬约束 #3 |
| 原版 Tent (跨 batch BN running stats) | 违反硬约束 #2 |
| Cross-sectional features 跨 sym 同时刻 | 评测 batch shuffle，无法保证多 sym 共存 |
| DANN with 5-class sym discriminator (NN only) | TS DG benchmark 显示 DANN 是弱 baseline |
| IRM | 5 个 env 不够 (linear theory needs envs > non-inv dim) |
| Online live retraining | 评测无 retraining 接口 |

**对应的 next worker tasks (T12 系列建议)**：

- **T12-A**: Cross-Sym Mixup augmented LightGBM 重训 → LOSO 5-fold（半天）
- **T12-B**: Group DRO via dynamic sample reweighting on LightGBM（1-2 天）
- **T12-C** (备用): Focal loss 替代标准 cross-entropy

**论文摘要目录** (`r11_papers/`)：

- `group_dro_sagawa_2020.md` — min-max worst-group + 强正则要求
- `v_rex_krueger_2021.md` — variance penalty across envs，比 IRM 稳
- `dann_ganin_2016.md` — adversarial domain confusion + GRL
- `tta_finmarkets_2026.md` — financial TTA 实证：BN-only 是 robust default，激进 TTA 有害
- `janestreet_yirun_writeup.md` — Jane Street 1st 架构 + mixup + 4-fold day-iso CV
- `mixup_for_finance.md` — mixup 在 stock prediction（LightGBM）全部 case 提升
- `wildtime_ts_dg_benchmark.md` — TS DG benchmark：DANN 弱、ERM 难超越

---

## 1.8 sym=2 brittleness 诊断（2026-05-06，T8 worker）

**根因**：sym=2 看起来是 **大盘蓝筹/ETF**，与 sym {0,1,3,4} 的 distribution 显著不同：
- amount_delta（CNY 量）+7.49 z-score（4-7× 其他 sym）
- spread1 -2.33 z（最窄 spread）
- mid_diff std (per-tick vol) -2.85 z（最低波动）
- 深档量 +2.34 z（最深订单簿）

**LOSO 模型训 {0,1,3,4} → 把 sym=2 normal state 误读为"买入压力"** → 预测 UP 62% vs 真实 18%。

5 个 sym-agnostic feature 建议（按优先级）：

| # | Feature | 公式 | Priority | 预期 LOSO 增益 |
|---|---|---|---|---|
| F1 | amount_delta within-window z-score | `(amount - rolling_mean_100) / rolling_std_100` | HIGH | **+3 ~ +5**（最大 bias 修） |
| F2 | spread-normalized price moves | `Δmidprice / max(spread1+1, ε)` | HIGH | +1 ~ +3 |
| F3 | relative vol quantile | `current_abs_move / std(Δmid_100t)` | MED | +2 |
| F4 | order-flow imbalance ratios | `mb / (lb + mb)` 等 | MED | +1 |
| F5 | class-prior calibration head | training-time prior 校正 | LOW | +0.5 ~ +1.5 |

**总预期**：F1 + F2 + F3 三件套 → LOSO sum 提升 +6 ~ +10（主要修 sym=2，附带其他 sym 小升）。

→ **下一个 worker 应该是 Scheme E = T2 features + F1/F2/F3 (+F4)**

详见 `experiments/T8_sym2_diagnostic/report.md`。

---

## 1.7 LOSO CV 验证 + Threshold post-processor（2026-05-06，T4 worker）

### LOSO Scheme A 5 fold 结果（label_60，验证假设：T2 是 brittle 模型）

| held-out sym | held-out cum_pnl (raw argmax) | held-out cum_pnl (T=0.5,δ=0.15) |
|---|---|---|
| 0 | -12.41 | -1.46 |
| 1 | -3.77 | +3.26 |
| 2 | -8.12 | +0.67 |
| 3 | +0.45 | +0.21 |
| 4 | +1.75 | +3.78 |
| **sum** | **-22.10** | **+6.45** |
| folds > 0 | 2/5 | **4/5** |

**惊人发现**：raw argmax 的 LOSO sum **-22.10** 与 mmpc_demo 平台 **-23.13** **量级 + 符号完全吻合** → LOSO 是平台行为的可信代理。
T2 IID（同 sym 训练 + 同 sym 测试）+19.25 → LOSO（跨 sym）-22.10 → **跨 sym 损失 41 分**。

### 关键发现：高置信度 threshold post-processor 是当前最大 lever

**机制**：默认 LightGBM argmax 在 ~34% prob 就出手 → OOD sym 上弱信号易翻面 → 手续费拖死。
强制 max(prob_0, prob_2) > 0.5 且 > prob_1 + 0.15 才出手 → 出手量降 86%（320k→45k），但单笔 alpha 从 -7e-5 翻到 +1.4e-4 → cum_pnl 从 -22.10 → **+6.45**。

**且不是只对 OOD 有效**：T2 Scheme B IID test cum_pnl 从 +9.36 → **+14.88**（同样 +59%）。
说明 thresholding 是**通用增强**，不只是 OOD 救命。

### iter_001 决策

- ❌ **iter_001_lgbm_schemeB（raw argmax）** — 不要提，LOSO 预测是 -22
- ✅ **iter_001c_lgbm_schemeB_thresh** — 推荐提交，预期 [-2, +6]

**新硬约束（写到 trick 库）**：

| 日期 | 类别 | trick 名 | 来源 | 状态 | 效果 |
|---|---|---|---|---|---|
| 2026-05-06 | post-proc | **高置信度阈值 gating (T=0.5, delta=0.15)** | T4 worker LOSO 验证 | **keep** | LOSO -22.10 → +6.45（label_60，5 fold sum）；IID +9.36 → +14.88（label_60）|
| 2026-05-06 | val 策略 | **必须用 LOSO CV，不能信 IID** | T4 worker 验证 | **must do** | T2 IID +19 → LOSO -22；41 分跨 sym 损失 |

---

## 1.6 本地 Pipeline 完整性审计（2026-05-06，audit_pipeline.py）

**结论：本地 pipeline (split / PnL / windowing) 100% 正确，没有 bug**：

- **A. Split**：train(800)/val(160)/test(240) 三集合无重叠，date 范围正确
- **B. PnL 公式**：手算与 `compute_pnl` 完全一致（pred=labels → 0；pred=2,真实=1 → -0.0002）
- **C. Windowing**：`feat_df.iloc[t-99:t+1]` 与官方 `main.py` 的 `iloc[index:index+100]` byte-perfect 相同
- **D. mmpc_demo 在单 session 上几乎全预测 1**（94-100%）—— 模型塌缩到多数类，PnL ≈ 0
- **E. 240-session per-session 分布**：

| horizon | sum | mean | pos/neg | mean_acc | mean_predflat |
|---|---|---|---|---|---|
| label_5 | +4.09 | +0.017 | 124/31 | 0.81 | **0.94** (94% 预测平) |
| label_10 | +6.16 | +0.026 | 130/45 | 0.72 | 0.85 |
| label_20 | +6.33 | +0.026 | 95/34 | 0.79 | **0.96** |
| label_40 | +6.27 | +0.026 | 90/47 | 0.69 | 0.88 |
| label_60 | +3.57 | +0.015 | 90/72 | 0.62 | 0.74 |

按 sym 分（label_20）：sym=0 +0.62, sym=1 +1.59, sym=2 +0.74, sym=3 -0.008, sym=4 **+3.40**
→ **5 个 sym 在本地全是非负**，但平台是 -7。

**所以差距不是 bug，是 distribution shift**：
- 本地：mmpc_demo 在 5 个训练 sym 的 date 96-119 上**绝大多数预测平** → cum_pnl 弱正
- 平台：mmpc_demo 预测**主动**了很多（44-60% active vs 本地 6-26%） → 大量乱猜亏 fee

**两种最可能的解释**：
1. **跨股票泛化**：平台 sym=0..4 实际是不属于训练集的股票（赛题明文允许）
2. **时间漂移**：平台 test 集是 date 119 之后的更新数据

**T2 Scheme B 在 5 个本地 sym 上的表现**（label_60，**强烈不均匀**）：
- sym=0: +2.96 (predflat 91%)
- sym=1: **-0.96** (predflat 20%)
- sym=2: +1.05 (predflat 57%)
- sym=3: +3.36 (predflat 93%)
- sym=4: **+12.84** (predflat 32%)

→ T2 的 +19.25 总分 **67% 来自 sym=4**！如果平台没有类似 sym=4 的股票，T2 也可能是负分。

**关键 takeaway / 后续硬约束**：
- **任何"本地正分"都不能直接外推平台**——必须用 leave-one-sym-out CV 评估
- **sym=4 在所有模型上都异常**（mmpc_demo 也是 +3.40 最高的 sym），可能是某只波动率 / 流动性显著不同的股票
- **优先做 sym-robust 特征工程**：避开 per-sym 量级敏感的特征，多用 ratio / OFI normalized / 形状特征
- **iter_001 候选**：T2 Scheme B 最强但 brittle；建议先做 LOSO CV 验证再提交，否则可能再吃一次 -7 ~ -23

---

## 1.5 已验证的事实（来自 iter_000 平台真实成绩，2026-05-06）

mmpc_demo（官方 demo）在平台 test 集上**全 5 horizon 都是负分**：

| label | platform cum_pnl | per-trade | n_active |
|---|---|---|---|
| label_5  | -9.079  | -0.000200 | ~45.4k |
| label_10 | -13.319 | -0.000199 | ~66.9k |
| label_20 | **-6.649** | -0.000134 | ~49.6k |
| label_40 | -16.065 | -0.000212 | ~75.8k |
| label_60 | -23.132 | -0.000184 | ~125.7k |

**Best of 5 = -6.649 at label_20**（最不亏）。

**关键 insight（指导后续训练）**：

1. **per-trade ≈ -fee 完美吻合** → mmpc_demo 是真正零 alpha；任何随机预测都会得到这个数字
2. **平台 test 集 ~210k 评测点**（用 label_60 active 数倒推），约为我们本地 240-session test 的一半
3. **长 horizon (label_60) 在平台上是 worst，不是 best**——因为 p_flat 低 → 模型激活率高 → 乱猜更多 → 损失放大。**翻转了我们 sanity-check 阶段对 label_60 的预期**
4. **本地 1-session 评测严重低估风险**——本地 sym0_date0_am 测得 ≈0 PnL（模型几乎全预测 1），但平台分布更广，模型乱预测 → 大额负 PnL。**强制要求：所有 iter 在本地必须跑 240-session 全 test 集评测**
5. **任何"全预测 1（不交易）"baseline 都拿 0 分，胜过 mmpc_demo**——这是绝对底线，任何提交必须 > 0
6. **5 horizon 中最容易学的可能是 label_20**（mmpc_demo 在 label_20 损失最少 = 平均预测错得最少）——可作为后续重点优化目标

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

---

## 7. R12 — Ensemble & Stacking 调研（2026-05-06，R12 worker）

### 主交付物

- `r12_ensemble_stacking.md` — **12 条 ensemble/stacking 方案**（每条含实现、来源、复杂度、适配度、期望增益）+ 失败方案排除 + Top-3 优先级建议 + 一周实施顺序。

### 关键 paper 摘要（`r12_papers/`）

- `r12_papers/snapshot_ensembles_huang_2017.md` — Snapshot Ensembles (ICLR 2017)，cosine cyclic LR + M snapshot 平均
- `r12_papers/swa_izmailov_2018.md` — Stochastic Weight Averaging (UAI 2018)，weight space averaging → flatter minima
- `r12_papers/ngboost_duan_2020.md` — NGBoost (ICML 2020)，probabilistic gradient boosting + uncertainty
- `r12_papers/stacking_bayesian_yao_2018.md` — Stacking > BMA in M-open (Bayesian Analysis 2018)，simplex-constrained meta
- `r12_papers/purged_kfold_lopez_de_prado.md` — Purged K-Fold + Embargo + CPCV，对应我们 LOSO 协议

### 核心 takeaways

1. **T6b 失败 = base 太相关 (ρ > 0.97 simple seed-only)**：修复路径按多样性强度排序：换算法 > 换特征子空间 > 换 horizon > 换超参 > 换 seed。
2. **Top-3 推荐方案**（已写入 r12_ensemble_stacking.md §4）：
   - 🥇 **方案 2** Stacking with Ridge meta over LOSO-OOF（zero new training，期望 +3 ~ +8）
   - 🥈 **方案 1** LightGBM + XGBoost + CatBoost simple mean（diversity 来自算法，期望 +2 ~ +6）
   - 🥉 **方案 6** Multi-horizon stacking（iter_002 已有原料，零成本，期望 +1 ~ +4）
3. **简单平均不够鲁棒**：Yirun Jane Street 1st 用 **middle-60% averaging**（trim outlier seeds），M. Kim Optiver 同样 trick；rank averaging 用于跨算法 calibration（但与我们 stateless 评测有兼容性问题）。
4. **NN 路线"免费 ensemble"**：Snapshot Ensemble (Huang 2017) + SWA (Izmailov 2018) 几乎零成本，PyTorch 原生支持；但前提是 NN 主路线要先打过 GBDT base，否则不值得做。
5. **不要用 BMA**：Yao et al. 2018 证明 M-open 下 stacking 严格优于 BMA；非负 simplex stacking 是首选。
6. **OOF metric 应直接是 cum_pnl**：scipy.optimize.minimize (Nelder-Mead) 在 5-8 维 weight 空间优化 OOF PnL，配合 bootstrap 防过拟合（方案 10）。
7. **不可用方案（违反硬约束）**：online learning、sliding-window retrain、sym embedding meta、跨 predict 调用维护 state、cross-sample rank averaging（test 顺序被打乱）。

---

## 8. R13 — Probability Calibration + Position Sizing 进阶（2026-05-06，T13 worker）

### 主交付物

- `r13_position_calibration.md` — **10 条进阶后处理方案 (A-J)**，每条含落地数学公式 / 在 LightGBM softmax 上能否直接用 / 来源 / 实现复杂度 / 期望 LOSO 增益；末尾给"如果只能跑 2 个方案"的明确 priority + 完整 7 天实施序列。

### 关键 paper 摘要（`r13_papers/`）

- `r13_papers/guo_temperature_scaling_2017.md` — Guo et al. 2017 (ICML)，T 单参 softmax 重缩放，**保 argmax 不变**；用 NLL 在 val 上拟合
- `r13_papers/kull_beta_calibration_2017.md` — Kull et al. 2017 (AISTATS/EJS)，3-param beta 校准 = log(s) + log(1-s) 两特征 logistic 回归，扩展 Platt 到非 sigmoid 失真
- `r13_papers/lopez_de_prado_bet_sizing_ch10.md` — AFML Ch10：z-score gating 公式 `z = (p - 1/K) / sqrt(p(1-p)); m = 2·Φ(z) - 1`；离散化；power form；meta-labeling
- `r13_papers/duan_ngboost_2020.md` — NGBoost，GBDT 输出全分布 (μ̂, σ̂)，可直接算 P(Δmid > +α)
- `r13_papers/conformal_prediction_2024.md` — Split conformal + Mondrian (group-conditional)，coverage 保证 1-α，3-class 自动产生 abstain 信号
- `r13_papers/cost_sensitive_threshold_tuning_2024.md` — Bayes EV 决策规则 `argmax_a Σ R[a,y]·p_y`，用 reward matrix（fee 已知）替代 hand-tuned T

### 核心 takeaways

1. **当前 T=0.55, δ=0.10 是 4 个弱点的耦合**：(a) 未校准 p；(b) 忽略 class prior 偏移和 Bernoulli 方差；(c) 全样本同阈值；(d) 浪费 multi-horizon 信息。每条方案各击破其一。
2. **Kelly 临界点 ≈ 当前 T**：fee/avg_move ≈ 1.25 → Kelly 阈值约 0.555；说明 grid-search 已 empirically 触底，进一步收益必须从 **校准 p 真实化** 或 **改换决策规则** 入手，不是再调 T。
3. **如果只能跑 2 个方案**：
   - 🥇 **Scheme D — LdP z-score gating**：`m = side·(2·Φ(z) - 1), z = (p_top - 1/3) / sqrt(p_top(1-p_top))`；20 行 patch，无需重训，**threshold_m 作为 1 hyper 替代 (T, δ)**。期望 LOSO h_10 +2 ~ +6。
   - 🥈 **Scheme E — Bayes EV with reward matrix**：`EV[a] = Σ_y R[a,y]·p_y; pred = argmax`，用 fee + 训练集 conditional means 构造 R。最简版本是训练 µ̂ regression head，`pred = 2 if µ̂ > +fee else 0 if µ̂ < -fee else 1`。期望 +3 ~ +8。
4. **推荐序列（7 天）**：A (temperature scale) → D (z-gate) → E (Bayes EV) → F (meta-labeling)。每步独立 ablation，可滚回。
5. **不推荐的方案**（违反硬约束）：online recalibration（stateless 限制）、per-sym calibration（test sym 可能新）、连续 Kelly 仓位（必须 0/1/2 离散）、cross-sample rank averaging（顺序被打乱）。
6. **Scheme F (meta-labeling) 是 +5~+12 的最高潜力 lever**，但要 2-3 天 + 引入 M2 模型，建议在 D/E 触底后再做。

