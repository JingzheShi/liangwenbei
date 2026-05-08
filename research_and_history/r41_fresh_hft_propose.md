# R41 — Fresh HFT Brainstorm（不带历史包袱）

> Worker：opus xhigh，**fresh-eye**（先**不**看 experiments / r3*.md / SUBMISSION_LOG）
> 完成日期：2026-05-07
> 输入：仅 `docs/data_schema.md`、`submission/RULES.md`、`submission/ENV_NOTES.md`、`CRITICAL_CONSTRAINTS.md`
> 时间预算：~40 min（实际 30 min 文献 + 10 min propose + 对比）

---

## §0 题目重述（无历史信息，仅来源于 schema/RULES）

- **数据**：5 sym × 100-tick 窗口（3s 一档）× 154 特征（量价 / LOB 10 档 / 订单流 6 类 / Kercheval-Zhang 衍生）
- **目标**：5 horizon (h=5/10/20/40/60) 三分类（涨/平/跌）
  - α 阈值：h=5/10 → 0.05%，h=20/40/60 → 0.1%
  - **类不平衡**：h=5 flat=76%，h=60 flat=54%
- **评分** = **5 个 horizon 中最高的 cum_pnl**
- **PnL 公式**：`(label-1)·Δmid - fee·|label-1|·|sum_mid_ratios|) / (mid_t+1)`，fee=0.0001 / side（roundtrip ~0.02%）
- **完美预测上限（单 session）**：h_60 = **2.74**（最高），h_5 = 0.57（最低）
- **硬约束（违反 = 0 分）**：
  1. date 评测时置 0 → 不能当 feature
  2. 测试点顺序被打乱 → Predictor 不能维护 state
  3. sym 0-4 但**可能含 OOD 股票** → sym-agnostic 模型，全局 normalization

---

## §1 文献调研要点（30 min WebSearch）

### 1.1 LOB 深度学习经典 + 最新（2024-2025）

| 工作 | 年份 | 核心 | 备注 |
|---|---|---|---|
| **DeepLOB** (Zhang Zohren) | 2018 | CNN+LSTM，FI-2010 上 F1=0.834 | 行业 baseline |
| **DeepLOBAtt** | 2021 | + attention, multi-horizon | DeepLOB 直系 |
| **Sirignano-Cont "Universal Model"** | 2019 | 一个 NN 跑全 stocks > per-stock 训 | **直接对应我们的 OOD 约束** |
| **Axial-LOB** | 2022 | Axial attention | 维度解耦 |
| **TLOB (dual attention)** | 2025 | spatial + temporal self-attention | [arxiv 2502.15757](https://arxiv.org/abs/2502.15757) |
| **LiT (LOB Transformer)** | 2025 | Top-20 levels, ms 级 reconstruct | crypto data |
| **Deep LOB Forecasting microstructural guide** | 2024-2025 | LOBFrame benchmark, CVML +244.9% | [arxiv 2403.09267](https://arxiv.org/abs/2403.09267) |
| **Differential Transformer for HFT** | 2023 | diff attention | MDPI |

### 1.2 微结构经典 + OFI

| 工作 | 核心 | 关键 takeaway |
|---|---|---|
| **Cont-Kukanov-Stoikov OFI** | signed queue change | **OFI 与短期 return 近似线性**；MLOFI（multi-level）显著 boost R² |
| **Hawkes self-exciting** | 6 类订单到达 = 互激发 | 比 Poisson 更 fit；**short-horizon 预测显著** |
| **Kyle's λ** (price impact) | dprice / dvolume | 流动性度量，作为 vol-feature |
| **Kercheval-Zhang 2015** | 144 LOB 衍生（已包含在我们 154d） | 题目特征源 |

### 1.3 标签 / Loss / 决策方法

| 思想 | 来源 | 适用性 |
|---|---|---|
| **Triple-Barrier Method** | Lopez de Prado 2018 | 用 vol-adjusted upper/lower barrier 替代固定 horizon → **label 噪声小 + 直接对应可交易性** |
| **Meta-Labeling** | Lopez de Prado | 一阶段挑 side（rule-based 即可），二阶段 ML 学 size（trade or not）→ **直接对应 PnL 公式的 (sign, |trade|) 分解** |
| **PnLLoss / SharpeLoss / MDDLoss** | [arxiv 2509.04541](https://arxiv.org/pdf/2509.04541)（2025） | 直接代替 MSE / CE，**比传统 loss 在交易指标上更好** |
| **Cost-Sensitive Bayes** | Elkan 2001 | 显式 cost matrix（含 fee）→ Bayes 决策最小化期望成本 |
| **Conformal Selective Classification** | [arxiv 2506.21802](https://arxiv.org/abs/2506.21802)（2025） | 拒绝低置信度预测 → 有界覆盖率保证 |

### 1.4 Kaggle 经验

| 比赛 | 第一名 trick | 我们能借鉴什么 |
|---|---|---|
| **Optiver Realized Vol 2021** | 反工程 time_id 顺序 + **676 个手工特征 from 3s snapshot** | 大量手工特征 → GBDT |
| **Optiver Trading at Close 2023** | feature engineering >> model；market urgency 特征最重要 | 同上 |
| **GResearch Crypto 2022** | **3 个独立 LightGBM 分别训 up/down/flat 市场** + 平均预测 | regime-conditioned ensemble |
| **JaneStreet 2024** | （信息不足；正在进行） | n/a |

### 1.5 跨股票泛化 / OOD

- **InvariantStock** ([arxiv 2409.00671](https://arxiv.org/html/2409.00671v1))：fundamental features 比 price features 更稳；feature selection 模块挑 invariant
- **FOIL** ([arxiv 2406.09130](https://arxiv.org/abs/2406.09130))：alternating 推断环境 + 学 invariant repr
- **Sirignano-Cont 2019**：pooled training + sym-agnostic features = OOD generalization 的 gold standard

---

## §2 12+ Disruptive Basic Ideas

> 所有 idea 都从"先决方法学选择"层面提出，不绑定具体超参。

### Idea-1：Regression on Δmid + EV-optimal Decision

**Target / Loss**：放弃 3-class CE，直接 **regress Δmid（或 PnL）**。
**Decision rule**：在 100-tick 窗口尾部直接计算每个 action ∈ {long, flat, short} 的期望 PnL = `pred_Δmid·(action-1) - fee·|action-1|·avg_mid`，argmax。

**理论合理性**：★★★★★
- 三分类用 α=0.05%/0.1% 阈值切分，但**最优交易阈值是 fee=0.02%，不是 α**，分类目标和经济目标错位
- 回归保留连续 magnitude → fee 阈值在决策时显式处理

**实施 simplicity**：★★★★★（LightGBM regression_l2 + numpy 后处理）

**风险**：Δmid 分布厚尾，需 huber loss 或 winsorize。

---

### Idea-2：Quantile Regression for Δmid + Risk-Aware Decision

**Target**：回归 q10、q50、q90 of Δmid（pinball loss）
**Decision**：long iff `q50 - 2·fee > 0` AND `q10 > -3·fee`（限制最坏情况）

**理论**：★★★★☆
- HFT 收益分布**显著厚尾**，平均值不能代表风险
- q10/q90 给出"赔付率界"，自动避免 vol-spike 时的灾难单

**simplicity**：★★★☆☆（LightGBM 支持 `objective=quantile`，但要训 3 个模型）

---

### Idea-3：Triple-Barrier 动态 Labels（Lopez de Prado）

**核心**：替换固定 horizon labels：
- 设 upper/lower barrier = ±k·realized_vol(W=20)（动态）
- 哪个先 hit = label 2 / 0；都没 hit = label 1
- vertical barrier = 60 ticks（最长 horizon）

**理论**：★★★★★
- 当前 labels 用固定 α 阈值 → vol 高时 |Δmid|>α 容易，vol 低时不可能 → labels 与 vol 强相关 = bias
- triple-barrier 把 label 转为"在 vol 标度下的可交易性"，**直接是 model 该学的东西**

**simplicity**：★★★☆☆（要写 cumsum 找 first-hit；可全量 numpy 向量化）

**风险**：会改变 label 分布，需重新对齐 5 horizon 的 mapping。可只生成一个 "h_60_triple_barrier" 标签。

---

### Idea-4：Meta-Labeling 架构

**Stage 1（side）**：简单 rule = sign(OFI) 或 sign(微回归 last-tick Δmid)
**Stage 2（size）**：ML 模型学"在 stage1 信号下，trade 是否 +EV"，二分类输出 0/1

**理论**：★★★★★
- 直接对应 PnL 公式的 `(direction) × (trade or not)` 因子分解
- 大量"flat"样本不浪费 stage2 模型 capacity（反正 stage1 给 flat 时直接 label=1）
- meta-label 是二分类 → 噪声小，更容易学

**simplicity**：★★★★☆（一个 stage1 rule + 一个二分类 LightGBM）

---

### Idea-5：PnL-Direct Loss（端到端 differentiable）

**Loss**：`L = -Σ_i E[PnL_i]` 其中 `E[PnL_i] = Σ_a p(a|x) · realized_pnl(a,Δmid_i)`
- 软 PnL 期望（用 softmax 概率加权 3 actions 的 realized PnL）
- 直接梯度下降优化训练集 cum_pnl

**理论**：★★★★★
- CE 假设我们想"分类正确"，但**真正的目标是 cum_pnl**
- soft-PnL 最大化 = 训练目标 ≡ 评分公式

**simplicity**：★★★☆☆（PyTorch 自定义 loss + 注意梯度对 prob 的传导）

**风险**：cum_pnl 大幅依赖少数极端样本 → 梯度方差大。建议加 Sharpe 项 or vol-adjust。

---

### Idea-6：单 horizon Specialist（h_60 only）

**核心**：评分 = MAX over horizons → 只需要**一个** horizon 强。h_60：
- 完美预测 PnL ceiling = 2.74（最高）
- flat=54%（最不平衡 → 信号最多）
- α=0.1%（容错最大）

策略：把所有训练 capacity 投到 h_60 单一目标。其他 horizon 全输出 1（不交易）。

**理论**：★★★★★
- 是评分公式 + 类分布 + 阈值的三重交汇推论，无脑选 h_60
- multi-task 反而稀释了 h_60 上的容量

**simplicity**：★★★★★（单模型）

**风险**：如果模型 h_60 真的崩塌，那就只能 h_60 崩；不像 multi-horizon 还能"换最好的 horizon"。需要谨慎做 LOSO 兜底。

---

### Idea-7：Leave-One-Sym-Out (LOSO) CV

**核心**：训练时永远用 4 sym，validate on 第 5 sym。所有 metric / threshold tuning / ensemble weight 都在 LOSO OOF 上做。

**理论**：★★★★★
- **唯一与平台 OOD sym 协议同构的 CV**
- 时间序列 split / random split 都不能模拟 OOD sym

**simplicity**：★★★★★（5 fold，sym 0..4 各 hold-out 一次）

**注意**：如果 LOSO best fold = 5 sym mean PnL；但平台测试集 sym 分布未知，可能某 sym 占比很高。建议 **5/5 fold 全正** 才提交（min-fold 兜底）。

---

### Idea-8：Universal Sirignano-Cont 范式

**核心**：**一个**模型训所有 5 sym 数据 pool；features 完全 sym-agnostic（无 sym ID，无 per-sym mean/std）；所有 normalization 用全局 robust statistics（median/MAD）。

**理论**：★★★★★
- Sirignano-Cont 2019 实证：pooled > per-stock，包括对训外 stocks
- 对应硬约束 §3：sym 可能 OOD → 必须 sym-agnostic

**simplicity**：★★★★★

**风险**：5 sym 之间 dynamics 不同会被平均掉。可在 feature 端用 within-window 自适应统计（z-score by 100-tick window mean/std）补救。

---

### Idea-9：OFI-Centric 特征工程（Cont-Kukanov-Stoikov）

**核心**：在现有 154 features 之上，重点构造 **multi-level OFI (MLOFI)**：
- 对 10 档每档计算 signed queue change
- 不同时间窗口（1, 5, 20, 60 ticks）累计
- 每档单独 + 加权平均（权重按档位距离衰减）
- 输出 ~30 个新 features

**理论**：★★★★★
- 文献证明 OFI 与短期 mid-return 关系**近似线性**且解释力远超 raw price/size
- 现有 154 特征中只有少量 _intst / _ind / _acc，没有显式 OFI

**simplicity**：★★★★☆（一系列 cumsum + diff，纯 numpy）

---

### Idea-10：Cost-Sensitive Bayes 决策（显式 fee 矩阵）

**核心**：训分类器输出 calibrated probabilities，再用 Bayes 公式：
- `argmax_a Σ_y p(y|x) · u(a, y)`，其中 `u(a, y)` = 该动作在该真实 label 下的实际 PnL（含 fee）

**理论**：★★★★★
- argmax 等价于 0-1 loss；交易成本不对称 → 需要 cost-sensitive
- 显式利用 fee（一个常数）和 |Δmid| 期望（条件期望）来重新决策

**simplicity**：★★★★☆（核心是 calibrated probs，可用 sigmoid 校准 + 已知 |Δmid| 条件期望表）

**风险**：calibration 不好会放大 fee 项；需要 LOSO calibration。

---

### Idea-11：Conformal Selective Classification（拒绝交易 = abstain）

**核心**：给每个测试点产生 conformal prediction set。
- 如果 set = {1}（only flat），直接输出 flat
- 如果 set = {0, 2} 或 {0,1,2}（不确定），也输出 flat（拒绝交易）
- 只在 set ⊂ {0, 2} 时才交易

**理论**：★★★★☆
- 直接控制"覆盖率 vs 准确率"trade-off，**有 finite-sample 保证**
- 自然解决"何时交易 vs 何时观望"问题，无需手调阈值

**simplicity**：★★★☆☆（需要 calibration set + 写 conformal score function）

---

### Idea-12：Volatility Regime Ensemble（GResearch 2022 winner）

**核心**：在 100-tick 窗口内计算 realized vol，分 3 regime（低/中/高）→ 训 3 个 LightGBM，inference 时按 regime 路由（或软门控加权）。

**理论**：★★★★☆
- HFT 在不同 vol regime 下的 dynamics 差异巨大（micro-trend vs mean-reversion 切换）
- GResearch winner 用此法

**simplicity**：★★★★☆（vol = std(close, 100); split into 3 buckets）

---

### Idea-13：Hawkes Intensity Features

**核心**：用 6 类订单（lb/la/mb/ma/cb/ca）的 _intst 时序，拟合 multivariate Hawkes：
- self-exciting intensity（lb→lb decay）
- cross-exciting（mb→cb 撤单导致市价单跟进）
- 输出 6 + 30 个 intensity-state features

**理论**：★★★★☆
- Hawkes 模型在 LOB clustering 上比 Poisson 显著好
- 可作为 GBDT 的额外特征

**simplicity**：★★☆☆☆（实时 Hawkes 拟合较慢；可用近似 EWMA self-excitation 替代）

---

### Idea-14：Time-of-day Features（time 字段被保留）

**核心**：constraint §2 说 time 保留实际时间戳。
- 加 `minutes_since_open`, `minutes_to_close`, `is_lunch_break`
- 开/收盘 vol 大于午盘 → 同样的 |Δmid| 信号在收盘时更可信

**理论**：★★★☆☆（intraday seasonality 实证强）

**simplicity**：★★★★★（但要确保 time 列在 config.feature 里，否则 DataFrame 不会送 time）

**风险**：time 不在默认 feature 列表，需要在 build_cache 阶段编码进 feature 列。

---

### Idea-15：Self-Supervised Pretrain + Fine-tune

**核心**：
- Pretrain：100-tick 窗口 → 预测下一个 tick 的 LOB（自回归）或 contrastive（同一 trajectory 不同切片 = positive）
- Fine-tune：5 horizon 三分类 head

**理论**：★★★★☆
- 标注 labels 高噪声（76% 都是 flat）；SSL 学到的 representation 能通用化
- TF-C 等成熟方法可借

**simplicity**：★★☆☆☆（需要训两阶段；GPU 时间长）

**风险**：pretrain 学到的 repr 可能 overfit train 5 sym 的微结构；要在 LOSO 验证。

---

### Idea-16：Multi-Horizon Vote Ensemble

**核心**：训 5 个 horizon 模型；inference 时让 5 个 horizon 预测互相 vote：
- 5 个都看好 long → 高置信度 long
- 4 看好 long、1 flat → 中置信度 long
- 不一致 → 不交易

汇总后**只输出某个 horizon 的最终决策**（评分仍按 max horizon 计）

**理论**：★★★★☆
- 多 horizon 是 weakly correlated 的"独立"信号源 → ensemble 降方差
- 跨 horizon vote 是天然 confidence proxy

**simplicity**：★★★★☆

---

### Idea-17：Asymmetric Threshold（up vs down 不同）

**核心**：观察现有 label 分布：down vs up 比例不对称（h_60 down=24.6%, up=21.8%）→ 用**两个独立阈值** thr_up / thr_down 而非对称 ±k·σ。

**理论**：★★★☆☆
- 暴跌行情比上涨更"暴力"→ 同样 prob 下 short 收益期望更大
- 直接 DE 优化 (thr_up, thr_down) 在 LOSO OOF 上

**simplicity**：★★★★★（仅 2D 网格搜索）

---

### Idea-18：Dual Head（direction + magnitude）

**核心**：模型输出两路：
- direction logits（3-class CE）
- magnitude regressor（|Δmid|, regression）

**Decision**：trade iff `pred_dir != flat AND pred_mag > 2·fee + ε`

**理论**：★★★★☆
- 因子分解：direction 易学，magnitude 难学（但只需 magnitude > fee 即可）
- 比单 regression 更鲁棒（direction confidence 单独门控）

**simplicity**：★★★★☆

---

### Idea-19：Adversarial Validation（train vs test 分布检测）

**核心**：建模 "X 来自 train 还是 test"二分类，AUC>0.6 表明分布漂移。
- 可用来识别哪些 features 漂移最严重 → drop
- 可用来发现 OOD sym 的特征签名

**理论**：★★★★☆
- 实战 Kaggle 经典；MLfinlab 重要工具

**simplicity**：★★★★★

---

### Idea-20：手工 600+ Features（Optiver winner 路线）

**核心**：直接借鉴 Optiver Realized Vol 1st：在 100-tick 窗口上计算 600+ summary statistics：
- bid/ask spread → mean, std, min, max, last, ratio_first_last, qm05, qm95
- imbalance → 同上
- volume-weighted average price (VWAP)
- 各阶 realized moment（vol, skew, kurtosis）
- micro-price = (a*bsize + b*asize) / (asize+bsize)
- multi-window slope（OLS over 20/50/100 ticks）

**理论**：★★★★★（feature engineering > model 是 tabular 共识）

**simplicity**：★★★☆☆（pandas rolling + numpy 向量化）

---

## §3 与 SUBMISSION_LOG 对比（速览表 + 后续 7-13）

> 仅看 SUBMISSION_LOG.md（不开 r3*.md / experiments/）

| Idea | 名称 | 我们做了？ | 出处 / 备注 |
|---|---|---|---|
| 1 | Regression on Δmid + EV-gate | ✅ | iter_013 BREAKTHROUGH（LOSO +36.23, 平台 +19.23） |
| 2 | Quantile regression + risk-aware | ⚠️ 没做 | T63_reg_sweep 名字暗示但不确定 |
| 3 | Triple-Barrier dynamic labels | ⚠️ 没做（重要！） | 没在 LOG 中出现 |
| 4 | Meta-labeling (side+size) | ⚠️ 部分？ | T72_binary_cascade 目录名暗示，但 LOG 未提 |
| 5 | PnL-Direct Loss（differentiable） | ⚠️ 部分？ | T57_pnl_aware_loss 目录名暗示；T60_rl_nn_pnl 也是；LOG 未明示成败 |
| 6 | Single horizon h_60 specialist | ✅ 已用 | iter_001/iter_007/iter_013 都聚焦 h_60 |
| 7 | LOSO CV | ✅ 普及 | iter_001+ 都用 LOSO |
| 8 | Universal pooled, sym-agnostic | ✅ | iter_001f Scheme D1 window-zscore |
| 9 | OFI-centric (MLOFI) | ✅ 部分 | R34 Stage 2 含 GOFI（但是否完整 Cont-Kukanov-Stoikov MLOFI 待查） |
| 10 | Cost-Sensitive Bayes（显式 fee 矩阵）| ⚠️ 没做（明确） | LOG 未提 |
| 11 | Conformal selective classification | ⚠️ 没做 | LOG 未提 |
| 12 | Vol regime ensemble (3 regime models) | ⚠️ 没做（明确） | LOG 未提；GResearch 经典 |
| 13 | Hawkes intensity features | ⚠️ 没做 | LOG 未提（R34 Stage 2 有 EWMA-OFI 但非 Hawkes） |
| 14 | Time-of-day features | ❌ 已试且失败 | git log: T78 ablation: regression+EV+6 time features → -0.72 vs iter_013 |
| 15 | SSL pretrain + fine-tune | ❌ NN 路线整体失败 | iter_007 negatives：DeepLOB / iTransformer / PatchTST / TimesNet 全 -16+ |
| 16 | Multi-horizon vote ensemble | ✅ T77（git log）+5.98 | git: T77: multi-horizon vote ensemble — +5.98 LOSO-equiv |
| 17 | Asymmetric threshold | ✅ | iter_013 thr_up=3.7e-4, thr_dn=1.6e-4（asymmetric） |
| 18 | Dual head（direction + magnitude）| ⚠️ 没做（明确） | LOG 未提 |
| 19 | Adversarial validation | ⚠️ 部分？ | T66_adversarial_val 目录暗示，LOG 未明示结果 |
| 20 | 600+ 手工 features (Optiver style) | ✅ 部分 | iter_001 Scheme B 已 2002-d features；R34 Stage 2 +113 features |

---

## §4 Top 5 我们没做的 idea（含具体实施建议）

### 🥇 Top-1：Triple-Barrier Labels（idea-3）

**为什么 disruptive**：
- 当前所有 work 都建立在"固定 α 阈值的三分类标签"上 → labels 本身有 vol-dependent bias
- triple-barrier 把"可交易性"直接编码进 label
- 重要：可与 iter_013 regression 范式 **正交**（对回归 target 也可生成 vol-adjusted Δmid_norm = Δmid / σ_recent）

**实施步骤**：
1. 写 `build_triple_barrier_labels.py`：对每个 t，扫描 [t+1, t+60]，找最早碰到 ±k·σ_W=20 的 tick；都没碰 → label=1
2. 用 k=2、3、4 三套 barrier 各生成一套 label，作为辅助多任务
3. 主目标仍 iter_013 regression on Δmid_norm；triple-barrier 作为"meta gate"——只在 triple-barrier label≠1 的样本上 trade
4. 在 LOSO OOF 评估 cum_pnl 增量

**预期增益**：+1~+3 LOSO-equiv（相对 iter_013 的 +36.23）

---

### 🥈 Top-2：Volatility Regime Ensemble（idea-12）

**为什么 disruptive**：
- iter_013 是单一 model（5 seed）；GResearch winner 实证：3 regime model > single model
- 100-tick 窗口内已有充足信息估算 regime（realized vol 分位数）

**实施步骤**：
1. 在 train 集上 compute realized_vol(W=100) → 三分位 cut → low/med/high regime
2. 用 iter_013 的 359-d 特征 + regression target，分别在 3 个 regime 子集上训 LightGBM
3. inference：算当前窗口 vol → 路由到对应 model（hard 路由），或软 sigmoid 加权（soft 门控）
4. EV-gate 用 regime-specific thresh（low-vol 区严，high-vol 区松）

**预期增益**：+0.5~+2 LOSO-equiv

**风险**：高 vol regime 是少数样本 → 模型欠训。可加重该 regime 的样本权（或 oversample）

---

### 🥉 Top-3：Cost-Sensitive Bayes Decision（idea-10）

**为什么 disruptive**：
- iter_013 用了 EV gate（thr_up/dn）但仍是手调 + DE 优化
- 完整 Bayes：`a* = argmax_a E_y[u(a,y)] = argmax_a Σ_y p(y|x) · pnl(a,y,|Δmid|)`
- 不是 thresh 调，而是把 fee 直接进决策公式

**实施步骤**：
1. 用 iter_013 的 regression model 输出 `pred_Δmid`
2. 用一个二阶 head（相同 backbone + 不同 head）输出 `pred_|Δmid|`（regression on abs value）
3. 决策公式：
   ```
   ev_long  = pred_Δmid - 2·fee·E[mid+1]
   ev_short = -pred_Δmid - 2·fee·E[mid+1]
   ev_flat  = 0
   action   = argmax(ev_long, ev_flat, ev_short)
   ```
4. 与 iter_013 的 EV-gate 对照：好处是 thresh 不再是单一标量，而是 per-sample 自适应

**预期增益**：+1~+2 LOSO-equiv，同时**显著缩小 LOSO→平台 gap**（iter_013 gap=-17 是 thr 过拟合）

---

### 🏅 Top-4：Conformal Selective Classification（idea-11）

**为什么 disruptive**：
- iter_013 LOSO=+36.23 → 平台=+19.23（gap -17）说明决策对**抽样分布**敏感
- conformal prediction 提供 distribution-free 覆盖率保证 → gap 自然缩小

**实施步骤**：
1. 用 iter_013 的 5-seed regression 得到 5 组 pred_Δmid → 计算预测方差作为 nonconformity score
2. 在 LOSO calibration set 上找 quantile threshold（例如 q=0.8 的 nonconformity score）
3. inference：方差 > threshold → 拒绝（输出 flat）；方差 < threshold → 用 EV-gate 决策
4. 校准 quantile q 在 LOSO 上 maximize cum_pnl

**预期增益**：+1 LOSO-equiv（不一定多），但**重点是缩小 platform gap**——更可控的覆盖率 → 私榜更稳

**simplicity 加分**：可在 iter_013 之上 zero-cost 加，无需重训

---

### 🏅 Top-5：Multi-Output Quantile Regression for Δmid（idea-2 升级版）

**为什么 disruptive**：
- iter_013 是 mean regression（Δmid_pred = E[Δmid|x]），但 HFT 收益分布**厚尾不对称**
- 直接预测 q10/q50/q90 of Δmid → 决策可考虑下尾风险

**实施步骤**：
1. LightGBM 支持 `objective=quantile`（指定 alpha=0.1, 0.5, 0.9）→ 训 3 个独立 model
2. Decision rule：
   - long iff `q_50 - 2·fee > 0` AND `q_10 > -3·fee`（避免 Δmid 概率分布左尾过深）
   - short 对称
3. LOSO 调整 q_low/q_high 的 alpha（不一定 10/90）

**预期增益**：+0.5~+1.5 LOSO-equiv，**同时降低 worst-case fold 损失**（iter_007 fold 2 是负的；quantile 可能修复）

---

## §5 关键 cross-cutting 观察

1. **iter_013 是当前最强（LOSO +36.23, 平台 +19.23）**——任何新 idea 都应在 iter_013 之上叠加 / 替换
2. **gap -17（LOSO vs 平台）是当前最大问题**——top 3, 4 都是缓解 gap 的方向（cost-sensitive Bayes 不需 thresh tuning；conformal 给 distribution-free 保证）
3. **NN 系（DeepLOB / TLOB / iTransformer）一致失败**——说明问题主要在 features / target / decision 而非 model class；GBDT + 好 features 已是 SOTA
4. **R34 Stage 2 的 sym-invariant features 是关键 lever**——R34 已挖掘了 quantile rank / GOFI / vol-burst / Kyle / EWMA-OFI；下一步是**Hawkes self-exciting features**（idea-13）和 **MLOFI 完整版**（idea-9）作为 Stage 3
5. **Triple-barrier labels 与 regression 正交，最有潜力**——可重新生成训练 target，是少数能在 fundamental 层面突破的 idea

---

## §6 提交建议（如果只能再提 5 次）

| 提交序 | 基底 | 加的 idea | 预期 LOSO-equiv |
|---|---|---|---|
| iter_014 | iter_013 | + Conformal selective（top 4，不重训）| +37 ~ +38；**关键：gap 缩小** |
| iter_015 | iter_013 | + Cost-Sensitive Bayes（top 3，加 mag head）| +37 ~ +39 |
| iter_016 | iter_015 | + Quantile regression（top 5，替换 mean） | +38 ~ +40 |
| iter_017 | iter_016 | + Triple-barrier labels（top 1，新 target） | +40 ~ +43 |
| iter_018 | iter_017 | + Vol regime ensemble（top 2，3 regime） | +42 ~ +45 |

私榜 = 公榜最后一次，所以**最稳的（top 4 conformal）放靠后某次 + 留至少 1 次回退提交**。

---

## §7 文献来源

1. [Universal features of price formation, Sirignano & Cont 2019](https://arxiv.org/abs/1803.06917)
2. [DeepLOB: Deep Convolutional NN for LOB, Zhang et al. 2018](https://arxiv.org/pdf/1808.03668)
3. [TLOB: Dual Attention Transformer for LOB, 2025](https://arxiv.org/abs/2502.15757)
4. [Deep LOB Forecasting: a microstructural guide, 2024-2025](https://arxiv.org/abs/2403.09267)
5. [Cont-Kukanov-Stoikov OFI](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159)
6. [Forecasting High-Frequency OFI using Hawkes, 2024](https://arxiv.org/html/2408.03594v1)
7. [Lopez de Prado, Advances in Financial ML, 2018](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
8. [Finance-Grounded Optimization for Algo Trading, 2025](https://arxiv.org/pdf/2509.04541)
9. [Classification with Reject Option via Conformal, 2025](https://arxiv.org/abs/2506.21802)
10. [InvariantStock: Invariant Features Across Markets, 2024](https://arxiv.org/html/2409.00671v1)
11. [Optiver Realized Vol 1st-place Solution](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
12. [Optiver Trading at Close 1st-place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
13. [GResearch Crypto Forecasting Wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)

---

**RESULT**: `task=r41_fresh_hft metrics={n_ideas=20, n_undone=8, top5_undone=triple_barrier|vol_regime|cost_sensitive_bayes|conformal_select|quantile_regression}`
