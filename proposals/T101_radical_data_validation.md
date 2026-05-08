# T101 — 颠覆性数据 / 验证策略改进 (radical data + validation rethink)

**作者**: Worker (opus, research-only proposal, 不训模型)
**日期**: 2026-05-08
**输入约束**: 只关注 data 与 validation 策略；model / loss / decision rule 不在范围
**核心 baseline**: iter_014 LOSO-equiv +36.22, 平台预期 ≈ +20.7 (transmission 0.70)
**目标**: 缩小 LOSO→Platform 30% 损耗；从 +20.7 提到 +25 ~ +28

---

## Part 1 — 现状诊断（First-Principles 重新审视）

### 1.1 LOSO-equiv → Platform 0.70 透传系数的最深根因

T80 校准把"LOSO 14.61 → 平台 4.07"和"LOSO 36.23 → 平台 19.23"两个锚点连成 ratio 0.70。
T83 已排除"thresh 在 442k test 上过拟合"是主因——所有 robust 策略 (kfold5_date / bootstrap15 / quantile_grid / ensemble_avg_3) 在 unbiased eval 上只差 0.30 PnL，远小于 −17 的绝对 gap。

**T83 关键发现**: 把 442k test 切成 half-A (date 96-107) 和 half-B (date 108-119)：
- iter_013 thresh 应用于 half-A: PnL = **20.75**
- 同 thresh 应用于 half-B: PnL = **15.40**
- **Half-A 比 Half-B 高 26%**——同一只股票、相邻 12 天，pnl/day 差 26%
- 即使 oracle thresh on half-B alone 也只 +15.46（无 lever 可挽）

**结论**: 0.70 透传系数的本质是**未对齐的时间漂移 (date drift)**。

平台 test 集 vs 我们 local test (date 96-119) 的关系：
1. 公榜 LOSO -2.23 ~ -3 是**已经在 date 96-119 内观测到的衰减**的延伸
2. 如果平台 test 跨 date 120+（比赛说明未明示但极有可能），**衰减会继续**
3. Our val 在 date 76-79（紧贴 train），抓不到这条衰减曲线

**反推**: 用 24-day local test 内的 -26% pattern 外推到平台 test：
- 假设平台 test 跨 date 120-143（24 天）
- Half-A→half-B 趋势继续：PnL 衰减 ~26% × (1 + γ)，γ ∈ [0.5, 1.0]
- 即平台 PnL ≈ local-half-B × (1 - γ × 0.13) ≈ 15.40 × 0.85 ≈ **+13**
- 公开 ratio 0.70 隐含的 time-OOD 损失 ≈ 36 - 19 = 17 ≈ exactly this

**这意味着**：我们一直在为"过去时段 (date 0-79)"优化模型，
**val 选 best_iter 也在过去时段**，**aug_a 也只是 per-(sample,feat) 高斯化**——
完全没有针对"未来时段"的任何归纳偏置。修这条是最大的 lever。

### 1.2 V4 split 是不是最优？

V4 = train 0-75 + val 76-79（仅 4 天）+ test 96-119。

**V4 的 3 个深层缺陷**：

| # | 缺陷 | 量化 |
|---|------|------|
| **a** | val 仅 4 天，best_iter 噪声极大 | iter_013 5 seed best_iter ∈ [70, 124]，方差 50+ rounds |
| **b** | val (76-79) 与 train (0-75) **零 gap**，与 test (96-119) 隔 16 天 | val 选出来的 best_iter 是"对训练分布过拟合最少"的点，不是"对未来分布最优"的点 |
| **c** | dates 80-95 的 16 天数据**完全闲置**——既不在 train，也不在 val，也不在 test | 16/120 = **13% 数据浪费**，且 80-95 是最接近 test 分布的早期信号 |

**T58 walk-forward CV 已经写好框架**（W1: 7-fold sliding，W2: expanding，W3: 大块 expanding），但 iter_011 后的所有提交全都退回单 split V4——T58 的成果**未被生产化使用**。

**iter_011 用 V4 替代 sym-内 cross val 时声称 "+1.42 LOSO-equiv"**——但那个 ablation 是在 sym-OOD 折叠下做的；当 iter_009+ 切到 full-5sym pooled training 后，V4 的相对优势没重新校准。**V4 在 full-pooled 训练下可能已不是最优**。

### 1.3 aug_a [0.80, 1.20] per-feature 是不是真"randomize sym dynamics"?

**T26 设计意图**：模拟"训练外 sym 特征量级有±20% shift"。
**实际做的事**：`X *= np.random.uniform(0.80, 1.20, size=X.shape)`——**每个 (sample, feature) 独立采样**。

**与目标对齐度评估**：

| 真实 sym OOD shift 性质 | aug_a 是否模拟 |
|------|------|
| 同一 tick 上 bid1..bid10 全同向偏移（价格 baseline 平移） | ❌ 各档独立扰动，破坏档间一致性 |
| size 类特征整组放大/缩小（流动性差异） | ❌ size 各档独立，破坏深度结构 |
| 跨 tick 的时序相关 noise | ❌ 同一 (t, feature) 跨样本独立 |
| 局部 microstructure shape（spread / OFI）保留 | ❌ spread = ask1*scale_1 - bid1*scale_2，被破坏 |
| 长尾"black swan"分布 | ❌ U[0.8,1.2] 是窄 bounded |

**结论**: aug_a 在数学上等价于 **per-element 高斯平滑 + LightGBM 自带的 feature_fraction bagging**，**几乎不模拟任何真实的 sym/date OOD shift**。
T31 phase1 单 seed 显示 [0.75,1.25] 比 [0.80,1.20] 强 +0.81，但 5-seed ensemble + DE thresh **反转**（更宽 aug 提升单模型 robustness 但降低 ensemble 多样性）——
说明 aug_a 当前的边际收益已经触底，**进一步收益必须换 aug 的"shape"**。

### 1.4 5-seed 是否充分利用了数据？

**当前**: 5 个 hyperparam-diverse 配置（不同 num_leaves / ff / bf / l2），各跑 1 次，共享同一份 train 数据。

**未用的多样性**：
- ❌ **同 hyperparam 不同 seed**: 真随机种子未单独 ablate
- ❌ **bootstrap 样本**: 每个 model 都看完全相同的 train（仅 ff/bf 内部 random sample）
- ❌ **不同时间窗口**: 5 个模型都用 date 0-75
- ❌ **不同 sym 子集**: 5 个模型都用全 5 syms（iter_009+ 后）

**5 hyperparam 配置实际"独立性"评估**: 看 T38 报告——M1 (iter_005b) 和 M3 (revol-only) cross-corr 0.97+，diversity 极弱。这意味着 **5-seed ensemble 实际就是 1.x seed**。

### 1.5 5 sym × 120 day × AM/PM × 2001 tick 结构未利用之处

| 维度 | 未充分利用的结构 |
|------|------|
| **sym** | T8 已发现 sym=2 是 outlier (large-cap/ETF)，sym=4 异常正贡献——但**没有把这些 outlier 排除/上下采样**做 sym-balanced training |
| **date** | dates 80-95 闲置；date 60-79（最近）和 date 0-19（最远）权重相同 |
| **AM/PM** | 两个 session 完全混合训练；早盘 9:40 与晚盘 11:20 的 microstructure 不同（Worker 调研提到），但**没有按 session 分组**做 stratified val |
| **tick (2001)** | 100-tick 滑窗起点 = tick 99..1940；前 99 tick 的 warmup 数据**完全没贡献**到任何预测点（除作为窗口前段输入） |

### 1.6 LOSO sym OOD 的本质 vs Platform OOD

**LOSO 测试**: train 4 known syms → test 1 known sym。模型见过这只 sym 的 *sibling distributions*（其他 4 syms 的 0-79 day），仅是**未见过这只 sym 的 96-119 day**。

**Platform 测试**: train 5 known syms → test "可能不来自训练集 5 只股票"。**真正的 distribution 全新**（一个我们没见过任何 day 的 sym）。

**严重程度对比**：
- LOSO ≈ "你认识 4 只哥哥姐姐，预测他们的弟弟妹妹"
- Platform ≈ "你认识 5 个家庭，预测一只完全陌生家庭的孩子"

**这是 LOSO inflated +6.46 的内在原因**：iter_002 → iter_013 LOSO 涨 +21.62，平台只涨 +15.16，
gap 增量 −6.46 来自更强的模型把 sym 内部 microstructure 学得更深，
反而对**真正陌生的 sym** 泛化变差。

**特别注意**: T80 数据中 iter_002 LOSO+14.61 时 gap=−10.54，iter_013 LOSO+36.23 时 gap=−17.00。
**Gap 随模型变强而扩大**——这意味着继续榨取 LOSO-equiv 的边际收益正在递减，0.70 系数将进一步下降。

---

## Part 2 — 颠覆性 Trick 提案（10 个，按预估收益排序）

> 标记：基础度=Trick 是否标准/无创新；颠覆性=与当前实践偏离程度；难度=实施成本；预估 LOSO=对 +36.22 的相对增益；风险=潜在回退或假阳性。

### 提案 1 ⭐⭐⭐: Late-Date Val with Embargo (V5 split)

**1 句描述**: 把 val 从 date 76-79 (4 天，紧贴 train) 改为 **date 89-95 (7 天，距 train end 9 天 embargo)**——early stopping 在更接近 test 分布的窗口上选 best_iter。

**为什么基础**: 金融 ML 标配的 "purged + embargo" 验证（López de Prado AFML Ch7）。

**为什么颠覆性**:
1. 当前 val=76-79 与 train=0-75 **零 gap**——best_iter 在选"刚刚还在分布内"的迭代数，对应"未来 24-44 天"的最优 iter 必然偏大（过拟合训练分布）
2. T83 已证明 half-A/half-B 26% 衰减是真实的，但**我们从来没在 val 阶段抓到这条衰减**
3. iter_013 5 seed best_iter ∈ [70, 124]——50+ rounds 噪声主要来自 4-day val 太短；7-day + embargo 后 best_iter 方差应砍半
4. **如果 V5 把 best_iter 平均拉低 30 rounds**（即"未来分布"上更早收敛），LOSO-equiv 单方向涨 +1 ~ +2，更重要的是**transmission ratio 从 0.70 → 0.78**，平台增益放大 11%

**实现难度**: 30 行 patch（修改 `train_regr.py:92-114` 的 `build_v4_split`），重训 5 seeds 约 15 min on GPU。

**预估 LOSO 改进**: +1 ~ +2 LOSO-equiv（直接），但**平台真增益期望 +2 ~ +4**（透传系数改善）。

**风险**:
- 训练数据从 76 天减到 89 天 — net **更多** train（76+9=85 vs 76）；正向无副作用
- 如果未来分布与 89-95 也不一致（又一层 drift），收益可能减半
- DE thresh 在 442k test 上重新调；不影响这个 trick 的本质收益

---

### 提案 2 ⭐⭐⭐: Linear Date-Decay Sample Weight (replace class-balanced)

**1 句描述**: 把当前 class-balanced 样本权重 `w_i = N/(K·N_class[y_i])` 替换为 **`w_i = (1 + 1.5 × date_i / 79)`**，即最近一天权重是最早一天的 2.5 倍。

**为什么基础**: 时间衰减是金融 TS forecasting 经典 trick (Optiver Trading at Close 2023 1st HYD 用 12-day retrain 5 次实现等效)。

**为什么颠覆性**:
1. T83 证明 half-A→half-B PnL 衰减 26%——**距离 train-end 越远，PnL 越差**
2. 当前 train 把 date 0 和 date 75 等权重处理；date 0（4 个月前）的 microstructure 应该被忽略，date 60-75 应该主导
3. 4 个月数据 vs 1 个月数据等权——这是**反直觉的过强假设**，金融数据漂移在月级时间尺度发生
4. **关键**: 与提案 1 (V5) 互补——V5 改 val 使 best_iter 更对 future-OOD，提案 2 改 train 使 model 本身更对 future-OOD
5. 与 class-balanced 互补：改成 `w_i = class_balanced × (0.6 + 1.4 × date/79)` 保留类别平衡，叠加日期加权

**实现难度**: 1 行 patch (`sw_tr = sw_tr_class_bal * (0.6 + 1.4 * train_full["date"][m_t] / 79)`)。

**预估 LOSO 改进**: +1 ~ +3 LOSO-equiv，**平台真增益 +1 ~ +2**。

**风险**:
- T57 试过 PnL-aware sample weight (linear/sqrt/cap) 全负 (-3 ~ -11)，但那是 PnL-magnitude weight，**不是 date weight**——根本不同的 lever
- 如果训练分布对早期 date 有"长 horizon 学习信号"被砍掉，可能轻微负向（监控 cross-corr）
- 系数 1.5 是凭直觉；建议 sweep ∈ {0.5, 1.0, 1.5, 2.0, 3.0}（5 model × 5 系数 = 25 次训，快速）

---

### 提案 3 ⭐⭐⭐: Sym-Coupled Group Scale Augmentation (replace per-element aug_a)

**1 句描述**: 当前 aug_a 对 (sample, feature) 独立采样 U[0.80, 1.20]；改为**每 sample 采样 3 个 group scale (price_group / size_group / orderflow_group)**，组内特征共享同一 scale，且 scale 间相关性可调。

**为什么基础**: Group-aware data augmentation 在 NLP / CV 是标配；金融特征天然分组（OHLC vs LOB-prices vs LOB-sizes vs OFI）。

**为什么颠覆性**:
1. 真实 sym OOD shift = **整组特征 correlated 偏移**（不同股票价格 baseline 不同 → 所有 price 类同步偏移；流动性不同 → 所有 size 类同步偏移）
2. 当前 per-element 独立扰动**破坏了 cross-feature 相关结构**（spread = ask1−bid1，独立扰动 ask1 和 bid1 相当于把 spread 变成噪声）
3. 给定 354 维特征，按官方 schema 自然分组：
   - **price group** (~80 dim): close/open/high/low, bid1-10, ask1-10, midprice1-10, bid_diff/ask_diff, avgbid/avgask
   - **size group** (~50 dim): bsize1-10, asize1-10, totalbsize/asize, volume_delta
   - **flow group** (~24 dim): _intst, _ind, _acc 6 类 × 4
   - **derived group** (~200 dim): spreads, rates, ewma, mlofi 等 — 留 untouched
4. 实现：
   ```python
   s_price = rng.uniform(0.85, 1.15)  # 每 sample 一个标量
   s_size = rng.uniform(0.70, 1.40)   # size 漂移更大（流动性差异更大）
   s_flow = rng.uniform(0.80, 1.20)
   X[:, price_idx] *= s_price
   X[:, size_idx] *= s_size
   X[:, flow_idx] *= s_flow
   ```
5. **Stage 1 实验信号**: T29 aug_a h_40 ablation 显示 aug 效果在 [0.80, 1.20] saturate；本提案是**改变 aug 的 shape，而非 magnitude**

**实现难度**: 60 行 patch（构建 feature group masks + 改 `aug_a_scale`）；重训 5 seeds 约 15 min。

**预估 LOSO 改进**: +1.5 ~ +4 LOSO-equiv（更重要：**改善 platform 上"陌生 sym"的真实 OOD 表现**——这是 0.70 系数的另一半 root cause）。

**风险**:
- 如果 size group scale 上限过宽（>1.4），volume_delta 等高方差特征会出现极端值，破坏 LightGBM 树结构（监控 leaf split count）
- 与 aug_a 不能简单叠加（需 ablation 二选一或叠加更弱版本）
- **关键 ablation**: 单独训一个 sym=2 sym=4 holdout 验证泛化是否提升（T8 已知这两 syms 是关键）

---

### 提案 4 ⭐⭐: Temporal Block K-fold for Best_iter Averaging

**1 句描述**: 替换 V4 单 split 早停，改为 **5-fold date-block CV** (train [0-15] [16-31] ... [64-79]，每折 holdout 一段做 val) 选 best_iter，最终在 0-79 全数据上 retrain 到 5 折平均 best_iter。

**为什么基础**: K-fold CV 标配。

**为什么颠覆性**:
1. iter_013 5 seeds best_iter ∈ [70, 124] 方差高度依赖 4-day val 噪声
2. 5-fold date-block 把 best_iter 估计的方差从单 1 个 4-day 折降到 5 个 16-day 折的 mean——**直接砍 ~70% noise**
3. 同时给出 **best_iter 的不确定性**（std），用于 ensemble 是否值得加额外 seed 的决策
4. 5 折训出的 5 个 model 也可作为 ensemble base（与 5-seed 配合，潜在 25-model ensemble）
5. **重要差异 vs T58**：T58 walk-forward CV 是为了估计"OOD 性能"，本提案是为了"选 best_iter"——目标不同

**实现难度**: 100 行 patch（重写 `build_v4_split`）；总训练 5×5 = 25 次 LightGBM × 60s = 25 min on GPU。

**预估 LOSO 改进**: +0.5 ~ +2 LOSO-equiv（best_iter 收紧）；**平台 +0.4 ~ +1.5**。

**风险**:
- K-fold 5 倍训练成本（25 次 vs 5 次），但仍 < 30 min on GPU 可接受
- 需要小心 fold 边界数据泄漏（embargo 1 day 即可）

---

### 提案 5 ⭐⭐⭐: 5-LOSO + 1-Full Ensemble (re-introduce LOSO models)

**1 句描述**: iter_009+ 切到 full-5-sym training 后丢弃了 LOSO 训练；提案**保留 5 个 LOSO 模型 (each holds 1 sym)** + 1 个 full-5sym 模型，**6 模型 prediction 平均**作为 final.

**为什么基础**: Bagging via different OOD experiences。

**为什么颠覆性**:
1. 最反直觉的点：**LOSO 训练单 model 弱**（少 20% data, 少 1 个 sym），但 5 个 LOSO model 各自见过不同的"4 sym → 1 sym 泛化"经验
2. **Platform OOD（陌生 sym）≈ 5 个 LOSO 模型的 OOD（陌生哥哥姐姐）的随机 mixture**——所以 LOSO 模型平均后对陌生 sym 期望泛化更稳
3. 当前 full-5sym 模型 implicitly 假设"test sym ∈ train syms"——T80 显示这是 LOSO 系数从 1.0 → 0.70 的关键
4. **关键数学论据**：full-5sym 模型对每个训练 sym 都 "memorize" 一些；陌生 sym 上等于失去全部 memorization。LOSO 模型把每个 sym 的 memorization 限制在 4/5，但**6 模型平均后 memorization 抑制 ~6 倍 lighter**
5. 完全在硬约束内：每个 model 都 sym-agnostic（不喂 sym ID），只是训练数据组成不同

**实现难度**: 120 行 patch（重启 LOSO 训练 pipeline，且与 V4 split + iter_013 features 兼容）；6 model × 60s × 5 seeds = 30 min on GPU。

**预估 LOSO 改进**: +1 ~ +3 LOSO-equiv（local），**平台真增益 +2 ~ +5**（系数 0.70 应改善到 0.85+）。

**风险**:
- LOSO 模型单独弱（已知 T11/T26 的事实）；ensemble 平均可能 dilute full-5sym 的强势——需要 grid search 权重 (例如 LOSO 模型平均 × 0.4 + full × 0.6)
- 与 V5 split (提案 1) + 提案 2 (date-decay) 兼容
- **核心论据需验证**：LOSO model average 在 sym=2、sym=4 这类 outlier sym 上的表现 vs full model，应做 ablation

---

### 提案 6 ⭐⭐: Bootstrap Bagging Replacing Hyperparam-Diverse Seeds

**1 句描述**: 当前 5 seeds = 5 hyperparam configs (固定 seed)；改为 **15 个 bootstrap × 同一 hyperparam config**（80% sample with replacement）作为 ensemble base，再 cross-arch 用 hyperparam 多样性。

**为什么基础**: Bagging (Breiman 1996)。

**为什么颠覆性**:
1. T38 证明 5 seeds cross-corr > 0.97（diversity 几乎全无）——hyperparam diversity 没产生真 ensemble 效益
2. Bootstrap sample diversity 是更基础的 source of randomness——15 boot + 1 hyperparam 比 5 hyperparam + 0 boot 多得多
3. **特别**: 与 LightGBM 自带的 `bagging_fraction=0.8` 重复但不冲突——bagging_fraction 在每轮内部重抽，bootstrap 在数据集级 fix 抽样
4. 2x ensemble size 但训练时间 3x（15 vs 5）——可接受
5. 与提案 5 (LOSO ensemble) 正交叠加

**实现难度**: 30 行 patch（外层 loop）；总训练 15 model × 60s × 1 hp = 15 min。

**预估 LOSO 改进**: +0.5 ~ +1.5 LOSO-equiv；**平台 +0.4 ~ +1**。

**风险**:
- 失去 hyperparam diversity 的 robustness 优势（如果某 hp 突然在 test 上失败，没有 fallback）
- 折衷方案：**3 hp × 5 boot = 15 model**

---

### 提案 7 ⭐⭐: Adversarial-Validation-Reweighted Sampling

**1 句描述**: 训一个二分类器 `is_test_distribution` (label=1 if date≥80, label=0 else)，用 OOF proba 给每个 train 样本 (date 0-79) 一个"看起来像 test 的程度" score，**用这个 score 做样本权重**。

**为什么基础**: 比赛圈经典 (Kaggle "adversarial validation" - Tunguz/Bestfitting 2018+)。

**为什么颠覆性**:
1. T66 试过类似但 "marginal"——失败原因可能是用 random val 而非 date-block val
2. 修正版：用 LightGBM 二分类训 `P(date ≥ 80 | x)`，用 isotonic 校准成 proba，作为 sample weight
3. 与 提案 2 (linear date-decay) 区别：date-decay 是**先验假设**线性衰减；adversarial val 是**数据驱动**——让模型自己识别哪些训练样本最像未来分布
4. 副产物：识别出 train-test 分布最不同的 features，可考虑 KS-fail filter（但 T75 已经 drop 了 11 个）
5. **关键改进 over T66**：用 5-fold date-block CV 训这个二分类器，避免 single split 噪声

**实现难度**: 80 行（训 binary classifier + isotonic + apply weights）；总时间 ~10 min。

**预估 LOSO 改进**: +1 ~ +2.5 LOSO-equiv；**平台 +0.7 ~ +1.7**。

**风险**:
- 与 提案 2 不能简单叠加（两个权重都 weight 同一 sample），择一或乘性结合
- 如果二分类器学到的是 trivial signal（如 date 本身），相当于温柔版的 date-decay；**必须 drop date 字段后训**

---

### 提案 8 ⭐⭐: Cross-Sym Mixup Restricted to Same-Date Pairs

**1 句描述**: 训练时随机抽样 (sym_a, t_a) 和 (sym_b, t_b)，**当 date_a == date_b** 时，做 feature 和 target 的 mixup λ=0.3：`X = (1-λ) X_a + λ X_b, y = (1-λ) y_a + λ y_b`。

**为什么基础**: Mixup (Zhang 2018) + cross-sym pooling。

**为什么颠覆性**:
1. T14/T52 试过 cross-sym mixup，**marginal**——但他们没限定 same-date
2. 同 date 不同 sym 共享市场 macro regime（同一交易日的板块联动 / 大盘 beta），mix 后样本有真实物理意义（"另一只股票在同一时刻的状态"）
3. 跨 date mix 没物理意义，会引入 regime noise—T14/T52 失败可能是这个
4. **与提案 3 (sym-coupled group scale) 互补**：sym-coupled 只在 feature 上模拟跨 sym shift，cross-sym mixup 在样本级真用了别的 sym 的数据
5. λ=0.3 偏弱保留主体；mix ratio 50% 的样本（即 50% mix, 50% 原样）

**实现难度**: 60 行（构建 same-date sym pair index，concat mixed batch）。

**预估 LOSO 改进**: +0.5 ~ +2 LOSO-equiv（小，因为 T14/T52 已部分 explored）；**平台 +0.4 ~ +1.4**。

**风险**:
- 与 aug_a / 提案 3 不要同时用最强配置；mix 已是强 augmentation
- 如果 same-date 同 sym pair 不多（5 sym × 1 date = 5 pair / sample），样本多样性受限

---

### 提案 9 ⭐: Tail-Date Curriculum (Pretrain-Finetune)

**1 句描述**: 两阶段训练：(1) date 0-39 大量训 200 round 学 baseline microstructure；(2) **freeze 树结构** + date 60-79 fine-tune 100 round（仅 boost residual / leaf weight 更新）。

**为什么基础**: Curriculum learning + transfer learning 标配。

**为什么颠覆性**:
1. LightGBM 不支持 native fine-tune，但**可以用 `init_model` 参数从 stage1 booster 继续训练 stage2 数据**
2. Stage1 学"普世规律"（任何时段 LOB 微结构都成立的特征 → label 关系）
3. Stage2 学"近期 regime"（最近 20 天的特定 alpha）
4. 与提案 2 (linear date-decay) 的核心区别：
   - date-decay 单次训练把权重平滑施加
   - curriculum 用两阶段，让早期 date 完全主导树结构形成、晚期 date 完全主导细节
5. iter_013 → iter_014 用 V4 walk-forward 实质上是 simplified curriculum；本提案是更显式的两阶段版本

**实现难度**: 100 行 patch（stage1/stage2 拆分 + init_model 参数）。

**预估 LOSO 改进**: +0.5 ~ +1.5 LOSO-equiv；**平台 +0.4 ~ +1**。

**风险**:
- LightGBM `init_model` + `keep_training_booster=True` 行为有些 fragile，不同版本 API 不同
- 与 提案 2 部分重叠；二选一

---

### 提案 10 ⭐⭐: External Pretraining on Optiver/FI-2010 + Fine-tune

**1 句描述**: 在 **公开 LOB 数据集 (FI-2010 / Optiver Realized Vol)** 上预训练同架构 LightGBM regression model 学 "通用 microstructure → mid-price move" 关系，然后用我们 354-d 特征 + 良文杯 train 数据 fine-tune。

**为什么基础**: Transfer learning 在 NLP / CV 标配；但**金融 LOB 上几乎没人做**（每个比赛闭门）。

**为什么颠覆性**:
1. 我们的 5 syms × 120 day train 是相当小的样本（~480k 训练点 with 100-tick window），加上 OOD 限制更难
2. FI-2010 是公开 5-stock × 10 day Helsinki LOB 数据集（~4M ticks），Optiver Realized Vol 数据 ~3M ticks
3. **如果跨市场 / 跨时间的 LOB → mid-price 关系存在**（Kolm 2023 "Deep OFI" 的数百股 cross-stock 模型证明这个 transfer 是可行的），预训练可以提升小样本 generalization
4. 关键挑战：特征列对齐——我们 154 维 schema vs FI-2010 144 维 vs Optiver 不同——但**核心 30 维 (LOB top 5 prices+sizes, OFI proxies) 可以对齐**
5. **绕过 0.70 transmission 系数的本质方法**：现在 LOSO→Platform gap 很大程度因为我们模型只见过 5 syms；预训练让模型见过 100+ syms

**实现难度**: 高 (3-5 day work)
- 数据下载（学术加速代理 ~1 hour）
- Schema mapping (1 day)
- 预训练 + fine-tune 实验（2 day）
- 提交 sanity check（1 day）

**预估 LOSO 改进**: +0 ~ +5 LOSO-equiv（高方差，因为转换可能完全失败也可能很强）；**平台 +0 ~ +4**。

**风险**:
- 高研发成本 (vs 提案 1-9 的 1-3 hour)
- FI-2010 / Optiver 标签定义不同（FI-2010 mid-price direction with different α），需特殊处理 fine-tune target
- **提交模型大小限制 2GB**——需确认 fine-tuned LightGBM 不超
- **强烈建议放在 stretch goal，不是首选**

---

## Part 3 — 推荐执行顺序与组合

### 优先级矩阵

| 提案 | 难度 | 单独预估 LOSO | 与基线兼容 | 推荐优先级 |
|------|------|--------|----------|-----------|
| 1. V5 late-date val + embargo | 低 | +1~+2 | ✅ | **🥇 必做** |
| 2. Linear date-decay weight | 极低 | +1~+3 | ✅ | **🥇 必做** |
| 3. Sym-coupled group aug | 中 | +1.5~+4 | ✅ | **🥇 必做** |
| 5. 5-LOSO + 1-full ensemble | 中 | +1~+3 | ✅ | **🥈 强推** |
| 4. Temporal K-fold best_iter | 低 | +0.5~+2 | ✅ | 🥈 强推 |
| 7. Adversarial val reweight | 中 | +1~+2.5 | ⚠️ 与 #2 重叠 | 🥉 备用 |
| 6. Bootstrap bagging | 低 | +0.5~+1.5 | ✅ | 🥉 备用 |
| 8. Same-date cross-sym mixup | 中 | +0.5~+2 | ✅ | 🥉 备用 |
| 9. Curriculum (stage1+stage2) | 中 | +0.5~+1.5 | ⚠️ 与 #2 重叠 | 备用 |
| 10. External pretraining | 极高 | +0~+5 (高方差) | ⚠️ 复杂 | 🚀 stretch |

### 推荐 1 周 sprint plan

**Day 1**: 提案 1 (V5 split) + 提案 2 (date-decay weight)
- 30 min patch + 30 min train + 30 min eval = 1.5 hour
- 期望 +2 ~ +5 LOSO 单步收益
- 如果失败，已验证两个核心假设

**Day 2-3**: 提案 3 (sym-coupled group aug) — 仔细 ablate
- 半天构建 feature group masks + patch aug 函数
- 半天 train + ablation (price-only / size-only / all-three)
- 期望额外 +1 ~ +3

**Day 4**: 提案 5 (5-LOSO + 1-full ensemble) + 提案 4 (K-fold best_iter)
- 重启 LOSO training pipeline
- 期望额外 +1 ~ +2

**Day 5**: 整合 + iter_015 提交
- DE thresh 重新调
- 平台 transmission 0.70 理论上应改善 → +25 ~ +28 平台分目标

### 不推荐的组合

- 提案 2 + 提案 9: 重叠（都是 time decay）
- 提案 2 + 提案 7 直接相加: 双重 weight overcount; 用乘法或择一
- 提案 6 全替换 5-seed: 损失 hyperparam robustness; 用 3 hp × 5 boot 折衷

### 监控指标

每个 trick 落地后必须看：
1. Local LOSO-equiv（绝对值, 期望 +X）
2. Half-A vs Half-B PnL ratio（提案 1/2 的核心 ablation——如果 half-B 涨幅大于 half-A，证明 trick 有效在打 future-OOD）
3. Per-sym PnL 方差（提案 3/5 应该让 sym-level std 下降）
4. Best_iter 5-seed 方差（提案 1/4 应该让 std 砍半）

---

## Part 4 — 推荐 Top 3（如果只能跑 3 个）

1. **提案 2 (Linear date-decay sample weight)** — 1 行代码，最低风险，最直接打 0.70 系数本质
2. **提案 3 (Sym-coupled group augmentation)** — 改 aug 的 shape 而非 magnitude，打 platform OOD 的 sym 部分
3. **提案 5 (5-LOSO + 1-full ensemble)** — 最颠覆性的认知，把 0.70 系数本质归因于 LOSO 与 platform OOD 的不同性质，并用 ensemble 对冲

期望叠加 LOSO-equiv: **+3 ~ +9**（保守 +3，乐观 +9）；
对应平台预估: **+22 ~ +25**（保守 +22 ≈ 不变，乐观 +25 ≈ 进入 +25 区间）；
**如果叠加 提案 1 (V5 split)**: 透传系数从 0.70 改善到 0.78~0.85，**平台再 +1.5~+3** → 最终目标 +27 ~ +28。

---

## 附 A — 不要做的事（明确 rule out）

| 不推荐 | 原因 |
|--------|------|
| 用 sym ID 做 embedding | 违反硬约束 #3（platform sym 可能新） |
| Per-sym normalization stats | 同上 |
| 跨 batch hidden state / cache | 违反硬约束 #2 |
| 在 96-119 上 pseudo-label 自训（无 label） | T65 已试 -0.61 |
| Magnitude weight 替代 class-balanced (linear/sqrt) | T57 已试 -3 ~ -11 |
| 全替换 5-seed → 30-seed 单 hp | 失去 hyperparam robustness（DE 噪声 ±0.05 已证明 hp 多样性还有用） |
| Aug 范围扩大到 [0.70, 1.30] | T37 已试 [0.75, 1.25]，5-seed ensemble 反而 -0.67 |

---

## 附 B — 关键代码位置

| 文件 | 提案影响 |
|------|---------|
| `experiments/T75_regression_dmid/train_regr.py:92-114` | 提案 1 (V5 split), 提案 4 (K-fold) |
| `experiments/T75_regression_dmid/train_regr.py:241` | 提案 2 (date-decay weight) |
| `experiments/T26_domain_randomization/build_aug.py:39-43` | 提案 3 (group scale) |
| 新建 `experiments/T101a_loso_ensemble/` | 提案 5 |
| 新建 `experiments/T101b_adversarial_val/` | 提案 7 |

---

```
RESULT: task=radical_data_validation_rethink top_recommendations=[V5_late_date_val_with_embargo, linear_date_decay_sample_weight, sym_coupled_group_augmentation, 5LOSO_plus_1full_ensemble] notes="0.70 transmission ratio root cause = uncalibrated time drift (half-A→half-B 26% PnL decay) + LOSO sym OOD ≠ platform fresh-sym OOD; V4 split wastes 16 days unused (80-95) and val too short (4 days, 0 embargo); aug_a per-(sample,feat) breaks cross-feature correlations and doesn't simulate sym OOD shift; 10 specific proposals ranked by ROI; top 3 (date-decay weight + sym-coupled group aug + 5LOSO ensemble) target +3~+9 LOSO with +22~+25 platform projection; with V5 split adding transmission improvement to 0.85+, projected platform +27~+28"
```
