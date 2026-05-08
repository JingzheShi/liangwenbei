# IDEA-CF-FRESH-OPUS — Fresh Eyes Brainstorm

> 视角: 假设我今天才拿到这个赛题,只看 problem statement / 字段定义 / PnL 公式,完全不知道历史实验做了什么。从 first principles 推理。

---

## 第 1 节: 我的核心洞察

经过 first-principles 分析,我认为这个赛题的 unique structure 由 **3 个一阶约束** + **2 个 二阶约束** 决定,而绝大多数表面 baseline 只优化前者:

**一阶约束(直接进 loss)**

1. **PnL 公式不是 accuracy 的代理,是 magnitude × correctness − fee**。  
   Round-trip cost ≈ **2 bp**(双边各 1bp)。α=0.1% 长 horizon 的"激活"label 净边际 ≈ **8 bp**, 而 α=0.05% 短 horizon 仅 **3 bp**。这意味着 **同样的 prediction noise, 短 horizon 被 fee 吃得多得多** —— 长 horizon 在数学上有 **2-3× 更宽的安全边际**。

2. **score = max over 5 horizons** —— 服务器替你做模型选择。  
   推论: **集中火力优化 1-2 个 best-bet horizon** 比 5 个全均匀更优。但保留 5 个是免费保险(如果 best-bet 翻车,其他 horizon 会救你)。

3. **决策规则的 freedom 比 model 自身大**。  
   PnL 的最优 Bayes 决策是 `side = argmax_s E[PnL|s, x]`。给定一个 noisy regression 头 r̂(x), 选阈值 τ 可以让 PnL 翻倍甚至 negative ↔ positive。**τ 调优的 effort/return 比 model 调参高一到两个数量级**。

**二阶约束(限制 model space, 但不进 loss)**

4. **5 stocks 太少, OOD novel sym 是真实风险**。任何 sym-specific signal(如 absolute price level, sym embedding)都是地雷。模型必须学的是 **微观结构动力学**,不是"这只股票常常跌"。

5. **Stateless + shuffled batch + 100-tick window** 把时序问题强制 reduce 到 "100×D snapshot → 5 个数字"。这等价于 **i.i.d. supervised learning**, 只是输入是 sequence。这非常友好于 GBDT(忽略时序结构)和 1D-CNN/Transformer(显式利用)。

**核心直觉**: 这个题不是"高级 deep learning 比拼" —— 它是 **"在严格的 OOD/stateless 约束下,用 PnL-aligned objective + 阈值调优,从 100-tick LOB 窗口榨出 bp 级别 alpha"**。简单方法 + 正确 loss + 正确 threshold,大概率打赢复杂方法 + 错的 loss。

**二阶心智模型**: 我会把整个 pipeline 看成 3 个独立优化阶段:
- (a) **representation**: 怎么把 100×D 压成有用的 ~500 维特征(或让 NN 学)
- (b) **prediction**: 怎么 estimate r̂ = E[Δmp/mp_t | x]
- (c) **decision**: 怎么从 r̂ 选 side ∈ {-1, 0, +1}

每个阶段错一步, PnL 都崩。其中 (c) 是大多数人低估的 — **它本质决定 50% 的 final PnL**, 因为 PnL 在阈值处不可微,小幅 τ 移动就能改写排行榜。

---

## 第 2 节: 14 个设计选择的回答

### Q1. Target 定义: 3-class label / Δmidprice 回归 / sign(Δ)·max(0,|Δ|−fee) / multi-task?

**答: 主目标用 Δmp/mp_t 连续回归; 辅助一个 3-class CE 头做 multi-task regularization。**

First-principles 推理: PnL 公式对预测的偏导是 `∂PnL/∂side ≈ Δmp − fee·sign(side)` —— **PnL 在 magnitude 上是线性的**。3-class label 把 magnitude 信息(label_5=2 可能意味 +5bp 也可能 +500bp)全部抹掉,而这正是 PnL 关心的部分。

但纯回归会 underfit "label 噪音区"(|Δ|≈α 边界),因为 MSE 在那里梯度小。所以多挂一个 CE 头作为 anchor: 它强迫模型分辨 "确定方向 vs 不确定", 并且在 long-tail 区域提供形状先验。Multi-task ≈ **regression 给 magnitude, CE 给 confidence**。

不选 `sign(Δ)·max(0, |Δ|−fee)` 作为直接 target: 它在 |Δ|<fee 的样本上 target=0 但样本不是噪音,会 underweight 这些点的信号。fee 应该进 **决策**,不该进 target。

### Q2. Loss function

**答: weighted MSE on Δmp, weight = `|Δmp|^0.5`, + 0.1× CE auxiliary。Stage 2 再用 PnL-surrogate fine-tune。**

推理: 

- **PnL 实际是被大移动主导的**: 如果 |Δmp| 服从重尾分布, 95% 的总 PnL 来自 top 5% 大移动样本。普通 MSE 在 |Δ|≈0 的海量噪音点上花了 95% 梯度。**Weight = |Δmp|^p (p∈[0.3, 0.7])** 让 loss 真正去关心交易上重要的样本。p=0.5 是个折中(p=1 太激进会 unstable)。
- **不直接用 PnL 作 loss(stage 1)**: PnL = side·Δmp − fee·|side| 在 side 上不光滑(side 是 sign 函数),梯度需要 surrogate(如 sigmoid),而 surrogate 加阈值后训练会病态(梯度沿阈值集中)。先用稳健 loss 训出可用 r̂, 再小步 PnL fine-tune。
- **CE auxiliary** 起 calibration & anchor 作用,权重小(0.1),不主导。

### Q3. 模型家族

**答: GBDT (LightGBM) 主力 + 1D-CNN (small) 做 stacking 互补。NN 单跑作为 high-risk-high-reward 实验项。**

推理: 1.77M 样本 × ~500 特征,这是 GBDT 的 sweet spot:
- 异质 scale (price ~1e-3, intst ~1e-4, indicator 0/1)无需 normalization
- 缺失/极值鲁棒
- 训练快(GPU LightGBM 12s/epoch),允许海量调参
- 特征重要性可解释,便于 debug pipeline

但 GBDT 完全忽略 100-tick 的时序结构 —— 它把每个滚动统计当成独立特征。**1D-CNN 在 100×D 上能学到 GBDT 错过的 lag-cross-correlation** (如 "8 ticks ago 的 lb_intst 突变 → 现在 ask 退缩 → 预兆下跌")。两者高互补。

不选纯 sequence model (Transformer): 在 5 stocks × OOD 约束下,大模型严重过拟合 risk,且 stateless 100-tick 窗口太短发挥不出 attention 优势。

### Q4. CV scheme

**答: 主 CV = Leave-One-Sym-Out (LOSO) × Last-N-Days-Holdout 联合; 用 LOSO 选超参, 用时间 holdout 选阈值/early stop。**

推理: OOD novel sym 是题面写明的 first-class 风险。Random split 会让模型记住 sym 特征(尤其当 sym 不当 feature 但 sym-specific patterns 通过 normalization 渗入)。LOSO 是**唯一直接测 OOD 泛化**的方案。

但 LOSO (4 folds) 信号噪声大且只能选粗超参。所以加一层时间 holdout(如 last 20 days)用于:
- early stopping (loss 早停)
- 阈值 τ 选取 (PnL 早停)
- 最终 submission horizon 选取

不选纯随机 split: 不测 OOD,**严重高估自己**。  
不选纯 leave-one-day-out: 5 stocks × 1 day 太小,fold 内 PnL 噪声 > 信号。

### Q5. Feature pipeline

**答: 80% 手工特征 + 20% 原始 100-tick raw (给 1D-CNN 用)。 主 GBDT 用 ~500 维手工特征。**

推理: 数据已经预处理得很重(KZ-2015 衍生,intst/ind/acc), **手工 + GBDT 是 Pareto 最优**。具体:

- 当前 tick 154 维 raw + 衍生
- 100-tick 滚动统计: 每个 base feature 取 (mean, std, slope-via-OLS, last-mean, max, min) → 多 ~ 6 × 50 ≈ 300 features
- **OFI/QI 类自定义**: order flow imbalance (mb_intst - ma_intst) 在多窗 (5/20/100) 求平均
- **微结构 ratio**: spread1/midprice, bsize1/asize1 各档
- **time-of-day** features: 见 Q5 末尾

总计 ≤ 500 特征。raw 100×D 给 CNN 做 stacker 用。

**Time-of-day 关键提示**: `time` 字段被保留(3s 间隔)。提取分钟号 / 距开盘多少分钟 / AM-PM phase / "0-1 normalized session position" → 这些在 HFT 里是 highest information gain per effort 的特征(开盘/收盘 dynamics 完全不同)。**不挖这个就是浪费 free signal**。

### Q6. Normalization

**答: per-feature global log1p (volume/intst) + raw (price, indicator); 加 per-window z-score 兜底覆盖 (cross-sym).**

推理: 严格的 sym-agnostic 约束:
- 价/return 字段(close, bid, ask, midprice, spread): 已经是 returns ~0,**不再 normalize**
- size/volume/intst (`*size*`, `volume_delta`, `*_intst`): 跨 sym 量级差异大, 用 **log1p** 收敛尾部, 再加 **per-window z-score** 让模型看相对值而非绝对值
- amount_delta (唯一带单位的 e3-e6): **必须 log1p**
- indicator (`*_ind`): 二值,**绝不 z-score**

per-window z-score 是 sym-agnostic 的关键: 即使遇到 sym=99,其窗口内的 mean/std 自洽,模型不会 IndexError。**不能用 per-sym 全局统计**。

GBDT 严格不需要 z-score,但即便如此,window-z-score 在树模型上也提供 "相对水平" 信号(比"今天 size 突然 +3σ"比 "size=0.0042" 更直接)。

### Q7. Augmentation

**用**:
- **Sym ID shuffling**: 训练 batch 内不让 model 看到 sym ID 关联(虽然 sym 不当 feature, 但同 sym 的 sample 可能因数据顺序聚集 → mini-batch 内 cross-sym mixing)
- **Random 100-tick crop**: 训练时随机用 80-100 ticks (推理时永远用 100); 让 model 不依赖 fixed length
- **Window-mask augmentation (CNN only)**: 随机 mask 一档 LOB level (如 mask bid5 整列), 强迫 model 不依赖某一档

**不用**:
- **Gaussian noise injection**: LOB 是结构化数据 (bid <= ask, 价格 ladder), Gaussian 破坏结构
- **MixUp/CutMix**: target 是回归值,MixUp 会让 magnitude 变 noisy
- **Time-reversal**: 物理上不对称(订单簿动力学有方向),会注入错误先验

理由: augmentation 应当 preserve task-relevant invariances。这个题的 invariance 是 sym 和 absolute time, 不是 noise / temporal direction。

### Q8. Training (ensemble & seeds)

**答: 5 LOSO folds × 3 seeds = 15 个 GBDT models, mean predict;再加 1-2 个 NN model 做 stacking。**

推理:
- LOSO ensemble: 4 fold(留 1 sym 测试), 重复 5 次 (每 sym 各 hold 一次)。这同时做了 cross-validation 和 ensembling。最终 inference 用全部 5×3=15 model 平均(每个看过 4/5 sym 数据,合起来覆盖全部)。
- 3 seeds 够: GBDT 在 1M+ 样本上方差不大,3 seeds 减阈值附近的 prediction noise。多 seed 边际收益快速递减。
- 模型大小约束 (2GB): LightGBM 单 model ~50MB → 15 model = 750MB ✓ 还够装一个 NN。

不做 5-seed bagging within fold: edge case 的 prediction 多样性主要来自 **不同 sym 训练**, 不来自不同 seed (后者只动 split 顺序)。

### Q9. Threshold / decision

**答: 直接在 LOSO holdout 上 grid search (τ_up, τ_down) 最大化 PnL, 每个 horizon 独立选。 用 Joint search 而非 single τ (因为 up/down 标签分布不对称, 长 horizon 尤甚)。**

推理: 决策阶段是 **highest leverage** 的步骤,但绝大多数 baseline 用 argmax(三类 CE) 或固定 τ=0.5 (二类) —— 这两种都把 fee 完全忽略。

正确做法:
1. Model 输出 r̂ (回归值或 P(up)-P(down))
2. 在 LOSO holdout 上做 2D grid: τ_up ∈ [0, 5e-3], τ_down ∈ [0, 5e-3], 步长 1e-4
3. 选 max PnL 的 (τ_up*, τ_down*)
4. **报告活动率 (n_active / n_total)**: 经验上 sweet spot 在 30-50% 活动率, 太低(<20%)说明信号弱, 太高(>70%)说明 τ 调小了
5. **每个 horizon 独立调**: 因为 5 个 horizon 的 P(up)/P(down) 分布尺度不同

可以加 **adversarial validation**: 把 holdout 50/50 切两半,τ 在前半选,后半验证,防止 τ 过拟合 holdout。

### Q10. Multi-horizon 处理

**答: 5 个独立单 horizon 模型 (主); 但 share 同一份 feature cache。 提交时 5 个 horizon 全部填,服务器自动选 max。**

推理: max-over-horizons 是 score 的关键 —— 这意味着 **不需要把 5 个 horizon 都做好,只需要 1 个做爆**。两条策略:

(a) 5 model 独立训练, 让每个 horizon 各自最优, 提交全部
(b) 1 model 多 head 共享 trunk, share 信息

两者权衡:
- (a) 每个模型完全 specialize, 但 trunk 重复学 ⇒ overhead 但 ceiling 高
- (b) trunk 共享,horizon 间 transfer learning, 但 capacity 被分摊 ⇒ ceiling 略低,但训练快/省 GB

对 GBDT: trunk-share 不存在(每棵树是 horizon-specific), 所以选 (a)。  
对 NN: (b) 是 multi-task,但 NN 不是主力,所以也选 (a)。

**关键策略**: **算力分配偏向 best-bet horizon**。基于本节顶部的数学(α=0.1% 净边际 8bp >> α=0.05% 净边际 3bp), **赌 h=40 或 h=60 最强**。这两个 horizon 用主 ensemble (15 models), 其他 3 个 horizon 用单 seed 单 fold (省时间) 当保底。

### Q11. OOD novel sym handling

**答: 主防线 = LOSO CV + sym-agnostic 特征; 辅助 = sym-shuffle augmentation; 高风险选项 = adversarial domain training (列入 high-risk)。**

推理:
- **必须做的**: LOSO CV(测 OOD 泛化),不用 sym 当 feature, normalization 严格 sym-agnostic
- **应该做的**: 训练时把 sym ID 当 dropout 处理 — 随机 25% 概率把 sample 视为"未知 sym",强迫模型仅用窗口信号
- **可以做的**: cross-sym percentile rank features (在 batch 内 rank), 但这破坏 stateless,放弃
- **可能值得**: adversarial — 加一个 sym classifier head, 主 trunk 反向 gradient 优化让它分不出 sym。这是 domain generalization 经典手法,可能 +大 也可能 -大,放 high-risk

不做的: 训练 sym-specific model 集合(违反约束)。

### Q12. Probability calibration

**答: 需要 — 但通过 τ 调优隐式做, 不需要单独 isotonic/Platt。**

推理: 一般 calibration 是为了让 P(class) 真实反映概率。但我们的下游决策是 PnL 最大化,不是 probability matching。**直接调 τ 已经在 holdout 上 calibrate 了 score → action 的映射**。

如果一定要做显式 calibration:
- LOSO holdout 上拟合 isotonic 把 r̂ → calibrated r̂
- 然后再调 τ
- 边际收益小(<1% PnL),不优先

不做的情况: 如果遇到 r̂ 在不同 fold 间 scale 漂移严重, 需要 per-fold z-score 后再 τ 选取。先观察再决定。

### Q13. Early stopping

**答: 主 metric = validation PnL (在 hold-out 上, 用 grid 选过 τ 后的). 备用 = horizon-specific F1_macro。 不要用 loss / accuracy。**

推理: 
- **loss 早停**: 模型可以 loss 一直降但 PnL 因为 τ-aware 边界开始恶化。loss 早停是 PnL 错的。
- **accuracy 早停**: 同样问题, accuracy 优化 majority class (flat) 而 PnL 关心 minority class precision。
- **F1_macro 早停**: 比 accuracy 好,因为 balance up/down,但仍忽略 magnitude。备用方案。
- **PnL 早停 (推荐)**: 在每 N iteration 后, 用快速 τ-grid 算 holdout PnL, monitor 5 epoch 不涨就停。

实现注意: τ-grid 算 PnL 在 LightGBM custom callback 里需要每 50 trees 跑一次, 不要每棵都跑(慢)。

### Q14. Submission strategy

**答: 5 个 horizon 全填(免费保险),把最大 effort 给 h=40, 其次 h=60, 然后 h=20, 最后 h=5/10。 12h 提交一次的限制要求每次都是 "下次最大改进" 的实验。**

推理:
- **不能只填一个 horizon**: 如果服务器期望全部, 没填会报错 / 0 分。即便不报错, 全填的成本几乎为 0(GBDT inference 快, model size 小)。
- **算力优先级**: 见 Q10 — h=40 (净边际 8bp + 39% 活动率) 是 best bet。
- **12h 提交节流**: 这是 brutal constraint。每次提交前必须有 LOSO holdout 上的 大幅 (>3%) PnL 改善信号才提交,否则浪费窗口。

冷启动建议: 首次提交一个 sanity baseline (linear regression on order book imbalance), 确认 pipeline 正确,然后才上 GBDT。

---

## 第 3 节: Top 13 实验方法 (按优先级)

> 强度标注: 弱 = 小幅改善, 中 = 显著改善 (~3-10% PnL), 强 = 关键 (~10-30% PnL)

### Tier S — 必做基础

| # | 方法 | 强度 | rationale |
|---|---|---|---|
| 1 | **GBDT regression on Δmp/mp_t**, weighted MSE (w=\|Δ\|^0.5), per-horizon 独立 model | 强 | 最 PnL-aligned 的 baseline, 单这一项就建立可观 PnL ground truth |
| 2 | **2D τ-grid search for decision thresholds** on LOSO holdout | 强 | 同一个 model 加正确 τ 调优 vs argmax,实测往往差 30-50% PnL |
| 3 | **手工特征 ~500 维**: 滚动统计 + OFI + microstructure ratios + **time-of-day**(分钟号 / session phase) | 强 | time-of-day 是免费 alpha, 大多数人会忘 |

### Tier A — 高 ROI 提升

| # | 方法 | 强度 | rationale |
|---|---|---|---|
| 4 | **LOSO 5-fold × 3-seed GBDT ensemble** mean predict | 中 | OOD 泛化 + 阈值方差 reduce, 一次设置长期收益 |
| 5 | **PnL-aware loss fine-tune (Stage 2)**: 用 Stage 1 GBDT 出来的 r̂ 作为 init, 用 \|Δmp\| weighted MSE 上加一个 PnL surrogate 的 fine-tune | 中 | 直接对齐 objective, 比纯 weighted MSE 再 +几% |
| 6 | **3-class CE auxiliary head (multi-task)**: 主 regression + 0.1× CE | 中 | 边界附近的样本 anchor, regularization 效果 |
| 7 | **per-horizon, per-sym-fold τ 选取 + holdout 二次验证**: 防 τ 过拟合 | 中 | 边际但稳定, 避免 lucky threshold |

### Tier B — 锦上添花

| # | 方法 | 强度 | rationale |
|---|---|---|---|
| 8 | **1D-CNN on 100×D raw window** as stacking 第二模型, blend weight 在 LOSO holdout 选 | 中 | 捕捉 GBDT 错过的 lag-cross-correlation, 提供 model-class diversity |
| 9 | **Sym-dropout augmentation** during training: 25% 把 sample 视为 "unknown sym" (其实没有 sym 字段, 但通过 z-score scale jitter 实现) | 弱-中 | OOD insurance,边际效果但 cheap |
| 10 | **Window length augmentation**: 训练时随机用 80-100 ticks | 弱 | 鲁棒性 + 轻微 ensemble 效果 |
| 11 | **Quantile regression**: GBDT 同时预测 Q25, Q50, Q75 of Δmp; 决策时 require Q25 > fee (up) 或 Q75 < -fee (down) — 高置信门槛 | 中 | PnL 关心 magnitude, quantile 直接给置信区间, 高置信 τ 更稳 |

### Tier C — 长尾尝试

| # | 方法 | 强度 | rationale |
|---|---|---|---|
| 12 | **Order flow regime feature**: 100-tick 内 abs return 的 std → volatility regime, 让 model 学 regime-specific signal | 弱 | volatility 是 HFT 公认重要 conditioning |
| 13 | **Cross-horizon stacking**: 用 horizon-5 model 的 r̂ 作为 horizon-60 model 的额外特征 (and vice versa) | 弱-中 | 不同 horizon 抓到不同尺度信号, 简单 stacking |

---

## 第 4 节: 5 个 high-risk-high-reward 实验

这些可能 0 分也可能 +20 PnL,放在主 pipeline 之外的 side bet:

### HRHR-1: End-to-end PnL maximization with differentiable decision
**思路**: NN 直接输出 logits over {-1, 0, +1}, 用 Gumbel-Softmax / straight-through estimator 让 side ∈ {-1,0,+1} 离散选择 differentiable。Loss = -E[PnL]。

**风险**: 训练极难收敛, 容易 collapse 到全 0 (no trade) 局部最优。需要精心 schedule (warm start from CE, slowly anneal Gumbel temperature)。

**回报**: 如果跑通,能 unify model + decision into single optimization, 理论最优。在小心调的情况下可以打掉 "regression + post-hoc τ" 的 sub-optimality (估计 +10-15% PnL)。

### HRHR-2: Adversarial domain generalization for sym-OOD
**思路**: 主 trunk 共享, 分两 head: (a) PnL prediction head, (b) sym classifier head。Reverse gradient layer 让 trunk 学 sym-agnostic representation。 训练时把 sym 作为 label (虽然 sym 不进 forward)。

**风险**: GAN-style 训练 unstable, weight 调不好会让 main loss 退化。  
**回报**: 如果 OOD novel sym 是真实 risk (题面写明), 这个直接 attack 风险源, 可能 +20 PnL on novel sym subset。

### HRHR-3: Self-supervised pretraining on next-tick prediction
**思路**: 用全部 2.4M ticks 预训一个小 transformer, target = next-tick 各特征 (BERT-style masked). 然后冻结 trunk, fine-tune horizon heads。

**风险**: pretraining 算力大(几小时); transformer 在 100-tick 短窗效果不一定比 CNN 好; OOD sym 上 pretrain 的 representation 不一定 transfer。  
**回报**: 如果 pretrain 抓到 universal microstructure dynamics, 单 model 可能匹敌 ensemble。+15-25 PnL 上限。

### HRHR-4: Bayesian queue dynamics hardcoded model (Cont 2014-style)
**思路**: 不学黑盒, 直接用 LOB queue dynamics 公式: arrival rates (lb/la/mb/ma/cb/ca) 推导 next-tick 价格变动概率分布, 解析或半解析地算 expected P(up)/P(down)。 0 学习参数 (或极少 calibrated 参数)。

**风险**: 模型 mis-specification, 所有的 derived 特征已经是黑盒, 重新建分析模型可能 conflict。  
**回报**: 极强 OOD 泛化 (机制性模型 universal), 可能在 novel sym 上稳吃 GBDT。+10 PnL 同时降低 OOD 风险。

### HRHR-5: Trade size / position aware (treat fee as per-trade not per-tick)
**思路**: 当前 fee 公式按 tick 算, 实际交易是 "建仓 → h tick 后平仓" 的 round trip。考虑两笔连续相反 prediction 时, 中间不需要平仓再开仓 (fee saved)。在 prediction 后处理: smooth 连续相同方向预测, 减少冗余进出场。

**风险**: 评测公式本身可能不允许 smoothing(每 tick 独立 fee), 这会偏离官方公式。需要先在公式上验证。  
**回报**: 如果合规,fee saving 可能 cumulative ~15-30% (每个连续 5-10 tick 的相同方向 trade 节省 ~80% fee)。

---

## Self-review (against "fresh" criterion)

> 重读 Top 5: GBDT regression / τ-grid / 手工特征 (含 time-of-day) / LOSO+seed ensemble / Stage 2 PnL fine-tune

我相信前 4 个是任何 reasonable team 都会做的 baseline (确实可能项目 80 个实验里有重复)。**但 #2 "正确的 τ 调优" 和 #3 "time-of-day 特征" 经常被忽略 / 做错** —— 它们是 fresh-eyes 视角真正的 "不显然但高价值" 项。

#5 (Stage 2 PnL fine-tune from regression init) 可能 fresh: 大多数人要么纯 weighted MSE 一把, 要么纯 PnL surrogate 直接训 (会失败)。**两阶段 init 是可能被忽略的 trick**。

Tier B-C (1D-CNN stacking, Quantile regression, cross-horizon stacking) 是中等 fresh, 不是显然路线。

HRHR 5 项中, **HRHR-4 (Bayesian queue dynamics)** 最反直觉 — 大多数 ML team 不会从 econometrics 文献切入, 而它在 OOD constraint 下可能是 hidden alpha。

**最高 expected information gain 的 single trick**: 我赌 **"在 LOSO holdout 上做 2D (τ_up, τ_down) grid search 取代 argmax / fixed τ"** —— 它不需要新模型/新特征/新代码, 只是一个 post-processing 步骤, 但在 fee=2bp 的 sharp boundary 下能改写排行榜。
