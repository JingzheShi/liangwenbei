# IDEA-CONTRARIAN — 颠覆性方法 Brainstorm Report

> 生成日期：2026-05-07
> 评测截止：2026-05-11（4 天）
> 当前 SOTA：iter_006 LightGBM 5-seed × 226-d Scheme C，LOSO h_60 sum_pnl **+13.61**
> 研究员立场：刺儿头，独立思考，绝不恭维

---

## 0. 致命前提：我们踩了什么坑

在列 idea 之前，先诚实列出当前范式的结构性问题——这决定了哪些方向有可能翻盘：

| 问题 | 证据 | 含义 |
|---|---|---|
| **分类目标 ≠ PnL 目标** | 4D thresh DE 仍能在 prob 输出不变时让 LOSO 变化 ±0.5 | 分类 loss 训练的是"预测方向正确率"，但评分是 PnL（precision-weighted） |
| **NN 无法做到 sym-agnostic** | T1/T19/T32 全部崩，val gap 巨大 | NN 会把 226-d feature 的相关性模式当 sym 签名 → OOD sym 测试集崩 |
| **交叉 sym 特征在推理端不可用** | T48 正式判定 ABORTED | C2 + date 置 0 + no-state 三重约束封死了 Volkova/hyd 风格的 cross-sym pooling |
| **Group DRO 改善单 seed 但无法超越 5-seed 集成** | T45 LOSO +8.66 vs baseline +13.61 | DRO 提升的是 per-sym 公平性，但 5-seed 平均本身已经在做类似事 |
| **特征已饱和** | T39 显示 226-d 加 r34 特征几乎零收益 | Scheme C 的 feature importance 已经 long-tail；继续加 LOB features 是边际收益 |

**结论**：下一个 +2 不来自"更多特征"或"更强正则化"，来自**目标函数、模型结构、或推理逻辑的范式转换**。

---

## 1. Idea A：KNN-Target Retrieval Feature（非参数"学习记忆"）

### 原理
nyanp（Optiver 2021 1st）核心招数的干净版本。训练时把所有 train 样本的 226-d feature 向量和 label 存起来；推理时每个 test 窗口在训练集里找最近 K 个邻居，把邻居的 target (0/1/2) 做 majority vote 或 softmax-weighted average，作为 2 个额外 feature 输入 GBDT 最后一层（或直接 ensemble）。

**为什么这比 GBDT 本身更有信息？**
GBDT 的 leaf node 等价于对 feature space 的 axis-aligned 分区，所有落在同一叶子的样本共享一个 prediction。KNN 用 L2 距离——捕捉的是 non-axis-aligned 邻域（斜着切），GBDT 学不到的几何关系。两者 ensemble 有多样性。

**约束兼容性**：
- ✅ C1（date 置 0）：KNN 特征纯粹来自 feature 向量，不用 date
- ✅ C2（顺序打乱）：每次 predict 调用独立查询 FAISS，无跨调用 state
- ✅ C3（sym-agnostic）：只用 226-d feature，不用 sym ID

### 实现 Plan
```python
import faiss
# 训练阶段：
index = faiss.IndexFlatL2(226)
index.add(X_train.astype('float32'))  # 1.47M × 226
# 推理阶段（Predictor.__init__ 里加载）：
D, I = index.search(X_test.astype('float32'), k=5)
knn_label_mean = y_train[I].mean(axis=1)   # float feature
knn_label_std  = y_train[I].std(axis=1)    # uncertainty estimate
knn_min_dist   = D[:, 0]                   # nearest neighbor distance
# → 3 个新 feature 加进 X_test
```

**数据需求**：仅需训练集（已有）+ FAISS  
**风险**：
- FAISS index 1.47M × 226 × 4 bytes ≈ 1.3GB，接近 2GB 模型限制。需要 PQ（product quantization）压缩或只用 top-5% 高重要度的 k 个子空间。
- 推理端查询延迟：batch=1 时 1.47M 中查 k=5，FAISS-L2 约 2-5ms，总体 OK。

**Upside**：+1~+3 LOSO sum（参考 nyanp 在 Optiver 2021 的效果）  
**实现时间**：1~1.5 天（写 + debug + 验证 inference latency）  
**优先级**：⭐⭐⭐⭐ 可行性高，risk 低，且与现有 GBDT 完全正交

---

## 2. Idea B：Multi-Horizon Meta-Stacker（5 头集成器）

### 原理
当前做法：5 个 horizon 独立训练 → 选最好的那个提交。这丢弃了大量信息：5 个 horizon 的预测之间有相关性，且"5 head 一致性"是信心的强信号。

**新范式**：训 5 个 horizon 的 LightGBM OOF 预测后，喂给一个 **meta-stacker**（L2 层）：

```
[P_h5, P_h10, P_h20, P_h40, P_h60]  ← 5 × 3 = 15 个 prob 值
             ↓
       meta-GBDT / logistic
             ↓
      final_pred ∈ {0, 1, 2}  ← 面向 PnL 优化
```

关键 insight：如果 h_5 预测"涨"但 h_60 预测"跌"，这是矛盾信号，最优策略可能是 abstain（预测"平"）。但如果 h_5/h_10/h_20/h_40 都预测"涨"，信号一致，可以激进下注。

这不只是 soft ensemble，而是**学习"何时信哪个 horizon"的自适应决策器**。

**约束兼容性**：✅ 完全无状态，纯 tabular 输入

### 实现 Plan
1. 用 iter_006 的 OOF 预测（h_60 已有）+ 其他 horizon 的 OOF 预测（需要训练 h_5/h_10/h_20/h_40 或复用已有）
2. 把 15-d prob vector 作为输入，训一个小 LightGBM（depth=3，这是 meta-layer）
3. meta-layer 的 objective = 直接优化 PnL（自定义 obj）
4. threshold 仍可用 DE 搜索

**风险**：
- 需要 5 个 horizon 的 OOF，目前可能只有 h_60 完整。补训其他 horizon 需要额外 GPU 时间。
- Meta-layer 的 overfit 风险（只有 5-fold × ~88k 样本，meta-input 只有 15-d，应该 OK）

**Upside**：+1.5~+5 LOSO sum（多 horizon 一致性是强信号，理论上能显著提升 precision）  
**实现时间**：1.5~2 天（训缺失 horizon OOF + meta-layer training + eval）  
**优先级**：⭐⭐⭐⭐ 这是目前最被低估的方向，且"集成 5 horizon"的想法完全在约束内

---

## 3. Idea C：直接 PnL 策略优化（单步 Bandit）

### 原理
**当前管道**：LightGBM(CE loss) → probabilities → threshold → trade → PnL
**问题**：CE loss 的梯度不知道"错了以后会损失多少 PnL"。预测"涨"结果"平"的代价 ≠ 预测"涨"结果"跌"的代价，但 CE 对它们一视同仁。

**新框架**：把每个训练样本看作一个 one-shot bandit：
- Agent 输出 action $a \in \{0, 1, 2\}$（跌/平/涨）的 probability $\pi(a|x)$
- Reward $r(a, y_{true}) = \text{pnl\_single}(a, midprice_{t+n} - midprice_t)$
- REINFORCE 梯度：$\nabla J = \mathbb{E}[\nabla \log \pi(a|x) \cdot (r - b)]$，其中 $b$ 是 baseline

对 LightGBM 的适配：
- 用自定义 objective（类似 R10 PnL loss），但按 REINFORCE 风格加权：对于预测"涨"但实际"跌"的样本，loss 权重 = |PnL loss|，即越亏本越强迫梯度更新
- 关键与 CE 的差异：**action 的"错误成本"不对称**

### 为什么 R10 已试过但我仍然推荐再试

R10 试的是把 PnL 直接作为分类 loss 的目标函数，但问题是：分类 loss 优化的是"预测出 up/down/flat 的准确性"，而 PnL 优化目标是"只预测高置信样本（precision-maximizing trade selection）"。这两个目标的梯度方向不同。

**真正需要的是**：一个 loss 函数，让模型在 prob 低的样本上学会"预测平"，在 prob 高的样本上才预测方向。这等价于训练一个 soft threshold。

实现：
```python
# 自定义 LightGBM objective
def pnl_bandit_obj(preds, dataset):
    y = dataset.get_label()  # 0/1/2
    probs = softmax(preds)   # B × 3
    # 期望 PnL（对所有 action 的期望奖励，不只是 argmax）
    r = compute_expected_pnl_reward(probs, y)  # B-dim
    # gradient: ∇ loss = -(r - baseline) * ∇ log π(argmax|x)
    grad, hess = reinforce_grad(probs, y, r, baseline=r.mean())
    return grad, hess
```

**风险**：
- REINFORCE 梯度方差极大（需要 baseline 减方差）
- LightGBM 自定义 obj 的 hessian 近似可能不稳定
- 可能比 CE + 4D thresh DE 差（因为 thresh DE 本身就是事后 PnL 优化）

**Upside**：若 work，理论上能学到"只在高置信时出手"的 intrinsic threshold，减少 thresh tuning 的 overfit  
**实现时间**：1~2 天（主要 debug 梯度计算）  
**优先级**：⭐⭐⭐ 中等。R10 已探索过 PnL loss，需要确认 R10 的具体负面结果在哪，避免重复踩坑

---

## 4. Idea D：LOB Pattern Mining → Regime-Adaptive Threshold

### 原理
当前 4D thresh $\{T_{up}, T_{dn}, d_{up}, d_{dn}\}$ 是全局固定的。但不同市场状态（开盘后高波动、午间低波动、单边趋势、震荡）的最优 threshold 不同：趋势行情 threshold 应低（更多交易），震荡行情 threshold 应高（减少被 noise 骗）。

**Regime 特征**（从当前 100-tick 窗口内计算，stateless）：
```python
features_for_regime = [
    X['spread_mean'],           # book 松紧
    X['imbalance_std'],         # 短期不平衡波动率
    X['midprice'].diff().abs().rolling(20).mean(),  # 最近 20 tick 平均波动
    X['volume_delta'].rolling(20).sum(),             # 成交量密度
    X['ask1'].diff().rolling(10).apply(sign_changes), # 价格反转次数
]
regime_feature = concat(features_for_regime)  # → 5-d vector
```

然后训一个 **Regime Classifier**，把市场分为 K 个状态（K=3~5），每个状态使用不同的 threshold：
```python
k = regime_clf.predict(regime_feature)
threshold = thresholds[k]
trade_decision = apply_threshold(lgb_prob, threshold)
```

**约束兼容性**：✅ Regime 特征从当前窗口计算，无跨调用 state

**风险**：
- Regime classifier 可能 overfit（5 fold，需要谨慎 CV）
- 实际上 regime 里的关键特征很多已经在 226-d 里了，meta threshold 等价于 per-sample threshold... 这其实是 conformal prediction 的一种形式
- **更直接版本**：feature-conditioned threshold — 把 LGB prob 和 5 个 regime feature 一起喂给一个 XGBoost 二分类器（"是否值得出手"），直接学 abstain 决策

**Upside**：+0.5~+2 LOSO sum  
**实现时间**：0.5~1 天  
**优先级**：⭐⭐⭐ 中高。这本质上是 Venn-Abers 的更 domain-aware 版本

---

## 5. Idea E：Siamese 买卖不对称特征工程

### 原理
来自 arXiv 2505.22678（Siamese LOB，Yang et al. 2025）：LOB 天然有 bid-ask 对称性——预测"上涨"对应"买盘强过卖盘"，与预测"下跌"对应"卖盘强过买盘"是镜像结构。

**当前 226-d 特征**：混合了 bid/ask 两侧信息，让模型自己发现不对称性。但这要花大量模型容量。

**新想法（纯 feature engineering，无需 NN）**：
1. 构造 **买卖镜像特征**：对所有 bid-side features，对应的 ask-side features，显式计算差值和比值
2. 买卖对称性违背指数：如果 bid_imbalance 和 ask_imbalance 不对称 → 方向信号
3. 买卖 Granger causality 特征：过去 20 tick 里，bid 变化是否在 Granger-cause ask 变化

具体来说：
```python
# 买卖力量差（直接方向信号）
features['buy_pressure']   = (bid_total_size - ask_total_size) / (bid_total_size + ask_total_size)
features['bid_ask_OFI_diff'] = OFI_bid - OFI_ask    # 已有，但明确命名
# 高阶：滚动 Granger causality
for w in [10, 20, 30]:
    features[f'bid_lead_ask_{w}'] = rolling_xcorr(bid_mid, ask_mid.shift(1), window=w)
# 买卖不对称性（level 间的不对称）
for l in range(1, 6):
    features[f'lob_asym_{l}'] = (bsize_{l} * (ask_{l} - mid)) - (asize_{l} * (mid - bid_{l}))
```

**约束兼容性**：✅ 纯函数变换  
**风险**：部分特征已经在 226-d 里（Scheme C 有 imbalance，bsize_rate 等），收益可能边际  
**Upside**：+0.3~+1.5 LOSO sum  
**实现时间**：0.5~1 天  
**优先级**：⭐⭐⭐ 快，低风险，与其他 idea 可叠加

---

## 6. Idea F：生成模型扩充训练数据（VAE / Normalizing Flow）

### 原理
当前数据扩增：uniform scale [0.8, 1.2]，仅改变幅度。这是极其粗糙的 augmentation，不保留 LOB 的微观结构约束（价格单调性、spread > 0 等）。

**新想法**：用 Conditional VAE 学习 LOB 的条件分布 $P(\text{window} | \text{label}, \text{regime})$，然后对每个训练样本生成 K 个 augmented 版本。

具体优势：
1. **对少数类（极涨/极跌）生成更多样本**：当前 60-80% "平"，模型对非平类泛化差
2. **Counterfactual augmentation**：对一个"平"样本生成"如果 OFI 更高会怎样" → 一个"涨"的 counterfactual，让模型学到因果关系
3. **Style transfer augmentation**：把 sym_0 的 LOB 风格迁移到 sym_1，显式训练 sym-invariant 特征

### 诚实评估：为什么这在初赛前不可能 work

1. **实现时间**：VAE 需要 1. 设计 architecture 2. 找合适 loss 3. 验证生成质量 4. 验证 downstream 效果。最少 3-5 天。
2. **LOB-Bench（2502.09172）的结论**：现有所有生成模型（LOBS5/RWKV/cGAN）在 long horizon 上都有"model derailment"——生成的 sequence 在几十步后就偏离 real 分布。5-minute LOB sequence（100 ticks × 226 dim）正好在这个危险区。
3. **DiffLOB (2602.03776)**：专门做 counterfactual LOB generation，JPMorgan 做的，作为 stress-test 工具而非 data aug。原因：生成的 counterfactual 难以保持 microstructure 约束（spread ≥ 0, best bid < best ask 等）。
4. **Class-conditional generation**：需要 class-conditional 架构，且生成的"涨"样本必须真实反映有利于上涨的市场状态，否则生成的是高方差噪声。

**结论**：生成模型 aug 是**最有潜力的长期方向**（理论上能让训练集 10x 扩充），但在初赛前是白日梦。放进 finals 清单。

**Upside（如果做好）**：+3~+8 LOSO（speculative）  
**实现时间**：5-10 天起步  
**优先级（初赛）**：❌ 不做；**优先级（决赛）**：⭐⭐⭐⭐

---

## 7. Idea G：Causal/Counterfactual 因果特征（Conservative Actor）

### 原理
HFT 的核心悖论：我们训练的模型假设"我的下单不影响市场"，但如果模型预测得很准（HFT 本质上 IS the market），那"预测精准 → 下单 → 市场移动 → 预测不再精准"。

**保守 Actor 思路**：
1. 识别高 impact 时段（book 极薄、spread 极宽、成交量爆发）
2. 在这些时段 voluntarily abstain，因为我们的下单本身会成为信号泄露
3. 训练一个 "market impact" 预测器：$\hat{I} = f(\text{spread}, \text{book_depth}, \text{volume\_intensity})$
4. 在 $\hat{I} > \theta$ 时，强制输出"平"（abstain）

**约束兼容性**：✅ 完全 stateless（impact 特征从当前窗口计算）

### 诚实评估：这在良文杯的小规模下没意义

1. 我们只有 5 只股票，每次出手的 PnL 是模拟计算，没有真实市场 impact。
2. 平台 evaluate 是给我们的预测序列打分，没有 feedback loop——预测精准不会导致市场移动。
3. 这个 idea 在现实交易系统里有价值（交易 desk 规模 → 需要 impact-aware execution），但在比赛评分里完全无关。

**结论**：有趣的哲学思考，零实际 PnL 价值。**直接排除**。

---

## 8. Idea H：非表格表示（GNN / SSM / CNN）

### 原理

**GNN on LOB**：bid_1..10 和 ask_1..10 作为 nodes，价格间距 / 时间相关性作为 edges，GNN 学 LOB 的全局结构。  
**Mamba/SSM**：把 100-tick × 226-d 作为序列，用 SSM 捕捉长程依赖。  
**Image CNN**：把 LOB 当 image（depth × time），用 CNN 提取空间模式。

### 为什么这条路已经基本封死

**NN 失败记录**：T1 / T19 / T32 三次失败，T32 是最强配置（DeepLOB + aug + dropout + label smooth + SWA），结果 **LOSO = −16.97**（vs LGB +11.46）。

**失败原因分析**（这很重要，要理解 why）：  
NN 会把 226-d feature 的"组合模式"当作 sym 的隐式签名。例如：sym_2 的 book 结构特别深，NN 会学到"deep book → sym_2 → 用 sym_2 专属规则"。当 test sym 是新的，NN 的"专属规则"触发混乱。GBDT 用 axis-aligned 分裂，不会构造这样的组合特征。

**GNN 会更好吗？**  
GNN 的 message passing 是非常 expressive 的——它能学到 LOB nodes 之间的复杂关系，但同时也有更强的"记住 sym-specific structure"的能力。如果 sym_1 的 LOB 图结构在度分布上与 sym_4 不同，GNN 会完美地记住这个区别（而不是泛化）。

**Mamba？**  
在 LOB 上没有强 evidence（R31/R35 都搜过）。MambaTab 在 tabular 设置下几乎无优势（feature 顺序无意义时）。

**唯一可能 work 的 NN 变体**：**显式 sym-invariance 约束 + 对抗训练**：
- Gradient Reversal Layer：在 backbone 后加一个 sym classifier，用梯度反转让 backbone 无法区分 sym → 强制 sym-agnostic 表示
- 这需要 NN，且训练不稳定
- 需要 2-3 天实验，且 3/3 失败的先例让预期偏低

**诚实评估**：非表格表示在初赛前的期望值是负的（NN 失败概率 > 70%，GPU 时间 > 4 小时）。

**Upside（如果成功）**：+2~+5 LOSO（因为 NN 能捕捉 GBDT 捕不到的交叉特征）  
**实现时间（仅 invariance NN）**：2-3 天  
**优先级（初赛）**：⚠️ 低；**优先级（决赛）**：⭐⭐⭐（专门做对抗 sym-invariance NN）

---

## 9. 没被要求但我认为最重要的：Idea I — 5 Horizon × 3 Class 软提交策略

### 原理
比赛规则：**5 个 horizon，取最好的那个参与排名**。这意味着我们的"最优提交策略"不只是找到最好的模型，而是找到 **PnL 最高的 (model, horizon, threshold) 组合**。

当前做法：专注 h_60（历史上最好），忽略 h_5/h_10/h_20/h_40。

**被忽略的机会**：如果 h_5 在某些 sym 上显著更好，我们应该知道。更激进：训练一个 **horizon selector** 模型，对每个测试样本决定用哪个 horizon 的预测。

但等等——平台评分是 **单个 horizon 提交** 的 PnL，不能混用不同 horizon 的预测（因为每个预测点的 horizon 是固定的）。所以这里的 optionality 是：**提交 5 次，每次用不同 horizon，选最高分那次算**。

**当前状态**：iter_006 是 h_60，其他 horizon 的 LOSO 有多好？如果 h_5 实际 LOSO +15 > h_60 +13.6，我们应该提交 h_5。

**这不是颠覆性，但是被漏掉的简单钱**。

---

## 10. 优先级矩阵

| Idea | Risk | Upside (LOSO Δ) | Time (天) | 初赛可行 | 优先级 |
|---|---|---|---|---|---|
| **A: KNN-Target Retrieval** | 低 | +1~+3 | 1~1.5 | ✅ | ⭐⭐⭐⭐ |
| **B: Multi-Horizon Meta-Stacker** | 低-中 | +1.5~+5 | 1.5~2 | ✅ | ⭐⭐⭐⭐ |
| **C: Direct PnL Bandit** | 中 | +0~+2 | 1~2 | ✅ | ⭐⭐⭐ |
| **D: Regime-Adaptive Threshold** | 低 | +0.5~+2 | 0.5~1 | ✅ | ⭐⭐⭐ |
| **E: Siamese Bid-Ask Features** | 低 | +0.3~+1.5 | 0.5~1 | ✅ | ⭐⭐⭐ |
| F: Generative Augmentation | 高 | +3~+8 | 5~10 | ❌ | ⭐⭐⭐⭐（决赛）|
| G: Causal Conservative Actor | N/A | ~0（比赛无意义）| — | ❌ | ❌ |
| H: GNN/SSM/CNN | 高 | −5~+5 | 2~3 | ⚠️低 | ⭐⭐（决赛 adversarial）|
| I: Horizon Selection Check | 极低 | 0~+2 | 0.5 | ✅ | ⭐⭐⭐⭐（立即！）|

---

## 11. Top 3：初赛前 5 天可执行

### 🥇 #1：Multi-Horizon Meta-Stacker（最高期望 upside）

**原理**：5 个 horizon LightGBM OOF → 15-d meta feature → small GBDT meta-layer → final decision

**Minimum Viable Experiment**：
1. 确认现有的 OOF predictions：h_60（iter_006 已有），其他 horizon 是否有 OOF？
2. 如果没有，快速训 h_5 和 h_10 的 5-seed LOSO（2-3 小时 GPU）
3. 把 5 × 3 = 15 个概率值 + 3 个 KNN features（Idea A）+ 5 个 regime features（Idea D）组成 23-d meta feature
4. 在 LOSO inner fold 上训 meta-GBDT（depth=3，防 overfit）
5. meta layer 的 objective = custom PnL（直接优化 trade PnL，因为 meta layer 层的训练集 PnL 是可知的）
6. 评估：LOSO sum meta_stacker vs iter_006 baseline

**实现时间**：2 天（含补训 h_5/h_10 OOF）  
**预期结果**：LOSO +15~+18  
**风险**：meta layer 需要 5 个 horizon 的 OOF，如果其他 horizon 没训过，需要补

---

### 🥈 #2：KNN-Target Retrieval Feature（确定性最高）

**Minimum Viable Experiment**：
```bash
# 0. 安装 FAISS
pip install faiss-gpu -q  # 或 faiss-cpu

# 1. 训练阶段（在 train_loso.py 里加）
import faiss
index = faiss.IndexFlatL2(226)
index.add(X_train_all_folds.astype('float32'))
faiss.write_index(index, 'knn_index.bin')
np.save('y_train_for_knn.npy', y_train_all_folds)

# 2. 推理端（Predictor.py 里）
index = faiss.read_index('knn_index.bin')
y_knn = np.load('y_train_for_knn.npy')
D, I = index.search(x_features.astype('float32'), k=5)
knn_feat = np.stack([
    y_knn[I].mean(axis=1),   # 平均邻居 label
    y_knn[I].std(axis=1),    # 邻居 label 方差（不确定性）
    D[:, 0],                  # 最近邻距离（越大越 OOD）
], axis=1)
x_aug = np.hstack([x_features, knn_feat])
```

**注意事项**：
- 评测平台没有网络，FAISS 必须离线安装到 requirements.txt（或直接把 faiss-cpu 的 so 文件打进 zip）
- index + y_knn 文件约 1.3GB + 12MB，总模型大小接近 2GB 限制，需要用 PQ 压缩 index
  - `index = faiss.IndexPQ(226, M=16, nbits=8)` → 226 × 4bytes → 226 bytes/sample → 1.47M × 226 bytes ≈ 330MB，OK！
- 在 LOSO 内部，KNN index 只用 train folds，不 leak test fold label

**实现时间**：1~1.5 天  
**预期结果**：LOSO +14.5~+16.5  
**风险**：PQ 压缩可能损失精度（但 approx KNN 在 226-d 上收益足够）

---

### 🥉 #3：Horizon Audit + Best-Single-Horizon Submission（立即！0.5 天）

**这不需要任何新代码**。如果 h_5 或 h_10 的 LOSO sum 实际高于 h_60，我们现在就在浪费分。

**Action**：
```bash
# 检查现有实验结果
ls experiments/T*/results.json | xargs grep -l "loso_sum\|cum_pnl"
# 找出各 horizon 最高分对应的实验
```

如果 h_5/h_10 没有足够的 LOSO 实验，快速训一个：
- h_5 LGB 5-seed 约 4 小时 GPU
- 比较 LOSO sum

**预期发现**：h_5 的 label 分布更好分离（α=0.05%，label 平衡性更好），可能有更高 PnL。如果 h_5 LOSO > h_60，换提交。

---

## 12. Top 2：决赛 Punt

### Finals #1：Conditional DDPM/Flow for LOB Data Augmentation

**为什么值得决赛投入**：
- 生成 synthetic 数据能 10x 扩充训练集，特别是对少数类（涨/跌）
- LOB-Bench 论文的 "model derailment" 问题在 short horizon（h_5/h_10）上相对轻
- 条件生成 + discriminator filter（real vs fake ROC）能保证质量
- 实现路线：class-conditional DDPM（label=0/1/2 作为 condition）在 100×226 的 LOB tensor 上

**风险**：
- 需要 diffusion model 收敛（至少 2 天 GPU training + 验证）
- 生成的数据需要通过 microstructure sanity check（bid < ask，spread > 0 等）
- 如果生成质量差，downstream GBDT 可能反而更差

**期望（如果做好）**：+3~+8 LOSO

---

### Finals #2：Adversarial Sym-Invariant Neural Network

**为什么值得决赛投入**：
- NN 失败的根本原因是隐式 sym 表示，不是架构问题
- 解决方案：Gradient Reversal Layer（Ganin & Lempitsky 2015）显式压制 sym 信息
- 架构：
  ```
  X(226-d) → backbone(128-d latent) → direction_head(3-class)
                         ↓
                  GRL (gradient reversal)
                         ↓
                  sym_classifier(5-class, only in training)
  ```
- 训练时：`loss = CE(direction) - λ·CE(sym_pred)`（GRL 自动取反）
- 推理时：GRL 不存在，只用 backbone + direction_head

**为什么初赛没时间做**：
1. NN 需要重写训练 pipeline（当前 T32 已废弃）
2. GRL 的 λ 调参 + stability debug 需要 2-3 天
3. 还需要验证 NN 能跑 symid=99 等 OOD sym 而不 IndexError

**期望（如果做好）**：NN LOSO +10~+16，ensemble with GBDT → +15~+20

---

## 13. 额外 Idea：Triplet Imbalance 快速补全

来自 r33（Optiver 2023 top 方案）：
```python
# 三元 triplet imbalance: (max-mid)/(mid-min)
for (a, b, c) in triplets:
    feats[f'triplet_{a}_{b}_{c}'] = (X[a] - X[b]) / (X[b] - X[c] + eps)
```

50 个 triplet 组合从 bid1/ask1/mid/spread1/depth_total 里取 → 50 个新特征。成本 0.5 天，预期 +0.2~+0.8。**这可以和 Idea A+B 并行实施**。

---

## 14. 实施路线图（5月7日-11日）

```
5/7(今天) → 快速 horizon audit（哪个 h 最高？），同时启动 T53-KNN 和 T54-MetaStack 设计
5/8       → T53 KNN index 训练 + 验证推理延迟；T47 multi-horizon aux OOF 补齐
5/9       → T54 Meta-Stacker training（用 KNN feat + 5 horizon OOF）；triplet features 加进 pipeline
5/10      → 评估：KNN 单独 LOSO？Meta-Stacker LOSO？选最好的配置打包提交
5/11      → 最终提交（12 小时限制，慎重选时间窗口）
```

---

## 15. 结论

**坦白话**：
- iter_006 的 +13.6 主要来自 h_60 LOSO 中 fold 4 的 +7.57（一只 sym 贡献了一半）。这是真实 alpha 还是 eval noise？这个问题我们回答不了，但说明 variance 很大。
- 真正的 ceiling-breaker 是"改变游戏规则"：要么目标函数不同（bandit PnL），要么 feature 空间不同（KNN retrieval），要么 model 层次不同（meta-stacker）。
- RL / 生成模型 / 因果建模在初赛 deadline 前是不切实际的。5 天内能跑出结果的只有 A+B+D+E+I。
- **不要再做 incremental feature engineering**——T39 的 saturation diagnosis 已经说得很清楚。

**最值得押注的单一 idea（如果只能选一个）**：Idea B——Multi-Horizon Meta-Stacker。理由：5 horizon 的协同信号是目前完全未被利用的信息维度，实现成本 2 天，失败 mode 温和（最差 = iter_006 不变）。
