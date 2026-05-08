---
name: r39_data_training_pipeline
description: 16 disruptive-but-basic ideas at data pipeline / training loop / evaluation layer (NOT feature/model). Top 5 each ≤ 4h.
type: research
---

# R39 — Data Pipeline / Training / Evaluation Disruptive-Basic Ideas

> **Date**: 2026-05-07
> **Worker**: R39 brainstorm (opus xhigh, 30 min budget)
> **Goal**: T75 `target=Δmid` 一刀换公式 → +9.79 LOSO。寻找**同 magnitude**、**同 basic**、**在 data/train/eval 层**的 lever。
> **NOT covered here**: 特征工程（→ r34/r36）、模型架构（→ r35）、决策规则微调（→ r37 已 ship 在 iter_013）。

---

## 0. 锚定 — 为什么聚焦 pipeline

T80 校准发现的硬事实：

| 量 | 数值 | 含义 |
|---|---|---|
| `Δ LOSO-equiv → Δ platform` 透传 | **0.70×** | 改 LOSO+1.0 只能换平台 +0.70 |
| iter_002 absolute LOSO−platform gap | **−10.5** | 基础模型已 inflated 10.5 分 |
| iter_013 absolute gap | **−17.0** | 越强模型 gap 越大 |
| Gap 增长源 | **DE 4D thresh 在 test set 上 fit** | 验证集污染主因 |

→ **结论**：当前 LOSO 增益 30% 在透传中蒸发，绝大多数因为 **(i) DE 在 442k test 上 fit + (ii) train/val split 太窄**。
任何能**让 LOSO ↔ platform 透传率从 0.70 → 0.85** 的 lever，单独就值 +5 platform，**比单独 +5 LOSO 更值钱**。

**这是 R39 的核心赌注**：data pipeline 层的 disruptive idea 不在"再加多少 LOSO"，而在"**让现有 LOSO 真的能透传**"。这是 T75 `target=Δmid` 同 magnitude 的下一个 lever。

---

## 1. TL;DR — Top 5 推荐（按 ROI 排序）

| Rank | Idea | 颠覆性 | 成本 | 预期 LOSO Δ | 预期透传率 Δ | Top 5? |
|---|---|---|---|---|---|---|
| 🥇 | **#1 K-fold cross-validated thresh on test (CV-DE)** | ★★★★★ | 2-3h | -1 ~ +0.5 (LOSO 看似掉) | **+0.10 ~ +0.20** | ✅ |
| 🥈 | **#2 Bootstrap thresh + median selection** | ★★★★ | 2h | -0.5 ~ +0.5 | **+0.05 ~ +0.15** | ✅ |
| 🥉 | **#3 Self-distillation (teacher → student soft targets)** | ★★★★★ | 4h | +0.5 ~ +2.0 | **+0.05 ~ +0.10** | ✅ |
| 4️⃣ | **#4 Drop label-boundary samples (|Δmid − α| < ε)** | ★★★★ | 1.5h | +0.5 ~ +2.0 | +0 | ✅ |
| 5️⃣ | **#5 2-stage residual boost (hard-sample mining)** | ★★★★ | 3.5h | +0.5 ~ +1.5 | +0 ~ +0.05 | ✅ |

**关键观察**：top-2 idea LOSO 看似掉一点，但**透传率提升直接换平台 +5**。**LOSO 不再是优化目标本身**，是 LOSO × 透传 = platform。

---

## 2. 完整 16 条 ideas（按层分组）

---

## A. 数据 split / 验证策略层

### #1 K-fold cross-validated thresh on test (CV-DE) ⭐ TOP-1

**一句话**：当前 DE 4D 在 442k test 整集上 fit → 直接 leak（即使是 stateless 的 thresh）。改为 **K=5 折切 442k**，每折 DE on 4 folds → eval on holdout fold → 平均 K 套 thresh 取均值/中位数。理论上等价于 OOF thresh，把 test-fit 的 inflation 消除。

**为何关键**：T62 已显示 split-train DE gap +0.28 LOSO；T80 校准证明 iter_013 整 −17 absolute gap 主要是 DE 4D thresh 在 test 上 fit 导致。**这个 idea 不为提 LOSO，是为提透传率**。

**为何没做**：T62 试过简单 50/50 split，没系统化 K-fold；DE 在小子集上慢的顾虑。

**实施**：
```python
# 442k → K=5 stratified by sym × session
folds = stratified_kfold(test_df, n_splits=5, group_keys=['sym', 'session'])
thresh_list = []
for k in range(5):
    train_folds = test_df[fold != k]
    eval_fold = test_df[fold == k]
    best = differential_evolution(
        lambda T: -loso_pnl_on(train_folds, T),
        bounds=[(0.40, 0.65)] * 4,
        ...
    )
    thresh_list.append(best.x)
final_thresh = np.median(thresh_list, axis=0)  # 或 mean
```

**预期**：
- LOSO 看似 -1 ~ +0.5（thresh 不再过拟 test → 数字降）
- **平台 +2 ~ +5**（透传率从 0.70 → 0.85+）
- **绝对最重要的"看似负但真的正"实验**

**契约**：✅ thresh 是 4 标量，inference 完全兼容
**成本**：2-3h（DE × 5 fold ≈ 5x 当前 DE 时间）

⚠ 重要：**LOSO 数字会掉**，必须用 T80 的预测公式 `plat = 19.23 + 0.70 × (LOSO − 36.23) + Δ_transmission_bonus` 评估。**只看 LOSO 会拒绝这个 idea**。

---

### #2 Bootstrap thresh + median selection ⭐ TOP-2

**一句话**：对 442k test bootstrap N=20 次（每次 sample 80% w/ replacement），DE on each → 20 套 thresh → 取 median。等价于 thresh 的 bagging。

**为何关键**：DE 4D 是 stochastic 优化（每次 init 不同会落到不同 local optima）。Bootstrap 检测稳定 thresh 区域 → 中位数 thresh **更接近 platform-true optimal**。

**为何没做**：DE 单次已 ~10 min，跑 20 次 = 3.5h，人觉得"多跑没意义"。

**实施**：
```python
np.random.seed(0)
thresh_list = []
for b in range(20):
    idx = np.random.choice(N, int(0.8 * N), replace=True)
    boot_df = test_df.iloc[idx]
    best = de_4d(boot_df)
    thresh_list.append(best.x)
final_thresh = np.median(thresh_list, axis=0)
# 也输出 std 检查 thresh 稳定性
```

**预期**：
- LOSO -0.5 ~ +0.5（稳定性提升，不一定 LOSO 数字升）
- **平台 +1 ~ +3**（透传率 +0.05 ~ +0.15）

**契约**：✅
**成本**：2h（自动化 20 次 DE）

可与 #1 叠加：先 K-fold split，每折再 bootstrap → 100 套 thresh 取 median。

---

### #6 Combinatorial Purged Cross-Validation for thresh (CPCV)

**一句话**：Marcos López de Prado 2018 的 CPCV：N=10 chunks，每次 hold-out k=2 chunks，产生 C(10,2)=45 个 paths。每 path DE thresh → 45 套 thresh 中位数。比 K-fold 多 **9 倍 path 数**。

**为何关键**：thresh tuning 的 finite-sample 噪声是 platform 透传率掉到 0.70 的硬来源。CPCV 是金融 ML 标配解。

**为何没做**：CPCV 实现略复杂，没人写过。

**实施**：
```python
from itertools import combinations
N_CHUNKS = 10
HOLD_K = 2
chunks = stratified_split(test_df, N_CHUNKS, by='sym × session')
paths = list(combinations(range(N_CHUNKS), HOLD_K))  # 45 paths
thresh_list = []
for hold_idx in paths:
    train = pd.concat([chunks[i] for i in range(N_CHUNKS) if i not in hold_idx])
    best = de_4d(train)
    thresh_list.append(best.x)
final_thresh = np.median(thresh_list, axis=0)
```

**预期**：LOSO 中性，**平台 +2 ~ +4**
**成本**：4h（45 次 DE，但每次 chunk 小约 8x 快 → 总 ~ 1h compute + 3h 实现）
**契约**：✅

---

### #7 Walk-forward DE：DE on val (date 76-79) only, eval on test (96-119)

**一句话**：当前 DE 在 442k test 上 fit。改为 DE 只在 val (date 76-79) 上 fit → 评估在 test (96-119)。**完全 unbiased**，符合时序原则。

**为何没做**：val 太小（只 4 天 ≈ 73k 样本），DE 信号弱；放弃了。

**实施**：
- DE 在 V4 val 73k 上 fit
- 评估时在 test 442k 上 apply

**预期**：
- LOSO -2 ~ +0（val 信号弱可能 underfit）
- 平台 +0 ~ +2

**契约**：✅
**成本**：1h
**风险**：val 太小可能噪声大，不一定胜 K-fold thresh。建议**与 #1/#2 配合**做 sanity check。

---

### #8 Anchored walk-forward k-fold（多 fold 而非单 split）

**一句话**：当前 V4 = 单 split (train 0-95% / val 5%)。改为 K=5 anchored walk-forward：
- Fold 0: train 0-19 / val 20-23
- Fold 1: train 0-39 / val 40-47  ← train 累积扩大
- ... 一直到 Fold 4: train 0-79 / val 80-95
每 fold 训 5 seed → 25 model 平均，每 fold OOF prob 用于 stacking。

**为何没做**：T58 walk-forward CV 是单 fold（W1/W2/W3）；anchored 5 fold 没系统化。

**实施**：5 fold × 5 seed = 25 model，每 fold ~50 min → 25 × 50 / GPU 并行 = ~4h

**预期**：LOSO +0.3 ~ +1.0（更多 OOF coverage），平台 +0.2 ~ +0.7
**契约**：✅
**成本**：4h

---

### #9 Sym-stratified walk-forward（跨 sym 平衡 train）

**一句话**：当前 V4 train=date 0-95%，5 sym 全部 in，但**date 顺序内 sym 比例不均**。改为分 fold 时**强制每 fold 5 sym 比例 = 全集比例**。

**为何没做**：未审计 fold 内 sym 比例。

**实施**：StratifiedShuffleSplit by `sym` 在 walk-forward 里
**预期**：LOSO +0.2 ~ +0.5（balance helps OOD calibration），平台 +0.1 ~ +0.3
**契约**：✅
**成本**：1h

---

## B. 训练数据增强层

### #10 Additive Gaussian noise (aug_b)

**一句话**：aug_a 是 [0.80, 1.20] **multiplicative** scale；aug_b 加 N(0, 0.005·σ_col) 高斯噪声 — implicit L2 reg + OOD 鲁棒。**从未试过**。

**为何没做**：aug_a 已工作，没人想试加性 noise。

**实施**：
```python
X_train_noisy = X_train + np.random.randn(*X_train.shape).astype(np.float32) * (col_std * 0.005)
```

**预期**：LOSO +0.2 ~ +0.6
**契约**：✅
**成本**：1.5h

---

### #11 Feature dropout (aug_c) — Cutout-style for tabular

**一句话**：训练时随机 mask 0~10% 的 feature 列（设为 0 或 col_mean）。强迫 model 在子集 feature 上工作 → ensemble 效果。已知在 GBDT colsample_bytree=0.8 类似但 dropout 更激进。

**为何没做**：GBDT 已有 feature_fraction，但是 row-level dropout（不同 sample mask 不同 cols）从未试过。

**实施**：
```python
mask_rate = 0.05
mask = np.random.binomial(1, 1-mask_rate, X.shape).astype(np.float32)
X_train_masked = X_train * mask
# 但 LightGBM 不直接支持 per-sample mask;
# 替代：训 K=5 model，每个 mask 不同 cols，再平均
```

**预期**：LOSO +0.2 ~ +0.5
**契约**：✅
**成本**：3h（K=5 model retrain）

---

### #12 Within-sym mixup（不跨 sym）

**一句话**：T52 cross-sym mixup 失败（OOD 破坏）。**within-sym** mixup（同 sym 内 mixup λ ~ Beta(0.4, 0.4)）是 vanilla mixup，不破坏 sym OOD 学习。从未试过。

**为何没做**：T52 失败给"mixup 不行"印象，但 within-sym 是不同变体。

**实施**：
```python
for batch in train_batches:
    sym_groups = batch.groupby('sym_id')
    for sym, group in sym_groups:
        idx_pairs = ...
        lam = np.random.beta(0.4, 0.4, size=len(group))
        X_mixed = lam * X[idx_a] + (1-lam) * X[idx_b]
        # 但 LightGBM 不接 soft target，要 NN 或转 regression
```

**预期**：LOSO +0.1 ~ +0.4（受 LightGBM 限制）
**契约**：✅
**成本**：3h
**注意**：需要 regression objective（#5 的延伸），**与 iter_013 regression 路线天然配合**。

---

### #13 Time-warping augmentation

**一句话**：100-tick window 沿时间轴随机 stretch/squeeze（线性插值到 80~120 行 → 重采样回 100 行）。增强对"采样频率轻微变化"的鲁棒性。

**为何没做**：A 股 LOB 是固定 3s 一档，frequency 不会变；但 platform 数据可能在边界上有 jitter，time-warp 提供保险。

**实施**：scipy.interpolate.interp1d 每行重采样
**预期**：LOSO +0 ~ +0.3（小 lever）
**契约**：✅
**成本**：2h
**评分**：低优先级，不进 top 5

---

## C. 训练样本 selection 层

### #4 Drop label-boundary samples (|Δmid − α| < ε) ⭐ TOP-4

**一句话**：3-class label 由 `1[Δmid > α]` (α=0.001) 切分。**临近边界的样本**（|Δmid − α| < 5e-5）**label 噪声极大**——稍微 jitter 就 flip。Drop 这些 ε-buffer 样本（约 5-10% 训练集）→ 模型只学清晰信号。

**为何关键**：T75 的 regression 已经 partly 解决（不再 hard binarize），但 iter_012 / iter_013 ensemble 还在用 3-class CE。这是 label noise 直接攻击。

**为何没做**：从未审计 label 噪声分布。

**实施**：
```python
delta_mid = compute_delta_mid_60(train)
buffer = 5e-5  # 0.05% — 5x 平均 jitter
keep_mask = (np.abs(delta_mid - 0.001) > buffer) & (np.abs(delta_mid + 0.001) > buffer)
X_clean = X_train[keep_mask]
y_clean = y_train[keep_mask]
# 训 5-seed on clean subset
```

**预期**：LOSO +0.5 ~ +2.0（去 label noise → 学到更纯信号）
**契约**：✅
**成本**：1.5h
**风险**：训练集少 ~7% 数据；可同时增 num_boost_round 5%。

⚠ 与 #5 (regression head) 路线**互补**：regression 已部分对齐，但 buffer-drop 进一步清理边界 → **iter_013 升级首选**。

---

### #5 2-stage residual boost (hard-sample mining) ⭐ TOP-5

**一句话**：Stage 1 训 5-seed 标准模型 → 在 train 集上获 prob → 找 prob 校准最差的 30% 样本（high prob 但 wrong）→ Stage 2 训第二个 model **only on those**，inference 时两 model logit avg。

**为何关键**：当前 ensemble 多样性饱和（T67 K=20 bagging 没收益），但**样本难度多样性**未利用。Hard sample 二阶段是 GBDT 标准 boosting 思想的应用版。

**为何没做**：把 boosting 仅当 LightGBM 内部，没在 dataset level 做 explicit。

**实施**：
```python
# Stage 1
model1 = lgb.train(params, train_dataset)  # 5-seed
prob1 = model1.predict(X_train)
correct = (prob1.argmax(-1) == y_train)
hard_mask = ~correct  # 或 entropy(prob1) > thresh
X_hard = X_train[hard_mask]
y_hard = y_train[hard_mask]
# Stage 2 only on hard
model2 = lgb.train(params2, lgb.Dataset(X_hard, y_hard))
# Inference
def predict(X):
    p1 = model1.predict(X)
    p2 = model2.predict(X)
    return 0.6 * p1 + 0.4 * p2
```

**预期**：LOSO +0.5 ~ +1.5
**契约**：✅
**成本**：3.5h（额外 model + DE 重 tune）
**风险**：stage 2 易过拟噪声，必须强 reg。

---

### #14 Curriculum learning by |Δmid| magnitude

**一句话**：先用 |Δmid| > 2α 的样本训前 30% rounds（明确信号），再加入剩余样本训完。
"先大信号，后小信号"。

**为何没做**：从未试过 curriculum。

**实施**：
```python
# Phase 1: 0-1000 rounds on big movers
big_movers_mask = np.abs(delta_mid) > 2 * 0.001
init_dataset = lgb.Dataset(X_train[big_movers_mask], y_train[big_movers_mask])
booster = lgb.train(params, init_dataset, num_boost_round=1000)
# Phase 2: continue on full data
full_dataset = lgb.Dataset(X_train, y_train)
booster = lgb.train(params, full_dataset, num_boost_round=2000, init_model=booster)
```

**预期**：LOSO +0.2 ~ +0.7
**契约**：✅
**成本**：2.5h

---

### #15 Self-distillation (teacher → student soft targets) ⭐ TOP-3

**一句话**：5-seed iter_013 ensemble 是 teacher → 它的 soft prob 当作 train target → 训 student LightGBM with `objective='cross_entropy'` (continuous label)。Student 学到 teacher 的"知识"，但因 target smoother → **泛化更好** + **OOD 校准 implicit 改善**。

**为何关键**：
- 直接攻击 LOSO→platform 透传率（teacher overfit 的 sharp prob 经 distillation 平滑后透传更稳）
- Distillation 是 NN 圈标配（Hinton 2015），**GBDT 圈被低估**
- 已有 5-seed teacher，免费可做

**为何没做**：GBDT distillation 不流行；team 默认"复制 teacher"没价值。

**实施**：
```python
# 1. Get teacher soft probs on train set (NEED OOF teacher to avoid leak!)
teacher_oof_prob = oof_predict_5seed(model_iter013, X_train)  # 3 col
# 2. Soften: T = 2 (temperature)
teacher_soft = softmax(np.log(teacher_oof_prob + 1e-9) / 2.0, axis=-1)
# 3. Student trains on soft target
student = lgb.train(
    params={'objective': 'cross_entropy', ...},
    train_set=lgb.Dataset(X_train, label=teacher_soft[:, 2]),  # P(class=up)
    # 训 3 个 binary student（每 class 一个）
)
# Inference: student avg with teacher
final_prob = 0.5 * teacher_prob + 0.5 * student_prob
```

**预期**：LOSO +0.5 ~ +2.0；**平台 +0.05 ~ +0.10 透传率提升**
**契约**：✅（更多 model.txt 但 inference 简单）
**成本**：4h（OOF teacher 已有，student 训 + DE 重 tune）

---

### #16 Drop ambiguous label samples + 训 confidence head

**一句话**：除了 #4 buffer drop，再加一个 **二元 confidence head**（pred = "可信" / "不可信"）。Inference 时 confidence < threshold 的 sample 强制 pred=1（不交易）。

**为何没做**：T72 binary cascade 是另一思路（UP/DOWN binary），confidence head 更通用。

**实施**：
```python
# Train confidence model
y_conf = (np.abs(teacher_prob.max(-1) - 0.95) < 0.1).astype(int)  # 高置信样本=1
conf_model = lgb.train({'objective': 'binary'}, lgb.Dataset(X, y_conf))
# Inference
conf_pred = conf_model.predict(X)
final_pred = np.where(conf_pred > 0.5, base_pred, 1)
```

**预期**：LOSO +0.2 ~ +0.5
**契约**：✅
**成本**：3h
**评分**：mid 优先级

---

## D. Threshold tuning 进阶（除 #1, #2, #6, #7 之外）

### #17 Penalized DE objective（L2 prior）

**一句话**：DE 目标 `argmin -PnL` → 改为 `argmin -PnL + λ · Σ(T - 0.50)²`（拉 thresh 向中心，防 corner overfit）。
λ 小（0.001）即可。

**为何没做**：怕降 LOSO；但 platform 透传率可能升。

**实施**：modify DE objective 1 line
**预期**：LOSO -0.3 ~ +0.3，**平台 +0 ~ +1.5**
**契约**：✅
**成本**：30 min

---

### #18 Bayesian / GP thresh selection

**一句话**：拟合 GP `(T_up, T_dn, d_up, d_dn) → PnL` over 100 random thresh evaluations → 取 GP posterior 的 `argmax E[PnL | data] - β·Var[PnL | data]`（risk-averse）。

**为何没做**：DE 简单粗暴够用；GP 实现复杂。

**实施**：sklearn GaussianProcessRegressor + scipy.optimize for posterior
**预期**：LOSO +0 ~ +0.5，**平台 +0.5 ~ +1.5**
**契约**：✅
**成本**：4h

---

## E. 评估方法层

### #19 Bootstrap PnL CI on val（不只点估计）

**一句话**：每个候选模型在 val/test 上 bootstrap N=100 次 → 每次 cum_pnl → 95% CI。决策时**用 lower bound** 而非 mean，**保 platform 不掉**。

**为何关键**：当前用 LOSO sum 比较模型，没量化方差；可能选择"高方差但 mean 高"的模型，platform 上塌。

**实施**：
```python
def bootstrap_pnl(preds, labels, deltas, N=100):
    rng = np.random.default_rng(0)
    pnls = []
    for _ in range(N):
        idx = rng.choice(len(preds), len(preds), replace=True)
        pnls.append(compute_pnl(preds[idx], labels[idx], deltas[idx]))
    return {'mean': np.mean(pnls), 'lower_5': np.quantile(pnls, 0.05), 'upper_95': np.quantile(pnls, 0.95)}
```

**预期**：LOSO 评估机制改变，**模型选择更鲁棒** → 平台 +0.5 ~ +1.5
**契约**：✅（仅评估）
**成本**：1h

---

### #20 Synthetic OOD test (sym shuffle / Gaussian perturbation)

**一句话**：把 442k test 的 sym ID **随机 shuffle**（破坏 sym-feature correlation），再 eval → 模型 PnL 不应大跌（true sym-agnostic 的 sanity check）。如果跌很多，说明 model 暗中 leak sym info。

**为何没做**：从未做 sym-shuffle robustness test。

**实施**：
```python
test_shuffled = test.copy()
test_shuffled['sym'] = np.random.permutation(test_shuffled['sym'].values)
loso_shuffled = eval_full_pipeline(test_shuffled, model)
print(f"LOSO drop after sym shuffle: {loso_orig - loso_shuffled}")
# 如果 drop > 5，说明 model 严重依赖 sym 暗信号
```

**预期**：debug 工具，不直接提分；但 catch sym-leak 后修复值 +1 ~ +3
**契约**：✅
**成本**：1h

---

## F. Inference-time 改进

### #21 TTA — predict 同 sample N 次 with aug，average

**一句话**：inference 时对每 sample 做 5 次 aug_a 缩放 → 5 个 prob → 平均。等价免费的 inference-time ensemble（同一 model）。

**为何没做**：speed concern；但 batch-vec inference 58x 加速后有空间。

**实施**：
```python
def predict_tta(X, model, n_tta=5):
    probs = []
    for _ in range(n_tta):
        X_aug = X * np.random.uniform(0.95, 1.05, X.shape).astype(np.float32)
        probs.append(model.predict(X_aug))
    return np.mean(probs, axis=0)
```

**预期**：LOSO +0.1 ~ +0.4
**契约**：✅（每次 predict 独立 random，符合 stateless）
**成本**：1h
**风险**：5x 推理时间 → 平台 1024 batch 限制需 check

---

### #22 Online calibration ban (确认不可做)

**一句话**：根据最近 batch 累计 PnL adjust thresh — **平台 stateless 禁止**。
✅ **不要做**。

---

## G. Hyperparam / 训练 trick

### #23 Cosine LR decay

**一句话**：当前 lr=0.05 固定。改 cosine: 0.10 → 0.005 over 4000 rounds。
LightGBM `learning_rates` callback 支持。

**为何没做**：r36 #9 提过，没实施。

**实施**：见 r36 #9
**预期**：LOSO +0.1 ~ +0.4
**契约**：✅
**成本**：1h

---

### #24 Lower LR + 2x rounds + stronger reg

**一句话**：lr 0.05 → 0.025，num_boost_round 4000 → 8000，min_data_in_leaf 200 → 500。
**T63 已部分覆盖**，但是否 explore 更深 reg space 待定。

**预期**：LOSO +0.2 ~ +0.5
**契约**：✅
**成本**：等 T63 结果

---

## 3. 排序矩阵（颠覆性 × 简易度）

| Rank | Idea | 颠覆性 | 成本 | 预期 LOSO | 预期透传率 | Top 5? |
|---|---|---|---|---|---|---|
| 1 | #1 K-fold CV-DE thresh | ★★★★★ | 2-3h | 中性 | **+0.10~0.20** | ✅ |
| 2 | #3 Self-distillation | ★★★★★ | 4h | +0.5~2.0 | +0.05~0.10 | ✅ |
| 3 | #4 Drop boundary samples | ★★★★ | 1.5h | +0.5~2.0 | 0 | ✅ |
| 4 | #5 2-stage residual boost | ★★★★ | 3.5h | +0.5~1.5 | 0 | ✅ |
| 5 | #2 Bootstrap thresh median | ★★★★ | 2h | 中性 | +0.05~0.15 | ✅ |
| 6 | #6 CPCV thresh | ★★★★ | 4h | 中性 | +0.05~0.20 | — |
| 7 | #19 Bootstrap PnL CI | ★★★ | 1h | (评估改) | +0.05~0.10 | — |
| 8 | #17 Penalized DE | ★★★ | 0.5h | -0.3~0.3 | +0~0.10 | — |
| 9 | #14 Curriculum learning | ★★★ | 2.5h | +0.2~0.7 | 0 | — |
| 10 | #10 Additive Gaussian noise (aug_b) | ★★ | 1.5h | +0.2~0.6 | 0 | — |
| 11 | #16 Confidence head + drop | ★★★ | 3h | +0.2~0.5 | 0 | — |
| 12 | #21 TTA inference | ★★ | 1h | +0.1~0.4 | 0 | — |
| 13 | #18 Bayesian GP thresh | ★★★ | 4h | +0~0.5 | +0.03~0.10 | — |
| 14 | #11 Feature dropout (aug_c) | ★★ | 3h | +0.2~0.5 | 0 | — |
| 15 | #8 Anchored walk-forward k-fold | ★★ | 4h | +0.3~1.0 | 0 | — |
| 16 | #20 Synthetic sym-shuffle test | ★★ | 1h | (debug) | +0~3 if leak | — |
| 17 | #9 Sym-stratified walk-forward | ★ | 1h | +0.2~0.5 | 0 | — |
| 18 | #13 Time-warping aug | ★ | 2h | +0~0.3 | 0 | — |
| 19 | #12 Within-sym mixup | ★★ | 3h | +0.1~0.4 | 0 | — |
| 20 | #23 Cosine LR | ★ | 1h | +0.1~0.4 | 0 | — |

---

## 4. 关键洞察

1. **当前最大 lever 不在 LOSO，而在透传率**。T80 显示透传 0.70 是硬瓶颈。R39 top-2（#1 + #2 + #6）都是为了从 0.70 → 0.85+，预期纯透传 +5 platform。这是与 r36/r37 完全不同的 lens。

2. **Self-distillation (#3) 是 GBDT 圈被严重低估的 trick**。NN 圈 Hinton 2015 标准，GBDT 几乎无人用。我们已有强 teacher (iter_013) 免费起步。

3. **Drop label-boundary samples (#4) 是 T75 regression idea 的"标签层延伸"**。T75 在 target 公式上换，#4 在 sample selection 上清理 boundary noise，是同思路下一波。

4. **2-stage residual boost (#5) 提供 explicit boosting between models**，与 LightGBM 内 boosting 互补。Hard-sample mining 在 NN 圈成熟，GBDT 没人 stage between models。

5. **Bootstrap & CV thresh 不会提 LOSO 数字**。如果 worker 派发时不交代清楚，验收人看 LOSO 掉就 reject 这个 idea。**必须用 T80 透传公式评估**：
   ```
   predicted_platform = 19.23 + 0.70 × (LOSO − 36.23) + transmission_bonus
   ```
   transmission_bonus 估计要做 holdout 验证（如：先在 80% test 上 fit thresh，再在 20% test 上算 PnL 比 K-fold 多多少）。

6. **TTA (#21) 是免费的 inference-time ensemble**。需 check 平台 1024 batch 是否容许 5x predict 时间。

---

## 5. 下一步建议（worker 派发优先级）

🥇 **T81_cv_de_thresh**：实现 #1 K-fold CV-DE → 在 iter_013 5-seed prob 上重 fit thresh（2-3h）
🥈 **T82_self_distill**：实现 #3 self-distillation on iter_013 teacher（4h）
🥉 **T83_drop_boundary**：实现 #4 buffer drop ε=5e-5 retrain 5-seed h_60（1.5h）
4️⃣ **T84_residual_boost**：实现 #5 2-stage hard-sample model（3.5h）
5️⃣ **T85_bootstrap_thresh**：实现 #2 bootstrap thresh median over 20 reps（2h）

5 个 worker **独立 ablation**：

- T81/T85（thresh 类）评估必须**双指标**：LOSO + bootstrapped held-out LOSO（split test 80/20，DE on 80, eval on 20）
- T82/T83/T84 评估常规 LOSO（已知与 platform 透传 0.70）

任何 T81/T85 在 holdout LOSO 上 ≥ +1.5 → ship
任何 T82/T83/T84 在 LOSO ≥ +1.0 → ship

**不要 5 个并行**——T81/T85 是 thresh 后处理，要先跑（不动 model 训练 cost 低）；T82/T83/T84 涉及重训，并行 GPU。

---

## 6. 已 RULE OUT（不进 16 条）

- ❌ **Online calibration / state**: 违反硬约束 #2
- ❌ **Per-sym normalization at inference**: 违反硬约束 #3（sym 可能新）
- ❌ **Temperature/Platt/Isotonic calibration**: T40 全部 destroy 信号
- ❌ **adversarial val set**: T66 已试 AUC=1.0 即 distribution 重合差 → no useful overlap
- ❌ **pseudo-labeling**: T65 fail
- ❌ **Class weight / focal**: T41 fail
- ❌ **Group DRO**: T45 fail
- ❌ **Cross-sym pooling**: T48 fail (inference incompat)
- ❌ **Cross-sym mixup**: T52 fail
- ❌ **PnL-aware loss**: T57 fail (best +11 << +13.6 baseline)

---

## 7. 与 r36/r37 的关系

| 报告 | 焦点层 | 已 ship |
|---|---|---|
| r36 | 特征 + decision rule + GBDT trick | T74 time / T76 snapshot / T77 ridge meta（部分 in iter_013） |
| r37 | "leaderboard gap" — decision rule + 简单 feature | T78 multi-horizon / **iter_013 = #1 regression head** ✅ |
| **r39** | **data pipeline + training loop + eval/thresh tuning（NEW LAYER）** | ⭕ 全新，无重叠 |

R39 与 r36/r37 **完全互补**：r36/r37 已 picked over feature + decision；r39 攻击 **how the model is trained and how thresh is selected**。

---

## 8. 给 PM 的执行建议

**短跑（1 天 8h）**：T81 (#1) + T83 (#4) 串行 → 先看 thresh 透传率 + label noise drop 是否 stack。
**中跑（2 天）**：+ T82 (#3) self-distillation。
**长跑（3-4 天）**：+ T84 (#5) + T85 (#2)。

**iter_014 候选**：iter_013 + #1 (CV-DE thresh) + #3 (distill) + #4 (drop boundary)。
**预期 iter_014 LOSO**: 36.23 → ~38.5 (Δ +2.3 from #3+#4)
**预期 iter_014 platform**: 19.23 + 0.70 × 2.3 + (透传 bonus from #1) → **~ 21.5 ~ 23.0**
比公榜 +29.18 仍差，但缩到 -6 ~ -8 范围（vs 当前 -10）。

如果 #1 透传 bonus 兑现 +3，iter_014 platform **可达 +24~26**，与公榜差 < 5。

---

RESULT: task=r39_pipeline metrics={n_ideas=24, top5=cv_de_thresh/self_distill/drop_boundary/2stage_residual/bootstrap_thresh} notes=核心赌注=透传率 from 0.70→0.85；前 5 全 ≤4h 实施；与 r36/r37 完全无重叠
