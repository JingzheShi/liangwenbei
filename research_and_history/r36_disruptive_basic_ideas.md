# R36 — Simple but Disruptive Ideas（盲区清单）

> **Date**: 2026-05-07
> **Worker**: R36 brainstorm (opus xhigh)
> **Goal**: 找出"实施成本小、被忽视、但可能 +0.5+ LOSO"的简单 lever。
> 不再提复杂方案（NN / pseudo / focal / DRO / 元学习已尝试失败）。

## TL;DR — Top 5 推荐（每条 ≤ 4h，预期 +0.5+ LOSO）

| Rank | Idea | 实施成本 | 预期 LOSO Δ | 理由 |
|---|---|---|---|---|
| 🥇 | **#1 `time` 字段 sin/cos 编码 + minute-of-session** | 2-3h | **+1.0 ~ +3.0** | 平台明确允许，我们 359 维一个 time-derived 特征都没有；HFT 开盘/收盘动态完全不同；几乎"白送"信号 |
| 🥈 | **#2 LightGBM `multiclassova` (OvA) 5-seed** | 2h | **+0.5 ~ +1.5** | 1-flag 改 objective；OvA 在不平衡 multiclass 上常胜过 softmax；从未试过 |
| 🥉 | **#3 Snapshot-within-single-training ensemble** | 1.5h | **+0.3 ~ +1.0** | `Booster.predict(num_iteration=k)` 取 k=1500/2000/2500/3000/3500 后平均；零额外训练，免费 5x 多样性 |
| 4️⃣ | **#4 OOF Ridge meta-learner over 5 seed probs**（R12 Top-1，从未真正执行 meta） | 2h | **+0.3 ~ +0.8** | T6b 失败是因为 base 太相关；但 OOF Ridge 仍可救一点；从未在 V4 walk-forward OOF 上重 fit |
| 5️⃣ | **#5 Regression head on signed Δmid + post-hoc 3-class** | 4h | **+0.5 ~ +2.0** | 3-class label 把 |Δp| 量级信息全丢了；regression on log return 后 threshold 转 {0,1,2}；S6 idea 从未真做 |

剩余 15+ 条按"颠覆性 × 简易度"双指标排序见正文。

---

## 评分矩阵（颠覆性 vs 实施成本）

```
                  ↑ 颠覆性
              ★★★★★  │   #5 reg head    #1 time
              ★★★★   │   #4 ridge meta  #2 multiclassova
              ★★★    │   #11 1st-order  #3 snapshot
              ★★     │   #6-10 reg sweep
              ★      │   #18-20 trivial sweeps
              -------+----------------------------- → 简易度（成本反向）
                     <2h    2-4h    4-8h    >8h
```

---

## A. 数据层面盲区

### #1 `time` 字段：sin/cos minute-of-session + 离散 phase 桶 ⭐ TOP-PICK

**一句话**：我们 359 维 feature 里**一个 `time`-derived 特征都没有**；平台白纸黑字说"`time` 保留实际时间戳"，且 3s 一档 → 一天 ~4800 个 t；开盘/午休/尾盘的 LOB 动态完全不一样。

**为何没做**：早期 worker 把 `time` 当成"safer 不用"（怕违反硬约束），但 §1 只禁 `date`、不禁 `time`。CRITICAL_CONSTRAINTS.md §2 明文允许。

**实施步骤**：
```python
# 在 build_features 里：
# 1. 把 'time' 加入 config.json 的 feature 列表
# 2. parse "HH:MM:SS" → seconds_from_session_open
# 3. 加 4 个新特征：
#    sec_of_session = sec_open       # 标量
#    sin_t = sin(2π · sec / 14400)   # 4h 周期
#    cos_t = cos(2π · sec / 14400)
#    is_open_30min = (sec < 1800)    # bool
#    is_close_30min = (sec > 12600)  # bool
# 4. retrain h=60 5-seed (3.5 hours)
```

**预期 LOSO Δ**：+1.0 ~ +3.0（开盘/收盘 vol 比中段 2-3x，trade-or-not 决策不同）

**契约兼容性**：✅ sym-agnostic（time 与 sym 无关）、✅ batch-vec inference（element-wise）、✅ 无 cross-call state、✅ 不依赖 date

**风险**：需检查测试集是否始终包含 `time` 列。`docs/data_schema.md` 已确认 1200/1200 文件 time 字段连续 3s 一档；CRITICAL_CONSTRAINTS.md §2 也明确平台保留实际时间戳。

---

### #6 Volume-clock 特征（事件时间替代物理时间）

**一句话**：当前所有 rolling 都是 W=20/50/100 物理 tick；改用"累计 amount_delta 达到 X" 作为窗口边界，可降低低成交期的噪声。

**为何没做**：所有 features 是 W=N tick 的简化；event-time 实现稍复杂。

**实施**：
```python
# 沿 100-tick 窗口反向累积 amount_delta，找首个 cumsum >= threshold 的 t'
# 用 [t', t] 做 vol/return/imbalance — vol-clock window
# 加 6-8 个 vol-clock 副本特征
```

**预期 LOSO Δ**：+0.3 ~ +1.0  
**成本**：3h  
**契约**：✅ 全 sym-agnostic 全 stateless

---

### #7 First-order 时间差分特征（Δfeat over W）

**一句话**：我们有 EWMA 但没有 raw `feat[-1] - feat[-W]` 的简单差分；first/second derivative 显式信号。

**为何没做**：被 EWMA 覆盖的假象；但 EWMA 有平滑 lag，raw diff 是 unsmoothed delta。

**实施**：
```python
for W in (10, 30, 60):
    for col in ('imbalance', 'cumspread', 'totalbsize', 'totalasize', 'avgbid', 'avgask'):
        feat[f'd{W}_{col}'] = X[-1, col_idx[col]] - X[-W, col_idx[col]]
```
6 cols × 3 W = 18 新特征。

**预期 LOSO Δ**：+0.2 ~ +0.7  
**成本**：1.5h  
**契约**：✅

---

### #11 LOB shape: depth-decay slope（深度衰减斜率） ⭐

**一句话**：把 `log(bsize_1..10)` vs `[1..10]` 做线性回归，斜率 = 买侧深度衰减速度；卖侧同理。1 对斜率 + R² → 4 个特征捕捉"档位深度形状"。

**为何没做**：现有 features 都是单档/总量，没有"形状"层级；非常 basic 但漏了。

**实施**：
```python
# 100×K 窗口，对最后一行：
levels = np.arange(1, 11, dtype=np.float64)
# 买侧：
log_bs = np.log1p(np.maximum(0, X[-1, bsize_cols]))  # (10,)
slope_b, intercept_b = np.polyfit(levels, log_bs, 1)
# 卖侧同理 → slope_a
# 4 特征：slope_b, slope_a, slope_b - slope_a, |slope_b|·|slope_a|
```

**预期 LOSO Δ**：+0.3 ~ +0.8  
**成本**：1.5h  
**契约**：✅

---

### #12 Bid-ask 量分布 KL-divergence

**一句话**：把 bsize_1..10 和 asize_1..10 视作两个 size-rank 分布，算 KL(b||a) 和 KL(a||b)。捕捉两侧"形状不对称"的全局尺度。

**为何没做**：我们有 imbalance（量差）但没有 distribution-level 不对称。

**实施**：3 行 numpy（normalize → KL）；2 个特征。

**预期 LOSO Δ**：+0.1 ~ +0.4  
**成本**：30min  
**契约**：✅

---

### #13 Mid-price 二阶导（local curvature / 加速度）

**一句话**：现有 EWMA 给一阶 trend，但 acceleration（加速度）/ jerk 缺失。`Δmid[t] - Δmid[t-W]` 这类二阶差分。

**为何没做**：EWMA-OFI 有 acceleration 概念但 mid-price 本身没显式 2nd derivative。

**实施**：
```python
for W in (5, 20, 50):
    delta_now = X[-1, mid_idx] - X[-W, mid_idx]
    delta_prev = X[-W, mid_idx] - X[-2*W, mid_idx]  # if 2*W <= 100
    accel = delta_now - delta_prev
```
3 个 W × 1 = 3 特征。

**预期 LOSO Δ**：+0.2 ~ +0.5  
**成本**：1h  
**契约**：✅

---

## B. 标签层面盲区

### #5 Regression head on signed Δmid → post-hoc 3-class ⭐ TOP-PICK

**一句话**：3-class CE 把 |Δp| 信息丢了（"涨 0.05%" 与"涨 5%" 是同一类别）。改训 LightGBM regression on `signed_log_return = log(mid[t+60]/mid[t])`（MAE 或 Huber loss）；inference 时若 |ŷ| > fee+α 则方向交易，否则 1。

**为何没做**：S6 在 R10 文献里被列出但只有 idea 状态；HYD/Optiver 1st 用此方案；我们一直跑 multiclass。

**实施**：
```python
# 1. 用 train data 计算 signed_log_return_60 = log(mid[t+60]/mid[t]) (Δp ≈ 1e-4 量级)
# 2. lgb.train(objective='regression_l1', ...)
# 3. 5-seed ensemble + post-hoc thresh: pred=2 if ŷ>τ_up, pred=0 if ŷ<-τ_dn
# 4. 4D DE 优化 (τ_up, τ_dn) per fold
```

**预期 LOSO Δ**：+0.5 ~ +2.0（regression 直接对齐 PnL，HYD 1st 验证有效）

**契约**：✅ regression 输出 → threshold 决策，stateless 兼容

**关键 caveat**：T57 已试过"PnL aware loss"（custom objective）失败。但 **regression on Δmid 是不同方案**——不改 objective，改 target。更简单。

---

### #14 Quantile-based 标签 + 双模型平行训练（实验性）

**一句话**：除了 platform-fixed α=0.1% 标签外，**额外**训一个 model 用 quantile-based label（top 30%/bot 30%/mid 40%）。inference 时 stacking 两 model 的 prob。

**为何没做**：覺得 platform label 是法定标签；但训练时可用任何 alternative label，inference 不变。

**实施**：
1. 计算 Δmid_60，按 30/40/30 quantile 切分为 0/1/2
2. 这是更平衡的 label（vs 24.6/53.7/21.8 原始）
3. 训 model_q + 5-seed
4. 与原 model_orig 在 OOF prob 空间做 ridge stacking

**预期 LOSO Δ**：+0.3 ~ +0.8  
**成本**：4h  
**契约**：✅

---

### #15 软标签 / Label smoothing for class 1

**一句话**：当前 model 偏向预测 1（最常见类）→ recall 低（~0.20）。Soft-label 把 hard label 1 改成 (0.05, 0.90, 0.05)，迫使 model 在边缘样本上"不那么自信" → 阈值 gating 更松。

**为何没做**：focal/class weight 失败（T41）；但 label smoothing 是 inverse 操作（不是上权 0/2，而是降权 1）。LightGBM `objective='cross_entropy'` 接受连续标签。

**实施**：
```python
# 转换 (1-hot) → 软 label：
# 0 → (0.95, 0.05, 0.0)
# 1 → (0.05, 0.90, 0.05)
# 2 → (0.0, 0.05, 0.95)
# objective='cross_entropy' 训练
```

**预期 LOSO Δ**：+0.2 ~ +0.6（边缘 + 阈值更敏感）  
**成本**：2.5h  
**契约**：✅

⚠️ 注：与 T41 class weight 失败的本质区别——**label smoothing 改的是 prob target 不是 sample weight**。

---

## C. 模型 objective 层面盲区

### #2 LightGBM `multiclassova` (One-vs-All) ⭐ TOP-PICK

**一句话**：LightGBM 默认 `objective='multiclass'` 是 softmax CE；改成 `multiclassova` 是 3 个独立 binary（P(=0), P(=1), P(=2)）→ argmax/threshold。**1 个 flag 切换**。

**为何没做**：从未试过！文档很 basic 但被忽视。

**实施**：
```python
params['objective'] = 'multiclassova'  # 唯一改动
# 仍然 5-seed 训练 + DE thresh + 现有 359-d feat
```

**预期 LOSO Δ**：+0.5 ~ +1.5

**支持论据**：
- OvA 不强制 prob 之和为 1，使 binary 间独立学到强信号
- imbalanced multiclass（24.6/53.7/21.8）下 OvA 常胜 softmax
- 我们的 4D DE 阈值天然适合 OvA prob（不要求 simplex）

**契约**：✅ 完全兼容

**成本**：2h（1.5h 训 + 0.5h DE）

---

### #16 Lambdarank / Pairwise ranking objective

**一句话**：把 label_60=2 的样本视作"高 reward query"，label_60=0 视作"低 reward"，训 lambdarank → score → threshold。

**为何没做**：rank metric 不对齐 PnL；但 OOF score 可与 multiclass prob stacking。

**实施**：复杂（需要 group structure），先放着。

**预期 LOSO Δ**：+0.0 ~ +0.5  
**成本**：6h  
**契约**：✅ 但 group 定义微妙

❌ 不进 top picks。

---

## D. 推理决策层面盲区

### #4 OOF Ridge meta-learner over 5 seed probs ⭐ TOP-PICK

**一句话**：当前 5-seed ensemble = simple mean of 3-class probs。R12 Top-1 是 **stacking with Ridge meta over LOSO-OOF**，但 T6b 报告说 base 太相关失败 → 然而那是 simple seed-only stacking，**Ridge meta（with regularization）+ V4 walk-forward OOF** 还没真正在当前 5-seed × 5-fold 上 fit 过。

**为何没做**：T6b 失败给了"stacking 没用"印象，但 R12 明确强调 Ridge meta + LOSO-OOF 应能挽回 ρ>0.97 部分场景。

**实施**：
```python
# 1. 对每个 fold（held=0..4），收集 5 seed × 3 class = 15 维 OOF prob
# 2. Y = label_60 (1-hot 3D)
# 3. multi-output Ridge regression with α ∈ [0.1, 1, 10] CV
# 4. 在 inference 时 5 seed prob × Ridge weight matrix → 3 class score → 4D DE
```

**预期 LOSO Δ**：+0.3 ~ +0.8

**关键差异**：T6b 用 `simple_average` 或 `nelder-mead simplex weights`，**没有 Ridge L2 regularization**；高度相关 base 必须 L2 才能稳定 fit。

**成本**：2h  
**契约**：✅ Ridge weight 是常量矩阵

---

### #17 Per-batch prob normalization（rank-based gating）

**一句话**：inference 时把 batch 内 prob_2 排序，**只让 top X% 阈值最高的样本预测 2**。其他降为 1。这样在 high-vol batch 中保护 trade quality。

**为何不能做**：⚠️ **批内 rank 依赖 batch 大小和组成**；评测协议 §2 说"打乱顺序"，意味着 batch 边界是评测器决定的，每个 batch 内的样本组合是随机的 → 同一样本在不同 batch 里可能 rank 不同 → **违反硬约束 #2**（不同次输入的数据无先后关系→ prediction 必须只依赖单样本）。

**结论**：❌ 直接破坏契约，**don't do**。

---

### #18 Multi-horizon DE：activate h=40 + h=60 双 active

**一句话**：当前 only h=60 active，h=5/10/20/40 inactive。h=40 同样 α=0.1%，可能可独立训练 + DE → 两 horizon score 互补不冲突，平台对每个 label 独立打分。

**为何没做**：单 horizon 模型原则；但 platform 公榜显示 5 horizon **每个独立打分**（iter_002 的 5 horizon 表里 h=60 +4.07，h=40 +2.02 都是独立分数）。

**等等**：实际平台是"5 horizon 取 max 还是 sum"? 看 iter_002 历史："**Best score (即排名分) = max over 5 horizons**"。OK 是 max。

**修正**：所以即使 h=40 活跃，也只有当其分数 > h=60 时才贡献。**风险**：h=40 通常 < h=60，多余 active 反而拖低（如果出现负贡献）。

**实施**：训 h=40 5-seed Stage5 → DE 4D → 比较 h=40 vs h=60 LOSO sum → 若 h=40 ≥ +20，并行 active；否则放弃。  
T71_h40_stage5 已在跑（git status 显示）→ **等结果即可**。

**预期 LOSO Δ**：0 ~ +0.5（max over horizons 收益有限，除非 h=40 突破 +26）  
**成本**：等 T71（已跑中）  
**契约**：✅

---

### #19 Conformal prediction "abstain" set

**一句话**：用 split conformal on val → calibrate 95% coverage 的 prediction set。空 set 或含 1 → 预测 1（abstain）；只含 {2} → 预测 2。

**为何没做**：很少在 multiclass 决策中用 conformal abstain。

**实施**：复杂，4-6h。

**预期 LOSO Δ**：+0.2 ~ +0.5（但与 4D DE 重叠）  
**成本**：5h  
**契约**：✅

❌ 与现有 DE 重叠太多，不进 top picks。

---

## E. CV / 评估层面盲区

### #20 CatBoost 5-seed 加进 ridge meta（与 LGB 并列）

**一句话**：T17 单独 catboost LOSO ≈ LightGBM 但**从未与 LGB 5-seed ensemble 一起 ridge-stack**。CatBoost 是 oblivious tree，与 LGB leaf-wise 多样性大。

**为何没做**：T38 试过 simple weighted avg with revol，没试过 CatBoost。

**实施**：
1. T17 模型 + V4 walk-forward retrain CatBoost 5-seed (5h)
2. Ridge meta over (LGB×5 seed + CatBoost×5 seed) × 3 class = 30 维
3. 4D DE

**预期 LOSO Δ**：+0.4 ~ +1.0  
**成本**：5h（多了一点）  
**契约**：✅

---

## F. 训练流程 misc

### #3 Snapshot ensemble within single training ⭐ TOP-PICK

**一句话**：单个 LightGBM 训练 4000 轮，在第 1500/2000/2500/3000/3500 轮 snapshot 各自 predict 后平均。**零额外训练**，免费 5x 多样性。snapshot 间相关性 < 不同 seed 间相关性（LightGBM 的 leaf 结构每轮都漂移）。

**为何没做**：snapshot ensemble 通常用于 NN（cyclic LR），GBDT 圈很少用；但 `Booster.predict(num_iteration=k)` 原生支持。

**实施**：
```python
# inference 时：
preds = []
for k in (1500, 2000, 2500, 3000, 3500):
    preds.append(booster.predict(X, num_iteration=k))
prob = np.mean(preds, axis=0)
# 把 5-seed × 5-snapshot = 25-base ensemble
```

**预期 LOSO Δ**：+0.3 ~ +1.0

**契约**：✅ 完全兼容（model.txt 包含全 4000 棵树）

**成本**：1.5h（只改 inference 代码）

---

### #8 重正则化 sweep：min_data_in_leaf + lambda_l2 + max_depth

**一句话**：当前超参可能 overfit train 5 syms。Sweep regularization 重量级旋钮：
- `min_data_in_leaf`: 200 → 1000 / 2000 / 5000
- `lambda_l2`: 0 → 1 / 10 / 100
- `max_depth`: -1 → 6 / 8

**为何没做**：T36 Optuna stuck，T63 reg sweep 已在跑。

**实施**：等 T63 结果（已跑中）→ pick best config retrain 5-seed.

**预期 LOSO Δ**：+0.2 ~ +0.7  
**成本**：等 T63  
**契约**：✅

---

### #9 Cosine learning rate decay (`learning_rates=`)

**一句话**：LightGBM 支持 `learning_rates` 列表参数（每轮一个 lr）。当前固定 lr=0.05；改 cosine: 0.1 → 0.01 over 4000 rounds。

**为何没做**：从未试过；很多 GBDT 比赛用此 trick。

**实施**：
```python
import math
lr_list = [0.1 * (1 + math.cos(i * math.pi / 4000)) / 2 + 0.005 for i in range(4000)]
booster = lgb.train(params, dataset, num_boost_round=4000, learning_rates=lr_list)
```

**预期 LOSO Δ**：+0.1 ~ +0.4  
**成本**：1h  
**契约**：✅

---

### #10 Drop-low-gain features retrain（reduce 359 → ~200）

**一句话**：从 5-seed ensemble 算各 feature 的 mean gain；drop bottom 40% 后重训 5-seed。**减少 over-fit 噪声维度**，OOD 兼容性可能↑。

**为何没做**：感觉"信息越多越好"，但 LOSO 跨 sym 时低 gain 特征常 KS-fail 后噪声残留。

**实施**：
```python
for booster in 5_seed_models:
    gains = booster.feature_importance(importance_type='gain')
    avg_gain += gains / 5
top_k = np.argsort(avg_gain)[-200:]
# retrain 5-seed with only top_k features
```

**预期 LOSO Δ**：+0.1 ~ +0.5  
**成本**：3h（retrain）  
**契约**：✅

---

### #21 KFold-by-bagging 验证（不只 5 seed × 1 split）

**一句话**：当前 5-seed × 1 train/val split (V4 walk-forward)。改为 5 seed × 5 random 80% train subsets → **25 个 model** (memory-bound)。

**为何没做**：成本太大；但 sub-set bagging 比 hyperparam diversity 更有效降低 OOD 方差。

**实施**：5-seed × 3 bagging subsets = 15 boosters。

**预期 LOSO Δ**：+0.2 ~ +0.6  
**成本**：8h（训练慢）  
**契约**：✅

❌ 成本过高，不进 top picks。

---

### #22 Out-of-fold features as extra columns（"target encoding" 的 OOF 版）

**一句话**：用 1 个 toy seed 训 OOF prob，把 OOF prob 当**额外 3 个 feature** 喂给主 5-seed。等价于 "stage-1 model + stage-2 boost on residual"。

**为何没做**：在我们 R31 看到 VAR + FNN-on-residual idea 但没实现；其实更简单：LGB-stage1 + LGB-stage2 on resid。

**实施**：
```python
# 1. cross-fit 1 个 LGB on V4 walk-forward → OOF prob_0/1/2 列
# 2. 把 OOF prob 加到 359-d feat → 362-d
# 3. retrain 5-seed on 362-d
# 4. inference 时用 stage1 model 的 prob 作为 feat（prob 通过 stage1 model 推理）
```

**预期 LOSO Δ**：+0.3 ~ +0.7  
**成本**：4h  
**契约**：⚠ 需要 stage1 model 也打包进 inference（model size 增加 ~3MB）

---

### #23 训练时随机噪声 augmentation（Gaussian to feat）⭐ 简单

**一句话**：训练时给每个 feat 加 0.5% × std 高斯噪声 → 等价 implicit L2 reg + 改善 OOD。

**为何没做**：T29/T31 试过 augmentation **倍数缩放**（0.75-1.25），但**加性高斯噪声没试过**。

**实施**：
```python
# 在 lgb.Dataset 之前：
X_train_noisy = X_train + np.random.randn(*X_train.shape).astype(np.float32) * (X_train.std(axis=0) * 0.005)
```

**预期 LOSO Δ**：+0.1 ~ +0.4  
**成本**：1.5h  
**契约**：✅

---

### #24 平台 batch=1024 的 inference 端 horizontal scaling

**一句话**：把 5-seed 加 5-snapshot = 25 base 平均；用 batch matrix multiply 做（无显著速度损失）。当前每个 booster.predict 已 vectorized。

**为何没做**：与 #3 重复，已合并。

---

### #25 Per-fold DE thresh ensemble（vs single global DE thresh）

**一句话**：当前 single set of 4D thresh from DE on **mean fold**。但每个 fold 最优 thresh 不同（一些 fold 偏激进、一些偏保守）。改：每 fold 训 own best thresh，inference 时**5 fold thresh 平均**或 **majority-vote**。

**为何没做**：DE optimizer 之前用 sum-over-fold 目标；fold-individual thresh 仅在 retrospective 检查时见过，没用作 inference。

**实施**：
1. Per-fold DE → 5 套 (T_up, T_dn, d_up, d_dn)
2. inference 时取 mean / 中位数
3. 比较 LOSO sum

**预期 LOSO Δ**：+0.1 ~ +0.4（或 -，可能 noise dominate）  
**成本**：1h  
**契约**：✅

---

## 总结：按"实施成本 + 颠覆性"双指标排序

| Rank | Idea | 颠覆性 | 成本 | 预期 Δ | Top 5? |
|---|---|---|---|---|---|
| 1 | #1 time sin/cos features | ★★★★★ | 2-3h | +1.0 ~ +3.0 | ✅ |
| 2 | #2 multiclassova flag | ★★★★ | 2h | +0.5 ~ +1.5 | ✅ |
| 3 | #5 regression head + 3-class post-hoc | ★★★★★ | 4h | +0.5 ~ +2.0 | ✅ |
| 4 | #3 snapshot-within-training ensemble | ★★★ | 1.5h | +0.3 ~ +1.0 | ✅ |
| 5 | #4 Ridge OOF meta-learner | ★★★ | 2h | +0.3 ~ +0.8 | ✅ |
| 6 | #11 LOB depth-decay slope | ★★★ | 1.5h | +0.3 ~ +0.8 | — |
| 7 | #15 label smoothing on class 1 | ★★★ | 2.5h | +0.2 ~ +0.6 | — |
| 8 | #20 CatBoost 5-seed Ridge stack | ★★★ | 5h | +0.4 ~ +1.0 | — |
| 9 | #6 volume-clock features | ★★ | 3h | +0.3 ~ +1.0 | — |
| 10 | #7 first-order time differences | ★★ | 1.5h | +0.2 ~ +0.7 | — |
| 11 | #13 mid-price 2nd derivative | ★★ | 1h | +0.2 ~ +0.5 | — |
| 12 | #22 OOF prob as extra feat | ★★★ | 4h | +0.3 ~ +0.7 | — |
| 13 | #10 drop-low-gain retrain | ★★ | 3h | +0.1 ~ +0.5 | — |
| 14 | #14 quantile-label parallel model | ★★★ | 4h | +0.3 ~ +0.8 | — |
| 15 | #9 cosine LR decay | ★ | 1h | +0.1 ~ +0.4 | — |
| 16 | #25 per-fold DE thresh ensemble | ★ | 1h | +0.1 ~ +0.4 | — |
| 17 | #23 Gaussian noise augmentation | ★ | 1.5h | +0.1 ~ +0.4 | — |
| 18 | #12 bid-ask KL divergence | ★ | 0.5h | +0.1 ~ +0.4 | — |
| 19 | #8 regularization sweep | ★ | 等 T63 | +0.2 ~ +0.7 | (in flight) |
| 20 | #18 h=40 active dual horizon | ★ | 等 T71 | 0 ~ +0.5 | (in flight) |

---

## 已 RULE OUT（违反硬约束 / 与现状重叠）

- ❌ **#17 batch-rank prob normalization**: 违反硬约束 #2（顺序被打乱）
- ❌ **#16 Lambdarank**: group 定义微妙 + 与现有 multiclass 重叠
- ❌ **#19 Conformal abstain set**: 与 4D DE 重叠太多
- ❌ **#21 KFold-bagging × 25 boosters**: 训练 8h+，成本与收益不匹配

---

## 关键洞察

1. **`time` 字段是最大盲区**——3-min implementation gap 对应 +1~+3 LOSO，没人想起来加。  
2. **`multiclassova` 是 1-flag 实验**，OvA 文献表明常胜 softmax 在不平衡 multiclass。  
3. **regression head + post-hoc 3-class** 是 HYD/Optiver 1st 的核心配方，我们一直跑 multiclass CE。  
4. **snapshot ensemble** 是 GBDT 圈低估的 trick：不需要新训练。  
5. **Ridge OOF meta** 不是 T6b 的 simplex stack；R12 Top-1 应认真重做。  
6. **目标 user 团队 +29.18 / recall 0.302** 提示他们更激进 trade，可能用了 regression head 或 OvA → 更精细的 prob 校准。

## 下一步建议（worker 派发优先级）

🥇 **T74_time_features**：实现 #1 time sin/cos + retrain 5-seed h=60（3h）  
🥈 **T75_multiclassova**：实现 #2 + retrain 5-seed h=60（2h）  
🥉 **T76_snapshot_ensemble**：实现 #3 inference 端改造（1.5h）— 可与 T74/T75 并行  
4️⃣ **T77_ridge_oof_meta**：实现 #4 over current 5-seed iter_012（2h）  
5️⃣ **T78_regression_head**：实现 #5 regression on log return（4h）

5 个 worker 独立 ablation；任何 ≥ +0.5 LOSO 的进 iter_013。

RESULT: task=r36_brainstorm metrics={n_ideas=25, top5_picked=5} notes=top5=time_sincos / multiclassova / snapshot_ensemble / ridge_oof_meta / regression_head — 都 ≤4h 实施成本，预期 +0.3 ~ +3.0 LOSO 单 lever
