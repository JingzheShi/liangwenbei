# 良文杯 2026 — 关键设计选择与 Tricks 速查

> **来源**：从 192 个 T 实验、22 次平台提交、1091 个 review 问题、3014 行综合答案（`FINAL_SUBMISSION_DECISIONS_EXPLAINED.md` / `HISTORY_EVIDENCE_DB.md` / `answers.md` / `0512_tricks_table.md` / `0512_final_report.md` / `AUDIT_BATCH_LEAKAGE.md`）中提炼的核心 design choices。
>
> **平台 SOTA**：T188v2 50+50 = **+35.64**（从 mmpc_demo baseline **−6.65** 起步，绝对增益 **+42.29**）
>
> **取舍标准**：能影响"未来 6 个月新人做正确 design choice"的保留；trivial 工程选择 / 无 ablation 的 HP 微调 / 命名 nitpick 一律剔除。
>
> **数字阅读规约**：
> - 「LOSO」= 5-fold leave-one-sym-out（Phase 0-1, T1-T35）
> - 「LOSO-eq」= 全 sym 训练 + DE 阈值搜索在 442k 行本地测试集上（Phase 2, T44-T74）
> - 「LOSO-eq + V4」= train 0-79 / val 80-95 / test 96-119 walk-forward（Phase 3-5, T75-T155）
> - 「holdout in-sample」= date 0-119 全数据训练 + 96-119 评分（Phase 6-7, T156-T192，**含 in-sample 污染**，仅看 delta）
> - 「平台」= 公榜得分（5 horizon 取最高，**唯一 OOD ground truth**）

---

## 目录

- [§0. 三次大突破速查](#0-三次大突破速查)
- [§1. 训练方法的关键 tricks](#1-训练方法的关键-tricks)
- [§2. 决策层的关键 tricks](#2-决策层的关键-tricks)
  - [§2.0 Execution Strategy 完整决策链（pred → action 的全 pipeline）](#20-execution-strategy-完整决策链pred--action-的全-pipeline)
  - [§2.1.1 Threshold 求法历史（0.0003 / 0.000216 是怎么得来的）](#211-threshold-求法历史00003--0000216-是怎么得来的)
- [§3. 特征工程的关键 tricks](#3-特征工程的关键-tricks)
- [§4. 模型架构的关键选择](#4-模型架构的关键选择)
- [§5. 推理优化的关键 tricks](#5-推理优化的关键-tricks)
- [§6. 评估方法论的关键 tricks](#6-评估方法论的关键-tricks)
- [§7. 试过但 FAIL 的关键尝试（负面教训）](#7-试过但-fail-的关键尝试负面教训)
- [§8. 数学/逻辑层面的疑点（USER-RAISED）](#8-数学逻辑层面的疑点user-raised)
- [§9. 平台 SOTA 完整决策链（TLDR 数据流）](#9-平台-sota-完整决策链tldr-数据流)
- [§10. 最关键的 8 条工程经验](#10-最关键的-8-条工程经验)

---

## §0. 三次大突破速查

整个项目 22 次平台提交里，只有 4 次 delta ≥ +1：3 次「大突破」+ 1 次「ensemble 收尾」。其他十几次都在 ±0.5 噪音区。

| 节点 | T-编号 | trick | 平台 PnL | Δ vs 前 SOTA | 突破本质 |
|---|---|---|---|---|---|
| iter_002 | T5b | LGB 多 h + 3-class CE | +4.07 | +10.72 (vs mmpc_demo −6.65) | 首次平台正分 |
| iter_010 | T61 | batch-vec 推理 58× 加速 | (前几次超时) | — | **首次通过 3h 限时** |
| **iter_013** | **T75** | **Δmid L2 regression + EV gate** | **+19.23** | **+15.16** | **1st 突破：损失对齐 PnL 目标** |
| **iter_015** | **T87** | **SPO+ DFL NN 微调（与 LGB 集成）** | **+28.16** | **+8.93** | **2nd 突破：决策焦点学习** |
| iter_018 | T127 | + per-sym β conformal abstain | +28.93 | +0.77 | conformal 首次实测有效 |
| **iter_019 v2** | **T140** | **M7 LGB 全数据重训（330 iter）** | **+34.44** | **+5.51** | **3rd 突破：GBDT 全数据安全** |
| T170 | T170 | T87 NN M7 全数据重训（11 epoch） | +34.64 | +0.20 | NN M7 增量小 |
| **T188v2** | **T188v2** | **50 NN + 50 LGB simple mean ensemble** | **+35.64** | **+1.00** | **当前 SOTA**：OOD variance reduction 兑现 |

**绝对总增益**：mmpc_demo `−6.65` → T188v2 `+35.64` = **+42.29**。

**关键观察**：
1. 三次大突破 = 15.16 + 8.93 + 5.51 = **+29.60**（占总增益 70%）
2. 其余 12.69 来自 batch-vec 通过限时（让任何提交成为可能）+ conformal + NN M7 + 50+50 ensemble
3. **15→18 个月里只增长了 +1.00**（T170 +34.64 → T188v2 +35.64）→ 已逼近平台 OOD 信噪比上限

---

## §1. 训练方法的关键 tricks

### 1.1 SPO+ Decision-Focused Learning【**第二次大突破核心**】

| 项 | 值 |
|---|---|
| 实验 | T81 (L2 pretrain) → T87 (SPO+ fine-tune) → T170 (M7 重训) |
| 本地 | NN alone LOSO-eq **+35.89** → NN+LGB **+40.09** (+4.20) |
| 平台 | iter_013 +19.23 → iter_015 **+28.16**（**+8.93**） |
| HP | **λ=30, lr=3e-5**（T136 sweep λ∈{1,10,30,100,300} × lr∈{1e-5,3e-5,1e-4,3e-4} 选出） |
| 协议 | Phase 1: L2 pretrain 早停（V4 best_epoch ≈ 9.2）；Phase 2: warm-start → `L = MSE + λ × SPO+` 微调 11 epoch |

**为啥 work**：L2 pretrain 给好初始化（pred 量纲 ≈ Δmid），SPO+ 把 NN 梯度直接对齐 "出手赚钱" 的 PnL 目标。SPO+ 的 surrogate gradient 在决策层（EV gate）上 Fisher-consistent，比裸 L2 优 +4.20。

**为啥两阶段而非纯 SPO+**：直接 PnL loss（T13/T134）**不收敛**——PnL 公式 `z×(Δmid − fee×sign(z))` 在 z∈{0,1,2} 上分段不连续，裸梯度方向不一致。L2 项作为"锚定"防止 SPO+ 把 pred 跑飞。

**Phase 2 联合 loss 的 λ 量级**：L2 是 squared scale（量级 ~1e-7），SPO+ 是线性 scale（量级 ~1e-4），λ=30 让两者贡献同量级、SPO+ 主导。

**替代失败**：
- T13/T134 direct PnL loss → 不收敛
- T116 GMADL direction-aware → LOSO 微弱，不超 SPO+
- R10 调研 10 个 PnL-aware loss → SPO+ 是唯一稳定收敛的

**理论 GAP**（USER-RAISED，详见 §8.4）：SPO+ Fisher consistency 原文针对线性 LP，改成 1D 三选项任务无数学证明；weighted SPO+（class-balanced）凸性也未证明。Empirically work。

---

### 1.2 M7 全数据重训【**第三次大突破核心**】

| 项 | 值 |
|---|---|
| 实验 | R_full_retrain 调研 → T128 LGB 可行性 → **T140 LGB**（**+5.51** 平台） → T170 NN（+0.20 平台） |
| LGB 配置 | date 0-119 全数据，**`num_boost_round=330`** = V4 best_iter ≈ 300 × 1.1，不留 val |
| NN 配置 | date 0-119 全数据，**11 epoch** = V4 best_epoch ≈ 9.2 × 1.1（向上取整），warm-start from T81 |

**为啥 work**：M7 把 80-119 的 30 天新数据补进训练集（+25% 样本），且 96-119 是离平台测试时段最近的 30 天，对 OOD 兑现增益最大。

**为啥 LGB 增量 > NN（+5.51 vs +0.20）**：
- LGB 从头训，M7 新数据直接进树
- NN 已 T81 L2 warm-start + SPO+ DFL，warm-start 的相对增益已饱和（T183 验证：NN random init vs T81 warm-start M7 后 holdout 持平）

**为啥不留 val**：留 val = 浪费 12% 训练样本；96-119 是 OOD 最相近时段，更值得用作训练。T140 LOSO 不变（M7 只改全数据模型，不影响 LOSO eval）但平台 +5.51 直接证明不留 val 优。

**iter 数怎么定**：用 V4 walk-forward 早停的 5 seeds 平均 best_iter × 1.1 安全系数。GBDT 在大数据上单调收敛，量化测试 330 vs 300 iters train loss 差距小。NN 11 epoch 经 T176 sweep（ep∈{7,11,15,20}）确认（S1_ep15 平台 +34.51，−0.13 vs T170）。

**已验证模型才能 M7**：T172/T184/T187 新架构（Transformer/CNN）in-sample 数字漂亮（+22 ~ +55）但**全部未提交**——in-sample 污染严重，OOD 风险高。M7 retrain 是 "已验证模型的 +5" 而非 "未验证模型的捷径"。

---

### 1.3 aug_a 数据增强（bid/ask 互换 + 涨/跌对换）

| 项 | 值 |
|---|---|
| 实验 | T25-T28（5 月） |
| 本地 | LOSO h_60 单 seed +6.30 → **+9.67 (+3.37)**；5-seed → **+11.46** |
| 实现 | NN: `x' = x × U(0.8, 1.2)` per feature；LGB: bid/ask 互换且 label 反转，concat 后训练集 ×2 |

**为啥 work**：LOB 的方向对称性（涨/跌、bid/ask 互换后系统应等价），强制模型学到对称表示。是项目早期最关键的一个 +3.37 单 trick。

**Per-feature 独立缩放的疑点**（CODE Q18）：破坏几何关系（bid1 vs bsize1 不能独立缩放）、对 z-score / quantile 特征做乘法物理意义模糊。但 ablation 证实 work，沿用进所有后续提交。

**替代失败**：
- T14 cross-sym mixup → 负面
- T138 mixup → 下降（混合样本破坏 EV 信号）

---

### 1.4 HP cycling 提供 ensemble 多样性

| 实验 | 配置 | 平台 |
|---|---|---|
| T179 40-NN（8 HP × 5 seeds, 只扩 NN） | 平台 +34.39 | **−0.25 vs T170** |
| **T188v2 50+50** | 5 NN HP × 10 seeds + 5 LGB HP × 10 seeds | **+35.64 (+1.00)** |
| T190 150+150 | 30 HP × 5 seeds（NN + LGB 都扩到 150） | +34.59 **(−1.05)** |

**关键洞察**：
1. **多样性来自 HP 差异**（R_multinn_spo 结论），不是 seed 多样性；T97 vs T87 同架构 corr ≈ 1.0 → 集成无增量（T163 平台 **−3.70** 教训）
2. **NN+LGB 必须两侧对称扩**：只扩一侧（T179）无 variance reduction
3. **扩容非单调**：5→50 +1.00（有效），50→150 −1.05（HP 多样性边际后变噪音）

**T188v2 NN HP grid**：LR ∈ {1e-4, 3e-4, 1e-3} × Dropout ∈ {0.05, 0.10, 0.15, 0.20} × Batch ∈ {2048, 4096, 8192}，但 idx 用 `(S-1)%3 / (S-1)%4 / (S-1)%3` → lcm(3,4,3) = **12 unique combos**（不是 README 写的 36），每 combo 重复 ~4.17 次。是 bug 还是设计：作者未明确，但 12 + 重复 RNG 多样性事实上 work。**GAP**：README/code drift。

**T188v2 LGB HP grid**（手工 5 组）：

| # | feat_frac | bag_frac | num_leaves | λ_L2 |
|---|---|---|---|---|
| 1 (baseline) | 0.8 | 0.8 | 127 | 1.0 |
| 2 | 0.6 | 0.7 | 127 | 1.0 |
| 3 | 0.7 | 0.85 | **63** | 2.0 |
| 4 | 0.5 | 0.6 | **255** | 0.5 |
| 5 | 0.4 | 0.5 | 127 | **3.0** |

每组 10 seeds，sweet spot 127 leaves 不动，扰动主要在 feature/bagging fraction。

---

### 1.5 不学 ensemble 权重 — simple mean is best（T192 大教训）

| 实验 | 方法 | 本地 holdout | 平台 |
|---|---|---|---|
| T139 stacker NN | NN 学综合 LGB+NN | LOSO 负面 | 未提交（OOF 过拟合）|
| **T192 H4 bounded NNLS** | NNLS [1/1500, 2/150] 学 150+150 权重 | **val +180.41, test +193.94** (ratio 1.075) | **+31.0** (**−4.64 vs SOTA**) |

**为啥 NNLS / OLS 失败**（NN-NN avg corr **0.9331**, LGB-LGB **0.8433**, design matrix condition number **54,182**）：极病态共线性使权重不可识别，任何 "最优" 权重都是 in-sample 噪音对齐。

**Simple mean 反而是病态共线性下的最优鲁棒选择**。

**w_lgb = 1.5 / w_nn = 1.0 来源**：iter_015 阶段 grid search，50+50 阶段没复 sweep；轻微 GAP 但实测稳定。

---

## §2. 决策层的关键 tricks

### 2.0 Execution Strategy 完整决策链（pred → action 的全 pipeline）

<!-- 补充 by worker 2026-05-24: Execution Strategy 完整 pseudocode + 设计理由 -->

> **本节定位**：把整个"出手/不出手"的决策 pipeline 拆解到 step 级，对齐 `Predictor.py:316-365` 实际代码。§2.1-§2.4 是这 6 个 step 的逐项细化；§9 是更高 level 的数据流速查（不含 step 级伪代码）。

#### 决策链伪代码（与 Predictor.py 行号一一对应）

```python
# Predictor.predict(batches: List[pd.DataFrame]) → List[List[int]]  shape (B, 5)
# 每个 batches[i] 是一个 100-tick × N_features 的 window；输出 5 个 horizon 的 action ∈ {0,1,2}
# action 0 = short / 1 = flat / 2 = long（平台 PnL 公式约定）

# ─── STEP 1: 特征构造（一次性，全 horizon 共享） ────────────────────────
feats = self._compute_batch_features(batches)            # (B, 359) float32
#  ├─ raw_last = X3d[:, -1, :]                            # 154 维窗口最后一 tick
#  │  └─ raw_last[amount_delta] = sign(v) * log1p(|v|)   # 仅此一列 log1p 压缩
#  ├─ extras_full = fast_features_batch.compute_batch_features(X3d, col_idx)  # 216 维派生
#  ├─ extras_full[~isfinite] = 0.0                        # NaN/Inf → 0（不报错）
#  └─ concat([raw_last, extras_full[:, keep_idx]])        # 154 + 205 = 359（drop 11 FAIL）

# ─── STEP 2: per-row conformal band 查表（只当 cw_enabled=True） ────────
if self._cw_enabled:                                      # thresholds.json 里 = true
    band_per_row = np.full(B, self._cw_default_band, dtype=np.float64)
    # default_band = 0.16 × 4.00e-4 = 6.40e-5（OOD 兜底）
    for i, df in enumerate(batches):
        if "sym" not in df.columns:    continue           # 缺列 → 用 default band
        try:
            s = int(df["sym"].iloc[-1])                   # 取 window 最后一 tick 的 sym
        except (ValueError, TypeError, IndexError):
            continue                                       # NaN/非数值 → default band
        if s in self._cw_band:                            # s ∈ {0,1,2,3,4} 才查表
            band_per_row[i] = self._cw_band[s]            # = β_sym × σ_sym（常量）
        # else: s = 99/OOD → 保留 default band
else:
    band_per_row = np.zeros(B)                            # disable 模式回退到 _ev_gate_predict

# ─── STEP 3: 主循环（per-horizon，复用 share_with 缓存） ───────────────
out = np.ones((B, 5), dtype=np.int64)                     # 全 flat 初始化
pred_cache: Dict[int, np.ndarray] = {}                    # **predict() 局部变量**，无跨调用 state

for hcfg in self._horizons:                               # 5 个 horizon：[5,10,20,40,60]
    if not hcfg.get("active", True):  continue
    H     = int(hcfg["h"])
    src_H = int(hcfg.get("share_with", H))                # h=5/10/20/40 都 share_with=60；h=60 自己

    # ── STEP 4: ensemble 预测（命中缓存就跳过） ────────────────────────
    if src_H in pred_cache:
        pred_dmid = pred_cache[src_H]                     # 5 horizon 实际只 forward 1 次
    else:
        lgbs   = self._lgb_lists.get(src_H)               # 50 个 lgb.Booster（h=60 唯一）
        nn_ens = self._nn_batched.get(src_H)              # 50 NN batched bmm
        w_nn, w_lgb = self._weights.get(src_H, (1.0,1.0)) # (1.0, 1.5) for h=60

        preds, ws = [], []
        if lgbs and w_lgb > 0:                            # 50 LGB CPU sequential mean
            preds.append(mean([b.predict(feats) for b in lgbs]))
            ws.append(w_lgb)                              # 1.5
        if nn_ens is not None and w_nn > 0:               # 50 NN GPU/CPU bmm mean
            preds.append(nn_ens.predict_mean(feats))      # 内部 NaN → 0, clamp(-clip,clip)
            ws.append(w_nn)                               # 1.0
        if not preds:                                     # 配错 share_with → silent flat
            continue
        pred_dmid = (1.0 * pred_nn + 1.5 * pred_lgb) / 2.5    # 等价写法
        pred_cache[src_H] = pred_dmid                     # 缓存复用

    # ── STEP 5: EV gate 决策（asymmetric thr + band） ─────────────────
    thr_up = hcfg["thr_up"]                               # h=60: 3.00e-4；h=5: 8.66e-5（sqrt 缩）
    thr_dn = hcfg["thr_dn"]                               # h=60: 2.16e-4；h=5: 6.24e-5
    actions = np.full(B, 1, dtype=np.int64)               # 默认 flat
    actions[pred_dmid >  (thr_up + band_per_row)] = 2     # long（严格 >，无 hysteresis）
    actions[pred_dmid < -(thr_dn + band_per_row)] = 0     # short（严格 <）

    out[:, HORIZON_TO_IDX[H]] = actions                   # 写到对应 horizon 列

return out.tolist()                                       # List[List[int]] shape (B, 5)
```

#### Step 级注解

| Step | 出处 | 设计动机 | 关键固化年代/T 编号 |
|---|---|---|---|
| **1. 特征构造** | `_compute_batch_features` (Predictor.py:273-289) | 一次性算 359 维，全 horizon 共享。`np.where(~isfinite, 0)` 兜底 NaN/Inf；`amount_delta` 仅 raw_last 列 log1p 压重尾（X3d 下游特征仍用 raw，见 §3.3 已知不一致） | T44-T68 SchemeP 完成（5 stage 累计 +205 维）；T59 drop 11 FAIL_NAMES |
| **2. Conformal band 查表** | `_extract_band_per_row` (Predictor.py:291-304) | 查表 = 零开销 inference（`self._cw_band` 是 __init__ 时算好的 dict `{sym: β×σ}`）。**sym 用 window 最后一 tick** 是为了避免跨 tick 的 sym 漂移歧义（实际同 batch 内 sym 不变）。OOD 兜底 = `default_β × default_σ = 0.16 × 4.00e-4 = 6.40e-5`（**比 sym=1 的 1.88e-4 还宽松，已知 gap** — 见 §8 与 ANSWERS_P2 §3.3） | T127 conformal 首次启用（iter_018 +0.77 平台）；T150 β sweep 固化 {0.10, 0.40, 0.30, 0, 0} |
| **3. Per-horizon 主循环 + 缓存** | Predictor.py:331-358 | `pred_cache` 是 **`predict()` 局部变量**（每次调用 fresh dict），所以跨调用 state = 0，满足硬约束 #2。`share_with=60` 让 h=5/10/20/40 复用 h=60 的 ensemble 输出，**5 horizon 实际只 forward 1 次**（cost = 1×，不是 5×） | T188v3_fullhorizon |
| **4. Ensemble 加权** | Predictor.py:347-357 | `(1.0 × pred_nn + 1.5 × pred_lgb) / 2.5` — **不是 stacking / NNLS**。w_lgb=1.5 源自 iter_015 阶段 grid search，50+50 没复 sweep（轻 GAP，T192 NNLS 证 OLS 学权重必崩 −4.64） | T188v2 固化 50+50 |
| **5. EV gate 加 band** | `_gate_with_band` (Predictor.py:255-262) | 决策语义：`pred > thr_up + band → long`、`pred < -(thr_dn + band) → short`、else flat。**这不是 argmax 分类**，是 EV 阈值 = "期望毛收益 > 3× fee + brittle sym 加宽"。严格 `>` `<`（非 `>=`）与训练 label 边界对齐，浮点等于阈值概率 ≈ 0 | T75 EV gate 引入（iter_013 +15.16 平台）；T127 band 叠加（iter_018 +0.77） |
| **6. 输出 List[List[int]]** | Predictor.py:365 | 形状 (B, 5) 对应 (n_samples, 5 horizons)；HORIZON_LIST 顺序固定 `(5, 10, 20, 40, 60)` | — |

#### Corner cases — Predictor.py 实测覆盖（来自 `__main__` smoke test，行 368-388）

| 输入异常 | 触发位置 | 行为 | 来源代码 |
|---|---|---|---|
| `df["sym"]` 缺列 | `_extract_band_per_row:296` | `band = default_band`（不 raise） | `if "sym" not in df.columns: continue` |
| `df["sym"].iloc[-1] = "abc"` 或 `NaN` | `_extract_band_per_row:299-300` | `band = default_band`（catch ValueError/TypeError/IndexError） | `try: int(...); except ...: continue` |
| `sym = 99`（OOD ID） | `_extract_band_per_row:302` | `band = default_band`（s 不在 _cw_band 字典内） | `if s in self._cw_band: out[i] = ...` else 不写入 |
| `feats[i,j] = NaN/Inf` | `_compute_batch_features:286` + `_BatchedMLPEnsemble.predict_mean:159` | 先 `np.where(~isfinite,0)`；NN 再 `torch.where(isnan,0)` + `clamp(-clip,clip)` | 双重 NaN 防御 |
| `pred_dmid = NaN`（极小概率，feats 全 NaN 时） | _gate_with_band:259-261 | `NaN > x = False`，`NaN < -x = False` → 默认 flat | numpy NaN 比较语义 |
| `share_with` 指向不存在 horizon | Predictor.py:353-354 | `if not preds: continue` → 该 horizon silent flat | preds 空列表跳过 |
| `batches = []` | Predictor.py:317-318 | `return []`（短路） | early return |

**结论**：所有"输入异常"路径都**不 raise**，且默认 flat（action=1，安全无交易）。这是项目"宁可错过不可错杀"哲学的代码层兑现。

#### 决策链每一步对应的"为啥这么设计"速查

1. **为啥 ensemble pred 是 weighted mean 而非 stacking?** — T139 stacker NN OOF 负面；T192 bounded NNLS val/test 1.075 但平台 −4.64（OLS 学到 in-sample 共同 noise）。在 NN-NN corr 0.9331 / LGB-LGB corr 0.8433 / condition number 54182 的病态共线性下，simple mean 是数学上的最优鲁棒选择。见 §1.5 / §7.5。
2. **为啥 thr_up=3e-4 / thr_dn=2.16e-4 而不是其他数?** — 详见 §2.1.1（新增子节）。
3. **为啥 band = β·σ 而不是动态计算?** — 硬约束 #2（test 顺序打乱）禁止任何 per-batch 统计。`self._cw_band` 是 __init__ 时算好的常量。任何动态 band（T159 / T168 / T180 / T181）平台全部 kill。
4. **为啥 share_with=60 而不是每个 horizon 独训模型?** — iter_002 短 horizon 平台灾难（h_5 −7.68, h_10 −8.64）。α=0.05%（h_5/10）< fee 0.02% × 2，结构性输；α=0.1%（h_60）才有 alpha 空间。score=max(per-horizon) 让短 horizon 只要不严重负就不掉 rank。见 §2.4。
5. **为啥 sqrt(h/60) 缩 thr 而不是 h/60 线性?** — **数学上方向反了 12×**（USER-RAISED Q281, 见 §8.1）；项目能 work 因为 (a) max 规则隐藏短 h；(b) hi-band sym 上 conformal band 反过来抵消缩放。**真 GAP，没有 thr-scaling ablation**。
6. **为啥 conformal default_β = 0.16 而非 max(β)=0.40?** — 是 mean(0.10, 0.40, 0.30, 0, 0) = 0.16，**无原理依据**（应该 max 更保守）。OOD 反而比 sym=1 还激进交易。**已知 gap**，见 ANSWERS_P2 §3.3。

#### 决策链涉及的"常量"汇总表（生产路径硬编码）

| 常量 | 数值 | 来源（哪个 T 实验定下来的） | 自此以后是否动过 |
|---|---|---|---|
| `thr_up` (h=60) | **0.0003** | T75/T76 DE 4D + decoupled（iter_013, 2026-07-25） | **从未动**（iter_018/019/T170/T188v2 都不调） |
| `thr_dn` (h=60) | **0.000216** | T76 decoupled DE | 同上 |
| `thr_up/dn` (h=5/10/20/40) | × sqrt(h/60) | T188v3_fullhorizon | T188v3 加入 5-horizon emit |
| `w_nn` / `w_lgb` | 1.0 / 1.5 | iter_015 (T87) grid search | **从未复 sweep** |
| `β_sym` | {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00} | T150 4-fold CV consensus sweep | **从未动** |
| `σ_sym` (used in band) | {0:2.40e-4, 1:4.71e-4, 2:4.52e-4, 3:4.24e-4, 4:4.28e-4} | T127/T150 5-NN ensemble disagreement std on V4 val (date 80-95) | **5→50 NN 没重算**（已知 gap，保守） |
| `default_β_for_ood` | **0.16** | mean(per_sym_β)（无原理依据，应 max） | 未动（潜在 gap） |
| `default_σ_for_ood` | **0.0003998317** | mean(per_sym_σ) | 同上 |
| `clip` | max(per-NN clip) | T81 训练时设定 | per-NN 独立，inference 取 max |
| `HORIZON_LIST` | (5, 10, 20, 40, 60) | 平台 spec | 不可变 |

**该表的用处**：任何动这 11 个常量都必须经过 §6.4 的"OOD 验证 + 22 次提交保守原则"复审；iter_016 v3 / T146 / T180 / T168 都因为想动常量被 kill。

#### 决策链 vs 训练链的 boundary（容易混淆）

- **训练时**：决策层不参与梯度（除 SPO+ 的 DFL surrogate gradient 是例外，T87）。NN 用 L2 + SPO+ 训，LGB 用 L2 regression 训，**两边都不知道 thr_up/thr_dn 长什么样**。
- **DE 搜阈值时**：在 trained 模型的 OOF pred 上 grid/DE，**模型本身不重训**。这是 thr 容易 OOF 过拟合的根源。
- **Predictor 推理时**：模型 + 决策层都"冻结"。`Predictor.predict()` 内 0 训练态、0 梯度、0 跨调用 state。

---

### 2.1 Δmid 回归 + 非对称 EV gate【**第一次大突破核心**】

| 项 | 值 |
|---|---|
| 实验 | T75（iter_013，整个比赛最大单次跃升） |
| 本地 | LOSO-eq +26.44 → **+36.23 (+9.79, +37%)** |
| 平台 | iter_002 +4.07 → iter_013 **+19.23 (+15.16)** |
| y 公式 | `y = (mp_{t+60} − mp_t) / (mp_t + 1)`（+1 是除零防御）|
| 阈值 | `thr_up = 0.0003, thr_dn = 0.000216`（非对称，DE 搜出后固化）|

**为啥 work**：3-class CE 把 Δmid 离散化为涨/平/跌 3 bucket，**损失了幅度信息**；EV gate 直接对应期望收益。把 LGB 从 3-class CE 换成 L2 Δmid regression 是整个比赛最大单次 alpha 跃升。

**为啥 thr_up ≠ thr_dn**（T76 decoupled threshold）：交易 fee 双向，但 LOB 数据在评测窗口内有微观结构 drift；非对称阈值匹配实际收益分布。

**为啥不再做 DE 阈值搜索**（5 重证据）：
1. iter_013 DE 透传 LOSO +36.23 → 平台 +19.23（**透传 53%**，DE 过拟合）
2. iter_016 v3 4-way + OOF DE 4D：LOSO **+43.71** → 平台 **+25.08**（**−3.08**）灾难
3. T126 per-sym OOF threshold → LOSO 高但**未提交**（已知过拟合）
4. T146 conformal-aware re-DE → LOSO **+43.96** 但 PM 决定**未提交**（教训沉淀够多）
5. T62 对称 k×mean_abs_pred → LOSO **+33.72**，鲁棒但比 DE 低 2.51

**最终选择**：thr_up/thr_dn 一旦定下来，所有后续大版本（iter_018/019/T170/T188v2/T190）**都不动**这两个值。

**没用 isotonic / Platt 校准**（T15/T131/T40/R13）：早期试过 isotonic + EV gate，LOSO **−1.02**（破坏排序质量）。回归 pred 直接当 EV，无 "概率" 概念，不需校准。

---

### 2.1.1 Threshold 求法历史（0.0003 / 0.000216 是怎么得来的）

<!-- 补充 by worker 2026-05-24: thr_up/thr_dn 的演变时间线、固化原因、过拟合教训 -->

> **本节定位**：补强 §2.1，专门讲 thr_up=0.0003 / thr_dn=0.000216 这两个数字**具体怎么演化出来 + 为什么从此再没动过**。证据全部来自 `HISTORY_EVIDENCE_DB.md` 主题 1 / `ANSWERS_P2_DECISION.md` §1.1-1.9 / `0512_tricks_table.md` T 实验行。

#### 演变时间线（10 个里程碑）

| 阶段 | T-编号 | 方法 | 数据集 | 目标函数 | 算法 | 本地 | 平台 | 命运 |
|---|---|---|---|---|---|---|---|---|
| ① 早期阈值 | T5a | SchemeB 上非对称 thr_up/thr_dn DE | LOSO 5-fold | h_60 PnL（3-class CE 上） | DE | LOSO h_60 +11.11（vs T2 −22.10）| — | 验证 DE 优于无 thr |
| ② DE 4D 试水 | T30 | DE 4D（thr_up/thr_dn/margin/weight）on aug_a 5-seed | LOSO 5-fold | h_60 PnL | DE | +11.46 → **+13.61** (+2.15) | — | 但 T30 已经 warning：**DE 在 5-fold LOSO 上可超调** |
| ③ 无偏对称替代 | T62 | k×mean_abs_pred 对称 thr | LOSO-eq 442k | h_60 PnL | grid k∈[0.5, 2.5] | k=1.25, **+33.72** | — | 鲁棒但**绝对低于 DE 2.51**；DE 仍是首选 |
| ④ 参数扫描确认 | T63 | regression EV gate 变体（absolute/asym）| LOSO-eq | h_60 PnL | grid | 边际 | — | 确认 asymmetric DE 是最优 form |
| ⑤ **DE 首次大规模** | **T75** | LGB Δmid regression + 4D DE | LOSO-eq 442k | h_60 PnL（regression pred 上） | DE | **+36.23** (+9.79, +37%) | **+19.23** (iter_013) | **第一次大突破**；但透传 53% 已经吹哨 |
| ⑥ **非对称引入** | **T76** | decoupled DE（thr_up / thr_dn 分开搜，**不再固定 ratio**） | LOSO-eq | h_60 PnL | DE per-dim | 边际正 | — | **DE 收敛到 thr_up≈3e-4, thr_dn≈2.16e-4**；PM 手工取整 |
| ⑦ per-sym OOF 探索 | T126 | 每只 sym 独立 4D OOF DE | per-sym OOF（~94k/sym）| per-sym h_60 PnL | DE | LOSO **高**（具体未记录）| **未提交** | 过拟合风险高 + OOD sym 无 model |
| ⑧ **DE 4D 灾难** | **iter_016 v3** | 4-way ensemble (T87+T89+T99+T95) 再跑 OOF DE 4D | OOF (V4 test, 442k) | h_60 PnL | DE | LOSO **+43.71** (+3.62) | **+25.08 (−3.08)** | **最经典 DE OOF 过拟合教训** |
| ⑨ conformal 替代 | T127 | 不再动 thr，转 per-sym β conformal 弹性带 | LOSO-eq + 4-fold | sweep β∈{0, 0.1, ..., 0.5} | grid | LOSO +1.75 | **+0.77** (iter_018) | **从此 thr 冻结，dynamic 部分交给 conformal** |
| ⑩ 再试 DE 失败 | T146 | v3 mae + conformal-aware DE 重搜 | OOF | h_60 PnL | DE | +2.76 LOSO | **未提交** | 教训沉淀够多，**PM 主动不冒险** |
| ⑪ adaptive thr 失败 | T168 V1-V4 | per-batch 自适应 thr（quantile/mean±k·std/running median）| holdout | h_60 PnL | runtime | V1-V4 **全负** | **未提交** | 触碰硬约束 #2，per-batch 统计在 shuffle 下崩 |

**iter_013 透传 53% 是分水岭**：DE 在 LOSO-eq 442k 上拿 +36.23，平台只拿 +19.23，**gap −17 ≈ 47% 失血**。从此整个项目对"DE 在更大 OOF 上必更过拟合"建立了硬直觉。

#### 最终值 0.0003 / 0.000216 的来源——明示血统

- **来源实验**：T75 (iter_013, 2026-07-25) DE 4D 搜索 → T76 decoupled 后产物。**这两个数字本质上是 iter_013 的"遗物"**，从 iter_013 (2026-07-25) 到 T188v2 (2026-09-11) 跨 7 周、22 次平台提交、200+ T 实验**完全没改过**。
- **整数化**：DE 收敛点 ≈ (3.0e-4, 2.16e-4)，PM 手工取整到 `0.0003` (恰好 30bps ≈ 3× fee 0.01%)；`0.000216` 不是手设的 0.0002 也不是 0.00022，是 DE per-dim 自由优化的浮点结果。
- **决策准则**：thr_up = 30bps，确保毛预期收益 ≥ 3 × 单边 fee（0.01%）= 3× cost，留余量给方差。

#### 非对称比例 1.3889 = thr_up / thr_dn — 为什么 down 信号门槛更低 28%

| 解释 | 项目内是否采纳 | 数学/经济根据 |
|---|---|---|
| **A. 模型本身有正预测偏差** | **是**（项目主采纳） | 训练目标 y=(mp_{t+60}−mp_t)/(mp_t+1) 在 5 只训练股票上**偏正**（涨多于跌的样本，整体牛市）；NN/LGB 学到 E[pred]>0；如果对称 thr，sell 信号被无谓抑制 → thr_dn 调低让卖信号通过 |
| **B. short-fill 比 long-fill 含金量高** | 否（项目里没引用） | 微观结构上 short 流动性更稀 → short 信号 base rate 低但单笔贡献大；理论合理但项目未引用此论据 |
| **C. DE 收敛产物事后合理化** | 部分（🟡 ANSWERS_P2 §1.2 标注） | 1.3889 是 T76 decoupled DE 独立搜 thr_up / thr_dn 的副产物，**不是从假设 A 推导的，而是搜出来后用 A 解释** |

**横向 invariant**：所有 horizon 都用 1.3889（因为 sqrt(h/60) 缩 thr_up 和 thr_dn 是同一个常数因子，比例代数保留）。**不是独立选择，是 share_with + sqrt 的代数后果**。

#### 为什么从此再没动 — 5 重证据链

1. **iter_013 透传 53%**（最早警告）：DE 拿 +36.23，平台 +19.23，过拟合 −47%
2. **iter_016 v3 平台 −3.08**（致命教训）：4-way + OOF DE 4D，LOSO 看似 +43.71 但平台 +25.08
3. **T126 per-sym OOF threshold 未提交**（自动 abandon）：5 只 sym 各 ~94k OOF → 过拟合更严重 + OOD sym 无 model
4. **T146 conformal-aware re-DE 未提交**（PM 主动 abort）：本地 +2.76 但教训沉淀够多
5. **T168 adaptive thr V1-V4 全负**（硬约束 kill）：per-batch 统计在 shuffle 下方差爆炸
6.（额外）**T192 NNLS 平台 −4.64**：任何在 in-sample 上学习的决策参数（thr or ensemble weight）都是同一类风险

→ 项目末期形成"**保守 thr，激进 retrain，渐进 ensemble**"三原则（§10 第 7 条）；thr 一旦定下不再动。

#### conformal 接管 dynamic 部分 — T127 之后的分工

- **静态部分（thr_up/thr_dn）**：T76 后冻结，全局固定
- **动态部分（按 sym 加宽弹性带）**：T127 起由 `β_sym × σ_sym` 接管，per-sym 4-fold CV consensus，**不在 OOF 上学**
- **"dynamic part 不在 OOF 上学" 的具体落地**：T150 β sweep 用 4-fold CV consensus（取多数票），不是单 fold OOF；σ 用 V4 val 段（80-95）算 5-NN ensemble disagreement，不用 OOF residual。这两层防御让 conformal 没成为"另一个 OOF DE 过拟合"

#### DE 搜索具体配置（T75/T76 阶段）

| 维度 | 值 |
|---|---|
| 算法 | `scipy.optimize.differential_evolution`（默认 best/1/bin） |
| 搜索空间（T75 4D）| thr_up ∈ [1e-5, 1e-3], thr_dn ∈ [1e-5, 1e-3], margin ∈ [0, 5e-4], weight ∈ [0.5, 2.0] |
| 搜索空间（T76 decoupled）| 同上但**强制 thr_up ≠ thr_dn 独立**，去掉 margin/weight（简化为 2D） |
| 目标函数 | LOSO-equiv 442k 本地测试集累计 h_60 PnL（按 EV gate 后） |
| 收敛点（T76）| thr_up ≈ 3.00e-4, thr_dn ≈ 2.16e-4 |
| 种子 | 5 个 random seed 平均（早期 stability check） |
| 风险防御 | **无显式防御**——这正是后续 iter_016 v3 灾难的根因 |

#### 没做的 ablation —— 真实 GAP 标注

| 缺失实验 | 严重性 | 后果 | 是否容易补 |
|---|---|---|---|
| thr_up ±10% 平台敏感度（0.00027 / 0.00033）| 🔴 真 GAP | 不知道 0.0003 在平台 OOD 上是 "正好" 还是 "凑巧不错" | 易（改 thresholds.json + 提交 2 次） |
| thr_dn ±10% 平台敏感度（0.000194 / 0.000238）| 🔴 真 GAP | 同上 | 易 |
| ratio = 1.0（对称对照）vs 1.3889 | 🔴 真 GAP | 验证"模型正偏差导致非对称"假设 A 的真伪 | 易 |
| ratio = 0.72 反向（thr_up < thr_dn）| 🟡 | 若仍 work，说明 ratio 不是来自模型偏差 | 易 |
| 5 个 horizon 用各自独立 DE 而非 sqrt 缩 | 🔴（叠加 Q281）| 解决 sqrt 方向反 12× 的 USER-RAISED gap | 中（需要为短 h 重训或共用 pred + 重 DE） |
| M7 之后的 bias 是否变化 → 是否需要重 DE | 🟡 | 已知 gap，但 M7 +5.51 已成 SOTA，不冒险 | 中 |

**这些 ablation 全部没做的原因**：每次平台提交 12h 限 1 次 + 22 次提交配额已耗 + iter_016 v3 教训太深 → PM "已 SOTA 不冒险" 原则压倒一切。

#### 最浓缩的一句话回答 Q2

**"thr_up=0.0003 / thr_dn=0.000216 是 T76 decoupled DE (iter_013 阶段, 2026-07-25) 在 LOSO-equiv 442k 本地集上搜出来的；从此跨 22 次平台提交完全没动；不动的根本原因是 iter_013 透传 53% + iter_016 v3 平台 −3.08 两个 OOF DE 过拟合教训，导致 PM 把任何'再搜阈值'都判为高风险；之后所有 sym/horizon 维度的 dynamic 调整由 T127 conformal `β_sym × σ_sym` 接管，β 用 4-fold CV consensus 而非单 fold OOF，σ 用 V4 val 段算的 5-NN disagreement std（5→50 没重算，已知保守 gap）。"**

---

### 2.2 Per-sym β Conformal Abstain Wrapper

| 项 | 值 |
|---|---|
| 实验 | T127 (iter_018) → T150 β sweep → T182 ablation 验证 |
| 平台 | iter_015 +28.16 → iter_018 **+28.93 (+0.77)** |
| β 值（per-sym, 4-fold CV consensus）| **{sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}** |
| 规则 | `|EV_pred| < β_sym × σ_sym → action=1`（不出手） |
| σ 来源 | per-sym validation residual std，**训练时算好写进 thresholds.json** 常量，inference 时只 load 不重算 |

**为啥 sym1/sym2 abstain 重**：sym2 是蓝筹/ETF（T8 诊断：amount_delta z-score **+7.49** 偏离），预测在其上 brittle，β=0.30 弃权过滤错预测。

**为啥 sym3/4 不弃权**（β=0）：模型在这两只上预测最稳，conformal 弃权反损 PnL。

**关键 ablation T182**（in-sample 污染的经典案例）：去 conformal 后 holdout 本地 **+6.62**（看起来 conformal 是负担），平台 **−0.30**（conformal 实际有效）。**证明本地 holdout 和平台 OOD 可以反向**，conformal 在 OOD 上真有用。

**OOD sym 处理**：当前 β 按 sym ID 索引；T149 NN OOD sweep 确认 T87 NN 在 OOD sym 下表现可接受。降级方案（未实现，备选）：全局 β=0.20 兜底。

**替代失败**：
- T126 per-sym OOF threshold → 过拟合，**未提交**
- T159 per-sym dynamic abstain → null
- T180 OOD GMM density abstain → V1 **−32**, V2 −57, V3 **−78**（测试集比训练更 in-distribution，弃权反损）
- T181 time-window hard abstain → V1 **−23.39**（T170 conformal 已过滤，硬弃权适得其反）

---

### 2.3 Ensemble 组成：50 NN + 50 LGB simple mean

详见 §1.4 / §1.5（HP cycling + 不学权重）。

**最终配置**：
- 50 NN（5 HP × 10 seed，HP cycling 实际 12 unique combos）
- 50 LGB（5 HP × 10 seed）
- 每族 simple mean → 跨族 `w_nn = 1.0, w_lgb = 1.5`
- 外层 per-sym conformal abstain
- **不加** CatBoost / GRU / Transformer（详见 §7 失败章节）

**Family-mean 后再加权 vs model-level 混权**：等价（每个 NN 权重 = w_nn/50/(w_nn+w_lgb)）。**不是 stacking / ridge**。

---

### 2.4 5-horizon share_with=60 + sqrt 缩放

| 项 | 值 |
|---|---|
| 配置 | h ∈ {5, 10, 20, 40, 60}；h=5/10/20/40 共享 h=60 ensemble 预测 |
| thr 缩放 | `thr_h = thr_60 × sqrt(h/60)`（h<60 时 thr 变小） |
| 推理 cost | 1× h=60 ensemble 推理（不是 5×） |

**为啥所有 horizon 共享 h=60 预测**：
1. **短 horizon 平台灾难**（iter_002 关键发现）：

   | h | LOSO 本地 | 平台 |
   |---|---|---|
   | 5 | +17.49 | **−7.68** |
   | 10 | **+21.86** | **−8.64** |
   | 20 | +19.70 | −5.09 |
   | 40 | +11.71 | +2.02 |
   | 60 | +6.30 | **+4.07** |

   → LOSO 短 h 看似最强，平台短 h 全负。α=0.05%（h_5/10）被手续费 0.02% 双边摧毁；α=0.1%（h_60）才有 alpha 空间。

2. **平台规则**：score = max(per-horizon PnL) → 短 horizon 提交不会拉低 max。
3. **整个项目专注 h_60**，所有特征/调参/ensemble 都围绕 h_60。

**⚠️ USER-RAISED Q281：sqrt(h/60) 方向数学上反了**（详见 §8.1）—— h=5 时差 **12 倍**。能 "work" 是因为 max 规则隐藏了短 h 的问题。

---

## §3. 特征工程的关键 tricks

### 3.1 SchemeP 359 维 = 154 raw + 216 derived − 11 drop

**演变**：T2 SchemeB 154 维（LOSO sum −22.10）→ T3 SchemeC 226 维（+72 维 MLOFI/WMP/RV/EWMA/time，LOSO h_10 +21.86）→ R34 Stage 1-5 累计 +205 维 → **SchemeP 370 维 → drop 11 → 359 维进模型**。

**R34 Stage 1-5 累计增益**（**特征工程比模型变体加起来贡献更多**）：

| Stage | 增量维 | 主要内容 | 本地增益 |
|---|---|---|---|
| Stage 1 (T44) | +54 维 | dual z-score W=20/50, signed RV, Kyle λ rolling, EWMA-OFI multi-α, cancel pressure | LOSO 5-seed **+13.67** (+4.0 vs +9.67) |
| Stage 2 (T51) | +59 维 | window quantile rank W=20/50/100, signed skewness, GOFI ~30 维, vol-burst, Kyle λ | LOSO 5-seed **+15.06** (+1.4) |
| Stage 3 (T53) | +14 维 | EWMA residual, multi-W RV ratio, Bipower variation, Cancel imbalance, Roll spread | LOSO **+16.84** (+1.78) |
| Stage 4 (T55) | edge gain | 少量 edge features | 合并进 SchemeP |
| Stage 5 (T68) | +20 维 | adaptive momentum, OFI toxicity, signed bipower, spread regime, trade persistence, liq asymmetry | LOSO-eq +26.44 (+0.50, 含 +7 方法论 inflation) |

**为啥保留全 359 维**（不剪枝）：
- T147 feature pruning 按 importance 剪枝 → LOSO 轻微正
- T152 adversarial-filtered 259 维 → LOSO 持平，**平台无确认提升**
- 保留全集更稳；feature pruning 平台无收益

**Drop 11（FAIL_NAMES）**：T59_FAIL（10 个）+ STAGE5_FAIL（1 个）。例如 `dualz_ask_diff1`、`qrank_W100_spread1/5/10/cumspread`、`kyle_lam_W50/W100`、`liq_asym_top5_W5`。Drop 标准是 KS test（train vs test distribution shift），具体阈值文档缺失（**GAP**）。注意：**这些特征仍在 fast_features_batch 里计算**（浪费 compute，未优化）。

---

### 3.2 窗口 z-score 归一化（**修复 sym=2 brittleness**）

| 实验 | 方法 | LOSO 本地 |
|---|---|---|
| **T7** | **窗口内 z-score（100-tick）= SchemeD1 308 维** | **+13.91**（sym=2: −0.98 → +2.24）|
| T10 | SchemeF 全局 z-score | +13.40（**不如 T7**）|

**为啥 work**：窗口 z-score 在每个 100-tick 切片内自适应 → 对 sym distribution shift 鲁棒，**满足硬约束 #3 sym-agnostic**。全局 z-score 用整训练集 mean/std → 对 OOD sym 不鲁棒。

**T8 sym=2 brittleness 诊断**：amount_delta z-score **+7.49** 偏离 → 触发 SchemeE 中的 `amount_delta` log1p 处理（详见 3.3）。

---

### 3.3 amount_delta sign-preserving log1p（局部）

`build_schemeP_cache.py:108`：`raw_last[amt_idx] = sign(v) * log1p(|v|)` 压缩重尾。

**疑点**（CODE Q47-Q48，**无 ablation**）：raw_last 列做 log1p，但 X3d 下游特征（kyle_inv, signed_dvol, vol_burst 等）用**原始 amount_delta**——同一底层量在不同特征里两种尺度表示。reviewers 推测下游特征内置 cbrt/std/mean ratio 归一化，所以 raw amt OK；但**没消融对比"全部 log1p" vs "全部 raw"**。设计/bug 未明，empirically work。

---

### 3.4 时间特征/sym 特征**禁止**（硬约束）

| 方向 | 实验 | 结果 |
|---|---|---|
| `date` 字段 derived 时段 | 隐式禁止 | 违反硬约束 #1（date 评测时置 0） |
| sym embedding / per-sym norm | 隐式禁止 | 违反硬约束 #3 |
| Alpha101/Alpha191 跨截面 | T22 IC **+21.80** | 违反硬约束 #2（batch shuffle），**未提交** |
| sin/cos 时间编码 (SchemeQ +6 维) | T74/T78 | 回归框架下 LOSO-eq **−0.72**，**未入 final** |
| Time-of-Day (TOD) | T151 v9 | LOSO 小幅正，T170 final **未用** |

**经验**：在 LOB regression + sym-agnostic 约束下，**任何 time-derived 特征都倾向负面**——可能因为 100-tick 窗口的统计本身已经隐式捕捉到 time-of-day pattern。

---

### 3.5 不加的特征家族（全部 ablate 过）

| 方向 | T-编号 | 死因 |
|---|---|---|
| Hawkes 过程 OFI | T141/T142/R_Hawkes_OFI | 增量边际 |
| SG filter (Savitzky-Golay) | T35/T91 | SG 对树模型 gain ≈ 0 |
| Trend features (mean_logret_W10-100) | T191 | **−10.33**, corr(T191, T188v2) = **0.9941**, IC_residual_pooled = **−0.17** → 完全冗余 + 负 IC |
| logret lag | R4/T124 | 无增量 |
| Pairwise 交互 | R_Pairwise_ALL | 计算量大，边际 |
| Roll/TSRV | R_roll_tsrv | 部分入 Stage 3，未成主力 |

**T191 trend features 是典型反例**：standalone +140.10 vs T188v2 +150.43（−10.33），**corr 0.9941 但 IC_residual 负**。教训：高 correlation + 微弱负残差信号 = 完全冗余，**这种特征加入 ensemble 反而稀释**。

---

## §4. 模型架构的关键选择

### 4.1 MLP `[359 → 256 → 128 → 64 → 1]` + LayerNorm + GELU + Dropout

**架构**：4 层渐缩 MLP，参数量 ~120k（relatively small，避免过拟合 1.47M 样本）。

**为啥 MLP 而非 GRU/Transformer/CNN**（强烈的失败证据链）：
- **T95 GRU**：hidden state 易在短 horizon 过拟合，是 iter_016 v3 失败主因
- **T172/T187 GroupTransformer**：in-sample +149.91 ~ +164.71（**全 in-sample 污染**），OOD 风险高，**未提交**
- **T184/T185 Deep CNN**：in-sample 看似好，OOD 未验证，**未提交**
- **T186 verdict**：`architecture_doesnt_fit` —— 深度架构在 SchemeP 359 维 aggregate-feature 框架下无法竞争 MLP
- 设计哲学：特征工程已把 100-tick 窗口的时序信息提炼成 aggregate features → MLP 看 aggregate features 是更鲁棒的设计

**LayerNorm（不是 BatchNorm）**：per-sample 归一化，与 SchemeP 窗口 z-score 互补；batch 内不同样本独立 → **符合硬约束 #2（batch shuffle）**。BatchNorm 在 batch shuffle 下会泄露 batch 统计。

**GELU + Dropout 0.10**（中心）：现代 NLP 标准；Dropout 在 0.05-0.20 间 cycling 提供 ensemble 多样性。

---

### 4.2 LGB 而非 CatBoost / XGBoost

| 实验 | 方法 | LOSO |
|---|---|---|
| T17 | CatBoost vs LGB | 持平 (~+22.2 ceiling) |
| T18 | XGBoost vs LGB | 持平 |
| T89 | CatBoost L2 5-seed | +40.13（与 T87 NN **corr 0.77**，高度共线）|
| T166 | CatBoost L2 5-seed M7 retrain | holdout +0.39（边际）|

**为啥 LGB**：ceiling 三者持平，但 LGB 的 leaf-wise growth + 高效 GPU 训练（项目实测 LGB GPU vs CPU 38s → 12s, **3.2×**）。CatBoost 与 T87 NN **corr 0.77** 高度共线，加入 ensemble 无增量。

**目标 / loss**：`objective='regression_l2'`，y = (mp_h − mp_t)/(mp_t + 1)。MAE（T137/T143/T145）不超 L2；Huber（T99）OOF 过拟合警告；Quantile（T132）边际。**L2 始终主力**。

**Class-balanced weight（训 L2 regression 但用分类 weight）**：`y_cls = label_h60` (三分类) → `weights = 1/freq(y_cls)`。语义模糊但 work：h_60 标签不平衡（多数 sample 是"不动"），少数类（涨/跌）weight 让模型重视决策边界。**T92 magnitude weight（按 |y| 加权）反效果**（LOSO 下降）→ 假设证伪。

---

## §5. 推理优化的关键 tricks

### 5.1 T61 batch-vec 特征提取 **58× 加速**【**首次通过 3h 限时**】

iter_007/008/009 因 254 min feature 计算**全部超时**（平台 3h 限制）。T61 用 NumPy/Pandas 向量化替代 Python for 循环：**254 min → 4.5 min（58×）**。

**iter_010 是项目历史性节点**：首次通过 3h 限时，开启所有后续提交可能性。

---

### 5.2 T188v3 batched torch CUDA bmm **186× 加速**

| 阶段 | NN-only per-1024 batch | e2e per-1024 batch |
|---|---|---|
| T188v2 (numpy sequential) | 1509ms (stated) / 554.8ms (实测) | 2142ms |
| T188v3 (torch CUDA bmm) | **8.1ms** | **641ms** |
| 加速 | **186×** (stated) / 68.5× (实测) | **3.3×** |

**实现**：50 NN 权重堆叠成单一 tensor `W_all shape (50, out, in)`，单次 `torch.bmm(x_repeat, W_all)` 替代 50 次 numpy sequential 循环。

**actions 一致性验证**：T188v3 vs T188v2 actions max diff **6.98e-10**（远低于 1e-4 容忍度）→ 数值等价。

**总体**：442k 行总时间 15.4 min → 4.6 min。

---

### 5.3 5-horizon share h=60（**5× → 1× cost**）

`T188v3_fullhorizon`：所有 5 horizon 用 h=60 ensemble 输出，按各自 thr 缩放决策。**单次推理服务 5 个 head**。

详见 §2.4。

---

### 5.4 CPU-only torch wheel

`requirements.txt: torch==2.5.1+cpu`，平台无 GPU 假设。代码有 CUDA 分支（`device = torch.device("cuda" if cuda.is_available() else "cpu")`）但 CPU torch 下永远 false——**冗余但保留** "训练-推理代码共用" 的残留。

**为啥 npz 不 pt**：避免 torch 序列化跨版本兼容问题；平台 `weights_only=False` 在新版 torch 中可能 reject pickled module；npz 是 numpy 通用格式。代价：手 extract per-layer 权重 → 维护成本（layer_idx 计数脆弱）。

---

### 5.5 Stateless Predictor 设计（满足硬约束 #2）

**关键**：`Predictor.predict()` 完全 stateless：
- `pred_cache` 是 `predict()` **局部变量**，每次调用独立
- 所有 reductions 沿 **time 维（window 100 步）** 或 **hidden 维**，**从未沿 sample 维**
- 所有 sigma/beta/feat_mean/feat_std/target_scale 都是**训练时算好的常量**，inference 时只 load
- **没有 BatchNorm**（手实现 LayerNorm 沿 hidden 维）、**没有 softmax/dim=0**、**没有 TTA/TTT**、**没有 batch-rank / cross-sample agreement vote**

**AUDIT_BATCH_LEAKAGE.md 结论**：3830 行代码审计，**0 高风险 / 0 中风险**。

**多次"踩坑"实验印证约束 #2 是真实硬限制**：T22 Alpha101（跨截面）/ T161 TTA / T162 TTT / T168 adaptive thr / T169 AR TTT / R_cross_sym → **全部未通过平台**，是 stateless 设计的反面证据库。

---

## §6. 评估方法论的关键 tricks

### 6.1 V4 walk-forward（train 0-79 / val 80-95 / test 96-119）

Phase 3 (T58/T64) 引入，是项目最干净的本地 OOD 代理。LOSO-eq → 平台透传系数 **53-71%**（详见 6.2）。

V4 用 time-OOF 而非 sym-OOF（LOSO），匹配平台 "评测在训练截止后的新数据" 的真实场景。

---

### 6.2 LOSO-equiv → 平台 透传系数表

| iter | LOSO-eq | 平台 | 透传 % |
|---|---|---|---|
| iter_013 | +36.23 | +19.23 | **53%** |
| iter_015 | +40.09 | +28.16 | **71%** |
| iter_018 | +41.49 | +28.93 | **70%** |

**透传系数 53-71% 区间方差大**：DE 阈值越激进透传越低（iter_013 DE 4D → 53%）。这是 §2.1 "thr_up/thr_dn 不再 DE 调" 的根源。

**没有可靠 "本地 → 平台" 转换公式**（GAP）：只能定性"本地涨大则平台可能涨小，本地涨小则平台可能持平甚至跌"。

---

### 6.3 4 个 in-sample 污染陷阱案例（**holdout in-sample 阶段必读**）

Phase 4 (T156+) 切到 holdout in-sample（M7 之后没有干净 V4 OOF）。**绝对值不可信，只看相对 delta**。但 4 个案例显示 delta 本身也可能反向：

| 实验 | 本地 holdout | 平台 | 教训 |
|---|---|---|---|
| **T182 无 conformal ablation** | **+6.62**（看似 conformal hurt）| **−0.30**（conformal 实际有用）| **本地 delta 与平台 delta 可以反号** |
| T180 OOD GMM abstain | −32 ~ −78 | 未提交 | 测试集比训练 in-distribution |
| T172 GroupTransformer | in-sample +149.91 | 未提交 | 全 in-sample 污染 |
| **T192 bounded NNLS** | **val +180.41, test +193.94** (ratio 1.075) | **+31.0 (−4.64)** | val+test 均 in-sample，OLS 学到 in-sample 共同 noise |

**T182 是最经典案例**：去 conformal 后本地 +6.62，平台 −0.30。**证明本地 holdout 和平台 OOD 可以完全反向**，只能用平台为最终裁判。

**T192 的 val/test ratio = 1.075** 是 in-sample noise alignment 的**烟雾弹**：看起来 val 和 test 都涨说明 robust，实际 val+test 都是 in-sample，OLS 学到的是它们的共同 noise。

---

### 6.4 平台是唯一 ground truth — 保守迭代策略

22 次平台提交里反复出现的模式：
- iter_013 DE 4D LOSO +36.23 → 平台 +19.23（53% 透传）⇒ 教训：DE 过拟合
- iter_016 v3 4-way LOSO +43.71 → 平台 +25.08（**−3.08**）⇒ 教训：复杂 ensemble 倒退
- T163 v2+T97+agree holdout +42.90 → 平台 −3.70 ⇒ 教训：同架构 ensemble 无 diversity
- T192 H4 NNLS test +193.94 → 平台 −4.64 ⇒ 教训：权重学习 in-sample overfit

**策略**：thr_up/thr_dn 一旦定下来不再动；新模型/loss 必须 OOD 验证；ensemble 扩容必须两侧对称且 ablation 渐进。

---

## §7. 试过但 FAIL 的关键尝试（负面教训）

> "知道哪些方向不 work" 是项目最有价值的知识资产之一。**这些方向在没有这份文档的情况下，未来 6 个月新人极可能再次踩坑**。

### 7.1 决策层/阈值类 失败

| 方向 | T-编号 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| OOF DE 4-way 阈值 | iter_016 v3 (T106-109) | LOSO +43.71 | **+25.08** | DE 在 442k 本地测试严重过拟合（−3.08）|
| per-sym OOF threshold | T126 | LOSO 高 | 未提交 | 过拟合每 sym OOF 分布；OOD sym 无 model |
| conformal-aware re-DE | T146 | +2.76 | 未提交 | 二次 DE 风险叠加 |
| Adaptive threshold V1-V4 (per-batch 统计) | T168 | 全负 | 未提交 | 触碰约束 #2（batch shuffle） |
| 对称 k×mean_abs_pred | T62 | +33.72 | — | 鲁棒但比 DE 低 2.51 |

### 7.2 Wrapper / TTA / TTT 类 失败（**全部触碰硬约束 #2**）

| 方向 | T-编号 | 本地 | 死因 |
|---|---|---|---|
| TTA K=5（5 次增强预测平均） | T161 | holdout 下降 | per-batch 统计自适应在乱序下失效 |
| In-row TTT W3a/b/c | T162 | W3a −0.30, W3b −1.98, W3c **−3.11** | 同上 |
| AR TTT（自回归 test-time training） | T169 | 退步 | 同上 |
| In-batch wrapper B/C | T177 | holdout 正 | 未提交（同上风险）|
| OOD GMM density abstain | T180 | V1 −32, V2 −57, V3 **−78** | 测试集比训练 in-distribution；弃权反损 |
| Time-window hard abstain | T181 | V1 **−23.39** | T170 conformal 已过滤，硬弃权适得其反 |

**元教训**：约束 #2 "测试点输入顺序被打乱" 是**真实硬限制**，6 个 T 实验全部未通过。任何 "per-batch 自适应" / "test-time adaptation" / "sample 间相互依赖" 的设计**结构性失败**。

### 7.3 模型架构类 失败

| 方向 | T-编号 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| 3-class CE NN (DeepLOB) | T1 / mmpc_demo | LOSO ~+8 | **−6.65** | 压缩幅度信息 |
| GRU regression | T95 | LOSO 微小 | iter_016 失败主因 | hidden state 在短 h 过拟合 |
| **第二组 NN T97**（同 T87 架构 + agreement filter） | T156/T163 | holdout +1.41 ~ +1.66 | **−3.70** vs v2 | **架构同源 diversity ≈ 0**（关键教训）|
| 4-way (T87+T89+T99+T95) | iter_016 v3 | LOSO +43.71 | **+25.08** | GRU + OOF DE 双重过拟合 |
| 5-way (+ CatBoost) | T167 | +0.11 | 未提交 | within noise |
| GroupTransformer | T172/T187 | in-sample +149.91 ~ +164.71 | 未提交 | 全 in-sample 污染，OOD 风险 |
| Deep CNN raw / hybrid | T184/T185 | in-sample 看似好 | 未提交 | OOD 未验证 |
| Snapshot ensemble | T135 | 边际 | 未提交 | 不超 5-seed |

### 7.4 损失函数类 失败

| 方向 | T-编号 | 本地 | 死因 |
|---|---|---|---|
| Direct PnL loss（无 SPO+ 正则化） | T13/T134/T57/T60 | 不收敛 | PnL 公式分段不连续，裸梯度方向不一致 |
| GMADL direction-aware loss | T116 | LOSO 微弱 | 不超 SPO+ |
| MAE LGB | T137/T143/T145 | LOSO +0.3-0.5 边际 / 平台预测 +30~32 | 不超 L2 M7 |
| Huber LGB | T99 | LOSO +44.72 (含 OOF 污染) | iter_016 失败 |
| Quantile LGB α=0.5 | T132 | 边际 | LGB 已 L2 work |
| Magnitude sample weight（按 \|y\| 加权） | T92 (R44) | LOSO 下降 | **R44 假设证伪**：幅度加权反损 PnL（噪音也大）|
| isotonic / Platt 校准 | T15/T40/R13/T131 | LOSO −1.02 ~ 持平 | 破坏排序质量 |

### 7.5 Ensemble 权重学习 失败

| 方向 | T-编号 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| Stacker NN | T139 | LOSO 负面 | 未提交 | OOF 过拟合 |
| **Bounded NNLS** [1/1500, 2/150] per-group | T192 H4 | val +180.41, test **+193.94** | **+31.0 (−4.64)** | **NN-NN corr 0.9331，design matrix cond=54182**，OLS 学 in-sample 噪音 |

### 7.6 Ensemble 扩容 失败

| 方向 | T-编号 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| 40-NN only（只扩 NN） | T179 | +0.63 | **−0.25** | 只扩一侧无 variance reduction |
| 150+150 | T190 | **−9.81** | **−1.05** | HP 多样性边际后变噪音 |
| LGB 10-seed (vs 5-seed) | T178 | null | — | 5-seed 已在 variance floor |

### 7.7 特征类 失败

| 方向 | T-编号 | 本地 | 死因 |
|---|---|---|---|
| Alpha101 / Alpha191（WorldQuant 跨截面） | T22 | IC **+21.80** | **违反约束 #2 batch shuffle**，未提交 |
| Hawkes 过程 OFI | T141/T142/R_Hawkes_OFI | 边际 | 增量不显著 |
| TOD (Time-of-Day) | T151 v9 | LOSO 小正 | T170 final 未用 |
| sin/cos time features (SchemeQ) | T74/T78 | LOSO-eq **−0.72** | 回归框架下时间特征负面 |
| Trend features (+5 维) | T191 | −10.33, corr 0.9941, IC_res **−0.17** | 完全冗余 + 负 IC |
| Savitzky-Golay filter | T35/T91 | 不超 SchemeC | SG 对树模型 gain ≈ 0 |
| logret lag | R4/T124 | 无增量 | KILL |
| Global z-score | T10 (SchemeF) | LOSO +13.40 | 不如窗口 z-score (T7 +13.91) |
| sym embedding / per-sym norm | 隐式 | — | 违反约束 #3 |
| date-derived features | 隐式 | — | 违反约束 #1 |

### 7.8 调研类 失败（未深入实验）

| 方向 | 来源 | 死因 |
|---|---|---|
| TimesNet / iTransformer / PatchTST / N-BEATS | R31 文献调研 | 用户明确判定 "AI 论文是垃圾"；通用时序在 LOB high-noise low-signal 上未实证有效 |
| Pseudo labeling | T65 | 负面 |
| Group DRO | R2 | fold2 改善但 fold4 牺牲，净效果负 |
| Bayesian optimization | R5 | 与 DE 相近，无明显优势 |
| 3-stack compound stacking | R_stack3_compound | OOF 过拟合 |

### 7.9 失败方向的元教训（**6 条**）

1. **OOF DE 阈值过拟合是 5 重证据**（iter_013 透传 53% / iter_016 平台 −3.08 / T126 未提交 / T146 未提交 / T192 NNLS 平台 −4.64）→ 任何在 OOF 上学习决策参数都高风险
2. **新架构必须 OOD 验证**：M7 retrain 只对**已验证**的 T75 LGB / T87 NN 有效；新架构（Transformer/CNN）in-sample 数字漂亮但 OOD 风险高
3. **TTA/TTT/per-batch 自适应反复触碰约束 #2**：6 个实验全部失败
4. **OLS/NNLS 权重学习因病态共线性失败**：NN-NN corr **0.9331**，权重不可识别
5. **短 horizon 模型独立训也救不了**：α=0.05% < fee=0.02% 双边，结构性输
6. **trend features 完全冗余**：corr(T191, T188v2) = **0.9941**，IC_residual = **−0.17** → 高 correlation + 微弱负残差 = 冗余特征加 ensemble 反而稀释

---

## §8. 数学/逻辑层面的疑点（USER-RAISED）

> 这些不是 "trick"，而是项目作者**未完全解决**的设计问题。任何继续这个项目的人都需要先看这一节。

### 8.1 ⭐ Q281 — sqrt(h/60) 阈值缩放方向数学上反了

**现状**：`thr_h = thr_60 × sqrt(h/60)`（短 h thr 变小），短 horizon 直接 share pred_h60。

**用户精确推导**（drift-with-noise 模型）：
```
E[Δmid_h] = μ·h        (drift 线性于 h)
std[Δmid_h] = σ·sqrt(h) (noise sqrt)
```

模型 pred 是 60-tick 归一化 Δmid。若想用 pred_h60 给 h=5 决策：

| 做法 | 公式 | 含义 |
|---|---|---|
| 正确 A（缩 pred） | `pred_h60 × (h/60) > thr_native` | 线性缩 |
| 正确 B（反缩 thr） | `pred_h60 > thr_native × (60/h)` | thr 变大 |
| 按 noise std 缩 thr（"标准化做法"） | `thr × sqrt(60/h)` | thr 变大 |
| **代码实际** | `thr × sqrt(h/60)` | **thr 变小** ← 方向反了！|

**差距**：与"按 noise std 标准化" 差 `60/h` 倍。h=60 差 1×，h=40 差 1.5×，h=20 差 3×，h=10 差 6×，**h=5 差 12×**。

**为啥现状能 "work"**：
1. **平台 score = max(per-horizon)** → 短 h 即使更差也不影响 +35.64 排名
2. 短 h 有 conformal 弹性带兜底（per-sym β·σ 加在 thr 上）：at h=5, sym=1，band = 1.88e-4，thr = 8.66e-5，**band 是 thr 的 2.17 倍** → effective_thr ≈ 原始 h=60 thr → **hi-band sym 几乎抵消 sqrt 缩放**（意外副作用）

**严重性**：
- 数学正确性：**错误**（方向反了 12×）
- 影响主分数：**小**（max 规则隐藏）
- 影响潜在分数：**未知**（可能短 h 优化后改变 max-rank）
- 是否易修复：**是**（改 thresholds.json 一行）

**用户建议补做实验**：submit 4 个变体（缩 `h/60` 线性 / 缩 `1.0` 不缩 / 缩 `sqrt(60/h)` 反向 / 关闭短 h 只交 h=60），对比短 h 平台分。如果任何变体让短 h 平台分超过 h=60 的 +35.64，就会改变 max-rank 结果。

### 8.2 GELU erf vs tanh 训练-推理不一致（CODE Q176）

| 阶段 | 实现 |
|---|---|
| 训练 | `nn.GELU()`（默认 `approximate='none'`，即 erf） |
| 推理 | `F.gelu(h, approximate='tanh')` |

**数值差异**：\|x\| > 1 范围 ~1e-4 数量级；4 层 hidden 累加后可能 ~1e-3；阈值 thr_up=3e-4 → 1e-3 误差可能 flip action。

**T188v3 actions 一致性 6.98e-10 不能 cover 此问题**：那个验证是 "T188v3 vs T188v2 同 GELU 内的等价"，**未验证训练 erf vs 推理 tanh 差异**。

**严重性**：中-高（潜在 bug，影响所有 50 NN 推理）。修复成本：**1 行**（推理改 `approximate='none'` 或训练改 tanh）。**作者未明确回应或修复**。

### 8.3 amount_delta 处理不一致（CODE Q47-Q48）

`raw_last` 列做 `sign × log1p(|v|)`，但 `X3d` 下游特征用原始 amount_delta。同一底层量两种尺度表示。

**未做 ablation** "全部 log1p" vs "全部 raw"。Reviewers 推测下游特征内置归一化（cbrt/std/mean ratio），所以 raw OK；模型可能学到 redundancy 而非冲突。设计/bug 未明，empirically work。

### 8.4 SPO+ 理论 GAP

- **Fisher consistency in 1D**（CODE Q174）：SPO+ (Elmachtoub & Grigas 2022) 原文针对线性 LP；改成 1D scalar 三选项任务，**Fisher consistency 未数学证明**
- **Weighted SPO+ 凸性**（CODE Q196）：`(spo_per * wb).sum() / wb.sum()` 加 class_balanced_weight，**凸 surrogate 性质未证明**

两个都是 empirically work（T87 平台 +28.16 支撑），但理论严谨性缺失。

### 8.5 Ensemble 权重 in-sample 过拟合教训（T192）

**用户/PM 反复强调**：val/test ratio > 1.05 是 in-sample noise alignment 的烟雾弹，看似 robust 实则共同噪音对齐。

**T192 H4 NNLS**：val +180.41, test +193.94, ratio 1.075 → 平台 **−4.64 vs SOTA**。

**普适教训**：**ensemble 权重必须用真正 OOD 数据学，不能用 in-sample holdout**。在病态共线性（corr > 0.9）下，simple mean > 任何 "学" 出来的权重。

### 8.6 USER-RAISED 汇总表

| 问题 | 来源 | 数学严重性 | 影响 PnL | 是否 ablation | 是否易修复 |
|---|---|---|---|---|---|
| **Q281 sqrt 缩放方向** | 用户精确推导 | **高**（方向反 12×）| max 规则隐藏 | **否** | **是**（1 行） |
| GELU erf vs tanh | CODE Q176 | 中（数值差异）| 可能小 | 否 | **是**（1 行） |
| amount_delta 不一致 | FEATURES Q20 | 中（设计/bug 未明）| 未量化 | 否 | 中 |
| SPO+ 1D Fisher consistency | CODE Q174 | 低（理论）| 0 | 否 | 不需修 |
| weighted SPO+ 凸性 | CODE Q196 | 低（理论）| 0 | 否 | 不需修 |
| Ensemble 权重 in-sample 教训 | T192 实证 | — | −4.64（已发生）| 已 | 教训已沉淀 |

---

## §9. 平台 SOTA 完整决策链（TLDR 数据流）

按数据流顺序列出每个起作用的设计，便于一目了然 reproducible 路径。

```
[1] 输入 raw data
    └── 5 sym × 120 dates × 2001 ticks，154 维 raw features

[2] 特征构造 (build_schemeP_cache.py + fast_features_batch.py)
    ├── 154 raw + 216 derived = 370 维 SchemeP
    │   ├── T3 no-time (69): MLOFI/WMP/RV/EWMA-intst
    │   ├── R34 Stage 1 (54): dual z-score, signed RV, Kyle inv, EWMA-OFI
    │   ├── R34 Stage 2 (59): QRank, RSkew, GOFI, Kyle λ, Vol burst
    │   ├── R34 Stage 3 (14): EWMA residual, RV ratio, Bipower, Cancel imbalance, Roll spread
    │   └── R34 Stage 5 (20): Adapt momentum, OFI toxicity, Signed BV, Spread regime, Trade persistence, Liq asymmetry
    ├── 100-tick 窗口 z-score 归一化
    ├── amount_delta sign-preserving log1p (raw_last 列；X3d 仍用原始)
    └── drop 11 FAIL_NAMES (T59_FAIL 10 + STAGE5_FAIL 1) → 359 维进模型

[3] 训练目标
    └── y = (mp_{t+60} − mp_t) / (mp_t + 1)，Δmid 回归

[4] 模型 1 — 50 LGB
    ├── 5 HP × 10 seeds (feature_fraction / bagging_fraction / num_leaves / λ_L2 变化)
    ├── aug_a bid/ask 互换 + label 反转，训练数据 ×2
    ├── class-balanced weight (3-class 标签 weight 应用在 L2 regression)
    ├── objective='regression_l2', learning_rate=0.05
    └── M7 全数据重训：date 0-119, num_boost_round=330 (V4 best_iter ×1.1)

[5] 模型 2 — 50 NN
    ├── MLP [359 → 256 → 128 → 64 → 1]
    ├── LayerNorm + GELU + Dropout, AdamW (wd=1e-4), CosineAnnealingLR
    ├── HP cycling: LR ∈ {1e-4,3e-4,1e-3} × Dropout ∈ {0.05,0.10,0.15,0.20} × Batch ∈ {2048,4096,8192}
    │   (idx (S-1)%3 / %4 / %3 → 12 unique combos × ~4 reps)
    ├── Phase 1: L2 pretrain on train(0-79)/val(80-95), max 50 ep, patience 10
    └── Phase 2: warm-start → SPO+ DFL fine-tune, λ=30, lr=3e-5,
                  M7 全数据 (date 0-119) 11 epoch
                  L = MSE + λ × SPO+ (class-balanced weighted)

[6] Ensemble
    ├── pred_nn = mean(50 NN)         (after target_scale 还原)
    ├── pred_lgb = mean(50 LGB)
    └── pred = (1.0 × pred_nn + 1.5 × pred_lgb) / 2.5

[7] 决策层 (EV gate, asymmetric)
    ├── thr_up = 0.0003, thr_dn = 0.000216 (全局固定，不分 sym)
    └── per-sym conformal abstain band:
        band[sym] = β[sym] × σ[sym]   (常量配置)
        β = {sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}
        σ from thresholds.json (per-sym validation residual std)

    if pred > thr_up + band → 买 (action=2)
    elif pred < -(thr_dn + band) → 卖 (action=0)
    else → 平 (action=1)

[8] 5-horizon emit
    ├── h ∈ {5, 10, 20, 40, 60}，h<60 共享 h=60 ensemble 输出
    └── thr_h = thr_60 × sqrt(h/60)   ⚠️ 方向数学上反了（USER-RAISED Q281），但 max 规则救场

[9] 推理优化
    ├── T61 batch-vec 特征 (58× 加速)
    ├── T188v3 batched torch CUDA bmm (50 NN 一次 forward, 186× 加速)
    ├── 5 horizon share h=60 (1× cost 不是 5×)
    ├── CPU-only torch 2.5.1 wheel (平台无 GPU)
    └── Stateless Predictor (pred_cache 是 predict() 局部变量，无跨 batch state)

[10] 最终平台分
    └── max over 5 horizons = h_60 score = +35.64
```

---

## §10. 最关键的 8 条工程经验

> 这 8 条是项目最普适的方法论沉淀，比任何具体 trick 更重要。

### 1. **本地 → 平台 gap 永远存在**
透传系数 53-71% 区间方差大；in-sample holdout 阶段甚至 delta 都可能反号（T182 本地 +6.62 → 平台 −0.30）。不要相信任何 in-sample holdout 数字，**平台是唯一裁判**。

### 2. **GBDT 全数据重训安全（+5.51）；NN 全数据重训增量小（+0.20）**
LGB 从头训，M7 新数据增量大；NN 已 warm-start + SPO+，M7 增量已饱和。**M7 是已验证模型的 +5，不是新架构捷径**。

### 3. **同架构 ensemble 不带 diversity（T163 教训 −3.70）**
T97 与 T87 同 MLP 架构，corr ≈ 1.0；加 agreement filter 本地 +1.41 但平台 −3.70。**Ensemble diversity 必须来自架构层面或 HP 层面，不能来自 seed 层面**。

### 4. **per-batch 自适应在平台打乱顺序下必死**
TTA / TTT / AR TTT / adaptive thr / OOD GMM abstain → **6 个实验全 kill**。约束 #2 是真实硬限制，任何 "test-time adaptation" 结构性失败。

### 5. **特征工程贡献 > 模型变体加起来**
R34 Stage 1-5 累积本地 +9（+13.67 → +16.84 → +26.44 含方法论），比 GRU / CatBoost / Transformer / Huber 等模型变体加起来都多。**特征工程是 alpha 真正来源**。

### 6. **OOF / in-sample 数据上学决策参数 = 过拟合赌博**
5 重证据：iter_013 透传 53% / iter_016 v3 平台 −3.08 / T126 未提交 / T146 未提交 / T192 NNLS −4.64。**任何在 OOF 上学习的阈值/权重都高风险**。Simple mean > 任何 NNLS/OLS 学权重（病态共线性下不可识别）。

### 7. **保守 thr，激进 retrain，渐进 ensemble**
thr_up=0.0003 / thr_dn=0.000216 一旦定下来所有后续大版本不动；M7 retrain 在已验证模型上激进推；ensemble 5→50 +1.00 但 50→150 −1.05（非单调），不冒进。

### 8. **数学正当性 ≠ empirically work**
sqrt(h/60) 方向反 12× 但因 max 规则隐藏；GELU erf/tanh 不一致但 work；SPO+ 1D Fisher consistency 无证明但 work。**Empirical 结果不能替代理论严谨性**，未来如果优化短 horizon，Q281 必须先解决。

---

## 附录 A：T 编号速查（按 phase）

- **Phase 0 (T1-T10)**: T1 NN baseline, T2 LGB baseline, T3 SchemeC 226 维, T5b iter_002 (+4.07), T7 窗口 z-score, T8 sym=2 诊断, T10 全局 z-score 失败
- **Phase 1 (T11-T35)**: T13 PnL loss 不收敛, T15 isotonic 校准, T17 CatBoost, T22 Alpha101 (硬约束), T23 平台校准, T25-28 aug_a (+3.37), T30 DE 4D 阈值, T35 SG filter 失败
- **Phase 2 (T44-T74)**: T44 R34 Stage1, T51 Stage2 (+15.06), T53 Stage3, T55 Stage4, T58 V4 walk-forward, T59 全 sym 训练 (+7 方法论), **T61 batch-vec 58× → 首次通过 3h**, T64 V4 正式化, T68 Stage5 (SchemeP 359 维完成), T74 SchemeQ
- **Phase 3 (T75-T92)**: **T75 Δmid 回归 +19.23 (1st 突破)**, T80 透传校准, T81 NN L2, **T87 SPO+ DFL +28.16 (2nd 突破)**, T89 CatBoost, T92 magnitude weight 证伪
- **Phase 4 (T95-T109)**: T95 GRU 失败, T97 第二组 NN, T99 Huber LGB, T106-109 iter_015/016 打包系列
- **Phase 5 (T115-T155)**: T127 iter_018 conformal +28.93, T128 M7 起点, T136 SPO+ 超参, T137 MAE vs L2, **T140 M7 LGB +34.44 (3rd 突破)**, T143 v3 MAE, T146 conformal DE, T147 feature pruning, T150 conformal β sweep, T152 adversarial 259 维
- **Phase 6 (T156-T169)**: T156 NN swap, T158 agreement, T161 TTA 失败, T162 TTT 失败, **T163 v2+T97 失败 −3.70**, T165 magnitude filter, T166 CB M7, T168 adaptive thresh 失败, T169 AR TTT 失败
- **Phase 7 (T170-T192)**: **T170 NN M7 +34.64**, T172 GroupTransformer (未提交), T176 T87M7 HP sweep, T178 LGB 10-seed null, T179 40-NN +34.39 (-0.25), T180 GMM abstain 失败, T181 time-window abstain 失败, **T182 无 conformal ablation (验证 conformal 真有效)**, T183 NN random init, T184-T187 deep CNN/Transformer 未提交, **T188v2 50+50 +35.64 (SOTA)**, T188v3 186× 推理加速, T189 GPU pkg, **T190 150+150 −1.05**, T191 trend features KILL, **T192 NNLS overfit −4.64**

## 附录 B：硬约束 3 条速查（必须遵守）

1. **`date` 字段评测时被置 0** → 不能当 feature / embedding / derive 时段
2. **测试点输入顺序被打乱** → Predictor 不能维护跨 batch state（LSTM hidden / 缓存 buffer / per-batch 统计）
3. **`sym` 0-4 但可能含训练外股票** → 模型必须 sym-agnostic（不能 sym embedding / per-sym normalization / per-sym 模型）

**违反 = 提交报错或 0 分**。整个项目踩坑实验（T22 Alpha101 / T161 TTA / T162 TTT / T168 adaptive / T169 AR TTT / R_cross_sym）反复印证。
