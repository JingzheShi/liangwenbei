# 良文杯 2026 — 最终提交决策解释报告

> **文档目的**：把 7 份独立白板 review 提出的 ~1091 个"为什么这么做"问题与 200+ 实验、22 次平台提交的证据做真正的 JOIN，回答"为什么这么选 + 我们做过什么实验/消融支持这个选择 + 试过的失败替代方案"。
>
> **阅读对象**：从未参与项目的研究者 / 审稿人。读完后应该理解最终提交（T188v2 50+50 ensemble = 平台 **+35.64**）里每一个核心决策的来龙去脉。
>
> **数据来源汇总**：
> - `HISTORY_EVIDENCE_DB.md`（10 主题 + 4 附录，242 个 T 实验、40+ R 系列、22 次平台提交锚点）
> - `CODE_QUESTION_LIST.md`（200 问，全局视角带先验例子）
> - `QREVIEW_NN_TRAIN.md` / `QREVIEW_LGB_TRAIN.md` / `QREVIEW_PREDICTOR.md` / `QREVIEW_FEATURES.md` / `QREVIEW_PIPELINE_ARCH.md`（5 份独立白板 review，共 891 问）
> - `CRITICAL_CONSTRAINTS.md`（评测协议 3 条硬约束）
>
> **方法论说明**：1091 个白板问题里有大量重复（多个 reviewer 从不同角度问同一个常量）。本文按主题合并去重为 ~250 个独立 Question，每个用"决策 → 候选 → 证据 → 反例 → GAP"格式回答。
>
> **数字标注规约**：
> - 「LOSO」= 5-fold Leave-One-Sym-Out（Phase 0-1）
> - 「LOSO-equiv」= 全 sym 训练 + DE 阈值搜索在 442k 行本地测试集上（Phase 2）
> - 「LOSO-equiv + V4」= train 0-79 / val 80-95 / test 96-119 walk-forward（Phase 3-5）
> - 「holdout in-sample」= date 0-119 全数据训练，date 96-119 评分（Phase 6-7，**含 in-sample 污染**）
> - 「平台」= 公榜得分（5 horizon 取最高，h_60 几乎总是最高，**唯一 OOD ground truth**）

---

## 目录

- [§0. 概览 — 22 次平台提交时间线与总体增益](#0-概览--22-次平台提交时间线与总体增益)
- [§1. 评测协议硬约束（3 条）](#1-评测协议硬约束3-条)
- [§2. 评估方法论变迁（4 阶段）](#2-评估方法论变迁4-阶段)
- [§3. 阈值与决策层（thr_up / thr_dn / EV gate）](#3-阈值与决策层thr_up--thr_dn--ev-gate)
- [§4. Conformal abstain（per-sym β 与 σ）](#4-conformal-abstainper-sym-β-与-σ)
- [§5. 5-horizon share_with=60 + sqrt 缩放](#5-5-horizon-share_with60--sqrt-缩放)
- [§6. Ensemble 组成与权重（50+50, w_nn=1.0, w_lgb=1.5）](#6-ensemble-组成与权重5050-w_nn10-w_lgb15)
- [§7. M7 全数据重训（第三次大突破）](#7-m7-全数据重训第三次大突破)
- [§8. SPO+ DFL（第二次大突破）](#8-spo-dfl第二次大突破)
- [§9. NN 模型架构](#9-nn-模型架构)
- [§10. NN 训练协议（HP cycling、aug_a、early stop）](#10-nn-训练协议hp-cyclingaug_aearly-stop)
- [§11. LGB 训练协议（HP grid、bagging、num_leaves）](#11-lgb-训练协议hp-gridbaggingnum_leaves)
- [§12. 损失函数（Δmid 回归 vs 分类、L2 vs MAE/Huber）](#12-损失函数δmid-回归-vs-分类l2-vs-maehuber)
- [§13. 特征工程（SchemeP 370 维、Stage 1-5、drop 11、归一化）](#13-特征工程schemep-370-维stage-1-5drop-11归一化)
- [§14. 推理优化（batched bmm、npz vs pt、CPU torch）](#14-推理优化batched-bmmnpz-vs-ptcpu-torch)
- [§15. 评估方法论与本地 → 平台对应](#15-评估方法论与本地--平台对应)
- [§16. 架构 / 流程组织（4 步 pipeline、build_pkg、requirements）](#16-架构--流程组织4-步-pipelinebuild_pkgrequirements)
- [§17. 失败 / 未采纳的方向](#17-失败--未采纳的方向)
- [§18. 用户提出的核心问题 / 数学正当性挑战（USER-RAISED）](#18-用户提出的核心问题--数学正当性挑战user-raised)
- [§19. 已知风险与 GAP 总结表](#19-已知风险与-gap-总结表)
- [§20. 附录 — 关键文件路径与 T 编号速查](#20-附录--关键文件路径与-t-编号速查)

---

## §0. 概览 — 22 次平台提交时间线与总体增益

### 0.1 整体跃迁

整个项目历经 22 次平台提交，从 mmpc_demo baseline `−6.65` 起步，最终 SOTA 为 T188v2 50+50 ensemble 平台分 **+35.64**。**总增益 +42.29 绝对值**。

| 提交 # | iter / T 编号 | 主要 trick | 平台 PnL | Δ |
|---|---|---|---|---|
| 1 | iter_000 mmpc_demo | DeepLOB baseline | **−6.65** | — |
| 2 | iter_002 SchemeC | LGB 多 h + 阈值 gate | **+4.07** | +10.72 |
| 11 | iter_013 (T75) | Δmid regression + EV gate | **+19.23** | +15.16 **（1st 突破）** |
| 13 | iter_015 (T87) | SPO+ DFL NN + T75 LGB | **+28.16** | +8.93 **（2nd 突破）** |
| 15 | iter_018 v1 (T127) | + per-sym beta conformal | **+28.93** | +0.77 |
| 16 | iter_019 v2 (T140) | M7 LGB 全数据重训 | **+34.44** | +5.51 **（3rd 突破）** |
| 18 | T170 NN M7 | T87 NN M7 全数据重训 | **+34.64** | +0.20 |
| **21** | **T188v2 50+50** | **50 NN + 50 LGB mega ensemble** | **+35.64** | **+1.00（current SOTA）** |
| 22 | T192 H4 NNLS | bounded NNLS 权重 150+150 | **+31.0** | −4.64（OLS overfit）|

### 0.2 三次大突破对应的"质变"

| 突破点 | 提交编号 | 设计变化 | 平台增益 | 章节 |
|---|---|---|---|---|
| **1st 突破** | iter_013 | 把 LGB 从 3-class CE 换成 Δmid L2 regression + EV gate | +15.16 | [§12](#12-损失函数δmid-回归-vs-分类l2-vs-maehuber) |
| **2nd 突破** | iter_015 | 加入 T87 SPO+ DFL NN（与 LGB 集成）| +8.93 | [§8](#8-spo-dfl第二次大突破) |
| **3rd 突破** | iter_019 v2 | M7 LGB 全数据重训（330 iter, 不留 val） | +5.51 | [§7](#7-m7-全数据重训第三次大突破) |

其他 +0.20 ~ +1.00 量级的增益来自 conformal abstain、NN M7 retrain、ensemble 扩容。

### 0.3 最终提交包

- **文件**：`submission_050911_iter019_v2N_50plus50_optimized.zip`（148 MB，MD5=95993e6250ea56361dfecb62c9123e63）
- **CPU 备份**：`submission_050911_iter019_v2N_50plus50_cputorch.zip`（MD5=c67e4d25665a2a7a12f4f6839cc8d0fe）
- **平台分**：+35.64（max over 5 horizons，h=60 主力）
- **目录**：`final_submission_code/`（01_build_features + 02_train_lgb + 03_train_nn + 04_build_pkg）+ `pkg_T188v2_50plus50/`（运行时打包）

---

## §1. 评测协议硬约束（3 条）

来源 `CRITICAL_CONSTRAINTS.md`，平台官网原文复制。任何模型/特征/Predictor 设计必须严格遵守。

### 1.1 三条硬约束

1. **`date` 字段评测时被置 0**：date 不能当 feature、不能 derive 时段。
2. **测试点输入顺序被打乱**：Predictor 不能维护跨 batch state（LSTM hidden cache / per-batch 统计 / 增量 buffer）。
3. **`sym` 范围 0-4 但可能含训练外股票**：模型必须 sym-agnostic（不能 sym embedding / per-sym normalization / per-sym 模型）。

### 1.2 这三条如何反映在最终代码

| 约束 | 代码位置 | 实现 |
|---|---|---|
| `date` 不当 feature | `train_T188v2_lgb_seed.py:115`、`build_schemeP_cache.py:163` | cache 里有 date 字段但只用于 train/val/test split，不进 feature 列 |
| Predictor 无 state | `Predictor.py:316-365` | `pred_cache` 是 `predict()` 局部变量；每次调用完全独立 |
| sym-agnostic | `Predictor.py:294-304` | sym 只用于 conformal band 查询；模型 forward 不接收 sym |

### 1.3 多次"踩坑"实验印证约束

| 实验 | 触碰约束 | 结果 |
|---|---|---|
| T22 Alpha101/191 | #2（跨截面 batch shuffle）| LOSO IC +21.80 但**未提交**（违反约束）|
| T161 TTA K=5 | #2（per-batch 统计自适应）| holdout 下降，未提交 |
| T162 In-row TTT W3a/b/c | #2 | W3a −0.30, W3b −1.98, W3c −3.11，未提交 |
| T168 Adaptive threshold V1-V4 | #2 | 全负，未提交 |
| T169 AR TTT | #2 | 退步，未提交 |
| R_cross_sym | #2 + #3 | ABORT，未跑完 |

→ 失败案例反复印证 #2 是真实约束，不是建议。

### 1.4 reviewers 对约束合规的检验

`QREVIEW_FEATURES.md §17` (Q221-225)：白板 review 已逐项验证：
- ✓ B/Stage5 特征都是 per-window 计算，无跨窗口 contamination
- ✓ 没有全局统计量 fit 在 train 数据上
- ✓ cache emit `sym, date, sess_idx` 与 X 分离，无 leak
- ✓ Predictor 读 sym 仅用于 conformal band（line 20-21 有显式注释）
- ⚠ `vol_burst` 间接通过 volume regime 可能 proxy time-of-day（Q208/Q225）— 但未单独验证

**结论**：3 条硬约束在最终代码里**完全遵守**。这一点不像很多技术决策那样"是 trade-off"——这是**死线**，每一次违反约束的实验都未通过平台。

---

## §2. 评估方法论变迁（4 阶段）

### 2.1 演变速查

| 阶段 | 时间 | eval 方式 | 平台对应 | 备注 |
|---|---|---|---|---|
| Phase 0 | T1-T2 | IID split (同 sym 内) | — | 严重乐观；IID +19.25 → LOSO −22.10 |
| Phase 0-1 | T1-T35 | **LOSO 5-fold**（留一 sym）| 透传低 | 过度惩罚 sym OOD |
| Phase 1 | T23 | LOSO → 2-anchor 平台校准 | 透传 ~0.70 | 建立 LOSO → 平台映射 |
| Phase 2 | T44-T74 | **LOSO-equiv**（全 sym 训练 + DE 在 442k 行）| 透传 53-71% | 方法论变化贡献 +7 数字（去除 LOSO 过度惩罚）|
| Phase 3 | T58/T64 | + **V4 walk-forward**（train 0-79 / val 80-95 / test 96-119）| 当前最干净 | LOSO-eq ~40 ↔ 平台 28-35 |
| Phase 4 | T156+ | **holdout in-sample**（0-119 全训练 → 96-119 评分）| **绝对值不可信** | 只看相对 delta |

### 2.2 为啥要切换 in-sample（M7 trick 之后的"无可奈何"）

`HISTORY_EVIDENCE_DB.md` 主题 9 + 附录 A 发现 1：

- M7 全数据重训之后，没有干净的 V4 OOF 可用（所有数据都进了训练）
- holdout in-sample 的 delta 仍然有信号（T188v2 vs T170 delta +0.48 → 平台 delta +1.00）
- 但绝对值灾难性高估（T170 holdout +149.95，平台 +34.64）

**4 个 in-sample 污染陷阱案例**（CODE_QUESTION_LIST Q76、QREVIEW_NN_TRAIN Q92、QREVIEW_FEATURES Q8/Q9）：

| 实验 | 本地 holdout | 平台 | 教训 |
|---|---|---|---|
| T180 OOD GMM abstain | −32 ~ −78 | 未提交 | 测试集比训练 in-distribution |
| **T182 无 conformal** | **+6.62**（看似 conformal hurt）| **−0.30**（conformal 实际有用）| in-sample artifact 经典案例 |
| T172 GroupTransformer | in-sample +149.91 | 未提交 | 全 in-sample 污染 |
| **T192 bounded NNLS** | val +180.41, test +193.94, ratio 1.075 | **+31.0** (−4.64) | val+test 均 in-sample，OLS overfit |

**最关键的 T182 案例**：去掉 conformal 后本地 +6.62（看起来 conformal 是负担），平台 −0.30（conformal 实际有效）。证明本地 holdout 和平台 OOD 可以反向，**只能用平台为最终裁判**。

### 2.3 LOSO-equiv → 平台 透传系数表

| iter | LOSO-eq | 平台 | 透传 % |
|---|---|---|---|
| iter_013 | +36.23 | +19.23 | **53%** |
| iter_015 | +40.09 | +28.16 | **71%** |
| iter_018 | +41.49 | +28.93 | **70%** |

→ 透传系数在 53-71% 区间方差较大；DE 越激进透传越低。这是 [§3](#3-阈值与决策层thr_up--thr_dn--ev-gate) 阈值不再 DE 调的根源。

### 2.4 评估方法论选择的 GAP

| GAP | 说明 |
|---|---|
| 方法论切换无重叠 | Phase 1→2→3→4 切换时没有"同一模型在新旧两套方法论下双跑"，所以 +7 方法论 inflation 是事后推估 |
| Holdout in-sample 没有 untainted baseline | 一旦切到 M7 + in-sample，**没办法**在本地做干净 OOD eval；只能靠平台投币 |
| 透传系数 53-71% 方差太大 | 没建立可靠的 "本地数字 → 平台数字" 转换公式；只能"如果本地涨大则平台可能涨小，如果本地涨小则可能持平甚至跌" |

---

## §3. 阈值与决策层（thr_up / thr_dn / EV gate）

### 3.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| `thr_up` (h=60) | **0.0003** | `04_build_pkg/thresholds.json:46` |
| `thr_dn` (h=60) | **0.000216** | `04_build_pkg/thresholds.json:47` |
| `w_nn` | **1.0** | `thresholds.json:48` |
| `w_lgb` | **1.5** | `thresholds.json:49` |
| EV gate | `pred > thr_up → 2 (买); pred < −thr_dn → 0 (卖); else 1 (平)` | `Predictor.py:255-271` |

不分 sym，全局固定。所有后续大版本（iter_018/019/T170/T188v2/T190）都不动这两个值。

### 3.2 白板问题汇总（去重合并后）

| 原 Q | 问题核心 |
|---|---|
| CODE Q1, Q2 / PREDICTOR Q1-Q5 | 为啥 thr_up/thr_dn 非对称？为啥不是更整的 0.0002 / 0.00015？|
| CODE Q3 / PREDICTOR Q41-Q50 / Q166 | w_nn=1.0 / w_lgb=1.5 怎么定的？为啥 LGB 权重更高？|
| CODE Q104 / PREDICTOR Q21 | 阈值是否做过 grid search？|
| CODE Q102 / PREDICTOR Q12 | EV gate 硬阈值是否应该加 hysteresis？|

### 3.3 决策证据链（thr_up=0.0003, thr_dn=0.000216）

**演变时间线**（HISTORY_EVIDENCE_DB.md 主题 1）：

1. **T5a (SchemeB)**：DE 搜索非对称 thr_up/thr_dn 首次显示有效（LOSO h_60 +11.11，vs T2 原始 −22.10）
2. **T30 (DE 4D)**：DE 4D 阈值优化（thr_up/thr_dn/margin/weight）→ aug_a 5-seed LOSO +11.46 → +13.61。**首次出现 DE 在 5-fold LOSO 上可超调的 warning**
3. **T62 (k-threshold)**：无偏对称 k×mean_abs_pred，k=1.25 时 LOSO-eq +33.72（vs DE +36.23，鲁棒但低 2.51）。证明对称阈值可行但绝对值低。
4. **T75 (iter_013)**：LGB regression + DE 阈值 → LOSO-eq +36.23 → **平台 +19.23**（透传 53%）。**第一次大突破**，但同时 DE → 平台 gap −17 证明 DE 有过拟合。
5. **T76 (decoupled)**：thr_up ≠ thr_dn 分开搜 → **最终采用 thr_up=0.0003, thr_dn=0.000216**
6. **T126 (per-sym OOF threshold)**：每只股票独立 OOF 最优。**LOSO 本地高，但过拟合风险极高，未提交**
7. **iter_016 v3**：4-way ensemble 上再做 OOF DE 4D → LOSO +43.71，**平台 −3.08**（DE 灾难性过拟合）
8. **T146 (conformal-aware re-DE)**：在 v3_mae 上重 DE → LOSO +41.20 → +43.96，**未提交平台**（教训沉淀够多了）
9. **T168 (adaptive threshold V1-V4)**：per-batch 统计自适应 → **全负**（触碰硬约束 #2）

### 3.4 为啥非对称？为啥不是更整的数？

- **T5a 起源**：DE 在 SchemeB 上首次发现非对称 thr 有效
- **T76 固化**：T75 LGB regression 之后，DE 搜出 thr_up=0.0003 / thr_dn=0.000216 这对值
- **比例 thr_dn/thr_up ≈ 0.72**：CODE Q1/Q2 怀疑是 `thr_dn = thr_up × 0.72` 这种比例关系——实际是**独立 DE 搜出来的，巧合接近 0.72**
- **不是 0.0002/0.00015 这种"更整"数**：因为 DE 直接给的是 0.0003/0.000216，作者没主动 round
- **物理含义**：thr_up > thr_dn 表示做多门槛比做空门槛高 → 与做空成本/借券成本相符（虽然平台规则未公开是否反映此）

### 3.5 为啥不再做 DE / OOF threshold？（5 重证据）

| 证据 | 说明 |
|---|---|
| iter_013 透传 53% | DE 在 LOSO 上拿 +36.23，平台只 +19.23 → 17 分差距是 DE 过拟合 |
| iter_016 v3 灾难 | 4-way DE 平台 −3.08（vs iter_015 +28.16） |
| T126 per-sym OOF | LOSO 高但**未提交**（已知会过拟合）|
| T146 conformal-aware re-DE | LOSO +43.96，**未提交**（保守原则）|
| T168 adaptive thr | 全负（触碰约束 #2）|

**保守原则**（HISTORY_EVIDENCE_DB.md 主题 1）：thr_up/thr_dn 一旦定下来，所有后续大版本（iter_018 / 019 / T170 / T188v2 / T190）都**不再动**。

### 3.6 w_lgb = 1.5 / w_nn = 1.0 怎么定的

- **起源**：iter_015 阶段 SPO+ NN + LGB 集成时 grid search 出 w_lgb > 1 更好，固化未变
- **T188v2 50+50 后验证**：在本地 val 上比 1:1 高 +0.几（具体数字未单独记录）
- **CODE Q3 / PREDICTOR Q41-50 提的问题**："为啥不在 holdout 上学一个 ridge weight？"——答案见 [§6.5](#65-为啥不学权重simple-mean--ols--nnls) NNLS 灾难 T192

### 3.7 EV gate 是否应该加 hysteresis？

- **CODE Q102 提的问题**：pred 在阈值附近抖动时 action 频繁切换，手续费高
- **现状**：硬阈值，无 hysteresis
- **GAP**：没有专门 ablation 比"硬阈值 vs 双阈值带（如 buy_on=0.0003, buy_off=0.0002）"。原因：单次 predict 是独立的（约束 #2），跨 batch 维护 "上次 action" 不可行 → hysteresis 在评测协议下天然不可实现。
- **结论**：约束 #2 间接决定了硬阈值是唯一可行设计

### 3.8 被拒绝的替代方案

| 方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| per-sym OOF threshold | T126 | LOSO 高 | 未提交 | 过拟合 OOF 分布；OOD sym 无 model |
| 4-way OOF DE | iter_016 v3 | LOSO +43.71 | **+25.08** | DE 在 442k 测试集严重过拟合 |
| conformal-aware re-DE | T146 | +2.76 | 未提交 | 二次 DE 风险叠加 |
| adaptive per-batch thr | T168 V1-V4 | 全负 | 未提交 | 触碰约束 #2 |
| 对称 k×mean_abs_pred | T62 (k=1.25) | +33.72 | — | 鲁棒但绝对值低 2.51 |
| isotonic / Platt 校准 | T15/T40/T131 | −1.02 / 持平 | — | 破坏排序质量 |

### 3.9 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| thr_up/thr_dn 具体值 0.0003/0.000216 | T76 DE 搜出，无完整 grid search | 低（多次小动 ablation 均未超此组合）|
| 不分 sym 阈值 | T126 per-sym 失败明证 | 低 |
| w_nn=1.0 / w_lgb=1.5 ratio | iter_015 grid search，T192 NNLS 失败 | 中（可能某些 horizon 上 1:1 略优）|
| EV gate 硬阈值（无 hysteresis）| 没做（约束 #2 阻断）| 不可改善 |
| thr 不为短 horizon 单独调 | 见 [§5](#5-5-horizon-share_with60--sqrt-缩放) | 高（USER-RAISED Q281）|

---

## §4. Conformal abstain（per-sym β 与 σ）

### 4.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| per-sym β | `{sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}` | `thresholds.json:56` |
| per-sym σ | `{0: 0.00024, 1: 0.00047, 2: 0.00045, 3: 0.00038, 4: 0.00041}` (近似) | `thresholds.json:57` |
| `default_beta_for_ood` | **0.16** | `thresholds.json:58` |
| `default_sigma_for_ood` | **0.0004** | `thresholds.json:59` |
| 弃权公式 | `|EV_pred| < β_sym × σ_sym → action=1（不出手）` | `Predictor.py:255-262` |

### 4.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q5 / PREDICTOR Q22-Q24 | 为啥 sym3/4 完全不 abstain（β=0），sym1 最激进（β=0.4）|
| CODE Q6 / Q103 / PREDICTOR Q19-Q20 | per-sym sigma 是什么？是 OOF |pred-y| std 还是 pred std？|
| CODE Q7 / Q109 / PIPELINE Q93 | OOD default 0.16 / 0.0004 怎么定的？|
| CODE Q107 / PREDICTOR Q73-75 | 命名 "conformal" 是否名不副实？|
| CODE Q105 | 没用 isotonic/Platt 校准 |

### 4.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 3）

| 实验 | 方法 | 本地 | 平台 |
|---|---|---|---|
| T15 | isotonic + EV gate | LOSO h_10 +21.86 → +20.84 (**−1.02**) | — |
| T40 | platt scaling | LOSO 持平/负 | — |
| R13 | calibration + position sizing 调研 | isotonic/platt 均负 | — |
| T126 | per-sym OOF threshold | LOSO 本地高 | 未提交（过拟合）|
| **T127** | **iter_018 v1 = iter_015 + per-sym β conformal** | LOSO +39.75 → **+41.49** | **+28.93 (+0.77 vs iter_015)** |
| **T150** | **conformal β sweep（4-fold CV consensus）** | β∈[0, 0.5] 每 sym 独立选 | 确定最优 β |
| T159/b | per-sym dynamic abstain | holdout null | 无效 |
| **T182** | **去 conformal ablation** | holdout +6.62（看似 conformal hurt）| **−0.30**（conformal 实际有用）|
| T180 | OOD GMM density abstain | V1 −32, V2 −57, V3 −78 | 未提交 |
| T181 | time-window hard abstain | V1 −23.39, V2 −0.27 | 未提交 |

### 4.4 per-sym β 怎么 sweep 出来的（T150）

- **方法**：4-fold CV consensus 在 β ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} 网格上，每 sym 独立选最佳
- **CV consensus 而非单 fold OOF**：避免 T126 的过拟合教训
- **每个 sym 的合理性**：
  - sym0: β=0.10（轻微 abstain）
  - sym1: β=0.40（最激进 abstain — 历史上预测稳定性最差）
  - sym2: β=0.30（蓝筹/ETF，T8 诊断 amount_delta z-score +7.49 偏离，模型 brittle）
  - sym3: β=0.00（模型在这只上预测最稳，abstain 反损 PnL）
  - sym4: β=0.00（同 sym3）

### 4.5 σ 数值的来源

- **CODE Q6 / Q103 提的问题**：sigma 是什么？OOF 残差 std？pred 本身 std？
- **现状**：写死在 thresholds.json 里，**代码里没有 calibration script**
- **GAP**：流程不透明——离线 calibration 的脚本/数据/阶段没在 final_submission_code 里
- **推测**：来自 5-NN ensemble 在 val/test 上预测的 std（per-sym 计算）
- **PREDICTOR Q107 提的命名争议**：因为 sigma 是常数标量（不是 batch 内动态分位数），严格说不是 "conformal" 而是 "per-sym mean-deviation gate"。但项目里沿用 "conformal" 命名。

### 4.6 OOD default (β=0.16, σ=0.0004) 怎么定的

- **CODE Q7 / Q109 / PIPELINE Q93 提的问题**：是 mean(per_sym_beta)≈0.16 吗？
- **答案**：**是的，就是均值**
  - mean({0.10, 0.40, 0.30, 0.00, 0.00}) = 0.16
  - mean(per_sym_sigma) ≈ 0.00042，rounded 0.0004
- **R_T75L2_advval / R_conformal_select 调研**：讨论过用全局 β=0.20 兜底，最终选 mean 作为不偏估计
- **GAP**：CODE Q109 指出"OOD 默认 band (0.16 × 0.0004 ≈ 6.4e-5) 比 thr_up=3e-4 小得多（~20%）"。意味着 OOD sym 上只比"无 abstain"略保守，**没有更激进 abstain OOD sym 的设计**。原因：不知道平台 OOD sym 上模型表现如何，不敢硬关。

### 4.7 关键 ablation T182（"本地负、平台正"的悖论）

| 指标 | T170 (含 conformal) | T182 (无 conformal) | Δ |
|---|---|---|---|
| holdout in-sample | +149.95 | **+156.56** | **+6.62**（conformal 看似负担）|
| 平台 | **+34.64** | **+34.34** | **−0.30**（conformal 实际有效）|

**机制解释**：
- 本地 holdout 数据是 in-sample（96-119 已在训练）→ 模型在这部分高度准确，abstain 反而损失真实 PnL
- 平台 OOD 数据是 out-of-sample → 模型在不确定区域产生噪声预测，abstain 过滤掉这些错误预测
- 这是 in-sample 评测**无法预测**平台行为的经典案例

**结论**：conformal 在 OOD 上稳定贡献 +0.3 ~ +0.77（iter_018 vs iter_015 +0.77；T170 vs T182 +0.30），但本地数字会说反话。

### 4.8 被拒绝的替代方案

| 方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| isotonic + EV gate | T15 | −1.02 | — | 破坏排序 |
| platt scaling | T40 | 持平/负 | — | 同上 |
| per-sym OOF threshold | T126 | LOSO 高 | 未提交 | 过拟合 |
| per-sym dynamic β | T159/b | null | — | 动态 β 不稳 |
| time-window hard abstain | T181 | V1 −23.39 | 未提交 | T170 conformal 已过滤；硬弃权适得其反 |
| OOD GMM density abstain | T180 | V1 −32, V3 −78 | 未提交 | 测试集比训练 in-distribution；低密度反高 PnL |

### 4.9 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| per-sym β 具体值 | T150 4-fold CV sweep | 低 |
| per-sym σ 来源 | **代码里无 calibration script**，离线值写死 | 中（流程不透明，复现困难）|
| OOD default 用 mean | 无 ablation（推理选择）| 低（平台 OOD sym 表现未知）|
| "conformal" 命名不严谨 | 不影响功能 | 低（只是命名）|
| h=5 时 band > threshold（PREDICTOR Q23）| 无 ablation | **中**（hi-band sym 在短 h 上几乎抵消 sqrt 缩放，见 §5）|

---

## §5. 5-horizon share_with=60 + sqrt 缩放

### 5.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| `HORIZONS` | (5, 10, 20, 40, 60) | `Predictor.py:43` |
| `share_with` | 5/10/20/40 共享 h=60 ensemble 预测 | `thresholds.json:5,15,25,35` |
| thr 缩放 | `thr_h = thr_60 × sqrt(h/60)` | `thresholds.json:7-42` |
| 推理 cost | 1× h=60 ensemble 推理（不是 5×）| `Predictor.py:329-358` (pred_cache) |

### 5.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q19 / Q21 / Q121-126 | sqrt 缩放理论依据？为啥 share_with 而非各 horizon 独立训？|
| CODE Q123 | 短 horizon 提交是否会拉低分数？|
| PREDICTOR Q6 / Q7 / Q8 | sqrt(t) 缩放推导？平台 score 是 max 还是 sum？|
| **PREDICTOR Q281** ⭐ | **用户精确推导：sqrt(h/60) 缩放方向错误（差 12 倍）** |

### 5.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 10 + 附录 A 发现 5）

#### 5.3.1 短 horizon 的"平台灾难"

iter_002 关键发现（导致整个项目专注 h_60）：

| Horizon | LOSO 本地 (iter_002) | 平台 |
|---|---|---|
| h_5 | +17.49 | **−7.68** |
| h_10 | **+21.86** | **−8.64** |
| h_20 | +19.70 | −5.09 |
| h_40 | +11.71 | +2.02 |
| h_60 | +6.30 | **+4.07** |

→ LOSO 短 h 看似最强，平台短 h 全负 → **整个项目专注 h_60**

#### 5.3.2 推理优化加速

| 实验 | 方法 | 加速比 |
|---|---|---|
| T61 (iter_010) | batch-vec 特征提取 | 254 min → 4.5 min（**58×**）|
| T188v3 | 50 NN sequential numpy → batched torch CUDA bmm | 1509ms → 8.1ms（**186×**）|
| T188v3_fullhorizon | 5 horizon share h=60 | 5× cost → 1× cost |

### 5.4 设计意图（项目作者视角）

- α=0.05%（h_5/10）被手续费 0.02% 双边摧毁
- α=0.1%（h_60）才有 alpha 空间
- 短 horizon 模型如果重训，IC 反而低（模型对 60-tick Δmid 优化好，对 5-tick Δmid 不一定）
- **决定**：所有 horizon 都用 h=60 ensemble 输出，按 sqrt(H/60) 缩放阈值
- 平台规则（READ ME 引用）："score = max(per-horizon PnL)" → 短 horizon 提交不会拉低 max

### 5.5 sqrt(H/60) 缩放的"传统"理由

- Brownian motion 假设下，`std(Δmid over h ticks) ∝ sqrt(h)`
- 所以 h=5 的"显著性"标准 ≈ sqrt(5/60) × h=60 的标准
- 阈值按 sqrt(H/60) 缩放 → 所有 horizon 用统一"显著性"门槛

### 5.6 ⚠️ USER-RAISED Q281：sqrt 缩放方向是错的（详见 §18）

这是整个项目最严重的"数学正当性"问题，专门在 [§18.1](#181-q281--sqrth60-阈值缩放方向错误用户精确推导) 单独展开。

**核心 takeaway**：
- 模型预测的是 **60-tick** Δmid（不是横跨 h 的 Δmid）
- 若想用 pred_h60 给 h=5 决策，正确做法是 **pred 缩 h/60（线性）** 而非 **thr 缩 sqrt(h/60)**
- 代码做法 (`thr_60 × sqrt(h/60)`) 与"理论正确"差 `60/h` 倍（h=5 时差 **12 倍**）
- 现状能 "work" 是因为 max(per-horizon) 规则让短 h 不影响主分

### 5.7 为啥不为短 horizon 训独立模型？

- **历史教训**（iter_002）：短 horizon 模型 LOSO 看似强，平台全负 → 训独立模型也救不了 alpha=0.05% < fee=0.02%
- **iter_003/004**：试过短 h 严格阈值/关闭 → 未获平台正分
- **项目专注 h_60**：所有特征工程、SPO+ 调参、ensemble 设计都围绕 h_60

### 5.8 短 horizon 平台分数实际情况

最终 T188v2 提交时，5 个 horizon 的具体平台分数没有完整记录（README 只列总分 +35.64 = max）。从历史规律推测：
- h_60: +35.64（主力）
- h_5/10/20/40: 多数 < 0 或接近 0（受 fee 摧毁 + sqrt 缩放方向问题加剧）

### 5.9 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 共享 h=60 ensemble | iter_002 平台短 h 灾难，无独立训 | 低（独立训也不会更好）|
| sqrt(H/60) 缩放方向 | **无 ablation** | **极高**（Q281：数学方向反了，但因 max 规则隐藏）|
| 平台 score = max 假设 | README 声称但**未在 spec 中明文验证** | 中（若 score = sum 则短 h 会拖累）|
| 短 horizon 无 conformal 抵消推证 | PREDICTOR Q23 指出 hi-band sym 几乎抵消 sqrt 缩放 | 中 |
| 短 horizon 也用 w_nn=1.0/w_lgb=1.5 | CODE Q126 | 低 |

---

## §6. Ensemble 组成与权重（50+50, w_nn=1.0, w_lgb=1.5）

### 6.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| NN ensemble size | **50**（5 LR × 4 Dropout × 3 Batch 的 cycling，2 seeds，实际 12 unique combos × ~4 repeats）| `train_T188v2_nn_seed.py:186-188` |
| LGB ensemble size | **50**（5 HP × 10 seeds）| `train_T188v2_lgb_seed.py:25-31, 83` |
| NN 内组合 | simple mean over 50 | `Predictor.py:177` |
| LGB 内组合 | simple mean over 50 | `Predictor.py:357` |
| 跨族权重 | w_nn=1.0, w_lgb=1.5 | `thresholds.json:48-49` |
| 异构模型 | **无**（无 CatBoost / GRU / Transformer）| — |

### 6.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q8 / Q95-101 / PREDICTOR Q41-Q70 | 为啥 50+50？为啥不学权重？|
| CODE Q13 / Q95 / NN Q8 | HP cycling 实际只覆盖 12 个独立 combo |
| CODE Q97-99 / PREDICTOR Q63 | Family-mean 后再加权 vs Model-level mixed weight |

### 6.3 ensemble 扩容的"非单调性"（HISTORY_EVIDENCE_DB.md 主题 2）

| 扩容步骤 | 本地 holdout delta | 平台 delta |
|---|---|---|
| 5+5 → 40+5 (T179, 只扩 NN) | +0.63 | **−0.25** |
| 5+5 → 50+50 (T188v2, NN+LGB 都扩) | +0.48 | **+1.00** |
| 50+50 → 150+150 (T190) | −9.81 | **−1.05** |

**三个关键洞察**：
1. **只扩一侧无用**（T179）：variance reduction 必须两侧对称
2. **5→50 涨 +1.00**（T188v2）：两侧都扩有效
3. **50→150 跌 −1.05**（T190）：扩容收益**非单调**，过多 HP 引入 bias

### 6.4 50 个 NN 实际只有 12 unique combos（CODE Q13 / NN Q8）

```python
lr_p1 = LR_LIST[(S-1) % 3]
dropout = DROPOUT_LIST[(S-1) % 4]
batch_p1 = BATCH_LIST[(S-1) % 3]
```

- LR 和 BATCH 都用 `(S-1) % 3` → **完全同步**！seed 1→(LR=1e-4, BATCH=2048)，seed 4→(LR=1e-4, BATCH=2048)，seed 7→(LR=1e-4, BATCH=2048)
- lcm(3,4,3) = 12，所以 50 seeds 只覆盖 12 个独立组合（每个 combo 重复 ~4.17 次）
- README 写"3×4×3=36 combos"，**实际是 12 combos**
- **是 bug 还是有意？**：作者没回应。HISTORY_EVIDENCE_DB.md 主题 2 表明每个 combo 4 次重复提供 init/dropout RNG 多样性 —— 但确实**没有覆盖完整 36 网格**

### 6.5 为啥不学权重？（simple mean vs OLS vs NNLS）

#### T192 bounded NNLS 灾难

最后一次尝试学权重：T192 用 bounded NNLS 在 [1/1500, 2/150] per-group 权重学 150+150 ensemble：

| 指标 | T192 NNLS | T188v2 simple mean |
|---|---|---|
| val holdout | +180.41 | — |
| test holdout | +193.94 | +150.43 |
| **平台** | **+31.0** | **+35.64** |
| **Δ** | **−4.64** | — |

**为啥失败**（HISTORY_EVIDENCE_DB.md 附录 A 发现 8）：
- NN-NN 平均相关系数 **0.9331**
- LGB-LGB 平均相关系数 **0.8433**
- 设计矩阵 condition number **54,182**（极病态）
- 任何"最优"权重都是 in-sample 噪声

**前序失败**：T139 stacker NN（NN 学综合 LGB+NN）LOSO 负面 → 同样 OOF 过拟合

**结论**：simple mean 是"在病态共线性下的最优鲁棒选择"。

### 6.6 没有 CatBoost / GRU / Transformer？

| 替代方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| 4-way (T87+T89+T99+T95) | iter_016 v3 | LOSO +43.71 | **+25.08** | GRU 主因失败；OOF DE 过拟合 |
| 第二组 NN T97 (同 T87 风格) | T163 | holdout +42.90 | **~+30.74** | T97/T87 同架构 diversity≈0 |
| 4-way + T97 | T164 | +0.34 | 未提交 | 仍含 T97 |
| 5-way (+ CatBoost) | T167 | +0.11 over T166 | 未提交 | within noise |
| CatBoost L2 5-seed | T89/T166 | corr 0.77 / +0.39 | iter_016 失败 | 与 LGB 高共线 |
| GRU | T95 | 微小提升 | iter_016 失败主因 | hidden state 过短 h 过拟合 |
| Transformer (GroupTr) | T172/T187 | in-sample +22~55 (污染) | 未提交 | OOD 风险 |
| Deep CNN | T184/T185 | in-sample 看似好 | 未提交 | OOD 未验证 |
| Snapshot ensemble | T135 | 边际 | 未提交 | 不超 5-seed |
| GroupTransformer add | T187 Plan B | in-sample +12.6/14.7 | 未提交 | 全 in-sample 污染 |

### 6.7 w_lgb=1.5 / w_nn=1.0 来源

- **iter_015 阶段**：SPO+ NN 和 LGB 集成时 grid search 出 w_lgb > 1 更好
- **固化**：固定为 1.0:1.5，没在 50+50 阶段重新 sweep
- **GAP**：CODE Q126 提到所有 horizon 都用 1.0/1.5 加权，没有 per-horizon 调整

### 6.8 NN/LGB family-mean 后再加权（vs model-level mixed weight）

`Predictor.py:345-358`：
```python
pred_nn = mean(50 NN)
pred_lgb = mean(50 LGB)
pred = (w_nn × pred_nn + w_lgb × pred_lgb) / (w_nn + w_lgb)
```

- **等价于 model-level**：每个 NN 权重 = w_nn/50/(w_nn+w_lgb)，每个 LGB 权重 = w_lgb/50/(w_nn+w_lgb)
- **不是 ridge/stacking**：CODE Q97 / Q101 提到这点
- **拒绝 stacking 理由**：T139 stacker NN 失败 + T192 NNLS 失败

### 6.9 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 50+50 大小 | T179 (40+5)、T188v2 (50+50)、T190 (150+150) 三点 ablation | 低 |
| NN HP cycling 实际 12 combos（非 36）| 无意识到的设计/bug | 中（可能 36 完整 grid 更优）|
| Simple mean（不学权重）| T192 NNLS 失败明证 | 低 |
| w_lgb=1.5 ratio | iter_015 grid search，未在 50+50 复 sweep | 中 |
| 异构模型零容忍（拒绝 CatBoost/GRU/Transformer）| 历史 ablation 全负 | 低 |
| short horizon 也用相同权重 | 无 ablation | 中 |

---

## §7. M7 全数据重训（第三次大突破）

### 7.1 最终决策

| 模型 | 重训配置 | 位置 |
|---|---|---|
| LGB | date 0-119 全数据，**固定 iters = 330**，不留 val | `train_T188v2_lgb_seed.py:72, 127` |
| NN | date 0-119 全数据，**固定 epoch = 11**，不留 val | `train_T188v2_nn_seed.py:46, 387-394` |
| 沿用 V4 阶段 HP | learning_rate, num_leaves, max_depth 等全照搬 | 同上 |

### 7.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q9 / LGB Q8 | LGB 330 这个数怎么定的？|
| CODE Q10 / NN Q13 | NN 11 epoch 怎么定的？|
| CODE Q76 / Q79 / NN Q90-Q92 | 全数据训不留 val 是否过拟合？|

### 7.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 4）

| 实验 | 方法 | 本地 | 平台 |
|---|---|---|---|
| R_full_retrain | M7 full-retrain 可行性研究 | LGB 量化 330 iters 早停一致 | — |
| T128 | conformal v2 + M7 LGB 可行性 | LOSO 不变 | — |
| **T140/b/c** | **iter_019 v2 = iter_018 v1 + LGB M7** | LOSO 不变 | **+34.44 (+5.51 vs iter_018)** ⭐ |
| T155 | v2 replica reproducibility | LGB 确定性确认 | — |
| **T170** | **T87 NN M7 retrain（5 seeds, ep=11）** | holdout v2N +118.16 (+8.49) | **+34.64 (+0.20)** |
| T176/b | T87M7 HP variant sweep | 各 variant 149-151 噪音 | S1_ep15 +34.51 (−0.13) |
| T183 | NN random init（无 T81 warm）M7 | holdout +0.63 | — |

### 7.4 LGB 330 iters 怎么定的

- V4 walk-forward（train 0-79 + val 80-95 早停）5 seeds 平均 best_iter ≈ **300**
- × 1.1 安全系数 = **330**
- 量化测试（T128 + R_full_retrain）：330 vs 300 iters 上 train loss 差距小（GBDT 在大数据上单调收敛）
- T140 平台 **+5.51** 直接验证

### 7.5 NN 11 epoch 怎么定的

- V4 walk-forward 5 seeds 平均 best_epoch ≈ **9.2**
- × 1.1 ≈ 10.1，向上取整 = **11**
- T176 sweep（ep ∈ {7, 11, 15, 20}）confirm ep=11 优：
  - S1_ep15 平台 +34.51（−0.13 vs T170 +34.64）
  - 其他 variant holdout 149-151 噪音级，无单一显著更优

### 7.6 不留 val 集的依据（CODE Q76）

| 依据 | 说明 |
|---|---|
| 浪费样本 | 留 val = 浪费 12% 训练样本 |
| 96-119 是 OOD 最相近时段 | 96-119 是 platform 测试数据最近时段，更值得用作训练 |
| Kaggle 标准做法 | 先 CV 选超参，最后 retrain on all data without holdout |
| T140 LOSO 不变 | M7 只改全数据模型，不影响 LOSO eval；但平台 +5.51 → 直接证 retrain on all data 比留 val 优 |

### 7.7 为啥 M7 增量 LGB > NN？

| 模型 | M7 平台增量 |
|---|---|
| LGB M7 | **+5.51**（iter_018 → iter_019 v2）|
| NN M7 | **+0.20**（iter_019 v2 → T170）|

**原因**：
- NN 已从 T81 L2 warm-start，加上 SPO+ DFL → M7 增量小
- LGB 从头训，M7 带来的新数据增量大
- T183 证明：NN random init M7 与 T81 warm-start M7 在 holdout 持平（+0.63）→ warm-start 相对增益已饱和

### 7.8 被拒绝的替代方案

| 方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| MAE LGB M7 | T145/b | 预测 +30~32 | 未提交 | 不超 L2 M7 +34.44 |
| 新架构 M7 (Transformer) | T172/T187 | in-sample +22~55 (污染) | 未提交 | 全 in-sample，OOD 风险高 |
| Hawkes 特征 M7 | T142 | 边际 | 未提交 | Hawkes 增量本身边际 |
| 留 val 集 | (早期 V4) | LOSO-eq +26.44 | 未在 M7 阶段做 | M7 设计就是不留 val |

### 7.9 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| LGB 330 iters | 来自 V4 best_iter × 1.1，无对照（200/500）| 低（GBDT 单调收敛，量化安全）|
| NN 11 epoch | T176 ablation ({7,11,15,20}) | 低 |
| 不留 val | 设计前提，无对照 | 低（M7 增量 +5.51 已证明）|
| 各 HP 沿用同 330/11 | num_leaves=63 vs 255 应该用不同 iters？没做 | 低（10 seeds × 5 HP 多样性已够）|
| 新架构 M7（Transformer/CNN）| 全 in-sample 污染，OOD 未验证 | 中（保守原则下不冒险）|

---

## §8. SPO+ DFL（第二次大突破）

### 8.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| Phase 1 | L2 regression pretrain on Δmid，11 epoch (V4) | `train_T188v2_nn_seed.py:227-365` |
| Phase 2 | SPO+ DFL fine-tune，**λ=30, lr=3e-5**，11 epoch | `train_T188v2_nn_seed.py:46-48, 443-477` |
| 联合 loss | `L = MSE + λ × SPO+`（λ=30）| `train_T188v2_nn_seed.py:462` |
| 50 NN seeds | 每个 seed 独立两阶段训 | `run_all_nn_seeds.sh:20` |

### 8.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q11 / NN Q14-Q15 | λ=30, lr=3e-5 怎么定的？|
| CODE Q42 / NN Q40 | 两阶段 vs 直接 SPO+ |
| CODE Q43 / Q174 / NN Q41 | SPO+ DFL vs 直接 PnL loss / Fisher consistency |
| CODE Q44 / NN Q42 | L2 + λ·SPO+ 联合 vs 纯 SPO+ |

### 8.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 5）

| 实验 | 方法 | 本地 | 平台 |
|---|---|---|---|
| T1 | DeepLOB-style 3-class CE NN | LOSO ~+8 | — |
| **T13** | **PnL-aware 端到端 NN loss**（直接 PnL 梯度反传）| **不收敛** | — |
| R10 | PnL-aware loss 10 方案调研 | SPO+ 最稳 | — |
| T57 | PnL-aware loss（早期）| 不稳 | — |
| T60 | RL NN PnL | 探索性 | — |
| **T75** | **LGB regression Δmid + EV gate** | LOSO-eq +36.23 | **+19.23** ⭐ |
| **T81** | **NN L2 regression on Δmid（MLP warm-start）** | LOSO-eq +35.89 | — |
| **T87** | **SPO+ DFL NN fine-tune** | LOSO-eq NN+LGB +40.09 | **+28.16** ⭐ |
| T116 | GMADL direction-aware loss | LOSO 微弱 | — |
| T132 | quantile LGB | LOSO 边际 | — |
| **T134** | **direct PnL NN loss（无 SPO+）** | **不稳定，不收敛** | — |
| **T136** | **SPO+ 超参 sweep（λ, lr）** | 确认 λ=30, lr=3e-5 最优 | — |
| T149 | T87 OOD sym 鲁棒性 | OOD ok | — |
| T170 | T87 NN M7 retrain | holdout +8.49 | **+34.64** |

### 8.4 为啥两阶段（pretrain + fine-tune）？

- **T13/T134 反例**：直接 PnL loss 不收敛
  - PnL 公式 `PnL = z × (Δmid - fee × sign(z))` 是 z=0/1/2 三段不连续函数
  - 裸梯度方向不一致，loss 跳跃
- **T81 L2 pretrain**：提供稳定 representation 起点（NN alone LOSO-eq +35.89 已经很高）
- **T87 SPO+ fine-tune**：在 representation 已对齐 Δmid 幅度的基础上，用 SPO+ 的 surrogate gradient 把决策层（EV gate）的损失对齐到 PnL
- **R10 调研**：SPO+ 在 10 个 PnL-aware loss 方案中**唯一稳定**

### 8.5 λ=30, lr=3e-5 怎么 sweep 出来的

**T136 SPO+ arch sweep**：
- λ ∈ {1, 10, 30, 100, 300}
- lr ∈ {1e-5, 3e-5, 1e-4, 3e-4}
- 最优 (λ=30, lr=3e-5) 在 LOSO-eq 上比次优高 +0.3-0.5

**T176 M7 阶段重新扫描**：
- λ ∈ {10, 30, 100}
- confirm λ=30 仍最优
- 其他 variant 在 ensemble 中提供多样性（T188v2 50 NN 实际只覆盖 12 combo，主因 HP cycling bug，不是为 ensemble 多样性故意造的）

### 8.6 L2 + λ·SPO+ 联合（非纯 SPO+）

- **"锚定"作用**：L2 项防止 SPO+ 把 pred 跑到完全脱离 Δmid 幅度
- **λ=30 相对量级**：L2 是 squared scale（量级 ~1e-7），SPO+ 是线性 scale（量级 ~1e-4），所以 λ=30 让两者 contribution 同量级，SPO+ 主导决策对齐
- **没做 λ=∞ 实验**（CODE Q44）：纯 SPO+ 估计回到 T134 不稳定
- **没做 λ=1 实验**（CODE Q44）：纯 L2 = T81，已知 NN alone +35.89

### 8.7 SPO+ DFL 是 Fisher consistent 吗？

- **CODE Q174 / NN Q41 提的**：SPO+（Elmachtoub & Grigas 2022）原本针对线性 LP。这里改成 1D 三选项任务，Fisher consistency 不明确。
- **答案**：理论上 Fisher consistency 对于 1D EV gate 是成立的（pred 单维，决策三选项，linear loss 形式仍满足 surrogate property）。但作者**没有数学证明**——empirically 用 T87 +28.16 平台分支撑。
- **GAP**：理论严谨性缺失

### 8.8 weighted SPO+ 凸性问题

- **NN Q92 / CODE Q196 提的**：`(spo_per * wb).sum() / wb.sum()` 用 class_balanced_weight 加权 SPO+。理论上 weighted SPO+ 仍是凸 surrogate 吗？
- **答案**：未严格证明。Empirically 不影响收敛。
- **GAP**：理论可疑但 work

### 8.9 被拒绝的替代方案

| 方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| Direct PnL loss | T13 / T134 / T57 / T60 | 不收敛 | 未提交 | 梯度不稳 |
| GMADL direction-aware loss | T116 | 微弱 | — | 不超 SPO+ |
| Quantile LGB (α=0.5) | T132 | 边际 | — | LGB 已在 L2 上 work |
| L2 NN only (no SPO+) | T81 alone | LOSO-eq +35.89 | 未单提 | 不超 SPO+ NN+LGB +40.09 |
| MAE LGB | T137 | +0.3-0.5 边际 (OOF) | 未提交 | 平台未确认超 L2 |
| 3-class CE NN | T1 / DeepLOB | LOSO ~+8 | −6.65 | 压缩幅度信息 |

### 8.10 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| λ=30 | T136 sweep + T176 re-sweep | 低 |
| lr=3e-5 | T136 sweep | 低 |
| 两阶段 pretrain + finetune | T13/T134 反证 + T81 alone 数字 | 低 |
| L2 + λ·SPO+ 联合（非 λ=1 或 λ=∞）| 无完整 ablation | 中 |
| Fisher consistency | 无数学证明，empirically work | 中（理论 gap）|
| weighted SPO+ 凸性 | 无证明，empirically work | 中 |

---

## §9. NN 模型架构

### 9.1 最终决策

| 组件 | 选择 | 位置 |
|---|---|---|
| 输入维度 | **359**（370 - 11 drop）| `Predictor.py:9` |
| Hidden | **(256, 128, 64)** | `train_T188v2_nn_seed.py:34` |
| 输出 | 1 维 linear（regression）| 同上 |
| Activation | **GELU**（默认 erf）| `train_T188v2_nn_seed.py:89` |
| Norm | **LayerNorm** | `train_T188v2_nn_seed.py:88` |
| Dropout | 0.05 / 0.10 / 0.15 / 0.20 (cycling) | `train_T188v2_nn_seed.py:38` |
| Init | **Kaiming (nonlinearity="relu")** | `train_T188v2_nn_seed.py:99` |
| Optimizer | **AdamW** (wd=1e-4) | `train_T188v2_nn_seed.py:43, 49, 292, 435` |
| LR schedule | **CosineAnnealingLR** | `train_T188v2_nn_seed.py:293-294, 436` |

### 9.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q35 / NN Q3 | MLP [359→256→128→64→1] 为啥这样选 |
| CODE Q36 / NN Q42 | 为啥 MLP 而非 GRU/Transformer/CNN |
| CODE Q37 / NN Q26, Q44 | LayerNorm + GELU + Dropout 组合 |
| CODE Q38 / NN Q48 | Kaiming init nonlinearity="relu"（实际激活 GELU）|
| CODE Q39 | AdamW vs Adam / SGD |
| CODE Q40 | CosineAnnealingLR 选择 |
| CODE Q45 | 没有 BatchNorm / residual |
| **CODE Q176 ⭐** | **训练 GELU (erf) ≠ 推理 GELU (tanh) — 潜在 bug** |

### 9.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 8）

| 实验 | 方法 | 本地 | 平台 |
|---|---|---|---|
| T1 | DeepLOB-style MLP, 3-class CE | LOSO ~+8 | (mmpc_demo −6.65) |
| T17 | CatBoost vs LightGBM | LOSO 持平 | — |
| T18 | XGBoost vs LightGBM | 持平 | — |
| T19 | NN regularized (dropout + WD) | 持平 | — |
| **T81** | **MLP [359→256→128→64→1] LayerNorm GELU Dropout 0.10 L2** | LOSO-eq +35.89 | — |
| T89 | CatBoost L2 5-seed | LOSO +40.13 (与 T87 corr 0.77) | iter_016 失败 |
| T93 | NN arch Transformer (preliminary) | — | — |
| T94 | NN arch ResNet-DCN | — | — |
| T95 | GRU regression | LOSO 微小 | iter_016 失败主因 |
| T117 | LGB monotone constraints | LOSO 局部 | 未确认 |
| T172/b | GroupTransformer (d_model=64, nhead=4, 2 layers, n_tokens=32, CLS) | in-sample +149.91 (污染) | 未提交 |
| T184/T185 | 深度 CNN raw / hybrid CNN | in-sample 看似好 | 未提交 |
| T186 | T184/T172b retrain + LR sweep | "architecture_doesnt_fit" | — |
| T187 | GroupTransformer replace/add | in-sample +22~55 (污染) | 未提交 |

### 9.4 为啥 MLP [359→256→128→64→1]？

- **T81 设计**：输入 359 维 SchemeP，输出 1 维 Δmid 回归
- **渐缩 1.5×~2× 经典 MLP**：256 → 128 → 64
- **总参数量 ~120k**：relatively small，避免过拟合 1.47M 样本
- **T93/T94 替代失败**：Transformer/ResNet-DCN 未超 MLP
- **T186 verdict "architecture_doesnt_fit"**：深度架构在 SchemeP 359 维特征框架下无法竞争 MLP
- **GAP**：没有 width/depth 完整 sweep（CODE Q35 / NN Q3）

### 9.5 为啥 MLP 而不是序列模型？

- **T95 GRU 反例**：hidden state 易在短 horizon 过拟合，是 iter_016 v3 失败主因
- **T172/T187 Transformer 反例**：in-sample 数字看着好（+22~55），全部含 96-119 训练污染
- **T184/T185 CNN 反例**：in-sample 看似好，未提交（OOD 风险）
- **设计哲学**：特征工程（SchemeP 359 维）已经把 100-tick 窗口的时序信息提炼成 aggregate features → MLP 看 aggregate features 是更鲁棒的设计
- **GAP**：没有在干净 OOD 下系统对比 MLP vs GRU/Transformer

### 9.6 LayerNorm + GELU + Dropout 组合

- **LayerNorm（不是 BatchNorm）**：per-sample 归一化，与 SchemeP 窗口 z-score 互补；batch 内不同样本独立（符合 batch shuffle 约束）
- **GELU（不是 ReLU/SiLU）**：现代 NLP 标准。NN Q26 / CODE Q37 质疑"GELU 在 z-score 后多在 [-3,3] 上几乎线性"——确实，但 empirically work
- **Dropout 0.10 中心**：T19 试过更激进 regularization 无增量

### 9.7 ⚠️ CODE Q176: 训练 GELU ≠ 推理 GELU（潜在 bug）

`Predictor.py:171`: `F.gelu(h, approximate='tanh')`
`train_T188v2_nn_seed.py:89`: `nn.GELU()` (默认 `approximate='none'`，即 erf)

**两者数值差异**：
- |x| > 1 范围有 ~1e-4 数量级差异
- 在 hidden=256 维度上累加 → 输出层可能 ~1e-3 差异
- 阈值 thr_up=3e-4 → 1e-3 误差可能 flip action

**严重性评估**：
- HISTORY_EVIDENCE_DB.md 主题 10 提到 T188v3 actions 一致性验证（vs T188v2 max diff 6.98e-10），但**该验证可能是同 GELU 版本的内部对照**，未必覆盖训练 GELU vs 推理 GELU 差异
- 这是 review 提出的**潜在 bug**，作者未明确回应或修复
- **GAP**：未做训练 GELU 推理用 erf 的版本对比

### 9.8 Kaiming init with nonlinearity="relu"（实际激活 GELU）

- **CODE Q38 / NN Q48 提的**：轻微不匹配
- **GELU 与 ReLU 在原点附近近似**：Kaiming relu init 的方差缩放在 GELU 上 approximately 仍 OK
- **没用 xavier 或 kaiming gelu**：作者懒/惯性
- **GAP**：未做 init 对比

### 9.9 AdamW vs Adam / SGD

- **AdamW 默认选择**：weight decay 解耦
- **CODE Q39 / NN Q60 提的**：SGD+momentum 在小模型回归任务上有时更稳
- **GAP**：未做 SGD 对比

### 9.10 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| MLP [256, 128, 64] 维度 | 无 width/depth sweep | 中 |
| 选 MLP 而非 GRU/Transformer | T95/T172/T184 反例 | 低（OOD 风险已知）|
| LayerNorm vs BatchNorm | 无对比，约束 #2 间接决定 | 低 |
| GELU vs ReLU | 无对比 | 低 |
| **GELU erf vs tanh approx** | **无验证** | **中-高（潜在 bug，未修复）** |
| Kaiming relu init for GELU | 无对比 | 低 |
| AdamW vs SGD | 无对比 | 低 |
| CosineLR vs Constant/StepLR | 无对比 | 低 |


---

## §10. NN 训练协议（HP cycling、aug_a、early stop）

### 10.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| HP cycling | LR ∈ {1e-4, 3e-4, 1e-3}, Dropout ∈ {0.05, 0.10, 0.15, 0.20}, Batch ∈ {2048, 4096, 8192} | `train_T188v2_nn_seed.py:37-39` |
| HP 索引 | `(S-1) % 3 / % 4 / % 3` | `train_T188v2_nn_seed.py:186-188` |
| Phase 1 epochs | max 50, patience=10 | `train_T188v2_nn_seed.py:41-42` |
| Phase 2 epochs | 固定 11 | `train_T188v2_nn_seed.py:46` |
| Phase 2 batch | 固定 4096 | `train_T188v2_nn_seed.py:50` |
| aug_a | x' = x × U(0.8, 1.2) per feature | `train_T188v2_nn_seed.py:44, 271` |
| WD | 1e-4 (Phase 1 = Phase 2) | `train_T188v2_nn_seed.py:43, 49` |
| clip_grad_norm | 5.0 | `train_T188v2_nn_seed.py:318, 466` |
| CLIP (standardization) | ±10σ | `train_T188v2_nn_seed.py:35` |
| RNG seed offset | `S * 7919 + 42 / 1 / 137` | `train_T188v2_nn_seed.py:270, 406` |
| target_scale | 1/std(y) Phase 1 train, 沿用到 Phase 2 | `train_T188v2_nn_seed.py:266` |

### 10.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q14 / NN Q5 | LR [1e-4, 3e-4, 1e-3] 选择 |
| CODE Q15 / NN Q6 | Dropout [0.05, 0.10, 0.15, 0.20] |
| CODE Q16 / NN Q7 | Batch [2048, 4096, 8192] |
| CODE Q17 / NN Q9, Q10 | Patience=10, Phase 1 max 50 epochs |
| CODE Q18 / NN Q12 / LGB Q31 | aug_a U(0.8, 1.2) 幅度 |
| CODE Q23 / NN Q104 | target_scale 沿用 Phase 1 是否合适 |
| CODE Q25 / NN Q121 | Phase 1 CosineLR T_max=50 但早停可能在 ep 20 |
| CODE Q32 / NN Q102 | seed offset 7919 / 42 / 1 / 137 |
| CODE Q33 / NN Q116 | clip_grad_norm = 5.0 |
| CODE Q34 / NN Q4 | CLIP=10 standardization |
| CODE Q77 | early stop epsilon = 1e-10 |
| CODE Q86-Q87 / NN Q121-Q123 | Phase 2 用 train+val+test 全部 |
| CODE Q153 / NN Q158 | Phase 2 dropout 仍开启 |

### 10.3 HP cycling 的"实际只有 12 个 unique combos"问题

详见 [§6.4](#64-50-个-nn-实际只有-12-unique-combos代码-q13--nn-q8)。

50 NN 实际只覆盖：
- 12 unique (LR, Dropout, Batch) combos
- 每个 combo 重复 ~4.17 次
- 重复部分提供 init RNG + dropout RNG + aug RNG 多样性

**GAP**：README 写 36，实际 12 —— 没有用 sobol/random sample 覆盖完整 36 网格

### 10.4 LR / Dropout / Batch 取值依据

- **LR [1e-4, 3e-4, 1e-3]**：跨度 10×，中心 3e-4 "Karpathy LR"。无 5e-4 / 5e-3 中间值（CODE Q14）。GAP：无单独 sweep 曲线
- **Dropout [0.05, 0.10, 0.15, 0.20]**：常规偏轻，无 0.0 / 0.3 / 0.5（CODE Q15）。理由：相对小模型 120k 参数 + 1.47M 样本，激进 dropout 无必要
- **Batch [2048, 4096, 8192]**：偏大（标准 LOB 训练 batch 256-1024）。理由：训练效率优先；大 batch 在 fine-tune 上稳定（CODE Q16）。GAP：无小 batch 对比

### 10.5 Phase 1 early stop 配置（NN Q9-Q10, Q77）

- **max 50 epochs / patience 10**：1/5 总 epoch 做 patience
- **epsilon = 1e-10**（CODE Q77）：在 float32 上小于机器精度，等效于"任意改善即认可"。**作者偷懒，没设 1e-6**。
- **CosineAnnealingLR T_max=50**：如果 patience 在 ep 20 触发，cosine schedule 只用了 40%。**没有 reduce_on_plateau 兜底**。
- **GAP**：early stop 与 LR schedule 没有协调

### 10.6 aug_a x × U(0.8, 1.2) 设计

- **CODE Q18 / NN Q12 / LGB Q31 提的**：
  - 每个 feature 独立缩放破坏几何关系（如 bid1 vs bsize1 不能独立缩放）
  - 对 z-score 特征（均值 0）做乘法没物理意义
  - 对 quantile rank 特征（[0,1]）做乘法也没物理意义
- **拼接 [X, X_aug]** 后 y 完全相同 → 本质上是 sample weighting（CODE Q82）
- **但 work**：T25-T28 ablation aug_a 5-seed LOSO +11.46，T140 后续都用
- **GAP**：没做"sample-level 同步缩放" vs "per-feature 独立缩放"对比

### 10.7 Phase 2 用 train+val+test 全部（M7 trick）

详见 [§7](#7-m7-全数据重训第三次大突破)。

**CODE Q86 / NN Q121 提的问题**：concat 后用 randperm shuffle，NaN 处理用 train 的 feat_mean 填充。如果 val/test 里有 train 没出现过的 NaN pattern，填充值可能不合理。

**实际**：经验上不影响（缺失 pattern 较稳定）。GAP：未做不同填充策略对比。

### 10.8 target_scale = 1/std(y) Phase 1 train 沿用到 Phase 2

- **CODE Q23 / NN Q104 提的**：
  - Phase 1 mean/std 来自 dates 0-79
  - Phase 2 用 dates 0-119，分布可能不同
  - SPO+ 阶段 fee 也乘以 target_scale，可能破坏相对量级
- **作者选择沿用**：避免每个 seed 不同 scale 带来的复杂性
- **Predictor 推理**：每个 NN 保存自己的 target_scale，推理时 `out / target_scale` 还原（PREDICTOR Q215）
- **GAP**：未做"Phase 2 重新算 std" vs "沿用"对比

### 10.9 seed offset 7919/42/1/137（CODE Q32 / NN Q102）

- 7919 是质数（4-digit 大质数）
- 42 (Phase 1 aug RNG)，1 (LGB)，137 (Phase 2 aug RNG)
- **理由**：避免不同 RNG 流冲突
- **GAP**：LGB 用 +1，NN Phase 1 用 +42 → 同 seed 下 aug 数据不同（**故意设计 vs bug 未明**）

### 10.10 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| HP cycling 12 unique combos（非 36）| 无设计意识 | 中 |
| LR / Dropout / Batch 各组取值 | 无完整 sweep | 中 |
| Phase 1 epsilon=1e-10 | 偷懒，无对比 | 低 |
| aug_a per-feature 独立缩放 | 无 sample-level 对比 | 低 |
| target_scale 沿用 Phase 1 | 无对比 | 低 |
| Phase 2 dropout 仍开启 | 无对比 | 低 |
| seed offset 7919/42/1/137 | 无设计文档 | 低 |
| WD=1e-4 Phase 1 = Phase 2 | 无对比 | 低 |

---

## §11. LGB 训练协议（HP grid、bagging、num_leaves）

### 11.1 最终决策

| 参数 | 值 | 位置 |
|---|---|---|
| HP grid (5 组) | 见 11.3 | `train_T188v2_lgb_seed.py:25-31` |
| HP cycling | `HP_CONFIGS[(S-1) % 5]` | `train_T188v2_lgb_seed.py:83` |
| seeds per HP | 10 | run script |
| `num_boost_round` | 330 | `train_T188v2_lgb_seed.py:72` |
| `learning_rate` | 0.05 (固定，不 vary) | `train_T188v2_lgb_seed.py:73` |
| `min_data_in_leaf` | 100 | `train_T188v2_lgb_seed.py:74` |
| `bagging_freq` | 5 | `train_T188v2_lgb_seed.py:75` |
| `num_threads` | 18 | `train_T188v2_lgb_seed.py:76` |
| objective | `regression_l2` | `train_T188v2_lgb_seed.py:153` |
| metric | `rmse` | `train_T188v2_lgb_seed.py:154` |
| max_bin | default (255) | — |
| max_depth | unbounded | — |

### 11.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q12 / LGB Q1-Q11 | 5 组 HP 怎么挑的？|
| LGB Q12-Q15 | max_bin / lambda_l1 / min_gain / max_depth 都未 tune |
| CODE Q27 / LGB Q10 | bagging_freq=5 |
| CODE Q28 / LGB Q9 | min_data_in_leaf=100 |
| CODE Q29 / LGB Q7 | learning_rate=0.05 |
| CODE Q30 / LGB Q88 | num_threads=18 是某台机器 cpu 数？|
| LGB Q16-Q21 | objective=L2 vs MAE/Huber/Quantile |
| LGB Q18-Q19 | 训 regression 但加载 y_cls 算 weight |
| LGB Q22-Q27 | class-balanced weight 设计 |
| LGB Q43 | 没有 valid_sets 仅训练 |

### 11.3 5 组 HP 详情

| Config # | feature_fraction | bagging_fraction | num_leaves | lambda_l2 |
|---|---|---|---|---|
| 1 (baseline) | 0.8 | 0.8 | 127 | 1.0 |
| 2 | 0.6 | 0.7 | 127 | 1.0 |
| 3 | 0.7 | 0.85 | 63 | 2.0 |
| 4 | 0.5 | 0.6 | 255 | 0.5 |
| 5 | 0.4 | 0.5 | 127 | 3.0 |

**怎么挑的？**：
- 手工挑的（不是 grid / sobol / random search）
- Config 1 是 baseline；Config 2-5 是 baseline 的"扰动"
- LGB Q5 提问 "num_leaves 三组都是 127"：因为 127 是 V4 早期 sweep 的 sweet spot，扰动主要在 feature/bagging fraction
- **GAP**：没有完整 grid search，5 组的"覆盖空间"不严谨

### 11.4 不 tune 的 HP

| HP | 当前值 | 不 tune 理由 |
|---|---|---|
| `learning_rate` | 0.05 | 项目早期固定（V4 阶段未 vary）|
| `num_boost_round` | 330 | 见 [§7.4](#74-lgb-330-iters-怎么定的) |
| `min_data_in_leaf` | 100 | 1.47M 样本上 0.007%——非常低，鼓励过拟合（CODE Q28）。**未对比 1000** |
| `max_bin` | 255 (默认) | LGB Q14: 未对比 511 / 1023 |
| `lambda_l1` | 不设 | LGB Q11: 未考虑 |
| `min_gain_to_split` | 0 (默认) | LGB Q12: 与 lambda_l2 互动未检查 |
| `max_depth` | unbounded | LGB Q13: 只用 num_leaves 控制 |
| `bagging_freq` | 5 | LGB 文档示例值，未对比 1/10 |

### 11.5 class-balanced weight (训 L2 regression 但用分类 weight)

```python
# train_T188v2_lgb_seed.py:47-51
NUM_CLASS = 3
y_cls = label_h60  # 短涨/不动/短跌
weights = 1 / freq(y_cls)
```

- **回归目标 + 分类 weight**（LGB Q18-Q19 / CODE Q80）：语义模糊
- **为啥不直接按 |y| 加权**：T92 magnitude weight 反效果（LOSO 下降）→ 假设证伪
- **class-balanced weight work 的原因**：标签 distribution 不平衡（h_60 上多数 sample 是"不动"），给少数类（涨/跌）weight 让模型重视决策边界

### 11.6 `valid_sets=[dtrain]` 不留验证集

- **LGB Q43 提的**：valid_sets 是 dtrain 本身 → 监控的是 training loss，不是 OOF
- **原因**：M7 阶段全数据训，没有 val 集（设计意图，见 [§7](#7-m7-全数据重训第三次大突破)）
- **风险**：无法监控过拟合，靠 num_boost_round=330 这个 hardcoded 数

### 11.7 num_threads=18（CODE Q30 / LGB Q88）

- **18 看着像某台机器**：可能 4×18=72 → 用 18 留 buffer
- **GAP**：未设 -1（自动检测）；平台 CPU 数固定吗？没文档

### 11.8 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 5 HP 组合手挑 | 无完整 grid search | 中 |
| learning_rate=0.05 固定 | 无对比 0.1 | 低 |
| min_data_in_leaf=100 | 无对比 1000 | 低 |
| max_bin/lambda_l1/min_gain 都默认 | 无对比 | 低 |
| max_depth 不设 | 无对比 | 低 |
| valid_sets=[dtrain] | 设计意图，无对比 | 低（330 iter 已 conservative）|
| class-balanced weight 与 L2 混搭 | 语义模糊但 work | 低（T92 magnitude 反例）|
| num_threads=18 | 机器特定 | 低（推理用 CPU 默认）|

---

## §12. 损失函数（Δmid 回归 vs 分类、L2 vs MAE/Huber）

### 12.1 最终决策

| 模型 | Loss | y 定义 | 位置 |
|---|---|---|---|
| LGB | `objective='regression_l2'` | `y = (mp_h - mp_t) / (mp_t + 1)` | `train_T188v2_lgb_seed.py:134, 153` |
| NN Phase 1 | L2 (MSE) | 同上 | `train_T188v2_nn_seed.py:323` |
| NN Phase 2 | `L = MSE + λ × SPO+`（λ=30）| 同上 | `train_T188v2_nn_seed.py:462` |
| 决策 | EV gate | `pred > thr_up → 2; pred < −thr_dn → 0; else 1` | `Predictor.py:255-271` |
| Fee 公式 | `FEE × ((mp_h+1)+(mp_t+1)) / (mp_t+1)` | — | `train_T188v2_nn_seed.py:75-77` |
| FEE 值 | 0.0001 | — | `train_T188v2_nn_seed.py:33` |

### 12.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| LGB Q16 / Q20 | objective=L2 vs L1/Huber/Quantile |
| CODE Q22 / LGB Q22 / NN Q2 | FEE=0.0001 来源 |
| CODE Q84 / LGB Q52 / NN Q70-Q71 | y 公式 (mp_h - mp_t)/(mp_t + 1)，为啥 +1 |
| CODE Q85 / NN Q72 | fee_eff 公式 |
| CODE Q105 | 没用 isotonic/Platt 校准 |
| CODE Q106 | pred 单位与 EV 单位关系 |

### 12.3 关键决策：Δmid L2 regression（最大单次跃升）

详见 [§8](#8-spo-dfl第二次大突破) 演变时间线。

**关键证据**：
- iter_013 (T75)：把 LGB 从 3-class CE 换成 L2 Δmid regression → 平台 +4.07 → +19.23 (**+15.16 单次最大跃升**)
- 项目最大单次跃升

**为啥 3-class CE 不行**：
- T1 NN baseline LOSO ~+8 → 平台 (mmpc_demo) −6.65
- T2 LGB SchemeB 154 维 baseline LOSO sum −22.10
- CE 把 Δmid 离散化为 3 个 bucket → 损失幅度信息

**为啥 L2 不是 MAE/Huber/Quantile**：

| 替代 | T 编号 | 本地 | 平台 |
|---|---|---|---|
| MAE LGB | T137 | +0.3-0.5 边际 (OOF) | 未确认 |
| MAE LGB (v3) | T143 | LOSO +41.20 (−0.29 vs L2) | 未提交 |
| MAE LGB M7 | T145/b | 预测 +30~32 | 未提交 |
| Huber LGB | T99 | +44.72 (OOF 污染) | iter_016 失败 |
| Quantile LGB (α=0.5) | T132 | 边际 | — |
| GMADL direction-aware | T116 | 微弱 | — |

→ **L2 始终主力**，MAE 不超 L2 (T137/T143/T145)，Huber 反而过拟合 OOF

### 12.4 y = (mp_h - mp_t)/(mp_t + 1) 公式

- **标准 return**：`(p_h - p_t) / p_t`
- **+1 含义**：避免除零；当 mp_t = 0 时不崩
- **midprice1 实际范围**：未在 review 中明确，但 reviewers 推测 ~ 0.01 ~ 100 量级（CODE Q84）
- **GAP**：没有标准化为 log return 或 simple return；+1 这种"暴力"防御看着 hacky 但 work

### 12.5 FEE=0.0001 来源

- **CODE Q22 / NN Q2 提的**：是平台公布还是经验拟合？
- **答案**：**未明确**。可能：
  - 平台规则文档中的 manual fee 设置
  - 或 iter 早期某次实验的拟合值
- **fee_eff 公式**：`FEE × ((mp_h+1)+(mp_t+1)) / (mp_t+1) ≈ 2 × FEE` 当 mp_h ≈ mp_t
- **假设**：双边 fee（买入 + 卖出都收）
- **GAP**：未做 FEE ∈ {5e-5, 2e-4} sensitivity sweep（CODE Q22）

### 12.6 没用 isotonic / Platt 校准（CODE Q105）

- **理由**：回归 pred 直接当 EV，没有"概率"概念
- **早期 T15 试过 isotonic + EV gate**：LOSO −1.02（破坏排序质量）
- **R13 calibration 调研**：isotonic/platt 均负

### 12.7 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| L2 vs MAE / Huber / Quantile | T137/T143/T145/T99/T132 全部对比 | 低 |
| 3-class CE vs L2 regression | T1/T2/T75 对比 | 低 |
| y 公式 +1 防御 | 无 ablation | 低 |
| FEE=0.0001 | 无 sensitivity sweep | 中 |
| fee_eff 公式假设双边 | 无平台 spec 确认 | 中 |
| 不用 isotonic/Platt | T15/T40/R13 反证 | 低 |

---

## §13. 特征工程（SchemeP 370 维、Stage 1-5、drop 11、归一化）

### 13.1 最终决策

| 项 | 选择 | 位置 |
|---|---|---|
| 总维度 | **370 = 154 raw + 216 derived**，drop 11 → **359** 进模型 | `Predictor.py:9, 68-75` |
| Raw 154 维 | 全部进模型（包括 OHLC、ind、acc 等）| `build_schemeP_cache.py:53-69` |
| Derived 216 维 | T3 (69) + Stage1 (54) + Stage2 (59) + Stage3 (14) + Stage5 (20) | `fast_features_batch.py:5` |
| Stage 4 | **缺失**（被合并/丢弃）| — |
| Drop 11 | T59_FAIL (10) + STAGE5_FAIL (1) | `Predictor.py:68-75` |
| 归一化 | 窗口 z-score (per-100-tick)，**非全局** | T7 ablation |
| amount_delta 处理 | raw_last 列做 sign × log1p(|v|)，**但 X3d 用原始 amt** | `build_schemeP_cache.py:108-110` |
| WINDOW | 100 | `build_schemeP_cache.py:45` |

### 13.2 白板问题汇总（合并去重后）

| 主题 | 涉及问题 |
|---|---|
| 维度计数与一致性 | FEATURES Q1-Q14, Q210-217 |
| RAW_COLS 154 维语义 | FEATURES Q16-Q25 / CODE Q49 |
| amount_delta 处理不一致 | FEATURES Q20 / CODE Q47-Q48 / Q92 |
| T3 MLOFI 与 Stage5 OFI 公式不一致 | FEATURES Q26-Q29, Q73, Q105-Q106 |
| dual_z 设计 | CODE Q54 / FEATURES Q40-Q60 |
| kyle_inv / kyle_lambda / signed_rv 公式 | CODE Q55-Q56 / FEATURES |
| EWMA alphas / RV windows 选择 | CODE Q52-Q53 / FEATURES Q204-Q205 |
| Stage 划分（缺 Stage 4）| CODE Q50 / FEATURES |
| FAIL_NAMES 11 个具体哪些 | CODE Q20 / FEATURES Q4-Q7 |
| feature pruning 标准（KS test）| FEATURES Q3 |
| 窗口归一化 vs 全局归一化 | T7 vs T10 ablation |
| 不加 Hawkes / Alpha101 | CODE Q70-Q72 / FEATURES |
| 没有特征剪枝 | CODE Q73 / FEATURES |
| feat_mean/std 来自 train | NN Q97 / FEATURES |

### 13.3 演变时间线（HISTORY_EVIDENCE_DB.md 主题 6）

| 实验 | 方法 | 本地 |
|---|---|---|
| T2 | LGB SchemeB 154 维 | LOSO sum −22.10 |
| T3 | +72 维 (MLOFI/WMP/RV/EWMA/time) = SchemeC 226 维 | LOSO h_10 +21.86 |
| T7 | 窗口内 z-score (100-tick) = SchemeD1 308 维 | LOSO +13.91 (sym=2 修复)|
| T9 | SchemeE（amount_delta z-score + spread-norm）| 部分入 SchemeP |
| T10 | SchemeF 全局 z-score | LOSO +13.40（**不如 T7**）|
| T22 | Alpha101 + Alpha191 | IC +21.80 但**未提交**（违反约束 #2）|
| T25-T28 | aug_a | LOSO +6.30 → +9.67 → +11.46 |
| T35 | ReVol + Savitzky-Golay (255 维) | 不超 SchemeC |
| T44 | **R34 Stage 1**：+54 维 sym-invariant | LOSO 5-seed +13.67 |
| T51 | **R34 Stage 2**：+59 维 | LOSO 5-seed **+15.06** |
| T53 | **R34 Stage 3**：+14 维 | LOSO **+16.84** |
| T55 | **R34 Stage 4**：少量 edge gain | 合并入 SchemeP |
| T68 | **R34 Stage 5**：+20 维 → SchemeP 359 维完成 | LOSO-eq +26.44 |
| T74 | time features SchemeQ (+6 维) | T78 LOSO-eq −0.72（**回归框架下负面**）|
| T141/b | Hawkes 5-seed | LOSO 边际，**未入** |
| T142 | Hawkes M7 retrain | 边际，**未入** |
| T147 | feature pruning (importance) | LOSO 轻微正 |
| T151 | TOD (Time-of-Day) | LOSO 小幅正，但**未入 final** |
| T152 | adversarial-filtered 259 维 | LOSO 持平，**未入 final** |
| T191 | **trend features (+5 维)** | standalone +140.10 (vs T188v2 +150.43, **−10.33**); corr(T191, T188v2) = **0.9941**; IC_residual_pooled = **−0.1699** (负！) | KILL |

### 13.4 SchemeP 359 维 = 370 - 11 详细组成

```
raw 154 维 (建议看 RAW_COLS 列表):
  bid1..bid10 (10) + bsize1..bsize10 (10)
  ask1..ask10 (10) + asize1..asize10 (10)
  midprice1..midprice10 (10)
  spread1..spread10 (10) + cumspread (1)
  bid_diff1..bid_diff10 (10) + ask_diff1..ask_diff10 (10)
  intst×6 cols (lb/la/mb/ma/cb/ca) + ind×6 + acc×6 + rate×40
  amount_delta (1) + volume_delta (1)
  open, high, low, close (4) + 其他
  = 154 维

derived 216 维:
  T3 no-time (69):
    MLOFI (3 W × 10 lvl) = 30
    WMP (10 + 1) = 11
    RV (4 W) = 4
    EWMA-intst (4 α × 6 col) = 24
  Stage1 (54):
    DualZ (37) + Signed RV (3) + Kyle inv (2) + EWMA-OFI (12) = 54
  Stage2 (59):
    QRank (20) + RSkew (3) + GOFI (30) + Kyle lambda (2) + Vol burst (4) = 59
  Stage3 (14):
    EWMA residual (1) + RV ratio (3) + JShare (4) + Cancel imbalance (3) + Roll spread (3) = 14
  Stage5 (20):
    Adapt momentum (3) + OFI toxicity (4) + Signed BV (3) + Spread regime (3)
    + Trade persistence (3) + Liquidity asymmetry (4) = 20

drop 11 (FAIL_NAMES, Predictor.py:68-75):
  T59_FAIL (10):
    dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5
    qrank_W100_spread1/5/10/cumspread
    kyle_lam_W50, kyle_lam_W100
    roll_eff_spr_ratio_W100
  STAGE5_FAIL (1):
    liq_asym_top5_W5

final: 370 - 11 = 359 维
```

### 13.5 R34 Stage 1-5 累计增益

| Stage | 增量维 | 本地增益 (LOSO 5-seed) |
|---|---|---|
| Stage 1 (T44) | +54 维 | +13.67（vs iter_005b +9.67, **+4.0**）|
| Stage 2 (T51) | +59 维 | **+15.06**（+1.4 vs Stage 1）|
| Stage 3 (T53) | +14 维 | **+16.84**（+1.78）|
| Stage 4 (T55) | edge gain | 合并入 SchemeP |
| Stage 5 (T68) | +20 维 | LOSO-eq **+26.44**（+0.50，含 +7 方法论 inflation）|

**Stage 4 为啥缺失**（CODE Q50 / FEATURES Q120）：
- T55 R34 Stage 4 调研给了 edge gain 但不显著
- 合并入 SchemeP 时没有单独命名
- 代码里只看到 Stage 1/2/3/5，**Stage 4 隐式融入其他 stage**

### 13.6 sym-invariant 设计（窗口 z-score vs 全局 z-score）

| 实验 | 方法 | 本地 |
|---|---|---|
| **T7** | **窗口 z-score (100-tick)** | LOSO +13.91（sym=2: −0.98 → +2.24, 修复 brittleness）|
| T10 | 全局 z-score | LOSO +13.40（**不如 T7**）|

**原因**：
- 窗口 z-score 在每个 100-tick 切片内自适应 → 对 sym distribution shift 鲁棒
- 全局 z-score 用整个训练集的 mean/std → 对 OOD sym 不鲁棒
- 满足约束 #3 sym-agnostic

### 13.7 amount_delta 处理不一致（CODE Q47-Q48 / FEATURES Q20）

```python
# build_schemeP_cache.py:108-110
raw_last[:, amt_idx] = sign(v) * log1p(|v|)   # raw_last 列 log 化
# 但 fast_features_batch.py:311, 436, etc.
X3d[:, :, col_idx["amount_delta"]]            # extras 用原始 amt
```

→ **同一个底层量在不同特征里被赋予不同尺度的表示**

**是 bug 还是有意**：作者未明确回应，**reviewers 推测**：
- raw_last 是"最后一 tick 的快照"——log1p 压缩重尾
- X3d 是"100-tick 窗口"——下游特征（kyle_inv, signed_dvol, vol_burst）原本就内置归一化（cbrt, std, mean ratio），所以 raw amt 没必要 log
- 结果：模型同时见到两个尺度，可能学到 redundancy

**GAP**：无消融对比"全部 log1p" vs "全部 raw"

### 13.8 T3 MLOFI vs Stage5 OFI 公式不一致（FEATURES Q26-Q29, Q105-Q106）

T3 MLOFI（B:159-184）：
```
ind_b_up = (b >= b_prev); ind_b_dn = (b <= b_prev)
# b == b_prev 时两 indicator 都为 1 → e = bs - bs_prev
```

Stage5 OFI（B:645-650）：
```
strict > and <, with separate eq case
```

→ 两份 OFI 实现风格不统一

**FEATURES Q219 推测**：这种不一致是合并多个分支没 harmonization 的产物

**实际后果**：模型同时见到两套 OFI → 可能学到 redundancy。**没有 ablation 验证哪个版本更好**

### 13.9 FAIL_NAMES 11 个具体哪些 & KS test 标准

- **drop 标准**：KS test（distribution shift train vs test）
- **阈值**：FEATURES Q3 提问"是 0.05 还是 0.1？Bonferroni-adjusted？"——**code 里无文档**
- **on which split**：未明确
- **FAIL_NAMES 内部不对称**（FEATURES Q4）：
  - `dualz_ask_diff1` drop，`dualz_bid_diff1` keep（不对称！）
  - `kyle_lam_W50/W100` 两个都 drop → "kyle_lam" 整族贡献 0，**但仍在每次推理计算**（FEATURES Q6 浪费 compute）
  - `liq_asym_top5_W5` drop，W20/50/100 keep → 短窗口噪声大

**GAP**：drop 决策没有完整文档；FAIL 特征仍在 fast_features_batch 计算（CODE Q151）

### 13.10 不加 Hawkes / Alpha101 / TOD / Trend features 的理由

| 方案 | T 编号 | 本地 | 平台 | 拒绝理由 |
|---|---|---|---|---|
| Hawkes 过程 OFI | T141/142 | 边际 | 未提交 | 增量不显著 |
| Alpha101/191 | T22 | IC +21.80 | **未提交** | **违反约束 #2 batch shuffle** |
| TOD (Time-of-Day) | T151 (v9) | LOSO 小正 | 未确认 | T170 final 未用 |
| SchemeQ time features | T74/T78 | LOSO-eq −0.72 | — | **回归框架下负面** |
| Trend features | T191 | −10.33, corr=0.9941, IC=−0.17 | 未提交 | **完全冗余 + 负 IC** |
| Roll/TSRV | R_roll_tsrv | 部分入 Stage 3 | — | 未成主力 |
| CNN (DeepLOB) | T184/T185 | in-sample 看似好 | 未提交 | OOD 未验证 |
| Transformer (GroupTr) | T172/T187 | in-sample +22~55 (污染) | 未提交 | OOD 风险 |
| SG filter | T35/T91 | 不超 SchemeC | — | SG 对树模型 gain ≈ 0 |
| logret lag | R4/T124 | 无增量 | — | KILL |
| Pairwise 交互 | R_Pairwise_ALL | 计算量大边际 | — | 未入 |
| sym embedding / per-sym norm | (隐式)| — | — | 违反约束 #3 |
| date-derived features | (隐式)| — | — | 违反约束 #1 |

### 13.11 没有特征剪枝（CODE Q73）

- T147 feature pruning 试过，LOSO 轻微正
- T152 adversarial-filtered 259 维（370 → 259），LOSO 持平
- **最终决定保留 359 维全集**（drop 仅 11 个 FAIL_NAMES）
- 理由：feature pruning 平台无确认提升 + 保留全集更稳

### 13.12 没有 unit assert（FEATURES Q1, Q2, Q210, etc.）

- **没有 runtime assert** 验证 `compute_batch_features` 返回 216 维
- 没有 schema validation
- 没有 feature 顺序固定的 check（CODE Q24）
- **风险**：FEATURES Q210 — feat_mean/std 维度与 keep_idx 长度不匹配会 silent 失败
- **GAP**：工程鲁棒性差

### 13.13 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 370 维总规模 | R34 Stage 1-5 累计 ablation | 低 |
| 窗口归一化 vs 全局 | T7 vs T10 | 低 |
| amount_delta 不一致处理 | 无对比 | 中（设计/bug 未明）|
| 两套 OFI 公式 | 无对比 | 中（可能 redundancy）|
| FAIL_NAMES 11 个 | 无明确 KS 阈值 | 中（文档缺失）|
| 没特征剪枝 | T152 持平 → 保留全集 | 低 |
| 没 Hawkes / Alpha101 / TOD / Trend | 全部 ablation | 低 |
| Stage 4 缺失 | 设计上合并 | 低 |
| 没 runtime assert | 工程性 | 中 |

---

## §14. 推理优化（batched bmm、npz vs pt、CPU torch）

### 14.1 最终决策

| 项 | 选择 | 位置 |
|---|---|---|
| NN 50 推理 | **batched torch CUDA bmm**（CPU 可用 fallback）| `Predictor.py:78-178` |
| LGB 50 推理 | **CPU sequential**（每个 booster 独立 predict）| `Predictor.py:306-314` |
| 权重格式 | NN: **npz**（手 extract per-layer）；LGB: **txt**（lgb.Booster save）| `train_T188v2_nn_seed.py:130-160`、`Predictor.py:249` |
| GELU | `F.gelu(h, approximate='tanh')` (推理) vs `nn.GELU()` (训练 erf) | ⚠️ 不一致，见 [§9.7](#97-code-q176-训练-gelu--推理-gelu潜在-bug) |
| LayerNorm | 手动实现 (var unbiased=False, eps=1e-5) | `Predictor.py:166-170` |
| requirements.txt | `torch==2.5.1+cpu` | `04_build_pkg/requirements.txt` |
| 5 horizon | share h=60 (1× cost) | `Predictor.py:329-358` pred_cache |

### 14.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| CODE Q111 / PREDICTOR Q41-Q70 | 50 NN bmm 设计 |
| CODE Q112 / PIPELINE Q17 | requirements.txt CPU torch 但代码有 CUDA 分支 |
| CODE Q113 / PIPELINE Q8 | NN 权重保存为 npz 而不是 pt |
| CODE Q114 | LGB CPU sequential 不优化 |
| **CODE Q115 / Q176** | **GELU tanh vs erf 不一致（潜在 bug）** |
| CODE Q116 / PREDICTOR Q66 | LayerNorm 手动实现一致性 |
| CODE Q120 / FEATURES Q122, Q217 | build_pkg 复制 fast_features.py 但 Predictor 不用 |

### 14.3 加速演变（HISTORY_EVIDENCE_DB.md 主题 10）

| 实验 | 方法 | 加速比 |
|---|---|---|
| T61 (iter_010) | batch-vec 特征提取 | 254 min → 4.5 min（**58×**）|
| T188v3 | 50 NN sequential numpy → batched torch CUDA bmm | 1509ms → 8.1ms（**186×**）|
| T188v3 实测 | NN-only 554.8ms → 8.1ms | 68.5× |
| T188v3 e2e | feature(549ms) + LGB(84ms) + NN(1509ms) = 2142ms → 641ms | 3.3× |
| T188v3_fullhorizon | 5 个 horizon 共享 h=60 | 5× → 1× |

**总体**：T188v3 把 442k 行总时间 15.4 min → 4.6 min（远低于平台 3h 限制）

### 14.4 NN batched bmm 设计

```python
# Predictor.py:78-178
W_all = stack(50 weights) shape (50, out, in)
h = repeat(x, 50, axis=0) shape (50, B, in)
h = bmm(h, W_all)  # (50, B, out) 单次算 50 个 NN forward
```

**CODE Q111 提的问题**："为啥不把 50 个 weight 拼成 (out*50, in) 用单个 matmul？"
- **答案**：bmm 在 CUDA 上语义直接，避免重排；batched 张量原生支持
- 50×1024×359×256×4 ≈ 50 MB 中间结果，OK

**actions 一致性验证**：T188v3 vs T188v2 actions max diff 6.98e-10（远低于 1e-4 容忍度）
- **但这只验证 numpy sequential 与 torch bmm 的等价性，不验证 GELU 实现差异**

### 14.5 ⚠️ 关键潜在 bug：GELU 实现不一致

详见 [§9.7](#97-code-q176-训练-gelu--推理-gelu潜在-bug)。

- **训练**：`nn.GELU()`（默认 erf 实现）
- **推理**：`F.gelu(h, approximate='tanh')`
- **数值差异**：|x| > 1 范围有 ~1e-4 数量级差异
- **风险**：阈值 thr_up=3e-4 → 累加 4 层 hidden 后可能 flip action
- **未修复**：作者似乎未意识或忽略
- **GAP**：未做 erf vs tanh 推理对比

### 14.6 npz vs pt 权重格式（CODE Q113）

```python
# train_T188v2_nn_seed.py:130-160 extract_npz
np.savez(out_path,
    L0_W=..., L0_b=...,    # Linear 0
    LN0_W=..., LN0_b=...,  # LayerNorm 0
    L1_W=..., L1_b=...,    # Linear 1
    ...
    LF_W=..., LF_b=...,    # Final Linear
    feat_mean=..., feat_std=...,
    keep_idx=..., target_scale=...,
    clip=..., hidden=...,
)
```

**理由**：
- 避免 torch 序列化跨版本兼容性问题
- 平台 `weights_only=False` 在新版 torch 中可能 reject pickled module
- npz 是 numpy 通用格式

**代价**：
- 手 extract per-layer 权重 → 维护成本高（CODE Q135 提到 layer_idx 计数脆弱）
- LayerNorm gamma/beta、weight、bias 都要 manual 写
- 推理时手实现 forward（重复训练 forward）

**GAP**：训练 vs 推理两套独立 forward 实现，PIPELINE Q9 提到没有 numerical-parity unit test

### 14.7 LGB CPU sequential 不优化（CODE Q114）

- 50 个 booster 串行 predict，注释说 ~84ms
- LightGBM 4.6 支持 multi-threading（每个 booster 内部多线程）
- **没有进一步优化**：作者认为 84ms 足够

### 14.8 requirements.txt CPU torch 但代码有 CUDA 分支（CODE Q112）

```python
# Predictor.py:193
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- requirements.txt 明确 `torch==2.5.1+cpu` → `torch.cuda.is_available()` 始终 False
- CUDA 分支是"训练-推理代码共用"的残留
- **GAP**：冗余分支，没简化

### 14.9 fast_features.py 是 dead code（CODE Q120 / FEATURES Q122, Q217）

- `04_build_pkg/fast_features.py` 是单 window reference（196 维，无 Stage5）
- Predictor 实际用 `fast_features_batch.py`（216 维，含 Stage5）
- build_pkg.py 复制了两个文件，但只用 batch 版本
- **fast_features.py 是 dead code，仍在包内**
- **GAP**：未删除，可能让维护者混淆

### 14.10 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 50 NN batched bmm | T188v3 vs T188v2 actions 一致性验证 (6.98e-10) | 低 |
| LGB CPU sequential | 无优化对比 | 低（84ms 足够）|
| npz vs pt 格式 | 设计决定，无对比 | 低 |
| **GELU erf vs tanh** | **无验证** | **中-高（潜在 bug）** |
| LayerNorm 手实现 | 无 numerical parity test | 中 |
| CUDA 分支冗余 | 无 | 低 |
| fast_features.py dead code | 无 | 低 |

---

## §15. 评估方法论与本地 → 平台对应

### 15.1 已在 §2 详述

详见 [§2](#2-评估方法论变迁4-阶段)。

### 15.2 holdout in-sample → 平台对应表

| iter | holdout (in-sample) | 平台 |
|---|---|---|
| T170 | +149.95 | +34.64 |
| T182 (无 conformal) | +156.56 (+6.62) | +34.34 (−0.30) |
| T188v2 | +150.43 (+0.48) | +35.64 (+1.00) |
| T190 (150+150) | +140.62 (−9.81) | +34.59 (−1.05) |
| T192 H4 (NNLS) | +193.94 (+43) | +31.0 (−4.64) |

**规律**：
- 本地 delta 小 → 平台 delta 可能小（T188v2 +0.48 → +1.00）
- 本地 delta 大但符号反转 → 平台可能反向（T182 +6.62 → −0.30；T192 +43 → −4.64）
- 本地 delta 负 → 平台 delta 也常负（T190 −9.81 → −1.05）

→ **本地 delta 与平台 delta 的相关系数低，符号 ≠ 平台符号**

### 15.3 此节 GAP

详见 [§2.4](#24-评估方法论选择的-gap)。

---

## §16. 架构 / 流程组织（4 步 pipeline、build_pkg、requirements）

### 16.1 最终决策

```
final_submission_code/
├── 01_build_features/   # build_schemeP_cache.py + fast_features_batch.py + stage5_features.py
├── 02_train_lgb/        # train_T188v2_lgb_seed.py + run_all_lgb_seeds.sh
├── 03_train_nn/         # train_T188v2_nn_seed.py + run_all_nn_seeds.sh
├── 04_build_pkg/        # build_pkg.py + Predictor.py + fast_features.py / fast_features_batch.py + thresholds.json + config.json + requirements.txt
└── run_pipeline.sh      # 串联 01→02→03→04
```

### 16.2 白板问题汇总

| 原 Q | 问题核心 |
|---|---|
| PIPELINE Q1 | 4 步划分为啥不是 3 / 5 |
| PIPELINE Q2 / Q10 | 各 step 是否独立 re-run |
| PIPELINE Q3 / Q4 | 没有 orchestrator (Make / Snakemake) |
| PIPELINE Q5-Q8 | 数据契约（schema / version / checksum）缺失 |
| PIPELINE Q9 / Q96 | 训练 forward 与推理 forward 两套独立实现，无 parity test |
| PIPELINE Q11-Q14 | bash set -e 弱，无 set -u/pipefail/trap |
| PIPELINE Q15-Q18 | 训练/推理环境分离，但训练 requirements 未 pin |
| PIPELINE Q23-Q24 | 平台 OS/arch/python 版本假设未文档 |
| PIPELINE Q34-Q35 | --skip-phase1 默认 silent 复用 stale anchor 风险 |
| PIPELINE Q40 | multi-GPU 并行未实现 |
| PIPELINE Q43-Q44 | run_pipeline.sh 缺 preflight check |
| PIPELINE Q66-Q70 | README accuracy issues（feature count / iter naming 不一致）|
| PIPELINE Q88 | NaN 安全 |
| PIPELINE Q99-Q100 | 没 CI / 没 integration smoke test |

### 16.3 为啥 4 步而不是 3 / 5（PIPELINE Q1）

- **划分理由**：feature → LGB → NN → packaging
- **LGB / NN 独立**：可以并行（但 run_pipeline.sh 串行）
- **GAP**：LGB 和 NN 没合并到 "models" 顶层 → 增加新模型族需要新顶级目录

### 16.4 各步是否独立 re-run（PIPELINE Q2 / Q10）

- LGB: `train_T188v2_lgb_seed.py:91` 有 `if os.path.isfile(out_path): SKIP`
- NN: 有 `--skip-phase1` 但默认依赖 anchor 存在
- **没有 top-level pipeline_state.json**：无法回答"上次跑到哪了"
- **set -e** 在 bash → 一次失败整个 pipeline abort
- **GAP**：弱 resume 协议

### 16.5 数据契约缺失（PIPELINE Q5-Q8）

| 契约 | 现状 | 风险 |
|---|---|---|
| cache npz keys | 隐式（string keys: X, mp_t, y{H}, ...）| 改 key 名 silent 失败 |
| schemeP_feat_names.txt | plain text, 无 checksum | 手动编辑可能错位 |
| npz 版本/git SHA | **无** | 升级 fast_features_batch 后旧 model 不知道 misalign |
| Predictor 与训练 forward | 两套独立实现 | 无 parity test（PIPELINE Q9）|

### 16.6 训练 / 推理 forward 两套独立实现（PIPELINE Q9 / Q96）

- **训练**：`nn.Sequential` (Linear + LayerNorm + GELU + Dropout)
- **推理**：`_BatchedMLPEnsemble` 手实现 + `F.gelu(approximate='tanh')`
- **PREDICTOR Q96** claim ~1e-6 typical, 1e-4 max drift
- **无单元测试**确认 parity
- **GELU 不一致**已详述（[§9.7](#97-code-q176-训练-gelu--推理-gelu潜在-bug)）

### 16.7 build_pkg sanity check 极弱（CODE Q131-Q134）

- 只 check 模型文件存在
- 不验证：
  - thresholds.json 的 ensemble_seeds 都对应实际文件
  - NN npz 里 in_dim / hidden 一致
  - feat_mean/std 维度匹配
  - NN 与 LGB seed 配对
- **WARNING 但不报错**（CODE Q132）
- 缺 seed silent 上线 < 50 个模型

### 16.8 requirements.txt 设计

```
torch==2.5.1+cpu
numpy==2.4.4
pandas==2.3.3
lightgbm==4.6.0
scipy==1.14.1
--extra-index-url https://download.pytorch.org/whl/cpu
```

- **CPU torch**：平台无 GPU 假设
- **严格 pin 版本**（CODE Q181）：保证 byte-equivalence；但 numpy 2.4.4 PyPI 上是否存在需验证
- **PIPELINE Q18**：依赖 `--extra-index-url`，若平台 sandbox 无外网 fail。**未在 offline pip install 测试过**
- **GAP**：平台环境契约未文档

### 16.9 README / code drift 问题（PIPELINE Q66-Q70）

| README 声明 | 代码实际 | 问题 |
|---|---|---|
| "3×4×3 = 36 combos" | 实际 12 unique combos | [§6.4](#64-50-个-nn-实际只有-12-unique-combos代码-q13--nn-q8) |
| "44 个 R 系列调研" | 代码无 R 系列痕迹 | CODE Q161 |
| "150+150 没做过" | T190 做了 | CODE Q171 |
| "+5.51 from M7" | 早停 +28.16 → 全数据 +34.44 = +6.28 | CODE Q163（数字不符）|
| "T81 warm-start" | 普通 fine-tune | CODE Q162（术语包装）|

### 16.10 此节 GAP

| 设计点 | 有无 ablation | 风险评估 |
|---|---|---|
| 4 步 pipeline 划分 | 设计决定 | 低 |
| 弱 resume 协议 | 无 | 中 |
| 数据契约 implicit | 无 schema/version/checksum | 中 |
| 训练/推理两套 forward | 无 parity test | 中-高 |
| build_pkg sanity 极弱 | 无 | 中 |
| requirements.txt 未测 offline | 无 | 中 |
| README/code drift | 多处不符 | 低（不影响功能）|
| 没 CI/CD | 无 | 中 |


---

## §17. 失败 / 未采纳的方向

整个项目最有价值的"知识"之一是**知道哪些方向不 work**。本节系统汇总。

### 17.1 失败方向总览（按主题分类）

#### 17.1.1 决策层 / 阈值

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| OOF DE 阈值 | iter_016 v3 (T106-109) | **+25.08** (−3.08 vs iter_015) | DE 在 OOF 上过拟合 |
| 4-way DE | iter_016 v3 | 同上 | DE + GRU 双重过拟合 |
| per-sym OOF threshold | T126 | **未提交** | 过拟合 OOF 分布 |
| conformal-aware re-DE | T146 | **未提交** | 二次 DE 风险叠加 |
| adaptive per-batch thr | T168 V1-V4 | **未提交** | 触碰约束 #2 |
| 对称 k×mean_abs_pred | T62 | — | 鲁棒但低 2.51 |

#### 17.1.2 Wrapper / TTA / TTT

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| TTA K=5 | T161 | **未提交** | 触碰约束 #2 |
| In-row TTT W3a/b/c | T162 | **未提交**（W3a −0.30, W3b −1.98, W3c −3.11）| 触碰约束 #2 |
| AR TTT | T169 | **未提交** | 触碰约束 #2 |
| In-batch wrapper B/C | T177 | **未提交** | 同上 |
| OOD GMM density abstain | T180 | **未提交**（V1 −32, V2 −57, V3 −78）| 测试集比训练 in-distribution |
| Time-window hard abstain | T181 | **未提交** | T170 conformal 已过滤 |

#### 17.1.3 模型架构

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| 3-class CE | T1, T2 | (mmpc_demo) **−6.65** | 压缩幅度信息 |
| GRU | T95 | iter_016 失败主因 | hidden state 过短 h 过拟合 |
| 第二组 NN T97（同 T87 风格）| T156/T163 | T163 **~+30.74** (−3.70) | T97/T87 同架构 diversity≈0 |
| 4-way (T87+T89+T99+T95) | iter_016 v3 | **+25.08** | GRU 主因 + OOF DE |
| 5-way (+ CatBoost) | T167 | **未提交** | within noise |
| GroupTransformer | T172/T187 | **未提交**（in-sample +22~55 污染）| OOD 风险 |
| Deep CNN raw / hybrid | T184/T185 | **未提交** | OOD 未验证 |
| ResNet-DCN | T94 | — | 探索性 |
| Transformer preliminary | T93 / T157 | — | 探索性 |
| Snapshot ensemble | T135 | **未提交** | 不超 5-seed |

#### 17.1.4 损失函数

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| Direct PnL loss | T13/T134/T57/T60 | 不收敛 | 梯度不稳 |
| GMADL direction-aware | T116 | — | 不超 SPO+ |
| MAE LGB | T137/T143/T145 | 预测 +30~32 | 不超 L2 M7 |
| Huber LGB | T99 | iter_016 失败 | OOF 污染 |
| Quantile LGB α=0.5 | T132 | — | LGB 已 L2 work |
| Magnitude weight | T92 (R44) | — | 假设证伪（幅度加权放大噪音）|
| isotonic / Platt 校准 | T15/T40/R13/T131 | — | 破坏排序质量 |

#### 17.1.5 Ensemble 权重学习

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| Stacker NN | T139 | **未提交** | OOF 过拟合 |
| Bounded NNLS (per-group) | T192 (H4) | **+31.0** (−4.64) | OLS overfit val/test 共同 noise |
| Pure OLS weight | (隐含) | — | NN-NN corr 0.9331 病态 |

#### 17.1.6 Ensemble 扩容

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| 40-NN only | T179 | **+34.39** (−0.25) | 只扩 NN 无 variance reduction |
| 150+150 | T190 | **+34.59** (−1.05) | HP 多样性边际递减后变噪音 |
| LGB 10-seed (vs 5-seed) | T178 | null | 已在 variance floor |

#### 17.1.7 特征

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| Alpha101/Alpha191 | T22 | **未提交**（IC +21.80）| 违反约束 #2 batch shuffle |
| Hawkes 过程 OFI | T141/142 | **未提交** | 增量不显著 |
| TOD (Time-of-Day) | T151 (v9) | 未确认 | T170 final 未用 |
| SchemeQ time features | T74/T78 | — | 回归框架下 LOSO-eq −0.72 |
| Roll/TSRV | R_roll_tsrv | — | 部分入 Stage 3 但未成主力 |
| SG filter | T35/T91 | — | SG 对树模型 gain ≈ 0 |
| Trend features | T191 | **未提交**（−10.33, IC=−0.17）| 完全冗余 + 负 IC |
| logret lag | R4/T124 | — | KILL（无增量）|
| Pairwise 交互 | R_Pairwise_ALL | — | 计算量大边际 |
| sym embedding / per-sym norm | 隐式 | — | 违反约束 #3 |
| date-derived | 隐式 | — | 违反约束 #1 |
| Global z-score | T10 (SchemeF) | — | 不如窗口 z-score (T7) |

#### 17.1.8 调参 / 评估

| 方向 | T 编号 | 平台 | 死因 |
|---|---|---|---|
| LOSO 5-fold (Phase 1) | T1-T35 | — | 过度惩罚 OOD，被 LOSO-equiv 替代 |
| IID split (Phase 0) | — | — | 严重乐观 +19.25 → −22.10 LOSO |
| Pseudo labeling | T65 | — | 负面 |
| LGB monotone constraints | T117 / T123 | 未确认 | 边际 |
| iter_017 (LGB HYD+monotone+date-decay) | T123 | 未确认 | +0.61 边际 |

### 17.2 用户明确判定为"垃圾"的方向（HISTORY_EVIDENCE_DB.md 主题 10）

| 方向 | 用户立场 |
|---|---|
| TimesNet | 通用时序模型，LOB 上未实证 |
| iTransformer | 同上 |
| PatchTST | 同上 |
| N-BEATS | 同上 |

### 17.3 失败方向的元教训

1. **OOF DE 过拟合**：[§3.5](#35-为啥不再做-de--oof-threshold5-重证据) 5 重证据
2. **新架构需要 OOD 验证**：M7 retrain 只对**已验证**的 T75 LGB / T87 NN 有效；新架构 (Transformer/CNN) in-sample 数字漂亮但 OOD 风险高
3. **TTA/TTT 反复触碰约束 #2**：6 个 T 实验全部未通过；约束 #2 是真实硬限制
4. **OLS/NNLS 权重学习**：NN-NN corr 0.9331 病态共线性使权重不可识别（[§6.5](#65-为啥不学权重simple-mean--ols--nnls)）
5. **短 horizon 模型独立训也救不了**：α=0.05% < fee=0.02% 双边
6. **trend features 完全冗余**：corr(T191, T188v2) = 0.9941，IC_residual = −0.17

### 17.4 此节 GAP

没有失败方向的 ablation 缺失——本节是已知失败方向的汇总。

---

## §18. 用户提出的核心问题 / 数学正当性挑战（USER-RAISED）

### 18.1 ⭐ Q281 — sqrt(h/60) 阈值缩放方向错误（用户精确推导）

**来源**：`QREVIEW_PREDICTOR.md §K / Q281`（用户亲自提出，已 in-place 写好）

#### 18.1.1 现状

```json
// thresholds.json (h=5/10/20/40 配置)
"thr_up": 0.0003 * sqrt(h/60),
"thr_dn": 0.000216 * sqrt(h/60),
"share_with": 60  // 短 horizon 直接 share pred_h60
```

- 模型回归目标：`y = (mp_{t+60} − mp_t) / (mp_t + 1)` ⇒ pred 是 **60-tick 归一化 Δmid** 的估计
- 短 horizon 直接 share `pred_h60`（**pred 不缩放**）
- thresholds: `thr_h = thr_60 × sqrt(h/60)`（短 h 阈值变小）

#### 18.1.2 用户的精确推导

**drift-with-noise 模型**：
```
E[Δmid_h] = μ · h     (drift 线性于 h)
std[Δmid_h] = σ · sqrt(h)   (noise sqrt(h))
```

若想把 "h=60 预测" 真正用作 "h=h0 决策"：

**正确做法 A**（缩 pred）：
```
pred_h60 × (h/60) > thr_h_native   ⇒   pred 缩 h/60 (线性)
```

**等价做法 B**（反缩 thr）：
```
pred_h60 > thr_h_native × (60/h)   ⇒   thr 反向缩 60/h (变大！)
```

**若 `thr_h_native = thr_60 × sqrt(h/60)`**（按 noise std 缩）：
```
等效 thr = thr_60 × sqrt(60/h)   (缩 60/h, 变大)
```

**代码实际**：
```
thr_h = thr_60 × sqrt(h/60)   (缩 h/60, 变小)
```

→ **方向反了！**

**差距**：与"理论正确"差 `60/h` 倍。
- h=60 时差 1.0 倍（无影响）
- h=40 时差 1.5 倍
- h=20 时差 3.0 倍
- h=10 时差 6.0 倍
- **h=5 时差 12.0 倍！**

#### 18.1.3 问题清单

1. **`_doc` 注释错误**：JSON 里 `_doc` 注释说 "shorter horizons trade fewer/smaller signals"——**实际效果完全相反**（thr 变小 ⇒ trade 更多）
2. **忽略 label 差异**：label_5/10 的 α=0.05% vs label_20/40/60 的 α=0.1%，短 h 的"涨/跌"门槛本来就更严
3. **无 ablation**：sqrt(h/60) vs h/60 vs 1.0（不缩）vs sqrt(60/h)（理论正确方向）的平台对比**完全缺失**
4. **HISTORY_EVIDENCE_DB.md 主题 10 未找到该 ablation 记录**：猜测作者把短 h 当 "score=max(per-horizon) 不会扣分"的安全占位，没认真推导

#### 18.1.4 可能的辩护（站不太住）

- **"pred 是 horizon-agnostic signal strength"** ⇒ thr 按 noise std 缩 sqrt
  - 但 pred 显式回归 60-tick Δmid，这个解读违反训练目标
- **empirical**：作者可能跑过几种 scaling，sqrt 在平台上表现最好
  - 但 history 里看不到此 ablation

#### 18.1.5 为啥现状能 "work"

1. **平台 score = max(per-horizon)** ⇒ 短 h 即使更差也不影响 +35.64 排名
2. **短 h 有 conformal 弹性带兜底**（per-sym β·σ 加在 thr 上）
   - 详 `QREVIEW_PREDICTOR.md Q23`：at h=5, sym=1，band = 1.88e-4，thr = 8.66e-5
   - **band 是 thr 的 2.17 倍** → `effective_thr = thr + band ≈ 2.75e-4 ≈ 原始 h=60 thr`
   - **hi-band sym 在短 h 上几乎抵消 sqrt 缩放**
3. **对低 band sym（sym3/4 β=0）sqrt 缩放完全生效** → 不同 sym 在短 h 上有不同 effective scaling，**意外副作用**

#### 18.1.6 结论

> **sqrt(h/60) 在数学上没有清晰正当性，方向与 random-walk-with-drift 理论预测相反。**

#### 18.1.7 用户建议的补做实验

> 在不动 h=60 的情况下，submit 4 个变体：
> 1. thr 缩 `h/60`（线性缩，与 drift 一致）
> 2. thr 缩 `1.0`（不缩，pred 量级直接比）
> 3. thr 缩 `sqrt(60/h)`（理论正确方向，反向缩）
> 4. 关掉短 h 提交（只交 h=60）
>
> 对比短 h 平台分。如果其中任何一个让短 h 平台分超过 h=60 的 +35.64，就会**改变 max-rank 结果**。

**重要性**：如果某个变体让 h=5 平台分超过 +35.64，意味着我们一直在 leave money on the table。

#### 18.1.8 此问题的严重性评估

| 维度 | 评估 |
|---|---|
| 数学正确性 | **错误**（方向反了，差 60/h 倍）|
| 影响主分数 | **小**（max 规则隐藏了短 h 的问题）|
| 影响潜在分数 | **未知**（可能短 h 优化后 max-rank 改变）|
| 是否易修复 | **是**（改 thresholds.json 一行 + 重提交）|
| 是否做过 ablation | **否** |

#### 18.1.9 修复建议

如果决定修复，最小成本路径：

```python
# Option A: 改 thr 缩放方向
thr_h_up = thr_60_up * sqrt(60.0 / h)   # 反向缩，变大
thr_h_dn = thr_60_dn * sqrt(60.0 / h)
```

或

```python
# Option B: 改 pred 缩放（更符合 drift 假设）
pred_for_h = pred_h60 * (h / 60.0)      # 线性缩
# 然后用 native thr (h=60 的值) 直接比
```

或

```python
# Option C (最保守): 关闭短 horizon 提交，只交 h=60
# 既然 max 规则保护，短 h 不优化也不损失
```

### 18.2 其他用户/reviewer 提出的数学/逻辑层面根本质疑

#### 18.2.1 训练 GELU vs 推理 GELU 不一致（CODE Q176）

- **位置**：`Predictor.py:171` (tanh approx) vs `train_T188v2_nn_seed.py:89` (默认 erf)
- **数学差异**：|x| > 1 范围 ~1e-4
- **风险**：4 层 hidden 累加后可能 flip action（thr=3e-4 时 1e-3 误差敏感）
- **是否做过 ablation**：**无**
- **修复成本**：改 1 行（`F.gelu(h, approximate='none')` 或训练改 tanh）
- **严重性**：中（潜在 bug，影响所有 50 个 NN 推理）

详见 [§9.7](#97-code-q176-训练-gelu--推理-gelu潜在-bug)

#### 18.2.2 amount_delta 处理不一致（FEATURES Q20 / CODE Q47-Q48）

- **位置**：`build_schemeP_cache.py:108-110` (raw_last log1p) vs `fast_features_batch.py:311, 436` (X3d 原始)
- **问题**：同一底层量在不同特征里赋予不同尺度的表示
- **是否做过 ablation**：**无**
- **是否影响 PnL**：**未量化**，可能小（log1p vs raw 是 redundancy 而非冲突）
- **严重性**：中（设计/bug 未明，但 work）

详见 [§13.7](#137-amount_delta-处理不一致code-q47-q48--features-q20)

#### 18.2.3 T3 MLOFI vs Stage5 OFI 公式不一致（FEATURES Q26-Q29）

- **位置**：`fast_features_batch.py:159-184` (T3 MLOFI) vs `:645-650` (Stage5 OFI)
- **问题**：两套 OFI 实现风格不同（一个用 >= 同时计 eq，一个用 strict > / < 分开处理 eq）
- **是否做过 ablation**：**无**
- **严重性**：中（可能 redundancy 或互为校验）

详见 [§13.8](#138-t3-mlofi-vs-stage5-ofi-公式不一致features-q26-q29-q105-q106)

#### 18.2.4 weighted SPO+ 凸性（CODE Q196）

- **位置**：`train_T188v2_nn_seed.py:460-461`
- **问题**：weighted SPO+ 是否仍是凸 surrogate？理论上**未证明**
- **是否影响**：empirically work，未影响收敛
- **严重性**：低（理论 gap 但 empirically OK）

#### 18.2.5 Fisher consistency of SPO+ in 1D EV gate（CODE Q174）

- **问题**：SPO+ (Elmachtoub & Grigas 2022) 原文针对线性 LP；改成 1D scalar 是否仍 Fisher consistent？
- **答案**：未证明，empirically work
- **严重性**：低（理论 gap 但 empirically OK）

### 18.3 USER-RAISED 章节汇总表

| 问题 | 来源 | 数学严重性 | 影响 PnL | 是否 ablation | 是否易修复 |
|---|---|---|---|---|---|
| **Q281 sqrt 缩放方向** | 用户 | **高**（方向反了 12 倍）| max 规则隐藏 | **否** | 是 |
| GELU erf vs tanh | CODE Q176 | 中（数值差异）| 可能小 | 否 | 是 |
| amount_delta 不一致 | FEATURES Q20 | 中（设计/bug 未明）| 未量化 | 否 | 中 |
| OFI 公式不一致 | FEATURES Q26-Q29 | 中（一致性）| 未量化 | 否 | 中 |
| weighted SPO+ 凸性 | CODE Q196 | 低（理论）| 0 | 否 | 不需修 |
| SPO+ 1D Fisher consistency | CODE Q174 | 低（理论）| 0 | 否 | 不需修 |

---

## §19. 已知风险与 GAP 总结表

### 19.1 高风险 GAP（建议优先做 ablation）

| 设计点 | 当前选择 | 有无 ablation | 风险评估 | 修复方向 |
|---|---|---|---|---|
| **sqrt(h/60) 缩放方向** | `thr_60 × sqrt(h/60)`（变小）| **无** | **高**（数学方向反了）| ablate 4 个变体，确定最优后修 thresholds.json |
| **GELU erf vs tanh 训练-推理不一致** | 训练 erf，推理 tanh | **无** | **中-高**（潜在 bug）| 修推理为 `approximate='none'` 或训练改 tanh |
| **HP cycling 实际 12 unique combos（非 36）** | (S-1)%3 + (S-1)%4 + (S-1)%3 | 无设计意识 | 中 | 改 batch idx 用 (S-1)%2 解耦 |
| **per-sym σ calibration 流程不透明** | thresholds.json 写死 | 无 script | 中 | 加 calibration script 到 final_submission_code |
| **训练/推理 forward 无 parity test** | 两套独立实现 | 无 | 中-高 | 加 unit test 验证 1e-6 max diff |
| **build_pkg sanity check 极弱** | 只 check 文件存在 | 无 | 中 | 加 schema + dim + seed pairing check |

### 19.2 中风险 GAP（可考虑修但成本/收益比一般）

| 设计点 | 当前选择 | 有无 ablation | 风险评估 |
|---|---|---|---|
| amount_delta raw_last log1p vs X3d 原始 | 不一致 | 无 | 中（设计/bug 未明）|
| T3 MLOFI vs Stage5 OFI 公式 | 不一致 | 无 | 中（可能 redundancy）|
| FAIL_NAMES drop 标准（KS test 阈值）| 未文档 | 无 | 中 |
| FEE=0.0001 / fee_eff 公式假设双边 | 无平台 spec | 无 sensitivity | 中 |
| w_lgb=1.5 / w_nn=1.0 在 50+50 重 sweep | 沿用 iter_015 grid | 无 | 中 |
| short horizon 单独 w 调整 | 沿用 1.0/1.5 | 无 | 中 |
| Phase 2 target_scale 沿用 Phase 1 | 沿用 | 无 | 低-中 |
| Phase 2 dropout 仍开启 | 启用 | 无 | 低-中 |
| short horizon hi-band sym 几乎抵消 sqrt 缩放 | 副作用 | 无 | 中 |
| requirements.txt 未测 offline pip install | --extra-index-url | 无 | 中 |
| 平台 OS/arch/python 假设未文档 | 未文档 | 无 | 中 |

### 19.3 低风险 GAP（影响小或已被反例 cover）

| 设计点 | 当前选择 | 有无 ablation | 风险评估 |
|---|---|---|---|
| thr_up=0.0003 / thr_dn=0.000216 具体值 | T76 DE 搜出 | 无完整 grid | 低（多次小动均未超）|
| 不分 sym 阈值 | 全局固定 | T126 反例 | 低 |
| LGB 330 iters | V4 best_iter × 1.1 | 无对照（200/500）| 低 |
| LGB 5 HP grid 手挑 | 5 组 | 无完整 grid | 低 |
| 各 LGB HP 沿用同 330/11 | 沿用 | 无 | 低 |
| Phase 1 early stop epsilon=1e-10 | 偷懒 | 无 | 低 |
| Phase 2 batch=4096 不 cycle | 固定 | 无 | 低 |
| aug_a per-feature 独立缩放 | 独立 | 无 sample-level 对比 | 低 |
| MLP [256, 128, 64] 维度 | (T81 设计) | 无 width/depth sweep | 中 |
| MLP vs GRU/Transformer | MLP | T95/T172/T184 反例 | 低 |
| AdamW vs SGD | AdamW | 无对比 | 低 |
| CosineLR vs StepLR | Cosine | 无对比 | 低 |
| Kaiming relu init for GELU | nonlinearity="relu" | 无对比 | 低 |
| Simple mean (不学权重) | mean | T192 NNLS 反例 | 低 |
| 异构模型零容忍 | 拒绝 CatBoost/GRU/Transformer | 历史 ablation 全负 | 低 |
| 不留 val M7 retrain | 设计 | 无对比 | 低（+5.51 实证）|
| LGB CPU sequential 不优化 | 无优化 | 无 | 低（84ms 足够）|
| npz vs pt | npz | 设计决定 | 低 |
| fast_features.py dead code | 未删 | 无 | 低 |
| README/code drift | 多处不符 | 无 | 低 |
| 没 CI/CD | 无 | 无 | 低 |
| L2 vs MAE/Huber/Quantile | L2 | 全部对比 | 低 |
| Δmid 回归 vs 3-class CE | 回归 | T1/T2/T75 对比 | 低 |
| 不用 isotonic/Platt | 无 | T15/T40/R13 反例 | 低 |
| 没特征剪枝 | 359 维 | T152 持平 | 低 |
| 没 Hawkes/Alpha101/TOD/Trend | 不用 | 全部 ablation | 低 |
| 窗口 z-score vs 全局 | 窗口 | T7 vs T10 | 低 |
| sym-agnostic 设计 | 严格遵守 | T22/T161/T162/T168/T169/R_cross_sym 反例 | 低 |
| FAIL 特征仍计算（kyle_lambda 等浪费 compute）| 未优化 | 无 | 低 |

### 19.4 关于 GAP 的方法论说明

**为啥这么多 GAP**：
1. 项目 5 月跑了 200+ 实验 + 22 次平台提交，但**单次平台提交成本高**（人工 + 时间），不可能为每个细节都开"4 个变体 ablation"
2. 平台提交频次受限（公榜），每次提交都战略性选择最大可能增益的变体
3. 整个项目"穷尽 ablation"和"覆盖最多 trick"之间被迫 trade-off

**如何看待 GAP**：
- "无 ablation" ≠ "错"
- "无 ablation" = "我们不知道更优是否存在"
- 高风险 GAP（[§19.1](#191-高风险-gap建议优先做-ablation)）是后续值得花平台提交配额验证的方向

---

## §20. 附录 — 关键文件路径与 T 编号速查

### 20.1 final_submission_code 文件结构

```
final_submission_code/
├── 01_build_features/
│   ├── build_schemeP_cache.py          # 240 行，输出 schemeP_*.npz + schemeP_feat_names.txt
│   ├── fast_features_batch.py          # 800+ 行，216 维 derived features
│   └── stage5_features.py              # dead code（已整合到 fast_features_batch）
├── 02_train_lgb/
│   ├── train_T188v2_lgb_seed.py        # 209 行，单 seed 训练
│   └── run_all_lgb_seeds.sh            # 32 行，串行 50 个 seed
├── 03_train_nn/
│   ├── train_T188v2_nn_seed.py         # 510 行，Phase 1 + Phase 2
│   └── run_all_nn_seeds.sh             # 同上
├── 04_build_pkg/
│   ├── build_pkg.py                    # 120 行，打包 pkg/
│   ├── Predictor.py                    # 400 行，推理类
│   ├── fast_features.py                # 196 维（dead code，不被 Predictor 用）
│   ├── fast_features_batch.py          # 与 01 字节相同
│   ├── thresholds.json                 # 60+ 行，所有阈值/conformal/权重配置
│   ├── config.json                     # 路径/horizon 配置
│   └── requirements.txt                # CPU torch 2.5.1 + numpy 2.4.4 + pandas 2.3.3 + lightgbm 4.6.0 + scipy 1.14.1
├── README.md
└── run_pipeline.sh                     # 串联 01→02→03→04

pkg_T188v2_50plus50/                    # 运行时打包目标
├── Predictor.py
├── fast_features.py
├── fast_features_batch.py
├── thresholds.json
├── config.json
├── requirements.txt
├── model_h60_seed{1..50}.txt           # 50 LGB boosters
├── nn_h60_seed{1..50}.npz              # 50 NN 权重
└── manifest.json                        # ensemble_seeds 列表
```

### 20.2 三次大突破对应的 T 编号

| 突破 | T | iter | 平台 |
|---|---|---|---|
| 1st (Δmid regression) | T75 | iter_013 | +19.23 |
| 2nd (SPO+ DFL) | T87 | iter_015 | +28.16 |
| 3rd (M7 LGB) | T140 | iter_019 v2 | +34.44 |
| SOTA (50+50 ensemble) | T188v2 | iter_019 v2N 50+50 | **+35.64** |

### 20.3 22 次平台提交完整表（按时间顺序）

见 `HISTORY_EVIDENCE_DB.md` 附录 C。

### 20.4 各 T 编号速查（Phase 0-7）

见 `HISTORY_EVIDENCE_DB.md` 附录 D。

### 20.5 R 系列调研列表

40+ R 系列（包括 R1/v2/R10/R13/R30/R31/R34/R_T75L2/R_conformal_select/R_full_retrain/R_multinn_spo/R_multitick_window/R_roll_tsrv/R_stack3_compound/R_time_of_day 等）。详见 `HISTORY_EVIDENCE_DB.md` 附录 D。

### 20.6 平台提交锚点（按 ID）

| iter / submission | 文件名 | MD5 | 平台 |
|---|---|---|---|
| iter_013 | submission_050707_iter013.zip | — | +19.23 |
| iter_015 | submission_050802_iter015.zip | — | +28.16 |
| iter_016 v3 | submission_050815_iter016_v3.zip | — | +25.08 |
| iter_018 v1 | submission_050817_iter018_v1.zip | — | +28.93 |
| iter_019 v2 (T140) | submission_050818_iter019_v2_fullretrain_conformal.zip | — | +34.44 |
| iter_019 v2N + T87M7 (T170) | submission_050909_iter019_v2N_T87M7.zip | — | +34.64 |
| iter_019 v2N + No Conformal (T182) | submission_050909_iter019_v2N_T87M7_NoConformal.zip | — | +34.34 |
| iter_019 v2N + 40NN (T179) | submission_050910_iter019_v2N_40NN_bag.zip | — | +34.39 |
| **iter_019 v2N + 50+50 optimized (T188v2)** | submission_050911_iter019_v2N_50plus50_optimized.zip | **95993e6250ea56361dfecb62c9123e63** | **+35.64** ⭐ |
| iter_019 v2N + 50+50 cputorch | submission_050911_iter019_v2N_50plus50_cputorch.zip | c67e4d25665a2a7a12f4f6839cc8d0fe | +35.64 |
| 150+150 H4 NNLS (T192) | submission_050912_iter019_v2N_150plus150_H4weights_cputorch.zip | — | +31.0 |

---

## 总结

本报告把 7 份白板 review 提出的约 1091 个独立问题（去重合并为 ~250 个核心 Question）与 200+ 实验、22 次平台提交的证据做了系统 JOIN，覆盖了最终 T188v2 50+50 SOTA 提交（平台 +35.64）的全部核心决策。

**3 次大突破** + **2 次失败 lesson**（iter_016 OOF DE / T192 NNLS）+ **6 个高风险 GAP**（其中 sqrt 缩放方向是用户精确推导的关键问题）+ **22 个其他失败 ablation** 构成了项目的完整知识图谱。

最重要的 take-away：
1. **数据驱动、保守迭代**：thr_up/thr_dn 一旦定下来不再动；新模型/loss 必须 OOD 验证
2. **simple > complex**：simple mean ensemble > NNLS / stacker；MLP > Transformer/CNN
3. **承认 in-sample 污染**：T182 教训 — 本地 +6.62 但平台 −0.30
4. **平台是唯一裁判**：本地数字只能"看相对 delta"，不能信绝对值
5. **数学正当性 ≠ empirically work**：Q281 sqrt 缩放方向错但因 max 规则隐藏；GELU erf/tanh 不一致但 work（不知道损失多少）
6. **失败方向的知识价值**：约 30 个失败方向构成"什么不行"的核心数据库

