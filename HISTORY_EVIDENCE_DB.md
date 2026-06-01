# 良文杯 2026 — 历史证据库（HISTORY_EVIDENCE_DB.md）

> **用途**：把最终 SOTA 提交（T188v2 50+50 ensemble = 平台 **+35.64**）中的每一个"魔法数字 / 设计决策"反向 join 到 200+ 实验、40+ R 系列调研、22 次平台提交的具体证据。
>
> **数据源**：`0512_final_report.md`（1105 行）、`0512_tricks_table.md`（170 行 + 192 trick + R 系列）、`PROGRESS.md`、`report.md`、`CRITICAL_CONSTRAINTS.md`、`experiments/T*/results.json`（242 个）、`experiments/R*/`（40+）、`audits/` 5 份。
>
> **数字标注规约**：
> - 「LOSO」= 5-fold Leave-One-Sym-Out（Phase 0-1，T1-T35）
> - 「LOSO-equiv」= 全 sym 训练 + DE 阈值搜索在 442k 行本地测试集上（Phase 2，T44-T74）
> - 「LOSO-equiv + V4」= train 0-79 / val 80-95 / test 96-119 walk-forward（Phase 3-5，T75-T155）
> - 「holdout in-sample」= date 0-119 全数据训练，date 96-119 评分（Phase 6-7，T156-T192，**含 in-sample 污染**）
> - 「平台」= 公榜得分（5 horizon 取最高，h_60 几乎总是最高）
> - 不同阶段的本地数字**不可直接比较**；平台 OOD 是唯一 ground truth。

---

## 0. 全局时间线速览（22 次平台提交）

| # | iter / T-编号 | 主要 trick | 本地 | 平台 PnL | Δ |
|---|---|---|---|---|---|
| 1 | iter_000 mmpc_demo | DeepLOB baseline | 几乎全 1 | **−6.65** | — |
| 2 | iter_002 SchemeC | LGB 多 h + 阈值 gate | LOSO h_60=+6.30 | **+4.07** | +10.72 |
| 11 | iter_013 (T75) | Δmid regression + EV gate | LOSO-eq +36.23 | **+19.23** | +15.16（**1st 突破**）|
| 13 | iter_015 (T87) | SPO+ DFL NN + T75 LGB | LOSO-eq +40.09 | **+28.16** | +8.93（**2nd 突破**）|
| 14 | iter_016 v3 | 4-way (CB+Huber+GRU) | LOSO +43.71 | **+25.08** | −3.08（OOF DE 过拟合）|
| 15 | iter_018 v1 (T127) | + per-sym beta conformal | LOSO +41.49 | **+28.93** | +0.77 |
| 16 | iter_019 v2 (T140) | M7 LGB 全数据重训 | LOSO 不变 | **+34.44** | +5.51（**3rd 突破**）|
| 17 | T163 v2+T97+agree | + 第二组 NN + agreement | holdout +42.90 | **~+30.74** | −3.70（T97 与 T87 同源）|
| 18 | T170 NN M7 | T87 NN M7 全数据重训 | holdout +8.49 | **+34.64** | +0.20 |
| 19 | T182 无 conformal | 去掉 conformal wrapper | holdout +6.62 | **+34.34** | −0.30（证明 conformal 有效）|
| 20 | T179 40-NN | 40-NN ensemble (8 HP × 5 seed) | holdout +0.63 | **+34.39** | −0.25（只扩 NN 无用）|
| 21 | **T188v2 50+50** | **50 NN + 50 LGB mega ensemble** | holdout +0.48 | **+35.64** | **+1.00（当前 SOTA）** |
| 22 | T192 H4 NNLS | bounded NNLS 权重 150+150 | holdout +193.94 | **+31.0** | −4.64（OLS overfit）|

**整个项目总增益**：mmpc_demo −6.65 → T188v2 +35.64 = **+42.29 绝对值**。

---

## 主题 1：阈值（threshold）—— thr_up=0.0003 / thr_dn=0.000216

### 背景
EV gate（`pred > thr_up → 买；pred < −thr_dn → 卖；否则平`）是 T75 回归突破之后整个项目的决策层。阈值定得不好直接吞噬 alpha 或把噪音当信号交易。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T5a** | SchemeB 上 DE 搜索非对称 thr_up/thr_dn | LOSO h_60 +11.11 | — | 早期阈值优化首次显示有效（vs T2 原始 −22.10）|
| **T30** | DE 4D 阈值优化（thr_up/thr_dn/margin/weight）| aug_a 5-seed LOSO +11.46 → **+13.61**（+2.15）| — | DE 在 5-fold LOSO 上可超调，warning 信号建立 |
| **T62** | 无偏对称 k-threshold（thr = k×mean_abs_pred）| k=1.25 时 LOSO-eq **+33.72**（vs DE +36.23，鲁棒但低 2.51）| — | 鲁棒但底数低；DE 仍是首选 |
| **T63** | regression EV gate 参数扫描（绝对值/非对称等）| 边际 | — | 确认原始 EV gate 设计最简洁 |
| **T75** | LGB regression + EV gate（DE 搜索）| LOSO-eq **+36.23** | **+19.23**（iter_013，透传 53%）| **第一次突破**，但 DE → 平台 gap −17 也证明 DE 有过拟合 |
| **T76** | decoupled threshold（thr_up≠thr_dn 分开搜）| 边际正 | — | 最终采用非对称 thr_up=0.0003, thr_dn=0.000216 |
| **T126** | **per-sym OOF threshold**（每只股票独立 OOF 最优）| LOSO 本地高 | — | **过拟合风险极高，未提交**；改用全局 thr + per-sym conformal |
| **T127** | iter_018 v1 = iter_015 + per-sym beta conformal | LOSO-eq +39.75 → **+41.49**（+1.75）| **+28.93**（+0.77 vs iter_015）| 转向 conformal abstain，而非动 thr |
| **T146** | conformal-aware DE re-threshold（v3_mae 上重 DE）| LOSO +41.20 → **+43.96**（+2.76）| — | DE 超调风险高，**未提交平台** |
| **T168** | adaptive threshold V1-V4（per-batch 统计自适应）| holdout T170 → V1-V4 全部退步 | — | **平台测试点顺序被打乱（硬约束 #2），per-batch 统计不稳定** |
| **iter_016 v3** | 4-way OOF DE 4D | LOSO **+43.71**（+3.62 vs iter_015）| **+25.08** | **平台倒退 −3.08**，4D OOF DE 在 442k 本地测试严重过拟合 |

### 最终选择
- **thr_up = 0.0003, thr_dn = 0.000216**（非对称，DE 搜索结果，固定全局）
- 不分 sym（per-sym OOF threshold T126 已证过拟合）
- 仅在 thr 之外叠加 per-sym beta conformal（T127/T150，详见主题 3）

### 选择理由（多源验证）
1. **iter_013 DE → 平台 53% 透传**：DE 阈值在 LOSO 上拿到 +36.23，但平台只拿到 +19.23（gap −17），证明 DE 有 OOF 过拟合
2. **iter_016 v3 灾难**：在 4-way ensemble 上再用 OOF DE 4D，LOSO +43.71 但平台 **−3.08**——验证了 T62 的 warning
3. **T146 未提交**：conformal-aware DE 本地 +2.76，但 PM 决定不冒险（教训沉淀已经够多了）
4. **保守原则**：thr_up/thr_dn 一旦定下来，所有后续大版本（iter_018/019/T170/T188v2/T190）都不动这两个值

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| per-sym OOF threshold | T126 | LOSO 高 | 未提交 | 过拟合每 sym 的 OOF 分布；OOD sym 无 model |
| 4-way OOF DE | iter_016 v3 | LOSO +43.71 | **+25.08** | DE 在 442k 本地测试集严重过拟合 |
| conformal-aware re-DE | T146 | +2.76 | 未提交 | 二次 DE 风险叠加 |
| adaptive per-batch thr | T168 V1-V4 | 全负 | 未提交 | 平台批次乱序破坏 per-batch 统计 |
| 对称 k×mean_abs_pred | T62 (k=1.25) | +33.72 | — | 鲁棒但绝对值低于 DE 2.51 |

---

## 主题 2：Ensemble 组成与权重 —— 50 NN + 50 LGB simple mean, w_NN=1.0 w_LGB=1.5

### 背景
集成是项目末期（T140 之后）的主要增量来源。简单平均还是学权重？多少个 NN/LGB 才够？加入异构模型（CatBoost/GRU/Transformer）能不能涨？这是被反复试错的核心问题，最终答案出乎意料地朴素。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T6/T6b** | 5-seed SchemeA NN ensemble | LOSO +8.87 (vs +6.45, +0.36) | — | 早期边际 |
| **T11** | SchemeC 5-seed h_10 | LOSO +21.86 → +22.22 (+0.36) | — | 边际 |
| **T106-109** | iter_015 v2 4-way (T87+T89+T99+T95) | LOSO +43.71 | **+25.08** | **4-way 失败**：T95 GRU 是主因 |
| **T119/T120** | iter_016 ablation 去掉 GRU | 简单集成更鲁棒 | — | ablation 教训：3-way > 4-way |
| **T123** | iter_017 (T87+LGB HYD+monotone+date-decay) | LOSO +40.70 (+0.61 vs iter_015) | 未确认 | 未提交平台 |
| **T125** | T75 LGB + CatBoost addon | 边际 | — | 不超 T87 组合 |
| **T127** | iter_015 v1（5+5）+ conformal | LOSO **+41.49** | **+28.93** | 5 NN + 5 LGB 是 baseline 集成 |
| **T139** | stacker NN（NN stacking 综合 LGB+NN）| LOSO 负面 | — | stacking overfit OOF |
| **T140/b/c** | M7 LGB（同 T127 集成 + LGB 全数据重训）| LOSO 不变 | **+34.44** | M7 trick 关键，但 ensemble 结构没动 |
| **T156/b/c** | NN swap：T97 替代/叠加 T87 | holdout +41.49 → +43.16 (+1.66) | — | 本地正，但 T97 与 T87 太相似 |
| **T158** | agreement filter（NN 和 LGB 方向一致才出手）| holdout +41.49 → +41.74 (+0.25) | — | 本地小正 |
| **T163** | **v2+T97+agreement combined（5+5+T97+agree）** | holdout **+42.90** (+1.41) | **~+30.74** | **平台 −3.70 vs v2**！T87/T97 架构同源 diversity≈0 |
| **T164** | 4-way (T87+T97+T95+LGB) | holdout +43.26 (+0.34) | — | 仍含 T97，不安全 |
| **T166** | + CatBoost L2 5-seed M7 retrain | holdout +0.39 | — | 边际正 |
| **T167** | 5-way (T87+T97+T95+CB+LGB) | holdout +0.11 over T166 | — | within noise，不入 final |
| **T170** | T87 NN M7 retrain（5+5）| holdout +149.95 (vs +109.68 LGB-only, +8.49 NN add) | **+34.64** | 新 SOTA（短暂）|
| **T178** | LGB 5-seed → 10-seed | holdout **null result** | — | **5-seed 已在 variance floor**；纯扩 seed 无红利 |
| **T179** | **40-NN (8 HP × 5 seed) + 5 LGB** | holdout +0.63 | **+34.39** | **−0.25 vs T170**！只扩 NN 不行 |
| **T183** | NN random init（无 T81 warm）M7 retrain | holdout +0.63（持平 T170）| — | 不同初始化 = diversity 来源之一 |
| **T188** | 50+50 初版（pretrain val/train 未分离）| ~+150 | — | warm-start 问题，被 v2 修正 |
| **T188v2** | **50 NN + 50 LGB simple mean** | holdout **+150.43** (+0.48 vs T170) | **+35.64** | **+1.00 vs T170，当前 SOTA**；NN+LGB 都扩 |
| **T190** | 150+150 ensemble | holdout **+140.62** (−9.81 vs T188v2) | **+34.59** | **−1.05 vs T188v2**，扩容收益非单调 |
| **T192** | bounded NNLS [1/1500, 2/150] per-group 权重学习 | val +180.41, test +193.94, ratio 1.075 | **+31.0** | **−4.64**，OLS overfit val/test 共同 noise |

### 最终选择
- **50 NN + 50 LGB**：5×5 NN HP grid × 2 seeds + 5 HP × 10 seed LGB
- **Simple mean**：每一侧内 simple mean，组间 w_NN=1.0, w_LGB=1.5
- **Conformal abstain wrapper** 在最外层
- **不加** CatBoost / Huber / GRU / Transformer

### 选择理由（多源验证）
1. **w_LGB=1.5 起源**：早期 iter_015 的 SPO+ NN 和 LGB ensemble 时 grid search 出 w_LGB > 1 更好，固化后没动；50+50 后再验证仍然 optimal（T188v2 在 val 上比 1:1 高 +0.几）
2. **NN+LGB 都扩才有效**（关键发现）：T179（只扩 NN→40）平台 −0.25，T188v2（两侧都扩→50/50）平台 +1.00，验证 variance reduction 必须两侧对称
3. **5→50 vs 50→150 非单调**：T188v2 +1.00（有效），T190 −1.05（反效果）；HP 多样性边际递减，过多 HP 引入 bias
4. **Simple mean 胜 NNLS**：T192 严重 overfit，val/test ratio 1.075 是 in-sample noise alignment 的烟雾弹

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| 4-way (T87+T89+T99+T95) | iter_016 v3 | LOSO +43.71 | **+25.08** | GRU 容易过短 h 过拟合；OOF DE 过拟合 |
| 第二组 NN T97 | T163 | holdout +42.90 | **~+30.74** | T97/T87 同架构 → diversity≈0 |
| 4-way + T97 | T164 | +0.34 | 未提交 | 仍含 T97，不安全 |
| 5-way (+ CatBoost) | T167 | +0.11 over T166 | 未提交 | within noise |
| 40-NN（只扩 NN）| T179 | +0.63 | **+34.39** | 只扩一侧无 variance reduction |
| 150+150 | T190 | **−9.81** vs T188v2 | **+34.59** | 收益非单调，HP 多样性变噪音 |
| bounded NNLS 权重 | T192 | val/test 1.075 | **+31.0** | OLS overfit；NN-NN corr 0.9331 病态矩阵 |
| stacker NN | T139 | LOSO 负 | 未提交 | OOF 过拟合 |
| Snapshot ensemble | T135 | 边际 | 未提交 | 不超 5-seed |
| GroupTransformer add | T187 Plan B | in-sample +12.6/14.7 | 未提交 | 全 in-sample 污染 |

### 未被采纳的关键失败
- **OLS/NNLS 权重学习**：T192 NN-NN avg corr **0.9331**，LGB-LGB **0.8433**，condition number **54182** — 共线性消除权重可识别性，任何"最优"权重都是 val 噪声
- **150+150 扩容**：HP 多样性扩到 30 HP × 5 seed LGB 后，新 HP 实际上不稳定，反而拖低
- **second NN 同架构**：T163 教训 — diversity 必须架构层面，不是 seed/HP 层面

---

## 主题 3：Conformal abstain + per-sym β

### 背景
平台 OOD 上，模型对部分 sym 的预测置信度较低（特别是 sym=2 蓝筹/ETF 和 sym=1）。Conformal 弃权（abstain wrapper：`|EV_pred| < β_sym × σ_pred` 时 action=1 不出手）能过滤掉这部分噪声预测。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T15** | isotonic + EV gate | LOSO h_10 +21.86 → +20.84 (**−1.02**) | — | 校准破坏排序质量，**早期校准全部负面** |
| **T131** | isotonic addon (iter_015) | LOSO 小正 | — | 不超 conformal |
| **T40** | probability calibration | LOSO 持平/负 | — | platt scaling 负面 |
| **R13** | calibration + position sizing 调研 | isotonic/platt 均负 | — | beta conformal 后续证有效 |
| **T126** | per-sym OOF threshold | LOSO 本地高 | — | **过拟合，未提交** |
| **T127** | **iter_018 v1 = iter_015 v1 + per-sym beta conformal** | LOSO-eq +39.75 → **+41.49** (+1.75) | **+28.93** (+0.77 vs iter_015) | conformal wrapper **首次验证有效** |
| **T128** | conformal v2 研究 + M7 LGB 可行性 | — | — | 同时催生 M7 trick |
| **T150** | **conformal beta sweep** (4-fold CV consensus) | sweep over β∈[0, 0.5] 每 sym | — | **确定最优 β = {sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}** |
| **R_conformal_select** | per-sym conformal beta 搜索调研 | 同 T150 结论 | — | 固化进所有后续提交 |
| **T159/b** | per-sym dynamic abstain（动态计算 β）| holdout null | — | 无效 |
| **T182** | **去 conformal ablation (T170 - conformal)** | holdout T170 +149.95 → **+156.56** (**+6.62**！) | **+34.34** (**−0.30 vs T170 +34.64**) | **关键 ablation**：本地 holdout 显示 conformal "hurt" 是 in-sample artifact，平台 OOD 上 conformal 实际有用 |
| **T177** | wrapper tricks B/C（per-batch 统计 wrapper）| holdout 正 | — | 未入 final |

### 最终选择
- **per-sym β = {sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}**
- 形式：`|EV_pred| < β_sym × σ_pred → action=1`
- σ_pred = 5 NN（或 50 NN）的预测标准差
- 固化进 iter_018 / iter_019 v2 / T170 / T188v2 / T190 等所有后续

### 选择理由（多源验证）
1. **T150 sweep 直接结果**：4-fold CV consensus 在 β∈{0.0, 0.1, 0.2, 0.3, 0.4, 0.5} 网格上，每 sym 独立选最佳；CV consensus 而非单 fold OOF（避 T126 过拟合）
2. **sym1/sym2 abstain 重**：sym=2 是蓝筹/ETF（T8 诊断：amount_delta z-score +7.49 偏离），预测器在其上 brittle，β=0.30 弃权
3. **sym3/sym4 不弃权**：β=0，模型在这两只上预测最稳，conformal 弃权反损 PnL
4. **平台 OOD 净效果 +0.30 ~ +0.77**：iter_018 vs iter_015 +0.77；T170 vs T182 +0.30
5. **本地负、平台正的 paradox**：T182 本地 holdout 显示无 conformal +6.62，但平台 −0.30 → in-sample 训练数据上 abstain 损失真实 PnL，OOD 上 abstain 过滤错误预测；这是 in-sample 评测无法预测平台行为的经典案例

### OOD sym（不在 0-4 训练范围内）的处理
- 平台说明 "sym：可能的范围 0-4（可能有不来自于 5 只训练集股票的数据）"——同一 ID 可能映射不同股票
- 当前 conformal β 按 sym ID 索引应用（β_sym = β[sym_id]）；如果 ID 是 0-4 但底层是新股票，β 可能不再 optimal
- **降级方案（未实现，备选）**：在 R_T75L2_advval / R_conformal_select 中讨论过用全局 β=0.20 兜底
- **测试外 sym 鲁棒性**：T149 NN OOD sweep 确认 T87 NN 在 OOD sym 下表现可接受（不崩）；模型完全 sym-agnostic（无 sym feature、无 per-sym normalization）

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| isotonic + EV gate | T15/T131 | −1.02 / 小正 | 未确认 | 破坏排序质量 |
| platt scaling | T40 | 持平/负 | 未确认 | 同上 |
| per-sym OOF threshold | T126 | 本地高 | 未提交 | 过拟合 |
| per-sym dynamic abstain | T159/b | null | 未提交 | 动态 β 不稳 |
| time-window hard abstain | T181 | V1 −23.39 | 未提交 | T170 conformal 已过滤；硬弃权适得其反 |
| OOD GMM density abstain | T180 | V1 −32, V3 −78 | 未提交 | 测试集比训练 in-distribution；低密度反高 PnL |

---

## 主题 4：M7 全数据重训（第三次大突破）

### 背景
模型在 V4 walk-forward（train 0-79, val 80-95, test 96-119）上调好超参后，**在 date 0-119 全部数据上重新训练**（"M7 trick"）。这把 80-119 这部分新数据补进了训练集，理论上等于多 1/3 训练样本。问题是：会不会过拟合？早停 epoch 怎么定？是否需要保留 val 集？

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **R_full_retrain** | M7 full-retrain 可行性研究 | LGB 量化 330 iters 早停一致 | — | **确认 LGB M7 安全可行** |
| **T128** | conformal v2 + M7 LGB 可行性探索 | LOSO 不变 | — | M7 trick **起点**：LGB 量化测试 |
| **T140/T140b/T140c** | **iter_019 v2 = iter_018 v1 + LGB M7 全数据重训** | LOSO-eq +41.49 (不变；M7 不改 LOSO，只改全数据模型) | **+34.44** (**+5.51 vs iter_018 +28.93**) | **第三次大突破**，M7 LGB trick 首次确认 |
| **T142** | Hawkes 5-seed full-retrain | LOSO 边际 | — | Hawkes 特征上 M7 |
| **T145/b** | iter_019 v4 MAE LGB M7 retrain | 预测平台 +30~32（低于 v2）| 未提交 | MAE M7 LGB 不超 L2 M7 LGB |
| **T155** | v2 replica reproducibility test | LGB 确定性确认 | — | M7 LGB 100% deterministic |
| **T166** | CatBoost L2 5-seed M7 retrain | holdout +0.39 | — | 边际正，后被 50+50 替代 |
| **T170** | **T87 NN M7 full-retrain（5 seeds, date 0-119，固定 epoch=11）** | holdout v2 +109.68 → v2N +118.16 (+8.49, in-sample 污染) | **+34.64** (**+0.20 vs v2 +34.44**) | **NN M7 trick 确认**，但增量小（NN 已 warm-start）|
| **T172** | GroupTransformer M7 retrain | in-sample +149.91 (污染) | 未提交 | 新架构 M7 retrain 高风险 |
| **T176/b** | T87 NN M7 HP variant sweep (ep/lr/lambda) | 各 variant holdout 149-151 噪音 | S1_ep15 **+34.51** (−0.13 vs T170) | 单 variant 无优势，但提供 HP 多样性 |
| **T183** | NN random init（无 T81 warm-start）M7 retrain | holdout +0.63 | — | 随机初始化 M7 与 T81 warm 持平 |
| **T187** | GroupTransformer add to v2 stack | in-sample +12.6/14.7 (污染) | 未提交 | 全 in-sample，新架构 M7 retrain 高风险 |

### 最终选择
- **LGB M7**：在 date 0-119 全数据上重训，**固定 iters = 330**（V4 walk-forward 早停的 5 seeds 平均 best_iter × 1.1）
- **NN M7**：在 date 0-119 全数据上重训，**固定 epoch = 11**（V4 walk-forward 5 seeds 平均 best_epoch=9.2 × 1.1，向上取整）
- **不留 val 集**：M7 retrain 时全部 120 天都进训练
- **沿用 V4 阶段的所有 HP**：learning_rate, num_leaves, max_depth 等全照搬

### 选择理由（多源验证）

#### LGB 330 iters
- T128 + R_full_retrain 调研：V4 walk-forward 5 seeds 平均 best_iter ≈ 300 → ×1.1 = 330
- 量化测试：330 iters 上 train loss vs 300 iters 相差小，模型未过拟合（GBDT 在大数据上单调收敛）
- T140 平台 **+5.51** 直接验证：iter_018 +28.93 → iter_019 v2 +34.44，比 LOSO 阶段的 0 增量大得多 → M7 增量只在 OOD 上兑现

#### NN 11 epoch
- T170 设计：V4 walk-forward 5 seeds 平均 best_epoch ≈ 9.2，× 1.1 ≈ 10.1，向上取整 **= 11**
- T176 sweep 确认：ep={7, 11, 15, 20}, lr={1e-5, 3e-5, 1e-4}, λ={10, 30, 100} 在 holdout 上 149-151 噪音级，没有单一 variant 显著超 ep=11
- T176 S1_ep15 平台 +34.51（−0.13 vs T170 +34.64），证 ep=15 略差

#### 不留 val 集
- R_full_retrain 调研结论：留 val 集 = 浪费 12% 训练样本，且 96-119 是 platform OOD 最相近时段，更值得用作训练
- 类比 Kaggle 的标准做法：先 CV 选超参，最后 retrain on all data without holdout
- T140 LOSO 不变（M7 改的只是全数据模型，不改 LOSO eval）但平台 +5.51 → 直接证 retrain on all data 比留 val 优

#### M7 增量为啥 LGB > NN
- LGB M7 平台 **+5.51**（iter_018 → iter_019 v2）
- NN M7 平台 **+0.20**（v2 → T170）
- 原因：NN 已从 T81 L2 warm-start，加上 SPO+ DFL，M7 带来的增量小；LGB 从头训，M7 带来的新数据增量大
- T183 NN random init M7 与 T81 warm-start M7 在 holdout 持平 (+0.63)，进一步证 warm-start 的相对增益已饱和

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| MAE LGB M7 | T145/b | 预测 +30~32 | 未提交 | 不超 L2 M7（v2 +34.44）|
| 新架构 M7 (Transformer) | T172/T187 | in-sample +22~55 (污染) | 未提交 | 全 in-sample，OOD 风险高 |
| 留 val 集 | 早期 V4 | LOSO-eq +26.44 | 未在 M7 阶段做 ablation | M7 设计就是不留 val |
| Hawkes 特征 M7 | T142 | 边际 | 未提交 | Hawkes 增量本身边际 |

---

## 主题 5：SPO+ DFL（第二次大突破）—— λ=30, lr=3e-5, NN 11 epoch

### 背景
T75 LGB 回归带来 +9.79，但 LGB 是 surrogate loss（L2），决策层 EV gate 与训练目标不对齐。**SPO+ Decision-Focused Learning** 把梯度直接对齐 PnL，理论上能进一步提升。但实测 direct PnL loss 不收敛（T13/T134），SPO+ 是稳定的两阶段方案。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T1** | DeepLOB-style 3-class CE NN | LOSO ~+8 | — | 早期 baseline；CE 压缩幅度信息 |
| **T13** | **PnL-aware 端到端 NN loss**（直接 PnL 梯度反传）| 不收敛 | — | 端到端 PnL 梯度不稳定 |
| **T19** | NN regularized (dropout + weight decay) | 持平 | — | 单纯正则化无效 |
| **R10** | PnL-aware loss 10 方案调研（SPO+/Huber/分位数/Y-shape）| SPO+ 最稳 | — | **指导 T87 路线选择** |
| **T57** | PnL-aware loss（早期尝试）| 不稳定 | — | 同 T13 |
| **T60** | RL NN PnL | 探索性 | — | 不收敛 |
| **T75** | **LGB regression Δmid + EV gate**（**第一次突破**）| LOSO-eq **+36.23** (+9.79) | **+19.23** | 回归范式确立 |
| **T81** | NN L2 regression on Δmid（MLP [359→256→128→64→1] LayerNorm GELU Dropout 0.10 L2 预训练）| LOSO-eq **+35.89**（与 LGB 互补） | — | **warm-start 模型** |
| **T87** | **SPO+ DFL NN fine-tune**（T81 warm-start → SPO+，λ=30, lr=3e-5, 5 seeds）| LOSO-eq NN alone +35.89 → NN+LGB **+40.09** (+4.20) | **+28.16** (iter_015, **+8.93 vs iter_013**) | **第二次大突破**，透传 71% |
| **T116** | GMADL direction-aware loss | LOSO 微弱提升 | — | 不超 SPO+ |
| **T132** | quantile LGB (alpha=0.5 ≈ MAE) | LOSO 边际 | — | 不超 SPO+ NN |
| **T134** | **direct PnL NN loss**（无 SPO+ 正则化）| 不稳定，不收敛 | — | 重复 T13 结论 |
| **T136** | **SPO+ 超参扫描**（λ, lr 等）| 确认 λ=30, lr=3e-5 最优 | — | **HP 固化** |
| **T149** | T87 OOD sym 鲁棒性扫描 | T87 在 OOD sym 下表现可接受 | — | confirm SPO+ OOD ok |
| **T170** | **T87 NN M7 full-retrain**（11 epoch, 5 seeds）| holdout +8.49 vs v2 LGB-only | **+34.64** | M7 + SPO+ NN |
| **T176/b** | T87M7 HP sweep（ep/lr/lambda variant 扫描）| holdout 149-151 噪音 | S1_ep15 +34.51 | 多 HP 提供 ensemble 多样性 |

### 最终选择
- **两阶段训练**：
  - Phase 1 (T81)：L2 regression pretrain on Δmid，11 epoch（或 V4 的 best_epoch × 1.1）
  - Phase 2 (T87)：SPO+ DFL fine-tune，**λ=30, lr=3e-5**，5 seeds（T188v2 扩为 50 seeds）
- **不直接 PnL loss**（T13/T134 证不收敛）
- **架构**：MLP [359 → 256 → 128 → 64 → 1]，LayerNorm + GELU + Dropout 0.10

### 选择理由（多源验证）

#### 为啥两阶段 pretrain + finetune
- T13/T134 直接 PnL loss 梯度不稳定：PnL 公式是分段不连续函数（label=0/1/2 三段），裸梯度方向不一致，loss 不收敛
- T81 L2 pretrain 提供稳定的 representation 起点（NN alone LOSO-eq +35.89 已经很高）
- T87 SPO+ fine-tune：在 representation 已对齐 Δmid 幅度的基础上，用 SPO+ 的 surrogate gradient 把决策层（EV gate）的损失对齐到 PnL
- R10 调研直接给出"SPO+ 最稳"结论，整个 RL/PnL-aware 路线在 10 方案中只有 SPO+ 稳定收敛

#### λ=30, lr=3e-5 怎么 sweep 出来的
- T136 SPO+ arch sweep：扫描 λ ∈ {1, 10, 30, 100, 300}, lr ∈ {1e-5, 3e-5, 1e-4, 3e-4}
- 最优 (λ=30, lr=3e-5) 在 LOSO-eq 上比次优高 +0.3-0.5
- T176 后续在 M7 retrain 阶段重新扫描 λ ∈ {10, 30, 100}，confirm λ=30 仍最优（其他 variant 在 ensemble 中提供多样性，T188v2 50 NN 包含 5 λ × 5 lr × 2 seed = 50）

#### NN 11 epoch（M7 阶段）
- 见**主题 4**：V4 walk-forward 5 seeds 平均 best_epoch ≈ 9.2，× 1.1 = 10.1，向上取整 11
- T176 S1_ep15 平台 +34.51 vs T170 ep=11 +34.64，差 −0.13，证 ep=11 优

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| Direct PnL loss | T13 / T134 | 不收敛 | 未提交 | 梯度不稳，loss 跳 |
| RL NN PnL | T60 | 探索性 | 未提交 | 同上 |
| GMADL direction-aware loss | T116 | 微弱 | 未提交 | 不超 SPO+ |
| Quantile LGB (α=0.5) | T132 | 边际 | 未提交 | LGB 已在 L2 上 work |
| L2 NN only (no SPO+) | T81 alone | LOSO-eq +35.89 | 未单独提交 | 不超 SPO+ +40.09 |
| MAE LGB | T137 | +0.3-0.5 边际 | 未提交 | 平台未确认超 L2 |
| 3-class CE NN | T1 / DeepLOB | LOSO ~+8 | −6.65 | 压缩幅度信息 |

---

## 主题 6：特征工程总览（SchemeA → SchemeP 演变 + R34 Stage 1-5）

### 背景
特征从 154 维 SchemeB 起步，逐步扩到 359 维 SchemeP（R34 5 个 Stage 累计 +205 维 sym-invariant 特征）。最后还做了 11 个特征的 drop（feature pruning T147 + adversarial filter T152），但最终包仍是 359 维全集。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T2** | LGB SchemeB 154 维 baseline | LOSO sum −22.10 | — | 跨 sym OOD 极差 |
| **T3** | 特征 v1：+72 维（MLOFI/WMP/RV/EWMA/time）= **SchemeC 226 维** | LOSO h_10 +21.86 | — | 进入 iter_002 |
| **T7** | **窗口内 z-score（100-tick）= SchemeD1 308 维** | LOSO +13.91 (sym=2: −0.98 → +2.24) | — | **修复 sym=2 brittleness** |
| **T8** | sym=2 brittleness 诊断 | amount_delta z-score +7.49 偏离 | — | sym-agnostic 归一化方向确立 |
| **T9** | **SchemeE**（amount_delta z-score + spread-norm）| 部分入 SchemeP | — | sym-agnostic 增强 |
| **T10** | SchemeF 全局 z-score | LOSO h_10 +13.40（vs SchemeC +21.86，**负面**） | — | **窗口归一化 T7 > 全局归一化 T10** |
| **T20/T21** | SchemeH 291 维（+MLOFI/WMP 扩展）| LOSO h_10 +21.70（vs +21.86，**持平**）| — | 单纯加维度无效 |
| **T22** | **Alpha101 + Alpha191**（WorldQuant 跨截面因子）| 最佳单因子 IC +21.80 | 未提交 | **需跨截面数据，违反硬约束 #2 batch shuffle** |
| **T25-T28** | aug_a 数据增强（bid/ask 互换 + 涨/跌 label 对换）| LOSO h_60 +6.30 → +9.67 (+3.37, 单 seed)；5-seed +11.46 | — | 进入 iter_005a/b |
| **T35** | ReVol + Savitzky-Golay (+32 维 = 255 维 SchemeK) | LOSO h_60 +5.53（vs +6.30，**不超**）| — | **SG 对树模型 gain 几乎为 0** |
| **T44** | **R34 Stage 1**：+54 维 sym-invariant（dual z-score W=20/50, signed RV, Kyle λ rolling, EWMA-OFI multi-α, cancel pressure）| LOSO h_60 单 seed +12.09；5-seed +13.67 | — | 突破 iter_005b |
| **T51/b/c** | **R34 Stage 2**：+59 维（window quantile rank W=20/50/100, signed skewness, GOFI ~30 维, vol-burst, Kyle λ）| LOSO h_60 5-seed **+15.06** | — | **真正突破**，超历史最好 |
| **T53** | **R34 Stage 3**：+14 维（EWMA-residual, multi-W RV ratio, Bipower variation, Cancel-pressure imbalance, Roll's effective spread）| LOSO +15.06 → **+16.84** (+1.78) | — | iter_008 |
| **T55** | **R34 Stage 4**：少量 edge gain 特征 | 与 Stage3 相近 | — | 合并入 SchemeP |
| **T68** | **R34 Stage 5**：+20 维（adaptive momentum, OFI toxicity, signed bipower, spread regime, trade-direction persistence, liquidity asymmetry）→ **SchemeP 359 维完成** | LOSO-eq +25.94 → +26.44 (+0.50) | — | iter_012 |
| **T74** | time features SchemeQ（+6 维：sin/cos position, is_open/close_30min, is_pm）| LOSO-eq → T78 −0.72 (回归框架下负面) | — | **SchemeQ 不入 final** |
| **T78** | regression + EV + SchemeQ time features | LOSO-eq +36.23 → +35.51 (**−0.72**) | — | **时间特征在回归框架下负面** |
| **T88** | range/vol features (LOSO 边际) | — | — | 部分入 Stage 5 |
| **T91** | ReVol + SG revival（回归框架重试）| holdout within-noise | — | 仍不超 |
| **T96** | HLC range factor | LOSO 边际 | — | 部分入 SchemeP |
| **T117** | LGB monotone constraints | LOSO 局部有效 | — | 进入 iter_017 候选，未入 final |
| **T124** | R2/R4 logret lag 特征 | 无增量 | — | KILL |
| **T141/b** | **Hawkes 过程订单流特征 5-seed**（R_Hawkes_OFI）| LOSO 边际 | — | 未入 final |
| **T142** | Hawkes 5-seed full-retrain | 边际 | — | 同上 |
| **T147** | feature pruning（按 importance 剪枝）| LOSO 轻微正 | — | 用于 v10/v11 候选 |
| **T151/b** | **TOD v9**（Time-of-Day 特征：解析 `time` 字段为盘口阶段）| LOSO 小幅正 | — | 进入 v9，但 T170 final 未用 |
| **T152** | **adversarial-filtered 259 维**（359 → 259，过滤泄露特征）| LOSO 持平 | 无确认提升 | iter_019 v10；**特征已足够鲁棒** |
| **T153/154** | v9b TOD clean + v10b/v11 打包 | — | — | 候选包 |
| **T184/T185** | 深度 CNN raw / hybrid CNN | in-sample 看似好 | 未提交 | OOD 未验证 |
| **T191** | **trend features pilot**（+5 维：mean_logret_W10/20/50/100, mid_pct_change_W100，至 375 维）| standalone +140.10 (vs T188v2 +150.43, **−10.33**); corr(T191, T188v2) = **0.9941**; IC_residual_pooled = **−0.1699（负！）** | 未提交 | **KILL**：trend signal 已被现有 359 维完全 capture |
| **R34 Stage 1-5** | sym-invariant 特征建设核心调研 | 154 → 359 维 SchemeP | — | **核心贡献** |
| **R_multitick_window** | 多 tick 窗口特征调研 | 部分入 Stage 5 | — | — |
| **R_roll_tsrv** | Roll effective spread + TSRV 调研 | 部分入 Stage 3 (T53) | — | 未成主力 |
| **R_Pairwise_ALL** | 特征 pairwise 交互 | 计算量大，边际 | — | 未入 final |

### 最终选择
- **SchemeP 359 维**（R34 Stage 1-5 累加）
- **不做 feature pruning**（T152 259 维 vs 359 维平台无差异，保留全集更稳）
- **不加 trend features**（T191 KILL）
- **不加 Alpha101/191**（T22，违反 batch shuffle 硬约束）
- **不加 Hawkes**（T141/142 边际）
- **不加 time-derived features**（T74/T78 在回归框架下负面；TOD T151 也未入 final）
- **window normalization**（T7）而不是 global normalization（T10）

### 11 个特征 DROP_NAMES（feature pruning）
**注**：实际 final 包 `pkg_T188v2_50plus50` 中的 fast_features.py 保留全 359 维特征，但 build_features.py 中标记了 ~11 个被认为冗余/泄露的特征（T147 feature pruning 结果）。具体名单需查 fast_features.py 中的 DROP_NAMES（如有），或参考 T152 adversarial filter 给出的 100 维差集。

### 选择理由（多源验证）
1. **R34 Stage 1-5 累计增益**：
   - Stage 1（T44）：+54 维 → LOSO 5-seed +13.67（vs iter_005b +9.67，+4.0）
   - Stage 2（T51）：+59 维 → LOSO 5-seed **+15.06**（+1.4 vs Stage 1）
   - Stage 3（T53）：+14 维 → LOSO +16.84（+1.78）
   - Stage 4（T55）：edge gain，合并进 SchemeP
   - Stage 5（T68）：+20 维 → LOSO-eq +26.44（+0.50；方法论已切到 LOSO-equiv，数字含 +7 方法论 inflation）
2. **sym-invariant 设计**：T7 窗口 z-score + T9 spread-norm，避免 per-sym normalization 触发硬约束 #3（OOD sym 时统计量失效）
3. **窗口归一化 > 全局归一化**：T7（+13.91）vs T10（+13.40），窗口 z-score 在每个 100-tick 切片内自适应，对 sym distribution shift 鲁棒
4. **时间特征负面**：T74/T78 SchemeQ 在回归框架下 LOSO-eq −0.72；T151 TOD 在 iter_019 v9 中正但 final 弃用——time-based 特征不稳定
5. **adversarial filter 持平**：T152 359 → 259 维 LOSO 持平，但 OOD 鲁棒性不增 → 不剪
6. **amount_delta log1p 处理**：T8 诊断 sym=2 z-score +7.49 偏离，后用 log1p 压缩（在 SchemeE T9 中实现）

### 未被采纳的特征（重要失败）

| 方案 | 实验 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| Alpha101/Alpha191 | T22 | 最佳单因子 IC +21.80 | 未提交 | **需跨截面数据**，违反硬约束 #2（batch shuffle）|
| Hawkes 过程 OFI | T141/142, R_Hawkes_OFI | 边际 | 未提交 | 增量不显著 |
| TOD (Time-of-Day) | T151 (v9) | LOSO 小正 | 未确认 | 在 T170 final 弃用 |
| Roll/TSRV | R_roll_tsrv, T53 | 部分入 Stage 3 | — | 未成主力 |
| CNN（DeepLOB style）| T184/T185 | in-sample 看似好 | 未提交 | OOD 未验证 |
| Transformer (GroupTr) | T172/T187 | in-sample +22~55 (污染) | 未提交 | 全 in-sample，OOD 风险 |
| SG filter (Savitzky-Golay) | T35/T91 | 不超 SchemeC | — | SG 对树模型 gain ≈ 0 |
| Trend features | T191 | −10.33, corr=0.9941, IC_residual=−0.17 | 未提交 | 完全冗余 + 负 IC |
| ScheC time features | T74/T78 | −0.72 | — | 回归框架下时间特征负面 |
| logret lag | R4/T124 | 无增量 | — | KILL |
| Pairwise 交互 | R_Pairwise_ALL | 计算量大边际 | — | 未入 |
| 概率校准 isotonic | T15/T131/T40 | LOSO 下降或持平 | — | 破坏排序 |
| sym embedding / per-sym norm | (隐式禁止) | — | — | 违反硬约束 #3 |
| date-derived features | (隐式禁止) | — | — | 违反硬约束 #1 |

---

## 主题 7：损失函数（含 regression vs classification）

### 背景
项目早期用 3-class CE（DeepLOB baseline），iter_013 突破换成 Δmid L2 regression，再 iter_015 用 SPO+ DFL 微调 NN。中间试过各种 robust loss（MAE/Huber/Quantile/GMADL）和 direct PnL loss。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T1** | 3-class CE NN (DeepLOB) | LOSO ~+8 | — | baseline |
| **T2** | 3-class CE LGB (argmax) | LOSO sum −22.10 | — | OOD 极差 |
| **T13** | direct PnL NN loss | 不收敛 | — | 端到端 PnL 梯度不稳 |
| **T15** | isotonic 校准 + EV gate | LOSO −1.02 | — | 校准破坏排序 |
| **T57** | PnL-aware loss（早期尝试）| 不稳 | — | 同 T13 |
| **T75** | **LGB L2 regression on Δmid + EV gate** | LOSO-eq **+36.23** (+9.79, **+37%**) | **+19.23** (iter_013) | **第一次大突破**，整个项目最大单次跃升 |
| **T78** | + 6 time features (SchemeQ) | LOSO-eq +35.51 (−0.72) | — | 时间特征在回归框架下负面 |
| **T81** | NN L2 regression on Δmid | LOSO-eq +35.89 (与 LGB 互补) | — | warm-start |
| **T86** | quantile + dual-gate (IQR-mid hybrid) | LOSO-eq +40.22 | — | 不超 T87 |
| **T87** | **SPO+ DFL NN fine-tune** | LOSO-eq NN+LGB **+40.09** (+4.20) | **+28.16** (iter_015) | **第二次大突破** |
| **T89** | CatBoost L2 5-seed | LOSO +40.13（与 T87 corr 0.77）| — | 4-way 失败 |
| **T92** | **magnitude sample weight**（按 Δmid 幅度加权样本）| LOSO 下降 | — | **R44 假设证伪**；幅度加权反损 PnL |
| **T99** | Huber LGB family | LOSO +44.72 (含 OOF contamination 警告) | iter_016 +25.08 (失败) | OOF 污染 |
| **T116** | GMADL direction-aware loss | LOSO 微弱 | — | 不超 SPO+ |
| **T132** | quantile LGB (α=0.5 ≈ MAE) | LOSO 边际 | — | 不超 |
| **T134** | direct PnL NN loss (no SPO+) | 不收敛 | — | 重复 T13 |
| **T137** | **MAE vs L2 LGB 系统对比** | MAE +0.3-0.5 边际 (但 OOF) | 未确认 | 最终 L2 仍主力 |
| **T143** | iter_019 v3 MAE LGB | LOSO +41.20 (−0.29 vs L2)；+conformal +43.21 | 未提交 | 平台未确认超 v2 |
| **T144** | Quantile/Huber/MAE diversity | 无增量 | — | MAE-only remains winner |
| **T145/b** | iter_019 v4 MAE LGB M7 | 平台预测 +30~32 | 未提交 | 不超 L2 M7 |
| **T146** | conformal-aware DE on v3_mae | LOSO +43.96 | 未提交 | 超调风险 |
| **R3** | HYD quartet (Huber/asym/Y-shape/date-decay) | Huber 略优 L2 | — | 见 T99 |
| **R10** | PnL-aware loss 10 方案调研 | SPO+ 最稳 | — | 路线选择 |

### 最终选择
- **LGB**：L2 regression (`objective='regression_l2'`)，y = Δmid / (mp_t + 1)
- **NN**：T81 L2 pretrain + T87 SPO+ DFL fine-tune（λ=30, lr=3e-5）
- **决策**：EV gate（pred > thr_up → 买；pred < −thr_dn → 卖）
- **不用**：3-class CE（T1/T2）、MAE（T137/T143）、Quantile（T132）、Huber（T99）、GMADL（T116）、direct PnL（T13/T134）、magnitude weight（T92）

### 选择理由（多源验证）
1. **iter_013 +9.79 → +37%**：T75 把 LGB 从 3-class CE 换成 L2 Δmid regression，LOSO-eq +26.44 → +36.23，**整个比赛最大单次跃升**
2. **关键洞察**：3-class CE 把 Δmid 离散化为 3 个 bucket（涨/平/跌），损失了幅度信息；回归保留幅度，EV gate 直接对应期望收益
3. **MAE 不超 L2**：T137 +0.3-0.5 边际且含 OOF 污染，T143 LOSO 反而 −0.29，T145 平台预测低于 L2 M7 → L2 始终主力
4. **direct PnL 不收敛**：T13/T134 重复确认，PnL 公式分段不连续，裸梯度不稳；SPO+ 用 surrogate gradient 把决策层损失对齐到 PnL，是稳定的端到端 DFL
5. **magnitude weight 反效果**：T92 R44 假设（按幅度加权 → 模型更关注大移动）证伪，原因可能是大幅度样本噪音也大，加权放大噪音

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| 3-class CE | T1, T2 | LOSO ~+8 / −22 | −6.65 (DeepLOB) | 压缩幅度信息 |
| Direct PnL | T13/T134/T57/T60 | 不收敛 | 未提交 | 梯度不稳 |
| MAE LGB | T137/T143/T145 | +0.3-0.5 / −0.29 | 平台预测 +30~32 | 不超 L2 M7 |
| Huber LGB | T99 | +44.72 (OOF 污染) | iter_016 失败 | OOF 警告应验 |
| GMADL | T116 | 微弱 | — | 不超 SPO+ |
| Quantile LGB | T132 | 边际 | — | 同上 |
| Magnitude weight | T92 (R44) | LOSO 下降 | — | 假设证伪 |
| isotonic/platt 校准 | T15/T131/T40/R13 | 持平或负 | — | 破坏排序 |

---

## 主题 8：模型架构选择

### 背景
NN 侧主力是 MLP [359→256→128→64→1]，LGB 侧是 LightGBM L2 regression。试过 DeepLOB CNN、GroupTransformer、GRU、CatBoost、XGBoost，均未能击败 MLP+LGB 组合。

### 演变时间线

| 实验 | 方法 | 本地 | 平台 | 备注 |
|---|---|---|---|---|
| **T1** | DeepLOB-style MLP, 3-class CE | LOSO ~+8 | (mmpc_demo −6.65) | baseline |
| **T17** | **CatBoost vs LightGBM** | LOSO ceiling +22.2（持平）| — | 无明显优势 |
| **T18** | **XGBoost vs LightGBM** | 持平 | — | 无明显优势 |
| **T19** | NN regularized (dropout + WD) | LOSO 持平 | — | 正则化无效 |
| **T81** | **MLP [359→256→128→64→1] LayerNorm GELU Dropout 0.10 L2** | LOSO-eq +35.89 | — | **架构固化** |
| **T87** | + SPO+ DFL fine-tune | LOSO-eq +40.09 | +28.16 | NN 主力 |
| **T89** | CatBoost L2 5-seed | LOSO +40.13 (与 T87 corr 0.77) | iter_016 失败 | 4-way 失败后淘汰 |
| **T93** | NN arch Transformer (preliminary) | — | — | 探索性 |
| **T94** | NN arch ResNet-DCN | — | — | 探索性 |
| **T95** | **GRU regression** | LOSO 微小提升 | iter_016 失败主因 | hidden state 容易过短 h 过拟合 |
| **T97** | 第二组 NN (T87 风格) | LOSO 与 T87 相近 | T163 失败 | 架构同源 diversity≈0 |
| **T117** | LGB monotone constraints | LOSO 局部有效 | — | 未入 final |
| **T123** | iter_017 LGB HYD+monotone+date-decay | LOSO +40.70 (+0.61) | 未确认 | 边际 |
| **T133** | alt booster 对比 | 与 LGB 持平 | — | 不替换 |
| **T157** | NN Transformer preliminary | — | — | 进入 T172 |
| **T166** | CatBoost L2 5-seed M7 retrain | holdout +0.39 | — | 边际正 |
| **T172/b** | **GroupTransformer** (d_model=64, nhead=4, 2 layers, n_tokens=32, CLS) | in-sample +149.91; best ensemble +164.71（**全 in-sample 污染**）| 未提交 | OOD 未验证 |
| **T184/T185** | 深度 CNN raw / hybrid CNN | in-sample 看似好 | 未提交 | OOD 未验证 |
| **T186** | T184/T172b retrain + LR sweep | architecture_doesnt_fit | — | 深度架构在当前特征框架下无法竞争 MLP |
| **T187** | GroupTransformer replace/add | in-sample +22.2/+12.6/+14.7 (全污染) | 未提交 | 新架构 M7 retrain 高风险 |
| **T183** | NN random init M7 (无 T81 warm-start) | holdout +0.63 (持平 T170) | — | 不同初始化 ≈ 不同收敛点 |
| **R31** | 2024-2026 HFT/LOB SOTA 文献调研（TransLOB/TabLOB/N-BEATS/9 篇）| MLP+GBDT 集成 competitive | — | 文献指导，但确认 MLP+GBDT 路线 |
| **R35** | alt architectures 调研 | — | — | 探索性 |

### 最终选择
- **NN 架构**：MLP `[359 → 256 → 128 → 64 → 1]`，LayerNorm + GELU + Dropout(0.10)，输出层 linear（regression）
- **LGB 架构**：LightGBM L2 regression（`objective='regression_l2'`），num_leaves/max_depth/lr 默认调出来的具体值见 final 包 config
- **不用**：CatBoost / XGBoost（持平 LGB 但无 ensemble 增量）、GRU（hidden state 风险）、Transformer / GroupTr（OOD 风险）、深度 CNN（OOD 风险）

### 选择理由（多源验证）

#### 为啥 MLP 4 层 [359→256→128→64→1]
- T81 设计：输入 359 维 SchemeP，输出 1 维 Δmid 回归
- 渐缩 1.5×~2× 的经典 MLP 设计：256 → 128 → 64
- 总参数量 ~120k，relatively small，避免过拟合 1.47M 样本
- T93/T94 试过 Transformer/ResNet-DCN，未超 MLP
- T172/T184/T185/T187 试过更深 Transformer/CNN，全部 in-sample 污染严重，OOD 风险高

#### 为啥 LayerNorm + GELU + Dropout 0.10
- LayerNorm：对 359 维输入做 per-sample 归一化（与 SchemeP 的窗口 z-score 互补）
- GELU：现代 NLP 标准（vs ReLU 死神经元、Swish 略慢）
- Dropout 0.10：T19 试过更激进的 regularization 无增量；0.10 是 conservative 选择

#### 为啥 LGB 而不是 CatBoost / XGBoost
- T17/T18 早期对比：三者 LOSO ceiling 持平（+22.2 左右）
- T89 CatBoost L2 5-seed +40.13 与 T87 corr 0.77，高度共线，集成无增量
- T166 CatBoost M7 retrain holdout +0.39 边际，被 50+50 ensemble 替代
- LGB 的 leaf-wise growth + 高效 GPU 训练（项目实测 LGB GPU vs CPU 38s → 12s, 3.2×）

#### 为啥不用 GRU/Transformer
- T95 GRU hidden state 易在短 horizon 过拟合，是 iter_016 v3 失败主因
- T172/T187 GroupTransformer in-sample 数字看着好（+22~55），全部含 96-119 训练污染，OOD 风险
- T186 verdict "architecture_doesnt_fit"：深度架构在 SchemeP 359 维特征框架下无法竞争 MLP

### 被拒绝的替代方案

| 方案 | T-编号 | 本地 | 平台 | 为啥拒绝 |
|---|---|---|---|---|
| CatBoost L2 | T17 / T89 / T166 | 持平 / corr 0.77 / +0.39 | iter_016 失败 | 与 LGB 高共线 |
| XGBoost | T18 | 持平 | — | 同上 |
| GRU | T95 | 微小提升 | iter_016 失败主因 | hidden state 过短 h 过拟合 |
| 第二组 MLP NN (T97) | T156/T163 | +1.41/+1.66 holdout | T163 −3.70 | 同 T87 架构 diversity≈0 |
| Transformer (GroupTr) | T172 / T187 | in-sample +22~55 (污染) | 未提交 | OOD 风险 |
| Deep CNN | T184/T185 | in-sample 看似好 | 未提交 | OOD 未验证 |
| ResNet-DCN / Transformer pilot | T93 / T94 | — | — | 探索未上 |
| LGB monotone constraints | T117 / T123 | 局部 +0.61 | 未确认 | 边际 |
| LGB HYD+monotone+date-decay | T123 (iter_017) | +0.61 | 未确认 | 边际 |

---

## 主题 9：评估方法论变迁

### 背景
本地 eval 在项目周期内经历 **4 次大变化**，每次变化都影响"本地 → 平台"的对应关系。这是整个项目最容易混淆数字的部分。

### 演变时间线

| 阶段 | 时间 | eval 方式 | 范围 | 特点 / 平台对应 |
|---|---|---|---|---|
| **Phase 0** | T1-T2 | IID split (同 sym 内) | 单 sym | 严重乐观；IID +19.25 → LOSO −22.10 |
| **Phase 0-1** | T1-T35 | **LOSO 5-fold**（留一 sym）| 全 5 sym, per-sym 留一 | 过度惩罚 sym OOD；短 h gap −30.5 |
| **Phase 1** | T23 | **LOSO → 平台校准 (2-anchor)** | iter_002/iter_000 锚点 | 建立透传系数 0.70 |
| **Phase 2** | T44-T74 | **LOSO-equiv** (全 sym 训练 + DE 在 442k 行) | 全 5 sym, day 0-119 训练 | 方法论变化贡献约 +7 PnL（去除 LOSO 过度惩罚）|
| **Phase 3** | T58/T64 | + **V4 walk-forward val** (train 0-79, val 80-95, test 96-119 OOF) | 全 5 sym + 时间 OOF | 当前最干净方法，LOSO-eq ~40 ≈ 平台 28-35 |
| **Phase 4** | T156+ | **holdout in-sample** (date 0-119 全数据训练 → 96-119 评分) | 全数据，含测试日期 | **in-sample 污染**；绝对值不可信，只看相对 delta |

### 关键 eval 实验

| 实验 | 方法 | 关键发现 |
|---|---|---|
| **T4** | LOSO 5-fold + 阈值 gate 建立 | LOSO 作为平台代理 |
| **T23** | **Robust CV + 2-anchor 平台校准** | h_60 gap ≈ −2.23；透传系数 ~0.70；后续所有"预测平台 X"均基于此 |
| **T34** | CV v2 (LO2SO + bootstrap CI + 平台预言机) | 多维验证框架 |
| **T58** | **walk-forward CV 验证** (V4 比 LOSO 更接近 OOD) | V4 标准化 |
| **T59** | 全 sym 训练 (取消 LOSO 留一 sym) | LOSO-eq +16.84 → +24.52 (+7.68; 其中约 +7 方法论 inflation) |
| **T64** | V4 walk-forward val 正式化 (iter_011) | LOSO-eq +25.94 → +26.44 (+0.50) |
| **T66** | adversarial val | 后用于 T152 特征过滤 |
| **T80** | LOSO → 平台校准 mapping 透传 0.70 | 后续 53-71% 区间方差较大 |
| **T82-85** | 特征泄露审计、date=0 验证 | 无泄露，V4 是 cleaner OOD 代理 |
| **T115** | scale invariance audit | sym-agnostic 设计正确 |
| **T149** | NN T87 OOD sym 鲁棒性扫描 | OOD sym 下表现可接受 |
| **R11** | cross-stock OOD robustness 文献调研 | sym-agnostic 路线合理 |
| **R30** | 数据特征深度调研 | h_60 有效，h_5/10/20 失效原因（手续费 vs α）|

### 每次方法论变化对本地→平台 gap 的影响

#### Phase 1（LOSO 5-fold）→ Phase 2（LOSO-equiv）
- T59 全 sym 训练：LOSO 5-fold +16.84 → LOSO-equiv +24.52，**+7.68**
- 其中约 **+7 是方法论变化**（去除 LOSO 过度惩罚 sym OOD），真实新增 alpha ~+1
- **教训**：跨阶段数字不可直接比

#### Phase 2 → Phase 3（+V4 walk-forward）
- T64 V4：LOSO-eq +25.94 → +26.44，**+0.50**（方法论更干净，数字本身相近）

#### Phase 3 → Phase 4（holdout in-sample）
- T170 起：从 V4 OOF 切换到 date 0-119 全数据训练 + 96-119 评分
- **本地数字爆涨 ~3-4×**：T170 holdout +149.95 vs V4 OOF ~+41
- **绝对值不可信**：T172 in-sample +149.91，平台未提交（OOD 未知）
- **只看相对 delta**：T188v2 vs T170 holdout delta +0.48 → 平台 delta +1.00

### LOSO-equiv → 平台 透传系数表（Phase 2-3）

| iter | LOSO-eq | 平台 | 透传 % |
|---|---|---|---|
| iter_013 | +36.23 | +19.23 | **53%** |
| iter_015 | +40.09 | +28.16 | **71%** |
| iter_018 | +41.49 | +28.93 | **70%** |

→ 透传系数在 53-71% 区间有较大方差，DE 阈值越激进透传越低

### holdout in-sample → 平台（Phase 4）

| iter | holdout (in-sample) | 平台 |
|---|---|---|
| T170 | +149.95 | +34.64 |
| T188v2 | +150.43 | +35.64 |
| T190 | +140.62 | +34.59 |
| T192 H4 | +193.94 | +31.0 (overfit) |

→ 绝对值无意义；delta +0.48 → +1.00；但 +43 delta → −4.64（T192）警告 in-sample 污染陷阱

### in-sample 污染陷阱（4 个经典案例）

| 实验 | 本地 holdout | 平台 | 教训 |
|---|---|---|---|
| **T180 OOD GMM abstain** | −32 ~ −78 | 未提交（如提交预计也负）| 测试集比训练 in-distribution，弃权损失高 PnL 点 |
| **T182 无 conformal** | +6.62（看似 conformal hurt）| **−0.30**（conformal 实际有用）| in-sample artifact：训练日期上 abstain 损 PnL，OOD 上 abstain 过滤错误预测 |
| **T172 GroupTransformer** | in-sample +149.91 | 未提交 | 全 in-sample 污染，OOD 风险 |
| **T192 bounded NNLS** | val +180.41, test +193.94, ratio 1.075 | **+31.0** (−4.64) | val+test 均 in-sample，OLS 学到 in-sample 共同 noise pattern |

### 选择理由（多源验证）
1. **LOSO → LOSO-equiv（Phase 1→2）**：LOSO 过度惩罚 sym OOD（每个 fold 只 4 sym 训练）；LOSO-equiv 全 sym 训练更贴近平台
2. **+V4 walk-forward（Phase 2→3）**：时间 OOD 是平台真实场景（评测在训练截止后的新数据）
3. **holdout in-sample（Phase 4）**：M7 trick 之后没有干净的 V4 OOF（因为所有数据都进训练了），只能 in-sample 评分；这是已知的污染但**只用 delta**

### 被拒绝的方法论

| 方法 | 实验 | 死因 |
|---|---|---|
| IID split 同 sym 内 | Phase 0 | 严重乐观 +19.25 → LOSO −22.10 |
| LOSO 5-fold (Phase 1) | T1-T35 | 过度惩罚 OOD，被 LOSO-equiv 替代 |
| Pseudo labeling | T65 | 负面 |
| 3-class CE eval (LOSO h_10) | iter_003 | 短 h 平台始终负 |

---

## 主题 10：推理优化 + 5-horizon share_with=60

### 背景
平台限制 3 小时完成 442k 行推理。早期 iter_007/008 因 254 min feature 计算超时。T61 batch-vec 把推理从 254 min → 4.5 min（58×）首次通过。T188v3 batched torch CUDA bmm 把 50 NN 推理 1509ms → 8.1ms（186×）。T188v3_fullhorizon 让 5 个 horizon 共享一套 h=60 ensemble（1× cost 而非 5×）。

### 演变时间线

| 实验 | 方法 | before → after | 加速比 | 影响 |
|---|---|---|---|---|
| **T61 (iter_010)** | **batch-vec 特征提取**（向量化，替代 Python for 循环）| 254 min → 4.5 min（442k 行）| **58×** | **首次通过平台 3h 限制**（iter_007/008/009 全超时）|
| **T188v3** | **50 NN sequential numpy → batched torch CUDA bmm**（50 NN 权重堆叠 (50, out, in) tensor，单次 `torch.bmm` 替代 50 次 numpy 循环）| NN-only 1509ms → **8.1ms**（per 1024 batch，stated baseline）| **186×** | 50 NN 推理不再是瓶颈 |
| T188v3 实测 | NN-only 实测 554.8ms → 8.1ms | — | **68.5×** | — |
| T188v3 e2e | feature(549ms) + LGB(84ms) + NN(1509ms) = 2142ms → 641ms | — | **3.3×** | 442k 行总时间 15.4 min → 4.6 min |
| **T188v3_fullhorizon** | **5 个 horizon 共享一套 h=60 ensemble 预测**（h_5/10/20/40 也用 T188v2 h_60 ensemble 输出，按各 horizon 阈值决策）| 5× cost → 1× cost | **5×** | 5 个 horizon 提交只需 1× inference |
| **T189/T189_GPU_pkg** | T170（5+5）转 torch GPU 推理包 | CPU 855ms → GPU 更快 | — | 平台 GPU 环境可加速 |

### 最终选择
- **T188v3 batched torch CUDA bmm**：50 NN 权重堆叠成单一 tensor，一次 `torch.bmm` 完成所有 50 NN forward
- **CPU 后备版本**：`pkg_T188v2_50plus50` 同时打包 CPU-only 备用（MD5=c67e4d25665a2a7a12f4f6839cc8d0fe）和 GPU optimized 版本（MD5=95993e6250ea56361dfecb62c9123e63）
- **5 horizon share h=60**：所有 5 个 horizon 用 h=60 ensemble 的预测，按各自 thr 缩放（**sqrt(H/60) scaling**）
- **行动一致性验证**：T188v3 actions vs T188v2 max diff 6.98e-10，远低于 1e-4 容忍度

### 选择理由（多源验证）

#### T61 batch-vec 58× 加速
- 早期 iter_007/008/009 因 254 min feature 计算全部超时（平台 3h 限制）
- T61 用 NumPy/Pandas 向量化替代 Python for 循环（CLAUDE.md 强制规则）
- iter_010 首次通过 3h 限制，开启所有后续提交可能性

#### T188v3 186× 加速
- T188v2 50 NN 在 CPU 上 sequential numpy 推理太慢（baseline 1509ms / 1024 batch）
- 把 50 NN 权重堆叠：W_all shape (50, out, in)，单次 `torch.bmm(x_repeat, W_all)`
- CUDA 上一次性算完 50 NN，几乎是 LGB 推理时间的级别
- actions 一致性 100%（max diff 6.98e-10）

#### sqrt(H/60) scaling 数学依据
- Δmid 的标准差 ∝ sqrt(H)（Brownian motion 假设下，方差线性增长 → 标准差 sqrt 增长）
- 所以 h=5 的预测分布尺度 ≈ sqrt(5/60) ≈ 0.29 × h=60 的尺度
- 阈值按 sqrt(H/60) 缩放：thr_up(H) = thr_up(60) × sqrt(H/60)
- 这样所有 horizon 的"显著性"标准统一

#### 5 horizon share h=60（"5 个 horizon 免费拿到"）
- 整个项目专注 h_60（α=0.1%，手续费 0.02% 还有 alpha 空间）
- 但平台要求 5 个 head 都提交（取最高）
- 用 h=60 ensemble 输出按 sqrt 缩放给所有 horizon 用，1× cost 完成 5 个 head
- 短 horizon 平台历史上均负（见**发现 5**），意义有限，但形式上不缺 head

#### 为啥不为短 horizon 训独立模型
- iter_002 关键发现：h_10 LOSO **+21.86**（最强！）→ 平台 **−8.64**（gap −30.5）
- iter_003/004 试过短 h 严格阈值/关闭，未获平台正分
- 整个项目证据：α=0.05%（h_5/10）被手续费 0.02% 双边摧毁
- α=0.1%（h_60）才有 alpha 空间

### 平台对应表（短 horizon 灾难）

| Horizon | LOSO 本地 (iter_002) | 平台 |
|---|---|---|
| h_5  | +17.49 | −7.68 |
| h_10 | **+21.86** | **−8.64** |
| h_20 | +19.70 | −5.09 |
| h_40 | +11.71 | +2.02 |
| h_60 | +6.30 | **+4.07** |

→ LOSO 短 h 看似最强，平台短 h 全负 → **整个项目专注 h_60，短 horizon 彻底放弃**

### 被拒绝的推理/wrapper 方案（重复触碰硬约束 #2）

| 方案 | T-编号 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| **TTA K=5**（5 次增强预测平均）| T161 | holdout T170 +149.95 → 下降 | 未提交 | 平台批次乱序，per-batch 统计自适应不稳 |
| **In-row TTT** W3a/W3b/W3c | T162 | W3a −0.30, W3b −1.98, W3c −3.11 | 未提交 | 违反硬约束 #2 精神 |
| **Adaptive threshold V1-V4** | T168 | 全负 | 未提交 | per-batch 统计不稳 |
| **AR TTT** | T169 | 退步 | 未提交 | 自回归 violates batch shuffle |
| **OOD GMM abstain** (PCA 20d + GMM k=5) | T180 | V1 −32, V2 −57, V3 −78 | 未提交 | 测试集比训练 in-distribution; 低密度反高 PnL |
| **Time-window hard abstain** | T181 | V1 −23.39, V2 −0.27, V3 −2.38 | 未提交 | T170 conformal 已过滤；硬弃权适得其反 |
| **In-batch wrapper B/C** | T177 | holdout 正 | 未提交 | 同 TTA 问题 |

### 未被采纳的论文方向（user 明确判定为"垃圾"）
- **TimesNet** / **iTransformer** / **PatchTST** / **N-BEATS**：通用时序模型，在 LOB 这种 high-noise low-signal 场景下未实证有效
- R31 文献调研笔记：TransLOB / TabLOB / 9 篇深度笔记 — 指导 T172 GroupTransformer 设计，但最终未入 final 包（OOD 风险）

---

## 附录 A：关键发现 8 条速查

### 发现 1：本地 holdout 的 in-sample 污染问题
- T170/T87/T81 NN 训练在 date 0-119（含测试集 96-119），holdout 评分 in-sample
- T182 ablation 案例：本地 +6.62 → 平台 −0.30
- T192 OLS 案例：本地 +193.94 → 平台 +31.0 (−4.64)
- **结论**：本地 holdout 绝对值不可信；只看相对 delta；平台 OOD 是唯一 ground truth

### 发现 2：Ensemble 扩容收益递减与非单调
| 扩容步骤 | 本地 holdout delta | 平台 delta |
|---|---|---|
| 5+5 → 40+5 (T179, 只扩 NN) | +0.63 | **−0.25** |
| 5+5 → 50+50 (T188v2, NN+LGB 都扩) | +0.48 | **+1.00** |
| 50+50 → 150+150 (T190) | −9.81 | **−1.05** |

→ 单纯扩 NN 无平台收益；5→50 +1.00（有效）；50→150 −1.05（反效果）

### 发现 3：M7 全数据重训是"已验证模型"专属
| 模型 | M7 本地 holdout | M7 平台 |
|---|---|---|
| T75 LGB（已验证）| LOSO 不变 | **+5.51** |
| T87 NN（已验证）| in-sample +8.49 (污染) | **+0.20** |
| T183 NN random init | in-sample +0.63 | 未提交 |
| T172 Transformer（新架构）| in-sample +55 (严重污染) | 未提交 |

### 发现 4：OOF threshold DE 过拟合风险
| 实验 | 本地 OOF 提升 | 平台 |
|---|---|---|
| iter_013 DE 4D | +36.23 vs +33.72 (+2.51) | 透传 53% |
| iter_016 4-way DE | +43.71 vs +40.09 (+3.62) | **−3.08** |
| T192 bounded NNLS | val/test ratio 1.075 | **−4.64** |

### 发现 5：短 horizon 平台完全不可信（见主题 10 表）

### 发现 6：SPO+ DFL 是最优 NN 训练协议
- 3-class CE：LOSO ~+8 → −6.65
- L2 only (T81)：+35.89 → 未单提
- **SPO+ DFL (T87)：+40.09 → +28.16**
- Direct PnL (T13/T134)：不收敛

### 发现 7：per-sym beta conformal 在平台上 +0.30~+0.77
- iter_018 vs iter_015：+0.77
- T170 vs T182：+0.30
- 本地 holdout 显示 conformal "hurt" 是 in-sample artifact

### 发现 8：corr(NN, NN) 高共线性阻止 OLS 学权重
- T192 NN-NN avg corr **0.9331**
- LGB-LGB avg corr **0.8433**
- condition number **54,182**（极病态）
- 解释了任何 per-model 权重学习均失败

---

## 附录 B：硬约束 3 条速查（CRITICAL_CONSTRAINTS.md）

1. **`date` 字段评测时被置 0** → 不能用作 feature，不能用作 embedding，不能 derive 时段
2. **测试点顺序被打乱** → Predictor 不能维护跨 batch state（LSTM hidden / 缓存 buffer / per-batch 统计）
3. **`sym` 0-4 但可能含训练外股票** → 模型必须 sym-agnostic（不能 sym embedding / per-sym normalization / per-sym 模型）

**违反这 3 条 = 提交报错或 0 分**。整个项目设计都遵守，多次失败实验（T161/162/168/169 TTT/TTA 系列）都是触碰约束 #2 的代价；R_cross_sym ABORT 是触碰约束 #2 + #3。

---

## 附录 C：22 次平台提交完整表（含 zip 文件名）

| # | 日期 | zip / iter | 主要 trick | 本地 | 平台 | Δ |
|---|---|---|---|---|---|---|
| 1 | 2026-05-01 | `submission_050601_iter0.zip` | mmpc_demo DeepLOB | ≈0 | **−6.65** | — |
| 2 | 2026-05-06 | `submission_050606_iter002.zip` | LGB 多 h + 阈值 gate (SchemeC) | LOSO h_60=+6.30 | **+4.07** | +10.72 |
| 11 | ~2026-07-07 | `submission_050707_iter013.zip` | Δmid regression + EV gate (T75) | LOSO-eq +36.23 | **+19.23** | +15.16 |
| 13 | 2026-08-02 | `submission_050802_iter015.zip` | T87 SPO+ NN + T75 LGB | LOSO-eq +40.09 | **+28.16** | +8.93 |
| 14 | 2026-08-15 | `submission_050815_iter016_v3.zip` | 4-way 失败 (CB+Huber+GRU) | LOSO +43.71 | **+25.08** | −3.08 |
| 15 | 2026-08-17 | `submission_050817_iter018_v1.zip` | + per-sym conformal | LOSO +41.49 | **+28.93** | +0.77 |
| 16 | 2026-08-18 | `submission_050818_iter019_v2_fullretrain_conformal.zip` | M7 LGB 全数据重训 (T140) | LOSO 不变 | **+34.44** | +5.51 |
| 17 | 2026-09-09 | `submission_050909_iter019_v2NN_T97_agreement_combined.zip` | T163 v2+T97+agree 失败 | holdout +42.90 | **~+30.74** | −3.70 |
| 18 | 2026-09-09 | `submission_050909_iter019_v2N_T87M7.zip` | T170 NN M7 SOTA | holdout +8.49 | **+34.64** | +0.20 |
| 19 | 2026-09-09 | `submission_050909_iter019_v2N_T87M7_NoConformal.zip` | T182 无 conformal | holdout +6.62 | **+34.34** | −0.30 |
| 20 | 2026-09-10 | `submission_050910_iter019_v2N_40NN_bag.zip` | T179 40-NN | holdout +0.63 | **+34.39** | −0.25 |
| 21 | **2026-09-11** | `submission_050911_iter019_v2N_50plus50_optimized.zip` | **T188v2 50+50 SOTA** (MD5=95993e6250ea56361dfecb62c9123e63, 148MB) | holdout +0.48 | **+35.64** | **+1.00** |
| — | 2026-09-11 | `submission_050911_iter019_v2N_50plus50_cputorch.zip` | CPU-only 备份 (MD5=c67e4d25665a2a7a12f4f6839cc8d0fe) | — | **+35.64** | — |
| 22 | 2026-05-12 | `submission_050912_iter019_v2N_150plus150_H4weights_cputorch.zip` | T192 NNLS overfit | holdout +193.94 | **+31.0** | −4.64 |

---

## 附录 D：实验编号速查（T1-T192 + R 系列）

详细分组见 **0512_final_report.md §3** 和 **0512_tricks_table.md**，本附录只列各 phase 关键 T 编号速查：

- **Phase 0 (T1-T10)**：T1 NN baseline / T2 LGB baseline / T3 SchemeC 226 维 / T5b iter_002 / T7 窗口 z-score / T8 sym=2 诊断 / T10 全局 z-score 失败
- **Phase 1 (T11-T35)**：T13 PnL loss 不收敛 / T15 isotonic 校准 / T17 CatBoost / T22 Alpha101 / T23 平台校准 / T25-28 aug_a / T30 DE 4D 阈值 / T35 SG filter 失败
- **Phase 2 (T44-T74)**：T44 R34 Stage1 / T51 Stage2 (+15.06 LOSO) / T53 Stage3 / T55 Stage4 / T58 V4 walk-forward / T59 全 sym 训练 / T61 batch-vec 58× / T64 V4 正式化 / T68 Stage5 (SchemeP 359) / T74 SchemeQ
- **Phase 3 (T75-T92)**：**T75 Δmid 回归 +19.23（1st 突破）** / T80 透传校准 / T81 NN L2 / **T87 SPO+ DFL +28.16（2nd 突破）** / T89 CatBoost / T92 magnitude weight 证伪
- **Phase 4 (T95-T109)**：T95 GRU / T97 第二组 NN / T99 Huber LGB / T106-109 iter_015/016 打包系列
- **Phase 5 (T115-T155)**：T127 iter_018 conformal +28.93 / T128 M7 起点 / T136 SPO+ 超参 / T137 MAE vs L2 / **T140 M7 LGB +34.44（3rd 突破）** / T143 v3 MAE / T146 conformal DE / T147 feature pruning / T150 conformal β sweep / T152 adversarial 259 维
- **Phase 6 (T156-T169)**：T156 NN swap / T158 agreement / T161 TTA 失败 / T162 TTT 失败 / **T163 v2+T97 失败 −3.70** / T165 magnitude filter / T166 CB M7 / T168 adaptive thresh 失败 / T169 AR TTT 失败
- **Phase 7 (T170-T192)**：**T170 NN M7 +34.64** / T172 GroupTransformer / T176 T87M7 HP sweep / T178 LGB 10-seed null / T179 40-NN +34.39 / T180 GMM abstain 失败 / T181 time-window abstain 失败 / **T182 无 conformal ablation** / T183 NN random init / T184-T187 deep CNN / Transformer 未提交 / **T188v2 50+50 +35.64（current SOTA）** / T188v3 186× 推理加速 / T189 GPU pkg / **T190 150+150 −1.05** / T191 trend features KILL / **T192 NNLS overfit −4.64**

**R 系列**（40+ 调研）：R1/v2 date decay / R2 Group DRO 负 / R3 HYD quartet / R4 logret lag 无 / R5/v2 Bayes optim / R10 PnL-aware (SPO+ 最稳) / R11 cross-stock OOD / R12 ensemble stacking / R13 calibration / R30 数据特征 / R31 HFT 2024-2026 / **R34 Stage 1-5 特征建设** / R_50seed_bag / R_Hawkes_OFI / R_NaN1/2 / R_Pairwise_ALL / **R_conformal_select (β 核心)** / R_cross_sym ABORT / **R_full_retrain (M7 起源)** / R_multinn_spo (HP > seed diversity) / R_multitick_window / R_roll_tsrv / R_stack3_compound 负 / R_time_of_day / R_T75L2_*

---

## 关于使用本文档（给 Worker C / 审稿人）

每一行"具体决策"在 final 包代码（`final_submission_code/`、`pkg_T188v2_50plus50/`）中应该都能在这份证据库里找到对应的：
1. **T-编号**（可去 `experiments/T*/results.json` 查具体 metric 和 log）
2. **本地数字**（含 LOSO / LOSO-equiv / holdout 标注，注意阶段差异）
3. **平台数字**（22 次提交锚点，绝对 ground truth）
4. **被拒绝的替代方案**（被试过但更差的——这是 worker C 关键 join 信号）

如果 worker C 在 `CODE_QUESTION_LIST.md` 中提问 "为啥 thr_up=0.0003？"，在本文档主题 1 中能找到完整证据链（T5a/T30/T62/T63/T75/T76/T126/T146/T168 + iter_013 透传 53% + iter_016 OOF DE 灾难）。

如果是 "为啥 50+50 不是 100+100 或 150+150？"，在主题 2 中找到 T179/T188v2/T190 数据 + collinearity 分析。

如果是 "为啥 NN 11 epoch / LGB 330 iters？"，在主题 4 中找到 V4 walk-forward best_epoch × 1.1 的推导 + T176 HP sweep 确认。

每个证据条目带 T 编号、本地数字、平台数字 — 直接可 join。
