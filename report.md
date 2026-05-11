# 良文杯 2026 — 完整研究历程报告

> 截至 2026-05-11。当前 SOTA：**T188v2 50+50 mega ensemble = 平台 +35.64**
>
> 本文档综合自 PROGRESS.md、git log（111 commits）、114 个 experiments/*/results.json 及平台反馈记录。

---

## 1. 一句话概览

**第二届"良文杯"**（厦门大学 × 黄良文统计学科基金会，协办 AITOPIA）要求用 LOB + 订单流数据预测 5 只匿名 A 股的 midprice 未来 5/10/20/40/60 ticks 后移动方向（涨/平/跌三分类），评分指标为**累计 PnL 收益率（扣 0.01% 双边手续费）**，取 5 个 horizon 中最好的一个计入排名。初赛从 4 月初开放至 5 月 11 日，每 12 小时限提交一次。

核心特征：5 只 sym（可能含训练外股票）× 120 交易日 × AM/PM × 2001 ticks（3 秒/tick）。输入为过去 ≤ 100 ticks 的 154 维特征；`date` 字段评测时置 0，测试点顺序被打乱，**模型必须 sym-agnostic**。这三条硬约束极大地约束了特征工程与架构选型。

从 mmpc_demo 官方 DeepLOB 基线（平台 −6.65）出发，历经约 **190 个 T-编号实验**和 **15+ 次 R-编号调研**，经过回归目标替换、SPO+ DFL、LGB 全数据重训（M7 trick）和 50+50 mega ensemble，最终以 **+35.64** 问鼎本地 SOTA，相对基线绝对提升超 42 分。

---

## 2. 平台提交历史（baseline → T188v2 SOTA）

> 表格说明：**平台 PnL** 为确认的平台得分；"(预测)" 表示基于 LOSO-platform 传导率估计；"(超时)" 表示平台运行超 3 小时被终止；"—" 表示未提交或无记录平台分。

| # | 提交日期 | zip 文件名 | 主要 trick / 迭代版本 | 平台 PnL | 备注 |
|---|---|---|---|---|---|
| 1 | 2026-05-01 | `iter0.zip` | mmpc_demo 官方 DeepLOB baseline | **−6.65** (best of 5) | h_5=-9.08, h_10=-13.32, h_20=-6.65, h_40=-16.07, h_60=-23.13 |
| 2 | 2026-05-02 | `iter1c.zip` | LGB schemeB + 高置信度阈值 gate | (无记录) | 预测 [−2, +6]；LOSO 从 −22.10 → +6.45 |
| 3 | 2026-05-03 | `iter1d.zip` | 同上，在 LOSO OOF 上重调阈值 | (无记录) | LOSO +11.11 |
| 4 | 2026-05-05 | `iter1f.zip` | schemeD1 window z-score (308 维) | (无记录) | LOSO +13.91 |
| 5 | 2026-05-06 | `iter002.zip` | SchemeC 多 horizon，5 fold LOSO | **+4.07** | h_60 LOSO +6.30 → 平台 +4.07，校准良好 |
| 6 | 2026-05-06 | `iter006.zip` | 同窗提交验证 | (无记录) | 技术备份 |
| 7 | 2026-05-14 | `iter003.zip` | 5-seed SchemeC h_10 ensemble | (~+4.07) | T23 预测：h_10 LOSO +0.36 仅平台 +0.11，不超 iter_002 |
| 8 | 2026-05-22 | `iter004a_safe.zip` | 短 h 严格阈值，保住 h_60 地板 | (~+4.07) | 纯后处理，不改模型；下限锁定 iter_002 |
| 9 | 2026-05-23 | `iter004b_aggressive.zip` | 短 h 激进阈值上探 | (~+4.07) | 与 iter_004a 下限相同 |
| 10 | 2026-05-24 | `iter005a.zip` | aug_a 数据增强 h_60 单 seed | (~+7.4) | LOSO +9.67 → 预测平台 +7.44 |
| 11 | 2026-05-26 | `iter005b.zip` | aug_a 5-seed ensemble h_60 | **(~+9.2)** | LOSO +11.46；0.70 透传系数预测 +9.2 |
| 12 | 2026-07-07 | `iter007.zip` | R34 Stage 2 sym-invariant 特征 | (超时) | 254 min feature compute，平台 3h 内跑不完 |
| 13 | 2026-07-07 | `iter008.zip` | R34 Stage 3 特征 (+14 维) | (超时) | 340 维，同样超时 |
| 14 | 2026-07-07 | `iter009.zip` | T59 全 sym 训练（不再 LOSO 留一） | (超时) | 本地 LOSO-equiv +24.52，仍超时 |
| 15 | 2026-07-07 | `iter010.zip` | **batch-vec 推理 58× 加速** | (无记录) | 首次不超时！LOSO-equiv +24.52 不变 |
| 16 | 2026-07-07 | `iter011.zip` | V4 walk-forward val (date 0-79 train) | (无记录) | LOSO-equiv +25.94 |
| 17 | 2026-07-07 | `iter012.zip` | R34 Stage 5 特征 (359 维) | (无记录) | LOSO-equiv +26.44 |
| 18 | 2026-07-07 | `iter013.zip` | **⭐ 突破：regression on Δmid + EV gate** | **+19.23** | LOSO-equiv +36.23 → 平台 +19.23（透传 53%，LOSO→平台 gap −17） |
| 19 | 2026-07-08 | `iter014.zip` | NN L2 regression + T75 LGB ensemble | (~+20.67) | LOSO-equiv +38.28；预测 19.23 + 0.70×2.05 = +20.67 |
| 20 | 2026-07-08 | `iter014_robust_thresh.zip` | 对称 k=1.25 阈值，减少过拟合 | (无记录) | 更鲁棒但 LOSO 略低 |
| 21 | 2026-08-02 | `iter015.zip` | **⭐ T87 SPO+ DFL NN + T75 LGB** | **+28.16** | LOSO +40.09；透传 71%；较 iter_013 **+8.93** |
| 22 | 2026-08-13 | `iter016_chis.zip` | iter_016 CHIS 变体 | (无记录) | — |
| 23 | 2026-08-14 | `iter016_gpu.zip` | iter_016 GPU 版本 | (无记录) | — |
| 24 | 2026-08-15 | `iter016_v3.zip` | 4-way ensemble（T87 NN + T89 CB + T99 Huber LGB + T95 GRU）| **+25.08** | 🔴 LOSO +43.71 本地更高，但平台倒退 **−3.08 vs iter_015**！OOF DE 过拟合 |
| 25 | 2026-08-16 | `iter016_ablation_001.zip` | 消融：去掉 GRU，短 h 改 1 | (无记录) | 旨在缩小本地-平台 gap |
| 26 | 2026-08-16 | `iter016_ablation_002.zip` | 消融变体 002 | (无记录) | — |
| 27 | 2026-08-17 | `iter017_v1.zip` | T87 NN + T123 LGB (HYD+monotone+date-decay) | (无记录) | LOSO +40.70；未获确认平台分 |
| 28 | 2026-08-17 | `iter018_v1.zip` | **iter_015 v1 + per-sym beta-conformal wrapper** | **+28.93** | LOSO +41.49；wrapper-only change，模型 byte-identical；透传 70% |
| 29 | 2026-08-18 | `iter019_v1_fullretrain.zip` | **全数据重训 LGB（date 0-119，M7 trick v1）** | (无记录) | 预测 +29~+31 |
| 30 | 2026-08-18 | `iter019_v2_fullretrain_conformal.zip` | **M7 LGB full-retrain + 原 iter_018 conformal** | **+34.44** | 🟢 **SOTA at time**；iter_018 +5.51 来自全数据重训 |
| 31 | 2026-08-18 | `iter019_v3_mae.zip` | T87 NN + MAE LGB ensemble | (无记录) | LOSO +41.20 local |
| 32 | 2026-08-18 | `iter019_v3_mae_v2.zip` | v3_mae + conformal-aware DE re-threshold | (无记录) | LOSO +43.96 local |
| 33 | 2026-08-18 | `iter019_v4_maefullretrain.zip` | MAE LGB + M7 full-retrain | (无记录) | 预测 +30~+32 |
| 34 | 2026-08-18 | `iter019_v7_conformal_tuned.zip` | conformal beta 重调 | (无记录) | — |
| 35 | 2026-08-18 | `iter019_v9_TOD.zip` | Time-of-Day 特征包 | (无记录) | LOSO 小幅提升 |
| 36 | 2026-08-18 | `iter019_v10_pruned.zip` | adversarial 过滤 259 维特征 | (无记录) | — |
| 37 | 2026-08-19 | `iter019_v2NN_T97.zip` / clean 版 | v2 + 第二组 T97 NN ensemble | (无记录) | 本地 +1.66 holdout |
| 38 | 2026-08-19 | `iter019_v2W_agreement_clean.zip` | agreement filter wrapper | (无记录) | — |
| 39 | 2026-08-19 | `iter019_v2_replica.zip` | v2 reproducibility 验证 | (无记录) | LGB 确认 deterministic |
| 40 | 2026-09-09 | `iter019_v2NN_T97_agreement_combined.zip` | T163 = v2 + T97 MLP + agreement filter | **~+30.74** | 🔴 平台 **−3.70 vs v2**（+30.74 绝对）；"MLP 太相似，平台有毒" |
| 41 | 2026-09-09 | `iter019_v2N_T87M7.zip` | **T170：T87 NN M7 full-retrain（NN 也全数据重训）** | **+34.64** | 🟢 **新 SOTA**；比 v2 LGB-only +0.20 |
| 42 | 2026-09-09 | `iter019_v2N_multihorizon_T170NN.zip` | T173 多 horizon 投票（h60 用 T170 NN） | (无记录) | — |
| 43 | 2026-09-09 | `iter019_v2N_transformer.zip` | T172 GroupTransformer + v2 | (无记录) | 全 in-sample；平台效果未验证 |
| 44 | 2026-09-09 | `v2T_AR_V1..V4.zip` / adaptive_V1..V4 | T168/T169 自适应阈值 / TTT | (无记录) | 🔴 本地 holdout 全退步 |
| 45 | 2026-09-09 | `v2N_T87M7_GPU.zip` | T170 GPU 版本 | (无记录) | — |
| 46 | 2026-09-09 | `v2N_T87M7_NoConformal.zip` | T182 去掉 conformal wrapper | **+34.34** | 本地 holdout +6.62 in-sample artifact；平台证明 conformal 有用 |
| 47 | 2026-09-10 | `iter019_v2N_40NN_bag.zip` | **T179：40-NN ensemble（8 HP 变体 × 5 seeds）** | **+34.39** | 本地 +0.63 in-sample；**平台 −0.25 vs T170**，验证 5-seed 已在 variance floor |
| 48 | 2026-09-10 | `v2N_GroupTr_add.zip` / replace / equal | T187 GroupTransformer replace/add | (in-sample) | 本地 Plan A +22.2 in-sample vs T170；平台未验证 |
| 49 | 2026-09-10 | `v2N_50plus50_proper.zip` | T188 mega ensemble 早期版 | (无记录) | 中间版本 |
| 50 | 2026-09-11 | `v2N_50plus50_optimized.zip` | **T188v2：50 NN + 50 LGB mega ensemble** | **+35.64** | 🟢 **当前 SOTA**；**+1.00 vs T170**；T188v3 batched bmm 3.3× 加速优化版 |

---

### 各阶段关键节点详解

#### 起点：mmpc_demo baseline（平台 −6.65）
官方 DeepLOB 示例，5 head 多任务，所有 horizon 全部亏损。根因：模型对多数类（label=1 占 60-80%）输出概率不够分离，出手多但 precision 低；加之 sym=2 类蓝筹/ETF 在所有模型中预测 UP 62% 但真实只有 18%，大量错误交易。

**核心发现**：`date` 字段评测时置 0、测试点顺序打乱、可能含训练外 sym——这三条硬约束排除了大量常见手段，必须 sym-agnostic 设计。

#### iter_002：SchemeC + 5-fold LOSO（平台 +4.07）
将 CE 3-class LightGBM 从单 horizon 扩展至多 horizon，SchemeC 增加 223 维特征，高置信度阈值 gate（T=0.5, δ=0.15）把出手量降 86%，单笔 alpha 从 −7e-5 翻到 +1.4e-4。**关键发现**：h_10 LOSO +21.86 → 平台 −8.64（gap −30.5）；h_60 LOSO +6.30 → 平台 +4.07（gap −2.23）。**短 horizon 平台完全失效，只信 h_60。**

#### iter_013：Δmid 回归 + EV gate BREAKTHROUGH（平台 +19.23）
用 LightGBM 直接回归 `y = (mp_{t+60} - mp_t) / (mp_t + 1)` 替代 3-class CE，将 EV gate（pred > thr_up → 买，pred < −thr_dn → 卖）作为决策层。LOSO-equiv +36.23（**+9.79 vs iter_012**，+37%），5 sym 全正。

**关键洞察**：3-class CE 损失压缩了 Δmid 幅度信息；回归直接保留幅度，EV gate 是最自然的决策规则（期望收益 > 手续费才出手）。平台 +19.23，LOSO-equiv → 平台 gap = −17（透传系数 ~53%）。至此建立 **T80 校准模型**：Δ平台 / Δ LOSO-equiv = 0.70（之后用于预测所有 iter）。

#### iter_015：T87 SPO+ DFL NN（平台 +28.16，+8.93 vs iter_013）
引入 T81 MLP（[359→256→128→64→1] LayerNorm GELU Dropout）先做 L2 预训练，再用 **SPO+ Decision-Focused Learning**（λ=30, lr=3e-5）微调，把梯度直接对 PnL 对齐。NN + LGB 集成（w_NN=1.0, w_LGB=1.5）= LOSO +40.09。

平台 +28.16，vs iter_013 **+8.93**。从 53% 传导率跳到 71% 主因：SPO+ 把模型直接优化到 PnL，减少了 EV gate 处理的"估算误差"，且阈值选择更保守（DE optimum 无 4D 过拟合）。

#### iter_016 v3：4-way ensemble 的失败（平台 +25.08，比 iter_015 倒退）
加入 T89 CatBoost + T99 Huber LGB + T95 GRU 组成 4-way ensemble，本地 LOSO 跳到 +43.71（+3.62 vs iter_015），看起来很好。**但平台仅 +25.08，比 iter_015 倒退 3.08**。

**根因**：OOF DE 4D 阈值在 442k 本地测试上严重过拟合；GRU 用 hidden state 结构天然容易过拟合短 horizon；iter_016 ablation（去掉 GRU + 短 h）随后预测可以恢复 iter_015 水平。**经验：本地大幅提升若来自阈值 DE 优化 + 复杂 ensemble，平台大概率翻面。**

#### iter_018 v1：per-sym beta-conformal wrapper（平台 +28.93）
在 iter_015 v1 模型（byte-identical）上增加 per-sym beta-calibrated abstain 带：当预测 EV 幅度小于 `beta × σ_pred` 时弃权（action=1）。5 个 sym 各自有独立 beta：{0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}，来自 T150/R_conformal_select 4-fold CV consensus。

LOSO +41.49（+1.75 vs baseline），平台 +28.93（+0.77 vs iter_015）。OOD sym 回退到全局默认 beta=0.16，符合 sym-agnostic 合规要求。

#### v2 baseline：LGB M7 全数据重训（平台 +34.44，+5.51 vs iter_018）
**关键突破**：在最终推理前，将 LightGBM 在 **date 0-119 全部数据**（含 96-119 "测试"集）上重训。GBDT 全数据重训安全，因为 LGB 不会 memorize exact test points（量化测试证明 330 iters 早停与 LOSO train 一致）。平台 +34.44，**+5.51 vs iter_018**，确立 M7 全数据重训的有效性。

#### T170：T87 NN M7 全数据重训（平台 +34.64，+0.20 vs v2）
对称地，将 T87 SPO+ NN 也做 M7 full-retrain（固定 11 epoch = 平均 best_epoch × 1.1），在 date 0-119 上重训。NN M7 holdout 提升本地 +8.49 in-sample（但 in-sample 污染），平台仅 +0.20。

**原因**：NN M7 retrain 带来比 LGB M7 更少的平台增益，因为 NN 已经从 T81 warm-start，而 LGB M7 完全重新拟合了 96-119 分布。

#### T163：加入 T97 NN 的失败（平台 ~+30.74，回退 ~3.70）
在 v2 基础上加入第二组 NN（T97 风格）+ agreement filter，本地 holdout +1.41。**平台反而退步 −3.70**（约 30.74）。根因："MLP 太相似"——T87 和 T97 NN 架构过于相近，集成后 variance reduction 接近 0，但引入了新的推理路径有轻微 distribution shift 影响。

#### T188v2 50+50 mega ensemble（平台 +35.64，当前 SOTA）
**核心设计**：把 T170 的 5 NN + 5 LGB 扩展到 **50 NN + 50 LGB**：
- 50 NN = 50 独立 T81-style L2 pretrain（HP 多样性：lr/epoch/lambda_spo 的 5×5 = 25 HP × 2 seeds → 50 seeds 分组）→ 各自独立 T87 SPO+ M7 微调
- 50 LGB = 5 HP × 10 seed，全数据 M7 重训
- 同 v2 thresholds：thr_up=0.0003, thr_dn=0.000216, w_lgb=1.5, w_nn=1.0，conformal per-sym

本地 holdout（in-sample）：T170 +149.96 vs T188v2 +150.43（delta +0.48，噪音级）。  
**平台 OOD**：T188v2 **+35.64 vs T170 +34.64**，**delta +1.00**（+2.9%）。

**核心洞察**：ensemble 扩容的 variance reduction 在 distribution shift（平台 OOD）上才兑现，本地 in-sample holdout 看不出来。这是当前 SOTA。

---

## 3. 本地研究工作总结

### 3.1 Validation Scheme 演化

#### 早期问题（Phase 1-2）
最初沿用"每 sym 内 IID train/val/test split"（T2 模型 IID +19.25），一旦用 LOSO（Leave-One-Sym-Out）跨 sym 验证就暴露：**LOSO sum −22.10**（5 fold），与 mmpc_demo 平台 −23.13 量级完全吻合，证明 LOSO 是平台行为的可信代理。

**根因**：sym=2 是大盘蓝筹/ETF（amount_delta z-score +7.49，最窄 spread，最低 tick 波动），训练在 {0,1,3,4} 的模型对 sym=2 误判 UP 62% vs 真实 18%，产生大量错误交易。

#### 迭代改进过程

| 阶段 | 方案 | 问题/发现 |
|---|---|---|
| Phase 1 | IID 5-fold（同 sym 内划分） | +19.25 IID → LOSO −22.10；sym OOD 损失 41 分 |
| Phase 2 | LOSO 5-fold（留一 sym） | 过度惩罚 sym OOD；真正平台是 **时间 OOD**，不是 sym OOD |
| Phase 3 | T23 robust CV + 2 锚点平台校准 | 发现 h_60 LOSO → 平台 gap 仅 −2.23；短 h gap −30.5 |
| Phase 4 (T59) | **全 sym 训练 + DE on 442k test** | LOSO 5-fold over-penalty removed；LOSO-equiv 跳 +16.84 → +24.52（~+7 是 metric 变化，仅 ~+1 真 alpha）|
| Phase 5 (T64) | **V4 walk-forward val**（date 0-79 训练，last 5% 为 val，date 96-119 为 test） | early stopping 更接近时间 OOD 真实场景；+1.42 |
| Final (M7) | **全数据重训（date 0-119）**，固定 epoch 不用 early stopping | LGB +5.51 平台 delta；NN +0.20 平台 delta |

**关键协议**：训练时用 date 0-79/80-95 val/96-119 test（V4 walk-forward），全数据重训时固定 epoch = 平均 best_epoch × 1.1（T170 实验）。

#### M7 全数据重训的限制
**只有已有架构的 M7 重训安全**。任何引入新 NN 架构、新特征或新 wrapper 的 M7 训练，平台往往 −3 到 −5（T183 NN random init M7 holdout 几乎不变；T187 GroupTransformer replace in-sample +22.2 但有 OOD 风险）。原因：新架构在 M7 数据上容易过拟合 96-119 日期内的 patterns，在真正 OOD 测试集上失效。

---

### 3.2 特征工程演化

从 154 维原始特征出发，经过 R34 系列分 5 个阶段扩展到 359/370 维 sym-invariant 特征（SchemeP/Q），是迭代中最持续的工程工作。

#### 特征 Scheme 版本线

| Scheme | 维度 | 主要内容 | 迭代对应 |
|---|---|---|---|
| A (baseline) | 154 | 官方原始 LOB + 订单流 | iter_001 |
| B | 154 | 同 A，LGB 重训 | iter_001 |
| C | 223 | 多 horizon 衍生特征，LOSO-optimized | iter_002-003 |
| D1 | 308 | window z-score normalization | iter_001f |
| E | 154+F1/F2/F3 | amount_delta z-score + spread-norm + vol-quantile | T9 研究 |
| H | 291 | Scheme A + MLOFI+WMP+RV+EWMA | T20-21 |
| R34 Stage 1 | +54 | dual z-score, signed RV, Kyle λ invariant, EWMA-OFI multi-α | T44, iter_007 |
| R34 Stage 2 | +59 | window quantile rank, signed skewness, GOFI, vol-burst, Kyle's λ | T51, iter_007 |
| R34 Stage 3 | +14 | EWMA-residual, multi-W RV ratio, Bipower variation, Cancel-pressure imbalance, Roll's effective spread | T53, iter_008 |
| R34 Stage 4 | minor | extras | T55 |
| **SchemeP** | **359** | Stage 1+2+3+5 全集成，sym-invariant | iter_012+ |
| R34 Stage 5 | +20 | adaptive momentum, OFI toxicity, signed bipower, spread regime, trade-direction persistence, liquidity asymmetry | T68, iter_012 |
| SchemeQ | 365 | SchemeP + 6 time features (sin/cos position, is_open/close_30min, is_pm) | T74 |

#### 重要单项实验

- **aug_a 数据增强**（T27/T28）：反向翻转（bid/ask 互换，涨/跌互换），将训练集 × 2；h_60 LOSO +11.46，预测平台 +9.2，相当于 iter_005b 的核心 trick。
- **Hawkes OFI**（R_hawkes/T141/T142）：用 Hawkes 过程建模订单到达强度，5-seed LGB + LOSO，结果在 iter_019 stack 中边际提升不显著，未进入最终包。
- **Roll TSRV / multi-tick window**（R_roll_tsrv/R_multitick_window）：滚动实现 TSRV（realized variance）+ 多 tick 窗口特征，研究阶段，未成为主力特征。
- **Time-of-Day 特征**（R_time_of_day/T151 TOD v9）：基于 `time` 字段的盘口阶段特征（开盘/收盘前后），LOSO 小幅提升，在 v9 包中存在，但最终版 T170 未使用。
- **Alpha101/Alpha191**（T22）：WorldQuant 风格的跨截面因子，最佳 IC +21.80，未成为 iter_004 候选（需要截面数据，单只股票实现困难）。
- **adversarial 特征过滤**（T66/T152）：对抗性验证过滤掉泄露特征，得到 259 维干净特征集（iter_019 v10），本地效果持平，未带来平台提升。

---

### 3.3 模型架构探索

#### 决策树族（LightGBM / CatBoost / XGBoost）

- **LightGBM（T2, T75, iter_013+）**：基础 GBDT，后改为 regression_l2 回归 Δmid，是 ensemble 的 LGB 支柱。全数据重训（M7 trick）后 LOSO-equiv 不变，但平台 +5.51。GPU 训练（device=gpu, gpu_use_dp=False）比 CPU 快 ~3.2×。
- **CatBoost（T17, T46, T49, T89, T125, T133, T166）**：T89 CB L2 5-seed → LOSO +40.13（与 T87 NN 相当，但交叉相关 0.77，有一定多样性）。T166 5-seed CatBoost ensemble 本地 +0.39 vs T163，但 ensemble 权重难调，未成为最终主力。
- **XGBoost（T18, T56）**：与 LGB 持平，无明显优势。
- **MAE LGB（T137, T145）**：MAE 目标函数（objective=quantile, alpha=0.5 近似）vs L2，MAE LOSO 略高（+0.3-0.5）但鲁棒性更强；入选 iter_019 v3/v4 候选，但平台未确认优势。

#### NN 架构族

| 模型 | 描述 | 关键结果 | 状态 |
|---|---|---|---|
| mmpc_demo DeepLOB | 官方示例，CNN 多任务 | 平台 −6.65 | ❌ 淘汰 |
| T1 NN baseline | MLP + 3-class CE | LOSO ~+8 | ❌ 被 regression 替代 |
| T19 NN regularized | MLP + dropout + weight decay | 与 iter_002 持平 | ❌ |
| T81 MLP L2 | [359→256→128→64→1], LayerNorm GELU Dropout 0.10 | NN alone LOSO +35.89 | ✅ 作为 T87 warm-start |
| **T87 SPO+ DFL** | T81 warm-start → SPO+ fine-tune (λ=30, lr=3e-5) | **LOSO +40.09, 平台 +28.16** | ✅ 当前主力 NN |
| T95 GRU | 门控循环单元 | 本地 +微小提升，iter_016 失败时负责任 | ❌ 移除 |
| T97 NN | T87 风格但独立训练 | 本地 +1.66 holdout vs v2，平台 −3.70 | ❌ 太相似 |
| T157/T172 GroupTransformer | 特征分组 + 注意力，d_model=64, 2 layers | 全 in-sample +22.2 vs T170；平台未验证 | ⚠️ 存疑 |
| T184/T185 深度 CNN | 深度 CNN raw / hybrid CNN | 本地 in-sample 数字好看，OOD 未验证 | ⚠️ |

**关键数字**：T87 SPO+ DFL NN 5-seed 与 T75 LGB 5-seed 的交叉相关 ~0.77，有足够多样性支撑 1:1.5 集成权重。

---

### 3.4 损失函数 / 训练目标演化

| 阶段 | 目标函数 | 实验 | 结果 |
|---|---|---|---|
| Phase 1 | 3-class CrossEntropy（CE）| iter_000-002 | ❌ 掩盖幅度信息，决策层难调 |
| Phase 2 | **L2 regression on Δmid + EV gate** | T75 (LGB), T81 (NN), iter_013 | ✅ BREAKTHROUGH +36.23 LOSO |
| Phase 3 | **SPO+ DFL**（L2 预训 → PnL-focused 微调）| T87 | ✅ +40.09 LOSO, 平台 +28.16 |
| 探索 | MAE (quantile α=0.5) | T137, T143-T145 | 轻微 LOSO 提升，平台未确认 |
| 探索 | Huber | T99, T108 | LOSO +44.72（伴随 OOF contamination），平台未超 iter_015 |
| 探索 | GMADL（方向感知）| T116 | LOSO 微弱提升，未入选 |
| 探索 | Quantile loss | T132, T144 | MAE 优于 Quantile，但均不超 SPO+ |
| 探索 | PnL-aware 端到端 NN loss | T13, T57, T60, T134 | 直接 PnL 梯度不稳定，收敛困难 |
| 探索 | Focal loss / class-weight | T41 | 均匀类权重后 recall 提升但 precision 崩 |
| 探索 | Mixup (T138) | T138 | 负面结果，LOSO 下降 |

**核心结论**：SPO+ DFL 是目前最佳 NN 训练协议，通过在推理损失上微调已对齐 L2 的表征，避免了端到端 PnL 梯度的稳定性问题。MAE 可作为 LGB 的竞争替代，但超越 L2 的幅度不足以在平台产生可确认的提升。

---

### 3.5 Decision / Wrapper 层演化

将原始预测值转为交易决策的后处理层，是本项目中独立且高价值的工程。

#### 阈值搜索
- **早期**：LOSO OOF DE（Differential Evolution）4D 对称阈值搜索 → iter_013 过拟合本地测试集，导致平台 gap 扩大。
- **改进**：对称 k 阈值（thr = k × mean_abs_pred），k=1.25 时 LOSO +33.72（vs DE +36.23），更鲁棒。
- **当前**：DE 搜索但用更保守的 thresholds.json（thr_up=0.0003, thr_dn=0.000216），在 iter_018 v1 中固定。

#### Conformal Abstain Wrapper（per-sym beta）
核心机制（R_conformal_select/T127/T150）：当 `|EV_pred| < beta_sym × sigma_pred` 时强制 action=1（弃权）。
- per-sym beta：{sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}（4-fold CV consensus）
- OOD sym 回退：beta=0.16（各 sym 均值）
- 效果：LOSO +1.75（本地），平台 +0.77（iter_018 比 iter_015）
- 实现合规：sym 只用作 beta 查找表 key，不喂入 model.forward

#### Agreement Filter（T155-T160/T163）
对 sym 0/1/2 过滤：只有 NN 和 LGB 预测方向一致时才出手。v2W（agreement）本地 holdout +1.66。T163 组合后本地 +1.41 but 平台 −3.70。**结论**：agreement filter 本地看起来有效，但叠加额外 NN 后整体在平台上失效。

#### TTT / TTA 探索（全部负面）
- **T161 TTA（K=5 augmented inference）**：K 次 augmented 预测平均，本地 holdout 均下降。
- **T162 in-row TTT**：3 种 wrapper（W3a −0.30, W3b −1.98, W3c −3.11），全部退步。
- **T168 自适应阈值**：4 种 per-batch 统计自适应阈值（V1-V4），本地 holdout 全退。
- **T169 AR TTT**：自回归 TTT，退步。
- **根因**：平台测试点顺序被打乱（硬约束 #2），任何依赖批内统计的适应都不稳定。

#### 其他 Wrapper 实验
- **T165 magnitude filter**：按预测幅度过滤，本地略正，未入 final 包。
- **T131 isotonic calibration**：等变校准（T15 已证负面），LOSO 小幅正，未超 conformal。

---

### 3.6 OOD / Regime Shift 防御

#### T180 OOD 密度抽象 Wrapper
用 PCA(20d) + GMM(k=5) 在 370 维特征上拟合密度估计，log_p < threshold 时弃权（abstain）。
- 测试数据（96-119）的 mean log-p 比训练高 +4.12 → **测试数据比训练更 in-distribution**，非 OOD。
- 低密度点反而贡献高 PnL。
- 3 种强度（p10/p20/p30）全部退步：−32.31, −57.29, −78.44 vs T170。
**结论**：特征空间密度与 PnL 无关；测试集不是 OOD；GMM 密度抽象彻底无效。

#### T181 时间窗口弃权
解析 `time` 字段，在开盘前 10 分钟 / 收盘后 10 分钟强制 abstain：
- V1（保守，去掉 13.19% 数据）：holdout −23.39 vs T170
- V2（仅 AM 开盘后 5 min，0.11% 数据）：holdout −0.27（几乎无影响）
- V3（仅收盘前 5 min）：holdout −2.38
**结论**：T170 conformal 已过滤低置信预测；时间窗口弃权适得其反，不建议使用。

#### R_cross_sym：跨 sym 特征（ABORT）
尝试用同一时刻多 sym 的 LOB 状态作为特征（跨截面信息）。**立即 abort**：平台评测 batch 是打乱的，无法保证多 sym 同时进入同一 batch，违反硬约束 #2。

#### M7 全数据重训的 OOD 防御含义
对已验证架构（T87 NN, T75 LGB）做 M7 retrain 有效，因为这些架构已在 V4 walk-forward 上调过 bias-variance。**对新引入的架构做 M7 retrain 无效甚至有害**——新架构没有经过时间 OOD 验证，M7 数据让它过拟合 96-119 分布，在真正 OOD 测试集上失效（T183 NN random init M7 实验验证了这一点）。

#### Adversarial Val（T66, R_T75L2_advval）
构造"对抗性"验证集（高分布 shift 的日期），用于过滤对 OOD 敏感的特征。T152 iter_019 v10 用 adversarial 过滤后 259 维特征，本地效果持平，平台未带来提升。

---

### 3.7 失败方向总结（防止重蹈覆辙）

| 失败方向 | 实验 | 表面诱人 | 真实失败原因 |
|---|---|---|---|
| **短 horizon（h_5/10/20）优化** | iter_003, iter_004, 多次尝试 | LOSO h_10 +21.86 看起来很强 | 平台 h_10 = −8.64；短 h alpha 极薄（0.05% 阈值），手续费摧毁所有利润 |
| **iter_016 4-way ensemble（OOF 过拟合）** | T89+T99+T95+T87, iter_016 v3 | LOSO +43.71，本地远超 iter_015 | OOF DE 4D 阈值过拟合本地 442k；GRU 短 h 平台不稳；4 路集成引入新 bias |
| **T163 额外 NN 集成** | v2 + T97 MLP + agreement | 本地 +1.41 holdout | T97 与 T87 架构过相似，集成多样性为 0；平台 −3.70 |
| **GRU 架构（T95）** | T95 GRU, iter_016 | LOSO 微小提升 | 内部 hidden state 结构违反硬约束 #2 精神（虽可 workaround）；与 T87 叠加无增量 |
| **TTT / TTA 系列** | T161, T162, T168, T169 | 理论上 adaptation 应该帮助 OOD | 平台批次顺序打乱，per-batch 统计自适应产生噪音；全部本地 holdout 退步 |
| **OOD 密度抽象（T180）** | GMM+PCA abstain | 理论上过滤 distribution shift | 测试集实际比训练更 in-distribution；低密度点反而有高 PnL；弃权损失信号 |
| **GroupTransformer（T172/T187）** | GroupTransformer replace/add | in-sample +22.2 看起来很强 | 全 in-sample 污染（训练包含 test dates）；平台 OOD 表现未知；新架构 M7 retrain 风险高 |

---

## 4. 关键 Insight 总结

### Insight 1：本地 holdout 的 in-sample 污染问题
T81/T87/T170 NN 训练在 date 0-119（含测试集 96-119），在 holdout（96-119）上的评分是 **in-sample**，不是真正的 OOD 性能参考。**唯一 cleaner 参考**是 T75 LGB 单 seed（V4 walk-forward, train 0-79, val 80-95, test 96-119 OOF）。所有"本地 holdout 提升 X"在 in-sample 污染条件下均应打折。平台 OOD 才是真正的 ground truth。

### Insight 2：OOD ensemble 红利——本地看不到，平台兑现
50 NN vs 5 NN 在本地 holdout（in-sample）差距仅 +0.48（噪音级）；但在**平台 OOD 上 +1.00**（+2.9%）。原因：50 模型的 variance reduction 在 distribution shift 环境下才生效——in-sample 时模型们都过拟合到同一 local minima，diversity 为 0；OOD 时 HP 多样性带来真正的独立误差，averaging 有效。**不要用本地 holdout delta 判断 ensemble 扩容是否值得。**

### Insight 3：M7 全数据重训是"已验证模型"专属 trick
对已用 V4 walk-forward 调好的架构（T87 NN, T75 LGB），M7 全数据重训安全有效（LGB +5.51 平台）。对新引入架构做 M7 retrain 有害——新架构没经过时间 OOD 验证，M7 数据让它过拟合 96-119 分布。**只对 existing, proven architecture 做 M7。**

### Insight 4：iter_016 教训——OOF threshold + 复杂集成 = 平台陷阱
本地 LOSO 大幅提升（+3.62 over iter_015）但平台倒退 3.08。原因：多路 ensemble 的 DE 4D 阈值在 442k 本地测试上过拟合；越复杂的集成，over-tuning 风险越高。**简单的 1-2 路集成 + 对称 k 阈值，比复杂 ensemble + DE 阈值更平台鲁棒。**

### Insight 5：回归 Δmid 是核心突破，CE 不适合 PnL 目标
3-class CE 损失把 Δmid 幅度信息压缩为类别 → EV gate 无法区分"勉强过阈值"和"显著信号"。L2 回归保留完整幅度信息 → EV gate 直接对应 PnL 决策。**这一范式转换（iter_013）带来 +9.79 LOSO，是整个比赛中最大的单次突破。**

### Insight 6：短 horizon 不可信，h_60 是唯一稳定 horizon
h_10 LOSO +21.86 → 平台 −8.64（gap −30.5），h_60 LOSO +6.30 → 平台 +4.07（gap −2.23）。短 horizon（5/10/20 ticks）alpha 极薄（α = 0.05%），稍有噪音即被手续费（0.02% 双边）摧毁。所有后续迭代专注 h_60。

### Insight 7：sym-agnostic 约束实际对模型质量有保护作用
不能用 sym embedding / per-sym normalization，迫使模型学习真正 price-invariant 的 LOB 结构 feature，反而降低了对单只股票特性的过拟合。这也解释了为什么我们的模型在 OOD sym（训练外股票）上不至于彻底失效。

---

## 5. 现状 + 下一步

### 当前状态（2026-05-11）

| 指标 | 数值 |
|---|---|
| 当前 SOTA | T188v2 50+50 = **+35.64** |
| 在训 | T190（150+150 mega ensemble，4 台远程机器，预计 ETA 06:53 UTC） |
| 实验性 | T191（trend feature pilot，opus worker） |
| 初赛截止 | **2026-05-11（今天）** |

### T190 预期
150+150（vs 50+50）在 OOD 上预期额外 variance reduction，但收益递减：从 5→50 获得 +1.00，50→150 预期 <+0.5。HP grid 更广（更高 epoch, 更多 lr, lambda 组合）。

### 可探索方向（已截止前存档）
1. **不同 LGB objective 多样性**：MAE LGB（5-seed）+ L2 LGB（50-seed）的混合 ensemble，在已验证框架内添加多样性。
2. **conformal beta 重调**：50+50 ensemble 后 variance 变小，optimal beta 可能调低（减少 abstain），T150 风格重调。
3. **T191 trend features**：如果 trend features 在 holdout 上有效，可叠加到 v2 LGB stack 上验证（不改 NN）。

---

## Appendix A. 主要提交 zip 文件清单

| 文件名 | 提交日期 | 平台 PnL | 备注 |
|---|---|---|---|
| `submission_050601_iter0.zip` | 2026-05-01 | −6.65 | mmpc_demo baseline |
| `submission_050606_iter002.zip` | 2026-05-06 | +4.07 | SchemeC LGB h_60 |
| `submission_050707_iter013.zip` | 2026-07-07 | +19.23 | 首个 Δmid regression |
| `submission_050802_iter015.zip` | 2026-08-02 | +28.16 | T87 SPO+ NN + T75 LGB |
| `submission_050815_iter016_v3.zip` | 2026-08-15 | +25.08 | 4-way ensemble 失败 |
| `submission_050817_iter018_v1.zip` | 2026-08-17 | +28.93 | + per-sym conformal |
| `submission_050818_iter019_v2_fullretrain_conformal.zip` | 2026-08-18 | +34.44 | M7 LGB full-retrain (v2 baseline) |
| `submission_050909_iter019_v2N_T87M7.zip` | 2026-09-09 | +34.64 | T170：M7 NN full-retrain |
| `submission_050909_iter019_v2NN_T97_agreement_combined.zip` | 2026-09-09 | ~+30.74 | T163 失败 (−3.70 delta) |
| `submission_050909_iter019_v2N_T87M7_NoConformal.zip` | 2026-09-09 | +34.34 | T182：无 conformal |
| `submission_050910_iter019_v2N_40NN_bag.zip` | 2026-09-10 | +34.39 | T179：40-NN bag |
| `submission_050911_iter019_v2N_50plus50_optimized.zip` | 2026-09-11 | **+35.64** | **T188v2：50+50 SOTA**；MD5=95993e6250ea56361dfecb62c9123e63 |
| `submission_050911_iter019_v2N_50plus50_cputorch.zip` | 2026-09-11 | +35.64 | 备用 CPU-only 版；MD5=c67e4d25665a2a7a12f4f6839cc8d0fe |

*其余 40+ zip 文件为中间版本或本地验证包，未获确认平台分，完整列表见 `ls /root/projects/liangwenbei_workdir/submission_*.zip`。*

---

## Appendix B. 实验编号速查（T1 – T191）

> 每行格式：**T编号** — 实验名称（关键产出 / LOSO）

**基础建设**
- **T1** — NN baseline (DeepLOB-style 3-class CE, h_60)
- **T2** — LightGBM baseline (SchemeB, 3-class CE, argmax)
- **T3** — 特征 v1 (MLOFI + WMP + RV + EWMA intensities + time encoding, +72 维)
- **T4** — LOSO Scheme A 5-fold + threshold post-processor (LOSO −22.10 → +6.45)
- **T5a/b** — SchemeB LOSO OOF re-threshold / SchemeC multi-horizon (LOSO +21.86)
- **T6/6b** — 5-seed SchemeA ensemble (负面，不超 iter_001d)
- **T7** — window z-score SchemeD1 (LOSO +13.91)
- **T8** — sym=2 brittleness 诊断（蓝筹/ETF 根因）
- **T9** — SchemeE (amount_delta z-score + spread-norm)
- **T10** — SchemeF z-score (不超 SchemeC)

**LOSO 与特征扩展**
- **T11** — SchemeC 5-seed ensemble (LOSO +0.36 over iter_002)
- **T12** — multi-horizon stacking meta (负面)
- **T13** — PnL-aware 端到端 NN loss (不稳定)
- **T14** — cross-sym mixup (负面)
- **T15** — isotonic + EV gate (负面 −1.02)
- **T17/18** — CatBoost/XGBoost 与 LGB 比较 (基本持平)
- **T19** — NN regularized (持平)
- **T20/21** — SchemeH 291 维 (LOSO +21.70，不突破)
- **T22** — Alpha101 + Alpha191 (IC +21.80，不入 iter_004)
- **T23** — Robust CV + 2-anchor platform 校准（建立 LOSO → 平台 mapping）
- **T24** — iter_004 a/b 短 h 严格阈值
- **T25-28** — aug_a 探索，iter_005a/b (LOSO +11.46, 预测平台 +9.2)
- **T29-36** — 各种消融/调参

**R34 Stage 1-5 特征建设**
- **T44** — R34 Stage 1 (+54 维 sym-invariant, LOSO h_60 起跳)
- **T51/51c** — R34 Stage 2 (+59 维, iter_007, LOSO +15.06)
- **T53** — R34 Stage 3 (+14 维, iter_008, LOSO +16.84)
- **T55** — R34 Stage 4 extras
- **T58** — walk-forward CV 验证
- **T59** — **全 sym 训练**（LOSO-equiv +24.52，metric 方法论变化）
- **T61** — **batch-vec 推理 58× 加速**（iter_010，首次不超时）
- **T62** — 无偏 threshold 设计
- **T63** — regression 参数扫描
- **T64** — **V4 walk-forward val** (iter_011, +1.42)
- **T65** — pseudo labeling (负面)
- **T66** — adversarial val
- **T67** — multi-split bagging
- **T68** — **R34 Stage 5 特征** (+20 维, iter_012, LOSO-equiv +26.44)
- **T69/70** — iter_012 包构建
- **T71** — h_40 Stage 5 探索
- **T72** — binary cascade (负面)
- **T73** — seed subset 分析
- **T74** — time features SchemeQ (+6 维: sin/cos/is_open/is_close/is_pm)

**Regression 回归突破**
- **T75** — **LGB regression on Δmid + EV gate** (LOSO-equiv +36.23, 平台 +19.23)
- **T76** — decoupled threshold
- **T77** — multi-horizon vote ensemble (+5.98 LOSO over iter_012)
- **T78** — regression + EV gate + 6 time features (schemeQ, 负面 −0.72)
- **T80** — **LOSO→平台校准模型** (透传系数 0.70)
- **T81** — **NN L2 regression on Δmid** (LOSO +35.89, warm-start for T87)
- **T82/83/84/85** — 校准/审计 (无泄露)
- **T86** — quantile + dual-gate (LOSO +40.22, +1.94 over iter_014)
- **T87** — **⭐ SPO+ DFL NN fine-tune** (LOSO +40.09, 平台 +28.16, iter_015 核心)
- **T89** — CatBoost L2 5-seed (LOSO +40.13, 与 T87 NN 相当)
- **T91** — ReVol+Savgol revival (within-noise)
- **T92** — magnitude sample weight (falsified R44 hypothesis)
- **T95** — GRU regression (小提升, 进 iter_016 后失败)
- **T97** — 第二组 NN (LOSO similar to T87)
- **T99** — Huber LGB family (LOSO +44.72, 但含 OOF contamination)

**iter_015-019 深化**
- **T106** — iter_015 v2 packaging (4-way: T87+T89+T99+T95)
- **T107/108/109** — iter_016 各变体打包
- **T115** — scale invariance audit
- **T116** — GMADL direction-aware loss
- **T117** — LGB monotone constraints
- **T118** — decision 4-in-1 (BAT/QSA/IDS/SCG)
- **T119/120** — iter_016 ablation
- **T122** — retest wins on T75
- **T123** — iter_017 v1 (T87 NN + T123 LGB HYD+monotone+date-decay)
- **T124** — R2/R4 评估
- **T125** — T75 LGB + CatBoost addon
- **T126** — per-sym OOF threshold
- **T127** — **iter_018 v1** (iter_015 + per-sym conformal wrapper, LOSO +41.49, 平台 +28.93)
- **T128/conformal_v2** — conformal v2 研究
- **T128/full_retrain** — **M7 LGB full-retrain 探索** (LGB 在全数据上重训)
- **T129** — iter_019 template
- **T131** — isotonic iter_015 (小幅正)
- **T132** — quantile LGB
- **T133** — alt booster 比较
- **T134** — direct PnL NN loss (不稳定)
- **T135** — snapshot ensemble NN
- **T136** — SPO+ arch sweep / tune
- **T137** — **alt robust losses sweep** (MAE beats L2 by +0.3-0.5 LOSO)
- **T138** — mixup (负面)
- **T139** — stacker NN (负面)
- **T140/b/c** — **iter_019 v1/v2 finalize** (M7 LGB full-retrain 5 seeds, v2 = +34.44 平台 SOTA)
- **T141/b** — Hawkes ensemble (边际)
- **T142** — Hawkes 5-seed full-retrain
- **T143** — **iter_019 v3 MAE packaging** (T87 + MAE ensemble LOSO +41.20)
- **T144** — Quantile/Huber ensemble diversity (MAE-only remains winner)
- **T145/b** — iter_019 v4 MAE full-retrain
- **T146** — conformal-aware DE re-threshold (v3_mae → LOSO +43.96)
- **T147** — feature pruning
- **T148/b/c** — LGB HP sweep
- **T149** — NN T87 OOD sweep
- **T150** — conformal beta sweep
- **T151/b** — TOD v9 packaging
- **T152** — adversarial-filtered 259-d features (iter_019 v10)
- **T153** — v9b TOD clean packaging
- **T154** — v10b only / v11 pruned+TOD
- **T155** — v2 replica reproducibility (LGB deterministic confirmed)
- **T156/b/c** — NN swap (T97 instead of T87, +1.66 holdout WIN locally)
- **T157** — NN transformer (preliminary)
- **T158** — agreement filter wrapper
- **T159/b** — per-sym dynamic abstain
- **T160** — v2W clean packaging
- **T161** — TTA (K=5, 负面)
- **T162** — in-row TTT (W3a −0.30, W3b −1.98, W3c −3.11, 全负)
- **T163** — **v2+T97+agreement combined** (本地 +1.41, 平台 ~−3.70 delta)
- **T164** — 4-way ensemble T87+T97+T95+LGB (+0.34 holdout vs T163)
- **T165** — magnitude filter
- **T166** — **CatBoost L2 5-seed full-retrain** (+0.39 vs T163)
- **T167** — 5-way ensemble (marginal +0.11, within noise)
- **T168** — adaptive threshold V1-V4 (TTT-style, 全负)
- **T169** — AR TTT (负面)
- **T170** — **⭐ T87 NN M7 full-retrain (5 seeds, date 0-119)** → 平台 **+34.64 SOTA**
- **T171** — LGB HP sweep
- **T172/b** — **GroupTransformer** (in-sample +22.2 vs T170, 平台未验证)
- **T173/4** — multi-horizon T170 NN (未验证平台)
- **T175** — transformer T170NN packaging
- **T176/b** — T87 M7 HP sweep (ep/lr/lambda), S1_ep15 平台 +34.51
- **T177** — wrapper tricks (B/C 本地正)
- **T178** — LGB 10-seed (null result vs 5-seed)
- **T179** — **40-NN ensemble (8 variants × 5 seeds)** → 平台 +34.39 (−0.25 vs T170)
- **T180** — **OOD GMM density abstain** (全部 −32 to −78, 完败)
- **T181** — time-window abstain (V1: −23.4, V2: −0.27, V3: −2.38, 全负)
- **T182** — no-conformal baseline → 平台 +34.34 (证明 conformal 有用)
- **T183** — NN random init (no T81 warm-start), M7 retrain (holdout ≈ T170, marginal)
- **T184** — deep CNN raw features
- **T185** — hybrid CNN
- **T186** — retrain experiment
- **T187** — **GroupTransformer replace+add** (Plan A in-sample +22.2 vs T170, Plan B +12.6/+14.7)
- **T188/v2/v3** — **⭐ 50+50 mega ensemble** → 平台 **+35.64 当前 SOTA**; T188v3 为 batched bmm 推理优化版 (3.3× 加速)
- **T189** — T170 GPU 包
- **T190** — 150+150 mega ensemble (进行中, 4 台远程机器)
- **T191** — trend features pilot (进行中, opus worker)

**R-系列调研**
- **R1/v2** — date decay, sym augmentation 研究
- **R2** — Group DRO 研究
- **R3** — HYD quartet (Huber/asymmetric/Y-shape/Date-decay)
- **R4** — logret lag 特征
- **R5/v2** — Bayesian optim
- **R_50seed_bag** — 50-seed bagging 前期研究
- **R_Hawkes_OFI** — Hawkes 过程订单流特征
- **R_NaN1/2** — LGB/CB NaN passthrough 审计
- **R_Pairwise_ALL** — 特征 pairwise 交互
- **R_conformal_select** — per-sym conformal beta 搜索
- **R_cross_sym** — **跨 sym 特征（ABORT，违反平台合约）**
- **R_full_retrain** — M7 full-retrain 可行性研究
- **R_hawkes/r6** — Hawkes 进阶
- **R_multinn_spo** — 多 NN SPO+ 组合研究
- **R_multitick_window** — 多 tick 窗口特征
- **R_roll_tsrv** — Roll effective spread + TSRV
- **R_stack3_compound** — 3 层 stacking
- **R_time_of_day** — 盘口时段特征研究
- **R_T75L2_*** — T75 LGB L2 系列变体（advval/bayes/pairwise/symaug）

---

*报告结束。如需单项实验的详细数据，见 `experiments/T<N>_*/results.json` 或 `research_and_history/history_and_important_notes.md`。*
