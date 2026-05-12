# 良文杯 2026 — 终版完整研究历程报告 (2026-05-12)

> **当前 SOTA**: T188v2 50+50 simple-mean = 平台 **+35.64**（h_60）
>
> 本文档综合自 PROGRESS.md、git log（130+ commits）、114 个 experiments/\*/results.json、SUBMISSION_LOG.md 及平台反馈记录。
>
> 本地 eval 方式在项目周期内经历多次变动（见第 7 节），不同阶段的本地 PnL 数字不可直接比较。

---

## 1. 项目概览

**第二届"良文杯"**（厦门大学经济学院 × 黄良文统计学科基金会，协办 AITOPIA）要求用 LOB + 订单流数据预测 5 只匿名 A 股的 midprice 未来 5/10/20/40/60 ticks 后移动方向（涨/平/跌三分类），评分指标为**累计 PnL 收益率（扣 0.01% 双边手续费）**，取 5 个 horizon 中最好的一个计入排名。

**数据规模**：5 只匿名 A 股 × 120 个交易日 × AM/PM 各一场 × 2001 ticks（3 秒/tick）= 1200 个 parquet，共 474MB。每次预测可用过去 ≤ 100 ticks（含当前），154 维原始特征。

**核心评分公式**：
```
pnl_single = [(label−1)·(mp_{t+60}−mp_t) − 0.0001·|label−1|·((mp_{t+60}+1)+(mp_t+1))] / (mp_t+1)
```
- `label=2`（涨）→ 买入再卖出；`label=0`（跌）→ 卖出再买回；`label=1`（平）→ 不出手
- 累计收益率 = 全部 single 之和（排名分）

**三条硬约束（评测时强制）**：
1. `date` 字段被置 0 → 不能作为特征
2. 测试点顺序被打乱 → Predictor 不能维护跨调用 state
3. `sym` 0-4 但可能含训练外股票 → 模型必须 sym-agnostic

从官方 mmpc_demo DeepLOB 基线（平台 **−6.65**）出发，历经约 **192 个 T 系列实验**和 **40+ 个 R/R34 系列调研**，约 **21 次有效平台提交记录**，最终以 **+35.64** 建立 SOTA，相对基线绝对提升超 42 分。

---

## 2. 平台提交完整清单（按时间排序）

> **说明**：
> - "本地 LOSO" 指 5-fold Leave-One-Sym-Out 累计 PnL（早期方法论）
> - "本地 holdout" 指 date 96-119 在 date 0-119 训练模型上的 in-sample 评分（后期方法论；注意：训练包含测试日期，存在 in-sample 污染）
> - "本地 LOSO-equiv" 指全 sym 训练 + DE 阈值搜索后的 442k 本地测试 PnL（中期方法论）
> - 平台 PnL 为确认的公榜得分（取 5 个 horizon 最好一个）

| # | 日期 | zip / iter | 主要 trick | 本地 PnL | 平台 PnL | Δ vs 前 SOTA | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-01 | iter_000_mmpc_demo | 官方 DeepLOB baseline | ≈0（本地单 session 几乎全预测 1） | **−6.65** (best of 5: h_20) | — | h_5=−9.08, h_10=−13.32, h_20=−6.65, h_40=−16.07, h_60=−23.13 |
| 2 | 2026-05-06 | iter_002 SchemeC | LGB 多 horizon + 阈值 gate | LOSO h_60=+6.30 | **+4.07** (h_60) | +10.72 | 首次正分；短 h 平台全负 |
| 3 | 2026-05-06* | iter_001c/d/f | LGB SchemeB/D1 + 阈值 | LOSO +8.15/+11.11/+13.91 | 无记录 | — | 系列调优，未获平台确认 |
| 4 | 2026-05-06* | iter_003 SchemeC 5-seed | h_10 5-seed ensemble | LOSO h_10=+22.22 | 无记录 | — | 仅本地提升 +0.36 |
| 5 | 2026-05-06* | iter_004a/b | 短 h 严格阈值 | h_60 LOSO=+6.30（不变） | 无记录 | — | 纯后处理，保住 h_60 地板 |
| 6 | ~2026-05-26* | iter_005b aug_a 5-seed | aug_a 数据增强 5-seed | LOSO h_60=+11.46 | 无记录 | — | 预测平台 +9.2 |
| 7 | ~2026-07-07 | iter_007 R34 Stage2 | +59 维 sym-invariant 特征 | LOSO h_60=+15.06 | (超时) | — | 254 min feature compute |
| 8 | ~2026-07-07 | iter_008 Stage3 | +14 维额外特征 | LOSO h_60=+16.84 | (超时) | — | 340 维，超时 |
| 9 | ~2026-07-07 | iter_010 batch-vec | 58× 推理加速 | LOSO-equiv=+24.52 | 无记录 | — | 首次不超时 |
| 10 | ~2026-07-07 | iter_012 Stage5+V4 | +20 维 Stage5 + walk-forward val | LOSO-equiv=+26.44 | 无记录 | — | V4 walk-forward +1.42 |
| 11 | ~2026-07-07 | iter_013 回归突破 | Δmid regression + EV gate | LOSO-equiv=+36.23 | **+19.23** | +15.16 | **第一次大突破**；透传系数 ~53% |
| 12 | ~2026-07-08 | iter_014 NN+LGB | NN L2 regression + T75 LGB | LOSO-equiv=+38.28 | 无记录 | — | 预测 +20.67 |
| 13 | 2026-08-02 | iter_015 T87 SPO+ | SPO+ DFL NN + T75 LGB | LOSO-equiv=+40.09 | **+28.16** | +8.93 | **第二次大突破**；透传 71% |
| 14 | 2026-08-15 | iter_016 v3 4-way | +CatBoost+Huber+GRU 4路集成 | LOSO=+43.71 | **+25.08** | −3.08 | **平台倒退**！OOF DE 过拟合 |
| 15 | 2026-08-17 | iter_018 v1 conformal | + per-sym beta conformal | LOSO=+41.49 | **+28.93** | +0.77 | 恢复并超过 iter_015 |
| 16 | 2026-08-18 | iter_019 v2 M7 LGB | LGB M7 全数据重训 | LOSO=+41.49（不变） | **+34.44** | +5.51 | **第三次大突破**；M7 trick 首次确认 |
| 17 | 2026-09-09 | T163 v2+T97+agree | +T97 NN + agreement filter | holdout=+42.90 | ~+30.74 | −3.70 | 倒退；"MLP 太相似" |
| 18 | 2026-09-09 | T170 NN M7 | T87 NN M7 全数据重训 | holdout in-sample +8.49 (vs v2) | **+34.64** | +0.20 | **新 SOTA**（短暂） |
| 19 | 2026-09-09 | T182 无 conformal | 去掉 conformal wrapper | holdout +6.62（无 conformal 更高） | **+34.34** | −0.30 | 证明 conformal 有效 |
| 20 | 2026-09-10 | T179 40-NN ensemble | 40-NN 8 HP × 5 seed | holdout in-sample +0.63 | **+34.39** | −0.25 | 5-seed 已在 variance floor |
| 21 | 2026-09-11 | **T188v2 50+50** | 50 NN + 50 LGB mega ensemble | holdout in-sample +0.48 (vs T170) | **+35.64** | **+1.00** | **当前 SOTA** |
| 22 | 2026-05-12 | T192 H4 bounded NNLS | bounded NNLS 权重 150+150 | holdout +193.94（严重过拟合） | **+31.0** | −4.64 | OLS overfit 确认；KILL |

\* 部分早期提交日期基于 git log 推断，可能与实际提交日期有出入。

---

### 各阶段关键节点详解

#### 节点 1：mmpc_demo 官方 baseline（平台 −6.65）

官方 DeepLOB 示例，5 head 多任务，5 个 horizon 平台全负。根因：模型对多数类（label=1 占 60-80%）输出概率不够分离，出手多但 precision 低；sym=2（蓝筹/ETF）误预测 UP 62% 但真实只有 18%，大量错误交易。本地评测（单 session sym=0 date=119 am）几乎全输出 1（不出手），显示本地 1-session 评测严重低估平台灾难性失败风险。

**关键发现**：全预测 1（不交易）= 0 分，已胜过 mmpc_demo。

#### 节点 2：iter_002 SchemeC LOSO（平台 +4.07）

将 CE 3-class LightGBM 从单 horizon 扩展至多 horizon，SchemeC 226 维特征，高置信度阈值 gate 把出手量降 86%，单笔 alpha 从 −7e-5 翻到 +1.4e-4。**关键发现**：h_10 LOSO +21.86 → 平台 −8.64（gap −30.5）；h_60 LOSO +6.30 → 平台 +4.07（gap −2.23）。**短 horizon 平台完全失效，只信 h_60。**

#### 节点 3：iter_013 Δmid 回归突破（平台 +19.23，+15.16）

用 LightGBM 直接回归 `y = (mp_{t+60} - mp_t) / (mp_t + 1)` 替代 3-class CE，将 EV gate（pred > thr_up → 买，pred < −thr_dn → 卖）作为决策层。LOSO-equiv **+36.23**（+9.79 vs iter_012，+37%），5 sym 全正。

**关键洞察**：3-class CE 损失压缩了 Δmid 幅度信息；回归直接保留幅度，EV gate 是最自然的决策规则。平台 +19.23，LOSO-equiv → 平台 gap = −17（透传系数 ~53%）。

#### 节点 4：iter_015 T87 SPO+ DFL NN（平台 +28.16，+8.93）

引入 T81 MLP（[359→256→128→64→1] LayerNorm GELU Dropout）先做 L2 预训练，再用 **SPO+ Decision-Focused Learning**（λ=30, lr=3e-5）微调，把梯度直接对 PnL 对齐。NN + LGB 集成（w_NN=1.0, w_LGB=1.5）= LOSO-equiv +40.09。平台 +28.16，透传 71%（vs iter_013 的 53%）。

#### 节点 5：iter_016 v3 4-way ensemble 失败（平台 +25.08，−3.08）

加入 T89 CatBoost + T99 Huber LGB + T95 GRU 组成 4-way ensemble，本地 LOSO +43.71（+3.62 vs iter_015）。**但平台仅 +25.08，比 iter_015 倒退 3.08**。根因：OOF DE 4D 阈值在 442k 本地测试上严重过拟合；GRU 内部 hidden state 天然容易过拟合短 horizon。

#### 节点 6：iter_018 v1 per-sym conformal wrapper（平台 +28.93，+0.77）

在 iter_015 v1 模型（byte-identical）上增加 per-sym beta-calibrated abstain 带：当 `|EV_pred| < beta_sym × sigma_pred` 时弃权（action=1）。per-sym beta：{sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}。LOSO-equiv +1.75，平台 +0.77。

#### 节点 7：iter_019 v2 M7 LGB 全数据重训（平台 +34.44，+5.51）

**关键突破**：将 LightGBM 在 date 0-119 全部数据上重训（M7 trick）。GBDT 全数据重训安全，量化测试证明 330 iters 早停与 LOSO train 一致。平台 **+34.44**，**+5.51 vs iter_018**，确立 M7 全数据重训的有效性。

#### 节点 8：T163 额外 NN 失败（平台 ~+30.74，−3.70）

在 v2 基础上加入第二组 NN（T97 风格）+ agreement filter，本地 holdout +1.41。**平台反而退步 −3.70**。根因："MLP 太相似"——T87 和 T97 NN 架构过于相近，集成后 variance reduction ≈ 0。

#### 节点 9：T170 NN M7 全数据重训（平台 +34.64，+0.20）

将 T87 SPO+ NN 也做 M7 full-retrain（固定 11 epoch = 平均 best_epoch × 1.1），在 date 0-119 上重训 5 seeds。NN M7 holdout in-sample +8.49 vs v2（但含 in-sample 污染），平台仅 +0.20。原因：NN 已从 T81 warm-start，M7 带来比 LGB M7 更少的增量。

#### 节点 10：T188v2 50+50 mega ensemble（平台 +35.64，+1.00，当前 SOTA）

把 T170（5 NN + 5 LGB）扩展到 **50 NN + 50 LGB**：
- 50 NN = 50 独立 T81-style L2 pretrain（HP 多样性：lr/epoch/lambda_spo 的 5×5 = 25 HP × 2 seeds）→ 各自独立 T87 SPO+ M7 微调（固定 11 epoch）
- 50 LGB = 5 HP × 10 seed，全数据 M7 重训
- 同 v2 thresholds：thr_up=0.0003, thr_dn=0.000216, w_lgb=1.5, w_nn=1.0，conformal per-sym

本地 holdout（in-sample）：T170 +149.96 vs T188v2 +150.43（delta +0.48，噪音级）。  
**平台 OOD**：T188v2 **+35.64 vs T170 +34.64**，**delta +1.00**（+2.9%）。

#### 节点 11：T192 bounded NNLS（平台 +31.0，−4.64，确认 overfit）

在 T190（150+150）ensemble 上用 bounded NNLS [1/1500, 2/150] per-group 学习权重。本地 holdout 大幅提升（H4 variant：+193.94，val_test_ratio=1.075），**平台 +31.0**，比 T188v2 SOTA 倒退 **−4.64**。val→test ratio > 1.05 是 in-sample noise alignment 的烟雾弹信号。

---

## 3. 逐实验详细记录（T1–T192 + R 系列）

> 按功能阶段分组，组内按时间顺序。本地 PnL 数字均注明方法论（LOSO / LOSO-equiv / holdout in-sample / holdout OOF）。

---

### Phase 0：项目建设与基线（T1–T10）

#### T1: NN baseline（DeepLOB-style 3-class CE, h_60）
**假设**：深度 CNN 在 LOB 数据上能学出 alpha。  
**方法**：DeepLOB 风格 MLP + 3-class CE，h_60，单 seed。  
**本地**：LOSO ~+8（SchemeC 评分框架）。  
**平台**：未提交。  
**结论**：❌ 被 regression 目标替代；3-class CE 的精度导向与 PnL 目标错位。

#### T2: LightGBM baseline（SchemeB, 3-class CE, argmax）
**方法**：LightGBM 154 维 SchemeB 特征，argmax 决策。  
**本地**：LOSO sum −22.10（5 fold，2/5 正）。  
**平台**：未提交（brittle on LOSO）。  
**结论**：❌ 跨 sym OOD 极差；揭示 sym=2 蓝筹/ETF brittle 问题。

#### T3: 特征 v1（MLOFI + WMP + RV + EWMA intensities + time encoding，+72 维）
**方法**：添加多尺度 MLOFI、WMP、已实现波动率、EWMA 指令强度及时间编码。  
**本地**：与 SchemeC 一起使用，LOSO h_10 +21.86（iter_002 核心）。  
**结论**：✅ 进入 iter_002；SchemeC 226 维基础。

#### T4: LOSO Scheme A 5-fold + threshold post-processor
**方法**：第一次系统 LOSO 验证；高置信度 gate（max(p0,p2)≥T 且 >p1+δ）。  
**本地**：SchemeA LOSO OOF +6.45（4/5 fold 正）；iter_001c 打包。  
**结论**：✅ 建立 LOSO 作为平台的可信代理。

#### T5a/b: SchemeB re-threshold / SchemeC multi-horizon
- **T5a**：在 SchemeB LOSO OOF 上重调阈值 → +11.11（iter_001d）。
- **T5b**：SchemeC 多 horizon 5-fold LOSO → h_10=+21.86（iter_002 核心）。  
**本地 before → after**：h_60 LOSO +6.30（vs T2 −22.10）。

#### T6/T6b: 5-seed SchemeA ensemble
**方法**：5 个 seed 概率平均后阈值 gate。  
**本地 before → after**：LOSO sum +8.87（vs iter_001c +6.45），不显著。  
**结论**：❌ 未突破 iter_002；5-seed 在此阶段无红利。

#### T7: window z-score SchemeD1（308 维）
**方法**：原始 154 维 + 100-tick 窗口内 z-score 归一化的 154 维 = 308 维。  
**本地 before → after**：LOSO +13.91（vs iter_001d +11.11）；sym=2 从 −0.98 翻到 +2.24。  
**结论**：✅ sym=2 修复有效；进入 iter_001f。

#### T8: sym=2 brittleness 诊断
**方法**：分析 5 个 sym 的特征分布差异。  
**发现**：sym=2 是蓝筹/ETF（amount_delta z-score +7.49，最窄 spread）；训练在 {0,1,3,4} 的模型对 sym=2 误判 UP 62% vs 真实 18%。  
**结论**：✅ 解释了 LOSO 5 fold 中 sym=2 fold 最差；sym-agnostic 归一化方向确立。

#### T9: SchemeE（amount_delta z-score + spread-norm）
**方法**：针对 T8 诊断的 sym-agnostic 归一化方案。  
**结论**：部分进入后续 SchemeP 体系；未单独提交。

#### T10: SchemeF z-score
**方法**：全局 z-score 归一化。  
**本地**：不超 SchemeC（LOSO h_10 +13.40 vs iter_002 +21.86）。  
**结论**：❌ 窗口内归一化（T7）优于全局归一化（T10）。

---

### Phase 1：Scheme 扩展与 LOSO 调优（T11–T35）

#### T11: SchemeC 5-seed ensemble（5-fold LOSO h_10）
**本地 before → after**：LOSO h_10 +21.86 → +22.22（+0.36）。  
**结论**：⚠️ 边际提升，进入 iter_003；但平台 h_10 从未正向，未确认价值。

#### T12: multi-horizon stacking meta
**方法**：用上层 meta 模型综合 5 个 horizon 输出。  
**本地**：LOSO +17.46（vs base +21.86），退步。  
**结论**：❌ 负面。

#### T13: PnL-aware 端到端 NN loss（PnL 梯度直接反传）
**方法**：用 PnL 公式计算梯度，直接训练 NN。  
**结论**：❌ 不稳定，不收敛；SPO+ DFL 后来以更稳的方式解决了这个问题。

#### T14: cross-sym mixup
**结论**：❌ 负面；与 sym-agnostic 目标冲突。

#### T15: isotonic + EV gate
**方法**：等变校准后阈值决策。  
**本地 before → after**：LOSO h_10 +21.86 → +20.84（−1.02）。  
**结论**：❌ 概率校准损害预测的排序质量。

#### T17: CatBoost vs LightGBM 比较
**结论**：基本持平（LOSO ceiling +22.2），无明显优势。

#### T18: XGBoost vs LightGBM 比较
**结论**：持平，无明显优势。

#### T19: NN regularized（dropout + weight decay）
**结论**：与 iter_002 持平，无提升。

#### T20/T21: SchemeH 291 维（MLOFI + WMP + RV + EWMA 扩展）
**本地**：LOSO h_10 +21.70（vs iter_002 +21.86），持平。  
**结论**：❌ 不超 SchemeC；单纯加维度无效。

#### T22: Alpha101 + Alpha191（WorldQuant 风格跨截面因子）
**发现**：最佳单因子 IC = +21.80；但需要截面数据，单只股票实现困难。  
**结论**：❌ 不入 iter_004；跨截面因子违反平台硬约束。

#### T23: Robust CV + 2-anchor platform 校准
**方法**：用 iter_002/iter_000 建立 LOSO → 平台 mapping（M5 heuristic：anchor 修正）。  
**发现**：h_60 LOSO → 平台 gap ≈ −2.23；短 h gap −30.5。建立**透传系数 ~70%** 预测模型。  
**结论**：✅ 关键基础设施；后续所有"预测平台 X"均基于此。

#### T24: iter_004 a/b（短 h 严格阈值）
**方法**：iter_004a 彻底关闭 h_5/10/20；iter_004b 短 h 超保守阈值。  
**本地**：h_60 LOSO +6.30（不变）；h_5/10/20 LOSO 正但预测平台可能无效。  
**结论**：⚠️ 纯后处理，锁定 iter_002 地板；未获平台确认。

#### T25–T28: aug_a 数据增强（反向翻转）
**方法**：bid/ask 互换，涨/跌 label 互换，训练集 × 2（aug_a）。  
**本地 before → after**：
- 单 seed h_60 LOSO：+6.30 → +9.67（+3.37）
- 5-seed ensemble h_60 LOSO：+9.67 → +11.46（+1.79）
**平台预测**：+9.2（T23 校准）。  
**结论**：✅ 数据增强有效；进入 iter_005a/b。但后来被 Stage 2 特征超越，未成主力。

#### T29–T34: 调参与 CV 建设
- **T30**：DE 4D 阈值优化：aug_a 5-seed LOSO +11.46 → +13.61（+2.15）
- **T34 CV v2**：建立 LO2SO + bootstrap CI + 平台预言机多维验证框架
**结论**：⚠️ T30 DE 证明在 5-fold LOSO 上可以超调，之后 iter_013 的 DE → 平台 gap 验证了这一风险。

#### T35: ReVol + Savitzky-Golay（SchemeK，255 维）
**方法**：ReVol 滚动波动率 + SG 平滑器 +32 维。  
**本地**：单 seed h_60 LOSO +5.53（vs iter_002 +6.30），不超。  
**结论**：❌ SG 对树模型 gain 几乎为 0；不入 iter_007。

---

### Phase 2：R34 系列特征建设（T44–T70）

> 这一阶段为 T44/T51/T53/T55/T68 的连续特征扩展，对应 R34 Stage 1-5。

#### T44: R34 Stage 1（+54 维 sym-invariant 特征）
**新特征**：dual z-score（w=20/50），signed realized vol，Kyle λ rolling，EWMA-OFI multi-α，cancel pressure。  
**本地**：单 seed LOSO h_60 +12.09（vs iter_005b +9.67）；5-seed 达 +13.67。  
**结论**：✅ 突破 iter_005b；Stage 1 进入 iter_007 候选。

#### T51/T51b/T51c: R34 Stage 2（+59 维，GOFI/quantile rank/vol-burst）
**新特征**：window quantile rank（W=20/50/100），signed skewness，GOFI generalized OFI（~30 dim），vol-burst ratio，Kyle's λ rolling（仅 OOD-robust 列）。  
**本地 before → after**：  
- 单 seed LOSO h_60：+12.09（Stage1）→ +13.67（Stage2）
- 5-seed LOSO h_60：+13.67 → **+15.06**  
**结论**：✅ **真正突破**；Stage 1+2 = 5-seed +15.06，超过历史最好。

#### T53: R34 Stage 3（+14 维）
**新特征**：EWMA-residual，multi-W RV ratio，Bipower variation，Cancel-pressure imbalance，Roll's effective spread。  
**本地 before → after**：LOSO h_60 +15.06 → +16.84（+1.78）。  
**结论**：✅ 进入 iter_008；超时问题尚未解决。

#### T55: R34 Stage 4（少量额外特征）
**本地**：edge gain，与 Stage3 相近。  
**结论**：⚠️ 不作为独立 iter；合并进 SchemeP。

#### T58: walk-forward CV 验证
**方法**：测试 V4 walk-forward（train 0-79, val 80-95, test 96-119）是否比 LOSO 更接近平台行为。  
**结论**：✅ V4 确认更接近时间 OOD 场景。

#### T59: 全 sym 训练（取消 LOSO，用全数据）
**方法**：不再用 LOSO 留一 sym，改为全部 5 sym 一起训练。  
**本地 before → after**：LOSO-equiv（5-fold OOF）跳 +16.84 → +24.52（+7.68）。  
**注意**：约 +7 是方法论变化（LOSO 过度惩罚 → 全 sym），真正新增 alpha 约 +1。  
**结论**：✅ 全 sym 训练确立为标准；iter_009 超时问题仍存在。

#### T61: batch-vec 推理加速（58× 速度提升）
**方法**：向量化 feature 提取，单 1024 batch 时间 254 min → 4.5 min。  
**本地 before → after**：推理时间 **254 min → 4.5 min**（58×）。  
**结论**：✅ **解决平台超时问题**；iter_010 首次通过平台 3h 限制。

#### T62: 无偏 threshold 设计
**方法**：探索对称 k 阈值（thr = k × mean_abs_pred），避免 DE 过拟合。  
**结论**：✅ k=1.25 时 LOSO +33.72，比 DE +36.23 鲁棒但低；最终采用 DE 但保守值。

#### T63: regression 参数扫描
**方法**：扫描 EV gate 的各种变体（非对称 thr, 绝对值 thr 等）。  
**结论**：⚠️ 边际，确认原始 EV gate 设计最简洁有效。

#### T64: V4 walk-forward val（date 0-79 训练，80-95 val，96-119 test）
**本地 before → after**：LOSO-equiv +25.94 → +26.44（+0.50，方法论更干净）。  
**结论**：✅ V4 成为标准训练协议；iter_011。

#### T65: pseudo labeling
**结论**：❌ 负面。

#### T66: adversarial val
**方法**：构造分布偏移最大的验证集过滤特征。  
**结论**：⚠️ 后来 T152 用此方法得到 259 维特征集，平台未带来提升。

#### T67: multi-split bagging
**结论**：⚠️ 边际，未超 5-seed ensemble。

#### T68: R34 Stage 5（+20 维）
**新特征**：adaptive momentum，OFI toxicity，signed bipower，spread regime，trade-direction persistence，liquidity asymmetry。  
**本地 before → after**：LOSO-equiv +25.94 → +26.44（+0.50）。  
**结论**：✅ Stage 5 = SchemeP 359 维完成；iter_012。

#### T69/T70/T70b: iter_012 包构建
**结论**：✅ 打包工作；iter_012 = LOSO-equiv +26.44（不超时，V4 walk-forward）。

#### T71: h_40 Stage 5 探索
**结论**：⚠️ h_40 局部有效，但平台评分取 h_60，未入 final 包。

#### T72: binary cascade
**结论**：❌ 负面，复杂度高。

#### T73: seed subset 分析
**结论**：✅ 确认 5-seed 足够稳定。

#### T74: time features SchemeQ（+6 维：sin/cos position, is_open/close_30min, is_pm）
**本地 before → after**：LOSO-equiv +26.44 → （对 T75 LGB 负面 −0.72，见 T78）。  
**结论**：❌ 时间特征对 LGB 反而降低鲁棒性；SchemeQ 不进入最终包。

---

### Phase 3：回归目标 + SPO+ DFL 突破（T75–T92）

#### T75: LGB regression on Δmid + EV gate（核心突破）
**假设**：回归目标保留幅度信息，EV gate 直接对应期望收益。  
**方法**：LightGBM objective=regression_l2，y = (mp_{t+n} - mp_t) / (mp_t + 1)，SchemeP 359 维，V4 walk-forward，5 seed，EV gate（thr_up/thr_dn DE 搜索）。  
**本地 before → after**（LOSO-equiv）：+26.44 → **+36.23**（**+9.79，+37%**）。  
**平台**（iter_013）：**+19.23**（透传系数 53%，gap −17）。  
**结论**：✅ **整个比赛最大单次突破**；确立 regression 范式。

#### T76: decoupled threshold
**方法**：上涨/下跌阈值分开搜索（非对称）。  
**结论**：⚠️ 边际，最终采用非对称 thr_up/thr_dn = 0.0003/0.000216。

#### T77: multi-horizon vote ensemble（h_5/10/20/40/60 投票）
**本地 before → after**：iter_012（SchemeC）+5.98 LOSO-equiv over iter_012。  
**结论**：✅ 此 trick 后来以 conformal 形式继续存在，但多 horizon 投票在回归框架下已过时。

#### T78: regression + EV gate + 6 time features（SchemeQ）
**本地**：LOSO-equiv +36.23 → +35.51（**−0.72**）。  
**结论**：❌ time features 在回归框架下负面；SchemeQ 不进 iter_013。

#### T80: LOSO → 平台校准模型
**发现**：基于 iter_002/iter_013 两个锚点建立线性校准：Δ平台 ≈ 0.70 × Δ LOSO-equiv。  
**结论**：✅ 透传系数 0.70 用于后续所有预测；在 iter_013-iter_015 阶段（53-71%）实际透传有较大方差。

#### T81: NN L2 regression on Δmid（MLP warm-start）
**方法**：[359→256→128→64→1]，LayerNorm，GELU，Dropout 0.10，regression_l2 预训练。  
**本地**：NN alone LOSO +35.89（vs LGB T75 +36.23，互补）。  
**结论**：✅ 作为 T87 SPO+ fine-tune 的 warm-start；独立性能与 LGB 相当。

#### T82/T83/T84/T85: 校准与审计
**发现**：无特征泄露（date=0 评测条件下），V4 walk-forward 是 cleaner 的 OOD 代理。  
**结论**：✅ 基础设施确认。

#### T86: quantile + dual-gate（IQR-mid hybrid）
**本地**：LOSO-equiv +40.22（+1.94 over iter_014 +38.28）。  
**结论**：⚠️ 比 T87 SPO+ 低，最终用 SPO+ 替代。

#### T87: SPO+ DFL NN fine-tune（⭐ 关键）
**方法**：T81 warm-start → SPO+（λ=30, lr=3e-5, 5 seeds）fine-tune，把梯度直接对 PnL 对齐。  
**本地 before → after**（LOSO-equiv）：T81 NN alone +35.89 → **+40.09**（+4.20）；NN+LGB ensemble。  
**平台**（iter_015）：**+28.16**（+8.93 vs iter_013）。  
**结论**：✅ **当前主力 NN**；在整个项目中始终是 NN 侧的最强选择。

#### T89: CatBoost L2 5-seed
**本地**：LOSO +40.13（与 T87 NN 相当，相关性 0.77）。  
**结论**：⚠️ 有多样性，进入 iter_016 候选；但 4-way ensemble 失败后淘汰。

#### T91: ReVol + Savitzky-Golay revival
**本地**：within-noise（不超 T87 组合）。  
**结论**：❌ 不进 iter_015 候选。

#### T92: magnitude sample weight（R44 假设验证）
**假设**：按 Δmid 幅度加权训练样本，大幅度样本权重高。  
**结论**：❌ R44 假设证伪；幅度加权反而损害 PnL。

---

### Phase 4：iter_015-019 深化（T95–T160）

#### T95: GRU regression
**方法**：门控循环单元，回归 Δmid。  
**本地**：LOSO 微小提升（进入 iter_016）。  
**结论**：❌ 在 4-way ensemble 中负贡献（hidden state 容易过拟合短 horizon）；后期移除。

#### T97: 第二组 NN（T87 风格，独立训练）
**本地**：LOSO 与 T87 相近（独立性不足）。  
**结论**：❌ 与 T87 太相似（架构同源），集成 diversity ≈ 0；T163 失败的根因。

#### T99: Huber LGB family
**本地**：LOSO +44.72（但含 OOF contamination）。  
**结论**：⚠️ 看起来高于 T87，但 OOF contamination 警告；进入 iter_016 后失败。

#### T106-T109: iter_015/016 打包系列
- **T106**：iter_015 v2 打包（4-way：T87+T89+T99+T95）
- **T107/T108/T109**：iter_016 各变体（CHIS/GPU/v3）  
**平台**（iter_016 v3）：**+25.08**（4-way ensemble 失败，倒退 −3.08）。

#### T115: scale invariance audit
**发现**：模型在 scale 变化下输出稳定，证明 sym-agnostic 设计正确。

#### T116: GMADL direction-aware loss
**方法**：方向感知梯度调整。  
**本地**：LOSO 微弱提升，不超 SPO+。  
**结论**：❌ 不入选。

#### T117: LGB monotone constraints
**方法**：对部分特征施加单调性约束。  
**结论**：⚠️ 局部有效，进入 iter_017 (T123 HYD+monotone+date-decay)。

#### T118: decision 4-in-1（BAT/QSA/IDS/SCG）
**结论**：⚠️ 探索性，边际。

#### T119/T120: iter_016 ablation
**方法**：去掉 GRU，短 h 改 1；消融确认 GRU 是 iter_016 失败的主因。  
**结论**：✅ ablation 教训：简单集成比复杂 4-way 更平台鲁棒。

#### T122: retest wins on T75
**结论**：✅ T75 LGB 可复现性确认。

#### T123: iter_017 v1（T87 NN + T123 LGB HYD+monotone+date-decay）
**本地**：LOSO +40.70（+0.61 vs iter_015 +40.09）。  
**平台**：无确认记录。  
**结论**：⚠️ 未证明平台有效。

#### T124: R2/R4 评估（logret lag 特征）
**结论**：❌ 无增量。

#### T125: T75 LGB + CatBoost addon
**结论**：⚠️ 边际，不超 T87 组合。

#### T126: per-sym OOF threshold
**方法**：每个 sym 独立 OOF 最优阈值。  
**结论**：❌ 过拟合风险高；改用全局阈值 + per-sym conformal。

#### T127: iter_018 v1（iter_015 v1 + per-sym beta conformal）
**方法**：T87 5-seed + T75 5-seed（byte-identical）+ conformal abstain wrapper。  
**本地 before → after**：LOSO-equiv +39.75（iter_015 v1）→ **+41.49**（+1.75）。  
**平台**（iter_018 v1）：**+28.93**（+0.77 vs iter_015）。  
**结论**：✅ conformal wrapper 验证有效；进入所有后续提交。

#### T128: conformal v2 研究 + M7 LGB 可行性探索
**发现**：LGB M7 full-retrain（date 0-119）安全可行；LOSO 不变，但模型更全面拟合已知数据。  
**结论**：✅ **M7 trick 的起点**；后在 T140 正式实现。

#### T129: iter_019 template
**结论**：✅ 统一打包框架建立。

#### T131: isotonic calibration（iter_015 + isotonic）
**本地**：小幅正，不超 conformal。  
**结论**：❌ 被 conformal 取代。

#### T132: quantile LGB
**方法**：quantile regression（alpha=0.5 ≈ MAE）。  
**结论**：⚠️ MAE 优于 Quantile，但均不超 SPO+ NN 组合。

#### T133: alt booster 比较
**结论**：⚠️ 与 LGB 持平，不替换主力。

#### T134: direct PnL NN loss（不稳定）
**结论**：❌ 梯度不稳定，不收敛；SPO+ 仍是最佳。

#### T135: snapshot ensemble NN
**结论**：⚠️ 边际提升，不超 T87 5-seed。

#### T136: SPO+ arch sweep / tune
**方法**：SPO+ 超参扫描（lambda, lr 等）。  
**结论**：✅ 确认 λ=30, lr=3e-5 是最优；进入 T87 最终配置。

#### T137: alt robust losses sweep（MAE vs L2）
**本地 before → after**：L2 LGB LOSO ≈ X → MAE LGB LOSO +0.3-0.5（边际提升）。  
**结论**：⚠️ MAE 轻微优于 L2；但平台未确认；最终 L2 LGB 仍为主力。

#### T138: mixup
**本地**：LOSO 下降。  
**结论**：❌ 负面。

#### T139: stacker NN
**结论**：❌ 负面。

---

### Phase 5：M7 全数据重训系列（T140–T155）

#### T140/T140b/T140c: iter_019 v1/v2 finalize（M7 LGB 全数据重训）
**方法**：LGB 在 date 0-119（全部 120 天）上重训，固定 iters = 330（V4 walk-forward 早停平均 × 1.1）。  
**本地 before → after**：LOSO-equiv +41.49（iter_018）→ 不变（M7 不改变 LOSO，只改全数据模型）。  
**平台**（iter_019 v2）：**+34.44**（+5.51 vs iter_018 +28.93）。  
**结论**：✅ **M7 trick 确认有效**；LOSO-equiv 不预测 M7 增量（only OOD 拿到）。

#### T141/T141b: Hawkes ensemble
**方法**：Hawkes 过程订单流特征，5-seed LGB + LOSO。  
**本地**：边际提升。  
**结论**：⚠️ 在 iter_019 stack 中边际不显著，未进入最终包。

#### T142: Hawkes 5-seed full-retrain
**结论**：⚠️ 与 T141b 类似；作为 iter_019 v3 候选之一。

#### T143: iter_019 v3 MAE packaging（T87 + MAE LGB）
**本地 before → after**：LOSO +41.49（v2）→ +41.20（T87+MAE，−0.29）；加 conformal → +43.21。  
**结论**：⚠️ 带 conformal 的 MAE ensemble LOSO 更高，但平台未确认超过 v2。

#### T144: Quantile/Huber ensemble diversity
**结论**：⚠️ MAE-only remains winner；多种 robust loss 混合无增量。

#### T145/T145b: iter_019 v4 MAE full-retrain
**本地**：M7 MAE LGB + T87 NN ensemble。  
**结论**：⚠️ 平台预测 +30-32，低于 v2（+34.44）；MAE M7 LGB 不超 L2 M7 LGB。

#### T146: conformal-aware DE re-threshold（v3_mae）
**本地 before → after**：iter_019 v3_mae LOSO +41.20 → +43.96（+2.76）。  
**结论**：⚠️ 看起来高，但 DE 超调风险；未提交平台。

#### T147: feature pruning
**方法**：按 importance 剪枝特征。  
**结论**：✅ 轻微正，用于 v10/v11 候选。

#### T148/T148b/T148c: LGB HP sweep
**结论**：⚠️ 找到轻微更优 HP，边际。

#### T149: NN T87 OOD sweep
**方法**：扫描 T87 对 OOD sym 的鲁棒性。  
**结论**：✅ 确认 T87 在 OOD sym 下表现可接受。

#### T150: conformal beta sweep
**方法**：对 5 个 sym 分别扫描 beta（0.0-0.5），4-fold CV consensus。  
**发现**：最优 beta = {sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}。  
**结论**：✅ 固化进入所有后续提交的 conformal 配置。

#### T151/T151b: TOD v9 packaging（Time-of-Day 特征）
**方法**：基于 `time` 字段的盘口阶段特征；LOSO 小幅提升。  
**结论**：⚠️ 进入 iter_019 v9，但最终版 T170 未使用。

#### T152: adversarial-filtered 259 维特征（iter_019 v10）
**方法**：对抗性验证过滤泄露特征，从 359 → 259 维。  
**本地**：LOSO 持平。  
**平台**：无确认提升。  
**结论**：❌ 未带来平台提升；特征已足够鲁棒。

#### T153/T154: v9b TOD clean + v10b/v11 packaging
**结论**：⚠️ 候选包，未获平台确认优势。

#### T155: v2 replica reproducibility test
**发现**：LGB 训练是确定性的（deterministic confirmed）；相同配置结果可完全复现。  
**结论**：✅ 基础设施确认。

---

### Phase 6：NN 扩展与 Wrapper 实验（T156–T169）

#### T156/T156b/T156c: NN swap（T97 替代/叠加 T87）
**本地 before → after**：v2 holdout +41.49 → +43.16（T97 替代 T87，+1.66）。  
**结论**：⚠️ 本地有效但 T97 与 T87 太相似；T163 组合后平台失败。

#### T157: NN transformer（preliminary）
**结论**：⚠️ 初步实验，进入 T172 完整实验。

#### T158: agreement filter wrapper
**方法**：只有 NN 和 LGB 预测方向一致时才出手。  
**本地 before → after**：holdout +41.49 → +41.74（+0.25）。  
**结论**：⚠️ 本地轻微正；但 T163 组合后平台失败。

#### T159/T159b: per-sym dynamic abstain
**方法**：每个 sym 动态计算 abstain 阈值。  
**本地**：holdout null result。  
**结论**：❌ 无效。

#### T160: v2W clean packaging（agreement filter）
**结论**：⚠️ 打包 agreement filter 版本。

#### T161: TTA（Test-Time Augmentation，K=5）
**方法**：5 次增强预测平均。  
**本地 before → after**：holdout T170 +149.95 → **下降**（两种方法均 hurt）。  
**结论**：❌ 平台批次顺序打乱，per-batch 统计自适应产生噪音。

#### T162: in-row TTT（Test-Time Training）
**方法**：3 种 wrapper（W3a/W3b/W3c）。  
**本地 before → after**：holdout T170 → W3a −0.30, W3b −1.98, W3c −3.11（全退步）。  
**结论**：❌ 违反硬约束 #2 精神；全部负面。

#### T163: v2+T97+agreement combined（平台 ~+30.74，−3.70 vs v2）
**本地**：holdout +42.90（+1.41 vs v2 +41.49）。  
**平台**：**~+30.74**（**−3.70 vs v2 +34.44**）。  
**结论**：❌ T97 与 T87 架构过相似，集成多样性为 0；agreement filter 的本地收益被 T97 OOD 失败抵消。

#### T164: 4-way ensemble（T87+T97+T95+LGB）
**本地**：holdout +43.26（+0.34 vs T163）。  
**结论**：⚠️ 略好于 T163 但仍含 T97，不安全。

#### T165: magnitude filter
**方法**：按预测幅度过滤，只保留高幅度预测。  
**本地**：holdout 略正。  
**结论**：⚠️ 不够显著，未入 final 包。

#### T166: CatBoost L2 5-seed full-retrain
**本地 before → after**：holdout T163 → **+0.39**。  
**结论**：⚠️ 边际正，但后续被 50+50 ensemble 替代。

#### T167: 5-way ensemble（T87+T97+T95+CB+LGB）
**本地**：holdout marginal +0.11 over T166。  
**结论**：❌ within noise；不入 final。

#### T168: adaptive threshold V1-V4（TTT-style）
**方法**：4 种 per-batch 统计自适应阈值（batch 内预测分布统计）。  
**本地 before → after**：holdout T170 → 全部退步（V1-V4 均负）。  
**结论**：❌ 平台测试点顺序被打乱，per-batch 统计不稳定。

#### T169: AR TTT（自回归 test-time training）
**本地**：holdout 退步。  
**结论**：❌ 负面；TTT 系列全部 kill。

---

### Phase 7：M7 NN 重训与扩展集成（T170–T192）

#### T170: T87 NN M7 full-retrain（5 seeds, date 0-119，平台 +34.64，新 SOTA）
**方法**：将 T87 SPO+ NN 在 date 0-119 全数据上重训，固定 epoch=11（avg best_epoch 9.2 × 1.1），warm-start from T81。  
**本地 before → after**（in-sample holdout）：v2 LGB-only baseline +109.68 → v2N（NN 加入）+118.16（+8.49，但 in-sample 污染）。  
**平台**（iter_019 v2N）：**+34.64**（+0.20 vs v2 +34.44）。  
**结论**：✅ **新 SOTA**；NN M7 增量比 LGB M7 小（LGB M7 +5.51 vs NN M7 +0.20），原因是 NN 已从 T81 warm-start，且 M7 NN 的 OOD 泛化有限。

#### T171: LGB HP sweep
**方法**：扫描 LGB 超参（learning_rate, n_estimators, num_leaves 等）。  
**结论**：⚠️ 轻微边际提升，未进入 final 包。

#### T172/T172b: GroupTransformer（特征分组 + 注意力）
**架构**：GroupTransformer (d_model=64, nhead=4, 2 layers, n_tokens=32, CLS token)；输入 359 维 → Linear(359, 32×64) → (B, 32, 64) super-tokens → TransformerEncoder → head。  
**本地**（in-sample，全部在 date 0-119 训练）：  
- Transformer only PnL：+149.91 in-sample  
- 最佳 ensemble [NN=1.0, TR=0.5, LGB=1.0]：**+164.71**（+55.04 vs v2 baseline +109.67）  
**平台**：未提交（⚠️ 全 in-sample 污染，无真实 OOD 验证）。  
**结论**：⚠️ in-sample 数字高度夸大；不安全提交。

#### T173/T174: multi-horizon T170 NN 投票
**方法**：h_5/10/20/40 各使用 T170 NN 风格模型投票，h_60 作主导。  
**结论**：⚠️ 理论有效，但短 horizon 平台始终不稳；未获确认。

#### T175: Transformer T170NN packaging
**结论**：⚠️ 打包工作，zip 构建完成但含 in-sample 风险。

#### T176/T176b: T87 NN M7 HP variant sweep（ep/lr/lambda）
**方法**：sweep NN M7 超参：ep={7,11,15,20}, lr={1e-5,3e-5,1e-4}, lambda={10,30,100}。  
**本地**：各变体 holdout 在 149-151 范围内波动（噪音级）。  
**平台**（S1_ep15 variant）：+34.51（vs T170 +34.64，−0.13）。  
**结论**：⚠️ HP 多样性已成为 T188v2 50+50 ensemble 多样性的来源，但单一 HP variant 无优势。

#### T177: wrapper tricks（B/C 两种变体）
**本地**：holdout 正，B/C 均轻微提升。  
**结论**：⚠️ 未入 final 包。

#### T178: LGB 10-seed（vs 5-seed）
**本地 before → after**：5-seed → 10-seed holdout **null result**（差异噪音级）。  
**结论**：❌ 5-seed 已在 variance floor；单纯加 seed 无红利。注意：这是 **LGB 5-seed 的 floor**；T188v2 的 50 LGB 通过 HP 多样性（5 HP × 10 seed）规避了这个问题。

#### T179: 40-NN ensemble（8 HP 变体 × 5 seeds）
**方法**：8 种 NN HP 变体（ep=7/11/15/20, lr=1e-5/3e-5/1e-4, lambda=10/30/100 的子集）× 5 seeds = 40 NN，配合 5 LGB。  
**本地 before → after**（in-sample holdout）：T170（5 NN）+149.95 → T179（40 NN）+150.57（**+0.63**）。  
**推理时间**：816 ms / 1024 batch（vs T170 855ms）。  
**平台**：**+34.39**（**−0.25 vs T170 +34.64**）。  
**结论**：❌ 验证 **5-seed 已在 variance floor**；40 NN 在平台上无提升（−0.25）。这与 T188v2 50+50 的 +1.00 形成对比——关键差异：T188v2 有 LGB 侧扩充（50 vs 5），而 T179 只扩了 NN 侧。

#### T180: OOD GMM density abstain wrapper
**方法**：PCA(20d) + GMM(k=5) 在 370 维特征上拟合密度估计，log_p < threshold 时弃权。  
**发现**：测试数据（96-119）mean log-p = −31.18，比训练（0-95）的 −35.30 高 **+4.12** → **测试数据比训练更 in-distribution**，非 OOD！低密度点反而贡献高 PnL。  
**本地 before → after**（holdout, T170 = +149.95）：
- V1 (p10)：+117.63（**−32.31**，8.0% abstain）
- V2 (p20)：+92.65（**−57.29**，15.0% abstain）
- V3 (p30)：+71.50（**−78.44**，22.6% abstain）  
**结论**：❌ GMM 密度抽象彻底无效；未构建 zip；测试集不是 OOD；低密度点有高 PnL。

#### T181: time-window hard abstain
**方法**：在开盘/收盘前后强制弃权（解析 `time` 字段）。  
**本地 before → after**（holdout, T170 = +149.95）：
- V1（保守，去 13.19%）：**−23.39**
- V2（仅 AM 开盘后 5 min，0.11%）：**−0.27**
- V3（仅收盘前 5 min）：**−2.38**  
**结论**：❌ T170 conformal 已过滤低置信预测；时间窗口弃权适得其反。

#### T182: 无 conformal wrapper（ablation）
**方法**：T170 但去掉 per-sym beta conformal wrapper。  
**本地 before → after**（holdout）：T170 with conformal +149.95 → T182 no conformal **+156.56**（**+6.62**！但 in-sample artifact）。  
per-sym delta：sym1（beta=0.4）delta +5.34，sym2（beta=0.3）+1.12。  
**推理**：conformal 在 holdout 上 hurt（过度 abstain），但平台不同。  
**平台**：**+34.34**（**−0.30 vs T170 +34.64**）。  
**结论**：❌ **conformal 在平台上有用**（+0.30 vs 无 conformal）。本地 holdout 显示 conformal 负面是 in-sample artifact——训练数据上的 abstain 损失真实 PnL，但平台 OOD 上 conformal 帮助过滤噪声预测。

#### T183: NN random init（无 T81 warm-start），M7 retrain
**假设**：random init + M7 是否可以与 T81 warm-start 竞争？  
**本地 before → after**（holdout）：T170 +149.95 → T183 **+150.57**（+0.63，几乎相同）。  
**结论**：⚠️ Random init M7 retrain 本地持平 T81 warm-start；但不同的初始化带来 diversity——T188v2 的 NN 多样性部分来自不同 HP，等效于不同收敛点。

#### T184/T185: 深度 CNN（raw features / hybrid CNN）
**方法**：深度 CNN 结构，直接处理原始特征序列。  
**本地**：in-sample 数字看起来不错，但 OOD 未验证。  
**结论**：❌ 未构建提交包；OOD 表现不明，风险过高。

#### T186: T184/T172b retrain with val + LR sweep
**方法**：尝试用 early stop + LR sweep 修复深度架构。  
**发现**：架构不适合（verdict: "architecture_doesnt_fit"）。  
**结论**：❌ Deep CNN/Transformer 系列在当前特征框架下无法与 MLP 竞争。

#### T187: GroupTransformer replace + add（Plan A/B）
**方法**：将 T172b GroupTransformer 加入 v2 stack。  
**本地**（in-sample）：
- Plan A（replace T87 NN with GroupTransformer）：holdout **+22.2 vs T170**（严重 in-sample inflate）
- Plan B（add to v2 stack）：+12.6 / +14.7  
**结论**：❌ 全 in-sample 污染（GroupTransformer 训练包含测试日期）；平台 OOD 表现未知；新架构 M7 retrain 高风险，不提交。

#### T188 → T188v2: 50+50 mega ensemble（当前 SOTA，平台 +35.64）

**T188 初版**：  
早期版本存在 warm-start 问题（T81 pretrain 未正确分离 val/train）。  
**本地 holdout（in-sample）**：~+150（边际优于 T170 +149.96）。

**T188v2 修正版（proper T81 pretrain）**：  
- 50 NN：seeds 1-50，各自独立 T81-style L2 pretrain（phase1 用 schemeP_val dates 80-95）→ T87 SPO+ M7 11 epoch 微调
- 50 LGB：seeds 1-50（5 HP × 10 seed），全数据 M7 重训
- 相同配置：thr_up=0.0003, thr_dn=0.000216, w_lgb=1.5, w_nn=1.0，conformal per-sym beta

**本地 before → after**（in-sample holdout）：
- T170（5+5）：+149.96 / 191,697 active
- T188v2（50+50）：**+150.43 / 192,413 active**（**delta +0.48**，噪音级）

**平台 OOD**：
- T170：+34.64
- **T188v2：+35.64**（**delta +1.00**，+2.9%）

**zip**：`submission_050911_iter019_v2N_50plus50_optimized.zip`（148MB，MD5=95993e6250ea56361dfecb62c9123e63）。

**结论**：✅ **当前 SOTA**。ensemble 扩容的 variance reduction 在 distribution shift（平台 OOD）上才兑现，本地 in-sample holdout 看不出。

#### T188v3: batched torch CUDA inference 优化（非模型变化）
**方法**：将 50 NN 的 sequential numpy 推理改为 batched torch CUDA bmm（所有 50 NN 权重堆叠成 (50, out, in) 张量，单次 torch.bmm 替代 50 次 numpy 循环）。

**推理速度 before → after**（per 1024 batch）：
- NN-only（baseline spec）：1509ms → V3 torch CUDA **8.1ms**（**186× 加速**）
- 本地实测 sequential numpy：554.8ms → V3 torch CUDA 8.1ms（**68.5×**）
- e2e（参考值）：2142ms → **641ms**（**3.3× 加速**）；442k 行总时间 15.4 min → 4.6 min

**行动一致性**：actions 与原 T188v2 100% 一致（max diff 6.98e-10，远低于 1e-4 容忍度）。  
**zip**：同 T188v2，使用该 optimized 版本提交（MD5=95993e6250ea56361dfecb62c9123e63）。

#### T188v3_fullhorizon：h=5/10/20/40 share h=60 ensemble
**方法**：最终打包时，所有 5 个 horizon 使用同一套 T188v2 50+50 h_60 ensemble 的预测，按各 horizon 的阈值决策。  
**结论**：⚠️ 共享 h=60 ensemble 作为 h=5/10/20/40 的预测 — 本质是"5 个 horizon 免费拿到"。但短 horizon 平台历史上均负，意义有限。

#### T189/T189_GPU_pkg: T170 GPU 推理包
**方法**：T170（5+5）转为 torch GPU 推理。  
**推理时间**：CPU +855ms → GPU 版本更快（benchmark 确认）。  
**结论**：✅ 打包完成；平台 GPU 环境可直接使用。

#### T190: 150+150 mega ensemble（平台 +34.59，−1.05 vs T188v2）
**方法**：从 50+50 扩展到 150 NN + 150 LGB（4 台远程机器并行训练）。  
- 150 NN：更广 HP grid（更高 epoch, 更多 lr, lambda 组合）
- 150 LGB：更多 seed（5 HP × 30 seed）

**本地 before → after**（in-sample holdout）：
- T170（5+5）：+146.24 / 191,697 active
- T188v2（50+50）：+150.43 / 192,413 active
- **T190（150+150）：+140.62 / 184,361 active**（**delta vs T188v2：−9.81**）

**平台**：**+34.59**（**−1.05 vs T188v2 +35.64**）。

**结论**：❌ 150+150 反而倒退；HP 多样性扩展后引入了更多 bias，抵消了 variance reduction 收益。ensemble 扩容收益非单调（5→50 获 +1.00，50→150 获 −1.05）。

#### T191: trend features pilot（IC residual 验证）
**新特征**：5 个 trend 特征（mean_logret_W10/20/50/100，mid_pct_change_W100），总维度 370 → 375（扣重复 → 364 有效维度）。  
**验证方法**：与 T188v2 的预测相关性 + IC residual（去除 T188v2 共线性后的剩余 IC）。

**结果**：
- corr(T191 pred, T188v2 pred) = **0.9941**（几乎完全共线）
- IC_residual_pooled = −0.1699（负！残差与标签负相关）
- T191 standalone holdout PnL = 140.10（vs T188v2 +150.43，低 10.33）

**结论**：❌ **KILL**。trend signal 已被现有 359 维特征完全 capture；IC residual 为负说明 trend 特征不仅冗余，还引入噪声。

#### T192: NNLS/OLS 权重学习（bounded NNLS，平台 +31.0，−4.64，overfit 确认）

**背景**：尝试在 T190 150+150 ensemble 上用 NNLS 学习 per-model 权重（代替 simple mean）。

**数据设置**：
- 权重学习集：schemeP_val.npz dates 80-95（294,720 行，**与模型训练集重叠**，in-sample）
- 评测集：schemeP_test.npz dates 96-119（442,080 行，同样 in-sample）

**关键指标**：
- T190 simple mean（baseline）：val +128.54，test +140.62
- T192 H4 bounded NNLS [1/1500, 2/150] per-group：val **+180.41**，test **+193.94**，val/test ratio = 1.075

**平台**（submission_050912_iter019_v2N_150plus150_H4weights_cputorch.zip）：**+31.0**（**−4.64 vs T188v2 SOTA +35.64**）。

**关键指标（彩灯）**：
- NN-NN 平均相关 = 0.9331（严重共线）
- LGB-LGB 平均相关 = 0.8433（严重共线）
- condition number = 54182（极病态矩阵）
- val→test ratio > 1.05 = OLS 在 in-sample 分布上 overfit 的烟雾弹信号

**结论**：❌ **OLS overfit 确认**。val 和 test 均 in-sample，OLS 学到了 val 期内 3 个特定 LGB 模型的噪声 pattern，并在同样 in-sample 的 test 期上"泛化"（实为噪声共同分量对齐）。真正的 OOD 平台上彻底失效。

**collinearity 分析**：
- OLS variant B（2 features aggregate）：w_nn = −0.98（认为 NN 是反预测的，荒谬）
- NNLS variant C（300 features）：只有 3 个 LGB 模型权重非零（memorizing 3 models' val noise）
- Lasso α=1e-5：所有权重归零（no signal robust to tiny L1）

---

### R 系列调研摘要

| 编号 | 主题 | 关键发现 | 结论 |
|---|---|---|---|
| R1/v2 | date decay + sym augmentation | date decay 对 GBDT 作用有限 | ⚠️ 边际 |
| R2 | Group DRO | fold2 改善但 fold4 牺牲；净效果负 | ❌ |
| R3 | HYD quartet（Huber/asymmetric/Y-shape/Date-decay）| Huber 略优 L2 | ⚠️ 见 T99 |
| R4 | logret lag 特征 | 无增量 | ❌ |
| R5/v2 | Bayesian optimization | 与 DE 相近 | ⚠️ |
| R10 | PnL-aware loss 10 方案调研 | SPO+ 最稳，端到端 PnL 梯度不收敛 | ✅ 确认 SPO+ |
| R11 | cross-stock OOD robustness 文献调研 | sym-agnostic 设计合理 | ✅ |
| R12 | ensemble & stacking 调研 | 12 方案 + 6 paper；OOF 过拟合风险高 | ✅ |
| R13 | probability calibration + position sizing | isotonic/platt 均负 | ❌ |
| R30 | 数据特征深度调研 | 解释 h_60 有效、h_5/10/20 失效原因 | ✅ |
| R31 | 2024-2026 HFT/LOB SOTA 文献调研 | 9 篇深度笔记；TransLOB/TabLOB/N-BEATS | ✅ |
| R34 Stage 1-5 | sym-invariant 特征建设 | 154 → 359 维 SchemeP | ✅ 核心贡献 |
| R_50seed_bag | 50-seed bagging 前期研究 | 为 T188v2 奠基 | ✅ |
| R_Hawkes_OFI | Hawkes 过程订单流特征 | 边际，未进入最终包 | ⚠️ |
| R_NaN1/2 | LGB/CB NaN passthrough 审计 | NaN 处理合规 | ✅ |
| R_Pairwise_ALL | 特征 pairwise 交互 | 计算量大，边际 | ⚠️ |
| R_conformal_select | per-sym conformal beta 搜索 | 确定 {0.10, 0.40, 0.30, 0.00, 0.00} | ✅ 核心 |
| R_cross_sym | 跨 sym 特征 | **ABORT**：违反平台合约（批次打乱） | ❌ |
| R_full_retrain | M7 full-retrain 可行性研究 | 确认安全可行 | ✅ 核心 |
| R_hawkes/r6 | Hawkes 进阶 | 边际 | ⚠️ |
| R_multinn_spo | 多 NN SPO+ 组合研究 | 多样性需 HP 差异而非 seed 差异 | ✅ |
| R_multitick_window | 多 tick 窗口特征 | 在 Stage 5 中部分进入 SchemeP | ⚠️ |
| R_roll_tsrv | Roll effective spread + TSRV | 有价值但未成为主力特征 | ⚠️ |
| R_stack3_compound | 3 层 stacking | OOF 过拟合，放弃 | ❌ |
| R_time_of_day | 盘口时段特征研究 | LOSO 小幅正，最终未进入 T170 | ⚠️ |
| R_T75L2_* | T75 LGB L2 系列变体 | advval/bayes/pairwise/symaug 均边际 | ⚠️ |

---

## 4. 关键发现总结（定量）

### 发现 1：本地 holdout 的 in-sample 污染问题

T81/T87/T170 NN 训练在 date 0-119（含测试集 96-119），holdout（96-119）评分是 in-sample，不是真正 OOD 性能参考。具体数字：

- T180 OOD abstain：本地 holdout −78 vs T170（严重退步）→ 平台未提交（但测试集比训练更 in-distribution，abstain 会损失 PnL）
- T182 无 conformal：本地 holdout **+6.62**（看起来好）→ 平台 **−0.30**（conformal 实际有用）
- T192 bounded NNLS：本地 holdout **+53 vs T190**（val/test ratio 1.075）→ 平台 **−4.64 vs T188v2**

**结论**：本地 holdout 数字的绝对值不可信；只有相对比较（已证架构间的 delta）有一定参考价值；平台 OOD 是唯一 ground truth。

### 发现 2：Ensemble 扩容的收益递减与非单调性

| 扩容步骤 | 本地 holdout delta | 平台 delta |
|---|---|---|
| 5+5 → 40+5（T179，只扩 NN） | +0.63 | −0.25 |
| 5+5 → 50+50（T188v2，NN+LGB 都扩） | +0.48 | **+1.00** |
| 50+50 → 150+150（T190，继续扩） | −9.81 | **−1.05** |

**关键洞察**：
1. 单纯扩 NN（T179）无平台收益；NN+LGB 两侧都扩才有 variance reduction 效果
2. 5→50 平台 +1.00（有效）；50→150 平台 −1.05（反效果）
3. 本地 holdout 方向与平台方向可以完全相反（T179：本地 +0.63 → 平台 −0.25）

### 发现 3：M7 全数据重训是"已验证模型"专属

| 模型 | M7 本地 holdout 变化 | M7 平台变化 |
|---|---|---|
| T75 LGB（已验证） | LOSO-equiv 不变 | **+5.51**（iter_018 → iter_019 v2） |
| T87 NN（已验证） | in-sample +8.49（污染） | **+0.20**（T170 vs v2） |
| T183 NN（random init，新） | in-sample +0.63 | 未提交（预估边际） |
| T172 Transformer（新架构） | in-sample +55（严重污染） | 未提交（OOD 风险高） |

**结论**：已验证架构（经过 V4 walk-forward 调优）做 M7 retrain 安全有效；新引入架构做 M7 retrain 危险（过拟合 96-119 分布），不提交。

### 发现 4：OOF threshold DE 过拟合风险

| 实验 | 本地 OOF 提升 | 平台变化 |
|---|---|---|
| iter_013 DE 4D（vs symmetric k） | +36.23 vs +33.72（+2.51） | 透传系数 53%（vs 预期 70%），gap 更大 |
| iter_016 4-way DE（vs iter_015） | +43.71 vs +40.09（+3.62） | 平台 **−3.08** |
| T192 bounded NNLS（val→test） | val/test ratio = 1.075 | 平台 **−4.64** |

**结论**：本地大幅提升若来自 OOF/val 上的阈值/权重 DE 优化，平台大概率翻面。更复杂的集成 + DE，过拟合风险越高。

### 发现 5：短 horizon 平台完全不可信

| Horizon | LOSO 本地 | 平台 |
|---|---|---|
| h_10 | +21.86（最强！） | **−8.64** |
| h_5 | +17.49 | **−7.68** |
| h_20 | +19.70 | **−5.09** |
| h_40 | +11.71 | **+2.02** |
| h_60 | +6.30 | **+4.07** |

α=0.05%（h_5/10）手续费（0.02% 双边）摧毁所有 alpha。**整个项目专注 h_60**（α=0.1%），短 horizon 彻底放弃。

### 发现 6：SPO+ DFL 是最优 NN 训练协议

| 训练目标 | 本地 LOSO-equiv | 平台 PnL |
|---|---|---|
| 3-class CE（DeepLOB） | ~+8（T1） | −6.65（平台） |
| L2 regression（T81 warm-start only） | +35.89 | 未提交 |
| **SPO+ DFL fine-tune**（T87） | **+40.09** | **+28.16** |
| MAE LGB（T137） | L2 +0.3-0.5（边际） | 未确认 |
| Direct PnL loss（T13） | 不稳定 | 未确认 |

SPO+ 的优势：在已对齐 L2 表征的基础上微调，避免端到端 PnL 梯度的不稳定性，同时把推理决策损失直接对齐。

### 发现 7：per-sym beta conformal wrapper 的平台效果

| 实验 | 无 conformal 平台 | 有 conformal 平台 | delta |
|---|---|---|---|
| iter_015 vs iter_018 | +28.16 | **+28.93** | **+0.77** |
| T170 vs T182 | +34.34（T182，无 conformal） | +34.64（T170，有 conformal） | **+0.30** |

**conformal 在平台上持续有效**（+0.3-0.8），即使本地 holdout 显示 conformal hurt（in-sample artifact）。

### 发现 8：corr(NN, NN) 高共线性阻止 OLS 学权重

T192 collinearity 数据：
- NN-NN 平均相关 = **0.9331**（150 个 NN 预测之间几乎完全共线）
- LGB-LGB 平均相关 = **0.8433**（50/150 个 LGB 预测共线性高）
- condition number = **54,182**（极病态，OLS 解不稳定）

这解释了为什么任何 per-model 权重学习（OLS/NNLS/Lasso）均失败：共线性消除了权重的可识别性，任何"最优"权重都只是拟合 val 期的噪声。

---

## 5. 失败方向归档（含数字）

| 方向 | 实验 | 本地 | 平台 | 死因 |
|---|---|---|---|---|
| **短 horizon 优化** | iter_003/004，多次 | LOSO h_10=+21.86 | −8.64 | α=0.05% 被手续费摧毁 |
| **4-way ensemble OOF 过拟合** | iter_016 v3 | LOSO +43.71（+3.62 vs iter_015） | +25.08（**−3.08**） | OOF DE 4D 过拟合 442k 本地测试 |
| **额外相似 NN 集成** | T163 v2+T97 | holdout +42.90（+1.41） | ~+30.74（**−3.70**） | T97/T87 架构相同，diversity=0 |
| **GRU 架构** | T95, iter_016 | LOSO 微小提升 | iter_016 失败主因之一 | hidden state 违反硬约束 #2 精神 |
| **TTT/TTA 系列** | T161/162/168/169 | holdout 全退步 | 未提交 | 平台批次打乱，per-batch 统计不稳 |
| **OOD 密度抽象（GMM）** | T180 | holdout −32 to −78 | 未提交 | 测试集比训练更 in-distribution；低密度点有高 PnL |
| **时间窗口弃权** | T181 | holdout −0.27 to −23.39 | 未提交 | T170 conformal 已过滤低置信；弃权损失信号 |
| **GroupTransformer** | T172/T187 | in-sample +22.2/+14.7（污染） | 未提交 | 全 in-sample，新架构 M7 retrain 高风险 |
| **150+150 mega** | T190 | holdout −9.81（vs 50+50） | +34.59（**−1.05**） | HP 多样性反成噪音；收益递减 |
| **OLS 权重学习** | T192 | holdout +53（val/test ratio 1.075） | +31.0（**−4.64**） | val+test 均 in-sample；OLS 拟合噪声 |
| **Trend features** | T191 | holdout −10（vs T188v2） | 未提交 | corr(T191, T188v2)=0.9941，完全冗余 |
| **CrossEntropy 目标** | T1, DeepLOB | LOSO ~+8 | −6.65 | 掩盖幅度信息；决策层无法利用 EV |
| **数据增强 mixup** | T138 | LOSO 下降 | 未提交 | 混合样本破坏 EV 信号 |
| **概率校准** | T15, T131, T40 | LOSO 下降或持平 | 负面 | 校准破坏排序质量 |
| **跨 sym 特征** | R_cross_sym | — | **ABORT** | 平台批次打乱，多 sym 无法同时进同一 batch |
| **40-NN 扩容（只扩 NN）** | T179 | holdout +0.63（in-sample） | **−0.25** | 5-seed 已在 variance floor；只扩 NN 无用 |
| **deep CNN** | T184/T185 | in-sample 看起来好 | 未提交 | OOD 验证缺失，风险高 |
| **各种 PnL-aware loss** | T13/T57/T134 | 不稳定 | 未提交 | 梯度不稳，不收敛 |
| **per-sym OOF threshold** | T126 | LOSO 本地高 | 未提交 | 过拟合每个 sym 的 OOF 分布 |
| **Alpha101/Alpha191** | T22 | IC +21.80 | 未提交 | 需跨截面数据，单只股票实现困难 |

---

## 6. 提速优化 timeline

| 实验 | 优化对象 | 时间 before → after | 加速比 | 影响 |
|---|---|---|---|---|
| T61（iter_010） | feature 提取 batch-vec | 254 min → 4.5 min（对 442k 行） | **58×** | 首次通过平台 3h 限制 |
| T188v3 NN-only | 50 NN sequential numpy → batched torch CUDA bmm | 1509ms → 8.1ms（per 1024 batch，stated baseline） | **186×** | NN 推理不再是瓶颈 |
| T188v3 NN-only（实测） | 554ms → 8.1ms | — | **68.5×** | 同上 |
| T188v3 e2e | feature(549ms)+LGB(84ms)+NN(1509ms)=2142ms → 641ms | — | **3.3×** | 平台预计 10-15 min 可完成 442k |
| T188v3_fullhorizon | 5 个 horizon 共享一套 h=60 ensemble | 单 horizon 时间 × 5 → 单 horizon 时间 | **5× horizons** | 5 head 免费提交 |
| T189 GPU pkg | T170 5-seed CPU → torch GPU | 855ms → 更快 | — | 平台 GPU 环境加速 |

---

## 7. 本地 eval 方式演进

> **重要**：不同阶段的本地 PnL 数字因 eval 方式不同，**不可直接比较**。

| 阶段 | 时间 | eval 方式 | 范围 | 特点 |
|---|---|---|---|---|
| Phase 0-1 | T1-T23 | **LOSO 5-fold**（留一 sym） | 全 5 sym，per-sym 留一 | 过度惩罚 sym OOD；短 horizon gap −30.5 |
| Phase 0-1 | T1-T23 | IID split | 同 sym 内划分 | 严重乐观；+19.25 IID → −22.10 LOSO |
| Phase 2 | T44-T70 | **LOSO-equiv**（5 sym 全训 + DE on 442k） | 全 5 sym，day 0-119 为训练 | 去掉 LOSO 过度惩罚；方法论变化贡献约 +7 PnL |
| Phase 3-5 | T75-T155 | **LOSO-equiv + V4 walk-forward val** | train 0-79, val 80-95, test 96-119（OOF） | 当前最标准方法；LOSO-equiv ~40 ≈ 平台 28-35 |
| Phase 6-7 | T156-T192 | **holdout in-sample**（date 96-119） | 全数据 0-119 训练 → 96-119 评分 | **in-sample 污染**（NN/LGB 训练包含 test 日期）；绝对值不可信，只看相对 delta |

**关键数字对应关系（Phase 3-5 LOSO-equiv 到平台的 mapping）**：
- iter_013：LOSO-equiv +36.23 → 平台 +19.23（透传 53%）
- iter_015：LOSO-equiv +40.09 → 平台 +28.16（透传 71%）
- iter_018：LOSO-equiv +41.49 → 平台 +28.93（透传 70%）

**关键数字对应关系（Phase 7 holdout in-sample 到平台）**：
- T170：holdout +149.95（in-sample）→ 平台 +34.64
- T188v2：holdout +150.43（in-sample）→ 平台 +35.64
- T190：holdout +140.62（in-sample）→ 平台 +34.59
- T192 H4：holdout +193.94（in-sample）→ 平台 +31.0（overfit）

---

## 8. 现状 + 后续方向

### 当前状态（2026-05-12）

| 指标 | 数值 |
|---|---|
| 当前 SOTA | T188v2 50+50 = **+35.64**（平台，h_60） |
| 初赛截止 | 2026-05-11（已截止） |
| 初赛成绩公布 | 2026-05-18 |
| 决赛答辩 | 2026-05-30 |

### 已 Kill 方向（确认无效，不再重试）

1. **OLS/NNLS per-model 权重**（T192）：val→test ratio>1.05 是 overfit 的烟雾弹；val 和 test 均 in-sample 时无效
2. **trend features**（T191）：corr=0.9941，完全冗余
3. **150+150 扩容**（T190）：收益非单调，−1.05 平台
4. **TTT/TTA 系列**（T161/162/168/169）：平台批次打乱是根本障碍
5. **OOD 密度抽象**（T180）：测试集不是 OOD，弃权损失信号
6. **短 horizon 优化**（iter_003/004）：α 太薄，手续费摧毁 alpha
7. **GroupTransformer**（T172/T187）：全 in-sample 污染，新架构 M7 retrain 高风险

### 可探索方向（如有后续实验窗口）

1. **Walk-forward val（val 跨训练边界）**：真正 OOD 的 per-model 权重学习需要 train 0-95, val 96-119, test 120+（但需要额外数据或重新定义范围）
2. **NN 架构多样性提升**：用完全不同的 NN 架构（如 CNN、GRU 改进版）替代第二组 NN，避免 T163 类型失败
3. **conformal beta 重调**：50+50 ensemble 后 variance 变小，optimal beta 可能调低（减少 abstain），T150 风格重调
4. **真正的 OOD ensemble 剪枝**：只保留在跨时间段（不同月份）表现稳定的模型

---

## Appendix A. 主要提交 zip 文件清单

| zip 文件名 | 日期 | 平台 PnL | 备注 |
|---|---|---|---|
| `submission_050601_iter0.zip` | 2026-05-01 | −6.65 | mmpc_demo baseline |
| `submission_050606_iter002.zip` | 2026-05-06 | +4.07 | SchemeC LGB h_60 |
| `submission_050707_iter013.zip` | ~2026-07-07 | +19.23 | 首个 Δmid regression |
| `submission_050802_iter015.zip` | 2026-08-02 | +28.16 | T87 SPO+ NN + T75 LGB |
| `submission_050815_iter016_v3.zip` | 2026-08-15 | +25.08 | 4-way ensemble 失败 |
| `submission_050817_iter018_v1.zip` | 2026-08-17 | +28.93 | + per-sym conformal |
| `submission_050818_iter019_v2_fullretrain_conformal.zip` | 2026-08-18 | +34.44 | M7 LGB full-retrain |
| `submission_050909_iter019_v2NN_T97_agreement_combined.zip` | 2026-09-09 | ~+30.74 | T163 失败 |
| `submission_050909_iter019_v2N_T87M7.zip` | 2026-09-09 | +34.64 | T170 NN M7 SOTA |
| `submission_050909_iter019_v2N_T87M7_NoConformal.zip` | 2026-09-09 | +34.34 | T182 无 conformal |
| `submission_050910_iter019_v2N_40NN_bag.zip` | 2026-09-10 | +34.39 | T179 40-NN |
| `submission_050911_iter019_v2N_50plus50_optimized.zip` | 2026-09-11 | **+35.64** | **T188v2 50+50 SOTA**；MD5=95993e6250ea56361dfecb62c9123e63 |
| `submission_050911_iter019_v2N_50plus50_cputorch.zip` | 2026-09-11 | +35.64 | CPU-only 备用；MD5=c67e4d25665a2a7a12f4f6839cc8d0fe |
| `submission_050912_iter019_v2N_150plus150_H4weights_cputorch.zip` | 2026-05-12 | +31.0 | T192 NNLS overfit |

---

## Appendix B. 实验编号速查（T1–T192）

> 格式：**T编号** — 核心内容（关键产出）

**T1–T10（基础建设）**  
T1: NN baseline (3-class CE, h_60)；T2: LGB baseline (SchemeB, LOSO −22.10)；T3: 特征 v1 (+72 维 MLOFI+WMP+RV+EWMA)；T4: LOSO 5-fold + threshold gate (iter_001c)；T5a/b: threshold sweep/SchemeC (LOSO h_10=+21.86)；T6/T6b: 5-seed SchemeA ensemble (edge gain 无)；T7: window z-score SchemeD1 (LOSO +13.91)；T8: sym=2 brittleness 诊断；T9: SchemeE (z-score)；T10: SchemeF (-v SchemeC)

**T11–T35（LOSO 与特征扩展）**  
T11: SchemeC 5-seed h_10 (+0.36)；T12: multi-horizon meta (-v base)；T13: PnL-aware NN loss (不稳定)；T14: cross-sym mixup (负面)；T15: isotonic+EV gate (−1.02)；T17: CatBoost (持平)；T18: XGBoost (持平)；T19: NN regularized (持平)；T20/T21: SchemeH 291维 (+21.70)；T22: Alpha101/191 (IC +21.80，不入 iter)；T23: CV calib + 平台校准 mapping；T24: iter_004a/b 短 h 阈值；T25-T28: aug_a (LOSO h_60 +11.46)；T29-T34: 调参/CV 建设；T35: ReVol+SG (持平)

**T44–T74（R34 Stage 1-5 特征建设）**  
T44: Stage 1 +54 维 (LOSO +12.09)；T51/b: Stage 2 +59 维 (5-seed +15.06)；T53: Stage 3 +14 维 (+16.84)；T55: Stage 4；T58: walk-forward CV 验证；T59: 全 sym 训练 (LOSO-equiv +24.52)；T61: batch-vec 58× 加速；T62: 无偏阈值设计；T63: regression 参数扫描；T64: V4 walk-forward val (+26.44)；T65: pseudo labeling (负)；T66: adversarial val；T67: multi-split bagging；T68: Stage 5 +20 维 (SchemeP 359 维完成)；T69-T70: iter_012 打包；T71: h_40 Stage 5；T72: binary cascade (负)；T73: seed subset；T74: time features SchemeQ

**T75–T92（Regression 突破）**  
T75: LGB regression Δmid + EV gate (LOSO-equiv +36.23, 平台 +19.23)；T76: decoupled threshold；T77: multi-horizon vote (+5.98)；T78: regression+SchemeQ (−0.72)；T80: LOSO→平台校准 (0.70x)；T81: NN L2 regression (+35.89)；T82-85: 校准审计；T86: quantile+dual-gate (+40.22)；T87: SPO+ DFL NN (LOSO +40.09, 平台 +28.16)；T89: CatBoost L2 5-seed (+40.13)；T91: ReVol+SG revival (noise)；T92: magnitude sample weight (falsified)

**T95–T109（iter_015-016 深化）**  
T95: GRU regression；T97: 第二组 NN (T87 类似)；T99: Huber LGB (+44.72, OOF 污染)；T106-T109: iter_015/016 打包系列

**T115–T155（M7 全数据重训系列）**  
T115: scale invariance audit；T116: GMADL loss；T117: monotone constraints；T118: decision 4-in-1；T119-T120: iter_016 ablation；T122: T75 retest；T123: iter_017 v1；T124: R2/R4 eval；T125: CB addon；T126: per-sym OOF thresh；T127: iter_018 v1 conformal (LOSO +41.49, 平台 +28.93)；T128: conformal v2 + M7 可行性；T129: iter_019 template；T131: isotonic；T132: quantile LGB；T133: alt booster；T134: PnL NN loss；T135: snapshot NN；T136: SPO+ tune；T137: MAE beats L2 (+0.3-0.5)；T138: mixup (负)；T139: stacker NN (负)；T140/b/c: iter_019 v1/v2 M7 LGB (平台 +34.44)；T141/b: Hawkes；T142: Hawkes full-retrain；T143: iter_019 v3 MAE (+41.20)；T144: diversity eval；T145/b: iter_019 v4 MAE full-retrain；T146: conformal DE re-thresh (+43.96)；T147: feature pruning；T148/b/c: LGB HP sweep；T149: NN OOD sweep；T150: conformal beta sweep；T151/b: TOD v9；T152: adversarial 259维 (v10)；T153-154: v9b/v10b/v11

**T155–T169（NN 扩展 + Wrapper 实验）**  
T155: v2 replica (deterministic confirmed)；T156/b/c: NN swap T97 (+1.66 holdout)；T157: transformer preliminary；T158: agreement filter；T159/b: per-sym abstain (null)；T160: v2W packaging；T161: TTA K=5 (负)；T162: in-row TTT (W3a −0.30, W3b −1.98, W3c −3.11)；T163: v2+T97+agree (holdout +1.41, 平台 −3.70)；T164: 4-way ensemble (+0.34)；T165: magnitude filter；T166: CatBoost M7 5-seed (+0.39)；T167: 5-way ensemble (+0.11)；T168: adaptive thresh V1-V4 (全负)；T169: AR TTT (负)

**T170–T192（M7 NN + 大规模 Ensemble）**  
T170: NN M7 full-retrain (in-sample +8.49, 平台 +34.64 SOTA)；T171: LGB HP sweep；T172/b: GroupTransformer (in-sample +55, 未提交)；T173/174: multi-horizon T170NN；T175: transformer packaging；T176/b: T87M7 HP sweep (S1_ep15 平台 +34.51)；T177: wrapper tricks B/C；T178: LGB 10-seed (null)；T179: 40-NN (in-sample +0.63, 平台 −0.25)；T180: OOD GMM abstain (−32~−78, 全负)；T181: time-window abstain (−0.27~−23.4)；T182: no-conformal (in-sample +6.62, 平台 −0.30)；T183: NN random init M7 (+0.63, 与 T170 持平)；T184: deep CNN raw；T185: hybrid CNN；T186: architecture_doesnt_fit；T187: GroupTr replace+add (in-sample +22.2/+14.7, 未提交)；**T188/v2/v3: 50+50 mega ensemble (平台 +35.64 SOTA)**；T188v3_fullhorizon: 5 horizon share h60 ensemble；T189: T170 GPU；**T190: 150+150 (平台 −1.05)**；**T191: trend features (corr=0.9941, KILL)**；**T192: bounded NNLS (val/test ratio 1.075, 平台 −4.64, overfit)**

---

## Appendix C. R 系列调研速查

R1/v2: date decay + sym aug；R2: Group DRO (负面)；R3: HYD quartet；R4: logret lag (无效)；R5/v2: Bayesian optim；R10: PnL-aware loss 调研（10 方案）；R11: cross-stock OOD 文献；R12: ensemble & stacking 调研；R13: calibration + position sizing；R30: 数据特征深度调研（h_60 有效原因）；R31: 2024-2026 HFT/LOB SOTA 文献（9 篇笔记）；R34 Stage 1-5: sym-invariant 特征建设 (154 → 359 维)；R_50seed_bag: 50-seed bagging 前期研究；R_Hawkes_OFI: Hawkes 过程订单流特征；R_NaN1/2: NaN passthrough 审计；R_Pairwise_ALL: 特征 pairwise 交互；R_conformal_select: per-sym beta 搜索（核心）；R_cross_sym: 跨 sym 特征 ABORT；R_full_retrain: M7 可行性研究；R_hawkes/r6: Hawkes 进阶；R_multinn_spo: 多 NN SPO+ 组合；R_multitick_window: 多 tick 窗口；R_roll_tsrv: Roll effective spread + TSRV；R_stack3_compound: 3 层 stacking (负面)；R_time_of_day: 盘口时段特征；R_T75L2_*: T75 LGB L2 系列变体（advval/bayes/pairwise/symaug）

---

*报告结束。数据来源：PROGRESS.md（258 行）、git log（130+ commits）、experiments/T\*/results.json（114 个文件）、SUBMISSION_LOG.md、report.md（41KB，2026-05-11）。*
