# AUDIT-ITER14-MISSED-TRICKS — 完整审计报告

**审计日期**: 2026-05-08  
**审计对象**: submission_050800_iter014_robust_thresh.zip (LOSO-equiv +36.22)  
**iter_014 实际组成**: 5-seed LGB regression Δmid_norm, h=60 only active, T83 kfold5_date thresh (thr_up=3.715e-4, thr_dn=1.660e-4), 354 特征 (359-d Stage 5 minus 11 KS-fail), NO NN, NO multi-horizon active

---

## 第 1 节：Missed-Trick 全清单

> **说明**：iter_014 baseline LOSO-equiv = **36.22**（same regression model as iter_013，仅 T83 改了 threshold，LOSO diff = +0.07，在噪声内）。
> 
> DE contamination 标注：所有"reported LOSO"数字都用 full-fit DE threshold（在 442k test 上调参），属于轻度污染。真实平台 lift = LOSO delta × 0.70（T80 校准的透传系数）。

| Rank | T# | Trick 名称 | Reported LOSO lift vs iter_014 | Val/unbiased estimate | 与 iter_14 兼容性 | iter_14 缺它的原因 | 整合成本 |
|------|-----|-----------|-------------------------------|----------------------|----------------|------------------|---------|
| 1 | **T81** | **NN regression + T75 LGB ensemble** | **+2.06** (+38.28 vs +36.22)¹ | NN alone = +35.89 (comparable to LGB +36.23); cross-corr 0.7749 → real diversity | ✅ 完全兼容 (h=60, EV gate, same features) | iter_014 选了更小/无 torch 依赖的 T83-only 包 | 中：训新 5 个 NN seed，Predictor.py 添加 NN 分支 |
| 2 | **T87** | **SPO+ DFL NN + T75 LGB** | **+3.87** (+40.09 vs +36.22)² | 单 seed SPO+ vs T81 delta: +0.52～+5.87; cross-corr T87↔T75=0.71 (更多样 vs T81 0.77) | ✅ 完全兼容 | T87 在 iter_014 决策之后完成；需要 T81 为前提 | 中高：T87 模型已存在 (5 NPZ)，需先上 T81 框架 |
| 3 | **T86** | **Quantile IQR midpoint (q_mid) + T75 LGB** | **+2.27** (+38.49 vs +36.22, w_t81=0)³ | q_mid alone ≈ +36.18 (≈T75)；ensemble diversity corr(q_mid,T75)=0.85 → 有真实信号 | ✅ 完全兼容 | T86 在 iter_014 之后完成 | **低**：q30/q70 模型已全部训好 (10 个 LGB txt)！只需改 Predictor.py |
| 4 | **T95** | **GRU 3-way ensemble (GRU+T81+T75)** | **+3.56** (+39.78 vs +36.22)⁴ | GRU alone = +9.06; cross-corr GRU↔T75=0.35 (高多样性)；1-seed + 4 seeds 待补 | ⚠️ 兼容但复杂 (需 per-tick raw LOB 窗口处理) | Post-iter_014；GRU packaging 复杂，只有 1 seed 完整 | 高：需补 4 个 GRU seed，Predictor.py 需重构 raw window 分支 |
| 5 | **T77v0** | **Multi-horizon raw vote (K=4, unbiased)** | val K4 = 7.82 vs iter_012 baseline val = 8.08 = **-0.26** on val | LOSO +28.42 < iter_013 +36.23 → **SUPERSEDED** | — | **已被 iter_013 超越**；T77v0 unbiased LOSO 低于 iter_013 | N/A |

---

### 脚注（Contamination 标注）

¹ **T81**: Reported LOSO +38.28 用 DE 调了 threshold (thr_up=4.21e-4, thr_dn=1.86e-4) on 442k. NN alone = +35.89，LGB alone = +36.23，ensemble diversity 来自 cross-corr 0.77，真实信号。估计无偏 LOSO lift = +1.0～+1.5，对应平台 +0.7～+1.1。

² **T87**: Reported LOSO +40.09 = T87 NN (5-seed) + T75 LGB，DE threshold on 442k. SPO+ per-seed 改进是真实的 (T81 seed1=26.48 → T87 seed1=32.34, +5.87)；cross-corr 0.71 比 T81 更多样。估计无偏 LOSO lift over actual iter_014 = +1.5～+2.5，对应平台 +1.0～+1.75。

³ **T86**: Reported LOSO +38.49 (q_mid + T75, no NN) 用 weight grid search on 442k，权重本身是 contaminated。但 q_mid alone = +36.18 (≈T75)，两者 corr=0.85。估计真实无偏 lift over actual iter_014 = +0.5～+1.5，平台 +0.4～+1.0。  
   - 注：含 T81 的最优三路 (w_q=2, w_t75=0.5, w_t81=1) = +40.22，也是 contaminated。

⁴ **T95**: Reported LOSO +39.78 = 3-way (GRU×0.5 + T81×1.0 + T75×1.5)，DE weights on 442k. GRU alone = +9.06（弱，但 corr=0.35 极低）。估计真实无偏 lift over actual iter_014 = +0.8～+1.5，平台 +0.6～+1.1。

---

### 关于 T77（已知 "missing" 案例的更正）

任务说 T77v2 val sanity +3.72 over iter_012，故 T77 被视为 missed trick。但经仔细审计：

| 指标 | 数值 | 含义 |
|------|------|------|
| T77 v0 K4 unbiased LOSO | +28.42 | 低于 iter_013 +36.23 |
| T77 v0 K4 val (date 76-79) | 7.82 | 低于 iter_012 baseline val 8.08 |
| T77 v2 DE LOSO | +45.02 | 6-dim DE 严重过拟合 |
| T77 v2 val sanity (+3.72) | 对 iter_012 基线 | iter_013 val 未测，但 regression 应远高于此 |

**结论**: T77 as-is 被 iter_013 超越（unbiased LOSO +28.42 < +36.23）。"Multi-horizon regression 投票" 的概念（训 h5/10/20/40 regression 模型）尚未测试，但那是新 T 实验，不是 proven missed trick。

---

## 第 2 节：Top 5 应优先整合的 Trick

### #1：T81 — NN Regression + LGB Ensemble（基础前提）

**描述**: 与 T75 LGB 完全对称的 MLP 回归 [359→256→128→64→1]，LayerNorm+GELU+Dropout0.10，134k params，L2 loss on Δmid_norm，相同 EV gate 结构。5-seed ensemble NN (w=1) + 5-seed LGB (w=1.5) average，再统一调 threshold。

**来源实验**: T81_nn_regression_pnl

**兼容性证据**:
- 同 feature pipeline，同 h=60，同 EV gate 结构 ✅
- Stateless predict ✅，sym-agnostic ✅，W≤100 ✅
- 预测一致性：inference 0.66s / 1024-batch ✅

**整合 plan**:
- 需要训练 5 个新 NN seed（约 60s/seed on GPU）
- 或直接使用 T87 的 NPZ 文件（5 个 seed 已存在 `experiments/T87_spo_dfl/model_T87_seed*.npz`）但那是 SPO+ 版本
- Predictor.py 需添加 NN 推理分支：load .npz → numpy matmul+layernorm → ensemble average
- 更新 thresholds.json 含 ensemble 权重和新 thr_up/thr_dn

**预期 LOSO + 平台 lift**:
- 保守 LOSO: +1.0～+1.5（无偏 cross-corr 多样性效益）
- 对应平台: **+0.7～+1.1**（基于 0.70 透传系数）

**风险**: NN 推理需 numpy-only 路径（无 torch 依赖时）；T87 NPZ 已做好 extract_nn_npz.py 参考。

---

### #2：T87 — SPO+ DFL NN（更优质的 NN 权重）

**描述**: 以 T81 NN 权重为 warm start，再用 SPO+ 决策聚焦学习（λ=30, lr=3e-5, 15 epochs），替代标准 L2 回归目标。核心改进：SPO+ surrogate 在 EV gate threshold 附近提供恒定次梯度（~2），修复了 T60/T81-B 的梯度塌缩问题。

**来源实验**: T87_spo_dfl，models: `experiments/T87_spo_dfl/model_T87_seed*.npz`

**兼容性证据**:
- 同 feature pipeline，同 h=60，同 EV gate ✅
- T87 NPZ 已通过 end_to_end_check.py 验证 ✅
- cross-corr T87↔T75 = 0.71（比 T81 的 0.77 更多样）

**整合 plan**:
- 最快路径：直接用 T87 NPZ 替换 T81 NN（模型文件已存在！）
- Predictor.py：同 T81 整合路径，只是换 .npz 文件
- ensemble: T87 NN (w=1) + T75 LGB (w=1.5) + 新调 threshold（可用 T83 kfold5_date 方法）

**预期 LOSO + 平台 lift**（vs actual iter_014 +36.22）:
- 估计无偏 LOSO: +1.5～+2.5
- 对应平台: **+1.0～+1.75**

**风险**: SPO+ fine-tune 用了 T81 warm start（若 T81 未先训练，需先训 T81 或直接用 T87 的）。T87 模型已存在，直接打包是最快路径。

---

### #3：T86 — Quantile IQR Midpoint + T75 LGB（零额外训练成本！）

**描述**: 用 LightGBM quantile regression (alpha=0.3 和 0.7) 训两套模型，取 q30/q70 均值作为 q_mid（IQR 中点，是 Δmid 的保守点估计）。将 q_mid 作为第 3 个预测器加入 ensemble：`pred = 2·w_q·q_mid + 0.5·w_t75·lgb + 1.0·w_t81·nn`（最优权重）。

**来源实验**: T86_quantile_dual_gate，所有 10 个 quantile 模型已在 `experiments/T86_quantile_dual_gate/model_T86_q{030,070}_seed*.txt`

**兼容性证据**:
- Quantile 模型同 359-d Stage 5 features，h=60 ✅
- q_mid alone = +36.18（≈ T75 +36.23，独立信号）
- corr(q30, q70) = -0.22（两分位真实学到不同信息）
- Stateless ✅，sym-agnostic ✅

**整合 plan（无需新训练）**:
- 不含 T81 的最优权重：w_q=1.0, w_t75=1.0, w_t81=0 → LOSO +38.49（DE 污染）
- Predictor.py 添加 q_mid 推理：load 10 个 quantile 模型，average q30+q70，与 T75 LGB 加权均值
- thresholds.json 更新为新的 thr_up/thr_dn（T86 结果：0.000321 / 0.000239 at best 3-way；2-way 约 0.000325）
- ZIP 大小约 +3MB（10 个额外 LGB 模型文件）

**预期 LOSO + 平台 lift**（vs actual iter_014 +36.22, no NN path）:
- 估计无偏 LOSO: +0.5～+1.5
- 对应平台: **+0.4～+1.0**

**风险**: 权重 (w_q=1.0, w_t75=1.0) 是在 442k test 上 grid-searched，有 DE 污染。建议用对称权重 (w_q=1, w_t75=1) 而非最优 grid，更保守但可信。对称权重下 LOSO ≈ +38.49（等价，因为 q_mid 与 T75 高度相关，权重比对结果影响小）。

---

### #4：T87（完整路径）或 T86 + T81 组合（推荐 iter_015 路线）

**描述**: 将 T87 NPZ + T86 q_mid + T75 LGB 三路合并，复现 T86 三路最优 (+40.22 LOSO)。

**整合 plan**:
1. Load T87 NPZ 5 seeds → NN average (等同于 T81 NN 的增强版)
2. Load T86 q030/q070 5 seeds × 2 → q_mid average
3. Load T75 LGB 5 seeds → LGB average
4. Weighted sum: 2.0 × q_mid + 0.5 × LGB + 1.0 × NN → EV gate

**预期**:
- LOSO: ~+40.22（DE 污染），无偏估计 +2.0～+3.0 over actual iter_014
- 平台: **+1.4～+2.1**（0.70 × 2.5 估计）

---

### #5：T95 — GRU 3-way Ensemble（高多样性，长期价值高）

**描述**: 1-layer GRU hidden=64, dropout=0.10，输入 raw LOB top5 per-tick 序列 (100 ticks × ~10 columns)。Window-normalization（per-batch per-feature）是关键。GRU alone = +9.06（弱），但 cross-corr 0.35 to T75/T81（远低于 T81 的 0.77）使其成为极佳 diversifier。

**整合 plan**:
- 需补 4 个 GRU seeds（单 seed 约 60s/seed）
- Predictor.py 需新增 raw window 处理路径（不用 fast_features_batch，直接取 raw top-5 价格/量）
- Window-normalization: per-feature mean/std of 100-tick window
- GRU forward → scalar output → 加入 EV gate ensemble

**预期 LOSO + 平台 lift**:
- 估计无偏 LOSO: +0.8～+1.5 over actual iter_014
- 对应平台: **+0.6～+1.1**

**风险**:
- Predictor.py 改动最大（需维护 raw LOB 窗口 + engineered feature 窗口双路径）
- GRU 模型小 (~70KB/seed)，zip 影响极小
- 平台超时风险：GRU forward 5 seeds 需 ~0.15s/1024-batch（可接受）

---

## 附：In-progress 实验预警（结果尚未确认）

| 实验 | 单 seed 先行结果 | 初步评价 |
|------|----------------|---------|
| T88 range-vol features | 无 results.json (worker done 但数据未写) | 未知 |
| T89 CatBoost regression | seed13: test_ev_k1=32.69 (低于 T75 均值 36.23) | 初步负/持平 |
| T90 alpha regression | 仍在 training | 未知 |
| T91 revol regr revival | 仍在 training | 未知 |
| T92 magnitude weight | 5-seed DE = +32.58 ← 已 confirmed **negative** (-3.65 vs iter_013) | ❌ Negative |
| T93 nn arch transformer | 无 worker-progress | 未知 |
| T94 nn arch ResNet DCN | 仍在 training (mlp_deep step) | 未知 |
| T96 HLC range factor | seed100: test_ev_k1=32.66 (低于 T75) | 初步负/持平 |

T92 已确认 negative，不重复。其余实验结果可能于本轮 worker 完成后更新。

---

## 总结

**已证明有效但 iter_14 未用的 trick 数量**: **4 个**（T81, T87, T86, T95）  
**已知 "missed" 案例 T77 更正**: SUPERSEDED（unbiased LOSO +28.42 < iter_013 +36.23）

**最快 ROI 路径**:  
1. 直接用 T87 NPZ + T86 quantile 模型打包 → iter_015（无需 GPU！模型已存在）  
2. 用 kfold5_date 方法重新调 threshold on new ensemble  
3. 预期 LOSO ~+40.22，平台 +1.4～+2.1 over iter_014
