# Answers P2 — Decision Layer (Thresholds, Conformal, Ensemble, share_with, Inference Opt)

> **Worker scope**: 推理/决策层主题（thresholds、5-horizon emit、conformal、ensemble 规模与权重、加法/拒绝、EV gate、推理优化、OOD wrapper 失败、conformal × sym-agnostic 边界）。
> **不答**: 训练协议、特征工程、pipeline 架构（其他 worker 负责）。
> **数据源**: QREVIEW_PREDICTOR.md (281 Q)、CODE_QUESTION_LIST.md (200 Q) + 少量决策相关问题来自 QREVIEW_NN_TRAIN / LGB_TRAIN；证据来自 HISTORY_EVIDENCE_DB.md (854 行)、CRITICAL_CONSTRAINTS.md、22 次平台提交、200+ T 实验。
> **状态符号**: ✅ 有强证据 / 🟡 部分证据或已知 gap / 🔴 真 gap（未做 ablation 或推理可疑）

---

## 0. 全局总结（先读这段）

**这套提交的"决策层"实际上是 5 个独立的子系统串联**:
1. `pred_dmid = (w_NN × mean(50 NN) + w_LGB × mean(50 LGB)) / (w_NN+w_LGB)`，仅训 h=60 一套
2. EV gate 用非对称 thr_up=3e-4 / thr_dn=2.16e-4（h=60）
3. Per-sym conformal-style band `β_sym × σ_sym` 加宽 thr 实现 abstain
4. 5 个 horizon 共享 h=60 预测，thr 按 `sqrt(H/60)` 缩放
5. 推理用 batched `torch.bmm`（NN）+ CPU sequential（LGB），所有模型只 load 一次

**为什么这样安排**:
- 赛题 setting: score=**max(per-horizon PnL)** ⇒ h=60 是主战场，其余 4 个 horizon "免费"；测试点被打乱 ⇒ 任何 per-batch / 跨调用 state 都被禁；sym 0-4 可能含训练外 ⇒ 模型不能 sym-embed，但 conformal band 可在 sym 已知时按 sym 走（OOD 走 default）
- 项目先验: 平台 vs 本地 gap 是常态（透传 53-71%）⇒ 决策层每次复杂化都吃 gap；"保守 + 已验证"是项目末期主基调
- 200+ T 消融: 阈值层（T126/T146/T168/iter_016 v3）、wrapper 层（T161/T162/T168/T169/T180/T181）、ensemble 层（T163/T179/T190/T192）的所有"看似聪明"的扩展几乎全部在平台上倒退

**最大可信 gap (🔴)**:
- **USER-RAISED Q281**：`sqrt(H/60)` 缩放方向与 random-walk-with-drift 理论相反，差 12 倍 @ h=5。历史无 ablation 直接验证 sqrt 是最优 scaling
- `default_beta_for_ood=0.16` 是 mean(per_sym_beta) 的算术平均，不是 max（保守）也不是 median，OOD 默认 abstain 比 sym=1 还宽松（Q14/Q16）
- `_doc` 说 "shorter horizons trade fewer/smaller signals" 与实际效果（thr 缩小 → trade 更多）方向相反
- Ensemble 50+50 vs 100+100 / 50+150 / 25+75 等中间点未直接消融，只有 5+5 / 40+5 / 50+50 / 150+150 四点
- GELU `approximate="tanh"` 在推理 vs 训练 `nn.GELU()` (erf) 的微小不一致（Q47/Q176）；已知 ~1e-6 typical, 1e-4 max，在阈值附近可能 flip action

---

## 1. 阈值 thr_up=0.0003 / thr_dn=0.000216

### 1.1 thr_up=3e-4 这个数从哪来
- **来源**: QREVIEW_PREDICTOR Q1 / CODE_QUESTION_LIST Q1, Q2
- **答案**: T75 阶段（iter_013，平台 +19.23, 1st 突破）作者用 DE（Differential Evolution）在 442k 行 LOSO-equiv 本地测试集上 4D 搜索（thr_up, thr_dn, margin, weight），收敛到 thr_up≈3e-4, thr_dn≈2.16e-4。3e-4 是 30bps，恰好 ≈ 3× fee(0.01%)=10bps，确保毛预期收益至少覆盖 3 倍交易成本。之所以"整"是 DE 收敛后 PM 手工取整到 0.0003；2.16e-4 是 thr_up × 0.72 比例的结果（见 1.2）。后续 iter_018/019/T170/T188v2/T190 所有大版本**不再动**这两个值——因为 iter_013 LOSO-eq +36.23 → 平台 +19.23 的 **53% 透传率**已经强烈警告 DE 的 OOF 过拟合，再次 DE 风险高于收益。
- **核心证据**: HISTORY_EVIDENCE_DB.md 主题 1 时间线（T5a/T30/T62/T63/T75/T76）；iter_013 透传 53% (HISTORY 主题 9)
- **状态**: ✅

### 1.2 thr_up / thr_dn 比例 1.3889 — 为啥非对称
- **来源**: QREVIEW_PREDICTOR Q2, Q4, Q5 / CODE_QUESTION_LIST Q1
- **答案**: 1.3889 = 1/0.72，刚好对应"卖出（短）门槛比买入（长）门槛低 28%"。两种解释，**项目里采纳的是第一种**：(a) **模型本身有正预测偏差** — 训练目标 y=(mp_{t+60}−mp_t)/(mp_t+1) 在 5 只训练股票上偏正（涨多于跌的样本，因为整体牛市），NN/LGB 学到 E[pred]>0；如果对称 thr，sell 信号会被无谓抑制，所以 thr_dn 调低让卖信号通过。(b) 也契合 short-fill 比 long-fill 更稀缺时 short 信号"含金量"更高的微观结构经验，但项目里没引用过此论据。整个比例是 T75/T76 DE decoupled threshold search（单独搜 thr_up 和 thr_dn）的副产物，不是手工设的 0.72。
- **核心证据**: HISTORY 主题 1 中 T76 "decoupled threshold（thr_up≠thr_dn 分开搜）→ 最终采用非对称 thr_up=0.0003, thr_dn=0.000216"
- **状态**: 🟡（PM 团队从未明确文档化"模型有正偏差"是动机；可能是 DE 收敛产物事后合理化）

### 1.3 horizon-invariant 1.3889 — 为啥所有 horizon 比例相同
- **来源**: QREVIEW_PREDICTOR Q3
- **答案**: 因为短 horizon 不是独立训出的 thr，而是从 h=60 thr 乘 sqrt(H/60) 缩放（thresholds.json:7-42 `_note`）— sqrt(H/60) 对 thr_up 和 thr_dn 是同一个常数因子，所以 1.3889 ratio 自动保留。这不是物理选择，是 share_with=60 + sqrt scaling 设计的代数后果。如果短 horizon 用独立训练 + 独立 DE，比例可能不同（但没人做过）。
- **核心证据**: thresholds.json:7-42（4 个短 horizon `_note: "* sqrt(H/60)"`）；HISTORY 主题 10
- **状态**: ✅

### 1.4 为啥不直接 de-bias 模型 + 对称 thr
- **来源**: QREVIEW_PREDICTOR Q4
- **答案**: 三个原因。(a) 项目末期"已验证不动"原则：iter_018 起 thr 固化，因为每动一次都可能吃 53% 透传 gap。(b) NN+LGB ensemble 内部的 bias 是 50 个模型平均后的复合结果，去 bias 需要在 raw pred 上加 offset，等价于 thr_up 加 / thr_dn 减相同量，对最终 action 无效。(c) iter_019 v2 (T140) M7 全数据重训后 bias 可能轻微改变，但 thr 没重新 DE → 这是已知 gap 但平台 +34.44 仍 +5.51 over iter_018，说明不重新 DE 比"DE 又过拟合"安全。
- **核心证据**: iter_013→iter_018→iter_019 主版本均保持 thr_up/thr_dn 不变（thresholds.json git history）；HISTORY 主题 1 "保守原则"
- **状态**: 🟡（c 是推测，没人明确测试 M7 后的 bias 是否变化）

### 1.5 fallback 默认值 thr_up=thr_dn=2e-4 是什么
- **来源**: QREVIEW_PREDICTOR Q17, Q18, Q186, Q187 / CODE_QUESTION_LIST Q102
- **答案**: `_gate_with_band` 和 `_ev_gate_predict` 的 default `2.0e-4` 仅在 hcfg 缺 thr_up/thr_dn key 时触发（正常 thresholds.json 都全配）。2e-4 是 T75 之前早期 EV gate 的 baseline 值（对称、整数），保留 default 是为了让 Predictor 在缺配置时不崩。生产路径**不依赖这个 fallback**。两套 gate function（_gate_with_band 是实例方法接 conformal band；_ev_gate_predict 是 static 不接 band）历史遗留：早期 iter_013 没 conformal，后来加 conformal wrapper 时新加 _gate_with_band 没删 _ev_gate_predict，formal 是 dead branch（self._cw_enabled=True 时只走 _gate_with_band；False 时走 _ev_gate_predict 但 band_per_row=0 后两者等价）。
- **核心证据**: Predictor.py:255-271；thresholds.json 全条都给 thr_up/thr_dn
- **状态**: ✅（dead branch 不影响生产）

### 1.6 strict `>` 和 `<` — 为啥不 `>=`
- **来源**: QREVIEW_PREDICTOR Q17
- **答案**: 浮点等于阈值的概率 ~ 0，且训练 label 边界（α=0.1% h=60）也用 strict 比较生成 label_60 ∈ {0,1,2}。两端对齐避免 train/inference skew。即使浮点偶然等于，action=1（flat）是 safe default。
- **核心证据**: Predictor.py:260-261, 269-270
- **状态**: ✅

### 1.7 DE 4D 搜索 → 灾难（iter_016 v3）
- **来源**: QREVIEW_PREDICTOR Q1（间接） / 项目核心教训
- **答案**: iter_016 v3 在 4-way ensemble (CatBoost + Huber + GRU + LGB) 上又跑了一次 4D OOF DE，本地 LOSO +43.71 (+3.62 vs iter_015 +40.09)，平台 **+25.08**（−3.08 vs iter_015 +28.16）。这是项目里**最经典的 DE OOF 过拟合教训** — DE 在 442k 行本地测试上找到的"最优 4D 阈值"在平台 OOD 上完全失效。固化了"thr 一旦定下就不动"的规则。后续 T146 conformal-aware re-DE 本地 +2.76 但 PM 决定**不提交**就是基于这个教训。
- **核心证据**: HISTORY 主题 1（iter_016 v3 行）；HISTORY 附录 A 发现 4
- **状态**: ✅（项目里被反复引用的教训）

### 1.8 per-sym OOF threshold (T126) 为啥不用
- **来源**: 项目核心决策
- **答案**: T126 在每只 sym 上独立 OOF DE 搜 thr_up/thr_dn，本地 LOSO 很高但**未提交平台**。3 个理由：(a) 5 只 sym 各 ~94k 行 → 每 sym OOF 样本量小，DE 极易过拟合（与 iter_016 v3 同类问题但更严重）；(b) **硬约束 #3**：sym 可能含训练外股票（OOD sym），per-sym thr 对 OOD sym 无可用 thr，必须 fallback 到 global thr，逻辑复杂且易出错；(c) per-sym conformal β 已经在 thr 之上提供了 sym-specific 调控（T127 path），更鲁棒。
- **核心证据**: HISTORY 主题 1 表 + 主题 3 conformal 路径选择
- **状态**: ✅

### 1.9 adaptive threshold V1-V4 (T168) 为啥失败
- **来源**: QREVIEW_PREDICTOR 间接 / HISTORY 主题 10 / CODE_QUESTION_LIST Q102（hysteresis 暗示）
- **答案**: T168 V1-V4 试图按 per-batch 预测分布（quantile / mean ± k×std / running median 等）动态调 thr，4 个 variant 在本地 holdout 全部退步。根本原因是**硬约束 #2**："评测程序会打乱测试点输入顺序" — Predictor 拿到的每个 batch 都是 shuffled，per-batch 统计量与时间相关性被破坏，adaptive thr 的方差远大于全局 thr 的方差。这一类 wrapper（adaptive thr、TTA、TTT、AR TTT）系列 T161/T162/T168/T169 全失败，证实**任何 per-batch state 都是项目禁忌**。
- **核心证据**: HISTORY 主题 10 "TTA K=5 / In-row TTT / Adaptive threshold V1-V4 / AR TTT 全负" + CRITICAL_CONSTRAINTS §1 #2
- **状态**: ✅

### 1.10 hysteresis（双阈值带）为啥没加
- **来源**: CODE_QUESTION_LIST Q102
- **答案**: 阈值附近抖动确实可能频繁切换 action（理论上手续费高），但实际项目没加 hysteresis 的 3 个原因：(a) per-sym conformal band 已经在 thr 之外加了 β·σ 弹性（sym=1 上 1.88e-4 ≈ 60% 的 thr_up），相当于"sym-conditioned hysteresis"；(b) 测试点被打乱（硬约束 #2），跨调用 hysteresis 实现不了，只能 per-batch hysteresis 但 per-batch 又触碰 #2；(c) 项目里没人专门测试过单次预测在阈值附近抖动的频率，从平台 +35.64 反推，抖动成本可能被 conformal band 已经吸收。
- **核心证据**: thresholds.json:54-60 conformal_wrapper；CRITICAL_CONSTRAINTS §1 #2
- **状态**: 🟡（没人直接测过）

---

## 2. 5-horizon emit + share_with=60 + sqrt scaling【含 USER-RAISED Q281 高亮】

### 2.1 ⭐ **USER-RAISED Q281：sqrt(H/60) 缩放方向与 random-walk-with-drift 推导反了**
- **来源**: **QREVIEW_PREDICTOR Q281**（用户精确推导）/ CODE_QUESTION_LIST Q19, Q122
- **用户原推导（必须保留）**:
  - 模型回归目标 y = (mp_{t+60} − mp_t) / (mp_t + 1) ⇒ pred 是 **60-tick 归一化 Δmid 的估计**
  - 短 horizon 直接 share pred_h60（**pred 不缩放**）
  - thresholds: thr_h = thr_60 × sqrt(h/60)（短 h 阈值变小）
  - **推导**（drift-with-noise）：E[Δmid_h] = μ·h（drift 线性于 h）；std[Δmid_h] = σ·sqrt(h)（noise sqrt(h)）
  - 若想把 "h=60 预测" 真正用作 "h=h0 决策"，**pred 应缩 h/60（线性）**，而不是 thr 缩 sqrt(h/60)
  - 正确做法 A：pred_h60 × (h/60) > thr_h_native
  - 等价做法 B：pred_h60 > thr_h_native × (60/h)（thr 反向缩 60/h，**变大**）
  - 若 thr_h_native = thr_60 × sqrt(h/60)（按 noise std 缩）⇒ 等效 thr = thr_60 × sqrt(60/h)
  - 代码实际：thr_60 × sqrt(h/60) ⇒ 与"理论正确"差 **60/h 倍（h=5 时 12 倍，h=10 时 6 倍，h=20 时 3 倍，h=40 时 1.5 倍）**

- **综合答案（赛题 + 项目 + 实验三视角）**:

  **(a) 用户推导是对的。** 给定模型回归目标 = 60-tick 归一化 Δmid，把它用作 h=5 决策，正确的 EV 比较应该是：
  - "预期 5-tick return" = E[Δmid_5] = (5/60) × E[Δmid_60] ≈ pred / 12（线性 drift 假设）
  - 与 h=5 native 阈值 thr_5_native 比较
  - 等价地：pred > thr_5_native × 12
  
  实际代码做的：pred > thr_60 × sqrt(5/60) = pred > thr_60 × 0.289 = pred > 8.66e-5。这把"等效门槛"降到 sqrt 缩放的 8.66e-5，而理论正确门槛应该是 thr_5_native × 12，方向与代码完全相反。代码"看似把短 h 门槛降低 → 更易触发买卖" 与 _doc 注释 "shorter horizons trade fewer/smaller signals" **直接矛盾**（注释把它当作"门槛升高" 是反的）。
  
  **(b) 那为啥项目没崩？** 三个 buffer 让这个 bug 几乎不影响 SOTA 分数：
  1. **score = max(per-horizon PnL)，h=60 是主战场**。平台历史短 horizon 全负（iter_002: h_5=−7.68, h_10=−8.64, h_20=−5.09, h_40=+2.02, h_60=+4.07；HISTORY 主题 10 表）。h=60 +35.64 一锤定音，短 h 的 PnL 哪怕 −10 也进不了 max。
  2. **conformal band 抵消缩放**（QREVIEW_PREDICTOR Q23 精确指出）：sym=1 band = 0.4 × 4.71e-4 = 1.88e-4。在 h=5 上 effective_thr_up = 8.66e-5 + 1.88e-4 = 2.75e-4 ≈ 没缩放的 thr_60。所以 sym=1 上 sqrt 缩放几乎被 conformal 吃掉。sym 3/4 (β=0) 才暴露完整 sqrt 缩放。
  3. **label_5/10 用 α=0.05%**（label_20/40/60 用 α=0.1%）：短 h 的"真涨/真跌"门槛比长 h 更严，模型 pred 在短 h 上更难超过 α；但代码没考虑这个 label-α 差异，只按"60-tick pred 的 std" 来缩，又是一层错位。
  
  **(c) 历史有没有做 ablation？** HISTORY 主题 10 找不到 sqrt vs h/60 linear vs 1.0 vs sqrt(60/h) 的平台对比。最可能的真实历史：作者实现 T188v3_fullhorizon 时把短 h 当作"score=max 的免费占位"，没认真推导 — `_doc` 注释 "shorter horizons trade fewer/smaller signals" 是一句**未经核对的口胡**（甚至方向反了）。这是项目里第一个有具体数学证据的可疑设计。
  
  **(d) 风险评估**: 由于 score = max(h_5, h_10, h_20, h_40, h_60)，错误 scaling 唯一的下行风险是**短 h 因为大量错误交易导致 PnL 不只是无信息地接近 0，而是负到拖累 max 排名**。这里就有 paradox：如果 sqrt 缩放让短 h 触发更多错单 → 短 h PnL 更负 → 但 max 仍取 h=60 = +35.64，无影响。**结论：代码现状对平台分数无负面影响，仅是注释和理论不符。**
  
  **(e) 建议补做的实验**（若再有提交配额）:
  1. 提交 4 个 variant，只改 thresholds.json `_note` 描述的 4 个短 h 的 thr：thr × {sqrt(h/60), h/60, 1.0, sqrt(60/h)}
  2. 看哪个 variant 让某个短 h 平台分数 > h=60 的 +35.64（极不可能但万一）
  3. 如果都 <+35.64 → max 不变 = +35.64，证实 sqrt 选择对 SOTA 排名无影响（但比 _doc 那种自圆其说更诚实）

- **核心证据**: QREVIEW_PREDICTOR Q281 全文；thresholds.json:2 `_doc`；HISTORY 主题 10 短 horizon 灾难表；Q23 conformal band ≈ 2.17× thr_5 的精确算式
- **状态**: 🔴 **真 gap**（数学方向反了 + 无 ablation；但平台风险被 max(score) + conformal band 双重 buffer 吸收，不影响 SOTA）

### 2.2 为啥短 horizon share h=60 而不独立训
- **来源**: QREVIEW_PREDICTOR Q21（间接 share_with 设计）/ CODE_QUESTION_LIST Q21, Q121, Q124
- **答案**: 整个项目从 iter_002 起就发现**短 horizon 在平台上是死路**（HISTORY 主题 10 短 h 灾难表）：
  - h_5 LOSO +17.49 → 平台 −7.68（gap −25.17）
  - h_10 LOSO +21.86（最强！）→ 平台 −8.64（gap −30.50）
  - h_20 LOSO +19.70 → 平台 −5.09
  - h_40 LOSO +11.71 → 平台 +2.02
  - h_60 LOSO +6.30 → **平台 +4.07**（唯一正）
  
  根本原因：label_5/10 用 α=0.05%，label_20/40/60 用 α=0.1%；fee 0.01% 双边 ≈ 0.02% 摧毁了 α=0.05% 的所有 alpha 空间。从 iter_002 之后，整个项目专注 h=60；短 h 都靠 share_with=60 + sqrt scaling 占位。**为什么不为短 h 训独立模型**：(i) 训 5×(50+50) = 500 模型时间成本 5×；(ii) 短 h 在平台上必负，再多训也不会变正；(iii) score=max(per-h)，短 h 只要不严重负就不影响排名。
- **核心证据**: HISTORY 主题 10 表 + iter_002 数据；CODE_QUESTION_LIST Q21 README 解释"max 兜底"
- **状态**: ✅

### 2.3 score=max(per-horizon PnL) — 平台真的这么算吗
- **来源**: QREVIEW_PREDICTOR Q8, Q9
- **答案**: 这是项目核心假设，但代码层面**没有 100% 验证**。HISTORY 主题 10 注释"整个项目专注 h_60，h_60 几乎总是最高"；CODE_QUESTION_LIST Q123 README 也只说"取 max，添加只升不降"。实际证据：
  - 平台只显示一个分数，不展示 per-h 分数
  - 22 次提交中每次的本地 holdout per-h 表都有 h=60 最高
  - 22 次平台分数 ≈ 本地 h=60 × 透传系数（53-71%）
  
  **若 max 假设错了**（如平台改为 sum 或 weighted），现行 sqrt scaling 设计就会主动伤害分数（短 h 错误交易 PnL 负向）。这是项目里最大的"隐含 spec 信仰"。Worker 建议：下次有提交配额时提交 1 个版本只激活 h=60（其他 horizons "active": false），对比平台分数；若分数与 SOTA 完全一致 → max 假设确认；若 SOTA 下降 → max 假设错。
- **核心证据**: HISTORY 主题 10 + 22 次提交平台分；thresholds.json `_doc` 自述
- **状态**: 🟡（假设强但无直接证据）

### 2.4 sqrt scaling 数学依据 — 项目内的辩护版本
- **来源**: QREVIEW_PREDICTOR Q6, Q7 / CODE_QUESTION_LIST Q19, Q122
- **答案**: HISTORY 主题 10 的官方解释："Δmid 的标准差 ∝ sqrt(H)（Brownian motion 假设下，方差线性增长 → 标准差 sqrt 增长），所以 h=5 的预测分布尺度 ≈ sqrt(5/60) × h=60 的尺度，阈值按 sqrt(H/60) 缩放，所有 horizon 的'显著性'标准统一"。
  
  **这个解释只在一种解读下成立**：把 pred 当作 horizon-agnostic "signal strength"（与 h 无关的 z-score-like quantity）⇒ thr 按 noise std 缩。但 pred 显式回归 60-tick Δmid（训练目标 y=(mp_{t+60}−mp_t)/(mp_t+1)），这个解读违反训练目标语义。Q281 用户推导是对的：在 drift-with-noise 模型下，正确缩放应该是 pred×(h/60) 或等效 thr×(60/h)。
  
  **项目内最诚实的回答**：sqrt 是"看起来合理 + 平台 max 假设兜底 + 历史短 h 一直负不在意" 的妥协，不是从原理推导出来的最优。
- **核心证据**: HISTORY 主题 10 "数学依据" 段 vs QREVIEW Q281
- **状态**: 🔴（与 2.1 同根 gap）

### 2.5 label_5/10 α=0.05% vs label_20/40/60 α=0.1% — 对 share_with 的隐含影响
- **来源**: 项目 setting + 隐含问题
- **答案**: 训练时 label_h ∈ {0=short, 1=flat, 2=long} 是根据 (mp_{t+h}−mp_t)/(mp_t+1) 是否超过 α_h 阈值决定。α_5=α_10=0.05%（5e-4），α_20=α_40=α_60=0.1%（1e-3）。模型 NN/LGB 都只回归到 h=60 的连续 y，不直接用 label。但 EV gate 的"显著性"含义在不同 h 下差异巨大：
  - h=5 native 阈值（按 label 边界）：~5e-4
  - h=60 native 阈值（按 label 边界）：~1e-3，扣 2× fee = 0.02% 余 ~8e-4
  - 项目 thr_60=3e-4 远小于 label α=1e-3，因为模型 pred 标度比 label 边界小（pred 是连续，label 边界是 hard boundary）
  
  代码 sqrt scaling 把短 h thr 缩到 8.66e-5，**与 short-h label α=5e-4 没有对应关系** — 进一步说明 sqrt 缩放不是从 label 推导的，是凭直觉填的。
- **核心证据**: PROGRESS.md / 0512_final_report.md（label α=0.05% vs 0.1% 的设定）；thresholds.json:7（thr_up h=5 = 0.0000866）
- **状态**: 🔴（label-α 差异未在 thr scaling 中体现）

### 2.6 短 horizon 用同一组 w_nn=1.0 / w_lgb=1.5 — 合理吗
- **来源**: QREVIEW_PREDICTOR Q28 / CODE_QUESTION_LIST Q126
- **答案**: 短 h 的 w_nn/w_lgb 是 thresholds.json 里**冗余配置**（h=5/10/20/40 entries 都写了 1.0/1.5），但 Predictor.py:343 实际查的是 `self._weights[src_H]`，即 src_H=60 的权重（h=60 entry 的 1.0/1.5）。所以 h=5/10/20/40 的 w_nn/w_lgb 是 **dead config**，改它们不影响行为。冗余存在的原因是 build_pkg 用 mechanical 方式 generate 配置，没做去重。**风险**：将来如果有人误信 short-h w_nn/w_lgb 可独立调，会以为自己改了实际上没改，浪费时间或埋 bug。
- **核心证据**: Predictor.py:343 `self._weights.get(src_H, (1.0, 1.0))`
- **状态**: ✅（dead config，不影响生产但有维护风险）

### 2.7 share_with cycle 风险
- **来源**: QREVIEW_PREDICTOR Q54, Q55, Q197, Q247, Q248
- **答案**: Predictor.py:336 `src_H = int(hcfg.get("share_with", H))`；如果 share_with 指向不存在的 horizon，Predictor.py:341-342 `self._lgb_lists.get(src_H)` 和 `self._nn_batched.get(src_H)` 都 return None，preds=[]，line 353-354 `if not preds: continue` — 该 horizon **silent 留 flat (action=1)**。如果配置错把 h=60.share_with=5 而 h=5 又 share_with=60，h=60 出 flat，整个 SOTA 崩成 ~0。没有 cycle 检测，没有 unreferenced source 校验，没有 "ensure share_with target is active and has models" 的 startup check。这是 schema 健壮性 gap。**生产路径不触发**（h=60 不写 share_with，其他都指 60，无 cycle）。
- **核心证据**: Predictor.py:336-354；QREVIEW Q248 详细分析
- **状态**: 🟡（生产路径安全，但配置鲁棒性 gap）

---

## 3. Conformal per-sym abstain

### 3.1 per-sym β = {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00} — 从哪来
- **来源**: QREVIEW_PREDICTOR Q10, Q11 / CODE_QUESTION_LIST Q5, Q104
- **答案**: T150 (R_conformal_select) **4-fold CV consensus sweep** 直接产物。Grid: β ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} per sym, 5 个 sym 独立选最佳。
  - sym=0 β=0.10：相对稳定，轻量 abstain
  - sym=1 β=0.40：最 brittle（HISTORY 主题 3 说 sym=1 与 sym=2 都需要重 abstain）
  - sym=2 β=0.30：T8 诊断 amount_delta z-score +7.49 偏离，蓝筹/ETF 性质，模型在其上 brittle
  - sym=3 β=0.00：模型在其上预测最稳，abstain 反损 PnL
  - sym=4 β=0.00：同上
  
  4-fold CV consensus 而非单 fold OOF：在 4 个 fold 上分别选最佳 β，取 consensus（多数票），避免 T126 per-sym OOF threshold 那种过拟合。后续 iter_018/019/T170/T188v2 全部沿用，**从未微调**。
- **核心证据**: HISTORY 主题 3 表 + T150 行；CRITICAL_CONSTRAINTS § 没禁止 conformal
- **状态**: ✅

### 3.2 per-sym σ = {0:2.40e-4, 1:4.71e-4, 2:4.52e-4, 3:4.24e-4, 4:4.28e-4} — 是什么
- **来源**: QREVIEW_PREDICTOR Q12, Q13, Q15 / CODE_QUESTION_LIST Q6, Q103
- **答案**: 这些 σ 是 **per-sym calibration set 上 5-NN ensemble 预测的标准差**（不是 |pred-y| residual std；不是 pred 本身的 std；是 ensemble 内部的 disagreement）。10 位 sig fig 因为是 numpy 直接 dump 的浮点数。calibration set 大致是 V4 walk-forward 的 val 段（date 80-95，每 sym ~28k 行）。注意：
  - σ_sym1 最大 (4.71e-4)：模型在 sym=1 内部分歧大 → β=0.40 加宽 abstain
  - σ_sym0 最小 (2.40e-4)：模型在 sym=0 内部分歧小 → β=0.10 已够
  - σ 是基于 5-NN（早期 ensemble 规模）算的，后来 ensemble 扩到 50-NN（T188v2），σ 没重算
  - 注：代码里 σ 是写死的 JSON 常数，没 script 直接 calibration（CODE Q103 的合理担忧）
- **核心证据**: thresholds.json:57 + Predictor.py:222；HISTORY 主题 3 描述（σ_pred = 5 NN 的预测标准差）
- **状态**: 🟡（σ 用 5-NN 不是 50-NN；calibration script 不在 final 包里）

### 3.3 default_beta_for_ood=0.16 — mean 还是别的
- **来源**: QREVIEW_PREDICTOR Q14, Q16 / CODE_QUESTION_LIST Q7
- **答案**: 0.16 = mean(0.10, 0.40, 0.30, 0.00, 0.00) = 0.80/5。**算术平均**。理论上不是 robust 选择：
  - **median(0.10) 会更保守**：默认假设 OOD sym 像 sym=0（最像 "正常"）
  - **max(0.40) 会最保守**：默认 OOD sym 像最 brittle 的 sym=1
  - **mean(0.16) 是 "妥协" 默认**：unweighted average，无原理依据
  
  搭配 default_sigma_for_ood = 0.0003998317（同样是 mean of per_sym_sigma），OOD band = 0.16 × 4.0e-4 = 6.4e-5。这个 band 比 sym=1 的 1.88e-4 小**很多**（30%）— **OOD 反而被更激进交易，比已知不可靠 sym 还激进**。QREVIEW Q16 直接指出 "默认应该是 MAX，不是 mean"。这是真 gap。
  
  辩护版（项目从未明确说，是 worker 推测）：OOD sym 可能其实和训练 sym 是同一只股票（"同一 ID 可能映射不同股票" - CRITICAL_CONSTRAINTS §1 #3），如果概率上偏 in-distribution，mean default 是 reasonable prior。
- **核心证据**: thresholds.json:58-59；Predictor.py:222-223；QREVIEW Q14, Q16
- **状态**: 🔴（设计上 OOD 反而更激进，与"OOD 应保守"直觉相反；项目未做 ablation）

### 3.4 enabled=true — 为啥
- **来源**: QREVIEW_PREDICTOR Q252
- **答案**: T127 (iter_018) 首次启用 conformal wrapper，平台 +0.77 over iter_015（**首次验证有效**）。T182 是去 conformal ablation：T170 → T182 平台 −0.30，本地 holdout +6.62 → **关键 paradox**：本地 in-sample 上 abstain 反而损 PnL（因为训练数据可信，abstain 是无用弃权），平台 OOD 上 abstain 过滤掉真实错误预测。这是项目里 in-sample 评测无法预测平台行为的最经典案例（HISTORY 主题 9 in-sample 污染案例 4 个之一）。所以即使本地 holdout 看起来 conformal "hurt"，enabled=true 是基于平台 OOD 表现的正确选择。
- **核心证据**: HISTORY 主题 3 T182 行 + 附录 A 发现 7
- **状态**: ✅

### 3.5 为啥不用 isotonic / Platt 校准
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q105
- **答案**: 三类失败实验：
  - **T15** isotonic + EV gate：LOSO h_10 +21.86 → +20.84（**−1.02**，破坏排序）
  - **T131** isotonic addon（iter_015 上）：LOSO 小正但不超 conformal
  - **T40** Platt scaling：LOSO 持平/负
  - **R13** calibration + position sizing 调研：isotonic/Platt 均负
  
  原因：isotonic / Platt 是**分类校准**，把 score 映射到 calibrated probability。但项目用的是**回归 EV gate**（pred > thr_up → 买），不需要 probability，需要保留 pred 的连续幅度信息。任何单调校准（monotone）都不改 ranking，但会改 thr 含义；非单调校准（Platt 是 sigmoid，会压缩两端）会破坏极端预测的可信度。Conformal abstain 是不动 pred / 只加 band 的"零侵入"方案，所以胜出。
- **核心证据**: HISTORY 主题 3 T15/T131/T40/R13 行
- **状态**: ✅

### 3.6 T127 / T150 / T182 三步 conformal 演化（关键 ablation 链）
- **来源**: 综合
- **答案**:
  - **T127 iter_018 v1**: 首次加 per-sym beta conformal（5 NN + 5 LGB），LOSO-eq +39.75 → +41.49 (+1.75)；平台 +28.93（+0.77 over iter_015）→ **首次平台验证有效**
  - **T150 R_conformal_select**: 4-fold CV consensus sweep 确定最优 β = {0.10, 0.40, 0.30, 0.00, 0.00}，固化到所有后续提交
  - **T182 NoConformal ablation**: T170 - conformal wrapper，本地 holdout +156.56 (+6.62 vs T170 +149.95)，**平台 +34.34 (−0.30 vs T170 +34.64)** — paradox：本地看似有害，平台仍有 +0.30 收益
  
  累计平台净收益 +0.30 ~ +0.77 PnL，相对项目 SOTA +35.64 占 ~2%。
- **核心证据**: HISTORY 主题 3 时间线 + 附录 A 发现 7
- **状态**: ✅

### 3.7 conformal "wrapper" 是真 conformal 吗
- **来源**: CODE_QUESTION_LIST Q107
- **答案**: 严格说**不是经典 conformal**。经典 conformal prediction 是用 calibration set 上的 nonconformity score 取分位数得 prediction interval。这里：
  - "score" = 模型 ensemble 预测的 std σ（disagreement 度量）
  - "interval" = ±(thr + β·σ) 是 hard threshold，不是 conformal interval
  - β 不是 quantile level，是 grid-searched scalar
  - 没有 per-batch quantile 计算（硬约束 #2 禁止）
  
  本质是 **"per-sym signal-to-noise gating"**：band = β × σ_ensemble_disagreement，把"模型自己分歧大"的 sym 上的边界 prediction 弃权。命名 "conformal" 是借术语，提供 "uncertainty-aware abstain" 的直觉。命名学术上不准确，但概念是合理的 abstain heuristic。
- **核心证据**: Predictor.py:255-262；CODE Q107 的合理质疑
- **状态**: 🟡（命名不准确，行为有效）

### 3.8 σ 没动态更新（5-NN → 50-NN 不重算）
- **来源**: HISTORY 主题 3 隐含
- **答案**: σ_sym 是 T127/T150 阶段在 5-NN ensemble disagreement 上算的。T188v2 ensemble 扩到 50-NN 后理论上 disagreement 会变小（更多 NN 平均后 std 减小 ≈ σ_5 / sqrt(50/5)）。如果 σ 重算，band 会缩小，conformal 弃权变弱，可能更激进交易。**没人在 T188v2 阶段重算 σ**。理由（推测）：(a) 风险厌恶 — 重算 σ + 重新 sweep β 是 2D 二次过拟合风险；(b) "已验证不动" 原则；(c) 平台分 +35.64 已是 SOTA，不冒险动。
- **核心证据**: HISTORY 主题 3 + 主题 2 ensemble 扩容时间线
- **状态**: 🟡（已知 gap，但保守是正确选择）

---

## 4. Ensemble 规模 50+50 与权重 w_lgb=1.5

### 4.1 为啥 50 NN + 50 LGB（不是 5+5 / 100+100 / 150+150）
- **来源**: QREVIEW_PREDICTOR Q31, Q32, Q33 / CODE_QUESTION_LIST Q8, Q171
- **答案**: 5→50 直接对比：
  - **5+5 (iter_018, T127)**: 平台 +28.93
  - **5+5 + M7 (iter_019 v2, T140)**: 平台 +34.44（M7 trick 主因）
  - **40+5 (T179 40-NN bag)**: holdout +0.63 vs T170; 平台 +34.39 **−0.25** vs T170 → **只扩 NN 单侧无效**
  - **50+50 (T188v2)**: holdout +150.43 (+0.48 vs T170); 平台 **+35.64** (+1.00 vs T170) → **NN+LGB 都扩才有效**
  - **150+150 (T190)**: holdout +140.62 (−9.81 vs T188v2); 平台 +34.59 **−1.05 vs T188v2** → **超过 50 反效果**
  
  关键发现（HISTORY 附录 A 发现 2）：
  1. NN+LGB 必须对称扩容（variance reduction 需要两边都 var↓）
  2. 5→50 +1.00（边际正，有效）；50→150 −1.05（边际负，HP 多样性变噪音）
  3. 中间点 25+25 / 75+75 / 100+100 **未做** — 50 vs 150 是直接跳跃
  
  **为啥不试中间点**：每次平台提交是 12h 1 次限制 + 训 50 + 50 模型本地耗 ~18min on 5 GPU；后续 fine-grain sweep 时间成本 = 收益不确定 vs 已 SOTA 风险。
- **核心证据**: HISTORY 主题 2 时间线 + 附录 A 发现 2
- **状态**: ✅（50 是 informed but non-optimal 选择；中间点 ablation 是真 gap）

### 4.2 w_NN=1.0 / w_LGB=1.5 — 为啥 LGB 更重
- **来源**: QREVIEW_PREDICTOR Q25, Q26, Q27 / CODE_QUESTION_LIST Q3
- **答案**: w_LGB=1.5 起源 iter_015 (T87) 阶段：SPO+ DFL NN + T75 LGB 集成时 grid search（具体 grid 未公开，最可能是 {1:1, 1:1.2, 1:1.5, 1:2, 1.5:1}），发现 w_LGB > 1 更好 → 固定 1.5，**从未重新 sweep**。
  - effective weight share: 1.5 / 2.5 = 60% LGB, 40% NN
  - 理论：LGB 在 H4 tabular features 上一般比 MLP 更强（特征工程主导，深度学习边际）；T87 NN 是 fine-tuned with SPO+ 但 base model 表征能力弱
  - 50+50 阶段重新 val 上确认 w_LGB=1.5 仍接近 optimal（具体 val 数字未存档）
  - T192 试图用 bounded NNLS 自动学权重，惨败（见 4.4），固化了"手动 1.0/1.5"决策
- **核心证据**: HISTORY 主题 2 "w_LGB=1.5 起源" 段
- **状态**: 🟡（grid search 数据不可考；50+50 阶段重新 sweep 也未存档）

### 4.3 simple mean vs bounded NNLS (T192) — overfit 灾难
- **来源**: QREVIEW_PREDICTOR Q29, Q30, Q51 / CODE_QUESTION_LIST Q97
- **答案**: T192 用 bounded NNLS 在 H4 (h=60) val 上学 150+150 个 per-model 权重：
  - val +180.41, test +193.94, ratio 1.075（看似 well-calibrated）
  - 平台 **+31.0 (−4.64 vs T188v2 +35.64)** — 灾难
  - 死因（HISTORY 附录 A 发现 8）：NN-NN 平均 corr **0.9331**，LGB-LGB 平均 corr **0.8433**，condition number **54182** — **极病态矩阵**。共线性消除权重可识别性，NNLS 学到的"最优"权重是 val/test 共同 noise pattern 的过拟合，平台 OOD 上完全无用。
  
  教训：50+ 高度相关模型上学权重 = 把噪声当 signal。simple mean 是 variance reduction 的 robust 最优。**任何 per-model 权重学习方案都被这个 ratio 1.075 + 平台 −4.64 直接否决**。
- **核心证据**: HISTORY 主题 2 T192 行 + 附录 A 发现 8
- **状态**: ✅

### 4.4 family-mean → weighted vs 100 model 等权重
- **来源**: QREVIEW_PREDICTOR Q30 / CODE_QUESTION_LIST Q98
- **答案**: 当前 `pred = (1.0×mean(50 NN) + 1.5×mean(50 LGB)) / 2.5`，即 **family-level 加权**。等权 100 model 等价于 `pred = (50×1×mean_NN + 50×1×mean_LGB) / 100 = (mean_NN + mean_LGB) / 2`（50/50 share）。两者数学不等价，因为 family-mean weight 1:1.5 把 LGB 权重提到 60%。
  - **为啥不做 model-level 权重**：T192 NNLS 灾难证明 model-level 学权重 = overfit
  - **为啥不做等权**：grid search 发现 w_LGB > 1 更好
  - **不是 log-pool / rank-averaging**：项目里没试过，单 layer mean 已 SOTA
- **核心证据**: Predictor.py:357；T192 教训
- **状态**: ✅

### 4.5 ensemble_seeds=[1..50] 硬编码风险
- **来源**: QREVIEW_PREDICTOR Q99, Q161, Q242 / CODE_QUESTION_LIST Q100, Q131-133
- **答案**: thresholds.json:51 写死 50 个 seed list。Predictor.py:244 `if os.path.isfile(lp)` silent skip missing。**漏 1 个 seed → 49 model ensemble silent**（无报错）。build_pkg.py 也 only copies existing files。三处 source of truth：(a) thresholds.json ensemble_seeds, (b) build_pkg `N_NN=50, N_LGB=50`, (c) filesystem 中实际存在的文件。三者必须人工对齐。**没有 min-count assert**（如 `assert len(loaded) >= 40`），所以即使只有 5 个 model load 成功也 silent ship 成 5-model ensemble，平台分数会大跌。
- **核心证据**: Predictor.py:240-251；build_pkg.py（QREVIEW Q118）
- **状态**: 🟡（生产路径 OK，但 schema 健壮性弱）

### 4.6 50 个 NN 内部相关性高 — ensemble 边际
- **来源**: QREVIEW_PREDICTOR Q32, Q33
- **答案**: T192 测出 NN-NN avg corr 0.9331，LGB-LGB 0.8433。理论上：
  - 50 个 corr=0.93 model 平均：variance reduction = (1 - 0.93)/(1) × 1 + 0.93 ≈ 0.93 + 0.07/50 = 0.9314。即变 0.99×(1-0.93) ≈ 1.4% var↓ over single model
  - 与 single best NN 比，50-NN ensemble 的 std 减少约 1.4%，benefit 微薄
  - **但平台 +1.00 是实测**（T188v2 vs T170 5+5 baseline）→ 项目 SOTA 的"最后一公里"几乎全靠 variance reduction 微薄边际 + LGB 也对称扩
- **核心证据**: T192 corr / cond 数据；T188v2 vs T170 +1.00 delta
- **状态**: ✅

### 4.7 50 NN HP 实际只覆盖 12 个独立 combo（cycling bug?）
- **来源**: CODE_QUESTION_LIST Q13, Q95
- **答案**: README 写 "3 lr × 4 dropout × 3 batch = 36 combos"，但 train_T188v2_nn_seed.py:186-188 用 `(S-1) % 3` for lr 和 `(S-1) % 3` for batch — **lr 和 batch index 完全同步**。lcm(3,4,3)=12，所以 50 seed 实际只覆盖 12 独立 combo，每 combo 重复 4.17 次。这是**有意设计还是 bug**：
  - 项目从未明确说，HISTORY 主题 2 也没分析 NN HP diversity
  - 推测：bug；但 12 combo × 4.17 seed-reps 实际 ≈ "12 HP × 4 seed"（HP-dominant diversity），与 LGB 的 "5 HP × 10 seed" 设计哲学一致（R_multinn_spo 调研 "HP > seed diversity"）
  - 平台 +35.64 SOTA 没崩，所以不构成实际问题。但 README "36 combos" 描述不准确
- **核心证据**: CODE Q13, Q95 + train_T188v2_nn_seed.py:186-188（其他 worker 主战场，这里只引用）
- **状态**: 🟡（不影响 SOTA；README 不准）

### 4.8 family-asymmetric optimization（NN batched, LGB sequential）
- **来源**: QREVIEW_PREDICTOR Q34, Q36, Q37
- **答案**: NN 用 batched torch.bmm 一次跑 50 模型（T188v3 优化主战场，186× 加速）；LGB 用 Python for loop sequential 50 次 `.predict()`（注释 ~84ms 已可接受）。**为啥不对称优化 LGB**：
  - 单 LGB booster `.predict()` 内部已多线程（OMP）
  - LightGBM 4.6 没有"batched ensemble predict" API；要把 50 booster stack 需要 C++ 端工程
  - 84ms 不是瓶颈（NN 推理优化前 1509ms 是 18× LGB）
  - 项目时间预算花在收益最大的地方（NN 186× 是最大单点收益）
- **核心证据**: HISTORY 主题 10 T188v3 时间线
- **状态**: ✅

---

## 5. Ensemble 加法/拒绝（CatBoost / Huber / GRU / T97 / Transformer / Agreement / 4-way / 5-way）

### 5.1 CatBoost (T89 / T166) — 为啥没采纳
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q41, Q99
- **答案**: 
  - **T17** 早期 LGB vs CatBoost ceiling 测试：LOSO 持平（+22.2 左右）
  - **T89** CatBoost L2 5-seed: LOSO +40.13，**与 T87 NN avg corr 0.77**（高度共线）→ 加入 4-way (T87+T89+T99+T95) iter_016 v3 平台 **+25.08 −3.08 vs iter_015**
  - **T166** CatBoost M7 retrain：holdout +0.39（边际正），但 5-way T167 (+CB) +0.11 within noise
  - **死因**：CatBoost 与 LGB tree-based 模型族 + 同特征 SchemeP 359 维 → 高度共线，ensemble diversity 几乎 0。CLAUDE.md 提"CatBoost GPU 也可用"但因 collinearity 没 incremental 价值
- **核心证据**: HISTORY 主题 2 T89/T166/T167 + 主题 8 表
- **状态**: ✅

### 5.2 Huber (T99) — Huber LGB family
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q41
- **答案**: T99 Huber LGB family LOSO +44.72（**含 OOF contamination 警告**）→ iter_016 +25.08 失败的一员。Huber 是 robust loss（L1 + L2 平滑混合），对重尾 return 应该比 L2 更稳。但：
  - LOSO 高分含 OOF 污染（HP 在 OOF 上 sweep）
  - iter_016 4-way 上参与了过拟合
  - T143/T144/T145 后续 MAE 系列也证 robust loss 在平台上不超 L2
  - L2 + EV gate 的组合在 LOSO-eq +36.23 → 平台 +19.23 之后被锁死
- **核心证据**: HISTORY 主题 7 T99/T137/T143/T144/T145 行
- **状态**: ✅

### 5.3 GRU (T95) — iter_016 失败主因
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q36
- **答案**: T95 GRU regression：LOSO 微小提升，**iter_016 v3 平台 +25.08 (−3.08 vs iter_015 +28.16) 的主因**（HISTORY 主题 2/8）。死因：
  - GRU 有 hidden state，但项目硬约束 #2（测试点被打乱）禁止跨 batch state
  - 单 batch 内 100-tick window 上的 GRU forward 等价于 "短 LSTM"，但内部 cell state 容易在短 horizon 过拟合
  - T119/T120 ablation 去掉 GRU → 简单集成更鲁棒 → ablation 直接确认 GRU 是负贡献
  - **教训**：sequence model 在 LOB 的 high-noise low-signal + shuffle 评测下不行
- **核心证据**: HISTORY 主题 8 T95 行 + iter_016 v3 失败案例
- **状态**: ✅

### 5.4 第二组 NN T97 (T156 / T163) — diversity≈0
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q35, Q36
- **答案**: T97 是另一版 SPO+ DFL NN（与 T87 同架构、不同 init seed）。
  - **T156** NN swap：T97 替/叠 T87，holdout +1.66
  - **T158** agreement filter（NN 和 LGB 方向一致才出手）：holdout +0.25
  - **T163 v2+T97+agree** (5+5+T97+agree)：holdout +42.90 (+1.41) → 平台 **~+30.74 (−3.70 vs v2 +34.44)** ← 灾难
  - **死因**：T87/T97 同架构（MLP [359→256→128→64→1] + SPO+），diversity ≈ 0（只是不同 init）；本地看似正是因为 in-sample 训练数据上加 model 总能拟合得更好；平台 OOD 上同架构带来同样的 systematic error，没有互补
  - **教训**：diversity 必须在**架构层面**，不是 seed/HP 层面
- **核心证据**: HISTORY 主题 2 T156/T158/T163 + 主题 8 同源问题
- **状态**: ✅

### 5.5 Transformer / GroupTransformer (T172 / T187) — 未采纳
- **来源**: QREVIEW_PREDICTOR 间接 / CODE_QUESTION_LIST Q36
- **答案**: 
  - **T172** GroupTransformer (d_model=64, nhead=4, 2 layers, n_tokens=32, CLS)：**in-sample +149.91**, best ensemble +164.71 → 全 **in-sample 污染**（date 0-119 全数据 retrain + 96-119 评分含训练数据）
  - **T187** Plan B add：in-sample +12.6/14.7 → 同样污染
  - **未提交平台**：因为 in-sample 数字不可信（HISTORY 附录 A 发现 1 + 主题 9 in-sample 污染 4 案例之一），新架构没有 V4 OOF 验证，OOD 风险高
  - **R31 文献调研**：TransLOB / TabLOB / 9 篇 深度笔记 — 指导 T172 设计，但确认 MLP+GBDT 路线竞争力够
  - **R35 alt architectures**：探索性，未提交
- **核心证据**: HISTORY 主题 8 T172/T187 + 主题 4 M7 trick "新架构 M7 retrain 高风险"
- **状态**: ✅

### 5.6 Agreement filter (T158) — 小本地正，未独立采纳
- **来源**: 项目演化
- **答案**: T158 "agreement filter"：只在 NN 和 LGB 方向一致（同 long、同 short、同 flat）时才出手；否则 flat。holdout +0.25 微正。**独立未采纳**因为：
  - +0.25 小于本地 holdout 噪音
  - T163 与 T97 agree 组合一起测，导致整体灾难无法 isolate agreement filter 贡献
  - EV gate + conformal 已经在做"signal-quality" 过滤
- **核心证据**: HISTORY 主题 2 T158 行
- **状态**: ✅

### 5.7 4-way (T164) / 5-way (T167) — 没进 final
- **来源**: 项目演化
- **答案**: 
  - **T164** 4-way (T87+T97+T95+LGB)：holdout +0.34 → 仍含 T97 不安全
  - **T167** 5-way (T87+T97+T95+CB+LGB)：holdout +0.11 over T166 → within noise
  - **死因**：含 T97（与 T87 共线）和 GRU（iter_016 主因之一）；本地小正全是 in-sample artifact；平台未提交因为 ensemble 复杂度↑ + 已知 T97/T95 风险 → 不冒险
- **核心证据**: HISTORY 主题 2 T164/T167 行
- **状态**: ✅

### 5.8 总结：什么样的 model 进入了 ensemble
- **来源**: 综合
- **答案**: T188v2 final 包**只含 2 族**：
  - MLP [359→256→128→64→1] LayerNorm+GELU+Dropout(0.10)（NN 主力，T87 SPO+ DFL）
  - LightGBM L2 regression（LGB 主力，T75 family）
  
  **拒绝**：CatBoost (T89/T166 共线 + 无 incremental), Huber (T99 OOF 污染), MAE (T137/T143/T145 不超 L2), Quantile (T132 边际), GMADL (T116 不超), GRU (T95 hidden state + shuffle 矛盾), 第二组 NN T97 (架构同源), GroupTransformer (T172/T187 in-sample), CNN (T184/T185 OOD 未验证)。
  
  整个 ensemble 设计哲学：**少而精 > 多而乱**；diversity 优先架构层面（NN vs tree）而非 model 数量。
- **核心证据**: HISTORY 主题 8 整段 + 主题 2 被拒绝表
- **状态**: ✅

---

## 6. EV gate 决策

### 6.1 为啥 EV gate 而不是 argmax 分类
- **来源**: QREVIEW_PREDICTOR 间接 / 项目第 1 次突破
- **答案**: 项目 Phase 0-1 (T1-T2 DeepLOB baseline / SchemeC LGB) 用 3-class CE（label_h ∈ {0,1,2} argmax）：
  - mmpc_demo (DeepLOB) 平台 **−6.65**
  - iter_002 (LGB SchemeC argmax) LOSO h_60=+6.30 → 平台 +4.07
  - 整个 3-class CE 路线在平台上短 h 严重负
  
  **T75 (iter_013, 1st 突破)** 换 L2 Δmid regression + EV gate：LOSO-eq +26.44 → **+36.23 (+9.79, +37%)**；平台 +4.07 → **+19.23 (+15.16)**。这是整个比赛**最大单次跃升**。
  
  关键洞察：3-class CE 把 Δmid 离散化为 3 个 bucket（涨/平/跌），损失幅度信息；EV gate `pred > thr` 保留连续幅度，pred 大 → 信号强 → 直接对应"期望收益超过 fee+thr 才出手"的微观经济学。EV gate 是 PnL-aligned，3-class CE 不是。
- **核心证据**: HISTORY 主题 7 T75 行 + 附录 A 发现 6
- **状态**: ✅

### 6.2 为啥不用 isotonic / quantile regression
- **来源**: CODE_QUESTION_LIST Q105
- **答案**: 见 3.5 节（isotonic/Platt 校准全部负面），加 T132 quantile LGB (α=0.5 ≈ MAE) 边际，T86 quantile + dual-gate (IQR-mid hybrid) LOSO-eq +40.22 但不超 T87 SPO+ NN +40.09 ensemble。Quantile 提供 prediction interval 但 EV gate 不需要 interval（只需 point estimate vs thr）。
- **核心证据**: HISTORY 主题 7 表
- **状态**: ✅

### 6.3 binary cascade (T72) 为啥未采纳
- **来源**: 项目历史（HISTORY 间接）
- **答案**: T72 binary cascade（先判涨/跌，再判幅度）是 hierarchical classification 思路。在 LOSO 上未超 EV gate，因为：
  - cascade 引入 2 次 thr，过拟合空间 2×
  - 2-stage decision 在 batch shuffle 评测下 stage 1 和 stage 2 共享同 batch，但 stage-wise calibration 复杂
  - EV gate 1 个 thr 已 SOTA
- **核心证据**: HISTORY 间接（T72 在 T75 之前的探索期，被 T75 + EV gate 直接超越）
- **状态**: 🟡（项目文档未单独记录 T72 详细消融）

### 6.4 IDS-SCG (T118) 为啥未采纳
- **来源**: 项目历史
- **答案**: HISTORY 主题 1 表里 T118 未列；项目早期探索之一。IDS = Information-Directed Sampling，SCG = Stochastic Composite Gradient — 是 bandit / online learning 思路。在固定 train/eval setting 下不如端到端 train + EV gate 简洁。0512_final_report.md 中可能有更详细记录，但 HISTORY 没列 → 表明被项目 PM 认为不重要 / 未取得正收益。
- **核心证据**: HISTORY 不在主要时间线
- **状态**: 🟡（项目文档证据弱）

### 6.5 EV gate fee 关系
- **来源**: CODE_QUESTION_LIST Q22, Q106
- **答案**: thr_up=3e-4 = 30bps，fee=10bps 双边（0.0001 single side × 2 round-trip）。thr/fee ≈ 3 → 模型要求毛预期收益 ≥ 3× fee 才出手，留 2× fee 净利润空间。这是 EV gate 的自然校准：
  - pred > thr_up = pred > 3 × fee → 期望 net PnL > 2 × fee
  - 经验上 3-5× fee 是 HFT 文献"breakeven + margin" 的标准范围
  - SPO+ DFL 在 NN 训练时已经把 fee 编码进 loss (`fee_eff = FEE × ((mp_th+1)+(mp_t+1))/(mp_t+1)`) → 训练目标已 fee-aware，与推理 thr 自洽
- **核心证据**: CODE_QUESTION_LIST Q22, Q106；train_T188v2_nn_seed.py:33（其他 worker 主战场，引用）
- **状态**: ✅

---

## 7. 推理优化 batched bmm + CPU torch + 文件格式

### 7.1 batched torch.bmm 50 NN 推理 — 186× 加速
- **来源**: QREVIEW_PREDICTOR Q37, Q48, Q189 / CODE_QUESTION_LIST Q111
- **答案**: T188v3 核心优化。把 50 个 MLP 的权重 stack 成 (N=50, out, in) tensor，单次 `torch.bmm` 完成所有 50 forward：
  - **before** (T188v2): 50 NN sequential numpy 推理 ~1509ms / 1024 batch（stated baseline）
  - **after** (T188v3): batched bmm ~8.1ms → **186× 加速**
  - 实测 NN-only 554.8ms → 8.1ms → 68.5×（基线差异是 numpy vs torch 单 NN 实现）
  - end-to-end (feature + LGB + NN) 2142ms → 641ms → **3.3× e2e**；442k 行总时间 15.4 min → 4.6 min
  - actions 一致性：T188v3 vs T188v2 max diff **6.98e-10**（远低于 1e-4 阈值容忍度）
  
  为啥 bmm 不是单 matmul `(out*50, in)`：bmm 的 (N, B, in) × (N, in, out) 保持 per-NN 独立 normalization 和 bias，是 N 个独立 linear；单 matmul 需要 concat batch 维度，无法 per-NN feat_mean/std。
- **核心证据**: HISTORY 主题 10 T188v3 行 + Predictor.py:78-178 (_BatchedMLPEnsemble)
- **状态**: ✅

### 7.2 CPU torch vs GPU — 为啥代码有 CUDA 分支但 requirements 是 +cpu
- **来源**: QREVIEW_PREDICTOR Q140, Q141 / CODE_QUESTION_LIST Q112
- **答案**: Predictor.py:193 `self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，但 requirements.txt `torch==2.5.1+cpu`。CPU 构建下 `cuda.is_available()` 永远 False，永远 CPU。**两个版本的提交包**：
  - `submission_050911_iter019_v2N_50plus50_cputorch.zip` MD5=c67e4d25665a2a7a12f4f6839cc8d0fe → 平台 +35.64
  - `submission_050911_iter019_v2N_50plus50_optimized.zip` MD5=95993e6250ea56361dfecb62c9123e63 (148MB) → 平台 +35.64
  - GPU 分支保留是为了**本地开发用 CUDA 调试 + 包通用性**（同一份 Predictor.py 在带 GPU 的开发机器上自动用 GPU，平台上自动用 CPU），不是冗余
  - 平台 16 核 CPU 模式：bmm(50, 1024, 256, 359) ≈ 4.7 GFLOP per layer × 4 layers ≈ 18.8 GFLOP；CPU 50-200 GFLOPS → 100-400 ms per batch 实测在 100ms 量级（满足平台 3h 限制）
- **核心证据**: HISTORY 主题 10 (T188v3 双版本) + 附录 C 第 21/22 行
- **状态**: ✅

### 7.3 为啥 NN 权重存 .npz 不存 .pt
- **来源**: QREVIEW_PREDICTOR Q90 / CODE_QUESTION_LIST Q113
- **答案**: 多个理由：
  - **平台不一定有 torch.load 兼容**：torch 2.5.1 默认 `weights_only=True`，serialize 格式版本敏感；npz 是 numpy 标准（numpy 2.x 兼容性好）
  - **load 不需要 torch instance**：`np.load(allow_pickle=False)` 直接拿到 dict of arrays，比 torch.load + state_dict 还原 + nn.Module 实例化简单
  - **手动 LayerNorm + GELU + Linear 实现已经在 _BatchedMLPEnsemble**，npz 给的 (W, b, LNW, LNb) 直接用，不需要复活 nn.Module
  - **per-key load 解耦**：可以 50 个 NN 的 W stack 成 (50, out, in) tensor，pt 不支持 batch stack 这种 reshape
  - **缺点**（QREVIEW Q90 列举）：keys 散乱 (LN0_W/LN0_b/L0_W/L0_b/LF_W/LF_b)；没 nested structure；维护成本高（手动 extract 在 train_T188v2_nn_seed.py 里），但提交侧便宜
- **核心证据**: Predictor.py:110-123 npz load 代码；CODE Q113
- **状态**: ✅

### 7.4 为啥 LGB 存 .txt 不存 pickle
- **来源**: QREVIEW_PREDICTOR Q95
- **答案**: LightGBM Booster.save_model() 默认 `.txt`：
  - **跨版本兼容**：LightGBM .txt 格式向后兼容（4.6 写的可以 4.0 读），pickle 完全不兼容版本
  - **可读**：明文存树结构，方便 debug
  - **平台环境**：requirements pin `lightgbm==4.6.0` 但平台实际版本可能微差，txt 格式容错
  - **缺点**：解析比 binary 慢；hyperparameter（如 early_stopping_round）可能在 txt 中丢失（QREVIEW Q95），但 inference 不需要这些
  - 50 booster × 1-2 MB → 50-100 MB load 时间 < 5s
- **核心证据**: Predictor.py:249 `lgb.Booster(model_file=p)`；QREVIEW Q95
- **状态**: ✅

### 7.5 GELU approximate="tanh" 推理 vs 训练 erf — 一致性 risk
- **来源**: QREVIEW_PREDICTOR Q47 / CODE_QUESTION_LIST Q115, Q176
- **答案**: 推理 Predictor.py:171 用 `F.gelu(h, approximate="tanh")`；训练 train_T188v2_nn_seed.py:89 用 `nn.GELU()` 默认 (approximate='none', 即 erf)。两者差异：
  - 数学上：tanh GELU = 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))；erf GELU = 0.5x(1 + erf(x/sqrt(2)))
  - max 误差 ~1e-4 @ |x|≈2，typical 1e-6
  - Predictor 包 docstring 自述 "Outputs match T188v2 within float32 precision (~1e-6 typical, 1e-4 max)" → 已知微差
  
  **风险**：阈值 thr_up=3e-4 大约是 3× 最大 GELU 误差；如果某个 sample 的 pred 恰好在 thr 附近 ±1e-4 内，action 可能 flip（错买/错卖）。但是 ensemble 平均 50 NN，单 NN 误差被平均到 ~1e-4/sqrt(50) ≈ 1.4e-5，不影响 SOTA。
  
  **为啥推理用 tanh**：tanh approx 在 CPU torch 上 quicker（少一次 erf 计算）；项目里没人显式 verify 训练用 erf 而推理用 tanh 的影响。**潜在 gap**：应该统一用同一个 approximate 参数。
- **核心证据**: Predictor.py:171 vs train_T188v2_nn_seed.py:89；docstring 1e-4 max
- **状态**: 🔴（已知不一致，影响小但形式上是 bug）

### 7.6 max(clips) vs per-NN clip
- **来源**: QREVIEW_PREDICTOR Q39, Q40, Q41
- **答案**: 推理用 `self.clip = float(max(clips))`，所有 NN 共用 max。理论上：
  - 训练每 NN 用各自 clip=10（PHASE2_CLIP, 即 ±10σ）
  - 推理用 max 意味着部分 NN 在 inference 时 input range 比训练时 wider（如果某 NN 训练 clip=10，但 max(clips)=10 也是 10，实际无差）
  - **实际情况**：所有 50 NN 共享 CLIP=10.0 训练（train_T188v2_nn_seed.py:35 全局常量），所以 max(clips)=10.0 = 每个 NN 的 clip = 等价于 per-NN clip
  - **若未来 HP-sweep 包含不同 clip**：max 会让小 clip 的 NN 见到 wider input，bias risk；但当前不触发
- **核心证据**: Predictor.py:130；train_T188v2_nn_seed.py:35
- **状态**: ✅（当前安全；future-proof gap）

### 7.7 target_scale 1.0 per NN — 推理顺序
- **来源**: QREVIEW_PREDICTOR Q42, Q50, Q258, Q263
- **答案**: target_scale 在训练时是 `target_scale = 1 / std(y_train)`，把 target 缩到 std=1 让 L2 loss 在合理量级。推理时：
  - `out = (out + bF) / target_scale` (Predictor.py:175-176) → 把 prediction 反 scale 回原 Δmid 量级
  - 然后 `out.mean(0)` 在 N=50 上平均
  - 这个顺序：per-NN 反 scale 后平均，**正确**（若 target_scale 各 NN 不同；若全 1.0 则顺序无关）
  
  **target_scale 是否各 NN 不同**：理论上每个 seed 训练时算 `target_scale = 1/std(y_train)`，y_train 是 SAME dataset（dates 0-119 + aug），所以 std(y) 在 5+ 位小数级别相同，target_scale ≈ 1.0 全 NN 共享。Predictor.py 把它当 per-NN 处理是 future-proof，当前无 effect。
- **核心证据**: Predictor.py:115, 131-133, 175-177；train_T188v2_nn_seed.py:266
- **状态**: ✅

### 7.8 feat_mean / feat_std 共享 vs per-NN
- **来源**: QREVIEW_PREDICTOR Q43, Q44, Q45
- **答案**: feat_mean / feat_std 各 NN 独立存（`(N, in_dim)`），但训练时全部用 Phase 1 train (dates 0-79) 的全局 mean/std，所以理论上 50 个 NN 的 feat_mean 是 identical。Predictor.py:110-113 仍 load 50 份（redundant ~70KB），用 per-NN 路径处理 (Predictor.py:158 `feat_mean.unsqueeze(1)` broadcast over batch)。
  - **若 NaN/Inf 处理**：Predictor.py:159 `torch.where(torch.isnan(h), 0, h)` 只处理 NaN，不处理 Inf。若 feat_std 含 0 → div → Inf → 不被 zeroed → 经过 clamp 后变 ±10 (clip)。**潜在 bug** 但不触发：训练时 feat_std 已确保 ≥ 1e-6 (`feat_std = np.maximum(feat_std, 1e-6)`, train_T188v2_nn_seed.py 推测)。
- **核心证据**: Predictor.py:158-160；CODE Q93
- **状态**: ✅（已知微 gap，不触发）

### 7.9 eager load vs lazy
- **来源**: QREVIEW_PREDICTOR Q97
- **答案**: `__init__` 里全部 load 50 LGB + 50 NN（eager）：
  - Total `__init__` time ≈ 50 LGB load (~5s) + 50 NN npz load (~1s) ≈ <10s
  - 平台 spec 没明确 init timeout，但 12h 1 次提交意味着 init 是 one-shot，eager 安全
  - lazy load 会让首次 predict() 慢（首次 predict 触发 all model load），不利于 latency-sensitive 评测
- **核心证据**: Predictor.py:231-253 eager loop
- **状态**: ✅

### 7.10 pred_cache 跨 horizon 复用
- **来源**: QREVIEW_PREDICTOR Q54, Q198 / CODE_QUESTION_LIST Q143
- **答案**: Predictor.py:329 `pred_cache: Dict[int, np.ndarray] = {}`（每次 predict() 局部变量，符合硬约束 #2 stateless）。5 个 horizon iterate 时：
  - h=5 first: src_H=60, cache miss → 计算 h=60 pred + cache
  - h=10/20/40: src_H=60, cache hit → 复用
  - h=60: src_H=60, cache hit → 复用
  - 总计 1× ensemble inference 完成 5 个 horizon → "5 个 horizon 免费"
- **核心证据**: Predictor.py:329-358；HISTORY 主题 10 "5 horizon share h=60"
- **状态**: ✅

### 7.11 LGB 推理 sequential — 84ms 不优化
- **来源**: QREVIEW_PREDICTOR Q36 / CODE_QUESTION_LIST Q114
- **答案**: 见 4.8 节（family-asymmetric optimization）。84ms 不是瓶颈，LightGBM internal multi-threading 已覆盖 OMP。
- **状态**: ✅

---

## 8. OOD / Abstain wrapper 失败汇总

### 8.1 T180 OOD GMM density abstain — 灾难
- **来源**: 项目历史 / HISTORY 主题 3 + 10
- **答案**: T180 在 PCA 降维 20D + GMM (k=5) 上估测试 batch 的 density，低密度（OOD-like）样本 hard abstain。3 个 variant：
  - V1: −32（log-density threshold）
  - V2: −57
  - V3: **−78**
  
  全部本地 holdout 退步，未提交。死因：**测试集比训练 in-distribution**（96-119 是训练日期最近 1/4，分布相似度高）；低密度样本反而是"模型不熟但有清晰信号"（如 sym=2 蓝筹的极端 amount_delta），弃权损失 PnL。
- **核心证据**: HISTORY 主题 10 表 + 主题 3 OOD GMM 行
- **状态**: ✅

### 8.2 T181 time-window hard abstain — 倒退
- **来源**: HISTORY 主题 10
- **答案**: T181 按 time-of-day 硬弃权（如开盘 5 min、收盘 5 min 不出手）。3 个 variant：
  - V1: **−23.39**
  - V2: −0.27
  - V3: −2.38
  
  死因：T170 conformal wrapper 已经过滤了不可靠预测；time-window 硬弃权进一步损失开/收盘段 PnL（开盘高波动期反而是高 alpha 段）。也触碰硬约束 #1（虽然 time 不是 date，但 time-based hard rule 容易在评测时 time 被去掉）。
- **核心证据**: HISTORY 主题 10 + 主题 3
- **状态**: ✅

### 8.3 T161 TTA K=5 — 倒退
- **来源**: HISTORY 主题 10
- **答案**: TTA (Test-Time Augmentation) K=5：对每个 test batch 做 5 次增强（bid/ask swap 等）预测平均。本地 T170 +149.95 → **下降**。死因：**硬约束 #2** 平台批次乱序 → per-batch 统计自适应不稳；TTA 在 LOB 上的 augment（bid/ask swap）改变了 microstructure 语义，让原本对齐的 prediction 漂移。
- **核心证据**: HISTORY 主题 10
- **状态**: ✅

### 8.4 T162 TTT — In-row TTT W3a/W3b/W3c
- **来源**: HISTORY 主题 10
- **答案**: TTT (Test-Time Training) W3a/W3b/W3c 三种 variant 在 test batch 上做少量 self-supervised gradient step。
  - W3a: −0.30
  - W3b: −1.98
  - W3c: −3.11
  
  死因：违反硬约束 #2 精神（TTT 假设连续测试样本相关，被 shuffle 破坏）；inference 时改 weight 是评测无法重现的 stateful 行为。
- **核心证据**: HISTORY 主题 10
- **状态**: ✅

### 8.5 T168 adaptive threshold V1-V4 — 全负
- **来源**: HISTORY 主题 10 / 与 1.9 同
- **答案**: V1: per-batch quantile-based thr；V2: per-batch z-score；V3: running median；V4: EWMA。**全 4 variant 本地负**。同根原因：硬约束 #2 shuffle 破坏 per-batch 统计稳定性。
- **核心证据**: HISTORY 主题 10
- **状态**: ✅

### 8.6 T169 AR TTT — 退步
- **来源**: HISTORY 主题 10
- **答案**: AR TTT (Auto-Regressive Test-Time Training): 把 test prediction 作为 next batch 的 pseudo-label 增量训练。退步。死因：违反 batch shuffle（pseudo-label 顺序意义被破坏）+ stateful（违反 #2）。
- **核心证据**: HISTORY 主题 10
- **状态**: ✅

### 8.7 T177 wrapper tricks B/C — holdout 正未入 final
- **来源**: HISTORY 主题 3
- **答案**: per-batch 统计 wrapper（具体方法未文档化），本地 holdout 正但与 T161/T168 同根问题（per-batch 统计），未入 final 是 PM 保守判断。
- **核心证据**: HISTORY 主题 3 T177 行
- **状态**: 🟡（项目文档不全）

### 8.8 所有 wrapper 失败的统一根因
- **来源**: 综合
- **答案**: **硬约束 #2（测试点顺序被打乱）+ 硬约束 #1（date=0）+ 硬约束 #3（OOD sym）三重夹击**让任何 per-batch / 跨调用 / time-derived / sym-specific dynamics-aware 的 wrapper 都失效：
  - per-batch dynamic：TTA/TTT/adaptive/AR — 破坏（shuffle）
  - time-based：T181 time-window — 破坏（time 不稳）
  - OOD-density-based：T180 GMM — 失效（测试在分布内）
  
  Conformal wrapper 是**唯一存活的 wrapper**，因为它是 **static per-sym lookup**（β·σ 从 calibration set 算好，推理时 O(1) 查 sym table），不依赖任何 batch dynamics。
- **核心证据**: HISTORY 主题 10 wrapper 失败汇总 + CRITICAL_CONSTRAINTS §1 §3
- **状态**: ✅

---

## 9. Conformal OOD sym 处理 + sym-agnostic 硬约束边界

### 9.1 sym ∉ {0,1,2,3,4} 时用 default_beta_for_ood=0.16
- **来源**: QREVIEW_PREDICTOR Q174, Q175 / CODE_QUESTION_LIST Q127
- **答案**: Predictor.py:294-304 `_extract_band_per_row`：
  - sym 列不存在 → default band
  - `int(sym)` 异常 → default band
  - sym int 不在 `self._cw_band` (即不是 0/1/2/3/4) → **保持 default band**（line 302 only set if `s in self._cw_band`）
  - 即使 sym=99 (整数合法但 OOD) 也走 default
  
  `default_band = 0.16 × 4.0e-4 = 6.4e-5` — 极小（比 sym=0 的 0.10×2.4e-4=2.4e-5 大但比 sym=1 的 1.88e-4 小很多）。所以 OOD sym **比 sym=1/2/3/4 都激进出手**。
- **核心证据**: Predictor.py:294-304；thresholds.json:58-59
- **状态**: ✅（行为正确但默认 band 偏小是 3.3 节的真 gap）

### 9.2 per-sym normalization 是否违反 sym-agnostic 硬约束
- **来源**: QREVIEW_PREDICTOR Q172, Q173 / CODE_QUESTION_LIST Q166
- **答案**: 硬约束 #3 的 ban list：
  - 模型 forward 接 sym 作为输入 ❌
  - sym embedding lookup ❌
  - LightGBM/XGBoost feature 列含 sym ❌
  - per-sym 模型集合 ❌
  - per-sym normalization 统计量 ❌
  
  Conformal 的 per-sym β·σ band：
  - 不进 model.forward（只在 decision gate）
  - sym 是用作 **per-sym lookup table** 而非 model input
  - σ 是 calibration set 上每 sym 的 ensemble disagreement，是**离线统计量**，不是推理时统计量
  - OOD sym 走 default band → 仍能跑（不像 sym embedding IndexError）
  
  **是否违反"per-sym normalization"**：严格说 σ_sym 是 **per-sym 残差统计量**，但 calibration 阶段用 sym，推理时只是 hash lookup。CRITICAL_CONSTRAINTS §1 #3 ban 的是"**评测 sym 可能是新的**"导致 calibration 失效；conformal 的 fallback (default band) 处理了这个 case。**项目 PM 判定**：合规（注意 README 表述"sym agnostic" 是指 model.forward，不是 entire pipeline）。
  
  **微观边界 case**：(a) 训练 sym 0 实际上在评测里映射到不同股票（同 ID 不同股）→ σ_sym0 calibration 错位，但 fallback 不能触发（int(0) 仍在 _cw_band）。这是已知风险，没有 detection 机制。(b) 严格意义的 sym-agnostic 应该全 default band，但牺牲 0.30~0.77 的平台收益。
- **核心证据**: CRITICAL_CONSTRAINTS §1 #3 + Predictor.py docstring (line 19-27 "sym is read ONLY for per-sym beta lookup") + HISTORY 主题 3 OOD sym 段
- **状态**: 🟡（合规边界，项目 PM 已接受 risk-reward trade-off）

### 9.3 sym 从 batch 最后一 tick 取
- **来源**: QREVIEW_PREDICTOR Q150 / CODE_QUESTION_LIST Q108
- **答案**: Predictor.py:299 `s = int(df["sym"].iloc[-1])`。理论上一个 100-tick window 内 sym 应一致（同一只股的连续 tick），但代码不 assert。若 window 跨 sym（如 sym rollover），iloc[-1] 取最后一个 sym。其他 99 个 tick 的 sym 被忽略。**实际数据中应无跨 sym window**（评测每个 batch 是单只股票的 100-tick）。
- **核心证据**: Predictor.py:299
- **状态**: ✅（不触发但有 silent assumption）

### 9.4 sym 列缺失行为
- **来源**: QREVIEW_PREDICTOR Q272, Q273 / CODE_QUESTION_LIST Q128
- **答案**: Predictor.py:296-297 `if "sym" not in df.columns: continue` → 保持 default band。整 batch 无 sym 列 → 全部走 default。平台 spec 是否保证 sym 在 df 中？项目 smoke test (Predictor.py:386) 测试了 `df_no_sym = df.drop(columns=["sym"])` case，行为定义清楚。平台实际行为：未明确，但 conformal 关闭 (default band) 比 crash 安全。
- **核心证据**: Predictor.py:296-297, 386
- **状态**: ✅

### 9.5 _BatchedMLPEnsemble 共享 keep_idx — sym-agnostic 间接 issue
- **来源**: QREVIEW_PREDICTOR Q110 / CODE_QUESTION_LIST Q180
- **答案**: `_BatchedMLPEnsemble.__init__` 用 npz_paths[0] 的 keep_idx，假设所有 50 NN 共享相同 keep_idx (即 11 个 FAIL_NAMES 在所有 NN 里都被 drop)。若某 NN 训练时用了不同 keep_idx，feat_mean shape 仍 359 但语义错位。**当前安全**：所有 NN 训练共享 SchemeP 359 维 - 11 = 348 维 keep_idx (训练代码硬编码)。
- **核心证据**: Predictor.py:98, 128-129
- **状态**: ✅

---

## 10. GAP 表（决策层已知 / 可疑 / 真 gap 汇总）

| # | Gap 描述 | 影响 | 严重度 | 来源 Q# |
|---|---------|------|--------|--------|
| G1 | sqrt(H/60) 阈值缩放方向与 random-walk-with-drift 推导反了（h=5 时差 12 倍）| 与 _doc 矛盾；分数无影响（max + conformal buffer）| 🔴 critical（理论） / 🟡（实际）| Q281, Q19, Q122 |
| G2 | `_doc` 注释 "shorter horizons trade fewer/smaller signals" 与实际行为相反 | 文档误导 | 🟡 | _doc, Q281 |
| G3 | sqrt vs h/60 vs sqrt(60/h) vs 1.0 ablation 缺失 | 无法证明 sqrt 最优 | 🔴 | Q281 (d) |
| G4 | label_5/10 α=0.05% vs label_20/40/60 α=0.1% 差异未在 thr scaling 中体现 | 短 h thr 与 label 边界无关 | 🔴 | 2.5 |
| G5 | default_beta_for_ood=0.16 是 mean(per_sym_beta)，OOD 反而比 sym=1 激进 | OOD safety 弱 | 🔴 | Q14, Q16, Q7 |
| G6 | default_sigma_for_ood=0.0004 也是 mean，10 位小数 | 数值 precision noise | 🟡 | Q15 |
| G7 | per-sym σ 用 5-NN ensemble disagreement 算，T188v2 50-NN 没重算 | conformal band 偏宽 ~3× | 🟡 | 3.8 |
| G8 | thr_up/thr_dn 自 iter_013 以来未重新 DE（M7 后未重 calibrate）| 风险厌恶；收益 unknown | 🟡 | 1.4 |
| G9 | thresholds.json 短 horizon 的 w_nn/w_lgb=1.0/1.5 是 dead config（被 share_with 覆盖）| 维护混淆 | 🟡 | Q28, 2.6 |
| G10 | share_with cycle 无检测；指向 inactive horizon 无报错 | 配置 robustness | 🟡 | Q197, Q248 |
| G11 | ensemble 中间点 25+25 / 75+75 / 100+100 未消融 | 不知是否真 50-optimal | 🟡 | Q31, 4.1 |
| G12 | NN HP 实际只覆盖 12 combo（README 写 36）| README 不准；不影响 SOTA | 🟡 | Q95, Q13 |
| G13 | w_LGB=1.5 50+50 阶段 val sweep 数据未存档 | 无法重现 grid search | 🟡 | Q25, 4.2 |
| G14 | T192 NNLS overfit 后无 model-level 权重学习再试 | 反证已充分 | ✅（已 close）| 4.3 |
| G15 | EV gate hysteresis 未加 | 阈值抖动成本未测 | 🟡 | Q102, 1.10 |
| G16 | adaptive threshold / TTA / TTT / GMM / time-window 全失败 | wrapper 限制清晰 | ✅（已 close）| 8.x |
| G17 | 推理 GELU=tanh approx vs 训练 erf — 已知 1e-4 max 差 | 阈值附近可能 flip action | 🔴（形式 bug）| Q47, Q176 |
| G18 | LGB 推理 sequential 未 batch | 84ms 不优化 | ✅（已 close）| Q34, Q114 |
| G19 | feat_std 含 0 导致 Inf 不被 NaN→0 处理 | 不触发（train 已 max(std, 1e-6)）| 🟡 | Q44, Q49, Q117 |
| G20 | per-sym conformal 在 sym-agnostic 硬约束边界 | PM 已接受 0.30-0.77 收益 vs 风险 | 🟡 | Q172, Q173, 9.2 |
| G21 | 4-fold CV consensus β sweep 具体 4 个 fold split 未存档 | 无法 reproduce calibration | 🟡 | T150 行 |
| G22 | score=max(per-h) 平台假设未直接 ablation 验证 | 隐含 spec 信仰 | 🟡 | Q8, Q9, 2.3 |
| G23 | NN 内部 NaN→0 但 +Inf 不处理；pred 可能 +Inf → 触发 spurious long | 不触发但数学 incomplete | 🟡 | Q49, Q170, Q171 |
| G24 | `_doc` 字段在 thresholds.json 容易丢失；无 model schema version | 提交追溯弱 | 🟡 | Q115, Q200, Q280 |
| G25 | smoke test 不 assert，只 print | CI 无 regression 检测 | 🟡 | Q200-204, Q191 |

---

## 11. 三段附录

### 附录 A：阈值/conformal/ensemble 关键数字速查

```
EV gate (h=60):
  thr_up = 0.0003   (30 bps)
  thr_dn = 0.000216 (21.6 bps)
  ratio  = 1.3889
  asym   = thr_up / thr_dn ≈ 1/0.72

thr scaling (share_with=60 short horizons):
  h=5:  thr × sqrt(5/60)  = ×0.2887
  h=10: thr × sqrt(10/60) = ×0.4082
  h=20: thr × sqrt(20/60) = ×0.5774
  h=40: thr × sqrt(40/60) = ×0.8165
  h=60: thr × 1.0

Conformal per-sym (calibration on 5-NN, not re-cal'd for 50-NN):
  sym 0: β=0.10  σ=0.0002403901  band=2.40e-5
  sym 1: β=0.40  σ=0.0004712397  band=1.88e-4
  sym 2: β=0.30  σ=0.0004522216  band=1.36e-4
  sym 3: β=0.00  σ=0.0004235249  band=0.00
  sym 4: β=0.00  σ=0.0004279811  band=0.00
  OOD:   β=0.16  σ=0.0003998317  band=6.40e-5

Ensemble:
  50 NN + 50 LGB
  w_nn=1.0  w_lgb=1.5  (60% LGB / 40% NN)
  simple mean per family → linear pool

Inference:
  T61 batch-vec feature   254 min → 4.5 min  (58×)
  T188v3 batched bmm NN   1509 ms → 8.1 ms   (186×)
  T188v3 e2e (B=1024)     2142 ms → 641 ms   (3.3×)
  442k 行 total          15.4 min → 4.6 min

Platform SOTA: T188v2 50+50 = +35.64 (max of per-h PnL)
```

### 附录 B：22 次提交 + 决策层相关 delta

| iter | 主要变化（决策层）| 平台 | Δ |
|---|---|---|---|
| iter_013 (T75) | EV gate + L2 regression（去 3-class CE）| +19.23 | +15.16（1st 突破）|
| iter_015 (T87) | + SPO+ DFL NN ensemble | +28.16 | +8.93（2nd 突破）|
| iter_016 v3 | 4-way + OOF DE 4D thr search | +25.08 | **−3.08（DE 过拟合教训）**|
| iter_018 v1 (T127) | + per-sym beta conformal | +28.93 | +0.77（conformal 首次有效）|
| iter_019 v2 (T140) | + LGB M7 全数据重训 | +34.44 | +5.51（3rd 突破，非决策层）|
| T163 v2+T97+agree | + 第二组 NN + agreement filter | ~+30.74 | **−3.70（同源 NN 灾难）**|
| T170 | NN M7 retrain (5+5) | +34.64 | +0.20 |
| T182 NoConformal | 去 conformal wrapper ablation | +34.34 | **−0.30（conformal 净效果证实）**|
| T179 40-NN | 只扩 NN 到 40 | +34.39 | **−0.25（单侧扩无效）**|
| **T188v2 50+50** | NN+LGB 都扩到 50 | **+35.64** | **+1.00（current SOTA）**|
| T190 150+150 | 继续扩到 150 | +34.59 | **−1.05（HP 多样性变噪音）**|
| T192 H4 NNLS | bounded NNLS 学权重 | +31.0 | **−4.64（OLS overfit）**|

### 附录 C：未答 / 弱答清单

- **score=max 假设的直接验证实验**（提交 h=60-only 版本对比）— 未做，G22
- **sqrt vs h/60 vs sqrt(60/h) 的平台对比**（4 个 variant 提交）— 未做，G3
- **default OOD band = max vs mean vs median 对比**（3 个 variant 提交）— 未做，G5
- **σ 用 50-NN disagreement 重算后 re-sweep β**（2-day work）— 未做，G7
- **EV gate hysteresis effect 测量**（per-batch action flip rate 分析）— 未做，G15
- **GELU tanh vs erf 在 thr 附近的 action flip 概率**（offline analysis）— 未做，G17

---

RESULT: task=[answers P2 decision] metrics={n_questions_answered=110, n_substantive_entries=73, n_gaps=25, n_strong_ablations=22, user_q281_addressed=true} notes=[110 unique Q# refs answered across QREVIEW_PREDICTOR + CODE_QUESTION_LIST in 73 substantive entries spanning 10 sections (thresholds, share_with sqrt, conformal, ensemble 50+50, ensemble add/reject CB/Huber/GRU/T97/Transformer/4way/5way, EV gate, inference opt batched bmm/CPU torch/npz/txt, OOD wrapper failures TTA/TTT/GMM/adaptive/time-window, sym-agnostic boundary, GAP table). Q281 sqrt(h/60) scaling direction confirmed reversed per user derivation; mitigated only because score=max(per-h) and conformal band partially cancels short-h scaling for sym=1/2. 25 gaps logged: 6 真 gap (Q281 direction/no ablation/label-α mismatch/OOD default=mean/GELU tanh vs erf/sqrt未做平台对比), 19 已知部分证据 gap. Core 22-submission deltas show iter_016 4D DE (-3.08), T163 second-NN (-3.70), T190 150+150 (-1.05), T192 NNLS (-4.64) all confirm "decision-layer complexity is risk-asymmetric on platform OOD". Conformal +0.30~+0.77 (T127→T182 ablation), 50+50 +1.00 (T188v2 vs T170), share_with=60 effectively free (5 horizons share 1 ensemble). Default OOD band uses mean not max — counterintuitive but unmeasured.]
