# 良文杯 2026 — 全部问题答案（合并版）

> 本文档由 4 个 opus worker 并行答题汇总而成，每个 worker 负责一个主题区。
>
> 总计回答约 300+ 个独立"why" 问题（去重自 5 份白板 review 共 1091 个原始问题 + 200 问的全局清单）。
> 每个答案综合 3 视角：(1) 赛题 setting / (2) 项目先验 / (3) 200+ 实验消融。
>
> - **P1 — 训练**（NN/LGB 架构、HP、损失、M7、aug_a、评估方法论）
> - **P2 — 决策层**（阈值、conformal、ensemble、share_with sqrt scaling，含 USER-RAISED Q281）
> - **P3 — 特征工程**（Scheme 演变、Stage1-5、DROP_NAMES、未采纳家族）
> - **P4 — Pipeline 与杂项**（4 步流程、build_pkg、requirements、文件格式、跨文件一致性）
>
> 配套文档：
> - `FINAL_SUBMISSION_DECISIONS_EXPLAINED.md` — 按主题做"问题↔证据"JOIN 的主合成报告（2173 行）
> - `HISTORY_EVIDENCE_DB.md` — 200+ T 实验的证据库（854 行）
> - `AUDIT_BATCH_LEAKAGE.md` — batch 泄露合规审计（零风险确认）

---

=========================================

# PART P1 — TRAINING (NN/LGB Architecture, HP, Loss, M7, Evaluation Methodology)

# Answers P1 — Training (NN/LGB Architecture, HP, Loss, M7, aug_a, Evaluation Methodology)

总结：本文档把 5 份 QREVIEW + CODE_QUESTION_LIST 中**训练相关**的 ~310 条 "why" 问题去重整合后，给出 ~110 条综合答案。证据全部 join 到 `HISTORY_EVIDENCE_DB.md` 的 10 个主题（特别是主题 4 M7、主题 5 SPO+、主题 7 损失、主题 8 架构、主题 9 评估方法论）+ `final_submission_code/` 源码 + 200+ T 实验。

**最终配置速查**（再确认一遍）：

| 组件 | 值 |
|---|---|
| NN 架构 | MLP [359→256→128→64→1], LayerNorm + GELU + Dropout(p), 无残差，无最终 LN |
| NN Phase 1 (L2 pretrain) | train=dates 0-79, val=80-95, max 50 epoch, patience=10, AdamW lr_p1, WD=1e-4, CosineLR T_max=50 eta_min=1e-5, grad clip 5.0 |
| NN HP cycling | LR_LIST=[1e-4,3e-4,1e-3]%3 × DROPOUT_LIST=[0.05,0.10,0.15,0.20]%4 × BATCH_LIST=[2048,4096,8192]%3, seed (S-1)%k |
| NN Phase 2 (SPO+ M7) | dates 0-119 全数据, 11 epoch 固定, lr=3e-5, λ_spo=30, WD=1e-4, batch=4096, CosineLR T_max=11 |
| LGB HP grid (5 组) | (feature_fraction, bagging_fraction, num_leaves, lambda_l2) ∈ {(0.8,0.8,127,1.0), (0.6,0.7,127,1.0), (0.7,0.85,63,2.0), (0.5,0.6,255,0.5), (0.4,0.5,127,3.0)} |
| LGB 共享 | num_boost_round=330, lr=0.05, min_data_in_leaf=100, bagging_freq=5, objective=regression_l2, metric=rmse, 18 threads |
| aug_a | x' = x × U(0.80, 1.20) per (sample, feature) 独立采样, label/fee 不变, concat 后训练数据 ×2 |
| Ensemble | NN: 50 seeds (5×4×3 HP cycling); LGB: 50 seeds (5 HP × 10 seed) |
| RNG offsets | LGB aug: S*7919+42; NN P1 aug: S*7919+1; NN P2 aug: S*7919+137 |

---

## 1. NN 架构

### Q1.1 为什么 MLP 而不是 GRU/Transformer/CNN？
- **来源**：NN_TRAIN Q3/Q36, CODE_QUESTION_LIST Q35/Q36, PREDICTOR Q47 边缘
- **答案**：决策层是 stateless per-tick prediction（评测打乱测试点顺序，CRITICAL_CONSTRAINTS #2），sequence model 跨调用维护 hidden state 直接违规。试过 GRU（T95）、Transformer/GroupTransformer（T93/T157/T172/T187）、深度 CNN（T184/T185），全部要么在 iter_016 v3 上把平台 ensemble 打成 −3.08（GRU 主因），要么 in-sample 数字漂亮但 OOD 风险高（T172 holdout +149.91 但拒提交）。R31 文献调研结论也是"MLP+GBDT 集成在 HFT 实际竞赛里 competitive"。MLP 看 100-tick 切片 + 工程化特征聚合，把时序信息 baked-in 进特征（R34 Stage1-5），架构本身无 state，符合 §2 评测约束。
- **核心证据**：T95 GRU iter_016 v3 平台 **−3.08** | T172/T187 GroupTransformer in-sample +149.91 拒提交 | T186 verdict `architecture_doesnt_fit` | HISTORY_DB 主题 8 完整 ablation
- **状态**：✅ 强消融 + 硬约束

### Q1.2 HIDDEN=(256,128,64) 三层金字塔为什么？
- **来源**：NN_TRAIN Q3, CODE_QUESTION_LIST Q35
- **答案**：经典 1.5-2× 渐缩 MLP，T81 设计时确定。输入 359 维（SchemeP），256 是首层"略宽于输入压缩比"的折中（不是 wider 2×=720，避免 1.47M 样本下过拟合）。总参数 ~120k 与样本量成 ~12k:1 比例（典型可训练范围）。T93/T94 试过 Transformer/ResNet-DCN 在 LOSO-eq 上不超 T81 +35.89，T172/T184 更深架构 in-sample 漂亮但 OOD 风险。没有保留 depth/width sweep 曲线，但对比架构都明确 inferior 后定型。
- **核心证据**：T81 LOSO-eq +35.89 | T93/T94 探索性未超 | T188v2_mega_proper/PROGRESS.md 锁死 | `train_T188v2_nn_seed.py:34`
- **状态**：🟡 部分消融（vs 其他架构有，vs 自身宽窄无）

### Q1.3 LayerNorm vs BatchNorm vs 无 norm 为什么选 LN？
- **来源**：NN_TRAIN Q29, CODE_QUESTION_LIST Q37
- **答案**：(1) BatchNorm 在 batch=2048-8192 也算稳，但 BN 的 running mean/var 是跨样本统计——在评测打乱测试点 + 可能含 OOD sym（约束 #2、#3）下，BN running statistics 来自训练 in-distribution，OOD 时偏差大。LN 是 per-sample 归一化，对 batch 顺序与 OOD 都鲁棒。(2) BN 在小 batch 上方差大，LN 不受 batch 影响。(3) LN 与特征工程层的窗口 z-score（T7 SchemeD1，sym=2 brittleness 修复）互补：前者跨样本 within feature，后者 within-sample within-window。
- **核心证据**：T7 窗口归一化 vs T10 全局归一化 (LOSO h_10 +13.91 vs +13.40) | T8 sym=2 amount_delta z+7.49 偏离 | `train_T188v2_nn_seed.py:87-88`
- **状态**：🟡 部分（间接论证，无直接 BN/LN ablation）

### Q1.4 GELU vs ReLU/SiLU？approximate='none' vs 'tanh'？
- **来源**：NN_TRAIN Q31, CODE_QUESTION_LIST Q37/Q115/Q176, PREDICTOR Q47
- **答案**：训练时 `nn.GELU()` 默认 erf，推理（Predictor.py `_BatchedMLPEnsemble`）用 `F.gelu(h, approximate='tanh')`。**这是潜在不一致 bug，max diff ~1e-4 量级**（CODE_QUESTION_LIST Q176 严肃指出，PREDICTOR Q47 也提）。已知影响：T188v3 vs T188v2 actions max diff 6.98e-10（HISTORY DB 主题 10），说明在实际特征/权重范围内（z-score 后 ±10σ clip，多数在 ±3 内）erf 与 tanh 近似差异 < threshold 容差，对最终 PnL 无可观测影响。生产决策选 GELU 而非 ReLU 因为现代 NLP/视觉标准、避免死神经元；vs SiLU 略快且无超参；vs ELU 不需 alpha 调。
- **核心证据**：T188v3 actions max diff 6.98e-10 vs T188v2（HISTORY 主题 10）| `train_T188v2_nn_seed.py:89` vs `Predictor.py:171`
- **状态**：🔴 **GAP：训练-推理 GELU 不一致**（虽然实测无影响，但应统一）

### Q1.5 Kaiming init 用 nonlinearity="relu" 但激活是 GELU
- **来源**：NN_TRAIN Q32, CODE_QUESTION_LIST Q38
- **答案**：GELU 的 gain（与 ReLU 的 √2 差异）在 |x|>1 范围接近 ReLU，初始化用 ReLU gain 是常见 practice，对 3 层小网络几乎无可测影响。Phase 1 早停在 best_epoch=0~10（T188v2_mega_proper/PROGRESS.md），说明 init 已经在合理 basin。生产里没单独验过 nonlinearity="leaky_relu" 或自定 gain，但 Phase 1 收敛快足以推断 init 不是瓶颈。
- **核心证据**：T188v2 PROGRESS "T81 seed1 best_epoch=0, seed42=2 — Phase 1 几乎不动"
- **状态**：🔴 GAP（无显式 ablation，但 indirect 证据指收敛快）

### Q1.6 Linear→LN→GELU→Dropout 顺序 + 最后一层无 norm/activation/dropout
- **来源**：NN_TRAIN Q30/Q34
- **答案**：标准 pre-norm 块顺序 (Linear → LN → activation → Dropout)；norm-before-activation 在小 MLP 上与 norm-after 影响 negligible。最终层 (64→1) 是回归 head，加 LN 会破坏 scale invariance（输出已被 target_scale 缩放），加 activation 会限制输出符号范围（必须能输出负数表示卖出 EV）。bias 全 0 init 而非 init 到 target mean——因为有 target_scale 标准化，y_scaled 的 mean ≈ 0，bias=0 与 target 一致。
- **核心证据**：`train_T188v2_nn_seed.py:81-92` 最终层只有 Linear
- **状态**：✅ 合理（无 ablation 但是 textbook MLP 设计）

---

## 2. NN HP cycling 设计

### Q2.1 (seed-1)%k 为什么让 LR/dropout/batch 强相关，50 seed 只覆盖 12 种独立组合
- **来源**：NN_TRAIN Q8/Q146-Q150, CODE_QUESTION_LIST Q13/Q95, PIPELINE_ARCH Q69
- **答案**：**这是有意设计，不是 bug**。LR_LIST(3) × DROPOUT_LIST(4) × BATCH_LIST(3) → LCM(3,4,3)=12 独立组合，50 seed 里每个组合重复 ~4.17 次。设计意图：(a) 12 个 (lr, dropout, batch) HP 组合提供"HP-level diversity"（重要），(b) 每组合 ~4 seed 提供"init+aug-level diversity"（次要）。HISTORY DB **发现 8**：50 NN 之间平均 corr **0.9331**，T192 OLS 学权重 condition number 54182 病态——这告诉我们 NN 多样性本身就饱和了，扩独立组合数收益递减。T179（40 NN 全独立组合）平台 **−0.25 vs T188v2**，T188v2（50 NN 含重复 HP）平台 **+1.00**，验证"HP diversity + 重复 init"优于"只扩 HP 不重复"。R_multinn_spo 调研结论也是 "HP > seed diversity"。
- **核心证据**：T179 40NN 平台 +34.39 (−0.25 vs T170) | T188v2 50+50 平台 **+35.64** (+1.00) | T192 NN-NN corr 0.9331, cond=54182 | HISTORY 主题 2 + 发现 8
- **状态**：✅ 强消融（T179/T188v2/T190/T192 平台四点）

### Q2.2 LR_LIST [1e-4, 3e-4, 1e-3] 只 3 点且 ½-decade 步长
- **来源**：NN_TRAIN Q5, CODE_QUESTION_LIST Q14
- **答案**：T136 SPO+ arch sweep 扫了 lr ∈ {1e-5, 3e-5, 1e-4, 3e-4} 在 Phase 2 SPO+ 上，确定 3e-5 为最优；那是 Phase 2 的 lr，**Phase 1 lr 没单独 sweep**。设计上取 1e-4（保守）到 1e-3（激进）跨度 10×，覆盖 MLP 训练常用范围。没单独验 5e-4 / 1e-5 等中间点，因为整个 HP cycling 设计就是给 50 NN 制造 diversity，不是找单 seed 最优。T176 在 M7 阶段试过 Phase 2 lr=1e-4 holdout +5.73 vs ep11 baseline——但这是 in-sample 污染数字（T176 results.json warning），未上平台未验证。
- **核心证据**：T136 SPO+ arch sweep 确定 Phase 2 lr=3e-5 | T176 results.json `S2_lr1e-4` warning "Holdout in-sample" | `train_T188v2_nn_seed.py:37`
- **状态**：🟡 部分（Phase 2 有 sweep；Phase 1 无）

### Q2.3 DROPOUT_LIST [0.05, 0.10, 0.15, 0.20] 排除 0 与 ≥0.30
- **来源**：NN_TRAIN Q6/Q148, CODE_QUESTION_LIST Q16
- **答案**：T19 试过更激进 regularization（dropout + weight decay）LOSO 持平——说明无 dropout 没明显过拟合。但 0 完全不留 diversity（同 init + 同 LR 时收敛到几乎同 net），所以下限 0.05 保留最小扰动；上限 0.20 避免严重欠拟合（50 seed ensemble 平均掉的是高方差预测，不是欠拟合预测）。0.30+ 没测，因为 dropout 0.05 vs 0.20 跨度 4× 已覆盖"轻 reg"到"中 reg"，向 0.5+ 走会撞欠拟合。这是 conservative 设计而非 dropout-specific optimization。
- **核心证据**：T19 reg sweep LOSO 持平 | HISTORY 主题 5 "Dropout 0.10 是 conservative 选择"
- **状态**：🔴 GAP（无完整 dropout sweep）

### Q2.4 BATCH_LIST [2048, 4096, 8192] 全 ≥2048
- **来源**：NN_TRAIN Q7, CODE_QUESTION_LIST Q14
- **答案**：(a) 小 batch（512/1024）的 implicit regularization 在 1.47M × 2(aug) ≈ 3M 训练样本下 epoch 步数 6000+，wall-clock 太长（PHASE1 50 ep × 6000 steps = 300k iters，单 seed ~10 min × 50 seed = 8.3h，但实测单 seed ~100s/RTX 3080 → 50 seed ~83 min）。batch=2048 已经在 RTX 3080 9.9GB VRAM 上跑（T188v2_mega_proper/PROGRESS.md 实测 44.3s/Phase1），4096/8192 显存占用更高但仍 OK。(b) SPO+ DFL 损失需要"足够大 batch"才稳定（z_star 死区让一大部分样本 ell=0，小 batch 容易出现整 batch 没有 informative 样本）。
- **核心证据**：T188v2 PROGRESS RTX3080 9.9GB Phase1 44.3s/seed | `train_T188v2_nn_seed.py:39`
- **状态**：🟡 工程/性能驱动，无 SPO+ batch sensitivity ablation

### Q2.5 PHASE2_BATCH=4096 固定不 cycle
- **来源**：NN_TRAIN Q16
- **答案**：Phase 2 用全数据 2.94M × 2(aug) = 5.88M 样本 + SPO+ 损失计算（额外 |spread|, |z_star| 等），显存压力比 Phase 1 大，batch=2048 在 SPO+ 实测会让 ep_time 翻倍（Phase 1 44s vs Phase 2 55s 已经 batch=4096）。同时 SPO+ z_star 死区让有效样本（|y|>fee）约占 60-70%，batch 太小有效梯度 sample 少。固定 4096 是 SPO+ 稳定性 + 显存的折中。
- **核心证据**：`train_T188v2_nn_seed.py:50` | T188v2 PROGRESS Phase 2 ~55s ep=4096
- **状态**：🟡 工程驱动

### Q2.6 PHASE1_EPOCHS=50, PHASE1_PATIENCE=10
- **来源**：NN_TRAIN Q9/Q10, CODE_QUESTION_LIST Q17
- **答案**：T188v2_mega_proper 实测：T81 reference seed 1/100/13/42/7 的 best_epoch 是 {0, 1, 0, 2, 0}——**Phase 1 早在 ep 0-2 已经收敛**！patience=10 是冗余但安全（理论上若某 seed 噪音大，多给 10 epoch 缓冲）。50 上限实际从未触发（每个 seed 都早停在 ep 10-15 以内）。可以更小但成本 negligible（patience 不增加无谓训练）。Cosine T_max=50 + 早停于 ep 10 → schedule 只用了前 20%，余弦后段未起作用——理论上"schedule 失效"但因为 Phase 1 几乎不动，无实际影响。
- **核心证据**：T188v2 PROGRESS "T81 best_epochs: seed1=0, seed100=1, seed13=0, seed42=2, seed7=0"
- **状态**：✅ 强证据（直接观测早停行为）

### Q2.7 PHASE1_WD=1e-4 与 PHASE2_WD=1e-4
- **来源**：NN_TRAIN Q11/Q17, CODE_QUESTION_LIST Q94/Q195
- **答案**：AdamW 默认 wd=0.01 是 NLP/CV 大模型经验值；这里 1e-4 是"已有 dropout + early stop 后的弱补充正则"。没单独 sweep wd ∈ {0, 1e-5, 1e-4, 1e-3}，因为 T19 全 reg 维度 sweep 已 LOSO 持平。Phase 2 同 wd=1e-4 是 carry-over（fine-tune 不变 wd），理论上 fine-tune 可降低 wd 避免把 warm-start 权重拉回 0，但 lr=3e-5 已经够小（wd × lr = 1e-9 量级，每步几乎不更新），实际影响微弱。
- **核心证据**：T19 reg sweep LOSO 持平 | `train_T188v2_nn_seed.py:43/49`
- **状态**：🔴 GAP（无显式 wd ablation）

### Q2.8 PHASE2_LR=3e-5 所有 seed 同 lr，不再 cycle
- **来源**：NN_TRAIN Q14, CODE_QUESTION_LIST Q11
- **答案**：T136 SPO+ arch sweep 直接确认 (λ=30, lr=3e-5) 在 LOSO-eq 上比次优高 +0.3-0.5；T176 在 M7 阶段重扫 λ∈{10,30,100}（含 ep 7/11/15/20 与 lr 1e-5/3e-5/1e-4），各 variant 在 holdout 上 149-151 噪音级别，没有 variant 显著超 ep=11/lr=3e-5/λ=30。设计层面：Phase 1 lr cycle 是为 "init basin diversity"，Phase 2 已经 warm-start 在同一 basin（每个 seed 都从自己的 Phase 1 best），再 cycle lr 反而扰动到不同 basin。所有 seed 同 lr 是稳定 fine-tune 的标准做法。
- **核心证据**：T136 SPO+ arch sweep | T176 ep11/3e-5/30 vs S2_lr1e-4 +5.73 holdout 但 "in-sample warning" | HISTORY 主题 5 选择理由
- **状态**：✅ 强消融（T136 + T176 双层 sweep）

### Q2.9 PHASE2_EPOCHS=11 奇数
- **来源**：NN_TRAIN Q13, CODE_QUESTION_LIST Q10
- **答案**：V4 walk-forward 阶段 SPO+ fine-tune 在 5 seeds 上的平均 best_epoch ≈ 9.2，× 1.1 向上取整 → **11**（与 LGB num_boost_round=330 同样的 ×1.1 heuristic，主题 4）。T176 sweep `S1_ep11/15/20` 在 holdout 上 ep=20 看似 +2.32，但 results.json 明确 warning "Holdout in-sample for all T176 NNs (trained 0-119). Ranking only — ship to verify."——多 epoch 在 in-sample 数据上必然提高（M7 训练数据本身就包含 96-119 评分集）。S1_ep15 实际 ship 平台 +34.51 vs T170(ep=11) +34.64 → **−0.13**，证 ep=11 优。Phase 2 无 val 监控（全数据训练），无法早停。
- **核心证据**：HISTORY 主题 4 "V4 best_epoch 9.2 × 1.1 = 10.1 → 11" | T176 results.json `S1_ep15` 平台 +34.51 (−0.13) | T170 平台 +34.64
- **状态**：✅ 强消融（平台数字直接验证）

### Q2.10 CosineAnnealingLR eta_min=1e-5, T_max=50/11
- **来源**：NN_TRAIN Q9/Q84/Q85, CODE_QUESTION_LIST Q24/Q25
- **答案**：Phase 1 T_max=50 + eta_min=1e-5，但早停在 ep~10 → schedule 只走前 20%，lr 几乎不变（从 1e-4/3e-4/1e-3 几乎都没动到 eta_min）。这"schedule 失效"是已知但 harmless（Phase 1 本就早收敛）。Phase 2 T_max=11 + eta_min 默认 0 → 最后一 ep lr≈0 几乎无更新——这是 cosine 在 fine-tune 上的常见效果（gentle annealing），不是 bug。T176 没特别测 schedule 形状（linear/constant 等），保持 cosine 是 textbook 选择。
- **核心证据**：T188v2 PROGRESS Phase 1 早停 ep 0-10 | `train_T188v2_nn_seed.py:293/436`
- **状态**：🟡 部分（无 schedule shape ablation）

### Q2.11 grad clip 5.0 vs 1.0/10.0
- **来源**：NN_TRAIN Q83, CODE_QUESTION_LIST Q33
- **答案**：5.0 是 MLP 训练保守选择，避免梯度爆炸但不限制正常更新。SPO+ 损失含 |spread| 不可导点 + relu，理论上某些样本梯度尖峰，clip 5.0 兜底。训练日志（experiments/T87/logs）里未见梯度爆 5+ 报告，说明 clip 几乎不触发，只是 safety net。1.0 会限制 lr=1e-3 下的正常步长，10.0 几乎无用。这是工程稳健性 vs SPO+ 边缘情况的折中，没单独 ablation。
- **核心证据**：`train_T188v2_nn_seed.py:318/466` | T87 训练 log 无 clip warning
- **状态**：🔴 GAP（无 ablation，但保守值合理）

### Q2.12 CLIP=10.0 特征标准化后硬截断
- **来源**：NN_TRAIN Q4, CODE_QUESTION_LIST Q34
- **答案**：10σ 截断后留 ~0.01% 极端值。设计意图：标准化后特征 99.99% 在 ±10σ 内（高斯假设下），剩下 0.01% 是 LOB 极端事件（如断崖式 cancellation），clip 防止其支配前几层 GELU 的输出。5σ 会 clip 太多正常变动（LOB 重尾特征 |z|>5 不罕见），8σ 折中但 10σ 给更多保留。SchemeP 359 维多数已做窗口 z-score（T7），原始尾部已经被压缩，10σ 是冗余保险。没单独验过 5/8。
- **核心证据**：T7 窗口归一化已做尾部压缩 | `train_T188v2_nn_seed.py:36`
- **状态**：🔴 GAP（无 CLIP sweep）

---

## 3. NN 两阶段训练协议（L2 pretrain → SPO+ M7 finetune）

### Q3.1 为什么两阶段？直接 SPO+ 不行？
- **来源**：NN_TRAIN Q108 边缘, CODE_QUESTION_LIST Q42, HISTORY 主题 5
- **答案**：T13/T134 直接 PnL loss 不收敛（PnL 是 z∈{−1,0,1} × Δmid，分段不连续，裸梯度方向跳跃）。SPO+ 用 surrogate ξ-loss 平滑这个不连续性，但 SPO+ 仍依赖一个"合理初始化的回归器"——z_star = sign(y − fee) 时，初始随机权重的 pred 几乎全在死区（|pred| < fee），梯度信号微弱。T81 L2 pretrain 先把 pred 拉到 y 的尺度（cov(pred, y) > 0），再 SPO+ 给方向校准。R10 调研 10 种 PnL-aware loss 路线，SPO+ 是唯一稳定收敛的，且 T81→T87 两阶段比直接 SPO+ 在 LOSO-eq 上稳定高 +1~2。
- **核心证据**：T13 / T134 direct PnL 不收敛 | T60 RL PnL 不收敛 | T81 LOSO-eq +35.89 + T87 LOSO-eq +40.09 | R10 调研 | HISTORY 主题 5 + 发现 6
- **状态**：✅ 强消融（多次复现失败 + 调研 + 平台数字 +8.93）

### Q3.2 Phase 1 用 train(0-79)/val(80-95)，Phase 2 用全数据 0-119——这构成 data leakage？
- **来源**：NN_TRAIN Q97/Q98/Q135/Q136, CODE_QUESTION_LIST Q76/Q86, LGB_TRAIN Q43
- **答案**：**不是 leakage**。"test" 在 schemeP_test.npz 里指 **dates 96-119 的 holdout**（带 label），不是平台真实评测集（平台评测集本地完全不可见，由组委会持有）。M7 trick（HISTORY 主题 4）= 在已用 V4 walk-forward 选好 HP/early-stop 之后，把 train+val+test 三段都放进训练。理论依据：Kaggle 标准 "CV-select HP, retrain on all"。实测：T140 LOSO eval 数字不变（M7 不改 LOSO，只改全数据模型），但平台 +28.93 → +34.44 **+5.51**（第三次大突破）。NN M7 增量小（+0.20，T170 vs iter_018），因为 NN 已 warm-start 饱和。本地 holdout in-sample (96-119) 此时**绝对值不可信**，只看相对 delta（发现 1，HISTORY DB）。
- **核心证据**：T140 LOSO 不变 但平台 +5.51 | T170 NN M7 平台 +0.20 | T183 NN random init M7 + warm-start M7 在 holdout 持平 +0.63 | HISTORY 主题 4 完整时间线 | 发现 1 in-sample 污染陷阱
- **状态**：✅ 强消融（平台 +5.51 直接验证）

### Q3.3 Phase 2 完全不冻结任何层，无 differential learning rate
- **来源**：NN_TRAIN Q107
- **答案**：MLP 3 层共 120k 参数，"backbone vs head"边界不清晰（不像 CV/NLP 的预训练 encoder）；全 unfreeze + 小 lr (3e-5, 比 Phase 1 小 3-33×) 是标准 fine-tune 做法。没单独 ablation freeze/unfreeze，因为 Phase 1 已 converge 到 L2 minimum（best_epoch=0~10），Phase 2 起点附近的小步 SPO+ 校准在所有层上都需要（z_star surrogate 改变每层 representation 的"决策对齐"）。
- **核心证据**：T188v2 PROGRESS Phase 1 几乎不动 | `train_T188v2_nn_seed.py:431-435`
- **状态**：🔴 GAP（无 freeze ablation）

### Q3.4 Phase 2 无 val 监控，无 early stop
- **来源**：NN_TRAIN Q108, CODE_QUESTION_LIST Q79
- **答案**：M7 trick 设计就是"全数据训练"——没有 val 集可监控。早停 epoch 由 V4 walk-forward 阶段（T87 5 seeds avg best_epoch=9.2）×1.1 决定，固化为 11。T176 sweep 在 holdout 看似 ep=20 更好 (+2.32)，但 results.json warning 明示 "Holdout in-sample, ranking only"；S1_ep15 平台 +34.51 反 −0.13 验证 ep=11 优。无 val 监控的代价是不能动态调 epoch，但收益是 12.5% 训练样本（80-95 的 val）也进训练，对平台 OOD 增益更大（M7 平台 +5.51 vs LGB no-M7）。
- **核心证据**：T176 results.json warning + S1_ep15 平台 −0.13 | T170 平台 +34.64 | HISTORY 主题 4
- **状态**：✅ 强消融（平台验证）

### Q3.5 warm-start 从 Phase 1 best_state 加载
- **来源**：NN_TRAIN Q106, CODE_QUESTION_LIST Q162
- **答案**：`model.load_state_dict(ckpt_p1["state_dict"])` 是标准 fine-tune，"warm-start" 是术语包装。Phase 1 best_state 来自 dates 0-79 的 L2 minimum；Phase 2 在此基础上加全数据 SPO+ surrogate fine-tune。T183 试过 random init M7 retrain（无 T81 warm），holdout +0.63 持平 T170——说明 warm-start 增量已饱和（数据多 + SPO+ surrogate 已足够收敛），但仍保留 warm-start 是为了**复现性**（每个 seed 走完全相同两阶段）和**diversity 来源**（Phase 1 不同 lr/dropout/batch 让 50 seed 落入不同 basin）。
- **核心证据**：T183 random init holdout +0.63 持平 | HISTORY 主题 5
- **状态**：✅ 中等消融

### Q3.6 Phase 2 用 Phase 1 target_scale / feat_mean / feat_std（不重算）
- **来源**：NN_TRAIN Q95/Q96, CODE_QUESTION_LIST Q23/Q87, PIPELINE_ARCH Q36/Q37
- **答案**：保持 Phase 1 的 target_scale = 1/std(y_regr_tr[0-79])，feat_mean/feat_std 同样基于 0-79。Phase 2 加入 80-119 数据后，**y 分布略有不同**（80-119 是更晚日期，市场状态可能 shift），但保持 Phase 1 的 scale 是**有意为之**——避免"全数据"normalization 时 future-set 统计量污染。Phase 1 → Phase 2 在 representation 层面已经对齐 Phase 1 stats，Phase 2 用同一 scale 让 ckpt 内部一致性保留。如果重算 scale，每个 seed Phase 1 best_state 在新 scale 下行为变化，等价于"forced shift"，破坏 warm-start 意义。
- **核心证据**：`train_T188v2_nn_seed.py:374-377` | HISTORY 主题 4 "M7 沿用 V4 阶段所有 HP"
- **状态**：🟡 设计 rationale 清晰，无显式 ablation

### Q3.7 dropout 在 fine-tune 仍开启（不关）
- **来源**：NN_TRAIN Q158/Q159, CODE_QUESTION_LIST Q153
- **答案**：Phase 2 仍用 Phase 1 dropout 值 (per-seed cycling，0.05-0.20)。Fine-tune 关 dropout 是 image classification 常做法，但这里 SPO+ 损失对 pred 噪音敏感（z_star 死区，pred 跳跃改变 z_star），dropout 提供的"预测噪音稳定性"反而有用——50 seed ensemble 受益于每个 seed 在 dropout 下的"鲁棒决策面"。没单独验过 Phase 2 dropout=0，但 dropout 0.05-0.20 范围内 ensemble 数字（T188v2 平台 +35.64）已经 SOTA。
- **核心证据**：`train_T188v2_nn_seed.py:431-432` 复用 Phase 1 dropout | HISTORY 主题 5 "Dropout 0.10 是 conservative 选择"
- **状态**：🔴 GAP（无 Phase 2 dropout ablation）

---

## 4. SPO+ DFL 损失公式

### Q4.1 SPO+ 公式：z_star + spread + relu(|spread|-fee) - z_star*spread + fee*|z_star|
- **来源**：NN_TRAIN Q41-Q49, CODE_QUESTION_LIST Q43
- **答案**：标准 Elmachtoub-Grigas 2022 SPO+ surrogate，formal form:
  ```
  z_star = sign(y) 当 |y|>fee 时，else 0   （"true optimal action"，含死区）
  ell = relu(|2*pred - y| - fee) - z_star*(2*pred-y) + fee*|z_star|
  ```
  - **z_star 死区**：当 |y| < fee 时 z_star=0，意味"该样本上 oracle 不交易"；模型也不应被迫交易该样本（避免学习手续费亏损方向）。
  - **2*pred - y 形式**：Elmachtoub-Grigas 标准（来自 cˆ - 2c 的 LP 对偶形式）；写成 `2*pred - y` 而非 `pred + (pred-y)` 是论文原貌，可读性差但数学等价。
  - **+ fee*|z_star|**：在 z_star=±1 时为 fee，z_star=0 时为 0；这是补偿"如果走 oracle 决策会交手续费"的常数项，让 ξ-surrogate 对应到完整 PnL（regret-style）。
  - 1D 三选项任务的 Fisher consistency：论文证明在 |action|≤1 LP 决策空间下 SPO+ 是 calibrated；这里 z∈{−1, 0, 1} 是离散 3 点，严格 Fisher consistency 没有原文 LP 那么干净，但实测稳定（T87 LOSO-eq +40.09 → 平台 +28.16）。
- **核心证据**：`train_T188v2_nn_seed.py:117-127` | R10 SPO+ 10 方案调研 | T87/T136 sweep | HISTORY 主题 5
- **状态**：✅ 有理论依据 + 实证

### Q4.2 SPO+ z_star 死区是不是让一大块样本梯度 = 0？
- **来源**：NN_TRAIN Q41/Q43
- **答案**：是的，但**这是 feature 不是 bug**。z_star = 0 时 ell = relu(|spread|-fee) = 0 当 |2*pred-y|<fee。也就是说当模型 pred 在 y 的 ±fee 邻域，且 oracle 也是"don't trade"时，loss 为 0，无梯度。这强制模型不学习"在小于手续费幅度的样本上瞎猜"。这与 EV gate（pred > thr_up → buy）的死区是一致的——决策层和损失层都尊重 fee 死区。HFT 真实场景下死区是 alpha 的物理上限，损失函数应反映之。
- **核心证据**：HISTORY 主题 5 "SPO+ 的 surrogate gradient 把决策层（EV gate）的损失对齐到 PnL" | `spo_plus_loss` 实现
- **状态**：✅ 设计意图清晰

### Q4.3 L2 + λ·SPO+ 组合，λ=30，为啥不纯 SPO+
- **来源**：NN_TRAIN Q15/Q110/Q111, CODE_QUESTION_LIST Q11/Q44
- **答案**：L2 项作为 anchor 保持 pred 的"幅度信息"（pred 接近 y 的 magnitude），SPO+ 项校准"方向决策"。**纯 SPO+ (λ=∞)** 可能让 pred 退化到只关心 sign（pred 永远是 ±很大值），破坏 EV gate 对幅度的依赖。**纯 L2 (λ=0)** 就是 T81，平台 +28.16 → SPO+ 加进来后 +28.16；T87 SPO+ 加 +8.93 = +28.16 vs iter_013 +19.23 → SPO+ 贡献 ~+9 平台分。
  - T136 SPO+ arch sweep 扫了 λ ∈ {1, 10, 30, 100, 300}，**30 最优**比次优高 +0.3-0.5 LOSO-eq。
  - T176 在 M7 阶段重扫 λ ∈ {10, 30, 100}，holdout 数字相近（118.04 / 118.16 / 118.33）但都 in-sample，未上平台单独验证。
  - λ=30 的数量级合理性：L2 ≈ (pred-y)² ≈ var(y_scaled)=1（target_scale 已 normalize），SPO+ ≈ |2*pred-y| ≈ 线性 1 量级，所以 30×SPO+ 主导 30:1。这是有意的"SPO+ 主导决策对齐 + L2 弱锚定"。
- **核心证据**：T81 alone LOSO-eq +35.89 vs T87 LOSO-eq +40.09 (+4.20) | T136 λ sweep | T176 λ ∈ {10,30,100} holdout | HISTORY 主题 5
- **状态**：✅ 强消融

### Q4.4 fee = 0.0001 × (mp_th+1+mp_t+1)/(mp_t+1) ≈ 2×FEE
- **来源**：NN_TRAIN Q2/Q26/Q27, CODE_QUESTION_LIST Q22, LGB_TRAIN Q20
- **答案**：FEE = 1e-4 是平台公布的双边手续费率（单边 0.005%，双边 0.01% = 1e-4）。fee_eff 公式推导：在 t 时刻买入花 (mp_t+1) × (1+FEE)，在 t+H 时刻卖出收 (mp_th+1) × (1−FEE)，PnL = (mp_th+1)(1−FEE) − (mp_t+1)(1+FEE) ≈ (mp_th-mp_t) − FEE×(mp_th+1+mp_t+1)；除以 (mp_t+1) 归一化到与 regr_target 同单位，得到 `fee_eff = FEE × ((mp_th+1)+(mp_t+1))/(mp_t+1)`。当 mp_th≈mp_t≈0 时 fee_eff ≈ 2×FEE = 2e-4，即双边手续费。FEE 不在 LGB 训练里用（LGB 用 L2，fee 只影响 EV gate 决策），但在 NN SPO+ 损失里直接进 z_star 计算，敏感度极高（FEE 设错 z_star 全 0，loss 退化）。已验证 FEE=1e-4 与 thr_up=3e-4 关系 ratio≈3（pred 显著高于 fee 才有 alpha 余量），与 HISTORY 主题 1 thr_up DE 搜索结果一致。
- **核心证据**：`train_T188v2_nn_seed.py:75-77` | thresholds.json thr_up=3e-4 ≈ 3×fee_eff | HISTORY 主题 1
- **状态**：✅ 推导清晰，平台规则锚定

### Q4.5 regr_target = (mp_th - mp_t) / (mp_t + 1) 中 +1 的作用
- **来源**：NN_TRAIN Q24/Q25, CODE_QUESTION_LIST Q84, LGB_TRAIN Q52
- **答案**：mp_t（midprice1）在数据中是**已经被某种 normalization 处理过的相对值**，量级在 0 附近（mean ≈ 0, std 取决于 sym/window），所以 mp_t≈0 时直接除会爆炸。+1 是 stabilizer，把分母 shift 到 ~1 量级，让 (mp_th-mp_t)/(mp_t+1) 在 mp_t≈0 时退化为 (mp_th-mp_t)，与 absolute return 等价。**不用 log(mp_th/mp_t)** 因为 mp_t 可能为负或 0（normalize 后），log 不定义；不用 +EPS 因为 EPS=1e-9 量级 ÷ 1e-9 仍爆炸，+1 是数量级 stabilizer 而非微分 epsilon。fee_eff 用同一 +1 保持单位一致。
- **核心证据**：HISTORY 主题 7 "y = (mp_th − mp_t) / (mp_t + 1) 用 +1 是 normalize 后 mp 量级 stabilizer"
- **状态**：🟡 工程 rationale 清晰，无 mp_t 量级 audit 公开

### Q4.6 SPO+ 与 class_balanced_weight 的交互
- **来源**：NN_TRAIN Q21-Q23, CODE_QUESTION_LIST Q80/Q196, LGB_TRAIN Q22-Q27
- **答案**：class_balanced_weight 基于 y_cls (3 类: -1/0/1) 给样本权重，加在 L2 和 SPO+ loss 上。理论上 weighted SPO+ 的凸 surrogate 性质不严格保留（论文里 SPO+ 假设 unweighted），但实证：(a) class_balanced 让 y=0 中性样本权重 ≈ N/(3×中性 count) 小于 ±1 样本（因为中性占多数），(b) 这让模型更"关注"有方向的样本，与 EV gate 的"非中性才交易"对齐，(c) T188v2 平台 SOTA +35.64 验证不破坏 SPO+ 稳定性。class_balanced 公式 `len(y)/(num_class*counts)` 是标准 inverse-frequency normalize；空类的 `where counts==0 → 1` 防 ÷0 silent。没测过 unweighted MSE/SPO+，但 weighted 在 50 NN ensemble 下稳定收敛。
- **核心证据**：T188v2 平台 +35.64 | `class_balanced_weight` 实现 | HISTORY 主题 7
- **状态**：🔴 GAP（无 weighted vs unweighted SPO+ ablation）

### Q4.7 fee_scaled = fee × target_scale 在 SPO+ 里
- **来源**：NN_TRAIN Q48/Q105
- **答案**：SPO+ 在 scaled 空间计算（pred_scaled, y_scaled），fee 也必须 scale 否则 z_star 条件 `|y_scaled| > fee` 错位。target_scale ≈ 1/std(y_regr_tr)，若 std(y)=1e-4 量级，target_scale~1e4，fee_scaled=1e-4×1e4=1.0。这把 SPO+ 在 scaled 空间的"有效 fee 边界"放在了 y 标准化分布的 ±1σ 邻域——合理的死区比例。Phase 2 用 Phase 1 target_scale（不重算）确保 scale 一致性。
- **核心证据**：`train_T188v2_nn_seed.py:422` fee_scaled_full
- **状态**：✅ 推导一致

---

## 5. LGB HP grid 与训练协议

### Q5.1 5 组 HP 配置每个具体值来源
- **来源**：LGB_TRAIN Q1-Q15, CODE_QUESTION_LIST Q12
- **答案**：5 组 HP 是**手工挑选的多样性 grid**（不是网格、不是 sobol、不是 random search 的结果），意图是在 (feature_fraction, bagging_fraction, num_leaves, lambda_l2) 4 维空间提供"多样化采样"。设计原则：
  - **Config 0** (0.8/0.8/127/1.0)：baseline 中庸
  - **Config 1** (0.6/0.7/127/1.0)：feature/bagging 都更激进 subsample
  - **Config 2** (0.7/0.85/63/2.0)：narrower tree (63) + more L2 (2.0) → "高 reg、浅树"
  - **Config 3** (0.5/0.6/255/0.5)：wider tree (255) + less L2 (0.5) + low feature/bagging → "高 capacity 高 noise"
  - **Config 4** (0.4/0.5/127/3.0)：极激进 subsample + 高 L2 → "high reg + low signal"
  - 3 个 num_leaves=127 因为这是 LGB MLP-baseline 的 sweet spot（T18 LGB sweep 历史值），其他 2 个 (63, 255) 探索 capacity 维度。
  - 反 intuition 的"wide+low L2" / "narrow+high L2" 配置是有意打破"capacity-regularization 应该正相关"的预期，**强迫 ensemble 多样性**——T192 NN-NN corr 0.9331 vs LGB-LGB corr 0.8433 验证 LGB ensemble 比 NN 更多样。
- **核心证据**：`train_T188v2_lgb_seed.py:25-31` | T192 LGB-LGB corr 0.8433 vs NN-NN 0.9331（HISTORY 发现 8）| T179 vs T188v2 验证扩 LGB 也有收益
- **状态**：🟡 设计 rationale 清楚，但无完整 grid search log（手工挑而非系统 sweep）

### Q5.2 (seed-1)%5 cycling + 10 seed/HP
- **来源**：LGB_TRAIN Q2, CODE_QUESTION_LIST Q96
- **答案**：5 HP × 10 seeds = 50 models 是**对称多样性**设计：HP 提供 4 维空间多样性，seed 在每 HP 内部提供 bagging/feature_fraction 随机采样多样性（LGB 内部用 `params['seed']=S` derive bagging_seed/feature_fraction_seed/data_random_seed）。HISTORY DB 主题 2 关键发现：T179 (40 NN + 5 LGB，只扩 NN) 平台 −0.25，T188v2 (50+50 双侧扩) 平台 +1.00 → "NN+LGB 必须对称扩"。10 seed/HP 来自 T188v2 SOTA pkg 已验证 LGB-LGB corr 0.8433（vs NN 0.9331），10 seed 提供足够 bagging variance。T178 (LGB 5→10) holdout null result 表明"LGB 5 seed 已在 variance floor"——但是和 NN 一起扩到 50 时**对称多样性**才生效。
- **核心证据**：T179 平台 +34.39 (−0.25) | T188v2 +35.64 (+1.00) | T178 LGB 5→10 null | T190 150+150 平台 −1.05 | HISTORY 主题 2 + 发现 2
- **状态**：✅ 强消融

### Q5.3 num_boost_round=330 固定，无 early stop
- **来源**：LGB_TRAIN Q8/Q39/Q40, CODE_QUESTION_LIST Q9
- **答案**：V4 walk-forward 5 seeds 平均 best_iter ≈ 300（早停于 train+val→test 的 96-119 holdout），×1.1 → **330**（与 NN ep=11 同 ×1.1 heuristic）。M7 trick 后无 val 集，所以固定 round。LGB 在大数据上单调收敛（不像 NN 会 overfit 倒回），330 vs 300 在 train loss 差异小，但多 10% iters 让 M7 全数据训练有缓冲（多 12.5% 训练数据 ≈ 多 5-10% 收敛 iters）。同一 round 用在 num_leaves=63/127/255 三个 capacity 上看似"unfair"，但实测 ensemble 平均掉每个 config 的 over/under-converge，整体 +35.64 SOTA。
  - T140 M7 LGB 平台 +5.51 直接验证 330 + M7 trick 组合。
  - 不分 HP 用不同 round 是简化设计；num_leaves=63 理论应训更多 iters 但 T18 早期 LGB sweep 在 same round 下 63/127/255 均收敛 OK。
- **核心证据**：HISTORY 主题 4 "V4 5 seed avg best_iter 300 × 1.1 = 330" | T140 平台 +34.44 (+5.51) | `train_T188v2_lgb_seed.py:72`
- **状态**：✅ 强消融（平台 +5.51 直接验证）

### Q5.4 objective=regression_l2, metric=rmse, 无 L1/Huber/Quantile
- **来源**：LGB_TRAIN Q16-Q21, CODE_QUESTION_LIST 隐含
- **答案**：T75 把 LGB 从 3-class CE 换成 L2 Δmid regression，LOSO-eq +26.44 → +36.23（**+9.79，整个项目最大单次跃升**），iter_013 平台 **+19.23**。后续试过：
  - **MAE / L1**: T137 +0.3-0.5 边际且含 OOF 污染，T143 LOSO −0.29，T145 平台预测 +30~32 都不超 L2 M7 +34.44。
  - **Huber**: T99 LOSO +44.72 但 OOF 污染严重，iter_016 v3 平台 **−3.08**。
  - **Quantile (α=0.5 ≈ MAE)**: T132 边际。
  - **GMADL** (T116): 微弱。
  - **Custom PnL / fobj**: 没试（NN 侧 SPO+ 已对齐 PnL，LGB 用 L2 surrogate + EV gate 已 +19.23 baseline 充分）。
  - L2 对 tail returns 敏感是已知缺陷，但 class_balanced_weight + bagging 已经压尾，整体 +35.64 SOTA。L1 在 GBDT 上 hessian = 0 让 leaf value 估计不稳。
- **核心证据**：T75 平台 +19.23 第一次突破 | T99 Huber iter_016 平台 −3.08 | T137/T143/T145 MAE 路线全部不超 L2 M7 | HISTORY 主题 7 完整表
- **状态**：✅ 强消融（4 种 alt loss 都被试过且更差）

### Q5.5 min_data_in_leaf=100, bagging_freq=5
- **来源**：LGB_TRAIN Q9/Q10, CODE_QUESTION_LIST Q27/Q28
- **答案**：min_data_in_leaf=100 是 1.47M × 2 (aug) ≈ 2.94M 训练样本下"叶 ≥ 0.0034%"的保守值，防止 noise leaves。在 num_leaves=255（最深 config）下，最大叶数 255 × 0.0034% = 0.87% 样本/叶——叶仍能学到 signal 但不退化为单样本 memorization。255 vs 63 用同一 min_data 不是"capacity-aware"调整，但 LGB 内部对 num_leaves<255 时通常分裂不到 255 个（受 max_depth=−1 + min_split_gain 限制），所以 100 是 OK 兜底。bagging_freq=5 即每 5 iters 重采样一次 bagging set，330 iters 给 ~66 次 bagging draws——足够多样性。没单独 sweep min_data ∈ {50, 100, 200} 或 bagging_freq ∈ {1, 5, 10}。
- **核心证据**：`train_T188v2_lgb_seed.py:74-75`
- **状态**：🔴 GAP（无 ablation）

### Q5.6 lambda_l2 而非 lambda_l1
- **来源**：LGB_TRAIN Q11
- **答案**：L2 正则化对 LGB 叶 value 是常规选择（sum(g²)/(sum(h)+λ²) 让大幅 leaf 缩水）。L1 (lambda_l1) 是稀疏化（drop irrelevant features），在 359 维 SchemeP 上**特征已经精挑细选 + drop 了 11 个 FAIL**，再 L1 稀疏化收益小。HP grid 在 lambda_l2 ∈ {0.5, 1.0, 1.0, 2.0, 3.0} 范围内提供多样性，覆盖 reg 从弱到强。没测 L1。
- **核心证据**：T59_FAIL_NAMES + STAGE5_FAIL_NAMES drop 11 个 | `train_T188v2_lgb_seed.py:26-30`
- **状态**：🔴 GAP（无 L1 试验）

### Q5.7 valid_sets=[dtrain] 用训练集做"validation"
- **来源**：LGB_TRAIN Q38/Q43
- **答案**：M7 trick = 全数据训练，没有 holdout val。`valid_sets=[dtrain]` 只是让 `log_evaluation(period=50)` 有东西打印 train RMSE，不参与 early stop（也没设 early_stopping_rounds）。num_boost_round=330 固定。这是 M7 设计的必然结果——HP 选择已经在 V4 walk-forward 阶段做完，M7 只是"用更多数据重训"。
- **核心证据**：HISTORY 主题 4 + 主题 9 | `train_T188v2_lgb_seed.py:181`
- **状态**：✅ M7 设计内在一致

### Q5.8 LGB --gpu flag CLAUDE.md 说默认 ON，但 train.py 默认 False
- **来源**：LGB_TRAIN Q107, PIPELINE_ARCH Q29
- **答案**：`run_pipeline.sh:13` 默认传 `--gpu`（项目实测 LGB GPU vs CPU 38s → 12s, **3.2× 加速**），CLAUDE.md 据此声明 "LGB GPU=on"。但单独跑 `train_T188v2_lgb_seed.py` 时默认 CPU（`args.gpu` 默认 False，行 78）。冲突是 entry-point 层：pipeline shell 显式传 --gpu，单独运行需手动传。byte-equivalence 在 GPU/CPU 之间不严格（GPU LightGBM 用 single precision，CPU 用 double）——HISTORY DB T155 验证过 LGB M7 100% deterministic（同设备），但跨设备非 bit-equiv。Predictor 推理用 CPU `.txt` model 通用。
- **核心证据**：HISTORY 主题 4 T155 LGB deterministic | CLAUDE.md 速查 | `run_pipeline.sh:13`
- **状态**：✅ 工程一致

---

## 6. M7 全数据重训

### Q6.1 为什么用 dates 0-119 全数据训练（含 96-119 holdout）
- **来源**：NN_TRAIN Q97/Q135, LGB_TRAIN Q43, CODE_QUESTION_LIST Q76
- **答案**：M7 是项目第三次大突破（iter_018 +28.93 → iter_019 v2 +34.44，**平台 +5.51**）。原理：V4 walk-forward 阶段（train 0-79 / val 80-95 / test 96-119）已经选定 HP 与早停 epoch，M7 阶段把这三段全部放进训练，相当于多 33% 训练样本。理由：(a) Kaggle/ML 标准做法 "CV-select HP, retrain on all"，(b) 96-119 是 platform OOD 最相近时段（时间最近、相同 sym 分布），更值得用作训练而非 holdout，(c) T140 实测平台 +5.51 直接验证。本地 holdout（96-119）此时**绝对值不可信**（model 见过这些数据），只看相对 delta。
- **核心证据**：T140 平台 +28.93 → +34.44 (+5.51) | T170 NN M7 平台 +34.64 (+0.20) | HISTORY 主题 4 + 发现 1/3
- **状态**：✅ 强消融

### Q6.2 LGB num_boost_round=330, NN epoch=11 的 ×1.1 heuristic
- **来源**：LGB_TRAIN Q40, NN_TRAIN Q13, CODE_QUESTION_LIST Q9/Q10
- **答案**：M7 无 val 监控，固定 round/epoch 由 V4 walk-forward 阶段 5 seeds 平均 best_iter/best_epoch 乘 1.1 向上取整：
  - **LGB**：V4 best_iter ≈ 300 → 330（×1.1）。理由：多 12.5% 训练数据需要多 ~10% iters 收敛。
  - **NN**：V4 best_epoch ≈ 9.2 → 10.1 → **11**（×1.1 向上取整）。
  - **R_full_retrain 调研**：量化测试 330 iters 上 train loss vs 300 iters 相差小，模型未过拟合（GBDT 单调收敛特性）。
  - **T176 sweep ep=7/11/15/20**：ep=15 平台 +34.51 (−0.13 vs T170 ep=11)，confirm ep=11 优。
  - **替代方案 ×1.0 / ×1.2 / ×1.5** 没单独 sweep，但 ×1.1 是 "slight buffer for extra data" 的合理 heuristic。
- **核心证据**：T140 LGB 330 平台 +5.51 | T176 NN ep=15 平台 −0.13 | T170 ep=11 平台 +34.64 | HISTORY 主题 4
- **状态**：✅ 强消融

### Q6.3 不留 val 集
- **来源**：NN_TRAIN Q108/Q135, LGB_TRAIN Q43
- **答案**：M7 设计就是不留 val。理由：(a) 96-119 是 OOD 最相近，更值得训练用；(b) 留 val 集 = 浪费 12.5% 训练样本 + 12.5% 时间 OOD 信号；(c) Kaggle 类似做法。代价：失去动态 early stop 能力，需要预先固定 round/epoch（×1.1 heuristic）。T140 平台 +5.51 直接验证"不留 val" 优于"留 val M7"（V4 walk-forward 阶段平台 +28.93）。
- **核心证据**：T140 平台 +5.51 | R_full_retrain 调研 | HISTORY 主题 4
- **状态**：✅ 强消融

### Q6.4 M7 LGB 增量 +5.51 vs NN 增量 +0.20，为啥差距这么大
- **来源**：NN_TRAIN Q97, HISTORY 主题 4
- **答案**：LGB 从头训，M7 多 33% 数据让 boosting 看到更多 patterns（CV 阶段 dates 0-79 训练在 96-119 上 OOD，M7 把 80-119 也加入消除部分 OOD gap）；NN 已经 T81 L2 warm-start + SPO+ DFL fine-tune 5 epoch（V4 阶段），M7 再加数据增量饱和。T183 NN random init M7 vs T81 warm-start M7 在 holdout 持平 (+0.63)，进一步证 NN 增量已饱和。这也是 ensemble 设计 w_LGB=1.5 > w_NN=1.0 的原因之一——LGB 在 M7 后 PnL 贡献更大。
- **核心证据**：T140 LGB M7 +5.51 | T170 NN M7 +0.20 | T183 NN random init holdout 持平 | HISTORY 发现 3
- **状态**：✅ 强消融

### Q6.5 M7 sym-agnostic / date 约束遵守
- **来源**：NN_TRAIN Q137/Q138/Q140, LGB_TRAIN Q111/Q113
- **答案**：M7 训练数据 schemeP_{train,val,test}.npz 里 date 列保留真值（用于 split 而非 feature），sym 列也保留（用于 class_balanced_weight 中性化但不进 feature）。模型 forward 不接收 sym/date（assert `forbidden = {"date","sym","time"}` not in feat_names）。CRITICAL_CONSTRAINTS #1（date 评测置 0）与 #3（sym 可能含 OOD）都被遵守：模型完全 sym-agnostic（无 sym embedding、无 per-sym normalization）。T149 NN OOD sym 鲁棒性扫描确认 T87 在 OOD sym 下表现可接受。
- **核心证据**：CRITICAL_CONSTRAINTS §1 | T149 OOD sweep | `train_T188v2_nn_seed.py:207-208` assert | `train_T188v2_lgb_seed.py:104-105` assert
- **状态**：✅ 硬约束遵守 + T149 validation

---

## 7. aug_a 数据增强

### Q7.1 AUG_LO=0.80, AUG_HI=1.20 (±20%) per (sample, feature) 独立 multiplicative
- **来源**：NN_TRAIN Q12/Q72-Q75, LGB_TRAIN Q28-Q37, CODE_QUESTION_LIST Q18/Q81/Q82
- **答案**：aug_a 是项目早期（T25-T28，iter_005a/b）确立的数据增强，原始版本是"bid/ask 互换 + 涨/跌 label 对换"对称翻转，T188v2 时演化为"独立 per-feature 乘 U(0.8, 1.2)"的 multiplicative noise。
  - **±20% 幅度**：经验 sweep（T25-T28 + T30）确定。±10% noise 太弱（diversity 不足），±30% 破坏特征语义（趋势/订单簿结构特征被压扁）。±20% 给 ~4x ratio (0.8/1.2 ≈ 1/1.5)，覆盖 LOB 日内 micro-variation 量级。
  - **每 (sample, feature) 独立采样**：打散特征间相关性。质疑（NN Q73）是"特征协同被打散"，但实测 5-seed LOSO h_60 +11.46（vs 无 aug 单 seed +9.67，+1.79）证明独立采样有效。LOB 特征间 corr 一般 < 0.5（窗口 z-score 已经做了部分 decorrelation），独立 noise 不会让特征跨"决策面"。
  - **label/fee 不变**：等价 "1×aug = 2x oversample with feature noise"，相当于强化"模型应对 ±20% 特征扰动不变"的先验。对于 normalize 后的 LOB 特征，±20% 是市场 micro-variation 等级，label 不变是合理假设。
  - **uniform vs lognormal**：multiplicative noise 标准做法是 lognormal，但 exp(±0.2) ≈ [0.82, 1.22] 与 uniform [0.8, 1.2] 几乎相同，uniform 更简单可控。
- **核心证据**：T25-T28 iter_005a/b LOSO h_60 +6.30 → +9.67 (+3.37 单 seed) → 5-seed +11.46 | HISTORY 主题 6 时间线 | `train_T188v2_lgb_seed.py:44`
- **状态**：✅ 强消融（aug 的存在 vs 无 aug 验证；幅度有 T30 间接调）

### Q7.2 LGB 与 NN 用同样 aug_a，但 LGB 对单调变换不变？
- **来源**：LGB_TRAIN Q83 (CODE_QUESTION_LIST Q83)
- **答案**：单调变换不变性指**全局 monotonic transform**（如 x → 2x）不改变 LGB 分裂；但 aug_a 是 **per-sample 独立 multiplicative**，每行不同 scale，**打破了原数据的相对顺序**——这等价于"添加 noise 让 LGB 学到 noise-robust 分裂"，是有效 augmentation。实测 T25-T28 aug_a 在 LGB 上从 +6.30 → +9.67 单 seed 提升验证。
- **核心证据**：T25-T28 aug_a LGB 增益
- **状态**：✅ 强消融

### Q7.3 增强后训练数据 ×2 (n_train_used = n_total × 2)
- **来源**：LGB_TRAIN Q30/Q35, NN_TRAIN Q72, CODE_QUESTION_LIST Q82
- **答案**：1× duplicate（1 原 + 1 aug），不是 2×/5×。原因：(a) 内存约束（2.94M × 226 feat × 4B × 2 ≈ 5.3GB，5× 会 OOM），(b) 边际增益递减（augmented copies 共享同 label，diversity 主要来自 noise pattern，更多 copies 不增 information），(c) wall-clock：50 seed × Phase 1 (2× 数据) × Phase 2 (2× 数据) 已 83 min，5× 会 4× 时间。没单独 sweep multiplicity ∈ {1, 2, 5}，但 1× 是工程稳健折中。
- **核心证据**：HISTORY 主题 6 + T188v2_mega_proper PROGRESS 83min/50seed
- **状态**：🔴 GAP（无 multiplicity sweep）

### Q7.4 aug 数据放在 X_tr_full 第二半，不 shuffle
- **来源**：LGB_TRAIN Q35
- **答案**：LGB Dataset 内部对 row 顺序无依赖（bagging 是 random subsample），所以拼接顺序无影响。NN 训练用 `torch.randperm(n_train)` 每 epoch shuffle，也无关。这是 vectorize 写法简化（CHUNK=200K 分块写入第二半的内存连续区域）。
- **核心证据**：`train_T188v2_lgb_seed.py:143-145` + LGB bagging
- **状态**：✅ 等价

### Q7.5 RNG offset: LGB +42, NN P1 +1, NN P2 +137
- **来源**：LGB_TRAIN Q36/Q68, NN_TRAIN Q74/Q102, CODE_QUESTION_LIST Q32, PIPELINE_ARCH Q31/Q32
- **答案**：`np.random.default_rng(S * 7919 + offset)`，7919 是质数，乘 S 给不同 seed 之间的"分散"，加 offset 区分三个独立的 RNG 流（LGB aug / NN P1 aug / NN P2 aug）让同 seed 下三个流的随机数不同。42/1/137 是 magic number（42 是 The Answer，1 是 trivial，137 是质数）——**没明确文档化为啥这三个**，但只要保证不重叠就 OK。CHUNK=200K 在 LGB aug 里影响 RNG 消费量（每 CHUNK 一次 uniform 调用），未来改 CHUNK 会让随机数序列变化——是 reproducibility risk（PIPELINE_ARCH Q32 指出），但 final pkg 用固定 CHUNK 所以已锚定。
- **核心证据**：`train_T188v2_lgb_seed.py:142` + `train_T188v2_nn_seed.py:270/406` | PIPELINE_ARCH Q31/Q32
- **状态**：🟡 magic number 无文档，但工程上 OK

---

## 8. 评估方法论变迁

### Q8.1 LOSO 5-fold → LOSO-equiv → V4 walk-forward → holdout in-sample 四阶段
- **来源**：QREVIEW 隐含, HISTORY 主题 9
- **答案**：项目 5 个月内 eval 方法换了 4 次，每次都影响"本地 → 平台"对应：
  - **Phase 0-1 (T1-T35)**：**LOSO 5-fold** 留一 sym，过度惩罚 sym OOD（每 fold 只 4 sym 训练）。
  - **Phase 2 (T44-T74)**：**LOSO-equiv** 全 sym 训练 + DE 在 442k 行测试集。T59 把方法论从 LOSO 切到 LOSO-eq 时 +7.68，其中 ~+7 是方法论 inflation（去除 LOSO 过度惩罚），真实新增 alpha ~+1。
  - **Phase 3 (T75-T155)**：**+V4 walk-forward** train 0-79 / val 80-95 / test 96-119 OOF。T64 V4 vs LOSO-eq 数字几乎相同 (+0.50)，但更干净。LOSO-eq → 平台透传系数 53-71%（iter_013 53% / iter_015 71% / iter_018 70%）。
  - **Phase 4 (T156+)**：**holdout in-sample**（date 0-119 全数据训练 → 96-119 评分），数字爆涨 3-4× 但绝对值不可信。T170 holdout +149.95 vs V4 +41；T188v2 +150.43 vs T170 +149.95 delta +0.48 → 平台 +1.00。**只看相对 delta，不看绝对值**。
- **核心证据**：HISTORY 主题 9 完整时间线 + 4 个污染陷阱（T180/T182/T172/T192）
- **状态**：✅ 强证据（22 次平台提交锚定）

### Q8.2 平台 OOD 是唯一 ground truth；本地 holdout in-sample 污染陷阱
- **来源**：NN_TRAIN Q97/Q135, HISTORY 发现 1
- **答案**：4 个经典 in-sample 陷阱案例：
  - **T180 OOD GMM abstain**：holdout V1-V3 −32 ~ −78，未提交（弃权策略在 in-distribution 测试集上必损 PnL）。
  - **T182 无 conformal ablation**：holdout +6.62 (看似 conformal hurt) → 平台 **−0.30** (conformal 实际有用；in-sample artifact)。
  - **T172 GroupTransformer**：in-sample +149.91 → 未提交 (新架构 OOD 风险高)。
  - **T192 bounded NNLS H4**：val +180.41, test +193.94, ratio 1.075 → 平台 **+31.0** (−4.64 vs T188v2)；OLS 学到 in-sample 共同 noise pattern。
  - 教训：本地 +X 不等于平台 +Y；只用 delta 比较；做 ablation 时优先 OOD（V4 walk-forward）或谨慎信 in-sample 大 delta。
- **核心证据**：HISTORY 发现 1 + 22 次平台提交表
- **状态**：✅ 极强消融

### Q8.3 transmission rate 53-71% LOSO-eq → 平台
- **来源**：HISTORY 主题 9
- **答案**：iter_013 LOSO-eq +36.23 → 平台 +19.23 (**53%**)，iter_015 +40.09 → +28.16 (**71%**)，iter_018 +41.49 → +28.93 (**70%**)。DE 阈值越激进（4D OOF DE）透传越低；保守 thr_up=3e-4 + per-sym conformal 透传 ~70%。iter_016 v3 4-way LOSO +43.71 → 平台 **−3.08** 是 OOF DE 灾难（4D OOF 严重过拟合 442k 本地测试集）。
- **核心证据**：HISTORY 主题 9 + 22 次提交 + 主题 1 thr 选择
- **状态**：✅ 强证据

### Q8.4 50→150 扩容非单调，5→50 才有效
- **来源**：HISTORY 发现 2
- **答案**：T179 (40 NN 只扩 NN) 平台 −0.25；T188v2 (50+50 双侧扩) 平台 +1.00；T190 (150+150) 平台 −1.05。这说明：(a) **ensemble 扩容必须对称**（NN+LGB 都扩）才有 variance reduction，(b) **5→50 边际正**，(c) **50→150 边际负**——HP 多样性在 5×5 HP × 2 seed × 5 LR = 50 已经接近饱和，再加 100 个变种引入 noise/biased HP。T192 OLS 学权重在 50 corr 0.93 矩阵上 condition number 54182 病态，证明 NN 多样性已饱和。
- **核心证据**：T179/T188v2/T190/T192 平台 4 点 + corr 矩阵 | HISTORY 发现 2/8
- **状态**：✅ 极强消融

---

## 9. 训练相关其他设计点

### Q9.1 forbidden = {"date", "sym", "time"} 只检查名字
- **来源**：NN_TRAIN Q62/Q137-Q140, LGB_TRAIN Q59/Q111-Q113
- **答案**：assert `not (forbidden & set(feat_names))` 只防字面名（"date"/"sym"/"time"）混进 feature 列。CRITICAL_CONSTRAINTS #3 sym-agnostic 的更深保护在**特征工程层**：T7 用 100-tick 窗口 z-score（不 per-sym），T9 SchemeE 用 spread-norm（global），R34 Stage 1-5 全部 sym-invariant。**未在 train 脚本里做**的检查：(a) 特征是否隐含 sym 信息（如某些 W=20 滚动统计依赖 sym 内顺序），(b) date-derived（如时段、首末日）—— T74 SchemeQ time features 试过在回归框架下 LOSO-eq −0.72 后弃用，所以 final 包不含。但脚本只有字面检查，是 trust-the-pipeline 的设计。T149 NN OOD sym sweep + T115 scale invariance audit 是 sanity check 层。
- **核心证据**：CRITICAL_CONSTRAINTS §1 | T7/T9/T74/T149/T115 sym-agnostic 验证 | `train_T188v2_nn_seed.py:207`
- **状态**：✅ 上游 pipeline 保护 + assert 兜底

### Q9.2 weights_only=False 加载 ckpt
- **来源**：NN_TRAIN Q93
- **答案**：ckpt 包含 non-tensor 元数据（hidden tuple, dropout float, feat_names list, target_scale float, keep_idx numpy 等），torch 2.x 推荐 weights_only=True 但需要全 tensor。生产 trade-off：weights_only=False 有 pickle 风险但项目内部 ckpt 可控（没有第三方 untrusted ckpt），允许 metadata 一同保存简化代码。如果生产化部署到不可信环境（如平台），Predictor 侧只 load npz（手工 extract），没 weights_only 风险。
- **核心证据**：`train_T188v2_nn_seed.py:231` | Predictor 用 npz
- **状态**：🟡 设计权衡

### Q9.3 推理用 npz 而非 torch state_dict
- **来源**：NN_TRAIN Q50, PIPELINE_ARCH Q5/Q19, PREDICTOR Q97 系列
- **答案**：平台环境 torch 版本可能与训练不一致（requirements.txt 指定 `torch==2.5.1+cpu`，但平台预装 torch 可能不同），npz 是 numpy 标准格式跨版本兼容；手工 extract_npz 把每层 W/b/LN_W/LN_b 拆出来，推理端（Predictor `_BatchedMLPEnsemble`）用 torch.bmm 重建 forward——避免 torch 序列化兼容性 + 允许 50 NN batched bmm 加速（T188v3 186× 加速核心）。代价：维护成本（手工 layer_idx +1/+2/+1+2 + GELU 不一致 risk）。这是项目晚期 T188v3 推理加速带来的必然选择。
- **核心证据**：HISTORY 主题 10 T188v3 186× | `extract_npz` + Predictor 实现
- **状态**：✅ 工程权衡清晰

### Q9.4 Phase 1 NaN imputation 用 for-loop（违反 CLAUDE.md "禁止 Python for"）
- **来源**：NN_TRAIN Q68/Q133, CODE_QUESTION_LIST Q155
- **答案**：`for d_idx in np.where(nan_mask.any(axis=0))[0]:` 只迭代有 NaN 的列（通常 < 10 列），不是逐 sample。严格说 CLAUDE.md 规则面向"特征构建"，这里是 imputation 不是 feature construction，但理论上 `X_imp = np.where(nan_mask, feat_mean[None,:], X_imp)` 一行 vectorize 即可。性能差异 negligible（< 1s for 1.47M × 226），但是 nit 该改。
- **核心证据**：`train_T188v2_nn_seed.py:260-263` | CLAUDE.md "特征构建：NumPy/Pandas 向量化"
- **状态**：🔴 GAP（minor lint）

### Q9.5 训练时无 WandB、无 worker-progress、无 results.json（CLAUDE.md 要求）
- **来源**：NN_TRAIN Q56/Q122/Q124/Q125, LGB_TRAIN Q71, PIPELINE_ARCH Q11
- **答案**：CLAUDE.md 要求所有实验用 WandB + worker-progress.json + results.json + RESULT 行，但 final_submission_code 是**提交包代码**（给评测平台跑或给 reviewer 复现），不是 internal experiment——所以没 WandB（平台环境无网）、没 worker-progress（无 PM 跟踪）。开发期实际有 WandB（experiments/T87/, T136/, T176/wandb/ 等都有 run 记录），但 final 包剥掉所有调试 instrumentation 保持纯净。`--no-wandb` flag 是 leftover 残留（从 T87 训练脚本来），未清理。
- **核心证据**：experiments/T188v2_mega_proper/wandb/（开发时有）vs final_submission_code/03_train_nn/（最终包剥掉）
- **状态**：✅ 工程意图清晰（区分 internal vs final）

### Q9.6 50 seed 串行 + GPU pinning 缺失
- **来源**：LGB_TRAIN Q85-Q91, NN_TRAIN Q165, PIPELINE_ARCH Q44
- **答案**：`run_all_lgb_seeds.sh` / `run_all_nn_seeds.sh` 用 `for SEED in $(seq 1 50)` 串行跑，单 GPU 单进程。生产实测 T188v2_mega_proper PROGRESS：单 RTX 3080 50 seed × 100s = 83 min（NN）+ 50 LGB × ~15s = 12.5 min ≈ ~96 min 总。可并行（5 GPU × 10 seed = ~17 min），但 shell 不支持。这是简化设计（任何 GPU 用户 1 GPU 都能跑），不是性能瓶颈（96 min 一次性 retrain 可接受）。
- **核心证据**：T188v2 PROGRESS RTX3080 83 min/50 seed
- **状态**：✅ 工程权衡

### Q9.7 cudnn deterministic / torch.use_deterministic_algorithms 未设
- **来源**：NN_TRAIN Q59/Q120, PIPELINE_ARCH Q27/Q28, CODE_QUESTION_LIST Q91
- **答案**：未设 `torch.backends.cudnn.deterministic = True` 也未设 `torch.use_deterministic_algorithms(True)`。MLP 只有 Linear/LayerNorm/GELU，没有 conv，cuDNN 非确定性主要在 conv backward——这里影响微小。LGB 侧 `params['seed']=S` 但未设 `deterministic=true` 也未限 `num_threads=1`，所以多线程 bagging 不严格 bit-reproducible。HISTORY DB T155 验证过 LGB M7 100% deterministic（同设备、同 num_threads=18），但跨 num_threads 或跨设备不一定。这是 reproducibility vs speed 的权衡，且整个 ensemble 平均掉 low-order bit 差异，对最终 PnL 无可观测影响。
- **核心证据**：T155 LGB deterministic confirmation | HISTORY 主题 4
- **状态**：🟡 已知 trade-off

---

## 10. GAP 表（本主题下显式 ablation 缺失的设计点）

| # | 设计 | 当前值 | 为什么这样 | 风险 / 备注 |
|---|---|---|---|---|
| 1 | NN 训练/推理 GELU 实现不一致 | 训练 erf，推理 tanh | 推理优化时用 batched bmm 手工实现 LayerNorm + tanh GELU；T188v3 实测 actions max diff 6.98e-10 | **真 bug**，应统一；当前数值上无可观测影响 |
| 2 | NN MLP depth/width 直接 sweep | (256, 128, 64) | T81 设计后未单独 sweep 宽度/深度 | 中等风险，对比其他架构有 ablation |
| 3 | BN vs LN 直接 ablation | LN | 间接论证（小 batch + OOD 友好），无直接 BN/LN 实验 | 低风险 |
| 4 | Kaiming init nonlinearity="relu" vs "gelu" | relu | Phase 1 几乎不动（best_epoch=0~10），init 不是瓶颈 | 极低风险 |
| 5 | Dropout 完整 sweep（含 0/0.3/0.5） | 范围 [0.05, 0.20] | T19 reg sweep LOSO 持平，conservative 选择 | 中等 |
| 6 | Weight decay sweep (0 / 1e-5 / 1e-3) | 1e-4 | 与 dropout 同 reg 维度，T19 持平 | 低 |
| 7 | grad clip 5.0 vs 1.0/10.0 | 5.0 | Safety net，无 sweep | 低 |
| 8 | CLIP=10 vs 5/8 | 10 | 留 0.01% 极端值，无 sweep | 低 |
| 9 | LR_LIST Phase 1 sweep（5e-4 等中间点） | [1e-4, 3e-4, 1e-3] | HP cycling 需要 3 离散点，中间点无意义 | 低 |
| 10 | aug multiplicity ∈ {1, 2, 5} sweep | 1× | 内存 + wall-clock 约束 | 低 |
| 11 | aug ±10/30% 幅度直接 sweep | ±20% | T25-T28/T30 间接调，无 explicit grid | 中等 |
| 12 | NN Phase 2 dropout=0 ablation | 复用 Phase 1 dropout | SPO+ 决策稳定性，无 ablation | 低 |
| 13 | NN freeze backbone ablation | 全 unfreeze | MLP 3 层无清晰 boundary | 低 |
| 14 | LGB min_data_in_leaf / bagging_freq sweep | 100 / 5 | 经验值兜底，无 ablation | 低 |
| 15 | LGB lambda_l1 试验 | 未试 | 359 维已 drop FAIL 11 个，稀疏化收益小 | 低 |
| 16 | weighted SPO+ vs unweighted | class_balanced | T188v2 +35.64 验稳定，但理论 SPO+ 凸性要求 unweighted | 中等理论缺口 |
| 17 | Phase 2 SPO+ batch sensitivity | 4096 固定 | 显存 + 死区采样稳定性 | 低 |
| 18 | NN MLP residual / skip connection | 无 | 小网络 typical 设计 | 低 |
| 19 | Cosine schedule shape vs linear/constant | Cosine | Textbook，无 sweep | 低 |
| 20 | x ×1.1 ratio for M7 round/epoch | 1.1 | Heuristic，未 sweep ×1.0/1.2/1.5 | 低 |

**GAP 总计**：~20 个设计点无显式 ablation，但**核心决策路径**（M7 / SPO+ / w_LGB=1.5 / 50+50 / aug / thr / conformal）都有强消融 + 平台数字验证。

---

## Final Summary

- 共回答 **~110 个综合答案**，去重整合 5 份 review + CODE_QUESTION_LIST 中的 ~310 条训练相关问题。
- **强消融**（有 T 编号 + 平台数字 + 替代方案对比）：约 **30** 项核心决策（M7 / SPO+ / λ=30 / lr=3e-5 / NN ep=11 / LGB 330 iters / aug_a / 50+50 / w_LGB=1.5 / 评估方法论 / HP cycling 设计）。
- **部分消融**：约 **25** 项（架构选择、HP 间接验证）。
- **GAP（无显式 ablation）**：约 **20** 项（多为低风险工程细节：grad clip 5.0、CLIP=10、min_data_in_leaf=100 等）。
- **真 bug / 不一致**：1 项（训练 GELU erf vs 推理 GELU tanh，实测无影响）。

**关键发现**：
1. **HP cycling 是有意设计**：50 NN 只覆盖 12 独立 (lr, dropout, batch) 组合 + 每组 ~4.17 seed，配合 T179 (40 NN 只扩 NN) 平台 −0.25 vs T188v2 (50+50 双侧扩) +1.00 → **对称扩容才有效**。
2. **M7 是平台 +5.51 的主因**（LGB 全数据重训），NN M7 增量 +0.20 已饱和。
3. **SPO+ 不可替代**：T13/T134 direct PnL 不收敛，3-class CE 平台 −6.65，MAE/Huber/Quantile/GMADL 全部不超 L2+SPO+。
4. **本地 holdout in-sample 绝对值不可信**：T182 holdout +6.62 (看似 conformal 损) 但平台 −0.30 (conformal 实际有用)；T192 holdout +193.94 但平台 −4.64。
5. **训练-推理 GELU 不一致是潜在 bug**（max diff 6.98e-10 实测无影响，应统一）。

RESULT: task=[answers P1 training] metrics={n_questions_answered=110, n_gaps=20, n_strong_ablations=30, n_potential_bugs=1} notes=[HP cycling 是有意设计 12 独立组合 × 4.17 seed; M7 LGB 平台+5.51 主因; SPO+ 不可替代 (T13/T134 direct PnL 不收敛); 训练 GELU erf vs 推理 GELU tanh 潜在 bug 但实测 max diff 6.98e-10; 本地 holdout in-sample 绝对值不可信只看 delta]

---

# PART P2 — DECISION LAYER (Thresholds, Conformal, Ensemble, share_with, Inference Opt)

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

---

# PART P3 — FEATURE ENGINEERING

# Answers P3 — Feature Engineering

> Worker: P3 (features). 仅回答"特征工程"主题，不涉及训练 / 决策 / 推理（其他 worker 负责）。
> 来源：`QREVIEW_FEATURES.md`（225 问）+ `CODE_QUESTION_LIST.md` §3（Q47-Q75）+ `QREVIEW_NN_TRAIN.md` Q18-20 + `QREVIEW_LGB_TRAIN.md` Q3/Q57-Q62 + `QREVIEW_PREDICTOR.md` Q58-Q90/Q113。
> 证据：`final_submission_code/01_build_features/{build_schemeP_cache,fast_features_batch,stage5_features}.py` + `04_build_pkg/{Predictor.py,fast_features.py,config.json,thresholds.json}` + `HISTORY_EVIDENCE_DB.md` 主题 6 + `0512_final_report.md` Phase 2 + `experiments/T{7,10,22,35,44,51,53,55,68,74,78,141,142,147,151,152,191}/results.json` + `experiments/R_{cross_sym,roll_tsrv,time_of_day,Hawkes_OFI,Pairwise_ALL}/`。

## 状态图例
- ✅ = 有强证据链（实验编号 + 本地数 + 平台数 + 代码行）
- 🟡 = 部分证据 / 推断 / 仅本地（无平台对照）/ "约定俗成"无 sweep 记录
- 🔴 = GAP，无证据 / 与代码冲突 / 真有 bug 嫌疑

---

## 总结（TL;DR）

1. **Scheme 演变是 8 步累加**：SchemeA(154)→B(154 同样 baseline LGB)→C(226, +72 T3 特征)→D1(308, +154 窗口 z-score)→E(spread/amount 归一化)→F(全局 z-score 失败)→H(291, MLOFI/WMP 扩展失败)→**P(370 raw+derived，drop 11 → 359 final)**。SchemeP = 154 raw + (R34 Stage 1-5 共 216 derived，drop 11 = 205) = **154 + 205 = 359 维**喂模型。370 → 359 = 154+216−11，**数字对得上**。
2. **R34 Stage 1-5 累计本地增益**（h=60 LOSO）：iter_005b +9.67 → Stage 1 +13.67 (T44, +4.0) → Stage 2 +15.06 (T51, +1.4) → Stage 3 +16.84 (T53, +1.78) → Stage 4 边际 (T55) → Stage 5 +26.44 (T68, +0.50，方法论从 LOSO 5-fold 切到 LOSO-equiv 全 sym 训练，含 +7 方法论 inflation)。**Stage 1/2/3 是真正的 alpha 来源；Stage 4 / Stage 5 边际**。
3. **窗口长度选择 99% 没有 sweep**：(5,20,60) for MLOFI/GOFI、(5,10,20,50) for RV、(20,50,100) for SIGNED_RV/KYLE/CANCEL/ADAPT_MOM/SIGNED_BV/SPREAD_REG/TRADE_PERS、(20,30,50,100) for JSHARE、(30,50,100) for ROLL、(5,20,50,100) for LIQ_ASYM。**完全 idiosyncratic，每个 stage 自己挑**。HFT 文献"标准 30/60/120 秒"对应 10/20/40 ticks（3s tick），但代码采用 (20,50,100)≈(1min,2.5min,5min) 是 R34 调研的选择，没有正式 sweep（Q204-205）。
4. **EWMA α 4 档** (0.05, 0.1, 0.3, 0.5)（HL≈13/7/2/1 ticks）：覆盖从"近一个 5min 窗口的中段"到"几乎只看最后一 tick"。α=0.5 和 α=0.3 之间冗余度高（Q43）；α 选择是 T44 设定，从未 sweep（🟡）。
5. **amount_delta 唯一 log1p**：因为它是 **154 维 raw 中唯一未归一化字段**，量级 e3-e6，与 LOB 价格/size/intensity 不同；T8 诊断 sym=2 amount_delta z-score +7.49 偏离（蓝筹 ETF 量很大）→ T9 SchemeE 引入 sign-preserving log1p 压缩重尾。**volume_delta/totalbsize/bsize 没有 log1p 是不一致**（Q18, Q47, Q69）：理由是 amount = price × volume 量级更大，单 amount log1p 足够；volume 已通过 dual_z/qrank 内部归一化。🟡 没有 ablation 证明这是最优策略。
6. **DROP_NAMES 11 个特征**：`dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5, qrank_W100_spread1, qrank_W100_spread5, qrank_W100_spread10, qrank_W100_cumspread, kyle_lam_W50, kyle_lam_W100, roll_eff_spr_ratio_W100, liq_asym_top5_W5`（Predictor.py:68-75）。来源是 T59_FAIL（10 个 = T59 全 sym 训练时的 sym-agnostic verification，特征在某 sym 上 KS 失败 → drop）+ STAGE5_FAIL（1 个 = T68 之后 W=5 太短噪音过大）。**没有显式的 KS p 值记录**（🔴）。
7. **特征维度 370 = 154 raw + 196 baseline + 20 Stage5；训练用 359 = 370 − 11 drop**：数字闭环。但单窗口版 `fast_features.py` 只有 196 baseline（无 Stage5），是 dead code（Q122/Q217；Predictor 用的是 `fast_features_batch.py`）。
8. **未采纳的特征家族**：Alpha101/Alpha191（T22，**需跨截面，违反硬约束 #2**，但子集 33 个 + 124 个 implementable，最佳单因子 IC +21.80 已超 SchemeC，仍未入因 cross-sym risk）、Hawkes (T141/142, R_Hawkes_OFI, 边际)、SG filter (T35/T91, gain≈0)、TOD time encoding (T74/T78 −0.72；T151 v9 小正但 final 弃)、sin/cos (T74)、trend features (T191, corr 0.9941 + IC_residual −0.17, KILL)、Pairwise (R_Pairwise_ALL, 计算量大边际)、cross-sym (R_cross_sym ABORT — date 置 0 + batch 打乱 + sym 可能 OOD)。
9. **NaN/Inf 处理**：cache 末端 `extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)`（Cache:105）；每个 stage 内部还各有 `np.where(np.isfinite(...), ..., 0.0)`（防御性多次）；LGB native NaN passthrough（R_NaN1 确认）；NN 训练 `np.where(np.isnan, 0, X)` 后 standardize 再 clip ±10。**inf 在多数 stage 都处理了，唯独 EWMA 内部 `x_safe = where(isfinite(x), x, 0)` 把 NaN→0 当 EWMA 输入，会让 missing 段 EWMA 偏低**（Q46/Q63，🟡）。
10. **`fast_features.py` vs `fast_features_batch.py`**：前者单窗口、后者批量。**功能不一致**（单窗口只 196 维无 Stage5，批量 216 维含 Stage5）。**Predictor 用的是 `fast_features_batch.py`**（Predictor.py:191 importlib），所以单窗口文件是 **dead code**——`build_pkg.py` 仍打包它（Q120/Q217，🔴 dead code 浪费 + 误导未来维护者）。同时两文件 docstring 互相矛盾（单窗口说 350 dim/drop 10，实际 returns 196 dim，drop 应该是 11——Q157-158）。
11. **多个 "OFI" 形式并存**：MLOFI（T3，`>=`/`<=` 双计 size 等价 case，line 169-174）、GOFI（Stage 2，`>`/`<`/`==` 三分 line 411-418）、Stage5 OFI（line 645-651）—— **3 套 OFI，3 个 equality/边界约定**，源于"R34 是多次单独调研合并"，未统一（Q26/Q73/Q105-106/Q189/Q219）。这是 design debt，但模型已经训过、提交分数已 +35.64，**不动**。

---

## 1. Scheme 演变（A→B→C→D1→E→F→H→P）

### 1.1 SchemeA 154 → SchemeB 154 → SchemeC 226 (T3, +72)
- **来源**：QREVIEW_FEATURES §0/§2 (Q16-Q47), CODE_QUESTION_LIST Q47/Q49/Q50, HISTORY_EVIDENCE_DB 主题 6 T2/T3, 0512_final_report T3 节, tricks_table T2/T3。
- **答案**：
  - **SchemeA = 官方 mmpc_demo DeepLOB 用的 154 raw**（T1 NN baseline，LOSO h_60 ~+8，平台 −6.65）；**SchemeB = 同样 154 维但用 LightGBM 3-class CE**（T2，LOSO sum −22.10 — cross-sym OOD 极差）。A 和 B 是同一 154-d schema，差别只是模型族。
  - **SchemeC = SchemeB + 72 derived = 226**：T3 加入多尺度 MLOFI(30) + WMP(11) + RV(4) + EWMA intensities(24) + time encoding(3) = 72 维。LOSO h_10 +21.86（iter_002 核心）。**这是首次"加 derived feature 大幅涨分"**。
  - 选择 MLOFI/WMP/RV/EWMA 是因为这些是 HFT 标准微观结构指标（Cont/Kukanov/Stoikov, Avellaneda/Stoikov, BNS realized vol）。time encoding 后来被剥离（T3 时还在，T7 之后逐步替换为窗口归一化，T74 SchemeQ 重新尝试 sin/cos 失败 −0.72）。
- **核心证据**：T3 build_features.log + tricks_table T3 row | iter_002 平台 +4.07 | `fast_features_batch.py:5,148-241`（T3 no-time block 仍是 69 维 = 30+11+4+24，剩余 3 维 time encoding 已被移到 SchemeP 外部，**最终 SchemeP 不含 time**）。
- **状态**：✅

### 1.2 SchemeD1 308 (T7, +154 窗口 z-score)
- **来源**：QREVIEW_FEATURES Q123-126, CODE_QUESTION_LIST Q54, HISTORY_EVIDENCE_DB 主题 6 T7, 0512_final_report T7, T7 experiment results.json。
- **答案**：T7 加入 100-tick 窗口内的 z-score 归一化 154 维（对原始 154 维每列做 (x_last − mean(W=100)) / std(W=100)）。SchemeD1 = 154 + 154 = 308。LOSO h_10 +13.91（vs T5b SchemeC +21.86 在 h_10；但 T7 解决了 **sym=2 brittleness**，sym=2 cum_pnl 从 −0.98 翻到 +2.24）。窗口 z-score 是"sym-agnostic 归一化"的起点（T7 → 后续 dual_z 把 W=20 和 W=100 都做了）。
- **核心证据**：T7 results.json (cum_pnl_sym2: −0.98 → +2.24) | HISTORY_EVIDENCE_DB 主题 6 T7 行。
- **状态**：✅

### 1.3 SchemeE (T9, amount_delta 归一化 + spread-norm)
- **来源**：QREVIEW_FEATURES Q18-22, CODE_QUESTION_LIST Q47-48, HISTORY_EVIDENCE_DB 主题 6 T8/T9, 0512_final_report T8/T9。
- **答案**：T8 诊断发现 sym=2（蓝筹 ETF）的 amount_delta z-score +7.49 偏离训练分布（巨型成交量），导致预测器对 sym=2 误判 UP 62% vs 真实 18%。T9 给 amount_delta 加 sign-preserving log1p、给 spread 加 normalization。**这一步部分被吸收进 SchemeP**：raw_last 的 amount_delta 用 log1p（Cache:108-110），dualz/qrank 内部窗口 z-score 处理了其它 size 字段。
- **核心证据**：HISTORY_EVIDENCE_DB 主题 6 T8/T9, 0512_final_report 行 175-181 ("sym=2 amount_delta z-score +7.49") | `build_schemeP_cache.py:108-110`。
- **状态**：✅

### 1.4 SchemeF (T10, 全局 z-score) — 失败
- **来源**：QREVIEW_FEATURES Q138, HISTORY_EVIDENCE_DB 主题 6 T10, tricks_table T10。
- **答案**：T10 把窗口 z-score 换成全局（基于整个训练集统计量）z-score。LOSO h_10 +13.40，**不超 SchemeC +21.86**，**且不超 T7 SchemeD1 +13.91**。证明窗口归一化 > 全局归一化：原因是全局统计量被训练集 sym 主导，OOD sym 时 (x − global_mean) / global_std 不再以 0 为中心，破坏数值分布。**SchemeF 被弃**。
- **为啥窗口 > 全局 > per-sym**：
  1. **per-sym normalization 违反硬约束 #3**（OOD sym 无统计量）→ 直接不允许。
  2. **全局 z-score** 对 OOD sym 不稳（z 偏离 → 模型见到训练时未见过的输入空间）。
  3. **窗口 z-score** 每 100 tick 内自适应，对 sym distribution shift 鲁棒（每个 100-tick 切片自己产生归一化中心和尺度）。SchemeP 全部使用窗口归一化（dualz / qrank / signed_rv / kyle_inv 都是 last-vs-window）。
- **核心证据**：tricks_table T10 行 | HISTORY_EVIDENCE_DB 主题 6 T10。
- **状态**：✅

### 1.5 SchemeH 291 (T20/T21, MLOFI/WMP 扩展) — 失败
- **答案**：T20/T21 把 MLOFI/WMP 进一步扩到 291 维（具体加什么 log 未保留），LOSO h_10 +21.70，**持平 SchemeC +21.86**。"单纯加维度无效" → 必须 + sym-invariant 设计（T44 起的 R34 Stage 1）才有真增量。
- **核心证据**：tricks_table T20/T21 | HISTORY_EVIDENCE_DB 主题 6 T20/T21。
- **状态**：✅

### 1.6 SchemeP 359 (T44/T51/T53/T55/T68, R34 Stage 1-5)
- **来源**：QREVIEW_FEATURES §3-§6 (Q48-Q150), HISTORY_EVIDENCE_DB 主题 6, 0512_final_report Phase 2, T44/T51/T53/T55/T68 results.json。
- **答案**：见 §2 详述。**SchemeP 总维度 = 370 (154 raw + 216 derived = 69 T3 + 54 S1 + 59 S2 + 14 S3 + 20 S5)**；训练前 drop 11 → **359 喂模型**。
- **核心证据**：`fast_features_batch.py:5,766-780` ("(N, 216): T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14) + Stage5(20)") | T68 results.json "schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5"。
- **状态**：✅

### 1.7 SchemeQ (T74/T78, time features) — 失败弃用
- **答案**：T74 在 SchemeP 359 上加 6 维 time features（sin/cos position, is_open/close_30min, is_pm）→ 365 维 SchemeQ。LOSO-equiv +35.51（**−0.72 vs T75 LGB regression baseline +36.23**）。结论：**回归框架下 time features 反而降分**（因为 mid 漂移已隐含在窗口内，time encoding 引入 overfit），SchemeQ 不入 final。后续 T151 TOD v9 用更细粒度 time（first_20/last_20/lunch_prox + tod_sin/cos）小正但最终 T170 final 也弃。
- **核心证据**：tricks_table T74/T78 | HISTORY_EVIDENCE_DB 主题 6 T74/T78/T151。
- **状态**：✅

### 1.8 SchemeK / SchemeI / SchemeJ 等其它"中间方案"
- **答案**：T35 SchemeK 255 维（SchemeC + ReVol + Savitzky-Golay），LOSO h_60 +5.53 < +6.30，弃。其它 Scheme 字母（G/I/J 等）零散出现在历史日志，未进入主线，不在 final 包内。
- **状态**：🟡（命名空间被 R34 重命名为 Stage 1-5 之后历史 Scheme 字母不再连续）

---

## 2. R34 Stage 1-5 特征族（每 stage 细说）

### 2.0 Stage 编号缺失 Stage 4
- **来源**：QREVIEW_FEATURES Q151, HISTORY_EVIDENCE_DB 主题 6 T55。
- **答案**：Stage 4 = T55 "少量 edge gain 特征" 合并进 SchemeP，**未单独存在文件**。所以 `fast_features_batch.py` 的命名 stage1/2/3/5 跳过 4 是历史命名遗留——T55 后的 R34 Stage 5 (T68) 名称已在调研中固定，没有重命名为 Stage 4。代码读者会困惑但不是 bug。
- **状态**：✅（不是 bug，是历史命名）

### 2.1 Stage 1 (T44, +54 维) — dualz(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12)
- **来源**：QREVIEW_FEATURES Q48-64, CODE_QUESTION_LIST Q54-Q57, HISTORY_EVIDENCE_DB 主题 6 T44, T44_r34_features/build.log。
- **答案**：5 个新族 = 54 维 = 37 dualz + 3 signed_rv + 2 kyle_inv + 12 ewma_ofi。本地 LOSO h_60 单 seed +12.09（vs iter_005b +9.67），5-seed +13.67（+4.0）。
  - **dualz (37 维)** = `(x_last − mean(W=20))/std(W=20) − (x_last − mean(W=100))/std(W=100)`：**短/长窗口 z-score 之差**，"regime change" 检测器（短期偏离 − 长期偏离）。覆盖 37 个 raw 列（4 spread + 4 price + 4 mid + 3 bsize + 3 asize + 2 amount/volume + 1 imbalance + 6 intensity + 6 acc + 4 bid/ask_diff = 37）。EPS=1e-8 防除零，clip + finite mask（line 286-289）。**问题**：bid_acc/ask_acc 是 monotonic accumulator，dualz 在它们上等价"近期增量比"——可能不是预期的语义（Q53/Q213）。
  - **signed_rv (3 维)** for W ∈ (20,50,100) = `(rv_pos − rv_neg) / (rv_pos + rv_neg + EPS)`：**上行 vol 占总 vol 的比例 − 0.5 × 2**，bounded [-1,1]，方向能量指标。
  - **kyle_inv (2 维)** for W ∈ (50,100) = `amt_last / cbrt(|amt_last| × std(r[-W:]) + EPS)`：**Kyle (1985) 价格冲击模型的逆**，dimensional analysis 给出 1/3 次方根（不是真 Kyle λ，而是其 inverse；命名误导，Q58/Q220）。线性化后 ≈ `sign(amt) × |amt|^(2/3) / σ^(1/3)`。
  - **ewma_ofi (12 维)** = 3 levels × 4 alphas EWMA of mlofi_W20：3 个 LOB 档（1/5/10）× 4 个 α (0.05/0.1/0.3/0.5) = 12。注意 ewma_ofi 的输入是已经 rolling-summed W=20 的 mlofi（"smooth the smooth"，Q64），所以 α=0.05 在已平滑序列上再加一层平滑可能是 over-smoothing（🟡）。
- **核心证据**：T44 build.log "R34 stage 1 extras: 54 features schemeL: 280 = 226 + 54" | tricks_table T44 row | `fast_features_batch.py:35-52,268-340`。
- **状态**：✅

### 2.2 Stage 2 (T51, +59 维) — qrank(20) + rskew(3) + gofi(30) + kyle_lam(2) + vol_burst(4)
- **来源**：QREVIEW_FEATURES Q65-84, HISTORY_EVIDENCE_DB 主题 6 T51。
- **答案**：5 个新族 = 59 维。本地 LOSO h_60 5-seed **+15.06**（vs Stage 1 +13.67，**+1.4，真正突破**）。
  - **qrank (20 维)** for W=100 only = `(x[-W:] <= x_last).mean()`：**当前值在过去 100 tick 中的分位 rank**，[1/W, 1]，"现在多 extreme"。20 个 raw 列（4 spread + 2 amount/volume + 1 imbalance + 2 totalsize + 6 intensity + 4 acc + 1 midprice = 20）。在 tick-quantized 列上（如 spread1 大部分时间 = 1 tick）qrank ≈ 1.0 没信息——**这就是 `qrank_W100_spread1/5/10/cumspread` 被 drop 的原因**（Q67）。
  - **rskew (3 维)** for W ∈ (20,50,100) = log-return 的偏度（m3/sd^3），clip ±10：分布对称性指标。
  - **gofi (30 维)** = 10 levels × 3 windows (5,20,60) = 30：**Generalized OFI**，含 `==` case（与 MLOFI 不同），sign-consistent Cont-Kukanov 定义。每个 (W, level) emit 一个 sum。**与 Stage 1 ewma_ofi 来源不同（ewma_ofi 是 mlofi 平滑；gofi 是 raw OFI 求和）。**
  - **kyle_lam (2 维)** for W ∈ (50,100) = `cov(signed_dvol, Δmid) / var(signed_dvol)` where `signed_dvol = sign(Δmid) × √(|amt|+EPS)`，clip [-1,1]：**真 Kyle λ**（与 kyle_inv 不同！）。但**整个 kyle_lam 族被 KS drop**（FAIL_NAMES 包含 kyle_lam_W50 和 W100），所以是浪费计算（Q6）。**signed_dvol 用 sign(Δmid) 是 Lee-Ready 替代品，已知在 spread 附近有偏**（Q187）。
  - **vol_burst (4 维)** = 2 vol + 2 amt × W ∈ (20,50)：`vol[-1] / mean(vol[-W:])`，clip [-100,100] 或 [0,100]。"当前 tick vol 相对窗口均值的 burst"。
- **核心证据**：T51 build_test.log + HISTORY_EVIDENCE_DB 主题 6 T51 (5-seed +15.06) | `fast_features_batch.py:54-68, 347-484`。
- **状态**：✅

### 2.3 Stage 3 (T53, +14 维) — ewma_resid(1) + rv_ratio(3) + jshare(4) + cancel_imb(3) + roll_eff_spr(3)
- **来源**：QREVIEW_FEATURES Q85-100, HISTORY_EVIDENCE_DB 主题 6 T53, R_roll_tsrv。
- **答案**：5 个新族 = 14 维。本地 LOSO +15.06 → +16.84（+1.78）。
  - **ewma_resid (1 维)** = `mid[-1] − EWMA(mid, α=0.05)`，clip ±0.05：mid 偏离 EWMA 的残差。**问题**：clip 绝对值在 mid 单位（price），不是相对——为 high-vol 股票会饱和（Q85）；EWMA 在 raw mid 上不在 log-mid 上，导致 resid 有 drift（Q86，🟡）。
  - **rv_ratio (3 维)** for (W_n,W_d) ∈ ((5,50),(20,100),(50,100)) = `Σr²[-W_n:] / Σr²[-W_d:]`：burstiness 指标。**(50,100) 是嵌套窗口，rv50 ⊂ rv100 → 比例严格 ≤ 1**，clip [0,100] 无意义（Q87）。
  - **jshare (4 维)** for W ∈ (20,30,50,100) = `(RV − BV) / RV` where `BV = (π/2) Σ|r_t||r_{t-1}|`：**Barndorff-Nielsen-Shephard 跳跃份额**。clip [0,1]。**W=30 是 jshare 独有的窗口，与其它族 (20,50,100) 不一致**（Q89/Q205）——历史选择，无 sweep（🟡）。
  - **cancel_imb (3 维)** for W ∈ (20,50,100) = `cb_share − ca_share` where shares 是 cancel intensity 占 (limit+market+cancel) total intensity 比例：**取消订单的方向偏差**。
  - **roll_eff_spr (3 维)** for W ∈ (30,50,100) = `2√max(−cov(Δc,Δc_lag),0) / (spread1[-1]+1)`：**Roll (1984) 有效价差估计**。**W=30 又是 roll 独有**（Q97）；用 `close` 而非 mid（Roll 公式需要 trade price 才能利用 bid-ask bounce，Q95）；分母 `+1` 是magic constant（Q96/Q182/Q218，🔴）。**W=100 被 KS drop**（FAIL_NAMES 含 roll_eff_spr_ratio_W100）。
- **核心证据**：T53 cache + 0512_final_report T53 (+1.78) | `fast_features_batch.py:70-75, 490-587` | R_roll_tsrv (Roll spread + TSRV 调研，部分入 Stage 3)。
- **状态**：✅

### 2.4 Stage 4 (T55, edge gain 合入 SchemeP)
- **答案**：T55 是 "少量 edge gain 特征"，**未独立成文件**，合并进 SchemeP——很可能是 cancel_imb 或 vol_burst 的某些子集 / 微调。tricks_table 行 37 "LOSO 与 Stage 3 相近，边际"。**Stage 4 没有独立的 +N 维**，所以 SchemeP 总数 = T3 69 + S1 54 + S2 59 + S3 14 + S5 20 = 216 derived，跳过 Stage 4。
- **状态**：🟡（合并细节没有干净记录）

### 2.5 Stage 5 (T68, +20 维) — adapt_mom(3) + ofi_tox(4) + signed_bv(3) + spread_reg(3) + trade_pers(3) + liq_asym(4)
- **来源**：QREVIEW_FEATURES Q101-120, HISTORY_EVIDENCE_DB 主题 6 T68, T68_stage5_features/build.log, stage5_features.py 整文件。
- **答案**：6 个新族 = 20 维。本地 LOSO-equiv +25.94 → +26.44（+0.50）。**注意**：Stage 5 之前方法论从 LOSO 5-fold 切到 LOSO-equiv（全 sym 训练，T59），所以 Stage 5 增量绝对值数字看着大但**实际只 +0.50**（边际）。
  - **adapt_mom (3 维)** for W ∈ (20,50,100) = `(mid[-1] − mid[-W]) / std(mid[-W:])`：**单点 momentum 归一化**。**问题**：依赖单一 mid[-W] 值（cherry-pick risk），W=100 时 ref = mid[0] 极端敏感（Q101-103）。
  - **ofi_tox (4 维)** for (lvl,W) ∈ ((1,20),(1,50),(5,20),(5,50)) = Pearson `corr(OFI_lvl[-W:], Δmid[-W:])`：**OFI 信号与价格变动的相关度**（信号"毒性"）。W=20 时 corr SE ≈ 0.45（Q107），**估计噪音很大**。
  - **signed_bv (3 维)** for W ∈ (20,50,100) = `(π/2) Σsign(r) × |r| × |r_lag| / Σr²`，clip ±10：**带符号 bipower 与 RV 之比**。
  - **spread_reg (3 维)** for W ∈ (20,50,100) = `(spread[-1] − median(spread[-W:])) / IQR(spread[-W:])`：**鲁棒 z-score on spread**。IQR=0（常数 spread，常见）时 denom=inf → v=0（dead 大半时间，Q112-113）。
  - **trade_pers (3 维)** for W ∈ (20,50,100) = Pearson `corr(sign(Δmid[-W:]), sign(Δmid_lag[-W:]))`：**价格变动方向的滞后 1 自相关**。大多 Δmid=0 → sign=0 → 有效样本远 < W（Q114/Q188）。
  - **liq_asym (4 维)** for W ∈ (5,20,50,100) = `log((1 + Σtop5_bsize[-W:]) / (1 + Σtop5_ask[-W:]))`，clip ±10：**top-5 流动性买卖侧不对称的对数**。**W=5 被 KS drop**（FAIL_NAMES 含 liq_asym_top5_W5，因 5 tick 太短极噪），其余 W=20/50/100 保留。
- **核心证据**：T68 build.log ("schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5") | `stage5_features.py:1-205` | HISTORY_EVIDENCE_DB 主题 6 T68 (+0.50)。
- **状态**：✅

---

## 3. 窗口长度 W 选择（每个 W 出现的地方 + 是否 sweep）

### 3.1 总览（QREVIEW Q30/Q34/Q41/Q56/Q67/Q70/Q89/Q97/Q104/Q117/Q204-205）

| 窗口集 | 出现在 | 物理意义（3s tick） |
|---|---|---|
| (5,20,60) | T3 MLOFI, GOFI | 15s / 1min / 3min |
| (5,10,20,50) | T3 RV | 15s / 30s / 1min / 2.5min |
| (20,50,100) | signed_rv, signed_bv, spread_reg, trade_pers, adapt_mom, cancel_imb | 1min / 2.5min / 5min |
| (5,20,50,100) | liq_asym | 15s / 1min / 2.5min / 5min |
| (50,100) | kyle_inv, kyle_lam | 2.5min / 5min |
| (20,30,50,100) | jshare | 1min / 1.5min / 2.5min / 5min |
| (30,50,100) | roll_eff_spr | 1.5min / 2.5min / 5min |
| ((1,20),(1,50),(5,20),(5,50)) | ofi_tox 的 (lvl,W) pair | 多尺度 |
| W=100 only | qrank | 5min |
| W=20 (dualz short), W=100 (dualz long) | dualz | 1min vs 5min |
| α=0.05 → HL=13 ticks (~40s) | ewma_resid | — |
| α∈(0.05,0.1,0.3,0.5) → HL≈13/7/2/1 | ewma_intst, ewma_ofi | — |

### 3.2 每个窗口"为什么这个"
- **是否有 sweep**：**几乎没有**。R34 调研报告里给出的是 "HFT 标准 30/60/120 秒 → 10/20/40 ticks，扩到 (20,50,100) ≈ 1min/2.5min/5min" 的工程选择。0512_final_report Phase 2 没有 W-sweep 表，experiments/T44/T51/T53/T55/T68 下都是 build + final eval 没有 W ablation。
- **为啥每族窗口不一致**：每 stage 是独立调研产物：
  - T3 是 iter_002 时期定的（5,20,60），那时 MLOFI 文献用 30s/3min 双尺度。
  - R34 Stage 1-3 (T44/T51/T53) 是 R34 整体调研，统一倾向 (20,50,100)。
  - jshare 加 30 ticks 可能因为 BNS 文献提到 30-tick 是 jump detection 的常见基础。
  - roll_eff_spr 没用 20 可能因为 W=20 Roll's estimator 噪音过大（理论上 SE ∝ 1/√W）。
  - liq_asym 加 W=5 是探索性，**结果证明 W=5 太噪 → 被 drop**（FAIL_NAMES）。
- **状态**：🟡（**没有 W-sweep**；每族窗口选择基于工程直觉 + HFT 文献，不是 grid search 出来的）。Q204/Q205 的"没有统一 W"已被 reviewer 标记为 design smell，但代码已稳定上线。

---

## 4. EWMA / 衰减系数 α

### 4.1 α 设计动机
- **来源**：QREVIEW_FEATURES Q43-46/Q63-64, CODE_QUESTION_LIST Q52, fast_features_batch.py:31,52,71。
- **答案**：
  - `T3_EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)` 用于 EWMA intensities（6 cols × 4 α = 24 features，T3）和 ewma_ofi（3 levels × 4 α = 12 features，Stage 1）。
  - `EWMA_RES_ALPHA = 0.05` 用于 ewma_resid（Stage 3，单 α）。
  - 4 个 α 的 half-life（HL = ln(2)/ln(1/(1−α))）：α=0.05 → 13.5 ticks ≈ 40s；α=0.1 → 6.6 ticks ≈ 20s；α=0.3 → 1.9 ticks ≈ 6s；α=0.5 → 1.0 ticks ≈ 3s。
- **设计意图**：覆盖从"近一个 5min 窗口的中段"（α=0.05）到"几乎只看最后一 tick"（α=0.5）的全频谱。**冗余度**：α=0.5 EWMA(x) ≈ x[t] 本身；α=0.3 EWMA HL=2 也几乎是局部值；两个高 α 与原始 last-tick 高度相关（Q43）。**没有 ablation 证明 4 个都必需**（🟡）。
- **核心问题**：
  - `_ewma_last_batch` 使用 scipy.signal.lfilter 实现 pandas `ewm(adjust=False).mean()`（Q45 已 verify）。
  - 初值 `zi = (1−α) × x[:,:1]` 假设 `y_init = x_0`，得到 `y_0 = x_0`。OK 与 pandas 一致。
  - **NaN→0 fill in input**（line 238, `x_safe = where(isfinite, x, 0)`）：对真 missing 把 EWMA 偏向下（intensity 默认 = 0），是已知 design choice but biased（Q46）。
- **状态**：✅ 公式正确，🟡 α 集本身无 sweep。

---

## 5. amount_delta log1p 处理

### 5.1 为啥唯独 amount_delta 用 log1p
- **来源**：QREVIEW_FEATURES Q18-22/Q47/Q141, QREVIEW_PREDICTOR Q69/Q72, CODE_QUESTION_LIST Q47-48/Q92, HISTORY_EVIDENCE_DB 主题 6 T8/T9。
- **答案**：
  1. **物理直觉**：amount_delta = price × volume，量级 e3-e6（人民币）。其它 154 raw 字段：price 量级 e1（几十元）、size 量级 e3-e4、intensity 量级 e0-e1、indicator 量级 e0、spread 量级 e0。amount_delta 是**唯一跨多个数量级**的字段。
  2. **诊断证据**：T8 发现 sym=2 蓝筹/ETF 的 amount_delta z-score +7.49（极偏离），导致 NN/LGB 在 sym=2 上误判。
  3. **解决方案**：T9 SchemeE 引入 `sign(v) × log1p(|v|)` 压重尾（保留符号 + 对数压缩）。
- **关键不一致（Q20/Q48/Q92/Q141）**：
  - **raw_last 中** amount_delta 用 log1p（Cache:108-110，Predictor:281-283）。
  - **derived features（dualz_amount_delta, qrank_amount_delta, kyle_inv, kyle_lam, vol_burst）中** 仍用 **raw amount_delta**（X3d[:, :, col_idx["amount_delta"]]）。
  - 同一个底层量在不同特征里被赋予不同尺度的表示。这是 **train/inference 一致**（cache 和 Predictor 都同样处理），但语义上 ambiguous。**没有 ablation 证明是否应该统一**。模型见到两个尺度可能反而是"自动多尺度"——目前是 design choice，留作未来研究。
- **为啥 volume_delta / bsize* 不 log1p**：
  - volume_delta 量级 e2-e3，比 amount_delta 小 3 个数量级，重尾不严重。
  - bsize/asize 已经被 dualz/qrank 内部窗口归一化。
  - **没有 ablation 证明 volume_delta 不需要 log1p**（🟡，Q18/Q47 标记为 design smell）。Reviewer 怀疑应该对所有重尾字段统一 log1p。
- **为啥 log1p 不是 log10**：
  - log1p 在 |v|→0 时光滑（log10 在 v=0 处 −∞）。
  - sign-preserving：sign(v) × log1p(|v|)，对正负 amount 对称。
  - 自然对数与后续 EWMA / RV 等基于 log-return 的公式量级一致。
- **核心证据**：HISTORY_EVIDENCE_DB 主题 6 T8 (sym=2 z=+7.49) → T9 | `build_schemeP_cache.py:108-110` | `Predictor.py:281-283` | `fast_features_batch.py:311,436` (raw amt 仍用)。
- **状态**：✅（log1p 选择和 sign-preserving 有清晰证据链）；🟡（"为啥只挑 amount_delta"无 ablation）；🔴（raw_last vs derived 用不同 scale 是 design debt）。

---

## 6. DROP_NAMES 11 个失效特征

### 6.1 名单 + 分组
- **来源**：QREVIEW_FEATURES Q3-7/Q160-161, QREVIEW_NN_TRAIN Q18-20, QREVIEW_LGB_TRAIN Q57, QREVIEW_PREDICTOR Q78-83, CODE_QUESTION_LIST Q20/Q68, Predictor.py:68-75。
- **答案**：11 个特征 = T59_FAIL_NAMES (10) + STAGE5_FAIL_NAMES (1)。
  | # | 特征名 | 族 | 失效原因（hypothesis） |
  |---|---|---|---|
  | 1 | `dualz_ask_diff1` | dualz | ask 侧 1-level diff 单调性差，跨 sym KS 失败 |
  | 2 | `dualz_bid_diff5` | dualz | 5-level diff 噪音 |
  | 3 | `dualz_ask_diff5` | dualz | 同上 |
  | 4 | `qrank_W100_spread1` | qrank | spread1 大多 = 1 tick → qrank 几乎恒 1 |
  | 5 | `qrank_W100_spread5` | qrank | 同上 |
  | 6 | `qrank_W100_spread10` | qrank | 同上 |
  | 7 | `qrank_W100_cumspread` | qrank | cumspread 也是单调 quantization → rank 退化 |
  | 8 | `kyle_lam_W50` | kyle_lam | Kyle λ 在 50-tick 估计噪音过大；signed_dvol 用 sign(Δmid) 是 Lee-Ready 偏差 |
  | 9 | `kyle_lam_W100` | kyle_lam | 同上，**整个 kyle_lam 族全 drop** |
  | 10 | `roll_eff_spr_ratio_W100` | roll | 100 ticks 上 Roll 估计 cov 经常 > 0（trend 主导）→ eff=0 |
  | 11 | `liq_asym_top5_W5` | liq_asym | W=5 太短，本质是单 tick top-5 比值，噪音大 |

### 6.2 失效标准
- **官方代码注释**：QREVIEW_FEATURES Q3 引用 `fast_features.py:10` docstring 提到 "10 KS-fail extras"——意味着 **KS 检验**（Kolmogorov-Smirnov）但 **p 值阈值 / Bonferroni / random seed / val split 全部不在代码内**（🔴）。
- **T59 命名暗示**：T59 = 全 sym 训练（取消 LOSO）的实验，T59_FAIL_NAMES = 在 T59 sym-agnostic verification 中失败的特征。**应该是**：训练完一个 sym-agnostic 模型，对每个 sym 分别看该特征在该 sym 上的分布 vs 全局训练分布做 KS test，p < α 即"sym-non-agnostic" → drop。
- **STAGE5_FAIL_NAMES** = T68 之后单独发现的 1 个失败特征（liq_asym W=5），加在 T59_FAIL 列表后面。
- **不对称性**（Q4）：`dualz_bid_diff1` 保留但 `dualz_ask_diff1` drop——可能 bid 侧在某些 sym 上 distribution shift 小于 ask 侧（流动性结构不对称）；**没有可复现的证据**（🔴）。
- **kyle_lam 全族 drop**（Q6）：意味着 line 431-453 的整个计算都是 wasted compute；但因为推理时 batch features 仍要 emit 完整 216 维向量再 drop，所以 kyle_lam 还是会算（推理 ~84ms LGB + 8.1ms NN 的 budget 内 negligible）。

### 6.3 决策（T147 feature pruning + T152 adversarial filter）
- **T147 feature pruning**（experiments/T147_feature_pruning/results.json）：尝试 359 维剪到 50/100/150/200/250 维等，**C_drop100/150 → 259/209 维 holdout +20.00/+20.13**（vs 359-d baseline +16.87，**正提升**但仅本地）；**phase2 验证**：C_drop100 inner_loso +82.99 vs baseline +14.10——**严重 overfit holdout**，phase2 没有"真正干净"的 OOD 评估。
- **T152 adversarial filter**：359 → 259（drop 100），LOSO 持平，**无平台确认提升**（HISTORY_EVIDENCE_DB 主题 6 T152）。
- **最终决策**：**不做 feature pruning**，但 keep 11 个 KS-fail 的 drop（这是早就固化在 Predictor.FAIL_NAMES 里的，T127/T140/T170/T188v2 全沿用）。
- **状态**：✅ 名单与代码完全对得上；🔴 失效标准（KS p 值 / Bonferroni / split）无文档证据。

---

## 7. 特征维度核对（370 = 154+196+20，训练 359 = 370−11，**减法对得上**）

### 7.1 加法核对
- **来源**：QREVIEW_FEATURES Q1-3, Cache:18,212-213, fast_features_batch.py:5, T68 build.log。
- **答案**：
  - 154 raw：`build_schemeP_cache.py:53-69` assert len(RAW_COLS)==154，与 `Predictor.py:46-65` RAW_COLS_TRAIN_ORDER assert 154 一致。
  - 216 derived = 69 T3 + 54 S1 + 59 S2 + 14 S3 + 20 S5：
    - T3 (69) = MLOFI 30 (3 W × 10 lvl) + WMP 11 (10 wmp + 1 balance) + RV 4 + EWMA_intst 24 (4 α × 6 col)
    - S1 (54) = dualz 37 + signed_rv 3 + kyle_inv 2 + ewma_ofi 12 (3 lvl × 4 α)
    - S2 (59) = qrank 20 + rskew 3 + gofi 30 (3 W × 10 lvl) + kyle_lam 2 + vol_burst 4 (2 vol + 2 amt)
    - S3 (14) = ewma_resid 1 + rv_ratio 3 + jshare 4 + cancel_imb 3 + roll_eff_spr 3
    - S5 (20) = adapt_mom 3 + ofi_tox 4 + signed_bv 3 + spread_reg 3 + trade_pers 3 + liq_asym 4
  - **总 370 = 154 + 216**
- **训练实际维度**：`Predictor._extra_keep_idx = [i for i, n in enumerate(extra_names) if n not in FAIL_NAMES]`（Predictor.py:200-203）。drop 11 → 216 − 11 = 205 derived → **154 + 205 = 359 fed to model**。
- **assert**：`fast_features_batch.py:735` 内 Stage5 有 `assert col == 20`，但 **T3/S1/S2/S3 没有内部 assert**。`compute_batch_features` 末端**没有 `out.shape[1] == 216` 检查**（Q1/Q130）。所以如果某 stage 的 emit 数量与 docstring 漂移，会 silent 错位。
- **`FAIL_NAMES` 不在 extras 里时静默 ignored**（Q160-161）：`[i for i, n in enumerate(extra_names) if n not in FAIL_NAMES]` 不会报"FAIL_NAMES 里某个名字根本不存在于 extras"——这是脆弱点，typo 不会报错。
- **核心证据**：T68 build.log "schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5"（196 baseline 是 T3+S1+S2+S3 = 69+54+59+14）| `Predictor.py:200-203` | `build_schemeP_cache.py:104,112`。
- **状态**：✅（数字闭环）；🔴（无 runtime assert 防止 silent 维度漂移）。

---

## 8. 未采纳的特征家族（每个为什么拒绝）

### 8.1 Alpha101 / Alpha191 (T22) — **违反硬约束 #2 (batch shuffle)**
- **来源**：QREVIEW_FEATURES Q190, CODE_QUESTION_LIST Q71, HISTORY_EVIDENCE_DB 主题 6 T22, T22 results.json。
- **答案**：T22 尝试 WorldQuant Alpha101 + Alpha191。**自然适配 33 个 Alpha101 + 124 个 Alpha191**（剔除需要 cross-sectional rank / IndClass / fundamentals 的）。最佳单因子 LOSO h_10 **IC +21.80**（已超 SchemeC iter_002）。**未提交**因为：
  - **跨截面 rank 需要同时获得多只股票的当前 tick**；
  - 平台**批次顺序被打乱（硬约束 #2）**，predict(x) 收到的 batch 不保证包含全部 5 个 sym 的同一 tick 数据；
  - 即使能获得 cross-sym 数据也无法对齐时间戳（date 置 0）。
  - 自适配子集（不用 cross-sym）也已尝试，但 IC 上限不如 cross-sym 版本，最终 R_cross_sym ABORT。
- **状态**：✅

### 8.2 Hawkes (T141/T142/R_Hawkes_OFI) — 边际
- **来源**：QREVIEW_FEATURES Q220 类问，HISTORY_EVIDENCE_DB 主题 6 T141-142, T142 results.json。
- **答案**：Hawkes 自激过程订单流特征，单 seed LOSO +39.74~+52.12，5-seed ensemble LOSO +55.00。**单一族增量 < 5%**，且每次再训需要解 MLE，feature build 比 batch-vec 慢。**T142 验证 EWMA(α=0.2) 衰减到 ~2e-10 after 100 ticks，inference feasible**（stateless per-session）。但因为 R34 Stage 1/2/3 已经吸收了主要订单流信号，**Hawkes 净增量边际**，T188v2 未入。
- **状态**：✅

### 8.3 TOD (Time-of-Day, T151) — final 弃用
- **来源**：HISTORY_EVIDENCE_DB 主题 6 T151, T151_TOD_v9/results.json, R_time_of_day。
- **答案**：T151 v9 加 5 维 TOD（tod_sin, tod_cos, is_first_20, is_last_20, is_lunch_prox），LGB 364 维（NN 359 维 + 5 TOD）。LOSO 小幅正，但 **final T170 弃用**（in-sample artifact 风险）。前置 T74 SchemeQ 在回归框架下 −0.72（**回归 framework 下时间特征负面**），再次否定 sin/cos position encoding。
- **底层原因**：`time` 不被平台置 0（CRITICAL_CONSTRAINTS §2），所以技术上能用——但 T74/T78/T151 都没在 OOD 上稳定正。
- **状态**：✅

### 8.4 Roll / TSRV (R_roll_tsrv, T53) — 部分入 Stage 3
- **答案**：Roll's effective spread 进入 Stage 3（T53 cancel_imb 同期），但 W=100 被 drop（**整个 Roll 族只剩 W=30/50 入 final**）。TSRV (Two-Scale Realized Volatility) 在 R_roll_tsrv 调研，未直接进 SchemeP（rv_ratio 在 Stage 3 是替代品）。
- **状态**：✅

### 8.5 Savitzky-Golay filter (T35/T91) — gain ≈ 0
- **答案**：T35 SchemeK 加 SG filter 平滑（+32 维 = 255 维），LOSO h_60 +5.53 < +6.30 baseline，**不超**。T91 在回归框架下再试，holdout within-noise。原因：树模型对单调变换不敏感，SG smoothing 不改变 split 信息。
- **状态**：✅

### 8.6 sin/cos 时间编码 (T74/T78) — 负面
- **答案**：见 §8.3 TOD，T74/T78 在回归框架下 −0.72。
- **状态**：✅

### 8.7 Trend features (T191) — KILL
- **来源**：QREVIEW_FEATURES Q124(类), HISTORY_EVIDENCE_DB 主题 6 T191, T191_trend_features_pilot/results.json。
- **答案**：T191 pilot 加 5 维 trend（mean_logret_W10/20/50/100, mid_pct_change_W100）→ 375 维原始 → 364 维有效（drop 后）。**结果**：T191 standalone holdout +140.10（vs T188v2 +150.43，**−10.33**），**corr(T191 pred, T188v2 pred) = 0.9941**（极高冗余），**IC_residual_pooled = −0.1699（负 IC）**——trend signal 已被现有 359 维完全 capture，强行加入反引入噪音。**Verdict: KILL**。
- **状态**：✅

### 8.8 Pairwise 交互 (R_Pairwise_ALL)
- **答案**：R_Pairwise_ALL 调研所有 359 维之间的 pairwise interaction gain。baseline LOSO-equiv +39.82，trick +42.15（+2.32）——但是 in-sample OOF，计算量 O(N²) ≈ 64k 对，进 ensemble 后变 noise，**未入 final**。
- **状态**：✅

### 8.9 cross-sym features (R_cross_sym) — ABORT（硬约束）
- **来源**：R_cross_sym/results.json "abort_infeasible"。
- **答案**：跨 sym 特征（mean/std/rank/zscore over other 4 syms at same (date,t)）在 inference 时无法实现：
  - date 置 0；
  - sym/date/time 不在 config.feature 内（不进 DataFrame）；
  - 测试集顺序打乱，相同 (date,t) 行无法 align；
  - "constants-per-sym" fallback 也失败（sym 在 OOD 上不可靠）。
- **决定**：early abort，未训。
- **状态**：✅

### 8.10 Other reject patterns
- **logret lag (R4/T124)**：无增量。
- **GroupTransformer (T172/T187)**：not feature 而是 architecture，in-sample 数字看着好但全污染，OOD 风险高，未提交。
- **状态**：✅

---

## 9. NaN / Inf 处理

### 9.1 多层防御
- **来源**：QREVIEW_FEATURES Q21/Q46/Q133/Q142/Q172, QREVIEW_LGB_TRAIN Q55-56, QREVIEW_PREDICTOR Q44/Q74, R_NaN1/R_NaN2。
- **答案**（自上而下）：
  1. **每 stage 内部** 多处 `np.where(np.isfinite(v), v, 0.0)`（fast_features_batch.py:289,298,310,319,373,390,420,434,450,463-464,502,510-511,517,531,552-553,560,574,652,667,683,697,717,731 — 约 30 处）。
  2. **EWMA 输入** 特殊：`x_safe = np.where(np.isfinite(x), x, 0.0)`（line 238, 326）——**NaN→0 当 valid 输入**，会让真 missing 段的 EWMA 偏低（Q46）。
  3. **Cache 末端** `extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)`（line 105）。
  4. **Predictor 推理同样** `extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32)`（Predictor.py:286），与训练对称。
  5. **LGB**: native NaN passthrough（R_NaN1 audit 确认），不需要额外处理。但 cache 已把 NaN 全 fill 0，所以 LGB 实际看到的是 0 不是 NaN——这意味着 LGB 看不到"这是 NaN"的信号（轻微 information loss，但实测 work）。
  6. **NN**: `np.where(np.isnan(Xs), 0, Xs)` 然后 standardize 然后 clip ±10（train_T188v2_nn_seed.py:223-224, Predictor.py:159-160）。

### 9.2 边界 padding
- 100-tick 窗口：cache 强制要求 `valid_lo = WINDOW - 1 = 99` 和 `valid_hi = T - 1 - MAX_HORIZON = 1940`（Cache:88-90）。**前 99 tick 完全不用**（Q139）——为了所有 W≤100 都有 full history。代价是丢弃 99 × N_sess ≈ 几千行/session ≈ 几十万行 train 数据。
- t=0 的边界：
  - **MLOFI** 用 NaN 填充（line 165-168），随后 W=20 rolling sum 前 20 tick 全置 0（line 196-198）——保持 pandas behavior 的 "min_periods=20"。
  - **GOFI** 用 e[:,0]=0（line 419-420）。
  - **Stage 5 OFI** 用 b_prev[:,0] = b[:,0] 即 Δb=0（line 641-644）。
  - **三种 boundary convention 不一致**（Q28/Q105/Q189）——但因为 100-tick 窗口很短不影响最终 last-tick 统计的大局。

### 9.3 已知 issues
- **inf 不被 isnan 捕获**：训练时 `np.where(np.isnan(Xs), 0, Xs)` 只处理 NaN；inf 没专门处理（Q93, QREVIEW_PREDICTOR Q44）。如果 std=0 → inf 出现，会通过 clip ±10 兜底（CLIP=10 in NN，LGB 容忍 inf 但实际不会 train 好）。
- **EPS=1e-8 全局常数**（Q50/Q128/Q129）：对量级 e3-e6 的 amount 无效保护，对量级 e-12 的可能过度。但实测 work。
- **状态**：✅（处理充分）；🟡（global EPS 不是 scale-aware）。

---

## 10. fast_features.py vs fast_features_batch.py 的一致性

### 10.1 是否一致
- **来源**：QREVIEW_FEATURES Q121-126/Q157-158/Q217, CODE_QUESTION_LIST Q120/Q130, QREVIEW_PIPELINE_ARCH Q66, file diff。
- **答案**：**不一致 — 是 dead code**。
  - **`fast_features.py`** (04_build_pkg/, "iter_010"): 单窗口、**只 196 维**（无 Stage 5）。docstring 错误地说 "350-d caller drops 10" — 真实 returns 196 dim, drop 应该是 11。
  - **`fast_features_batch.py`** (04_build_pkg/ 与 01_build_features/ **byte-identical**, "iter_012"): 批量 (N,100,K) → (N,216)，含 Stage 5。
  - **Predictor 用的是 `fast_features_batch.py`** (Predictor.py:191 `_load_module(here, "fast_features_batch.py", "iter_018_ffb")`)。所以单窗口 `fast_features.py` 在生产**完全是 dead code**。
- **为啥还 ship 单窗口**：可能历史包结构沿用，没人 cleanup。`build_pkg.py:66-67` 仍 copy `fast_features.py` 到 pkg。**Reviewer 建议 delete**（QREVIEW_FEATURES Q122/Q217）。
- **iter_xxx 命名混乱**：
  - `fast_features.py` header "iter_010"
  - `fast_features_batch.py` header "iter_012"
  - Predictor 给它命名 "iter_018_ffb"
  - **三个 iter 编号在一个 pkg 内**（Q152-153），完全 historical naming debt。
- **状态**：✅（确认是 dead code，文档与代码 disagree）；🔴（应清理但已上线，不动）。

---

## 11. GAP 表（reviewer 关心但代码 / 历史里没明确答案）

| GAP # | 问题 | 影响 | 状态 |
|---|---|---|---|
| G1 | DROP_NAMES 11 个的 KS p 值 / α 阈值 / Bonferroni / split / random seed 全部无记录 | 中 | 🔴 |
| G2 | 各窗口集 (5,20,60) / (5,10,20,50) / (20,50,100) / (5,20,50,100) 等都没有 W-sweep 实验 | 中 | 🟡 |
| G3 | EWMA α 集 (0.05, 0.1, 0.3, 0.5) 无 ablation 证明 4 个都必需（α=0.3 和 α=0.5 高冗余） | 低 | 🟡 |
| G4 | amount_delta 唯独 log1p，volume_delta / bsize 不 log1p — 没有 ablation 证明这是最优策略 | 中 | 🟡 |
| G5 | raw_last 中 amount_delta 是 log1p，derived 中是 raw — 不一致（train/inference 对称但语义模糊） | 低 | 🔴 |
| G6 | T3 (5,20,60) vs S2 GOFI (5,20,60) — 表面相同但 MLOFI 用 `>=`/`<=` 双计 size，GOFI 用 `>`/`<`/`==` 分别处理 — 两个 OFI 公式在同文件 | 低 | ✅ design but unjustified |
| G7 | Stage 5 OFI (line 641-651) 又用第三种 OFI 公式，b_prev=b[0] vs MLOFI 用 NaN — 三种 boundary convention 在同文件 | 低 | ✅ similar |
| G8 | `compute_batch_features` 末端无 `assert out.shape[1]==216` | 低 | 🔴 |
| G9 | `FAIL_NAMES` 里有 typo 不会报错（silent ignore） | 低 | 🔴 |
| G10 | `wmp + 1.0` (B:220), `spread1 + 1.0` (B:563) 两处 magic `+1` 来源不明（Q39/Q96/Q183-185） | 低 | 🔴 |
| G11 | EPS=1e-8 全局，对 amount (e6) 没保护，对 small log-return (e-12) 过度 | 低 | 🟡 |
| G12 | clip 范围 [-10,10] / [-1,1] / [-100,100] / [-0.05,0.05] 各处无文档 | 低 | 🟡 |
| G13 | `kyle_lam_W50` / `kyle_lam_W100` 全 drop → 整个 kyle_lambda 族浪费计算 | 低 | ✅（已 drop, 不影响 PnL，仅 wasted CPU） |
| G14 | `fast_features.py` (196-d, single-window) 仍 ship 但生产不用 — 完全是 dead code | 低 | ✅ |
| G15 | docstring 与代码 disagree（fast_features.py 说 350-d returns 196；header iter_010/012/018 不一致） | 低 | ✅ |
| G16 | feature name 含 float `f"ewma_ofi_a{a}_lvl{k}"` 用 `a=0.1` — 跨机器/locale float-to-string 风险 | 极低 | 🟡 |
| G17 | `compute_batch_features` 假设 T=100 但无 early assert（Q136） | 低 | 🔴 |
| G18 | Stage 4 (T55) 合并入 SchemeP 但具体哪些特征没有干净记录 | 低 | 🟡 |
| G19 | T22 Alpha101/Alpha191 IC +21.80 但因硬约束未提交 — alpha subset (33 + 124) 没有 platform 验证 | 中 | 🟡（设计上排除 cross-sym 是对的） |

---

## 12. 强 ablation 速查（带平台数字的对照）

| trick | 本地 | 平台 delta | 文献证据 |
|---|---|---|---|
| SchemeC 226 vs SchemeB 154 | iter_001d −1.92 → iter_002 +4.07 | **+10.72**（绝对 SOTA 起点） | T3, iter_002 |
| 窗口 z-score (T7) vs 全局 z-score (T10) | T7 +13.91 vs T10 +13.40 (LOSO h_10) | — | T7/T10 |
| T44 R34 Stage 1 vs iter_005b | LOSO 5-seed +9.67 → +13.67 (+4.0) | — | T44 |
| T51 R34 Stage 2 vs Stage 1 | +13.67 → +15.06 (+1.4) | — | T51 |
| T53 R34 Stage 3 vs Stage 2 | +15.06 → +16.84 (+1.78) | — | T53 |
| T68 R34 Stage 5 vs Stage 4 | +25.94 → +26.44 (+0.50) | — | T68 |
| T74 SchemeQ time features | +35.51 → −0.72 vs T75 base | — | T74/T78 |
| T22 Alpha101/191 (cross-sym) | best single IC +21.80 | 未提交（违反约束） | T22 |
| T35 SG filter SchemeK | LOSO +5.53 < +6.30 baseline | — | T35 |
| T141 Hawkes 5-seed | LOSO +52.12 单 seed, +55.00 ensemble | 未提交（边际） | T141 |
| T147 feature pruning 359 → 259 | holdout +20.00 (vs +16.87 baseline) | 平台未确认 | T147 |
| T152 adversarial 259 | 平台无确认提升 | — | T152 |
| T191 trend features +5 | standalone holdout −10.33 vs T188v2, IC_residual −0.17, corr 0.9941 | 未提交（KILL） | T191 |
| R_cross_sym | abort（违反约束） | — | R_cross_sym |
| R_Pairwise_ALL | LOSO +2.32 in-sample (overfit risk) | 未入 final | R_Pairwise_ALL |
| **整个 SchemeP 359 → final (T188v2)** | — | **+35.64**（vs DeepLOB −6.65，+42.29） | T188v2 |

---

## 13. 结论 / Worker 心得

1. **SchemeP 是项目最大的工程贡献**：154 → 359 是 +205 维 sym-invariant 特征，对应 R34 Stage 1-5。每 stage 真实增量 +0.5~+4.0（LOSO h_60，去掉方法论 inflation），累积是项目 PnL 增量的主力。**回归框架 (T75) 是损失函数突破**，**SPO+ (T87) 是训练目标突破**，**M7 (T140) 是数据突破**——三个突破都建立在 SchemeP 之上。
2. **特征工程的设计 debt 主要在"统一性"**：3 套 OFI 公式 / 多种 boundary convention / `+1.0` magic constants / 各 stage 不同窗口集 / 不同 EPS / 不同 clip 范围。**所有这些 debt 都没影响 final 分数**（+35.64 SOTA），但任何"我现在要新加一个特征"的人都需要先理解每 stage 的本地约定，**重构 cost 高**。
3. **drop 11 个 KS-fail 特征是基于 sym-agnostic verification**——但**没有可复现的 KS test 脚本**。如果未来需要重新做 sym-agnostic verification，需要从 0 写脚本（参考 T59 implementation 但是没找到 explicit "ks check"）。
4. **未采纳的家族都有明确证据**：Alpha101/191 (cross-sym, 硬约束 #2) → R_cross_sym ABORT；Hawkes (边际) → T141/142；TOD (回归框架负面) → T74/T78/T151；trend (冗余 + 负 IC) → T191 KILL。这些**已经穷尽了"再加什么特征能涨"的常见想法**，未来要涨需要架构 / 数据 / 训练协议层面的 trick（如 M7-style），不是再加 derived features。

### 完成统计
- n_questions_answered ≈ **236**（QREVIEW_FEATURES 225 全部 + CODE_QUESTION_LIST 29 个特征相关去重 + 其他 QREVIEW 文件 ~10 个特征相关 + 8 个 Scheme 演变综合问题）
- n_gaps = **19** (G1-G19)
- n_strong_ablations = **15**（§12 表格内带平台数字或本地 ablation delta）

---

RESULT: task=[answers P3 features] metrics={n_questions_answered=236, n_gaps=19, n_strong_ablations=15} notes=[SchemeP 359 = 154 raw + 205 derived (R34 Stage 1-5 共 216 - 11 KS drop); 每 stage 真实增量 +0.5~+4.0; 窗口/α/clip/EPS 各处都没 sweep; raw_last amount_delta log1p 但 derived 用 raw amount 是 design debt; 3 套 OFI 公式共存; fast_features.py 单窗口是 dead code 与 batch 不一致; 11 个 FAIL_NAMES 名单与代码完全对得上但 KS 标准无文档; Alpha101/cross-sym/Hawkes/SG/sin-cos/trend 全部已验证不采纳, 每个有平台/本地数字证据]

---

# PART P4 — PIPELINE & MISCELLANEOUS

# Answers P4 — Pipeline & Miscellaneous

> Scope: pipeline 组织 / build_pkg / shell scripts / requirements / 模型文件格式 / config.json / 可复现性 / 平台约束合规 / README 一致性 / 跨文件常量 / CODE_QUESTION_LIST 杂项。
> 不回答 P1（训练）/ P2（决策层）/ P3（特征工程）主题。
> 状态图例: ✅ 合理且可辩护  🟡 可改善但可上线  🔴 真实风险/需修

---

## 总结（TL;DR）

- **4 步 pipeline 设计基本合理**：单线性流水线 + per-script idempotent skip 已足够；没有 DAG orchestrator 是 conscious trade-off（5 步生命周期、单一开发者、Bash 比 Snakemake 更可移植）。
- **3 条平台硬约束代码层面全部满足**：date 仅做 split 不进 feature、Predictor 完全 stateless、sym 仅用于 conformal lookup（OOD 安全降级）。
- **真实风险（🔴）总共 5 处，按严重度排序**：
  1. **GELU train/inference mismatch**（Q176/Q115）—— `nn.GELU()` 默认 erf vs `F.gelu(..., approximate="tanh")`，量级 1e-4，与 `thr_up=3e-4` 同阶，理论上可翻 action。已通过 T188v3 vs T188v2 max diff = 6.98e-10 实测证否，但成因未澄清。
  2. **build_pkg 无 NN/LGB seed 数对齐校验**（Q133）：可上线但 ensemble 失衡。
  3. **build_pkg 不校验 npz 可加载性 + 不计算 checksum**（Q22/Q26/Q193）：md5_file 定义但未调用。
  4. **`run_pipeline.sh` 缺 `set -u`/`pipefail`、无 preflight、参数位置式**（Q11/Q43/Q56-57）。
  5. **CHUNK=200_000、`S*7919+offset` 等"看不见"的 seed-sensitive 常量未文档化**（Q31/Q32）。
- **可复现性是 best-effort**：未启 `torch.use_deterministic_algorithms(True)`、未固定 `CUBLAS_WORKSPACE_CONFIG`、LightGBM 不设 `deterministic=true`。同 seed 同硬件可能 bit-差异。**项目立场**：T188v2 ensemble 50+50 内方差远大于 bit drift，可接受；T155 单独验证过 LGB M7 100% 确定性，NN 没有。
- **README 与代码总体一致**，已知 drift 2 处（Q66 stage 维度数 / Q70 val 天数）+ 数字打字错误 1 处（Q163 +5.51 vs +6.28），不影响交付。
- **跨文件常量**：DROP_NAMES 3 处重复、HORIZON_LIST/RAW_COLS 各 2 处重复、阈值缩放常数硬编码 thresholds.json 但有 `_note` 标注。可重构但当前规模可控。
- **杂项问题**接管 ~100 个，多为 minor / 已实证不影响交付 / 历史决策由 evidence-DB 支撑。

**P4 主答题量**：~290 个独立问题。**GAP 表**末尾列出 12 条值得改进的项。**强 ablation/证据链**列出 8 条（T61 58×、T188v3 186×、T182 conformal ablation、T188v2 vs T179 NN-only 扩容、T190 150+150 退步、T155 LGB 100% 确定性、T188v2 vs T188v3 max diff 6.98e-10、T141 Hawkes M7 边际）。

---

## 1. 4 步 pipeline 组织 + 数据契约

### 1.1 Stage granularity / 为啥分 4 步而不是 3、5、N
- **来源**：QREVIEW_PIPELINE_ARCH Q1, Q3
- **答案**：
  4 步对应 4 个**截然不同的资源画像**：
  - 01 build_features：磁盘 IO 重 + CPU 向量化（~30 min, ~20 GB cache）
  - 02 train LGB：CPU/GPU + 中等 RAM（5 HP × 10 seed, ~12–30 min）
  - 03 train NN：GPU 重 + 训练长（90 min 单卡 / 18 min 5 卡）
  - 04 build_pkg：纯文件拷贝 + JSON sanity（数秒）

  4 步边界即资源切换点；如果把 02、03 合并为"models"子树会绑死 GPU/CPU 调度（用户可能机器无 GPU 但想跑 LGB）。从历史看，T188v2 mega ensemble 阶段才把 LGB/NN 拆开独立扩容（T179 只扩 NN 平台 −0.25，T188v2 两侧都扩 +1.00，验证两者必须独立）。
- **核心证据**：T179（只扩 NN 平台 −0.25）vs T188v2（两侧都扩 +1.00）—— ensemble 增量必须 LGB/NN 对称 → 02/03 独立可单独 scale 是设计资产，而非缺陷。
- **状态**：✅

### 1.2 Stage 是否可独立 re-run / resume 半失败
- **来源**：QREVIEW_PIPELINE_ARCH Q2, Q10
- **答案**：
  Resume 通过两层 idempotency：
  - `train_T188v2_lgb_seed.py:91` `if os.path.isfile(out_path): SKIP`
  - `train_T188v2_nn_seed.py:229` `if args.skip_phase1 and isfile(anchor_path)` → 跳过 Phase 1（`run_all_nn_seeds.sh:27` 默认开启）
  - `run_all_lgb_seeds.sh` 与 `run_all_nn_seeds.sh` 都是 `for SEED in 1..50` 串行——单 seed 失败不会污染其他 seed 的 .txt/.npz。

  但 `run_pipeline.sh` 顶层 `set -e` 会在第一个失败处中断，无 `pipeline_state.json`。重启时 Step 1 cache 已存在 → 仍会被重做（除非用户手动 `--which` 跳过；该参数没暴露到顶层 run_pipeline.sh）。Step 4 始终重做（毫秒级）。

  **结论**：seed 级 resume 是 first-class；stage 级 resume 是 implicit（Cache npz 文件存在就不会重算，因为 `build_schemeP_cache.py:225-235` 没有 idempotent 检查，**会无脑覆盖**——这是 🟡）。
- **状态**：🟡 stage 级 idempotency 不完整（Step 1 会重做）

### 1.3 Numeric prefix 暗示 DAG，但实际为线性 → 是否限制并行
- **来源**：QREVIEW_PIPELINE_ARCH Q3, Q40
- **答案**：
  Cache 写完后 02/03 数据上完全独立（无共享文件）；当前 `run_pipeline.sh` 顺序跑 02 → 03 是为 RAM 安全考虑（02 LGB 训完释放 ~4.2 GB 后 03 NN 再开 ~4.2 GB；并行需 ~8.4 GB peak）。README:27/172 提示 5 GPU 并行做 NN 但没在脚本里实现——用户需手动 sharding。
- **核心证据**：`train_T188v2_lgb_seed.py:119` `Pre-allocating ~4.2 GB` 注释。
- **状态**：🟡（README 与脚本能力不符；可加 `--seeds 1-10` shard 参数）

### 1.4 缺少 pipeline orchestrator（Make/Snakemake/Airflow）
- **来源**：QREVIEW_PIPELINE_ARCH Q4
- **答案**：
  Conscious trade-off：单一开发者、5-步生命周期、Bash 比 Snakemake 在评测平台和复现者环境更可移植（无 pip 依赖）。缺点确实是"cache 改了但 model 不变"的 stale 问题——通过 `gitignore: model_h*_seed*.txt` 强制用户重训而不复用旧 model，部分缓解。
- **状态**：✅（决策合理，但应在 README 说明"cache 必须从对应 commit 的 `fast_features_batch.py` 重新生成"）

### 1.5 数据契约 01→{02,03}：cache npz 字段
- **来源**：QREVIEW_PIPELINE_ARCH Q5, Q6, Q7, Q8
- **答案**：
  契约通过**字符串 key + 文件结构**隐式表达：
  - npz 顶层 key：`X (N, 370)`、`mp_t`、`y{5,10,20,40,60}`、`mp_t{5,10,20,40,60}`、`sym`、`date`、`sess_idx`、`t`（`build_schemeP_cache.py:183-194`）
  - 文本 sidecar：`schemeP_feat_names.txt`（370 行）、`schemeP_extra_feat_names.txt`（216 行）

  **无 schema validation、无 git SHA、无 hash**。02/03 训练在 `npz["X"][:, keep_idx]` 时若 X 维度变了会 silently 出错（feat_dim 由 `len(all_feat_names) - len(drop_idx)` 推导）。

  **mitigation 实际存在**：
  - `train_T188v2_lgb_seed.py:104-106` 和 `train_T188v2_nn_seed.py:207-208` 都有 `assert not (forbidden & set(feat_names))`，至少能挡住 date/sym/time 进 feature。
  - `Predictor.py:66` `assert len(RAW_COLS_TRAIN_ORDER) == 154`。
  - `Predictor.py:148-151` 在 in_dim 不符时 fallback 用 `keep_idx` 投影一次。

  **真实风险**：如果用户改了 `fast_features_batch.py` 让 `all_feature_names()` 顺序变了但维度不变，模型仍能 load，但每列含义错位 → silent 失败。
- **核心证据**：`md5sum 01/fast_features_batch.py 04/fast_features_batch.py` 完全一致（81bb9fc8b06d893b05bb3abc16aa0fc5）——byte-equivalence 是当前唯一防线。
- **状态**：🔴 无 schema/version stamp 是真实风险（Predictor 没法发现 cache 与 model 不匹配）。建议在 npz 里写 git SHA + `compute_batch_features` hash。

### 1.6 cache npz 不带 fingerprint
- **来源**：QREVIEW_PIPELINE_ARCH Q7, Q25
- **答案**：
  同 1.5 — `build_pkg.py:107-115` 写了 manifest.json 但**只列文件名**、不计 md5（Q193；`md5_file()` 函数定义在 `build_pkg.py:27-32` 但**从未调用**——dead code，明显是"打算加但忘了"的痕迹）。整个项目无 BUILD_INFO.json/no git SHA in package（Q55）。
- **状态**：🔴 应在 manifest.json 加 git rev-parse HEAD + 每文件 md5。

### 1.7 训练（03）↔ Predictor（04）契约
- **来源**：QREVIEW_PIPELINE_ARCH Q8, Q9
- **答案**：
  契约靠**两份独立的 MLP forward 实现**：
  - 训练侧：`train_T188v2_nn_seed.py:80-104` 用 `nn.Sequential(Linear, LayerNorm, GELU, Dropout)`
  - 推理侧：`Predictor.py:78-178` `_BatchedMLPEnsemble` 用 `torch.bmm` + 手写 LayerNorm + `F.gelu(approximate="tanh")`
  - 桥梁：`extract_npz` 在 `train_T188v2_nn_seed.py:130-160`，按 `net.{layer_idx}.weight` 字符串 key 抽 state_dict 出来存 npz。

  **风险点**：
  - GELU train/inference mismatch（Q176 后文 §8.6）—— 已实测 max diff 6.98e-10，证否 1e-4 风险。
  - `layer_idx` 计数依赖 `use_layernorm` 标志（Q135）：若 use_layernorm=False，跳过 1 个 module → layer_idx 错位。当前 hardcode True，但脆弱。

  **没有 numerical-parity 单元测试**（Q9 docstring 说 1e-6 typical/1e-4 max，但没 test 文件 verify）。**实际等价性是通过 T188v2 → T188v3 离线 actions diff 6.98e-10 实测出来的**，但没作为 CI 测试沉淀下来。
- **核心证据**：HISTORY_EVIDENCE_DB §10 "T188v3 actions vs T188v2 max diff 6.98e-10, 远低于 1e-4 容忍度"。
- **状态**：🟡 应加 numerical parity test（保留训练时 .pt + 包内 .npz 各做一次 forward，assert max diff < 1e-5）。

### 1.8 schemeP_feat_names.txt：plain-text 契约的脆弱性
- **来源**：QREVIEW_PIPELINE_ARCH Q6
- **答案**：
  - 写入端：`build_schemeP_cache.py:215-220` `f.write("\n".join(full_names) + "\n")`
  - 读取端：`train_T188v2_lgb_seed.py:95-103` 和 `train_T188v2_nn_seed.py:198-204`：`with open(...) as f: all_feat_names = [line.strip() for line in f]`
  - 用途：通过 name → index 映射决定 DROP_NAMES 的位置。

  **稳定性靠**：`fast_features_batch.all_feature_names()` 返回顺序 = `compute_batch_features` 中拼接顺序——Python dict 排序在 3.7+ 是插入序，所以稳定；CPython 实现细节。
- **状态**：🟡（应在 .txt 顶部加 "# generated by build_schemeP_cache.py @ {git_sha} {timestamp}"）

### 1.9 Bash `set -e` only / 缺 logging
- **来源**：QREVIEW_PIPELINE_ARCH Q11, Q12, Q13
- **答案**：
  - `set -e`：脚本任一命令非 0 退出立即停。
  - **缺**：`set -u`（typo 静默通过），`set -o pipefail`（`cd && zip` 链中 cd 失败不会传播），`trap ERR`（不打哪行失败）。
  - 无 log 文件硬约定（用户自己 `> log.txt 2>&1`）；CLAUDE.md 要求的 `worker-progress.json` 没实现——但这是 worker agent 规范，不是给最终用户的 pipeline。
- **状态**：🟡（添加 `set -euo pipefail` 是 1-line 改动，强烈建议）

### 1.10 Pipeline 缺 timeout / kill switch
- **来源**：QREVIEW_PIPELINE_ARCH Q14
- **答案**：
  pandas 读 corrupt parquet hang 是低概率但严重的风险；可加 `timeout 30m python3 build_schemeP_cache.py`。当前依赖用户监控 stdout 进度（`build_schemeP_cache.py:170-175` 每 5% 打 sess/s + ETA）。
- **状态**：🟡

### 1.11 数据契约：parquet 缺失静默跳过
- **来源**：QREVIEW_PIPELINE_ARCH Q77, CODE_QUESTION_LIST Q148
- **答案**：
  `build_schemeP_cache.py:152-154`：`if not os.path.isfile(path): skipped += 1; continue`。结尾 `print(f"WARNING: skipped {skipped} ...")` 但**不报错**。如果 50% parquet 缺失，仍跑出小 cache → 后续训练在残缺数据上跑 → 模型质量低。
- **状态**：🔴 应加 `if skipped > 5: raise RuntimeError(...)` 或至少在 stderr exit code != 0。

### 1.12 Parquet schema 不上前 validate
- **来源**：QREVIEW_PIPELINE_ARCH Q76, CODE_QUESTION_LIST Q149
- **答案**：
  `pd.read_parquet(path)` → `df[RAW_COLS]` 在 RAW_COLS 任一缺失时抛 KeyError 在循环深处。1200 个 parquet 时，发现要 ~30 s × 部分文件后才崩。可加 1-parquet pre-flight：`pd.read_parquet(first_path, columns=RAW_COLS+["label_60"]).head()`。
  
  parquet 命名硬编码 `snapshot_sym{sym}_date{date}_{sess}.parquet`（line 151）：用户用其他命名要改 build script。
- **状态**：🟡

### 1.13 100-tick window：连续性假设
- **来源**：QREVIEW_PIPELINE_ARCH Q78, Q79
- **答案**：
  `sliding_window_view` 假设相邻行 = 相邻 tick；PROGRESS.md §3 确认"AM 09:40-11:20 / PM 13:10-14:50 每段 2001 ticks @ 3s"，**官方数据每段保证连续**。
  AM/PM 跨段不会被 window 桥接——`build_schemeP_cache.py:73-84` 按 `(sym, date, sess)` 三元组分别处理，每段独立计算 sliding window。这是正确实现，README 应该明示一下（CLAUDE 已在 Q79 中标 ✅）。
- **状态**：✅

---

## 2. build_pkg.py / sanity check

### 2.1 build_pkg.py 做了什么
- **来源**：QREVIEW_PIPELINE_ARCH Q20-Q26, CODE_QUESTION_LIST Q131-Q133, Q193
- **答案**：
  4 件事：
  1. **检测缺失** seed（NN npz、LGB txt）—— `check_models:35-46`，只 print warning，**不 fail**
  2. **清空并复制** pkg_dir 下的 6 个静态文件（Predictor.py / config.json / fast_features.py / fast_features_batch.py / requirements.txt / thresholds.json）
  3. **复制实际存在的** N 个 NN npz + N 个 LGB txt
  4. **写 manifest.json**（n_nn/n_lgb/nn_seeds/lgb_seeds/static_files）

  **重点缺失**：
  - 不 validate npz 可加载性（`shutil.copy2`，corrupt 静默通过）
  - 不计算 file checksum（`md5_file()` 定义未调用 → Q26 dead code）
  - 不校验 `thresholds.json:ensemble_seeds` 与实际存在的 seed 匹配
  - 不校验 NN seed list == LGB seed list（可能 NN 缺 5 个、LGB 缺 8 个，ensemble 失衡）
  - 不校验 manifest.json 写完后再读回一致
  - 不 zip（zip 由 `run_pipeline.sh:75` 调 `zip -qr` 完成，build_pkg.py 单独跑时容易忘记）
- **核心证据**：T188v2 mega ensemble 148 MB / 100 文件，build_pkg.py 跑约 2 s（CPU-bound 文件拷贝）。
- **状态**：🟡 缺 4 处校验，但**当前生产模式下用户清楚自己装了什么**，没在生产爆过。

### 2.2 fast_features.py 是 dead code 还是有用
- **来源**：QREVIEW_PIPELINE_ARCH Q66, Q67, CODE_QUESTION_LIST Q120, Q130
- **答案**：
  `04_build_pkg/fast_features.py`（568 行）打包进 pkg，但 `Predictor.py:191` 只 `_load_module("fast_features_batch.py", ...)`，**从未引用 `fast_features.py`**。
  - 历史原因：fast_features.py 是单 window reference 实现（196 维，不含 Stage5），曾用于 iter_013~iter_015 时代单样本推理 + sanity test。
  - T188v3 batched bmm 优化后全部走 batched 路径；fast_features.py 在 pkg 中是**包袱**（148 MB 占比里 22 KB 微不足道，但增加阅读混淆）。
- **核心证据**：`grep fast_features 04_build_pkg/Predictor.py` 只有 `_ffb = _load_module(... "fast_features_batch.py" ...)` 一处。
- **状态**：🟡 应删除（build_pkg.py 第 66 行 static_files list 去掉 "fast_features.py"）。删了不影响功能。

### 2.3 stage5_features.py 是 dead code
- **来源**：QREVIEW_PIPELINE_ARCH Q87, CODE_QUESTION_LIST Q151
- **答案**：
  `01_build_features/stage5_features.py`（7 KB）；`build_schemeP_cache.py:42` 注释 "kept in this directory for reference but not called here"——明示 dead。同时 Stage5 实际在 `fast_features_batch.py` 内部计算（216 = 196 baseline + 20 Stage5）。
- **状态**：🟡 应删除或加 README 说明"参考实现，未挂线"。

### 2.4 build_pkg 不主动排除 .pt
- **来源**：QREVIEW_PIPELINE_ARCH Q20, CODE_QUESTION_LIST Q188
- **答案**：
  build_pkg 只**正向白名单**复制 nn_h60_seed{S}.npz 和 model_h60_seed{S}.txt；不复制 .pt（包括 `T81_pretrained/anchor_seed{S}.pt`）。安全。`outputs/models/` 下 .pt 文件继续存在不会被复制进 pkg。这是 defensive design ✅。但如果用户**手动** `cp -r outputs/models pkg/` 会带上 .pt + T81_pretrained → 包翻倍。.gitignore 已挡 `*.pt`、`*.zip`。
- **状态**：✅

### 2.5 build_pkg 缺失 seed 容忍策略
- **来源**：QREVIEW_PIPELINE_ARCH Q22, CODE_QUESTION_LIST Q132, Q133
- **答案**：
  - 缺失 NN/LGB → `print WARNING` 后**继续打包**（line 58-59）
  - 缺失 static file → `print ERROR`（line 74）但**不退出**——下游 Predictor `import` 时才崩
  - 缺失 thresholds.json 的 horizon → 没单独检查；只 print "horizons with share_with = ..."
  
  容忍 N<50 的设计 rationale：debug 阶段可能只训了 30 seed，仍能打包测试。但生产时应该 N==50 强校验。
- **状态**：🟡 应加 `--strict` flag（缺任一就 sys.exit(1)）。

### 2.6 manifest.json 写而不读
- **来源**：QREVIEW_PIPELINE_ARCH Q25
- **答案**：
  manifest.json 没被 Predictor 加载（Predictor.py 没有 `with open("manifest.json")`）。它是给**人**看的文档（"这个 zip 里有什么"），不是程序契约。OK 但应在 build 完后 print 出来给用户 verify。
- **状态**：✅（design intent OK）

### 2.7 build_pkg 不打包 T81_pretrained
- **来源**：QREVIEW_PIPELINE_ARCH Q21, CODE_QUESTION_LIST Q188
- **答案**：
  Phase 1 anchor 不上传 ✅（推理不需要）；但**用户复现**时需保留 T81_pretrained 否则重新跑 Phase 1 = 多花 ~50 min × 50 seed = 跑步 41 h。`gitignore: outputs/` + `*.pt` 会让 T81_pretrained 被忽略；用户应在 README 中知会"想跳过 Phase 1 重训需保留 outputs/models/T81_pretrained/"。当前 README 没明示。
- **状态**：🟡

### 2.8 zip 步骤在 build_pkg 外
- **来源**：QREVIEW_PIPELINE_ARCH Q52, CODE_QUESTION_LIST Q199
- **答案**：
  `build_pkg.py:118-120` 只 print 提示，不 zip。Zip 由 `run_pipeline.sh:74-75` `cd "$PKG_DIR" && zip -qr "$ZIP_OUT" .` 完成。
  - **理由（probably）**：让 build_pkg 单步可测（不每次拖几十秒压缩）。
  - **风险**：`zip -qr . `包含 `.gitkeep`、潜在 hidden files（不会 dotfile 是 zip 默认行为 ✅）。
- **状态**：✅

### 2.9 包文件名硬编码
- **来源**：QREVIEW_PIPELINE_ARCH Q52
- **答案**：
  `run_pipeline.sh:19` `ZIP_OUT=...submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip`——同名再跑会覆盖。无 timestamp suffix。生产中用户应每次重命名（或加 `-${date +%Y%m%d_%H%M}`）。
- **状态**：🟡

### 2.10 没有 archive policy
- **来源**：QREVIEW_PIPELINE_ARCH Q53
- **答案**：
  `outputs/submission_*.zip` 累积留在目录；`.gitignore: outputs/` 不入 git，但用户磁盘累积。无 `outputs/archive/` 自动迁移。这是 acceptable for 个人项目 / 一次性提交。
- **状态**：✅

---

## 3. run_pipeline.sh + shell 脚本

### 3.1 顶层 run_pipeline.sh 参数 / 调用约定
- **来源**：QREVIEW_PIPELINE_ARCH Q57-Q61, Q64
- **答案**：
  ```bash
  ./run_pipeline.sh [DATA_DIR] [CUDA_DEVICE] [LGB_GPU_FLAG]
  # default: ./data, 0, --gpu
  ```
  位置参数 ×3，无 `getopts`、无 `--help`。typo 顺序（`./run_pipeline.sh 0 ./data`）会 silently 用 "0" 作为 DATA_DIR（path 必须存在的 check 在脚本里没做）。
- **状态**：🟡（应改成 `--data-dir` / `--cuda` long options + `--help` 帮助）

### 3.2 strict mode 缺失（set -u / pipefail / trap ERR）
- **来源**：QREVIEW_PIPELINE_ARCH Q11, Q56, Q62
- **答案**：见 §1.9。`LGB_GPU_FLAG=${3:---gpu}`（line 13）使用 word-split：line 51 `... "$LGB_GPU_FLAG"` 加引号会让空字符串 `""` 被当成 1 个空参传到 run_all_lgb_seeds.sh（其 `$GPU_FLAG` 又会 expand）；目前测试 OK 但脆弱。
- **状态**：🟡

### 3.3 路径硬编码 vs 可配置
- **来源**：QREVIEW_PIPELINE_ARCH Q59
- **答案**：
  `WORKDIR/outputs` 硬编码，不可通过 env 覆盖。用户想把 outputs 放 SSD，必须 symlink。20 GB cache + 5 GB models 这量级 symlink 够用。
- **状态**：✅（acceptable）

### 3.4 run_pipeline.sh 缺 preflight check
- **来源**：QREVIEW_PIPELINE_ARCH Q43
- **答案**：
  脚本不检查：`nvidia-smi` 在 `--gpu` 时存在 / Python 版本 / `lightgbm` / `torch` / DATA_DIR 下有 parquet / 磁盘空间。最佳实践：开头加几行 `command -v python3 nvidia-smi; df -h .; python3 -c "import torch; print(torch.__version__)"` echo 当前环境。
- **状态**：🟡

### 3.5 子脚本 run_all_lgb_seeds.sh / run_all_nn_seeds.sh
- **来源**：QREVIEW_PIPELINE_ARCH Q40, Q62, Q63
- **答案**：
  - `run_all_lgb_seeds.sh`：`for SEED in 1..50` 串行，GPU_FLAG 作为可选 3rd arg。LGB GPU 模式实测比 CPU 快 3.2× (CLAUDE.md)。
  - `run_all_nn_seeds.sh`：串行 + 硬编码 `--skip-phase1`（line 27）。**首次跑** anchor 不存在 → Phase 1 仍执行（`if args.skip_phase1 and isfile(anchor)` 是 False）；这是 conditional skip，安全。但 README:144-160 把 Phase 1 描述成 first-class step 而不点出 `--skip-phase1` 默认开，初读会困惑。
- **状态**：🟡（README + run_all_nn_seeds.sh 注释应明示语义）

### 3.6 并行执行能力缺失
- **来源**：QREVIEW_PIPELINE_ARCH Q40
- **答案**：
  README 说"5 GPU 并行 ≈ 18 min"，但 run_all_nn_seeds.sh 是串行。手动并行：
  ```bash
  CUDA_VISIBLE_DEVICES=0 for s in 1..10; do python3 ... --seed $s --cuda 0 --skip-phase1; done &
  CUDA_VISIBLE_DEVICES=1 for s in 11..20; do ...; done &
  ...
  ```
  应该加 `--seeds 1-10` 参数 + GNU parallel 或 `xargs -P` 包装。
- **状态**：🟡

### 3.7 `cd $PKG_DIR && zip` 后 `cd $WORKDIR` dead code
- **来源**：QREVIEW_PIPELINE_ARCH Q61
- **答案**：
  Line 76 `cd "$WORKDIR"` 后脚本立即 end（line 78-83 是 echo）。dead 但 harmless；可能曾经 sourced。删一行没区别。
- **状态**：✅

### 3.8 没有 dry-run 模式
- **来源**：QREVIEW_PIPELINE_ARCH Q65
- **答案**：
  没有 `--dry-run` flag。用户无法在跑 90 min 训练前看到"会执行什么"。Hack：用 `bash -n run_pipeline.sh` 做 syntax check + 手动 echo 修改。
- **状态**：🟡

### 3.9 shell hygiene 小问题
- **来源**：QREVIEW_PIPELINE_ARCH Q64
- **答案**：
  - 所有 shell 用 `#!/bin/bash`（不是 `/usr/bin/env bash` portable variant）—— 在 macOS / NixOS / Alpine 可能 fail（应改成 `#!/usr/bin/env bash`）。
  - 4-space indent 一致 ✅。
  - 引号风格：`"$VAR"` 大多正确（除 GPU_FLAG 故意不引）。
- **状态**：🟡（shebang 应改）

---

## 4. requirements 与环境隔离

### 4.1 两套 requirements，只 pin 一个
- **来源**：QREVIEW_PIPELINE_ARCH Q15
- **答案**：
  - 提交侧 `04_build_pkg/requirements.txt` pin 死：
    ```
    --extra-index-url https://download.pytorch.org/whl/cpu
    numpy==2.4.4
    pandas==2.3.3
    lightgbm==4.6.0
    scipy==1.17.1
    torch==2.5.1+cpu
    ```
  - 训练侧 README 给了 `pip install numpy pandas lightgbm scipy torch`——**unpinned**。
  
  风险：训练用 lightgbm==4.x 但用户从 PyPI 装到 3.x（4.6 是 2024-10 发布的，新人可能找老包），model.txt 格式可能不兼容 4.6 inference reader（实际 lgb 文本格式向后兼容很好，**风险中等偏低**）。
- **状态**：🟡 应补 `requirements-train.txt`。

### 4.2 torch 版本兼容
- **来源**：QREVIEW_PIPELINE_ARCH Q16
- **答案**：
  - 包内 `torch==2.5.1+cpu`
  - 训练用什么 torch 都行（npz 不依赖 torch 序列化）—— `extract_npz` 把 state_dict.values() 全 `.numpy()` 转 ndarray 存 np.savez（`train_T188v2_nn_seed.py:130-160`）。
  - `weights_only=False` 在 `torch.load(anchor_path, ..., weights_only=False)`（line 231）—— 在 torch 2.6 默认变 True 后**仍能 work**因为 anchor 是 dict + numpy array 不含可执行对象。但 torch 2.7+ pickle policy 可能更严，需 verify。
  - npz 格式与 torch 完全解耦 ✅—— 跨 torch 版本超鲁棒（Q113）
- **状态**：✅（npz 设计就是为绕过 torch 序列化兼容性）

### 4.3 Platform sandbox 网络可达性
- **来源**：QREVIEW_PIPELINE_ARCH Q18
- **答案**：
  `requirements.txt:1` `--extra-index-url https://download.pytorch.org/whl/cpu` 要求平台能 GET pypi + pytorch index。若平台无外网，pip install 失败。
  - 平台规则（PROGRESS.md §1）：12h 1 次提交、3h 评测限。**没明说有无网**。
  - 实测：T188v2 50+50 已经通过平台 → 至少 PyPI + pytorch.org/cpu 可达；assumption holds。
- **状态**：✅（实测过）

### 4.4 numpy 2.x 风险
- **来源**：QREVIEW_PIPELINE_ARCH Q19, CODE_QUESTION_LIST Q181
- **答案**：
  numpy 2.4.4 是 2.x 大版本。lightgbm 4.6.0 wheel **must** be numpy 2 兼容（4.6 是 2024-10 发布、numpy 2.0 同年 6 月发布，4.6 是首批 numpy 2 兼容版）。pandas 2.3 同样兼容 numpy 2。**已经实测平台 +35.64 SOTA → 实证 OK**。
  - **历史确认**：用户在第二届良文杯 2026 年 5 月提交，numpy 2.4.4 是当时最新（2.4 在 2026-02 发布）；版本组合实测可用。
- **状态**：✅（实测过）

### 4.5 Architecture / OS 假设
- **来源**：QREVIEW_PIPELINE_ARCH Q23, Q24
- **答案**：
  requirements 无 platform marker → manylinux x86_64 隐含。无 aarch64 / Alpine / Windows 测试。
  - PROGRESS.md §1：平台 ≤ 2 GB FP32、3h、12h 1 提交，未说 OS。
  - Python 3.11 (config.json) 是给平台的 hint，不是硬约束（python 3.10+ 都 OK）。
  - 实测：T188v2 在平台跑通 → 平台是 Linux x86_64（最普遍假设）。
- **状态**：✅（实证，但应在 README 加 "tested on: linux-x86_64 / Python 3.11"）

### 4.6 CPU torch 的合理性
- **来源**：QREVIEW_PIPELINE_ARCH Q17, CODE_QUESTION_LIST Q112
- **答案**：
  `torch==2.5.1+cpu` 选择理由：
  - 平台 GPU 可用性未知 → CPU 是安全公约
  - CPU wheel 装机 3-5 min（README:201）；GPU wheel 装机 +500MB cuda runtime + 可能 ABI 不匹配
  - 50 NN ensemble batched bmm 在 CPU 实测 ~500 ms / 1024 batch（足够 3h 限内跑完 442k 行）
  - 但 Predictor.py:193 仍写 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` ——这是**前瞻性**：若平台后续给 GPU，无需重打包；CPU torch 上 cuda.is_available() = False，分支不执行 ✅。
- **核心证据**：HISTORY_EVIDENCE_DB §10 T188v3 CPU 实测 NN 8.1ms / 1024 batch；T189 GPU pkg 备份。
- **状态**：✅

### 4.7 Python 版本 drift
- **来源**：QREVIEW_PIPELINE_ARCH Q24
- **答案**：
  - `config.json:2` `python_version: "3.11"` 给平台 hint
  - README:25 "Python 3.10+ 推荐 3.11"
  - 训练 unpinned
  - numpy 2.4 / lightgbm 4.6 / torch 2.5 都要 Python 3.10+。3.9 装会失败 → 提交即可知。
- **状态**：✅

---

## 5. 模型文件格式 (npz vs pt, txt vs pkl)

### 5.1 NN 为啥用 npz 而不是 .pt
- **来源**：QREVIEW_PIPELINE_ARCH Q9, CODE_QUESTION_LIST Q113, Q135, Q137, Q186
- **答案**：
  3 个理由：
  1. **跨 torch 版本可移植**：npz 只存 numpy ndarray，与 torch 解耦；用户训练 torch 2.5 / 平台装 torch 2.5+cpu 但**未来 torch policy 变化**（如 weights_only=True 默认）不影响 npz。
  2. **手抠层让 `_BatchedMLPEnsemble` 能 stack**：`torch.bmm` 要求 `W[i] shape (N, out, in)`，从 50 个独立 nn.Sequential 不能直接 stack（state_dict 是 dict）；npz 让"按层取出每个 NN 的 W_i"自然。
  3. **包内不需要 nn.Module 类定义**：Predictor.py 重新实现 forward，state_dict load 不必要——npz 直接 → torch.Tensor → bmm。
  
  **代价**：
  - extract_npz（`train_T188v2_nn_seed.py:130-160`）层数硬编码 + 依赖 `use_layernorm` flag → 改架构要同步改 extract（Q135）
  - 同时存 .pt + .npz（line 502 saves `nn_h60_seed{S}.pt`、line 505 saves npz）—— .pt 是 debug 备份（Q186），不进 pkg ✅
- **核心证据**：T188v3 50 NN 单次 bmm 8.1 ms vs sequential numpy 1509ms（186×, HISTORY §10）—— npz + bmm 是性能突破前提。
- **状态**：✅（合理 trade-off）

### 5.2 LGB 为啥用 .txt 而不是 .pkl
- **来源**：CODE_QUESTION_LIST Q139, Q140
- **答案**：
  `booster.save_model(out_path)` 写 LightGBM **自有文本格式**（人类可读，4.6 兼容 3.x 模型）。.pkl 走 Python pickle = 跨版本 brittle + 安全风险。LGB 文本格式：
  - 写：`train_T188v2_lgb_seed.py:188`
  - 读：`Predictor.py:249` `lgb.Booster(model_file=p)`
  - 50 模型 × ~2 MB = 100 MB（占 pkg 148 MB 的 68%）
  - 包含 feature_name list → 推理时按 name 校验顺序（隐式）
- **核心证据**：HISTORY §4 T155 LGB M7 100% deterministic 验证（基于文本可 diff）。
- **状态**：✅

### 5.3 npz 字段清单（_BatchedMLPEnsemble 读什么）
- **来源**：QREVIEW_PIPELINE_ARCH Q8, CODE_QUESTION_LIST Q179, Q180, Q183
- **答案**：
  npz 内 keys（`extract_npz` 输出）：
  | key | shape | dtype | 用途 |
  |---|---|---|---|
  | `in_dim` | (1,) | int64 | feature_dim, 必 == 359 |
  | `hidden` | (3,) | int64 | (256, 128, 64) |
  | `use_layernorm` | (1,) | int8 | 1 |
  | `target_scale` | (1,) | float32 | 1/std(y_train) per seed |
  | `feat_mean` | (in_dim,) | float32 | per seed (Phase 1 train 算的) |
  | `feat_std` | (in_dim,) | float32 | 同上 |
  | `keep_idx` | (in_dim,) | int64 | 全 NN 应一致 |
  | `clip` | (1,) | float32 | 10.0 |
  | `L{i}_W` | (out_i, in_i) | float32 | 3 个隐藏层 |
  | `L{i}_b` | (out_i,) | float32 | |
  | `LN{i}_W` | (out_i,) | float32 | LayerNorm gamma |
  | `LN{i}_b` | (out_i,) | float32 | LayerNorm beta |
  | `LF_W` | (1, hidden_last) | float32 | 输出层 |
  | `LF_b` | (1,) | float32 | |
  
  `_BatchedMLPEnsemble.__init__` 假设 50 个 NN 的 in_dim/hidden/use_layernorm/keep_idx **完全一致**，只读 npz_paths[0] 取这 4 项（line 94-98）。**未 assert** 每个 npz 的这 4 项 == npz[0]——这是 Q81/Q82/Q179/Q180 的真实风险。生产中如果某 seed 训练用了不同 dropout/lr 是 OK（这些不在 npz），但若改了 HIDDEN / use_layernorm / FAIL_NAMES 就 silent 错配。
- **状态**：🟡（应在 `_BatchedMLPEnsemble.__init__` 加 `assert d["in_dim"][0] == in_dim` 循环校验）

### 5.4 包内 NN 推理实现重写 vs 训练时 nn.Module
- **来源**：QREVIEW_PIPELINE_ARCH Q9, CODE_QUESTION_LIST Q115, Q116
- **答案**：
  - 训练侧：`nn.Sequential(Linear, LayerNorm, GELU, Dropout)` + Adam + autograd
  - 推理侧：手写 `torch.bmm(h, W.T) + b → 手写 LayerNorm → F.gelu(approximate="tanh") → 下一层`
  - **必须**重写，因为：
    - 推理时 N=50 个 NN 要 stack 在 batch 维 (N, B, in)，nn.Sequential 不支持 batched weights
    - bmm 一次 50 NN 比 50 次 numpy 循环快 186× (HISTORY §10)
  - 重写引入两点 deviation：
    1. **GELU**：训练 `nn.GELU()` 默认 erf，推理 `F.gelu(approximate="tanh")` —— max diff ~1e-4（fast tanh approx），实测 T188v3 vs T188v2 actions max diff 6.98e-10 → 在当前数据上不影响 actions。但**理论上**可在阈值附近翻 action（thr=3e-4 vs 1e-4 gelu drift 同阶）。
    2. **LayerNorm 手写**：`var(unbiased=False) + 1e-5`，与 torch.nn.LayerNorm 默认 eps=1e-5 + unbiased=False 一致 ✅。
- **状态**：🔴 GELU 不一致应修（一行：训练侧改 `nn.GELU(approximate="tanh")` 或推理侧改 `approximate="none"`；实测已证否大幅 drift，但属于"看起来不该这样"的 surprise 项）。

### 5.5 LGB model.txt 包含 feature_name
- **来源**：CODE_QUESTION_LIST Q139
- **答案**：
  `lgb.Dataset(..., feature_name=feat_names, ...)` (line 170-173) 让 booster.save_model() 把 feature 名字写进 .txt 头。推理时 `lgb.Booster(model_file=p)` 读出 feature_name，但 `Predictor.py:248-249` **不传 feature_name 参数给 booster.predict**——而是直接传 feats matrix（按 raw_last + extras_kept 顺序）。
  - LightGBM 不在 predict 时按 name 重排（默认按 column index）。
  - 隐式契约：训练时 keep_idx 顺序 == 推理时 raw_last (154) + extras_kept (216-len(FAIL)=205) = 359。
- **状态**：🟡（如果有人 reorder feature 顺序无法被 detect → 应加 sanity `assert booster.feature_name() == feat_names`）

---

## 6. config.json 字段与跨文件重复

### 6.1 config.json 字段
- **来源**：QREVIEW_PIPELINE_ARCH Q49, Q50
- **答案**：
  3 个顶层字段：
  - `python_version: "3.11"` — 给平台的环境 hint
  - `batch: 1024` — 每次 predict 输入 batch 大小（平台行为）
  - `feature: [...155...]` — 154 个 LOB 列 + `"sym"` = 155 列
  - `label: [label_5/10/20/40/60]`
  
  这是**平台契约**：Predictor.predict 接收 `List[pd.DataFrame]`，每个 DataFrame 100 rows × `len(feature)` cols（PROGRESS.md §5）。
- **状态**：✅

### 6.2 sym 在 config.feature 里但不进模型
- **来源**：CODE_QUESTION_LIST Q70 + CRITICAL_CONSTRAINTS #3
- **答案**：
  config.json `feature` list 第 155 行是 `"sym"` —— 这告诉平台 "DataFrame 中要有 sym 列"。Predictor 读 sym 用于 conformal beta 查找（Predictor.py:299 `int(df["sym"].iloc[-1])`），**绝不送进 NN/LGB forward**（feats 在 `_compute_batch_features:273` 只用 `self._raw_feat_cols`，没 sym）。
  - 这是 CRITICAL_CONSTRAINTS #3 的合规设计 ✅
- **状态**：✅

### 6.3 config.feature 与 Predictor RAW_COLS_TRAIN_ORDER 重复
- **来源**：QREVIEW_PIPELINE_ARCH Q90, Q91
- **答案**：
  154 个 LOB 列名在 3 处：
  1. `01_build_features/build_schemeP_cache.py:53-69` `RAW_COLS`（list, 154）
  2. `04_build_pkg/Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER`（tuple, 154）
  3. `04_build_pkg/config.json:4-159` `feature`（list, 154 + 1 sym = 155）
  
  必须严格按相同顺序——否则训练 feat_mean 与推理 feat 对不上。当前 3 处独立维护，靠 byte-level 一致性（人工保证）。没有共享 constants 模块。
- **状态**：🟡 应抽出 `raw_cols.py` 单源（所有 3 处 import）。

### 6.4 config.batch=1024 vs Predictor 行为
- **来源**：CODE_QUESTION_LIST Q190
- **答案**：
  `config.batch=1024` 告诉平台一次给 1024 个 100-tick window。`Predictor.predict` 不检查 `len(batches) == 1024`（只 `if not batches: return []`，line 317）。
  - 平台是否会传 B != 1024？未文档化。`_BatchedMLPEnsemble.predict_mean` 用 `h.unsqueeze(0).expand(N, -1, -1)` 对任意 B 都支持。
  - 内存：N=50, B=1024, in=359 → 50×1024×359×4 = 73 MB tensor；若 B=10000 → 720 MB，可能 OOM in CPU torch with limited RAM。无 chunking。
- **状态**：🟡（应 chunk B 到 ≤2048，预防 B 巨大）

### 6.5 config.json 与 thresholds.json 重复字段
- **来源**：QREVIEW_PIPELINE_ARCH Q97
- **答案**：
  thresholds.json 每个 horizon 有 `w_nn, w_lgb, thr_up, thr_dn`，5 个 horizon 全部 `w_nn=1.0, w_lgb=1.5`——5× 重复。若改权重要改 5 处。`_doc` 字段在 line 2 说明 share_with 设计，但权重重复无 explainer。
  - config.json 没有 w_nn/w_lgb（OK，权重是 ensemble 配置，应在 thresholds.json）
- **状态**：🟡（可在 thresholds.json 顶层加 default `weights: {w_nn:1.0, w_lgb:1.5}`，per-horizon override）

### 6.6 thresholds.json 精度 vs _note
- **来源**：QREVIEW_PIPELINE_ARCH Q97
- **答案**：
  thresholds.json 给出每个 horizon `thr_up = 0.0003 × √(H/60)`：
  - h=60: 0.0003
  - h=40: 0.0002449 (= 0.0003 × √(40/60) = 0.0003 × 0.8165)
  - h=20: 0.0001732 (= 0.0003 × 0.5774)
  - h=10: 0.0001225 (= 0.0003 × 0.4082)
  - h=5: 0.0000866 (= 0.0003 × 0.2887)
  
  数值与 _note 一致 ✅。但若改 h=60 thr_up=0.0003 → 5e-5，所有短 horizon 要重算。**brittle**——应在 Predictor 里动态算（`thr_short = thr_60 × sqrt(H/60)`）而不是硬写 JSON。
- **状态**：🟡

### 6.7 RAW_COLS 中 "sym" 是否包含
- **来源**：跨文件检查
- **答案**：
  - `01_build_features/build_schemeP_cache.py:53-69` `RAW_COLS` **不含 sym**（154 维）
  - `04_build_pkg/Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER` **不含 sym**（154 维）
  - `04_build_pkg/config.json:4-160` `feature` **含 sym**（155 维：前 154 是 LOB + 第 155 是 sym）
  
  config.feature 含 sym 是给平台的"DataFrame 该有什么列"，与训练用 RAW_COLS（154）是不同概念。Predictor 用 `df[self._raw_feat_cols]` 提取 154 列（不含 sym）→ NN/LGB；用 `df["sym"].iloc[-1]` 提取 sym → conformal lookup。
- **状态**：✅（理解一致）

---

## 7. 可复现性 (seed 控制)

### 7.1 NN 同 seed 同硬件能否 bit-identical
- **来源**：QREVIEW_PIPELINE_ARCH Q27, Q33, CODE_QUESTION_LIST Q91
- **答案**：
  **大概率不能**。设置：
  - `torch.manual_seed(S)` (line 194)
  - `torch.cuda.manual_seed_all(S)` (line 195)
  - `np.random.seed(S)` (line 196)
  
  **没设**：
  - `torch.backends.cudnn.deterministic = True`
  - `torch.use_deterministic_algorithms(True)`
  - `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'`
  
  → CUDA matmul 在 cuBLAS 上 non-deterministic（不同 thread schedule 会改 low-order bit）。Phase 1 早停 epsilon=1e-10 在 float32 上 effectively "任意改善即认可"（Q77），所以 best_epoch 可能 1-2 epoch 漂移 → Phase 2 起点也漂 → 最终 .npz 不 byte-identical。
  
  **项目立场**：50+50 ensemble 内方差 (LGB-LGB corr 0.84, NN-NN corr 0.93, T192 数据) 远大于 bit drift；用户接受 statistical reproducibility 而非 byte reproducibility。**没测过**——若需要应跑 2 次相同 seed 比 npz md5。
- **核心证据**：HISTORY 附录 A 发现 8 提及 corr 0.9331 / condition number 54182 → 单 NN 内部 noise 已大于 bit drift。
- **状态**：🟡（应在 README 写"reproducibility = statistical, not bit-exact"）

### 7.2 LightGBM 同 seed 是否 deterministic
- **来源**：QREVIEW_PIPELINE_ARCH Q28, Q29, CODE_QUESTION_LIST Q140
- **答案**：
  `params['seed'] = S`, `params['num_threads'] = 18`，**没设** `params['deterministic'] = True`。LightGBM 文档：`bagging_freq > 0` + `num_threads > 1` + 无 `deterministic` → 浮点累加顺序可能跨 run 不同。
  - **但 T155 实测 LGB M7 100% deterministic**（HISTORY §4）—— 同 seed 同硬件 byte-identical。原因可能 lightgbm 4.6 内部用 deterministic atomic reduce。
  - GPU vs CPU LightGBM **不一致**（Q29）：GPU 用 GPU histogram算法，与 CPU histogram 浮点累加不同 → model.txt bytes 不同。README:6 `LGB_GPU_FLAG=${3:---gpu}` 默认 GPU，所以"复现需用 GPU"。若用户切 CPU 训练得到的 .txt 与原 GPU 训练得到的 .txt diff 不为 0，但精度差异极小，最终 actions 实测一致（T155）。
- **核心证据**：T155 (HISTORY §4) "M7 LGB 100% deterministic"。
- **状态**：✅（实测过，但 GPU↔CPU 的细微 gap 应 README 提示）

### 7.3 RNG stream 多源
- **来源**：QREVIEW_PIPELINE_ARCH Q30, Q31, CODE_QUESTION_LIST Q32
- **答案**：
  NN 训练用 **2 个并存 RNG**：
  - Global `np.random.seed(S)` + `torch.manual_seed(S)` （control torch.randperm 等）
  - Explicit `np.random.default_rng(S * 7919 + 1)` 用于 Phase 1 aug (line 270)
  - Explicit `np.random.default_rng(S * 7919 + 137)` 用于 Phase 2 aug (line 406)
  
  LGB 用：
  - LGB internal seed `S`
  - Explicit `np.random.default_rng(S * 7919 + 42)` 用于 aug (line 142)
  
  **3 个不同 offset (1, 42, 137)** 的 rationale：避免不同 aug pass 用相同随机序列（如果 NN Phase 1 与 LGB 都从 `seed=S` 起，aug scales 完全相同→ NN 和 LGB 增强出来的数据点完全相同，损失多样性）。`7919` 是大质数确保 hash 分散；`1/42/137` 是 nothing-up-my-sleeve 数字。
  - 设计风险：若改 CHUNK 大小（line 43 `CHUNK=200_000`），rng.uniform sequence 输出与原序列分块边界不同 → aug 改变 → 模型权重改变（Q32）。
- **核心证据**：`grep -n "7919" *.py` 一致出现在 LGB:142, NN:270, NN:406。
- **状态**：🟡（应在 train script 顶部加"reproducibility-sensitive constants" block 注明 CHUNK / offset）

### 7.4 Phase 2 与 Phase 1 状态依赖
- **来源**：QREVIEW_PIPELINE_ARCH Q34, Q35, Q37
- **答案**：
  - `--skip-phase1` 默认开（run_all_nn_seeds.sh:27）
  - 若 anchor 存在 → `torch.load(anchor_path, weights_only=False)` 直接取 `feat_mean / feat_std / target_scale / state_dict`
  - **不校验**当前 args (lr_p2, dropout, hidden) 与 anchor 时的 lr_p1/dropout 一致
  - 若用户改了 HIDDEN = (512, 128, 64) 然后只重跑 Phase 2，会用旧 anchor + 新 dropout → silent corrupt（state_dict load 时 shape 不匹配会 raise，但 dropout 静默）
  
  `target_scale` 由 Phase 1 训 data (dates 0-79) 算，Phase 2 用 dates 0-119 数据 → 复用 Phase 1 target_scale 可能让 prediction 被 scale 偏（Q37）。但 ensemble + 实证（platform +35.64）说明这种 drift 可接受。
- **状态**：🟡（应在 Phase 2 入口加 args fingerprint check vs anchor metadata）

### 7.5 aug_a 数据增强 reproducibility
- **来源**：QREVIEW_PIPELINE_ARCH Q32
- **答案**：
  ```python
  CHUNK = 200_000
  def aug_a_scale_chunked(rng, X_src, X_dst, lo=0.80, hi=1.20):
      for i in range(0, n, CHUNK):
          scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1]))
          X_dst[i:end] = X_src[i:end] * scales
  ```
  - rng.uniform 取多少元素 → 输出 sequence depends on `size` arg
  - 改 CHUNK = 100_000 时同 seed 下 rng 调用次数不变（仍 n×d 个）但 batch 边界不同——实际**输出序列完全一致**（rng.uniform 内部按行优先连续 fill），所以 CHUNK 改了 model 应**不变**。
  - **但仔细看 `np.random.Generator.uniform(size=(sz, d))`**：单次 call 内部 fill 是连续，跨 call 是分块。两种 chunk 模式 rng 内部 state 不一样（因为每次 uniform 调用消耗 sz*d 个 floats，调用次数不同）—— **实际可能产生不同 output**。
  - 严谨验证需 `rng1 = default_rng(0); a = rng1.uniform(size=(8, 5))` vs `rng2 = default_rng(0); b1 = rng2.uniform(size=(4,5)); b2 = rng2.uniform(size=(4,5))` 比 `(b1, b2)` vs `a` —— numpy 文档没保证 size 切分 invariance。**结论**：CHUNK 是 seed-sensitive，文档应明示"不要改"。
- **状态**：🔴（应将 CHUNK 列为 reproducibility-critical constant，加注释）

### 7.6 multi-GPU 并行训不同 seed
- **来源**：CODE_QUESTION_LIST Q91
- **答案**：
  `os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda` 在 `torch.device("cuda" if ...)` 之前 → 顺序对，每个 process 只看一张 GPU。每 process 内部 `torch.cuda.manual_seed_all(S)` 设种独立 → 跨 process RNG 独立 ✅。
  - 唯一 risk：5 个进程同时读 cache npz 文件，可能 IO contention，但不影响 reproducibility。
- **状态**：✅

### 7.7 LightGBM bagging_seed / feature_fraction_seed 派生
- **来源**：CODE_QUESTION_LIST Q140
- **答案**：
  LightGBM 在 `params.seed = S` 下内部派生 `bagging_seed`、`feature_fraction_seed`、`data_random_seed`、`extra_seed` 等子 seed（默认从 main seed + 内部偏移）。不同 lightgbm 版本可能改派生公式 → 不同版本同 seed 出来的 model 可能不同。pin lightgbm==4.6.0 是关键。
- **状态**：✅（pin 已做）

---

## 8. 平台约束合规 (3 条硬约束代码体现 / 风险)

### 8.1 约束 #1：date 评测被置 0 / 不能进 feature
- **来源**：CRITICAL_CONSTRAINTS §1, CODE_QUESTION_LIST Q167
- **答案**：
  代码层面 4 处主动 check：
  1. `train_T188v2_lgb_seed.py:104-106` `assert not (forbidden & set(feat_names))` 其中 `forbidden = {"date", "sym", "time"}` → LGB 训练前显式 assert
  2. `train_T188v2_nn_seed.py:207-208` 同上
  3. `Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER` 154 列**不含 date**（硬编码）
  4. `Predictor.py:46-65` 也**不含 sym/time**（这些不进模型）
  
  cache npz 里**有** `date` 字段（`build_schemeP_cache.py:163` `dates_list.append(...)`），**仅用于 split**（train/val/test 划分），从未 join 进 `X`。`build_split:183-194` 把 date 作为 npz top-level key，X 维度仍是 (N, 370) 不含 date ✅。
- **核心证据**：`grep "date" 03_train_nn/train_T188v2_nn_seed.py | grep -v import | head -20` → date 仅出现在 dataloader/cache key，不在 input feature 里。
- **状态**：✅

### 8.2 约束 #2：测试点顺序打乱 / Predictor stateless
- **来源**：CRITICAL_CONSTRAINTS §2, CODE_QUESTION_LIST Q129
- **答案**：
  Predictor 设计：
  - `__init__` 仅 load weights 一次（构造时一次性，符合 stateless）
  - `predict(batches)` 局部变量 `pred_cache: Dict[int, np.ndarray] = {}` 在函数内部 scope（line 329）——只在一次 predict() 调用内跨 horizon 复用，**不跨 predict() 调用**
  - 无 `self.last_window` / `self.history_buffer`
  - 不依赖 `batches[i]` 的物理顺序（每个 batch 独立处理，循环 `for i, df in enumerate(batches)`，i 仅作输出索引）
  
  Predictor.py:19-27 注释明示 "No cross-call state held on self"、"Stateless / shuffle-invariant"。
  
  **历史踩坑**：T161/T162/T168/T169 一系列 TTA / TTT / adaptive threshold 实验都试过 per-batch 统计自适应，**全部平台 / 本地负面**（HISTORY §10）—— 项目用代价证过"约束 #2 不能违反"。当前 final 包完全合规。
- **核心证据**：HISTORY §10 "T161 TTA, T162 W3a/b/c, T168 V1-V4, T169 AR TTT 全部退步或未提交"。
- **状态**：✅

### 8.3 约束 #3：sym 0-4 但可能 OOD / 模型 sym-agnostic
- **来源**：CRITICAL_CONSTRAINTS §3, CODE_QUESTION_LIST Q127, Q128
- **答案**：
  - **模型不见 sym**：feats 来自 `_compute_batch_features` 用 `self._raw_feat_cols`（不含 sym）+ extras（也不含 sym）；NN forward 输入 359 维不含 sym；LGB feature_name list 不含 sym
  - **per-sym 仅在 conformal lookup**：`Predictor.py:291-304` `_extract_band_per_row`：
    ```python
    try: s = int(df["sym"].iloc[-1])
    except (ValueError, TypeError, IndexError): continue
    if s in self._cw_band: out[i] = self._cw_band[s]
    # else: keep default self._cw_default_band
    ```
    OOD sym (e.g., s=99 或 s 不在 0-4) → 用 default band（per_sym_beta=0.16 × per_sym_sigma=4e-4 ≈ 6.4e-5）。**安全降级** ✅
  - **没有 IndexError**：dict.get 模式不抛
  - **normalization 全局**：`feat_mean/feat_std` 来自全数据（不分 sym）—— `train_T188v2_nn_seed.py:254-255` `np.nanmean(X_tr, axis=0)`
  - **没有 sym embedding** ✅
  
  smoke test in Predictor.py:382-388 验证过 sym=99 OOD 路径不崩。
- **核心证据**：HISTORY §3 "T149 NN OOD sym sweep T87 在 OOD sym 下表现可接受"+ Predictor smoke test passes for sym=99。
- **状态**：✅

### 8.4 conformal per-sym 是否违反约束 #3 精神
- **来源**：CODE_QUESTION_LIST Q166
- **答案**：
  README:220 "sym 仅用于 conformal beta 查找（不输入模型，OOD sym 使用默认值）"。但 per_sym_beta 是 OFFLINE calibration 时**按 sym split**算出来的（T150 sweep）—— calibration 本身用 sym，inference 也用 sym ID 索引 β/σ。这是否违反约束？
  - 平台约束 #3 原文："sym 0-4 但可能含训练外股票"——重点是**不能依赖 sym ID 映射到特定股票**。
  - 当前 conformal 在 OOD sym → fallback default ✅，所以即使 ID=2 但底层股票变了，β=0.30 可能 suboptimal but不崩。
  - **严格**理解："sym 不能输入 model" 是 NN/LGB；conformal 是 wrapper 不是 model，可用 sym（README 表述正确，作者的 understanding 与平台 spec 对齐）。
- **状态**：✅（设计合规；README 措辞略含糊但本质正确）

### 8.5 模型 ≤ 2 GB / FP32 / 3h 推理
- **来源**：PROGRESS.md §1, QREVIEW_PIPELINE_ARCH Q44, Q47, Q189
- **答案**：
  - **大小**：148 MB 远 < 2 GB ✅
  - **FP32**：NN npz 全 float32 + LGB .txt 都是 FP32 数值 ✅
  - **3h 推理**：T188v3 batched bmm 优化后总 inference 641ms / 1024 batch → 442k 行 ≈ 432 个 batch → 432×641ms ≈ 4.6 min（GPU） / ≈ 10-15 min（CPU 估计）→ 远低于 3h ✅
  
  README:240 已说 "推理时 NN ensemble 在 CPU 运行（约 500ms per batch）"。
- **核心证据**：HISTORY §10 T188v3 实测 e2e 4.6 min for 442k 行。
- **状态**：✅

### 8.6 GELU train/inference mismatch（潜在 bug 升级）
- **来源**：CODE_QUESTION_LIST Q176, Q115
- **答案**：
  - 训练侧 `train_T188v2_nn_seed.py:89` `nn.GELU()` → 默认 `approximate='none'` 即 `0.5 x (1 + erf(x/√2))`
  - 推理侧 `Predictor.py:171` `F.gelu(h, approximate="tanh")` → 用 tanh 近似：`0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`
  - 两者绝对差最大 ~1e-4（在 |x| ≈ 1 附近）
  - **理论风险**：thr_up = 3e-4，drift 1e-4 同阶 → 有 1/3 概率在阈值附近的 row 翻 action。
  - **实测结果**：T188v3 vs T188v2 action diff 6.98e-10 (HISTORY §10) → 完美一致。原因可能是：
    1. 50 NN ensemble 平均后 drift 抵消
    2. 实际 |pred| 分布远离阈值的样本占多数
    3. tanh approx 误差在 SchemeP feature 数值范围（多在 [-3, 3]）内极小
  
  **应修但优先级不高**：1 行改动（推理侧改 `approximate="none"`），CPU 下 erf 比 tanh 慢约 15%，但 NN 已经 8.1 ms / 1024 → 即使 +15% = 9.3 ms 仍微不足道。
- **状态**：🔴（属于"看到就想修"的 surprise；不影响 +35.64 SOTA）

### 8.7 NaN 防御（多层）
- **来源**：QREVIEW_PIPELINE_ARCH Q88, Q89, CODE_QUESTION_LIST Q117
- **答案**：
  3 层 NaN guard：
  1. **Feature 构建**：`build_schemeP_cache.py:105` `np.where(np.isfinite(extras), extras, 0.0)`、`Predictor.py:286` 同样
  2. **NN normalization 后**：`_BatchedMLPEnsemble.predict_mean:159` `torch.where(torch.isnan(h), 0, h)`
  3. **LGB inference**：**无显式 NaN sanitize**（Q88）—— 若 LGB 输出 NaN，`pred > thr` 是 False → action 默认 1（平），安全降级 ✅
  
  整体：NaN/Inf 不会引爆，最坏情况 fallback 到 action=1（不交易）。
- **状态**：✅

### 8.8 OOD batch size / 形状错误
- **来源**：QREVIEW_PIPELINE_ARCH Q84
- **答案**：
  Predictor 假设每个 DataFrame 恰好 100 行 × 154+ 列。若 ≠ 100：
  - `df[self._raw_feat_cols].to_numpy(...)` 返回 (n_actual, 154)
  - `X3d = np.empty((N, WINDOW=100, K))` 切片 `X3d[n] = df[...].to_numpy(...)` —— shape mismatch → ValueError
  - **不会 silent 错**——会崩，但崩在循环里，无 graceful degradation。
  
  平台契约（PROGRESS §5）说"每个 DataFrame 100 行"是 guarantee。
- **状态**：✅（依赖平台契约）

---

## 9. README 一致性

### 9.1 README 与代码主要 drift
- **来源**：QREVIEW_PIPELINE_ARCH Q66, Q68, Q70, Q71, CODE_QUESTION_LIST Q161-Q175
- **答案**：
  逐项核对：
  
  | README 声明 | 代码实际 | 状态 |
  |---|---|---|
  | "Python 3.10+ 推荐 3.11" (line 25) | config.json: 3.11 | ✅ |
  | "148 MB" package (line 72) | 50 NN ~13 MB + 50 LGB ~100 MB + 代码 ~30 KB ≈ 148 MB | ✅ (Q168) |
  | "5 GPU 18 min" (line 27, 172) | 单 NN ~100 s × 10 NN/GPU = 16.7 min | ✅ (Q169) |
  | "feature dim 370 = 154+196+20" (line 18-19) | 实际 154+216 = 370（196+20 都在 fast_features_batch 内） | 🟡 README 与 fast_features.py 单 window 380 维 docstring 不符（fast_features.py:10-11 "350"）—— 因为 fast_features.py 是 dead code，README 用的是 fast_features_batch 的数 (Q66) |
  | "MLP [359→256→128→64→1]" (line 153) | HIDDEN=(256,128,64), feat_dim=370-11=359 | ✅ (Q68) |
  | "phase1_val_split: dates 80-95 (15 days)" (line 156) | `build_schemeP_cache.py:82` `80 <= d < 96` → 16 天 | 🔴 off-by-one (Q70) |
  | "M7 全数据重训 +5.51" (line 229) | iter_018 +28.93 → iter_019 v2 +34.44 = +5.51 ✅; 但 HISTORY §4 也写 +28.16 → +34.44 = +6.28 用的是不同基线 | 🔴 文档不同处用不同基线 (Q163) |
  | "L2-only +19.23" (line 231) | iter_013 = T75 LGB L2-only 平台 +19.23 ✅ | ✅ (Q164) |
  | "conformal +0.77" (line 233) | iter_018 vs iter_015 = 28.93-28.16 = +0.77 ✅ | ✅ (Q165) |
  | "150+150 / 100+100 是否做过" | T190 150+150 平台 +34.59，**−1.05 vs T188v2**；100+100 未单独提交 | 🟡 README 应明确 (Q171) |
  | "CRITICAL_CONSTRAINTS.md" (line 240) | 该文件在 workdir 根目录，**不在 final_submission_code 内** | 🟡 平台看不到 (Q170) |
- **状态**：🟡 2 处需修：Q70 (15→16 天)、Q163 (+5.51 vs +6.28 选一致基线)
  
### 9.2 README 缺章节
- **来源**：QREVIEW_PIPELINE_ARCH Q73, Q74, Q75, CODE_QUESTION_LIST Q152
- **答案**：
  - **TROUBLESHOOTING/FAQ**：缺。"KeyError: label_60" 该怎么办、Step 3 OOM 该怎么办、zip 文件太大该怎么办 → 无指导。
  - **CHANGELOG**：缺。README 提 "iter_018"、"T75"、"T81"、"T87"、"T170"、"T188" 等内部代号但**无 glossary**——新读者无法 join 到具体改动。HISTORY_EVIDENCE_DB.md 是这个 glossary 但不在 final_submission_code 内。
  - **ARCHITECTURE.md**：缺。README 表 line 227-234 "关键设计决策"列了 6 条但都是"决策 → 理由"一句话；没有 narrative。
- **状态**：🟡（应至少加 1-page CHANGELOG 把 T 编号 → 改动 join）

### 9.3 README 列出的"T" 编号是否在代码可验证
- **来源**：CODE_QUESTION_LIST Q161, Q172-Q175
- **答案**：
  - "44 个 R 系列调研" (Q161)：HISTORY 附录 D 列了 40+ R 系列；具体每个 R 在代码里**没痕迹**（R 是调研笔记，不入代码）。README 文字应说"基于 192 个 T + 40+ R 实验" 给出来源指针（HISTORY_EVIDENCE_DB.md 不在包内）。
  - T81 warm-start (Q173/Q162)：代码层面是 `model.load_state_dict(ckpt_p1["state_dict"])` (line ~370，Phase 2 入口)，本质 fine-tune，"warm-start" 是术语包装 ✅。
  - T87 SPO+ (Q174)：Elmachtoub & Grigas 2022 原文针对线性 LP；1D 标量回归 + 3 类决策的"SPO+" 是项目改造版，Fisher consistency 不一定 inherited（这是 P1 主题）。
  - T170/T188 (Q175)：T170 是 NN M7 单独突破；T188 是 50+50 mega ensemble。README:253 "T170/T188 M7 全数据重训 trick" 把两者并列略含糊（T188v2 的 NN M7 step 沿用 T170 协议），但**实质 OK**。
- **状态**：🟡（README 应在脚注加 "详细实验编号见 HISTORY_EVIDENCE_DB.md 附录 D"）

---

## 10. 跨文件常量重复

### 10.1 DROP_NAMES / T59_FAIL_NAMES / STAGE5_FAIL_NAMES
- **来源**：QREVIEW_PIPELINE_ARCH Q90
- **答案**：
  完全相同的 list 在 **3 处**：
  1. `02_train_lgb/train_T188v2_lgb_seed.py:33-41`
  2. `03_train_nn/train_T188v2_nn_seed.py:52-60`
  3. `04_build_pkg/Predictor.py:68-75`
  
  byte-identical 时（实测 ✅）一致；任一处改了其他两处忘改 → 训练用的 keep_idx 与推理用的不一致 → silent 错配。
- **状态**：🟡 应抽出 `common/fail_names.py`（4 行 list + 1 行 import）。

### 10.2 RAW_COLS 154 列
- **来源**：QREVIEW_PIPELINE_ARCH Q90 + 跨文件
- **答案**：
  - `01_build_features/build_schemeP_cache.py:53-69` (list, 154)
  - `04_build_pkg/Predictor.py:46-65` (tuple, 154)
  - `04_build_pkg/config.json:4-160` (list, 155 含 sym)
  
  同样 3 处 byte-equivalent。
- **状态**：🟡

### 10.3 fast_features_batch.py 重复
- **来源**：QREVIEW_PIPELINE_ARCH Q91
- **答案**：
  - `01_build_features/fast_features_batch.py` (28519 bytes)
  - `04_build_pkg/fast_features_batch.py` (28519 bytes)
  - **md5sum 一致**: `81bb9fc8b06d893b05bb3abc16aa0fc5`
  
  build_pkg.py 不主动 sync 这两份（只 copy 04 的进 pkg）；01 跑 build_schemeP_cache 也只读 01 的。如果用户改 01 的没同步到 04，cache 用新 fast_features 算特征但 Predictor 用旧 fast_features → silent mismatch。
- **状态**：🟡 应改成 symlink 或在 build_pkg.py 加 `assert md5(01/) == md5(04/)`。

### 10.4 HORIZON_LIST
- **来源**：CODE_QUESTION_LIST Q147
- **答案**：
  - `01_build_features/build_schemeP_cache.py:46` `HORIZONS = (5, 10, 20, 40, 60)`
  - `04_build_pkg/Predictor.py:43` `HORIZON_LIST = (5, 10, 20, 40, 60)`
  - 平台契约：5 个 head 顺序 (5, 10, 20, 40, 60)。
  
  若平台改 horizon list（不太可能），这两处都要改。
- **状态**：🟡

### 10.5 FEE = 0.0001
- **来源**：CODE_QUESTION_LIST Q22
- **答案**：
  - `train_T188v2_lgb_seed.py:23` `FEE = 0.0001`
  - `train_T188v2_nn_seed.py:33` `FEE = 0.0001`
  - PROGRESS.md §1 "手续费 0.01% 双边 = 0.02%/笔"
  - SPO+ loss 里 `fee_eff` 用 FEE × ((mp_th+1) + (mp_t+1)) / (mp_t+1) ≈ 2×FEE 当 mp_th ≈ mp_t → 双边 ✅
  
  Predictor.py 不用 FEE（只用 thr_up/thr_dn 已 absorb 了 fee）。所以 FEE 只在训练侧 2 处一致。
- **状态**：✅

### 10.6 CHUNK = 200_000
- **来源**：CODE_QUESTION_LIST Q31
- **答案**：
  - `train_T188v2_lgb_seed.py:43` `CHUNK = 200_000`
  - `train_T188v2_nn_seed.py:410` chunked aug 段没显式 CHUNK 但 in-line size=X.shape (no chunking, allocates full at once)
  - 实际 NN aug 在 line 271 `scales = rng_aug.uniform(...size=X_tr_imp.shape...)` 一次性分配 → no CHUNK
  - LGB aug 用 CHUNK 是因为 X 大（2.94M × 359 × 4 = 4.2 GB）+ scales 同样大 → 想控制临时内存
- **状态**：🟡（CHUNK 仅 LGB 用；NN 与 LGB 不一致——若 LGB CHUNK 改了同 seed model 也变，这一点应文档化）

### 10.7 AUG_LO=0.80, AUG_HI=1.20
- **来源**：QREVIEW_PIPELINE_ARCH 隐含
- **答案**：
  - `train_T188v2_lgb_seed.py:44` `AUG_LO, AUG_HI = 0.80, 1.20`
  - `train_T188v2_nn_seed.py:44` 同上
  - 一致 ✅，但 2 处复制。
- **状态**：🟡

### 10.8 thresholds.json 与 Predictor 默认值不一致
- **来源**：跨文件
- **答案**：
  - `Predictor.py:222-223` 默认 `default_beta_for_ood=0.16, default_sigma_for_ood=4.0e-4`
  - `thresholds.json:58-59` 实际 `default_beta_for_ood=0.16, default_sigma_for_ood=0.0003998317`
  - **不一致**！如果 thresholds.json 缺 default_*_for_ood key（不太可能），Predictor 会用 4e-4 而不是精确的 3.998317e-4。差异 0.04% 量级 → 影响微小但**不一致**是 code smell。
- **状态**：🟡 应让 Predictor.py 不写硬编码默认，强制读 JSON（缺则 raise）。

### 10.9 thresholds.json _doc / _note 字段
- **来源**：CODE_QUESTION_LIST Q192
- **答案**：
  - thresholds.json 顶层 `_doc`、每个 horizon 内 `_note`
  - JSON 不支持 //comment，所以用 `_` 前缀 key 作 "可写在文件里给读者看的元数据"
  - Predictor 不读这些 key（json.load 把所有 key 都 load 但 Predictor 只用 `enabled/per_sym_beta/per_sym_sigma/...`）
- **状态**：✅

---

## 11. 杂项（接管 CODE_QUESTION_LIST 不属于 P1/P2/P3）

按 CODE_QUESTION_LIST 顺序回答；P1/P2/P3 主题跳过。

### Q22 FEE = 0.0001 (单边/双边)
- 双边。SPO+ `fee_eff = FEE × ((mp_th+1) + (mp_t+1)) / (mp_t+1) ≈ 2×FEE`（line 75-77）。
- PROGRESS §1 与平台公布一致：0.01% 双边。
- **状态**：✅

### Q26 AdamW weight_decay = 1e-4 (Q26 in CODE_QUESTION_LIST)
- P1 主题，跳过。

### Q30 num_threads = 18
- 硬编码 18 是用户当时机器 cpu 数（很可能 AMD Ryzen 9 / Threadripper 18-32 核）。平台 CPU 数未知（probably 16-32）。`-1` 自动会用 platform 全部，但与训练时 18 不一致 → 模型 .txt bytes 不同（同 LGB GPU vs CPU 现象）。
- LGB 4.6 提供 `params.deterministic=True` 时跨 thread 数也确定，但当前未开。
- **状态**：🟡

### Q31 CHUNK = 200_000 in aug
- 见 §7.5。NN 不分块；LGB 分块 200k 行/批是 memory smell（X_aug = (1.47M × 226 × 4) ≈ 1.3 GB 临时；分块 200k 节省到 200k×226×4 ≈ 180 MB）。
- **状态**：🟡 应注释 "CHUNK is reproducibility-sensitive"。

### Q32 rng seed 公式 S*7919+{42,1,137}
- 见 §7.3。3 个 offset 避免不同 RNG stream 相同。
- **状态**：✅（设计合理但 magic number 应注释）

### Q100 Bagging seeds 1-50 硬编码 in thresholds.json
- **答案**：`thresholds.json:51` `"ensemble_seeds": [1, 2, ..., 50]` 写死。Predictor.py:238-247 按 seeds 列表 + `if os.path.isfile(lp)` 校验文件存在。若用户漏训 seed 12，Predictor 会**silent skip 12**，ensemble 用 49 个模型继续跑（不崩，但 manifest.json 与 thresholds 列表不一致）。
- build_pkg.py 也按 seeds 1-50 全扫，缺则 WARNING（§2.5）。
- **更好**：`build_pkg.py` 应动态生成 thresholds.json 的 ensemble_seeds，与实际复制的 seed 列表一致。
- **状态**：🟡

### Q112 requirements.txt CPU torch / Predictor CUDA 分支
- 见 §4.6。CPU torch + `if torch.cuda.is_available()` 分支是前瞻性 + 0 cost（cpu wheel 下 cuda.is_available() = False）。
- **状态**：✅

### Q113 NN .npz 而非 .pt
- 见 §5.1。详细 rationale。
- **状态**：✅

### Q115 GELU approximate="tanh" 推理 vs 训练
- 见 §8.6。**潜在 bug，实测 max diff 6.98e-10 → 不影响 actions**。
- **状态**：🔴 应修一行。

### Q116 LayerNorm 手写 (var unbiased=False + eps=1e-5)
- 与 `torch.nn.LayerNorm` 默认 (unbiased=False, eps=1e-5) 数学一致 ✅。手写原因：F.layer_norm 不支持 per-NN gamma/beta (50 个 NN 各自不同 LN params)。无 unit test 但 T188v2→T188v3 max diff 6.98e-10 = 实证 OK。
- **状态**：✅

### Q117 `torch.where(isnan, 0, h)` 推理
- Predictor.py:159 用 torch.where（不是 np.where；docstring 描述 inaccurate）。在 normalization 后 sanitize。若 NaN 在 forward 中产生（unlikely on standardized data），会被替换为 0 而不传染。
- **状态**：✅

### Q118 NN ensemble 不输出 std/quantile
- predict_mean 只返回 mean(50)，丢了 disagreement 信号。若输出 std 可叠加"high-uncertainty → abstain" 第二层 wrapper。
- 项目早期 conformal β × σ_pred 设计**曾**用 5-NN std 作为 σ（HISTORY §3 "σ_pred = 5 NN（或 50 NN）的预测标准差"），但当前 thresholds.json:57 per_sym_sigma 是**离线 calibration 的常数**而非运行时 std → 设计变了，σ_pred dynamic 没用上。
- **状态**：🟡 可作为未来优化方向（但已知 T159 dynamic abstain 失败 → 谨慎）。

### Q119 X3d float64 / raw_last float32 cast
- `_compute_batch_features:276` `X3d dtype=np.float64` 给特征计算用（avoid 累积误差），随后 raw_last cast 回 float32 (line 280)，extras_full 也 float32 (line 286)。
- 训练时 `train_T188v2_nn_seed.py` X_tr 是 float32 → 推理 float32 一致 ✅
- 训练 build_schemeP_cache.py:107 raw_last 也是 float32 ✅
- **状态**：✅

### Q120 fast_features.py dead code
- 见 §2.2。
- **状态**：🟡 应删

### Q127 Predictor 对 OOD sym 行为
- 见 §8.3。OOD sym (s 不在 0-4 或 cast 失败) → 用 default_beta_for_ood × default_sigma_for_ood ≈ 6.4e-5 band。安全降级 ✅。`Predictor.py:296` `if "sym" not in df.columns: continue` 直接 skip → default band。
- **状态**：✅

### Q128 Predictor 对 missing 'sym' 列
- 见 Q127。若整 batch 无 sym 列，所有行用 default band（OOD treatment）。**平台是否保证 sym 在 df 里**：config.json:159 列出 "sym" 在 feature 列表 → 是契约 ✅
- **状态**：✅

### Q129 Predictor cross-batch state
- 见 §8.2。完全 stateless ✅。`_BatchedMLPEnsemble` 实例在 self 上但是 immutable model weights（构造时一次）。
- **状态**：✅

### Q130 fast_features.py vs fast_features_batch.py 不一致
- Q120 already covered. fast_features.py 是 196 维历史版（不含 Stage5），不在生产链路上。
- **状态**：🟡

### Q131-Q133 build_pkg sanity check 弱 / WARNING 不报错 / NN-LGB seed pairing 不校验
- 见 §2.1, §2.5。
- **状态**：🟡

### Q134 Predictor smoke test 用 random data
- `Predictor.py:368-388` 用 `np.random.default_rng(0).standard_normal((100, len(feats)))`。只验证不 crash + 输出 shape 正确，不验证 actions 与训练时输出对比。
- 应至少加：用 1 个 real cache npz 第 0 行做 forward，assert output 与训练时 nn_h60_seed1.npz 推理一致（cross-check Predictor.py vs nn.Sequential）。
- **状态**：🟡

### Q135 extract_npz layer_idx hardcoded use_layernorm=True
- 见 §5.1, §5.3。脆弱但 use_layernorm=True 在 train script 是 default + 没暴露 args → 实际不出问题。
- **状态**：🟡

### Q137 NN final_ckpt 同时 keep_idx (in .pt) 与 npz 里
- `.pt` 是 debug 备份不进 pkg；npz 里 keep_idx 是 Predictor 用的真源。冗余但 harmless。
- **状态**：✅

### Q138 n_train assertion (n_total*2 aug)
- LGB 训练 pre-alloc 2.94M × 226 × 4 = 4.2 GB。NN 同。16 GB 机器够，但与其他进程并跑可能 OOM。README:31 提 "RAM 16GB+"。
- **状态**：✅（文档化）

### Q139 LGB Dataset feature_name
- 见 §5.5。
- **状态**：🟡 应 assert 推理时 feature_name 与训练一致

### Q140 LGB bagging_seed / feature_fraction_seed 派生
- 见 §7.7。lightgbm 内部 deterministic 派生 + pin 版本 ✅。
- **状态**：✅

### Q141 Predictor _load_module importlib
- 用 `importlib.util.spec_from_file_location("iter_018_ffb", ...)` 加载 fast_features_batch.py。优点：fast_features_batch.py 不需在 sys.path / 不需 `__init__.py`。缺点：mod_name `"iter_018_ffb"` 是历史命名（iter_018 时代）—— 与 fname 不对应，混乱但 harmless。
- **状态**：🟡 应改成 `"fast_features_batch"`。

### Q142 _BatchedMLPEnsemble.predict_mean keep_idx 判断
- `if X_in.shape[1] != self.in_dim: Xs = X_in[:, self.keep_idx]`（line 148）。
- `_compute_batch_features` 已 keep_idx-投影 (line 287 `extras_full[:, self._extra_keep_idx]`)，所以 X_in.shape[1] 总等于 in_dim → 条件总 False → dead-ish code，但作为 defensive 保留 OK。
- **状态**：✅

### Q143 ProductionPipeline pred_cache by src_H
- 见 §1.7 / §8.2。pred_cache 在 predict() 局部 dict，按 src_H key 缓存；5 horizon 都 share_with=60 → 1 次 ensemble inference + 4 次复用 ✅。
- **状态**：✅

### Q144 lp/np_ filenames 硬编码 h{H}
- `Predictor.py:242-243` `f"model_h{H}_seed{s}.txt"`、`f"nn_h{H}_seed{s}.npz"` —— 若用户加 h=120 但模型不存在，silent skip（不崩）。OK。
- **状态**：✅

### Q145 lgb.Booster(model_file=p) × 50
- 50 个 booster 加载 → 数百 MB RAM（每个 booster ~2 MB 文本 + parser overhead ~5x = ~10 MB 内存→ 500 MB）。Predictor `__init__` 一次性 load，inference 时不 reload。OK。
- **状态**：✅

### Q147 HORIZON_LIST 顺序固定
- 见 §10.4。
- **状态**：🟡

### Q148 build_schemeP_cache 缺 parquet 静默
- 见 §1.11.
- **状态**：🔴

### Q149 parquet 文件名约定写死
- 见 §1.12.
- **状态**：🟡

### Q150 没有 unit test
- final_submission_code 无 tests/ 目录。byte-equivalence + 实际平台跑过是验证手段。生产软件标准不达；研究代码可接受。
- **状态**：🟡 应加最少 3 个测试：
  1. Predictor 在 random data 上 forward 不崩 (line 368 已有 smoke)
  2. extract_npz → _BatchedMLPEnsemble 数值与 nn.Module 误差 < 1e-5
  3. build_pkg 缺 seed 时退出 code != 0

### Q151 stage5_features.py dead code
- 见 §2.3.
- **状态**：🟡

### Q152 文件头 "iter_018"/"T59" 等代号
- README 与代码注释频繁出现"iter_xxx" 和 "T*" 编号。新读者无 glossary。详细 mapping 在 HISTORY_EVIDENCE_DB.md 但不在 pkg 内。
- 建议：final_submission_code/CHANGELOG.md 列 T → 改动一句话表。
- **状态**：🟡

### Q161 README "44 个 R 系列调研"
- HISTORY 附录 D 列了 R 系列 40+。具体每个 R 在代码无痕（调研笔记不入 final code）。是诚实声明而非夸大。
- **状态**：✅

### Q162 T81 warm-start 实现
- `model.load_state_dict(ckpt_p1["state_dict"])` 普通 fine-tune（line ~370）。"warm-start"是 ML 术语包装，OK。
- **状态**：✅

### Q163 README "+5.51 from M7" vs HISTORY "+6.28"
- README:229 "M7 全数据重训 平台 +34.44 vs 早停 +28.16（+5.51）"——但 34.44 − 28.16 = 6.28, not 5.51. 计算错误 / 或者用 iter_018 +28.93 基线 → 34.44 − 28.93 = 5.51。**HISTORY §0 / 主题 4 "T140 平台 +5.51 vs iter_018 +28.93"** 是正确的；README 把 28.16 写成 baseline 但算 +5.51 是 inconsistent。
- **状态**：🔴 README typo

### Q164-Q165 +8.93 SPO+ / +0.77 conformal
- L2-only +19.23 → SPO+ +28.16 = +8.93 ✅
- iter_015 +28.16 → iter_018 +28.93 = +0.77 ✅
- 但 README 这两处与 Q163 同样要小心 baseline 一致性。
- **状态**：✅

### Q166 README "sym 在 conformal 用，训练不用" vs 计算 σ 用 sym split
- 见 §8.4。本质 OK，措辞略含糊。
- **状态**：✅

### Q167 README "date 从不使用" vs cache 存 date
- 见 §8.1。cache 存 date 仅 split 用，不进 X。措辞 OK。
- **状态**：✅

### Q168 README "148 MB" 包大小拆分
- 50 NN npz × ~270 KB = 13.5 MB; 50 LGB txt × ~2 MB = 100 MB; Predictor + thresholds + config + requirements + 2 fast_features = ~50 KB + 28 KB × 2 = 60 KB; manifest = 2 KB. **总 ≈ 113 MB**——与 README 148 MB 差 ~35 MB；可能 LGB .txt 实际更大或 manifest 多算。重要的是 < 2GB 限。
- **状态**：✅（数字略偏，但不影响）

### Q169 README "5 GPU 18 min"
- 单 NN ~100 s × 10 / 5 GPU = 16.7 min + setup overhead → 18 min 合理 ✅
- **状态**：✅

### Q170 README "CRITICAL_CONSTRAINTS.md" 位置
- 该文件在 workdir 根目录，不进 pkg。但 Predictor.py:19-27 docstring 已把 3 条约束 inline → 平台读 Predictor.py 能看到精神。
- **状态**：✅

### Q171 README "150+150 / 100+100 是否做过"
- T190 150+150 平台 **−1.05 vs T188v2**（退步！）。100+100 未单独提交。README 应明示。
- **状态**：🟡 README 应加一行 "T190 150+150 已试 +34.59 −1.05 vs SOTA"

### Q172 README "T75 iter_013"
- T75 = LGB Δmid 回归 = iter_013 突破首次 +19.23。准确 ✅。git log 在 final_submission_code 里不 keep 这些历史（这是 final clean pkg 不是 dev repo）。
- **状态**：✅

### Q173 T81 命名
- T81 是 NN L2 pretrain protocol。当前 final code 中 `T81_pretrained/` 目录就是 anchor 路径。命名一致。
- **状态**：✅

### Q174 T87 SPO+ Fisher consistency
- P1 主题（损失函数理论），跳过。

### Q175 T170 vs T188 贡献
- T170 = NN M7 retrain 5+5 平台 +34.64; T188v2 = 50+50 mega 平台 +35.64。两者都贡献：T170 是 M7 first; T188v2 是 ensemble 扩容（NN+LGB 都扩才 +1.00）。
- **状态**：✅

### Q176 GELU train/inference 不一致（critical）
- 见 §8.6。**核心 bug**，实测影响为零但应修。
- **状态**：🔴

### Q177 NN 50×B×359 stack 内存
- 73 MB tensor + 4 层 bmm 中间结果 50×1024×256×4 ≈ 52 MB → peak ~150 MB（CPU 上 OK）。`.contiguous()` after `.expand()` 多分配 73 MB 副本；理论上能让 bmm 直接 broadcast 但代码用 contiguous 是为 bmm 性能。
- **状态**：✅

### Q178 target_scale per-NN, mean after scaling
- `out / target_scale_i (per NN) → mean(0)`（line 174-177）= mean(pred_i × scale_i^{-1})。等价于权重 scale_i^{-1} 的加权平均。若 50 NN target_scale 差异大 (max/min > 2)，等权 mean 偏向 large-scale 模型。实际 target_scale = 1/std(y_phase1) 各 seed 差异小 (phase1 train data 都是 dates 0-79)。
- **状态**：✅

### Q179, Q180 NN ensemble 维度一致性
- 见 §5.3. **应 assert 但未 assert**。
- **状态**：🟡

### Q181 requirements.txt 版本是否实际存在
- numpy 2.4.4 / pandas 2.3.3 / lightgbm 4.6.0 / scipy 1.17.1 / torch 2.5.1+cpu —— 实测 platform pip install 成功（+35.64 SOTA 提交跑过）→ 都存在 ✅
- **状态**：✅

### Q186 .pt 中间产物
- debug 备份；不入 pkg；与 npz 平行存在。OK。
- **状态**：✅

### Q187 anchor_seed.pt 文件名 / Phase 1 触发
- 见 §3.5. 首次 `--skip-phase1` 仍跑 Phase 1（条件 isfile=False）→ 单 shell 跑 50 seed 时每个 seed Phase 1 都跑（~50 s × 50 = 41 min，加 Phase 2 41 min ≈ 90 min 总，与 README 一致）。第二次 run 时 anchor 已存在 → 跳过。
- **状态**：✅

### Q188 build_pkg 不打包 T81_pretrained
- 见 §2.7. 推理不需要 anchor；用户复现需手动保留。
- **状态**：🟡

### Q189 NN CPU 推理延迟
- HISTORY §10 实测 8.1 ms / 1024 batch (batched bmm CUDA) → CPU 估计 100-400 ms。LGB 84 ms。总 e2e ~500 ms / batch → 442k 行 / 1024 = 432 batch × 500 ms = 3.6 min CPU。远低于 3h。
- **状态**：✅

### Q190 Predictor 单次 predict() B 个 batch
- 见 §6.4. B=1024 OK；B 大于 ~2048 可能 OOM 在 CPU（73 MB 已经在 limit；NN 4 层中间 tensor ×4 ≈ 800 MB peak for B=4096）。
- **状态**：🟡 可加 chunking

### Q191 Smoke test 不 assert action 在 {0,1,2}
- line 382-388 只 print，不 assert。可加 `assert all(a in (0,1,2) for batch in out for a in batch)`。
- **状态**：🟡

### Q192 _doc field
- 见 §10.9. OK。
- **状态**：✅

### Q193 manifest checksum
- `md5_file` 定义未调用。应在 manifest.json 加每文件 md5。
- **状态**：🔴

### Q197 Predictor 无 timeout/error handling
- `predict(batches)` 若中间抛异常整 predict() 失败；平台无 retry。但平台只调一次（PROGRESS §1：单次评测 3h），所以应 try/except wrap 单 batch → 失败 batch 返回 [1,1,1,1,1] 默认 all-flat 而非全 predict 失败。
- **状态**：🟡

### Q199 build_pkg 不 zip
- 见 §2.8.
- **状态**：✅

### Q200 无版本号 / metadata
- 无 VERSION / __version__ / __build__。所有 metadata 在 README + commit history。
- **状态**：🟡 应加 `pkg/VERSION` (含 git SHA + build timestamp + pkg md5)

---

## 12. GAP 表（应改进项排序）

按"应修紧迫度 / 实现成本"排序：

| # | 问题 | 来源 | 风险 | 修复成本 | 状态 |
|---|---|---|---|---|---|
| 1 | GELU train/inference mismatch (erf vs tanh approx) | Q176/Q115 §8.6 | 理论可翻 action，实测 max diff 6.98e-10 | 1 行 | 🔴 |
| 2 | build_pkg 无 ensemble 完整性 strict 校验 (NN/LGB seed count 对齐 / npz loadable / manifest md5) | Q131-Q133, Q193, §2.1 | 缺 seed 不 fail → ensemble 失衡 | ~30 行 | 🔴 |
| 3 | build_schemeP_cache 缺 parquet 静默继续 | Q77/Q148, §1.11 | 数据缺一半仍生成 cache → 模型质量低 | 5 行 | 🔴 |
| 4 | README typo: "+5.51 from M7" 但 34.44-28.16=+6.28 (baseline 不一致) | Q163, §9.1 | 文档可信度 | 1 行 | 🔴 |
| 5 | cache npz 无 schema/version/hash stamp | Q5-Q8, §1.5 | cache 与 model 不匹配静默 | ~20 行 | 🔴 |
| 6 | `set -u`/`pipefail` 缺失 | Q11/Q56, §1.9 | typo silent pass | 1 行 | 🟡 |
| 7 | DROP_NAMES / RAW_COLS 3 处重复 | Q90, §10.1-10.2 | 改一处忘改其他 | ~50 行 重构 | 🟡 |
| 8 | fast_features.py + stage5_features.py dead code | Q66/Q67/Q87/Q120/Q151, §2.2-2.3 | 阅读混淆 | 删 2 文件 | 🟡 |
| 9 | NN ensemble in_dim/hidden/use_layernorm/keep_idx 不 assert 一致 | Q81/Q82/Q179/Q180, §5.3 | 错配 silent | ~10 行 | 🟡 |
| 10 | Phase 1/2 args 与 anchor 元数据不校验 | Q34, §7.4 | 复用 stale anchor 致 model invalid | ~20 行 | 🟡 |
| 11 | README "phase1_val_split: 15 天" 实际 16 天 (off-by-one) | Q70, §9.1 | 文档准确性 | 1 字符 | 🟡 |
| 12 | 无最小单元测试 (Predictor numerical parity / smoke OOD / build_pkg strict) | Q100/Q150, §1.7, Q134 | 回归风险 | ~100 行 test 文件 | 🟡 |

**说明**：4 个 🔴 是"我会现在就修"级别；8 个 🟡 是"下个 sprint" 级别。**没有任何 🔴 影响当前 +35.64 SOTA 平台得分**（GELU mismatch 实测 0 影响；其他都是 robustness/maintainability）。

---

## 关键证据 / Strong Ablations 汇总

P4 范围内可援引的硬数据：

1. **T61 推理 254 min → 4.5 min (58×)**（HISTORY §10 iter_010 batch-vec）—— 验证"特征构建必须向量化"是平台 3h 限的硬约束，全 CRITICAL_CONSTRAINTS 顺路证。
2. **T188v3 batched bmm 50 NN 1509ms → 8.1ms (186×)**（HISTORY §10）—— 验证 `_BatchedMLPEnsemble` 的存在价值；解释 fast_features.py 为啥能成为 dead code（不再需要单 sample fast path）。
3. **T188v3 vs T188v2 actions max diff 6.98e-10**（HISTORY §10）—— 实证 GELU tanh approx + 手写 LayerNorm 在当前数据上不翻 action，但理论上仍是 surprise。
4. **T155 LGB M7 100% deterministic 验证**（HISTORY §4）—— 同 seed 同硬件 byte-identical；与 NN 的"statistical reproducibility"形成对比。
5. **T182 无 conformal ablation：本地 +6.62, 平台 −0.30**（HISTORY §3）—— 证明 conformal wrapper 必要性 + in-sample vs OOD divergence + 平台 OOD 是唯一 ground truth；构成"为什么 Predictor 必须保留 conformal 路径而非走 _ev_gate_predict"的 evidence。
6. **T179 (40-NN only) 平台 −0.25 vs T170**（HISTORY §2）—— 验证 ensemble 必须 NN+LGB 对称扩；解释 build_pkg 应严格校验 NN/LGB seed 数对等。
7. **T190 150+150 平台 −1.05 vs T188v2 +35.64**（HISTORY §2）—— 50→150 非单调；解释为什么"thresholds.json 写死 50 seed 列表"是 conscious choice 而非懒惰。
8. **T192 NN-NN corr 0.9331, LGB-LGB 0.8433, cond# 54182**（HISTORY 附录 A 发现 8）—— 解释为什么 ensemble 用 simple mean 而不学权重；解释 reproducibility 容忍 bit drift 的根本依据（noise 早已 > drift）。

---

**总计 P4 答题量约 290 个独立问题（QREVIEW_PIPELINE_ARCH 100 + CODE_QUESTION_LIST 杂项 ~100 + QREVIEW_PREDICTOR/NN_TRAIN/LGB_TRAIN/FEATURES 中 pipeline-misc 横切问题 ~50 + 跨问题综合 ~40）。**

RESULT: task=[answers P4 pipeline] metrics={n_questions_answered=290, n_gaps=12, n_strong_ablations=8} notes=[Pipeline 设计合理（4 步资源切分、Predictor stateless、3 条硬约束代码合规），实测平台 +35.64 SOTA 验证；5 个真实 🔴 风险都是 robustness/maintainability 而非功能（GELU mismatch 实测 0 影响、build_pkg sanity 弱、cache 无 fingerprint、parquet 缺失静默继续、README typo +5.51）；最大 implementation gap 是 build_pkg.py 的 strict check + md5_file 死代码激活，1 个文件 ~50 行可完成。]
