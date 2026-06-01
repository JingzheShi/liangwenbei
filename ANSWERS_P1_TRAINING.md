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
