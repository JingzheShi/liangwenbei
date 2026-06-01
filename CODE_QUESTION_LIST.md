# T188v2 50+50 提交代码 —— "为什么这么做？" 问题清单

> 阅读对象：`final_submission_code/` 下全部源文件（README + run_pipeline.sh + 01_build_features + 02_train_lgb + 03_train_nn + 04_build_pkg）。
>
> 立场：挑剔审稿人。对任何"魔法数字 / 看似随意的实现细节 / 反直觉的选择 / 与文档不一致的地方"都要质疑。
>
> 本文件**只列问题，不给答案**。答案由 Worker C 负责回答。
>
> 章节：1) 超参/魔法常量 · 2) 架构选择 · 3) 特征工程 · 4) 训练协议 · 5) Ensemble 设计 · 6) 决策层 · 7) 推理优化 · 8) 5 horizon share_with=60 · 9) 代码实现细节 · 10) README 声明核对

---

## 1. 超参 & 魔法常量

### Q1: thr_up 与 thr_dn 为什么不对称？
- **位置**：`04_build_pkg/thresholds.json:46-47`
- **代码摘要**：`"thr_up": 0.0003, "thr_dn": 0.000216`（h=60）
- **问题**：为什么做多阈值是做空阈值的 1.39 倍？这种非对称是基于数据集偏置（label 类分布不平衡？mid 价漂移？）、做空成本估计、还是手动 grid search 出来的？有没有做过 thr_up==thr_dn 的对照？为什么不是 0.0002 / 0.00015 这种更简单的组合？
- **变体/替代方案**：(a) 对称阈值；(b) per-sym 阈值；(c) per-date 阈值；(d) 数据驱动地从 OOF 预测分布的 quantile 取阈值；(e) ROC-curve / EV-curve 优化解出。

### Q2: 阈值 0.0003 / 0.000216 这两个具体数字是怎么调出来的？
- **位置**：同上
- **问题**：0.0003 显得"整"；0.000216 显得"恰好 0.0003 × 0.72"。是不是 `thr_dn = thr_up × 0.72` 这种比例关系？还是分别独立 grid search？search 空间是什么？每个候选值的平台得分是多少？

### Q3: w_nn = 1.0, w_lgb = 1.5 ——为什么 LGB 权重更高？
- **位置**：`04_build_pkg/thresholds.json:48-49`、`Predictor.py:10`
- **问题**：在大量文献里 NN 通常是"主力 + LGB 辅助"或 1:1，本提交反过来用 1.5:1 偏向 LGB。是否做过 sweep？候选值是 {1:1, 1:1.5, 1:2, 1.5:1}？每个值的 OOF / 平台得分？为什么不学一个最优权重（如 Ridge 在验证集上拟合 NN_pred 与 LGB_pred 的线性组合）？

### Q4: WINDOW = 100 为什么固定 100？
- **位置**：`01_build_features/build_schemeP_cache.py:45`、`04_build_pkg/Predictor.py:42`
- **问题**：为什么是 100 而不是 50 / 200 / 300？平台是否硬性要求 100-tick 窗口？做过窗口长度消融吗？更短的窗口（如 50）会怎样？更长（如 200）会怎样？

### Q5: per-sym beta 为什么不一样？
- **位置**：`04_build_pkg/thresholds.json:56`
- **代码摘要**：`per_sym_beta = {0:0.1, 1:0.4, 2:0.3, 3:0.0, 4:0.0}`
- **问题**：sym3/4 完全不 abstain（beta=0），sym1 最激进（beta=0.4）。这些数是怎么标定的——网格搜索 each sym 独立？还是某种 conformal 校准统计量？为什么 sym3、sym4 不需要 abstain 而 sym1 需要 0.4？

### Q6: per-sym sigma 是什么？
- **位置**：`04_build_pkg/thresholds.json:57`
- **代码摘要**：`per_sym_sigma = {0:0.00024, 1:0.00047, ...}`
- **问题**：这些 sigma 数值是 holdout 上 |pred - y| 的 std？是 pred 本身的 std？是某种 conformal score 的 quantile？训练时怎么算的？什么数据切片？

### Q7: default_beta_for_ood=0.16, default_sigma_for_ood=0.0004 怎么定的？
- **位置**：`04_build_pkg/thresholds.json:58-59`、`Predictor.py:222-223`
- **问题**：0.16 看起来像 mean(per_sym_beta)≈0.16；0.0004 像 mean(per_sym_sigma)≈0.00042。是平均值吗？为什么不是中位数？为什么不是 max？OOD sym 的处理策略是基于什么假设——它一定更难预测？

### Q8: 50 + 50 ensemble 大小为什么选 50？
- **位置**：`02_train_lgb/run_all_lgb_seeds.sh:20`、`03_train_nn/run_all_nn_seeds.sh:20`
- **问题**：为什么不是 5+5 / 20+20 / 100+100？README 提到 50+50 比 5+5 提升 +1.00 PnL。是否做过 25 / 75 / 100 的对照？边际收益从哪开始递减？100 个模型的成本（训练 + 推理时间 + 包大小 ~148MB）是否值得？

### Q9: LGB num_boost_round = 330 为什么这个数？
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:72`
- **问题**：330 不是常见整数（如 300/500/1000）。是某次 walk-forward 早停的中位数？是某个 OOF 实验最优？没有 early stopping（M7 全数据无 val），怎么定的？为什么 5 组不同 HP 都用同一个 330（按理 num_leaves=63 应该比 num_leaves=255 训更多轮）？

### Q10: NN Phase 2 固定 11 epoch 为什么？
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:46`
- **问题**：11 epoch 也不是整数。为什么不是 10 / 15 / 20？Phase 2 全数据没有 val，怎么定的 11？做过 sweep 吗？每个 epoch 数的平台得分？

### Q11: Phase 2 lr = 3e-5、lambda_spo = 30 怎么定的？
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:47-48`
- **问题**：3e-5 是 Phase 1 最大 lr（1e-3）的 1/33；lambda_spo=30 是 L2 项权重的 30 倍。这两个数有 sweep 数据吗？lambda_spo ∈ {0, 1, 10, 30, 100} 都做过吗？为什么 30 而不是 50？

### Q12: LGB HP grid 的每个具体值怎么选的？
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:25-31`
- **代码摘要**：5 组 HP，feature_fraction ∈ [0.4, 0.5, 0.6, 0.7, 0.8]，bagging_fraction ∈ [0.5, 0.6, 0.7, 0.8, 0.85]，num_leaves ∈ {63, 127, 127, 127, 255}，lambda_l2 ∈ [0.5, 1.0, 1.0, 2.0, 3.0]
- **问题**：5 组配置看起来"覆盖"了空间但不是网格、不是 sobol、不是 random search。是手工挑的？依据是 OOF 多样性？组 1（0.8/0.8/127/1.0）显得像基线，其他 4 组是基线的扰动？是否做过完整网格搜索？为什么 num_leaves 三组都是 127？

### Q13: NN HP 组合：cycling 设计的合理性？
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:37-39, 186-188`
- **代码摘要**：lr cycling % 3, dropout cycling % 4, batch cycling % 3。
- **问题**：README 写"3×4×3=36 combos"，但 lr 和 batch 都用 (S-1)%3，**这两个 index 完全同步**——seed 1→(0,0)、seed 4→(0,*)、seed 7→(0,*) 等。lcm(3,4,3)=12，所以 50 seeds 实际只覆盖 12 个独立组合（4.17 次重复），不是 36！这是**有意设计**还是 bug？如果有意，为什么不用 (S-1)%2 给 batch 来解耦？

### Q14: LR list 为何 [1e-4, 3e-4, 1e-3]？
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:37`
- **问题**：3e-4 是 "Karpathy LR"。1e-4 与 1e-3 是 ±1 个数量级。为什么不加 5e-4 / 5e-3？这 3 个值哪个最优？

### Q15: Dropout list [0.05, 0.10, 0.15, 0.20]
- **位置**：同上 line 38
- **问题**：常见 dropout 是 0.1 / 0.2 / 0.5。0.05 很轻、0.20 中等。为什么不含 0.0（无 dropout）和 0.3 / 0.5？这些值的 OOF 表现？

### Q16: Batch list [2048, 4096, 8192]
- **位置**：同上 line 39
- **问题**：标准 LOB 训练 batch 一般 256-1024。这里偏大。为什么不含 1024 / 16384？大 batch 会破坏 SGD noise 多样性，是有意为之吗？

### Q17: Patience = 10、Phase 1 max 50 epochs
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:42-43`
- **问题**：patience=10 在 LOB tick 数据上够长吗？50 epochs 上限意味着如果 50 epoch 都改善就直接停（没收敛？）。曾否触发 50 epochs 上限的 seed？

### Q18: AUG_LO=0.80, AUG_HI=1.20 缩放幅度
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:44`、`03_train_nn/train_T188v2_nn_seed.py:44`
- **问题**：±20% 这种量级是怎么选的？对 z-score 特征（均值为 0，正负方向无对称含义）和 quantile rank 特征（范围 [0,1]）做这种乘法没物理意义。是否做过 0.9-1.1 / 0.5-1.5 / 0.95-1.05 的对照？

### Q19: sqrt(H/60) 阈值缩放的理论依据？
- **位置**：`04_build_pkg/thresholds.json:7-42`
- **问题**：对于布朗运动，σ(Δp over h ticks) ∝ √h，所以阈值缩放 √h 看似自然。但 mid 价不一定是 BM；可能有自相关、跳跃。为什么是 √(H/60) 而不是 H/60 线性？或基于经验 σ(pred for h) 的 quantile？做过对照吗？

### Q20: 11 个 DROP_NAMES 具体哪些？
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:33-41`、同 NN 文件 52-60、`04_build_pkg/Predictor.py:68-75`
- **代码摘要**：T59_FAIL = 10 个（dualz_ask_diff{1,5}, dualz_bid_diff5, qrank_W100_spread/cumspread, kyle_lam_W{50,100}, roll_eff_spr_ratio_W100）+ STAGE5_FAIL = `liq_asym_top5_W5`
- **问题**：这些到底"失效"是什么标准——KS test fail？特征重要性 ≈ 0？OOF 上相关性 < 阈值？为什么 ask_diff1 失效但 bid_diff1 没失效（不对称）？为什么 spread1/5/10 的 qrank 都失效但 spread2/3/4/6/7/8/9 没在列里（因为压根没生成那些 qrank）？为什么 kyle_lam 两个窗口都失效但 kyle_inv 都没失效？

### Q21: 为什么短 horizon 用 share_with=60 而不是单独训
- **位置**：`04_build_pkg/thresholds.json:5,15,25,35`
- **问题**：h=5 和 h=60 的目标量纲完全不同——h=5 的 Δmid 噪声占比大，h=60 信号更强。共享同一个 ensemble 然后缩放阈值，物理上不成立。为什么不为 h=5/10/20/40 各训 50+50 模型？README 解释是"平台取 max 只升不降"——但这是兜底而非最优解。

### Q22: FEE = 0.0001 是平台公布的吗？
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:23`、`03_train_nn/train_T188v2_nn_seed.py:33`
- **问题**：0.01% 是单边还是双边？平台规则是否明文公布？SPO+ 损失里用 fee_eff = FEE × ((mp_th+1)+(mp_t+1))/(mp_t+1)，这个公式假设双边、买入和卖出都按 FEE × 当时价格收。如果平台用别的规则，整个 SPO+ 校准就偏了。

### Q23: target_scale = 1/std(y) 的依据
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:266`
- **问题**：把 target 缩放到 std=1，可以让 L2 loss 在合理量级。但 SPO+ 阶段 fee 也乘以 target_scale，是否会因为 target_scale 改变而破坏 fee 的相对量级？计算用 Phase 1 train（dates 0-79）的 std；Phase 2 用 dates 0-119 时数据分布不同，target_scale 仍沿用 Phase 1 的值是否合理？

### Q24: 11 epoch + CosineAnnealingLR(T_max=11) 让 lr 从 3e-5 衰减到 0
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:436`
- **问题**：最后一个 epoch lr ≈ 0，可能没贡献。为什么用 cosine 而不是恒定 lr 或者 linear warmup？

### Q25: Phase 1 Cosine 的 T_max = 50 但早停可能在 20 epoch
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:293-294`
- **问题**：T_max=50 意味着 cosine 周期假设训 50 epoch；但如果 patience=10 在 epoch 20 触发早停，模型只看到了 cosine 前 40% 的 schedule（lr 从 1e-3 仅衰减到 ~0.85e-3）。schedule 是否应该用 T_max=PHASE1_PATIENCE 或 reduce_on_plateau？

### Q26: AdamW weight_decay = 1e-4
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:43, 49`
- **问题**：1e-4 是 PyTorch AdamW 默认；为什么这里要复用而不是基于这个回归任务 tune？

### Q27: bagging_freq = 5 in LGB
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:75`
- **问题**：每 5 轮重采样数据。常见值是 1 / 5 / 10。5 是 LightGBM 文档示例值？做过对照吗？

### Q28: min_data_in_leaf = 100
- **位置**：同上 line 74
- **问题**：100 行/叶子在 1.47M 样本上是 0.007%——非常低，鼓励过拟合。为什么不是 1000？做过对照吗？

### Q29: learning_rate = 0.05 for LGB
- **位置**：同上 line 73
- **问题**：常见 0.05 / 0.1。0.05 配合 330 轮意味着等价于 0.1 配合 165 轮。是否对比过？

### Q30: num_threads = 18
- **位置**：同上 line 76
- **问题**：18 是某台特定机器的 cpu 数？为什么不是 -1（自动）？提交平台 CPU 数固定吗？

### Q31: CHUNK = 200_000 在 aug 拼接中
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:43`、`03_train_nn/train_T188v2_nn_seed.py:410`
- **问题**：分块为了控制内存。为什么 200k？1.47M 行需要 8 块。是测试出的最优？

### Q32: rng seed 公式 `S * 7919 + 42`、`S * 7919 + 1`、`S * 7919 + 137`
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:142`、`03_train_nn/train_T188v2_nn_seed.py:270, 406`
- **问题**：7919 是质数；42 / 1 / 137 不是。为什么不直接用 seed？为什么 LGB 和 NN Phase 1 不用同一个 offset，导致同 seed 下 aug 数据不同？

### Q33: clip_grad_norm = 5.0
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:318, 466`
- **问题**：5.0 是常见值但偏大；常用 1.0。是否做过 1.0 / 2.0 / 5.0 / 10.0 对照？

### Q34: CLIP = 10.0 (standardization clip)
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:35`
- **问题**：std-clip 到 ±10σ 还会留下 ~0.01% 极端值。为什么 10 而不是 5 / 8？

---

## 2. 架构选择

### Q35: MLP [359→256→128→64→1] 隐藏层为啥这样选
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:34`、`Predictor.py:9`
- **代码摘要**：HIDDEN = (256, 128, 64)
- **问题**：3 层逐步收窄到 64。为啥不是 (512, 256, 128) 或 (128, 128, 128) 或 (256, 256)？做过宽度/深度消融吗？

### Q36: 为什么用 MLP 而不是 GRU / Transformer / 1D-CNN
- **问题**：LOB 数据本质是时间序列，100-tick 窗口里有时序结构；MLP 只看 last-tick + 聚合特征，丢了 tick-level 动态。为什么不用序列模型？做过对照吗？

### Q37: LayerNorm + GELU + Dropout 组合
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:81-93`
- **问题**：为什么 LayerNorm 而不是 BatchNorm？为什么 GELU 而不是 ReLU/SiLU？GELU 在 LOB 数值范围（z-score 后多在 [-3,3]）上几乎线性，与 ReLU 差别不大。这是经过测试还是惯性选择？

### Q38: Kaiming 初始化 (nonlinearity="relu")
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:99`
- **问题**：实际激活是 GELU，但初始化按 ReLU 配置。轻微不匹配——是否应该用 nonlinearity="gelu" 或 xavier？

### Q39: AdamW vs Adam / SGD
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:292, 435`
- **问题**：AdamW 是默认选择，但在小模型 + 回归任务上，SGD+momentum 有时更稳。是否做过对照？

### Q40: CosineAnnealingLR 选择
- **位置**：同上 293-294, 436
- **问题**：与 ReduceLROnPlateau / OneCycleLR / StepLR / constant 的对比？

### Q41: 为什么 LightGBM 而不是 XGBoost / CatBoost
- **问题**：CatBoost 在带 categorical 的金融数据上常更优；XGBoost 在精度上有时优于 LGB。是否做过对照？CLAUDE.md 提到 LGB GPU 比 CPU 快 3.2x，但 CatBoost GPU 也可用。

### Q42: 两阶段（L2 pretrain → SPO+ fine-tune）vs 直接 SPO+
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:227-503`
- **问题**：为什么不直接从随机初始化跑 SPO+？SPO+ 损失梯度可能噪声大、不稳定，但 11 epoch 不算长——pretrain 真的必要吗？README 说"L2-only +19.23"，但没说"直接 SPO+ 无 pretrain"的得分。

### Q43: SPO+ DFL vs 直接 PnL loss
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:117-127`
- **问题**：SPO+ 是 Elmachtoub & Grigas (2022) 提出的、原本用于线性 LP 的 surrogate。这里改成 1D 三选项（long/flat/short）任务后是否仍有 Fisher consistency？为什么不直接对 PnL = z × (Δmid - fee × sign(z)) 做光滑代理（如 expected-PnL 通过 softmax 近似 argmax）？

### Q44: L2 + λ·SPO+ 联合损失而非纯 SPO+
- **位置**：`03_train_nn/train_T188v2_nn_seed.py:462`
- **问题**：保留 L2 项是为了"锚定"还是为了梯度平滑？λ=30 的相对量级意味着 SPO+ 主导。为什么不让 λ=∞（纯 SPO+）？为什么不 λ=1？

### Q45: 没有 BatchNorm / residual connection
- **问题**：359→256 这种维度跳变，没 residual 也行——但 (256, 128, 64) 三层信号通路狭窄，加 residual 可能稳定。是否做过对照？

### Q46: 没有学习 NN-LGB 加权 / stacking
- **位置**：`Predictor.py:357`
- **问题**：固定权重均值，为什么不在 holdout 上学一个 ridge 或 stacking？

---

## 3. 特征工程

### Q47: 154 维原始特征里 amount_delta 单独做 sign-preserving log1p
- **位置**：`01_build_features/build_schemeP_cache.py:108-110`、`04_build_pkg/Predictor.py:283`
- **问题**：只对 amount_delta 做 log1p，其他原始特征（volume_delta, bsize* 等同样有重尾）不做。为什么只挑 amount_delta？是否应该对所有重尾特征统一处理？

### Q48: amount_delta 在 raw_last 是 log1p，在 extras 里却是 raw —— 不一致！
- **位置**：`01_build_features/build_schemeP_cache.py:104-110`、`fast_features_batch.py:311, 436` 等多处
- **问题**：raw_last vector 第 5 列 amount_delta 是 sign × log1p(|amt|)。但 Stage 1/2 中很多特征（dualz_amount_delta、qrank_amount_delta、kyle_inv, kyle_lambda, vol_burst 等）用 X3d[:, :, col_idx["amount_delta"]]，即原始 amount_delta。**同一个底层量在不同特征里被赋予不同尺度的表示**。这是有意设计（让模型见到两种尺度）还是 bug？

### Q49: 154 原始 + 196 extras + 20 Stage5 = 370，drop 11 → 359
- **位置**：`build_schemeP_cache.py:212`
- **问题**：原始 154 全部进入模型，没有任何剪枝。其中 close、open、high、low 是当前 tick 的"高低开收"——但 tick 数据 1 个 tick 内 OHLC 几乎相同。这 4 列是否有信息量？

### Q50: 5 个 stage（T3 + Stage1 + Stage2 + Stage3 + Stage5）为什么分这么多？Stage 4 在哪？
- **位置**：`fast_features_batch.py` 整个文件
- **问题**：T3 有 69 个、Stage 1 有 54、Stage 2 有 59、Stage 3 有 14、Stage 5 有 20。Stage 4 缺失（被丢弃？）。每个 stage 的设计动机是什么？是不同实验回合（T3=第3轮调研、Stage1=R34 输出）的产物，简单堆在一起？

### Q51: T3 MLOFI 用 `>=` / `<=` 同时计数
- **位置**：`fast_features_batch.py:169-173`
- **代码摘要**：`ind_b_up = (b >= b_prev); ind_b_dn = (b <= b_prev)`
- **问题**：当 b == b_prev（价格不变）时**两个 indicator 都为 1**，公式变成 `e = bs - bs_prev`（实际的 size 变化）。设计意图正确（Cont 2014 MLOFI 在 price-equal 时跟踪 size），但与 Stage 5 OFI 用 strict `>` 和 `<` 不一致（Stage 5 单独处理 `==` case，line 645-650）。两份 OFI 实现风格不统一——是有意要双信号源吗？

### Q52: T3 EWMA alphas = (0.05, 0.1, 0.3, 0.5)
- **位置**：`fast_features_batch.py:31`
- **问题**：4 个 alpha 的等效窗口 ~20, 10, 3.3, 2 ticks。为啥不取 (0.01, 0.05, 0.1, 0.3) 覆盖更长时间尺度？

### Q53: T3 RV windows = (5, 10, 20, 50)
- **位置**：`fast_features_batch.py:30`
- **问题**：100-tick 窗口里取 RV 用最长 50。为啥不要 RV_W100？

### Q54: dual_z（37 个）= z_short(W=20) - z_long(W=100)
- **位置**：`fast_features_batch.py:286-290`
- **问题**：z_s - z_l 是 z-score 差分；意义是"短期相对长期偏离"。为什么不直接给 z_short 和 z_long 两个特征让模型自学差？为什么 short=20, long=100？做过其他窗口对吗？

### Q55: kyle_inv 公式 `amt_last / cbrt(|amt_last| * std(r))`
- **位置**：`fast_features_batch.py:312-321`
- **问题**：基于 Kyle (1985) 模型变换的"逆 Kyle"。1/3 次根来源于 dimensional analysis？公式中只用 amt_last（不是 amt_W:）是否丢失了窗口信息？

### Q56: signed_rv 公式 `(rp - rn) / (rp + rn)`
- **位置**：`fast_features_batch.py:299-308`
- **问题**：信号 ∈ [-1, 1]，正 = 上涨能量主导。为什么不也提供 rp, rn 原始值？

### Q57: qrank 用 float32 计算
- **位置**：`fast_features_batch.py:368`
- **代码摘要**：`x32 = x.astype(np.float32, copy=False); ... (seg <= last_val).mean(...)`
- **问题**：rank 不依赖精度，为什么显式 cast 到 float32？是否因为某次实验里 ref 实现用了 float32，为了 byte-equivalence？这种"为了数值复现性的固化"是否会引入隐患？

### Q58: gofi 与 t3_mlofi 重复
- **位置**：`fast_features_batch.py:396-429` vs 159-184
- **问题**：t3_mlofi 用 `>=` / `<=`；gofi 用 `>` / `<` / `==`，并对 ask 做了**符号反转**（`bid_cont - ask_dn - ask_eq`，注意减 ask_eq，而 bid_eq 是加）。两个特征族在做同一件事但公式不一致。模型同时输入两套——是为了"互为校验"还是无意冗余？

### Q59: kyle_lambda 用 sqrt(|amt|) 作为 signed dollar volume
- **位置**：`fast_features_batch.py:436-453`
- **问题**：signed_dvol = sign(Δmid) × √|amt|。√ 是为了压重尾，但为什么不是 log？lambda 被 clip 到 [-1, 1]，clip 是否会把强信号截掉？

### Q60: vol_burst 公式 last / mean(W)
- **位置**：`fast_features_batch.py:455-466`
- **问题**：当前 tick 的成交量除以窗口均值。clip 到 [-100, 100] / [0, 100]。如果用 z-score 化（(last-mean)/std）是不是更好？

### Q61: rv_ratio_W_pairs = ((5,50), (20,100), (50,100))
- **位置**：`fast_features_batch.py:72`
- **问题**：(5,100) 缺失，(5,20) 缺失。为什么只挑这 3 对？

### Q62: jshare_W = (20, 30, 50, 100) 含 30
- **位置**：`fast_features_batch.py:73`
- **问题**：其他特征族普遍用 (20, 50, 100)，jshare 多了个 30。为什么这个特殊？

### Q63: cancel_imb clip 到 [-1, 1]
- **位置**：`fast_features_batch.py:553`
- **问题**：cancel share 比例 in [0,1]，差值 in [-1,1]。clip 在这里多余但无害——为什么写？

### Q64: roll_eff_spr_ratio 公式 2*sqrt(max(-cov, 0)) / (qspread+1)
- **位置**：`fast_features_batch.py:556-576`
- **问题**：Roll's effective spread 经典公式是 2√(-cov(Δp, Δp_{prev}))，前提是 cov<0（bid-ask bounce）。代码里 max(-cov, 0) 处理 cov>0 的情况返回 0。对 LOB 数据 cov 可能并不可靠地为负。这个特征实际有信号吗？

### Q65: Stage 5 / OFI 用 prev 填充而非 NaN
- **位置**：`fast_features_batch.py:641-644`
- **代码摘要**：`b_prev[:, 0] = b[:, 0]` (vs T3 用 NaN)
- **问题**：Stage 5 在 t=0 时 b_prev = b[0]，所以 Δb=0；T3 用 NaN 然后填 0。语义不同——Stage 5 把第一个 tick 当作"无变化"，T3 当作"未知"。为什么不统一？

### Q66: Stage 5 spread_reg 用 IQR 归一化
- **位置**：`fast_features_batch.py:687-699`
- **问题**：当 IQR=0（窗口内 spread 恒定）取 inf 让分子无效；clip 到 [-10, 10]。为什么不用 std？

### Q67: Stage 5 liq_asym top-5 vs top-10
- **位置**：`fast_features_batch.py:721-733`
- **问题**：用 top-5 而不是 top-10 sum。为什么 5？做过 1 / 3 / 10 对照吗？

### Q68: Stage 5 liq_asym_top5_W5 被 drop（在 STAGE5_FAIL_NAMES 里）
- **位置**：`train_T188v2_lgb_seed.py:40`
- **问题**：W=5 太短，信号几乎是单 tick 比值的对数。一开始为什么要生成这个特征？

### Q69: Stage 5 trade_pers = lag-1 autocorr(sign(Δmid))
- **位置**：`fast_features_batch.py:701-719`
- **问题**：sign(Δmid) 只取 -1/0/+1。当大部分 Δmid=0（tick 价不变）时这个 corr 几乎全是噪声。是否应该过滤掉 Δmid=0 的 tick？

### Q70: 没有 Hawkes 自激发特征 / 无 sin/cos 时间编码
- **问题**：LOB 数据有显著 time-of-day 效应。但 date 被屏蔽、tick 索引 t 也没进入特征。这是因为评测时被打乱（CRITICAL_CONSTRAINTS）？是否可以引入"距开盘 tick 数 / total ticks"作为受限的时序特征？

### Q71: 没有 Alpha101 / Alpha191 跨截面因子
- **问题**：5 个 sym 同时出现，可以构造截面 z-score / rank。但模型 sym-agnostic，每个 sample 独立计算特征。这个限制是怎么权衡的？

### Q72: 没有 Roll-effective-spread / TSRV 等知名估计量
- **问题**：roll_eff_spr 在 Stage3 但被 drop。TSRV、jumps detection（Lee-Mykland）都没用。为什么放弃？

### Q73: 没有特征剪枝
- **问题**：370 维全输入，drop 11 后留 359。为什么不基于 LGB importance / SHAP 剪到 ~150 / 200？信噪比可能更高。

### Q74: Stage 2 skew clip 到 [-10, 10]
- **位置**：`fast_features_batch.py:391`
- **问题**：return skew 量级一般 < 5，clip 10 是兜底。为什么不 5？

### Q75: 没有截面特征 / 不分 sym 归一化
- **问题**：Stage 1 dual_z 已经做了滚动 z 内部归一化；模型推理时也用全局 feat_mean/std。为什么不允许 per-sym standardization——CRITICAL_CONSTRAINTS 禁止 sym 输入模型，但**预处理阶段**用 sym 也是禁的吗？

---

## 4. 训练协议

### Q76: M7 全数据重训 vs walk-forward 留 val
- **位置**：`train_T188v2_nn_seed.py:387-394`、`train_T188v2_lgb_seed.py:127`
- **问题**：M7 用 dates 0-119 全部训。但 dates 80-95 是 val、96-119 是 test。在 Phase 2 重训时 val/test 全部一起用——是否会过拟合到本地 holdout？README 说 +34.44 vs +28.16，但平台数据是 OOD，过拟合 holdout 怎么没拖累？

### Q77: Phase 1 早停 patience epsilon = 1e-10
- **位置**：`train_T188v2_nn_seed.py:331`
- **代码摘要**：`if val_mse < best_val_mse - 1e-10`
- **问题**：1e-10 在 float32 上小于机器精度，等效于"任意改善即认可"。改成 1e-6 / 1e-4 是否更稳？

### Q78: Phase 1 early stop 仅用 mse，没用 corr / PnL
- **位置**：同上 331
- **问题**：mse 最优 ≠ 决策最优。是否应该用 corr / PnL-on-val 早停？或多目标？

### Q79: Phase 2 不用 val、不早停
- **位置**：`train_T188v2_nn_seed.py:443-477`
- **问题**：固定 11 epoch，无法监控是否过拟合。如果某 seed 在 epoch 8 就开始过拟合，多跑 3 epoch 就毁了。

### Q80: class_balanced_weight 用 3 类频率
- **位置**：`train_T188v2_nn_seed.py:63-67`、`train_T188v2_lgb_seed.py:47-51`
- **问题**：用 3-class label_h60（短涨/不动/短跌）的频率算 sample weight。但回归目标是 Δmid。回归 + 分类权重的混搭语义模糊——为什么不直接按 |y| 或 |y|^α 加权？

### Q81: aug_a：x' = x × U(0.8, 1.2) 对每个 feature 独立
- **位置**：`train_T188v2_nn_seed.py:271`、`train_T188v2_lgb_seed.py:54-61`
- **问题**：每个 feature 独立缩放破坏了特征之间的几何关系（比如 bid1 与 bsize1 物理上不能独立缩放）。为什么不做 sample-level 同步缩放（所有特征用同一个 scale），或者只对 amount/volume 类做？

### Q82: aug_a 让训练数据翻倍
- **位置**：同上
- **问题**：拼接 [X, X_aug] 后 y 完全相同——本质上是 sample weighting。为什么不直接 dropout features 或者 mixup？

### Q83: NN 与 LGB 用了同样的 aug_a
- **问题**：GBDT 对单调变换不变（feature × constant 不改变分裂点 ranking）；但 aug 是**每个 sample 独立缩放**，所以打破了原数据的相对顺序——这是有效的。但等价于在数据上加噪音，是否会让 LGB 学到无意义模式？

### Q84: y = (mp_th - mp_t) / (mp_t + 1.0)，为啥分母 + 1？
- **位置**：`train_T188v2_lgb_seed.py:134`、`train_T188v2_nn_seed.py:70-72`
- **问题**：标准 return 是 (p_h - p_t) / p_t。这里 + 1 看起来像"避免除零"。但如果 midprice 已经被某种 normalization 处理过（量级在 1 附近？），那 +1 会显著改变 magnitude。midprice1 的实际数值范围是多少？

### Q85: fee_eff 公式 `FEE × ((mp_th+1)+(mp_t+1)) / (mp_t+1)`
- **位置**：`train_T188v2_nn_seed.py:75-77`
- **问题**：化简 ≈ FEE × (1 + (mp_th+1)/(mp_t+1)) ≈ 2×FEE 当 mp_th ≈ mp_t。看起来是双边 fee。但平台 fee 规则是什么？

### Q86: Phase 2 数据顺序：concat train + val + test
- **位置**：`train_T188v2_nn_seed.py:390-394`
- **问题**：concat 后用 randperm shuffle。但 NaN 处理用 train 的 feat_mean 填充——val/test 里若有 train 没出现过的 NaN pattern，填充值可能不合理。

### Q87: Phase 2 用 Phase 1 的 feat_mean/feat_std
- **位置**：`train_T188v2_nn_seed.py:374-375`
- **问题**：Phase 1 mean/std 来自 dates 0-79；Phase 2 用 dates 0-119。standardization 偏移 →训练时输入分布漂移。为什么不重新算？

### Q88: 整个数据集放入 CPU RAM 后再分 batch
- **位置**：`train_T188v2_nn_seed.py:425, 283`
- **问题**：X_full = 2*1.47M*359*4 ≈ 4.2 GB（Phase 1 train 之外还有 aug）。Phase 2 全数据 + aug ≈ 4.2 GB。单 GPU 训 50 seed 串行，每次都重新 load。可否预 cache 到 disk？

### Q89: 用 torch.randperm 在 CPU 上 shuffle，每 batch 同步到 GPU
- **位置**：`train_T188v2_nn_seed.py:306-313`
- **问题**：每 batch host→device 传输 ~ batch_size × 359 × 4 字节，对训练速度影响如何？是否应该 pin_memory + prefetch？

### Q90: predict_chunked 用 batch=16384
- **位置**：`train_T188v2_nn_seed.py:107-114`
- **问题**：用更大 batch 加速 inference。但 LayerNorm 是 per-sample 操作，与 batch size 无关。为什么挑 16384？

### Q91: NN 训练时 `torch.cuda.manual_seed_all(S)`，但 `torch.manual_seed(S)` 在 CUDA_VISIBLE_DEVICES 之后才设
- **位置**：`train_T188v2_nn_seed.py:177, 194-196`
- **问题**：CUDA_VISIBLE_DEVICES 在 cuda init 前设置才生效。代码先 `os.environ[...] = args.cuda` 再 `torch.device("cuda" if ...)`。OK 顺序对。但 multi-GPU 并行训不同 seed 时 GPU0 上跑 seed 1-10，GPU1 上跑 11-20 等——确保 RNG 独立吗？

### Q92: amount_delta 在 raw_last 做 log1p，但 NN/LGB 共享同一 raw_last
- **位置**：`train_T188v2_lgb_seed.py:131`、`build_schemeP_cache.py:108-110`
- **问题**：训练时 cache 里 raw_last[amount_delta] 已经是 log1p；Predictor 推理时也做同样 log1p。一致。但**Stage 1 dualz_amount_delta** 用 X3d 原始 amount_delta，extras 全程不 log1p。同名特征用两个量级——Q48 提过，这里强调 train/inference 一致性是 OK 的，但语义上有歧义。

### Q93: 训练时 `np.where(np.isnan(Xs), 0, Xs)` 之后 `clip(±10)`
- **位置**：`train_T188v2_nn_seed.py:223-224`
- **问题**：NaN→0 后再 clip。但 inf 没显式处理（只在 isnan check）。如果 std=0 导致 div 出 inf？feat_std = maximum(feat_std, 1e-6) 保护，但仍可能 inf 经其他路径产生。

### Q94: Phase 1 weight_decay=1e-4 vs Phase 2 也 1e-4
- **位置**：`train_T188v2_nn_seed.py:43, 49`
- **问题**：fine-tune 阶段是否应该降低 wd（避免把 pretrained 权重拉回 0）？是否对照过 wd=0 的 Phase 2？

---

## 5. Ensemble 设计

### Q95: 50 NN HP 实际只覆盖 12 个独立 combo
- **位置**：`train_T188v2_nn_seed.py:186-188`
- **问题**：（重要、与 Q13 同根）README 说 36 combos，实际 12 个。50 seeds 里每个 combo 重复 4 次。是否应该用 sobol / random sample on 36 grid？或在 README 修正？

### Q96: 5 LGB HP × 10 seeds = 50 model，每组 HP 用 10 个 seed
- **位置**：`02_train_lgb/train_T188v2_lgb_seed.py:83`
- **问题**：每个 HP 复 10 次仅靠 LGB 内部 seed (`seed=S`) 区分。但 LGB 主要随机性来自 feature_fraction / bagging_fraction 采样，10 个 seed 的多样性是否足够？是否应该 5 HP × 5 seeds = 25？

### Q97: 简单均值，不学权重
- **位置**：`Predictor.py:357`
- **问题**：50 个 NN / 50 个 LGB 均值——但 HP combo 之间表现差异大（如 lr=1e-3 vs lr=1e-4），等权可能让差模型拉低均值。为什么不学一个 ridge weight？

### Q98: NN 与 LGB 在 ensemble 内独立平均后再加权
- **位置**：`Predictor.py:345-358`
- **问题**：`pred_nn = mean(50 NN), pred_lgb = mean(50 LGB), pred = (w_nn*pred_nn + w_lgb*pred_lgb)/...` 而不是 `pred = mean(all 100 models with weights)`。两种数学等价（等权时），但加权 1.0:1.5 是 family-level 还是 model-level？目前是 family-level。

### Q99: 没有不同模型族（无 CatBoost、Transformer、GRU）
- **问题**：现有 ensemble 只 MLP + LGB 两族。CatBoost / random forest / 简单线性 baseline 不需要太多工程开销，是否做过对照？

### Q100: Bagging seeds 1-50 顺序
- **位置**：`thresholds.json:51`
- **问题**：硬编码 `[1..50]`。如果训练时漏掉 seed 12，是否会 silently 失败？build_pkg 检查存在性后只 ship 存在的——但 thresholds.json 仍写 1-50，Predictor 在 `_BatchedMLPEnsemble` 仅读真实存在的 npz。是否应该动态生成 thresholds.json？

### Q101: seed 数 50 vs HP 数 5（LGB）/12（NN）
- **问题**：LGB 每 HP 10 seed → 多样性主要来自 bagging seed。NN 每 unique HP 4.17 seed → 多样性来自 init + dropout。两族多样性结构不同，但平均权重一样——是否应该 LGB 取均值（family-mean）后再与 NN family-mean 加权？目前正是如此（Predictor），但是否考虑过 model-level mixed weight？

---

## 6. 决策层

### Q102: EV gate 三选项硬阈值
- **位置**：`Predictor.py:255-271`
- **问题**：`pred > thr_up → 2, pred < -thr_dn → 0, else 1`。比 classification 的 softmax + argmax 更"硬"。如果模型预测在阈值附近抖动，会频繁切换 action（手续费高）。是否应该加 hysteresis（双阈值带）？

### Q103: per-sym sigma 的来源未在代码中
- **位置**：`thresholds.json:57`
- **问题**：per_sym_sigma 是写死的 JSON 值。代码里没有"如何计算"的 script。这是离线 calibration（如 OOF 残差 std）但流程不透明。提交时如何保证 sigma 与当时的模型对应？

### Q104: per_sym_beta 离散值 {0.0, 0.1, 0.3, 0.4}
- **位置**：`thresholds.json:56`
- **问题**：是 grid search 出的最优？grid 是 {0.0, 0.1, 0.2, ..., 0.5}？为什么 sym3、sym4 是 0（不需要 abstain）——是统计噪声还是真有信号？

### Q105: 没用 isotonic / Platt 校准
- **问题**：回归 pred 直接当 EV，不做单调校准；分类设定下 isotonic regression 是标准做法。这里不适用？

### Q106: pred 是 (mp_h-mp_t)/(mp_t+1)，与 EV 单位的关系
- **问题**：阈值 0.0003 是相对收益（30bps）。但 fee≈0.0001（10bps）。所以 thr/fee ≈ 3。是否做过 thr_up = α × fee 的形式化处理？

### Q107: Conformal "wrapper" 的实际语义
- **位置**：`Predictor.py:255-262`
- **问题**：`eff_thr = thr + beta × sigma`，sigma 是常数（per_sym 标量），beta 是常数。所以 eff_thr 只是 per-sym 加了个常数 offset，不是真正 conformal（应该用 batch 内的某分位数）。命名 "conformal" 是否名不副实？

### Q108: sym 从 batch 最后一 tick 取
- **位置**：`Predictor.py:299`
- **问题**：100-tick 窗口最后一行的 sym。理论上窗口内不应跨 sym，但代码没断言。若数据有混合 sym 的 window，行为是什么？

### Q109: OOD sym 默认 band = 0.16 × 0.0004 ≈ 6.4e-5
- **位置**：`Predictor.py:222-223`
- **问题**：默认 band 比 thr_up=3e-4 小得多（~20%），意味着 OOD sym 只比"无 abstain"略保守。是否应该更激进 abstain（OOD 时不出手）？

### Q110: gate 直接 numpy 比较，没 vectorize 优化
- **位置**：`Predictor.py:259-262`
- **问题**：B 行直接 numpy where。OK。但 band_per_row 是 (B,) array，每行不同——`thr + band_per_row` 广播后 OK。无性能问题。

---

## 7. 推理优化

### Q111: 50 NN 用 torch.bmm 批量化
- **位置**：`Predictor.py:78-178`
- **问题**：W[i] 形状 (50, out, in)，h 形状 (50, B, in)，bmm 一次算 50 个网络的前向。N=50 × B=1024 × in=359 × hidden=256 → 第一层中间结果 50×1024×256×4 ≈ 50 MB。OK。但**为什么是 batched bmm 而不是把 50 个 weight 拼成 (out*50, in) 用单个 matmul**？两种方式效率差异？

### Q112: requirements.txt 是 CPU torch
- **位置**：`requirements.txt:1, 6`
- **问题**：`torch==2.5.1+cpu`。所以 Predictor 里的 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 在平台上始终 CPU。代码冗余的 CUDA 分支——是否应该简化？

### Q113: NN 权重保存为 .npz 而不是 .pt
- **位置**：`train_T188v2_nn_seed.py:130-160`
- **问题**：避免 torch 序列化兼容性问题（不同版本 torch 互不兼容）。但 LayerNorm gamma/beta、weight、bias 都要手 extract 然后 manual layer_norm reimplementation。维护成本高。为什么不用 torch.jit.script？

### Q114: LGB 推理是 CPU sequential（一个一个 predict）
- **位置**：`Predictor.py:306-314`
- **问题**：50 个 booster 串行预测，注释说"~84ms"。LightGBM 4.6 支持 multi-threading；是否所有 booster 共享同一个 prediction multi-thread pool？是否可以 stack 后 batch predict？

### Q115: NN 前向用 `GELU(approximate="tanh")`
- **位置**：`Predictor.py:171`
- **问题**：PyTorch 默认 GELU 是 erf 实现；tanh 近似快但与训练时 GELU 是否一致？训练代码（line 89）只用 `nn.GELU()`（默认 erf）。**推理 GELU != 训练 GELU**！这会带来误差。

### Q116: LayerNorm 手动实现 var(unbiased=False) + 1e-5
- **位置**：`Predictor.py:166-170`
- **问题**：torch.nn.LayerNorm 默认 eps=1e-5、unbiased=False。手动实现是为了支持 per-NN gamma/beta（F.layer_norm 不支持 batched gamma），但需要保证语义和训练时 `nn.LayerNorm` 完全一致。是否 unit test 过？

### Q117: 推理时 `np.where(np.isnan(h), 0, h)`
- **位置**：`Predictor.py:159`
- **问题**：在 GPU/CPU tensor 上调用，是 torch.where 而非 np.where。OK。但是否会在某些 batch 上出现 NaN 影响均值？

### Q118: `nn_ens.predict_mean(feats)` 返回 mean over N，不输出 std/quantile
- **位置**：`Predictor.py:177`
- **问题**：丢失了 ensemble disagreement 信息——本可用作 abstain 信号。

### Q119: 推理时 `_compute_batch_features` 用 float64 X3d，输出 float32
- **位置**：`Predictor.py:276`
- **问题**：float64 算特征防止数值不稳定 OK。但 raw_last 又 cast 回 float32。是否在 LayerNorm 前 cast 到 float64 防累积误差？

### Q120: build_pkg.py 复制 fast_features.py 但 Predictor 不用它
- **位置**：`build_pkg.py:66-67`
- **问题**：fast_features.py 只算 196 维（无 Stage5），与 Predictor 用的 fast_features_batch.py 不一致。打包到 pkg 里**完全是 dead code**。为什么 ship？

---

## 8. 5 horizon share_with=60

### Q121: 短 horizon 不训独立模型
- **位置**：`thresholds.json:5, 15, 25, 35`
- **问题**：h=5 的 Δmid 与 h=60 完全不同（量级、噪音、可预测性）。共享 h=60 预测然后缩放阈值，理论上不正确。为啥不训 h=5/10/20/40 各自的 50+50？

### Q122: sqrt(H/60) 缩放阈值——理论 vs 经验
- **位置**：`thresholds.json:7-42`
- **问题**：σ(BM over H) = σ × √H，所以阈值缩放 √H 是默认假设。但 pred 是 h=60 的预测，量纲是"60-tick return"。要"复用"到 h=5 的判定，应该 pred × (5/60) 缩放预测，而不是 thr × √(5/60)。是否在做错误的缩放方向？

### Q123: 短 horizon 提交不会拉低分数？
- **位置**：README:18, 232
- **问题**：README 说"平台取 max，添加只升不降"。但短 horizon 触发的 action 是否会进入 PnL 计算的某个 join？平台规则确认这一点了吗？万一短 horizon 提交了 action=0 而 long 横盘，亏 fee × 量级。

### Q124: 短 horizon 用同一组 NN/LGB ensemble seed list
- **位置**：`thresholds.json:51`
- **问题**：每个 horizon 都共享 50 NN + 50 LGB（通过 share_with 复用预测）。模型只为 h=60 训练，对其他 h 没有针对性。是不是应该至少 fine-tune 1 epoch 到目标 h？

### Q125: thresholds.json 里 short horizon 的 thr_up/dn 都是 sqrt 比例
- **位置**：`thresholds.json` 全表
- **问题**：thr_up (h=5)=0.0000866=0.0003×√(5/60)。逐项核对——但 thr_dn 也是 0.000216×√(5/60)=0.0000624。统一 sqrt 缩放。是否应该不同 horizon 独立 tune？

### Q126: w_nn / w_lgb 短 horizon 仍是 1.0/1.5
- **位置**：`thresholds.json:9-10` 等
- **问题**：所有 horizon 都用 1.0/1.5 加权。如果短 horizon 上 NN 比 LGB 更好（更敏感），权重不应该调？

---

## 9. 代码实现细节

### Q127: Predictor 对 OOD sym 行为
- **位置**：`Predictor.py:294-304`
- **问题**：try int(sym); 若 ValueError/TypeError/IndexError → 用 default band。但如果 sym 是 float 100.0（数值合法但 OOD）会 cast 成 100 然后查不到 → 用 default。**没有显式校验 sym in 0..4**。是否需要？

### Q128: Predictor 对 missing 'sym' 列行为
- **位置**：`Predictor.py:296-297`
- **代码摘要**：`if "sym" not in df.columns: continue`
- **问题**：保持 default band。OK。但 conformal_wrapper.enabled=True 时如果整 batch 无 sym 列，所有行 band=default。平台是否保证 sym 在 df 中？

### Q129: Predictor cross-batch state
- **位置**：`Predictor.py:316-365`
- **问题**：每次 predict() 完全独立——`pred_cache` 局部变量。OK 符合 CRITICAL_CONSTRAINTS。但 `_BatchedMLPEnsemble` 实例在 self 上，weights 加载一次。这是构造时一次，符合 stateless 要求。

### Q130: fast_features_batch 与 fast_features 内部不一致
- **位置**：见 Q120
- **问题**：fast_features.py 仅 196 维（无 Stage5），fast_features_batch.py 216 维。两个 docstring 都说"single-window reference"，但功能集不同。极易混淆。

### Q131: build_pkg sanity check 极弱
- **位置**：`build_pkg.py:35-46`
- **问题**：只 check 模型文件存在与否。不 verify：(1) thresholds.json 里的 ensemble_seeds 都对应实际文件；(2) NN npz 里的 in_dim / hidden 一致；(3) feat_mean/std 维度匹配。一个缺失 / 错配的 seed 会让 Predictor crash 在 platform。

### Q132: build_pkg WARNING 但不报错
- **位置**：`build_pkg.py:58-59`
- **问题**：缺 NN/LGB 只 print warning，仍打包。可能导致 < 50 个模型上线。

### Q133: build_pkg 不验证 NN 与 LGB seed 配对
- **位置**：同上
- **问题**：nn_h60_seed{S} 与 model_h60_seed{S} 数量可能不同——Predictor 各取各的 list，但 ensemble 失衡。

### Q134: Predictor smoke test 用 standard_normal
- **位置**：`Predictor.py:368-388`
- **问题**：合成数据测试只验证不 crash；不验证数值正确性、不与训练时输出对比。

### Q135: NN npz extract_npz 的 layer_idx 计数
- **位置**：`train_T188v2_nn_seed.py:147-156`
- **代码摘要**：每个 hidden block 是 Linear + LayerNorm + GELU + Dropout = 4 modules。手动 `layer_idx += 1`/`+= 2`/`+= 1`（共 4）。若 use_layernorm=False，跳过 1 个 → layer_idx 错位。
- **问题**：use_layernorm=True 是硬编码默认 + Phase 1 ckpt 也只存 use_layernorm=True。但 extract 函数 take `ckpt["use_layernorm"]`，若有人改成 False，extract 会取错权重。脆弱。

### Q136: NN 训练时 phase1 中保存 `phase1_val_split` string 但 Phase 2 不读
- **位置**：`train_T188v2_nn_seed.py:363, 509-512`
- **问题**：元数据冗余但无害。一致性即可。

### Q137: NN final_ckpt 也保存了 keep_idx，但 npz 已经存 keep_idx
- **位置**：`train_T188v2_nn_seed.py:144, 487`
- **问题**：两个地方都有 keep_idx。Predictor 读 npz 里的 keep_idx，pt 里的 keep_idx 是冗余备份。

### Q138: assert n_train 用 \( total，但 Phase 2 用 train+val+test 全部
- **位置**：`train_T188v2_lgb_seed.py:117`
- **代码摘要**：`n_train_used = n_total * 2`（aug 翻倍）
- **问题**：n_train_used = (train+val+test) × 2 ≈ 2.94M 行。pre-alloc ~ 2.94M × 359 × 4 ≈ 4.2 GB。RAM 限制？

### Q139: LGB Dataset feature_name 包含 359 列名
- **位置**：`train_T188v2_lgb_seed.py:171-173`
- **问题**：保存到 model.txt 里。提交平台读 model.txt 时若 feature 顺序不一致会错。Predictor 把特征拼为 raw_last(154) + extras_kept，确保顺序与训练时一致——靠人工保证。脆弱。

### Q140: LGB params seed=S，但 bagging_seed/feature_fraction_seed 不显式设
- **位置**：`train_T188v2_lgb_seed.py:163`
- **问题**：LightGBM 内部派生这些 seed from main seed，但版本间行为可能变。byte-equivalence 风险。

### Q141: Predictor `_load_module` 用 importlib.util
- **位置**：`Predictor.py:181-185`
- **问题**：动态加载 fast_features_batch.py。优点：避免 package import；缺点：mod_name "iter_018_ffb" 是历史命名，与 fname 不对应。混乱。

### Q142: `_BatchedMLPEnsemble.predict_mean` 输入维度判断
- **位置**：`Predictor.py:148-151`
- **代码摘要**：`if X_in.shape[1] != self.in_dim: Xs = X_in[:, self.keep_idx]`
- **问题**：用 in_dim 判断"是否已经 drop fail features"。但 Predictor 总是传入 already-dropped feats（Q130，_compute_batch_features 已 drop），所以条件总是 False。dead-ish code。

### Q143: ProductionPipeline 在不同 horizon 重复用同 ensemble，需要靠 `pred_cache`
- **位置**：`Predictor.py:329-358`
- **问题**：pred_cache 字典缓存按 src_H 索引。若 horizons 列表里 share_with 都是 60，则一次预测 + 4 次复用。OK。

### Q144: `lp` 与 `np_` 文件名硬编码 `h{H}`
- **位置**：`Predictor.py:242-243`
- **问题**：h=60 写死。如果 thresholds.json 加入 h=120 但模型没训，会 silent skip。已经在 `if os.path.isfile(lp)` check，所以不 crash 但可能错漏。

### Q145: 调用 `lgb.Booster(model_file=p)` 50 次
- **位置**：`Predictor.py:249`
- **问题**：每个 booster ~ 数 MB。50 个加载到内存 ~ 数百 MB。Predictor 实例化开销大。

### Q146: `_extract_band_per_row` 用 Python for 循环
- **位置**：`Predictor.py:291-304`
- **问题**：B 个 batch 各算一遍 sym 查找。可以预 vectorize：sym_array = np.array([df["sym"].iloc[-1] for df in batches])。**已经基本是这个写法**——但 try/except 每行——OK。

### Q147: HORIZON_LIST 顺序固定
- **位置**：`Predictor.py:43`
- **问题**：(5, 10, 20, 40, 60) ——平台输出 5 列依此顺序。平台 spec 是否固定？

### Q148: build_schemeP_cache 跳过 missing parquet 不报错
- **位置**：`build_schemeP_cache.py:152-155`
- **问题**：只 print warning。如果用户数据缺一半，仍生成 cache。下游训练在 1/2 数据上跑，最终模型质量差。

### Q149: parquet 文件名约定写死
- **位置**：同上 line 151
- **代码摘要**：`f"snapshot_sym{sym}_date{date}_{sess}.parquet"`
- **问题**：必须这种命名。用户若有不同命名（如 `sym{sym}.date{date}.{sess}.parquet`）需要自己改 build script。

### Q150: 没有 unit test
- **问题**：整个 final_submission_code 没 test 目录、没 assert（除了少数 dim check）。byte-equivalence 验证靠人。生产代码？

### Q151: stage5_features.py 在 01_build_features 但未被 import
- **位置**：`stage5_features.py`
- **问题**：build_schemeP_cache.py 不导入它（Stage 5 已经融入 fast_features_batch.py）。但保留在目录里——dead code 容易让维护者误以为它在用。

### Q152: 文件头注释提到 "iter_018", "iter_010", "iter_009", "iter_012", "T59", "T75", "T81", "T87", "T170", "T188" 等代号
- **位置**：多处文件 docstring
- **问题**：这些历史 iter / T 编号对外部读者完全不可解。是否需要 glossary？

### Q153: Phase 2 fine-tune 时 dropout 仍生效
- **位置**：`train_T188v2_nn_seed.py:431-432`
- **代码摘要**：`model = MLPRegr(..., dropout=dropout_p2)`
- **问题**：fine-tune 阶段是否应该把 dropout 关掉（设为 0），让模型更准确地学决策边界？

### Q154: Phase 1 中 X_va_std 提前 standardize（用 numpy）
- **位置**：`train_T188v2_nn_seed.py:280`
- **问题**：而 X_tr_raw 是 raw，在 forward 里 GPU 上 standardize。两种路径（CPU numpy vs GPU torch）的数值不完全一致（浮点累加顺序）。

### Q155: Phase 1 X_tr_imp NaN imputation 用 feat_mean（在 standardize 前）
- **位置**：`train_T188v2_nn_seed.py:258-263`
- **问题**：fill NaN 后 standardize 会让 fill 值变成 0。OK。但循环 `for d_idx in np.where(nan_mask.any(axis=0))[0]` 是 Python for——可以全 vectorize。

### Q156: NN Phase 2 重新算 NaN imputation，复用 Phase 1 feat_mean
- **位置**：`train_T188v2_nn_seed.py:400-404`
- **问题**：复用 Phase 1 mean 处理 Phase 2 数据。即使新数据有 train 时没有的 NaN pattern。

### Q157: aug 数据加在原数据后面（concat），不是 inplace mix
- **位置**：`train_T188v2_nn_seed.py:271-277`
- **问题**：[X, X_aug] 后 torch.randperm shuffle。OK。但内存翻倍。

### Q158: dualz 用 EPS=1e-8 防除零
- **位置**：`fast_features_batch.py:286`
- **问题**：当 std=0 时 z = (last-mean)/1e-8 → 极大值。后续 NaN→0 fill 把 nan 处理掉，但 inf 不会。`np.where(np.isfinite(dual), dual, 0.0)` 行 289 处理这种情况。OK。

### Q159: stage5 `(b >= b_prev)` 第一 tick 用 b[0]=b_prev[0]，等于 0 diff
- **位置**：`fast_features_batch.py:641-650`
- **问题**：第一 tick e=0。OK。但 T3 用 NaN→0，结果一样。两种风格不统一。

### Q160: ofi 的 Pearson corr 用窗口 W 内自己算 (mx, my, ...)
- **位置**：`fast_features_batch.py:657-668`
- **问题**：corr 在小 W=20 上方差大；clip 到 [-1, 1] 是好的。但当 sxx 或 syy = 0（一段 constant）时 denom = EPS，v 可能非常大但 clip。OK。

---

## 10. README 与代码核对

### Q161: README 说"44 个 R 系列调研"
- **位置**：`README.md:248`
- **问题**：代码里没看到 44 个 R 系列的痕迹。具体每个 R 是什么？哪些被采纳（如 R34 → Stage 1/2/3）？哪些被拒绝？

### Q162: README "T81 warm-start" 怎么实现的
- **位置**：`README.md:159, 253`
- **问题**：代码里 Phase 1 → Phase 2 通过 `model.load_state_dict(ckpt_p1["state_dict"])` 实现，是普通 fine-tune。"warm-start"是术语包装吗？

### Q163: README 表格"+5.51 from M7"
- **位置**：`README.md:230`
- **问题**：早停 +28.16 vs 全数据 +34.44 = +6.28，但 README 写 +5.51。哪个是错的？

### Q164: README 表格"+8.93 from SPO+"
- **位置**：`README.md:232`
- **问题**：L2-only +19.23 vs SPO+ +28.16 = +8.93。OK 一致。但 +19.23 是 L2-only **早停**还是 **L2-only M7**？

### Q165: README 表格"+0.77 conformal"
- **位置**：`README.md:234`
- **问题**：相对 baseline +0.77。具体对照实验是什么？

### Q166: README 提"sym 在 conformal 里用，但训练不用"——但 conformal 算 sigma 时也用了 sym
- **位置**：`README.md:220` vs `thresholds.json:57`
- **问题**：per_sym_sigma 离线 calibration 阶段必然 split by sym。是否违反"sym agnostic"？严格说 Predictor 没用 sym 训练；calibration 用 sym OK——但 README 表述模糊。

### Q167: README 提"Constraint compliance: date 从不使用"——但 build_schemeP_cache 保存 date 字段
- **位置**：`README.md:218` vs `build_schemeP_cache.py:163-164`
- **问题**：cache 里有 date 数组，仅用于分 train/val/test split。模型不用 date，OK。但 README 措辞会让人疑惑。

### Q168: README 提"148 MB 包大小"
- **位置**：`README.md:72`
- **问题**：50 NN npz + 50 LGB txt + 代码。具体每部分大小？

### Q169: README "5 GPU 18 min 训 50 NN"
- **位置**：`README.md:27, 172`
- **问题**：单 NN ≈ 100s（Phase1 50s + Phase2 50s），50 个 / 5 GPU = 10 NN / GPU × 100s = 16.7 min。OK 大致一致。

### Q170: README 提"CRITICAL_CONSTRAINTS.md"——这是哪里的文件
- **位置**：`README.md:240`、`fast_features_batch.py:11`、`Predictor.py:19`
- **问题**：在 workdir 根目录而非 final_submission_code 内。提交平台是否能看到？

### Q171: README "150+150 / 100+100 是否做过"
- **问题**：50+50 是声明的最优，但没说更大 ensemble 是否做过。是否到 100 已经饱和？

### Q172: README 说 "T75 LightGBM Δmid 回归（iter_013 突破）"
- **位置**：`README.md:252`
- **问题**：iter_013 是哪个 commit？git log 里有对应吗？

### Q173: README "T81 MLP L2 预训练协议"
- **位置**：同上
- **问题**：是不是 iter_018 v1 stack 的"T81"？命名混乱。

### Q174: README "T87 SPO+ Smart Predict-then-Optimize"
- **位置**：同上
- **问题**：Elmachtoub & Grigas 2022 原文是关于线性 LP 的 SPO+；这里改成 1D scalar—— Fisher consistency 证明仍成立吗？

### Q175: README "T170/T188 M7 全数据重训 trick"
- **位置**：同上
- **问题**：T170 是 first iter，T188 是 final？两个 T 哪个真正贡献了？

---

## 附：交叉问题（跨章节挑战）

### Q176: Predictor GELU 是 tanh approx，训练用 erf——不一致！
- **位置**：`Predictor.py:171` vs `train_T188v2_nn_seed.py:89`
- **问题**：训练用 `nn.GELU()`（默认 `approximate='none'`，即 erf），推理用 `F.gelu(h, approximate='tanh')`。两者在 |x| > 1 范围有 ~1e-4 数量级差异。这会影响 PnL 吗？为什么不统一？（**这是潜在 bug，严肃**）

### Q177: 训练时 N×B 数据加载，推理时 (50, B, 359) 一次性 stack——内存 vs 速度权衡
- **位置**：`Predictor.py:155`
- **问题**：B=1024 时 50×1024×359×4 ≈ 73 MB tensor。CPU 上 bmm 50 次同样耗时吗？

### Q178: target_scale 是 per-NN（每个 seed 不同），推理时 sum then mean——为什么 mean 在 scale 之后？
- **位置**：`Predictor.py:174-177`
- **代码摘要**：`out = out / self.target_scale; out = out.mean(0)`
- **问题**：每个 NN 输出 z = pred * target_scale_i^{-1}，然后 mean。等价于 weighted mean of pred_i × scale_i^{-1}。如果 target_scale 在 50 个 seed 上差异大，这种 mean 是否最优？

### Q179: NN ensemble 内 hidden / in_dim 一致性靠注释保证
- **位置**：`Predictor.py:84-86, 95-98`
- **问题**：docstring "All NNs share the same in_dim, hidden dims, and keep_idx" 但代码没 assert。如果某个 npz 维度不同，bmm 会 crash 在第一层。

### Q180: feat_mean 是 per-NN，但 keep_idx 是单一 self.keep_idx
- **位置**：`Predictor.py:98, 128-129`
- **问题**：所有 NN 用 npz[0] 的 keep_idx。若某 NN 训练时用了不同 keep_idx，feat_mean shape 仍 359 但语义不同 →错配。

### Q181: requirements.txt 严格 pin 版本
- **位置**：`requirements.txt`
- **问题**：`numpy==2.4.4` （注意 — numpy 2.x），`pandas==2.3.3`。这些版本是否实际存在？numpy 2.4 在公开 PyPI 上的发布时间和兼容性如何？（**待验证**——numpy 2.x 序列的最新版本与所列号是否匹配）

### Q182: scipy.signal.lfilter 用于 EWMA
- **位置**：`fast_features_batch.py:82-100`
- **问题**：调 lfilter 不如直接 cumprod 实现快（lfilter 内部 C 但函数调用开销大）。一些 seed 下用 jit 编译会快很多。

### Q183: PHASE2_BATCH = 4096 fixed，不跟随 Phase 1 batch_p1
- **位置**：`train_T188v2_nn_seed.py:50`
- **问题**：Phase 1 batch 在 {2048, 4096, 8192} 切换；Phase 2 固定 4096。是否应该用 Phase 1 同 batch？大 batch 在 fine-tune 阶段会稀释 SPO+ 梯度信号。

### Q184: 训练 GPU 使用率 / 内存监控
- **问题**：代码里没 GPU memory check / torch.cuda.empty_cache。50 seed 串行训，GPU 不释放可能逐渐占满（虽然 Python 对象释放应该 OK）。

### Q185: 模型为何不输出概率/不确定性
- **问题**：纯 scalar pred。如果输出 (pred_mean, pred_std)，可以在 abstain 时用 std 作为信号。架构 trivially 修改。

### Q186: 中间产物 `.pt` 在 npz 之外保存——但提交包不要 pt
- **位置**：`train_T188v2_nn_seed.py:502`
- **问题**：torch.save 一份 pt 文件。仅供 debugging？

### Q187: NN Phase 1 ckpt 文件名 `anchor_seed{S}.pt`
- **位置**：`train_T188v2_nn_seed.py:211`
- **问题**：`run_all_nn_seeds.sh` 用 `--skip-phase1`。如果 anchor 不存在会怎么样？答：Phase 1 还是会跑（条件是 `args.skip_phase1 and isfile`）。但单 shell 跑 50 seeds 会从无 anchor 开始，每个 seed Phase 1 都跑。

### Q188: build_pkg 不打包 `T81_pretrained/` 子目录
- **位置**：`build_pkg.py:77-94`
- **问题**：Phase 1 ckpt 不上传。OK 因为推理不需要。但用户若复现，需要保留 T81_pretrained。

### Q189: NN ensemble 推理在 CPU 上的实际延迟
- **位置**：`Predictor.py:78-178`
- **问题**：平台 CPU 上 bmm(50, 1024, 256, 359) ≈ 4.7 GFLOP per layer，× 4 层 ≈ 18.8 GFLOP。CPU ≈ 50-200 GFLOPS → 100-400 ms per batch。LGB 50 个 ≈ 84 ms。总特征+ensemble ≈ 500ms。是否满足平台时延要求？

### Q190: Predictor 单次 predict() 处理 B 个 batch
- **位置**：`Predictor.py:316`
- **问题**：B 由平台决定（config batch=1024）。如果平台一次给 B=10000 会内存爆？

### Q191: Smoke test 用 sym=99（OOD）但没断言 action 落在 [0,1,2]
- **位置**：`Predictor.py:382-388`
- **问题**：只 print，不 assert。

### Q192: `_doc` 字段在 thresholds.json 顶层
- **位置**：`thresholds.json:2`
- **问题**：JSON 不支持 comment，所以加 `_doc` key。Predictor 不读它，OK。

### Q193: 没有 manifest checksum 验证
- **位置**：`build_pkg.py:107-115`
- **问题**：manifest.json 写了 seed 列表，但不写文件 md5。下载后无法校验完整性。

### Q194: NN 在 GPU 上跑 11 epoch，但 batch 在 CPU—GPU 传输
- **位置**：`train_T188v2_nn_seed.py:451`
- **问题**：每 batch 用 `to(device, non_blocking=True)` 但 X_tr_raw 是 CPU tensor。pinned memory 没开启，可能 IO 限速。

### Q195: PHASE2_WD = 1e-4 = PHASE1_WD
- **位置**：`train_T188v2_nn_seed.py:43, 49`
- **问题**：fine-tune 与 pretrain 同 wd——重要决策（见 Q94），重复列出。

### Q196: 训练用 `class_balanced_weight` 来 weight loss——但 SPO+ 没考虑 weight 的语义
- **位置**：`train_T188v2_nn_seed.py:460-461`
- **问题**：SPO+ loss 对每个 sample 算 ell，然后 `(spo_per * wb).sum() / wb.sum()`。但 SPO+ 假设的是单 sample regret；weighted SPO+ 是否仍是凸 surrogate？理论上可疑。

### Q197: Predictor 没有 timeout / error handling
- **位置**：`Predictor.py:316`
- **问题**：若某 batch 计算抛异常，整个 predict() 失败。平台会怎么处理？

### Q198: 模型不输出 horizon-specific predictions（共享 h=60）
- **位置**：`Predictor.py:338`
- **问题**：见 Q121-126 章节，同根问题。

### Q199: build_pkg 不 zip
- **位置**：`build_pkg.py:118-120`
- **问题**：只生成 pkg 目录，需要手工 `cd pkg && zip ...`。run_pipeline.sh 里有 zip 步骤，但 build_pkg.py 单独使用时容易遗漏。

### Q200: 没有版本号 / metadata
- **位置**：整个代码
- **问题**：没有 `VERSION` 文件 / 没有 `__version__`。再次提交时无法追溯。

---

**总计：200 个问题（按主题分类）**

| 章节 | 问题数 | 起止 |
|------|------|------|
| 1. 超参 / 魔法常量 | 34 | Q1-Q34 |
| 2. 架构选择 | 12 | Q35-Q46 |
| 3. 特征工程 | 29 | Q47-Q75 |
| 4. 训练协议 | 19 | Q76-Q94 |
| 5. Ensemble 设计 | 7 | Q95-Q101 |
| 6. 决策层 | 9 | Q102-Q110 |
| 7. 推理优化 | 10 | Q111-Q120 |
| 8. 5 horizon share_with=60 | 6 | Q121-Q126 |
| 9. 代码实现细节 | 34 | Q127-Q160 |
| 10. README 与代码核对 | 15 | Q161-Q175 |
| 附录: 交叉问题 | 25 | Q176-Q200 |
