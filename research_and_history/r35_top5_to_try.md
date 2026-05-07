# R35 Top 5 — 精选下一步尝试

> 选择标准：(a) 与 iter_006 (LGB 5-seed h_60, LOSO +13.61) 互补；(b) 兼容硬约束（sym-agnostic / stateless / no-date）；(c) 实施成本 ≤ 1 工作日；(d) 预期非负 LOSO Δ。
>
> Top 5 按"做的顺序"排，不是"重要性"。

---

## #1 — LightGBM Dart booster, h_60 5-seed（最低风险）

### 为什么先做
- **代码改动最小**：T29/T31 的 `train_loso.py` 复用，只改 `params['boosting_type']='dart'`。
- 多篇 2024-2025 金融时序文章直接选 Dart 作 booster（Nature HSS 2025、MDPI Systems 2025）。
- 实测 Dart 比 gbdt **MSE −13%、train-test gap −22%** → 我们正缺缩小 OOD gap 的杠杆。
- 与 #2 CatBoost 是不同 tree split / regularization mechanism → ensemble 多样性来源不同。

### 实施 plan
1. 复制 `experiments/T31_aug_optim/train_loso.py` → 新 `experiments/T44_lgb_dart_h60/`。
2. 改 params：
   ```python
   params.update({
       'boosting_type': 'dart',
       'drop_rate': 0.1,           # 每 iter drop 10% trees
       'max_drop': 50,             # 最多 drop 50 trees
       'skip_drop': 0.5,           # 50% iter 不 drop（速度）
       'uniform_drop': False,
       'xgboost_dart_mode': False,
       'num_iterations': 2000,     # Dart 不能 early_stop，固定多
       # 移除 'early_stopping_rounds'
   })
   ```
3. **保留**：feature set (Scheme C 226-d)、aug_a、5 seed 列表、5 fold LOSO。
4. 训完后：
   - 单独评 Dart 5-seed h_60 LOSO sum（vs iter_006 +13.61）。
   - **混合 ensemble**：Dart pred + iter_006 LGB pred 加权平均，sweep weight ∈ [0, 1]，搜 4D thresh DE。
5. WandB project: `r35_alt_arch`，run name `lgb_dart_h60_seed{s}_fold{f}`。

### 预期结果
- Dart 单独：LOSO sum ∈ [+10, +15]（同量级或略好）。
- Dart × gbdt 混合：**LOSO sum +14 ~ +16**（Δ +1 ~ +3 vs iter_006）。

### 风险
- Dart 训练时长 2-3× → 5 seed × 5 fold ≈ 4-8 h CPU（acceptable，可以一晚跑完）。
- Dart 不支持 `early_stopping`，需固定 num_iter（用 iter_006 best_iter ×1.5 做参考）。

---

## #2 — CatBoost h_60 5-seed（最高确定性 ROI）

### 为什么做
- **空白格**：iter_006 是 LGB h_60，CatBoost h_60 完整 5-seed **从未训过**（T17 只 h_10 + 单 seed）。
- Optiver 2023 1st 用 CatBoost (权重 0.5 in 3-model ensemble) → A 股 LOB 类似的 tabular setup 上 CB 与 LGB 等强。
- **Ordered boosting** 对 OOD（held-out sym）友好——每棵树估计不依赖 own-sample → 比 standard GBDT 减 leakage。
- 与 LGB 的 Histogram split 算法不同 → ensemble 多样性。

### 实施 plan
1. 复制 `experiments/T17_catboost/train_loso.py` → `experiments/T45_catboost_h60/`。
2. 改：
   ```python
   horizons = [60]
   seeds = [42, 7, 13, 21, 99]   # 与 iter_006 同 seed list
   params = dict(
       loss_function='MultiClass',
       depth=6,
       learning_rate=0.05,
       l2_leaf_reg=3.0,
       bagging_temperature=0.0,
       random_strength=1.0,
       border_count=128,
       boosting_type='Ordered',  # 关键：default Plain 改 Ordered
       task_type='GPU',          # GPU 训快 3×
       devices='0',
       iterations=2000,
       early_stopping_rounds=100,
       eval_metric='MultiClass',
   )
   ```
3. **WandB**：每 fold log 5 个 metrics（per-fold cum_pnl, n_active, accuracy, train_time, best_iter）。
4. 评估顺序：
   - (a) CatBoost 5-seed h_60 单独 LOSO sum；
   - (b) **2-model ensemble**：CB pred + iter_006 LGB pred 等权 → DE 4D thresh search；
   - (c) **3-model ensemble**：CB + LGB + Dart（如果 #1 通过）。
5. CRITICAL：训完用 CPU 预测脚本验证一遍（提交平台是 CPU）；CatBoost CPU/GPU 模型可互通 load。

### 预期结果
- CB 单 5-seed h_60 LOSO sum ∈ [+10, +15]。
- LGB + CB 等权 ensemble + 重 thresh：**+14 ~ +17**（Δ +1 ~ +4 vs iter_006）。

### 风险
- CatBoost-GPU wheel 安装：先 `python3 -c "import catboost; print(catboost.__version__)"` 验证；缺则 `pip install catboost --timeout 60`。
- GPU 显存：depth=6 × 226-d × 88k 单 fold 用 < 4 GB，应 OK。
- 推理端：要确认提交包内的 Predictor 可用 CPU load CatBoost model（实测可以，CB save 出来的 .cbm 文件 CPU 也能 predict）。

---

## #3 — Conformal / Venn-Abers calibration on iter_006

### 为什么做
- iter_006 的 4D thresh DE 隐含假设 LGB `predict_proba` 是 calibrated 的，**实际 GBDT 概率严重偏离 sigmoid**（Manokhin 2017 等多个文献）。
- T30 早期试过 conformal 但没广用——可能是 isotonic 在我们小 fold（88k）上 overfit。**Venn-Abers 双 isotonic 更稳**。
- 完全 stateless（calibration mapping 是预先 fit 好的 lookup function）→ 符合硬约束。
- 成本 < 1 h，**最差 = iter_006 不变**（zero downside）。

### 实施 plan
1. 新建 `experiments/T46_conformal_venn_abers/`。
2. 复用 iter_006 的 5-seed avg `predict_proba` OOF 输出（已存在 parquet）。
3. 对每个 (held_fold, class) 学 isotonic mapping：
   ```python
   from sklearn.isotonic import IsotonicRegression
   for k in [0, 1, 2]:  # 3 classes
       cal_p = lgb_oof_proba[cal_idx, k]
       cal_y = (y_cal == k).astype(int)
       iso_k = IsotonicRegression(out_of_bounds='clip').fit(cal_p, cal_y)
       calibrated[:, k] = iso_k.predict(test_proba[:, k])
       calibrated /= calibrated.sum(axis=1, keepdims=True)
   ```
4. **Venn-Abers**：用 `venn-abers` PyPI 包或自写 50 行实现（双 isotonic：upper + lower bound 平均）。
5. 用 calibrated proba **重新跑 4D thresh DE** 优化（计算很快）。
6. WandB: `r35_conformal_venn_abers`。

### 预期结果
- LOSO sum **+13.61 → +14 ~ +16**（Δ +0 ~ +3）。
- 主要 win：**降 false-positive trade rate**，提升 PnL/trade ratio；某些 fold（fold 1 一直 OOD 难）可能提升明显。

### 风险
- 如果 LGB proba 已经"够 calibrated"（reliability diagram 接近对角线），收益会很小 → 但成本也极小。
- Calibration set 划分要用 LOSO 内部的 inner-CV（不能用 val/test fold）→ 需要重新 split 训练数据 80/20 calibration。

---

## #4 — XGBoost h_60 5-seed + monotone constraints

### 为什么做
- **第三个 GBDT** → 三 GBDT ensemble (LGB + CB + XGB) 是 Optiver / Jane Street 类比赛的 winning recipe。
- T18 h_10 单 seed 只跑过且 fold 0 崩 → 加 monotone_constraints + 5 seed avg 后稳定性预期改善。
- XGBoost 2.0+ 的 GPU `device='cuda'` API 训练快。
- 与 CB 的 Ordered boosting 和 LGB 的 leaf-wise 都不同 → ensemble 多样性。

### 实施 plan
1. 复制 `experiments/T18_xgboost/train_loso.py` → `experiments/T47_xgb_h60_mono/`。
2. 改：
   - horizons = [60], seeds = [42, 7, 13, 21, 99]
   - params:
     ```python
     params = dict(
         tree_method='hist',
         device='cuda',
         objective='multi:softprob',
         num_class=3,
         max_depth=6,
         learning_rate=0.05,
         min_child_weight=100,
         subsample=0.8,
         colsample_bytree=0.8,
         reg_lambda=2.0,
         reg_alpha=0.0,
         monotone_constraints=mono_dict,  # 见下
     )
     ```
   - **Monotone constraints**：从 T39 feature importance 选 top 5 方向性特征（OFI, mid_price_diff, bid-ask_imbalance, spread_inv, depth_ratio），约束 `+1` 或 `-1`。其他特征 `0`（unconstrained）。
3. 评估：
   - (a) XGB 5-seed h_60 LOSO sum；
   - (b) **3-model ensemble** (LGB + CB + XGB) 等权 / DE-tuned weight + DE 4D thresh。
4. WandB: `r35_xgb_h60_mono`。

### 预期结果
- XGB 单 ≈ +8 ~ +12（XGB 在我们 setup 历史上略弱于 LGB/CB）。
- **3-GBDT ensemble**：**+14 ~ +18**（这是最有可能突破 +15 的组合）。

### 风险
- Monotone constraints 设错方向会反向 hurt → 先无 monotone 跑一遍 baseline，再叠加。
- XGBoost-GPU 在 cuda 环境下偶尔 OOM（226-d 特征 × 88k 行 × hist 应该没事）。

---

## #5 — MLPLOB（轻量 NN sanity check）

### 为什么做（最低优先级，只为彻底关上 NN 这扇门）
- DeepLOB / regularized NN / DeepLOB+aug 都失败 (T1/T19/T32)，**但 MLPLOB（纯 MLP-Mixer）从未试过**。
- MLPLOB 是 Berti 2025 的发现："最简单的 NN 反而是 SOTA"。
- 如果 MLPLOB 也不行 → **强证据**：我们 setup 不可能从 NN 拿额外 alpha，未来彻底放弃 NN 路线。
- 如果 MLPLOB 出奇能 + 1 ~ + 2 → 加进 ensemble 作多样性来源。

### 实施 plan
1. 新建 `experiments/T48_mlplob_h60/`。
2. **架构**（MLPLOB simple，从 [TLOB 仓库](https://github.com/LeonardoBerti00/TLOB) 复刻 ~80 行）：
   ```python
   class MLPLOB(nn.Module):
       def __init__(self, n_feat=226, hidden=256, depth=4, dropout=0.4):
           super().__init__()
           layers = []
           dims = [n_feat] + [hidden]*depth
           for i in range(depth):
               layers += [
                   nn.Linear(dims[i], dims[i+1]),
                   nn.LayerNorm(dims[i+1]),
                   nn.GELU(),
                   nn.Dropout(dropout),
               ]
           self.encoder = nn.Sequential(*layers)
           self.head = nn.Linear(hidden, 3)
       def forward(self, x):  # x = (B, 226)
           return self.head(self.encoder(x))
   ```
3. **训练 setup**：
   - 输入：226-d Scheme C feature（不是 100-tick raw → 与 GBDT 同输入，对照才公平）。
   - Optimizer: AdamW lr=1e-3, wd=1e-2, cosine LR over 30 epoch。
   - **Aug**：保留 T32 用过的 aug_a（z-score space gaussian noise σ=0.05）。
   - **Label smoothing 0.05** + cross-entropy。
   - Early stop on **val test_held-out fold cum_pnl**（不是 val accuracy）。
4. 5-fold LOSO，3 seed (节省 GPU 时间，不到 5 seed)。
5. **关键 sanity check**：在 fold 0 跑通后，**先看 val→test gap**。如果仍 > 5 cum_pnl → 立即停（之前的 NN 锅复现），不浪费 GPU。
6. WandB: `r35_mlplob_h60`。

### 预期结果
- 70 % 概率：LOSO sum ∈ [−5, +5]（弱于 iter_006，最终 ensemble 也不加分）。
- 30 % 概率：LOSO sum ∈ [+8, +13] 且与 LGB 不相关 → 加进 ensemble 提 +1 ~ +2。

### 风险 / 退出条件
- fold 0 val→test gap > 5 → **停止**，记录 "NN 路线确认死路" 到 PROGRESS.md。
- GPU 时间不超过 4 h × 1 卡，超时即停。

---

## 综合执行顺序与时间预算

| 顺序 | 任务 | 预算 | 依赖 |
|---|---|---|---|
| Week 1 Day 1 | #1 LGB Dart h_60 5-seed | 4-8 h CPU + 1 h analysis | 无 |
| Week 1 Day 2-3 | #2 CatBoost h_60 5-seed | 2-4 h GPU + 1 h analysis | 无 |
| Week 1 Day 3 | #3 Conformal/Venn-Abers on iter_006 | < 1 h | iter_006 OOF（已有）|
| Week 1 Day 4 | #4 XGBoost h_60 5-seed + mono | 1-2 h GPU + 1 h analysis | #2 完成（拿到 mono direction 思路）|
| Week 1 Day 5 | **3-GBDT ensemble + conformal**：LGB+CB+XGB → conformal → DE 4D thresh | 2-3 h | #1, #2, #3, #4 都完 |
| Week 2 Day 1 | #5 MLPLOB sanity check | ≤ 4 h GPU | 独立 |

**核心赌注**：3-GBDT ensemble + conformal calibration → **LOSO sum +15 ~ +18** (vs iter_006 +13.61 → Δ +1.4 ~ +4.4)。
**保守预期**：至少 #1 或 #2 + #3 conformal 能拿到 +1 ~ +2 的 robust 提升。

---

## 不在 Top 5 的理由速查

| 排除 | 主因 |
|---|---|
| TabPFN v2/2.5 | 行数超限（10k/50k vs 我们 1.47M），inference 太慢 |
| FT-Transformer / SAINT | NN 路线已 3 次失败，FT-T 单模型期望 < LGB |
| Mamba for tabular | feature 顺序无意义，需要重做 100-tick 序列 pipeline |
| LGB+MLP gating | T32 已证 NN gate weight = 0 是最优 |
| LOBster pre-train + fine-tune | Domain shift 巨大 + 100+ GPU-day 不可复现 |
| Multi-task NN | 依赖一个能跑通的 NN backbone，前提不满足 |
| NGBoost / Soft Trees | calibration 收益由 #3 conformal 替代 |

---

## Sources

- [TLOB / MLPLOB GitHub](https://github.com/LeonardoBerti00/TLOB)
- [TLOB / MLPLOB paper (arxiv 2502.15757)](https://arxiv.org/abs/2502.15757)
- [LightGBM Dart docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost Ordered boosting](https://catboost.ai/en/docs/concepts/algorithm-main-stages_boosting)
- [Venn-Abers calibration tutorial (Manokhin)](https://valeman.medium.com/how-to-calibrate-your-classifier-in-an-intelligent-way-a996a2faf718)
- [Conformal prediction in finance (Sciencedirect 2025)](https://www.sciencedirect.com/science/article/pii/S266654682500103X)
