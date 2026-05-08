# R40 — Model Architecture / Decision Rule 颠覆但 basic 的 idea 清单

> **Date**: 2026-05-07
> **Worker**: R40 brainstorm (opus xhigh)
> **Status**: research-only（不写训练代码）
> **背景**: iter_013 (LightGBM 5-seed **regression on Δmid_norm** + EV-gate) = LOSO-equiv +36.23 / 平台 +19.23 (h_60)。
>           T80 透传系数 0.70：每 +1 LOSO ≈ +0.70 平台。
>           **关键结构性遗漏**：iter_013 把 "regression target" 这条 lever 拉满了，但**只在 LightGBM 上做**；**所有其他模型类型（NN / CatBoost / XGBoost / LGBM-Other-Objective）都还在用 3-class CE 或者根本没试**。
>
> **TL;DR — Top 5 (每条 ≤ 4h，预期 +1.0+ LOSO，单 lever)**：
>
> | Rank | Idea | 实施成本 | 预期 LOSO Δ | 颠覆性 |
> |---|---|---|---|---|
> | 🥇 | **#1 MLP regression on Δmid_norm + EV-gate** | 3-4h | **+1.0 ~ +5.0** | NN 路线**4 次都用 3-class CE**；regression target 在 NN 上从未试，重大盲区 |
> | 🥈 | **#2 CatBoost regression on Δmid_norm + EV-gate** | 2.5h | **+0.5 ~ +2.5** | T46 CatBoost classification = +12.64（低于 LGB CE +13.6），但 **CatBoost regression 没试**；Optiver 2023 1st 配方 |
> | 🥉 | **#3 XGBoost regression on Δmid_norm + EV-gate** | 2h | **+0.3 ~ +1.5** | T18 XGB classification 弱；regression 没试；GBDT 多样性补 LGBM |
> | 4️⃣ | **#4 LightGBM Quantile regression (5 quantile heads)** | 3h | **+0.5 ~ +2.0** | iter_013 用 L2 单点估计；quantile q=0.10/0.25/0.50/0.75/0.90 把 EV-gate 升级为"P(Δmid > fee | predicted distribution) > τ" → 决策直接含 uncertainty |
> | 5️⃣ | **#5 Volatility-conditional EV thresholds** (no retrain, decision-rule only) | 1h | **+0.3 ~ +1.0** | iter_013 用 1 对全局 (thr_up=3.7e-4, thr_dn=1.6e-4)；high-vol bucket 提阈值、low-vol bucket 降阈值，state-less 兼容 |

---

## 0. 评分矩阵（颠覆性 vs 实施成本）

```
                  ↑ 颠覆性 (与现有 stack 差异度)
              ★★★★★  │   #1 MLP-regr     #6 1D-CNN-regr     #12 LGBM-PnL-obj
              ★★★★   │   #4 quantile     #2 CB-regr         #11 mixture-density
              ★★★    │   #3 XGB-regr     #7 transformer-regr #14 FT-T-regr
              ★★     │   #5 vol-cond     #8 LGBM-Huber       #15 ensemble L1+L2+Huber
              ★      │   #16 decision-tweaks
              -------+----------------------------- → 简易度（成本反向）
                     <2h    2-4h    4-8h    >8h
```

---

## A. NN with Regression Target（**最大盲区，重头戏 — 6 个变体**）

**核心论点**：
- T1 / T19 / T32 (DeepLOB) / T60 (direct PnL NN) **全部 4 次失败**——但它们**都用 3-class CE 或者直接最大化 expected PnL**。
- T75 在 LightGBM 上把 target 从 3-class CE 换成 **regression on Δmid_norm + EV-gate** → +9.79 LOSO（+37%）。
- **完全未试的组合 = NN backbone + regression target on Δmid_norm + EV-gate**。
- T60 失败模式：directly maximize expected PnL，loss landscape 极不平滑，5 fold variance 巨大（-6.49 ~ +12.87）。**Regression L2/L1/Huber 的 loss landscape 是 convex/smooth**，应该完全不同。

### #1 Simple MLP regression on Δmid_norm + EV-gate ⭐ TOP-1

**架构**（3-4 layer，no residual，~150 k params）：
```python
# input: 359-d Stage 5 features (与 iter_013 完全一致)
# target: y = (mp_t60 - mp_t) / (mp_t + 1)   # 与 iter_013 完全一致
class MLPRegressor(nn.Module):
    def __init__(self, in_dim=359, hidden=256, depth=4, dropout=0.3):
        layers = []
        d = in_dim
        for i in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(hidden, 1)]    # single-scalar regression head
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# loss = SmoothL1Loss (Huber) — robust to tail outliers; default β=0.0001
# optimizer: AdamW lr=1e-3 weight_decay=0.02
# scheduler: CosineAnnealing 20 epochs
# aug: per-feat scale [0.85, 1.15] (与 iter_013 同样的 aug_a)
# val: V4 walk-forward (date 0-75 train / 76-79 val) 同 iter_013
# decision rule: 与 iter_013 完全一致 (thr_up=3.7e-4, thr_dn=1.6e-4) 或重做 DE
```

**为何颠覆**：
1. **NN 在 359-d Stage 5 上从未跑过 regression**——T1/T19/T32 都是 100-tick raw window，T60 是 direct PnL loss。
2. **Loss landscape 完全不同**：T60 的 -E[reward] 是非凸的；Huber regression 是 convex，5-seed ensemble 方差应远小于 T60。
3. **NN 学到的 nonlinearity ≠ GBDT 切分**——LGBM 在切分点之间是 piecewise-constant，NN 是 smooth → 在 Δmid 接近 fee 边界（决策最敏感区）的 ŷ 估计更准。
4. **直接 ensemble LGBM-regr + NN-regr 平均 ŷ** → 5-seed × 2 model = 10-base ensemble，与现有 4D DE 完全兼容。

**预期 LOSO Δ**：单 NN-regr LOSO ≈ +30 ~ +38（接近 iter_013），**ensemble (LGBM-regr + NN-regr) → +37 ~ +41**（+1 ~ +5 vs iter_013）

**成本**：3-4h（pilot 1 seed × 1 fold，~30 min on RTX 4090；full 5-seed × 5-fold ~2.5h）

**契约**：✅ sym-agnostic（无 sym embedding）、✅ stateless inference（forward(X)→scalar）、✅ batch-vec、✅ 不依赖 date

**风险**：NN 在 OOD 跨 sym 上比 GBDT 弱是历史经验。**缓解**：
- 训 ensemble 队友角色（α=0.3 weight），不替换 LGBM-regr
- 用 SWA (Stochastic Weight Averaging) 平滑 minimum
- aug_a 范围与 iter_013 同（不调宽）

---

### #6 1D-CNN regression on raw 100-tick window + EV-gate ⭐

**架构**：
```python
# input: (B, 100, K)  where K ≈ 50 raw LOB features (price/size 各 10 档 + 时间编码)
# 不用 Stage 5 features（避免 GBDT 的 hand-crafted 已饱和），直接喂 raw window
class CNNRegressor(nn.Module):
    def __init__(self, in_ch=50, conv_chs=[64, 128, 256], kernel=5):
        # 3 个 1D conv block + GAP + MLP head
        ...
        self.head = nn.Linear(256, 1)   # regression scalar
```

**为何颠覆**：
- T1 (DeepLOB CNN classification) 失败，**但 raw window + regression target 没试过**。
- 1D-CNN 学的是局部时序 pattern（OFI burst、price drift），**与 GBDT 的全局静态特征互补**。
- regression target 比 3-class CE 携带 |Δp| 量级信息。

**预期 LOSO Δ**：单模型 0 ~ +5（OOD 风险大），**与 LGBM-regr ensemble → +1 ~ +3**

**成本**：4-6h（含 raw window cache 重做）

**风险**：DeepLOB-class 架构 OOD 跨 sym 历史不稳；优先做 #1 再考虑 #6。

---

### #7 Transformer encoder regression on 100-tick window + EV-gate

**架构**：单层 Transformer encoder（4 head, d=128），无 positional encoding（用 relative bias），头是 mean-pool + linear → scalar。

**为何颠覆**：TLOB / MLPLOB Berti 2025 显示纯 MLP > Transformer，但**没在 regression target 上对比**。
**预期 LOSO Δ**：0 ~ +3（与 #6 相似）
**成本**：6h
**优先级**：低于 #1/#6；先做 MLP，再 1D-CNN，最后 Transformer。

---

### #14 FT-Transformer regression on 359-d features + EV-gate

**架构**：feature tokenizer + 3-layer transformer + CLS token + linear head。
**为何颠覆**：tabular DL SOTA 之一，但**从未跑 regression on Δmid**。
**预期 LOSO Δ**：0 ~ +2
**成本**：5h
**优先级**：低（与 #1 相似输入但更复杂；先看 #1 表现再决定）

---

### #11 Mixture Density Network (MDN) regression + probabilistic EV-gate ⭐

**架构**：NN 输出 (μ, σ) 或 K=3 component mixture (π_k, μ_k, σ_k)。
```python
# loss: -log p(y | μ, σ) under mixture-of-Gaussians
# decision: trade if P(Δmid > +fee | predicted N(μ, σ)) > τ
#   = trade if μ - z·σ > fee, where z = quantile_threshold
```
**为何颠覆**：iter_013 是 point estimate ŷ，**没有 uncertainty quantification**；MDN 让 EV-gate 变成 confidence-aware（高 σ → 不交易）。
**预期 LOSO Δ**：+0.5 ~ +2.5（自动屏蔽低 confidence trade）
**成本**：4h
**契约**：✅
**优先级**：与 #4 quantile 重叠，二选一即可。

---

### #19 Multi-head NN: regression main + classification aux ⭐ (NN 走通后的加强)

**架构**：共享 backbone → 双 head：
- main: regression on Δmid_norm (loss = Huber × 1.0)
- aux: 3-class CE (loss × 0.2，作为 regularizer)

**为何颠覆**：multi-task learning 在 Optiver 2023 1st 验证有效（HYD 用 5 个 horizon aux head）；T1/T19 单任务 NN 失败，可能 aux task 帮助 representation。
**预期 LOSO Δ**：+0.3 ~ +1（仅在 #1 走通后值得做）
**成本**：4h（基于 #1 的代码扩展）

---

## B. Hybrid LGBM + NN Stack（3 个变体）

### #20 LGBM-regr base + NN residual correction ⭐

**思路**：
1. LGBM 5-seed regression 预测 ŷ_lgb
2. NN 训练 target = `Δmid_norm - ŷ_lgb` (residual)
3. 最终 ŷ_final = ŷ_lgb + ŷ_nn_residual
4. EV-gate on ŷ_final

**为何颠覆**：
- iter_013 LGBM-regr 已捕捉大部分线性 + 树状 pattern → residual 是 GBDT 学不到的 nonlinear 部分。
- NN 不需要从零学，只需 fine-tune residual → loss landscape 简化，过拟合风险低。
- R31 paper P14 (Hybrid VAR + FNN) 验证 linear backbone + NN-on-residual 配方有效。

**预期 LOSO Δ**：+0.5 ~ +2.5（GBDT 学剩下的线性 + 树切分；NN 学剩下的曲面）
**成本**：5h（含 residual cache + NN 训练）
**契约**：✅（推理时 ŷ_lgb + ŷ_nn 都是 stateless scalar）

---

### #21 NN feature embedding → LGBM stage-2

**思路**：NN 训练 backbone（regression target），抽中间层 embedding 32-dim；把 embedding 当 32 个新 feature 喂给 LGBM 5-seed，再 EV-gate。
**为何颠覆**：NN backbone 学非线性表征，LGBM 决策；典型 stacking。
**预期 LOSO Δ**：+0.3 ~ +1.5
**成本**：6h
**风险**：NN backbone 不稳 → embedding 不稳；先 #20 后 #21。

---

### #22 LGBM-regr + NN-regr 简单加权 ensemble (no residual)

**思路**：训独立 NN-regr (#1)，再 ŷ_final = α·ŷ_lgb + (1-α)·ŷ_nn，α DE-tune。
**为何颠覆**：最简版 #20，作为 #1 的"消费方式"。
**预期 LOSO Δ**：与 #1 单独相同（+1 ~ +5）
**成本**：仅 0.5h（在 #1 之上）
**优先级**：与 #1 一起做，零额外成本。

---

## C. CatBoost / XGBoost Regression 重做（**填遗漏**）

### #2 CatBoost regression on Δmid_norm + EV-gate ⭐ TOP-2

**思路**：
- T46 CatBoost classification = +12.64（< iter_006 +13.6）
- T17 早期单 seed 同样 classification
- **CatBoost regression on Δmid_norm 完全没试**

**为何颠覆**：
- **Optiver 2023 1st 配方就是 CatBoost regression**（HYD MAE on continuous return）。
- CatBoost ordered boosting → OOD 跨 sym 比 LGBM 更稳（每棵树用 OOF residual，target leakage 减少）。
- 与 LGBM-regr ensemble 多样性高（不同 split 策略）。

**实施**：
```python
from catboost import CatBoostRegressor
m = CatBoostRegressor(
    iterations=2000, learning_rate=0.05,
    loss_function='RMSE',  # 或 'MAE'，'Huber:delta=0.0001'
    task_type='GPU', devices='0',
    bootstrap_type='Bayesian', bagging_temperature=1.0,
    random_seed=seed,
    early_stopping_rounds=40,
)
m.fit(X_train, y_regr, eval_set=(X_val, y_val_regr), use_best_model=True)
# inference: m.predict(X) → scalar; ensemble = mean over 5 seeds
# 与 LGBM-regr ensemble: ŷ = 0.5·ŷ_lgb + 0.5·ŷ_cb，再 EV-gate
```

**预期 LOSO Δ**：
- CatBoost-regr 单独 ≈ iter_013 ± 2
- **(LGBM-regr + CatBoost-regr) ensemble → +0.5 ~ +2.5**

**成本**：2.5h（CatBoost-GPU h_60 ≈ 5-10 min/seed/fold × 5 × 5 = 2-4 h；GPU 加速）

**契约**：✅（CatBoost CPU 推理 < 5 ms/window，提交端没问题）

---

### #3 XGBoost regression on Δmid_norm + EV-gate ⭐ TOP-3

**思路**：T18 XGB classification 弱；regression 同样没试。
**为何颠覆**：
- XGBoost depth-wise split + GPU 'gpu_hist' 快速训练
- 可加 `monotone_constraints` 强制 OFI 单调正、spread 单调负
- GBDT 多样性补 LGBM/CatBoost

**实施**：
```python
import xgboost as xgb
params = dict(
    objective='reg:squarederror',  # 或 reg:pseudohubererror
    tree_method='gpu_hist', device='cuda',
    learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints={'OFI_w20': 1, 'spread1': -1, ...},  # 可选
)
booster = xgb.train(params, dtrain, num_boost_round=2000,
                    evals=[(dval, 'val')], early_stopping_rounds=40)
```

**预期 LOSO Δ**：
- XGB-regr 单独 ≈ iter_013 - 1 ~ +1
- **(LGB + CB + XGB) tri-GBDT regression ensemble → +1.0 ~ +3.0**

**成本**：2h（XGB GPU 训快）

**契约**：✅

---

### #23 LGBM-regr + CatBoost-regr + XGBoost-regr 三 GBDT regression 加权 ensemble ⭐

**思路**：每个 GBDT 5-seed × 5-fold = 25 booster；3 类共 75 booster；DE-tune 3 weight 比例。
**为何颠覆**：完整 GBDT diversity stack，**全部 regression target**（vs 历史的 classification ensemble 失败）。
**预期 LOSO Δ**：+1.0 ~ +3.5（diversity 在 regression target 上比 classification 上更可能凑效，因为 ŷ 是连续的 → 平均更平滑）
**成本**：5-6h（含三 model 训练 + DE）
**优先级**：在 #2 #3 单独验证 ≥ +0.3 后再做。

---

## D. LightGBM 其他 Regression Objective

### #4 LightGBM Quantile regression (5-quantile head) + distribution-aware EV-gate ⭐ TOP-4

**思路**：
- 训 5 个 booster，每个 objective='quantile' alpha={0.10, 0.25, 0.50, 0.75, 0.90}
- 推理时得 (q10, q25, q50, q75, q90) 5 个 quantile 估计
- decision rule:
  ```python
  # 估计 P(Δmid > +fee) ≈ fraction of quantiles > +fee
  p_up = sum(q > +fee for q in [q10, q25, q50, q75, q90]) / 5
  p_dn = sum(q < -fee for q in [q10, q25, q50, q75, q90]) / 5
  if p_up >= τ_up: action = 2 (long)
  elif p_dn >= τ_dn: action = 0 (short)
  else: action = 1
  ```

**为何颠覆**：
- iter_013 是 point estimate ŷ → **EV-gate 不含 uncertainty**；同样 ŷ=4e-4 但 σ=2e-4 vs σ=1e-3 应不同决策
- Quantile regression 直接给 distribution → confidence-aware decision
- LightGBM 原生支持 `objective='quantile'`，5 booster × 5 seed = 25 booster，可全部 GPU 训
- R12 / R13 报告里提到但**没人执行 quantile regression**

**预期 LOSO Δ**：+0.5 ~ +2.0（高 σ trade 自动 abstain）
**成本**：3h（5 quantile × 5 seed = 25 boosters，GPU 并行 ~1.5h + decision rule + DE-tune τ_up, τ_dn）
**契约**：✅
**等价**：与 #11 MDN 的 NN 版功能等价；选简单的 LGBM-quantile 先试。

---

### #8 LightGBM Huber regression (replace L2 with Huber)

**思路**：iter_013 用 `objective='regression_l2'`；改成 `regression_huber` (含 δ 参数)。
**为何颠覆**：
- L2 对大 |Δp| 异常值高度敏感（HFT 偶有 limit-up/down）
- Huber 在小残差用 L2，大残差用 L1 → 鲁棒
- **完全 drop-in**：1 行 params 改动

**预期 LOSO Δ**：+0.2 ~ +0.8（少量 +1 ~ +2 噪声样本不再扭曲 ŷ）
**成本**：1h（重训 5-seed）
**契约**：✅
**优先级**：极低成本，必做的"消停" lever。

---

### #9 LightGBM L1 (MAE) regression

**思路**：`objective='regression_l1'`。
**为何颠覆**：HYD Optiver 1st 用 MAE。
**预期 LOSO Δ**：+0.1 ~ +0.5（与 Huber 重叠较多，但单独 ablate 可能略好）
**成本**：1h
**优先级**：与 #8 一起跑。

---

### #15 Multi-objective ensemble: L1 + L2 + Huber LGBM

**思路**：3 个 LGBM-regr ensemble，每个 5-seed × 5-fold = 15 model；不同 loss 误差结构互补。
**预期 LOSO Δ**：+0.3 ~ +1.0（比单一 objective 好）
**成本**：3h（在 #8 #9 已训之后零成本）
**契约**：✅

---

### #10 LightGBM Tweedie regression (zero-inflated)

**思路**：`objective='tweedie'` variance_power ∈ [1.1, 1.9]；处理 Δmid 高度集中在 0 周围的零通胀分布。
**为何颠覆**：Δmid_60 分布 ~50% 在 |x|<1e-4 内，是典型 zero-inflated；Tweedie 是这类 GLM 的经典处理。
**预期 LOSO Δ**：0 ~ +1（理论 fit 但 LightGBM Tweedie 实测有时漂移）
**成本**：1.5h
**优先级**：在 #8 #9 之后。

---

## E. Decision Rule 颠覆（不重训，仅改推理决策）

### #5 Volatility-conditional EV thresholds ⭐ TOP-5

**思路**：iter_013 用全局 (thr_up=3.7e-4, thr_dn=1.6e-4)；改成基于 last-window vol 的动态阈值：

```python
# stateless: vol 来自当前 100-tick window 内的统计
sigma_t = std(mid_diff over window[-20:])    # 20-tick rolling vol
sigma_baseline = 0.0002    # 全局基准（trained constant）
vol_ratio = sigma_t / sigma_baseline    # 高 vol → ratio > 1
thr_up_t = 3.7e-4 * vol_ratio**alpha    # alpha ~ 0.3-0.7 DE-tune
thr_dn_t = 1.6e-4 * vol_ratio**alpha
```

**为何颠覆**：
- High-vol 时 noise 大 → 提高阈值（避免乱 trade）
- Low-vol 时 signal-to-noise 好 → 降阈值（捕获更多 marginal trade）
- **stateless**（vol 由 window 算出，不跨 call）
- iter_013 完全没用动态阈值
- R31 paper P23 (Volatility-Conditional MoE) 验证此思路

**预期 LOSO Δ**：+0.3 ~ +1.0
**成本**：1h（仅改 Predictor.py，再 DE-tune α）
**契约**：✅
**风险**：DE α 可能 overfit；用 V4 walk-forward val DE，再 442k test ablate。

---

### #16 Asymmetric Bayesian thresh with prior

**思路**：在 EV-gate 加先验偏置。`pred_normalized = ŷ - μ_prior`，μ_prior 从 train 集均值算（应为 ~0 但实测 iter_013 在 test 上有 +4e-5 ~ +9e-5 bias，T75 报告）。
**为何颠覆**：iter_013 DE-asymm thresh 已经隐式校了 bias；显式去 bias 后 DE 收敛更稳，可能 overfit 减小。
**预期 LOSO Δ**：+0.1 ~ +0.5
**成本**：30min
**契约**：✅
**优先级**：与 #5 互补，可一起做。

---

### #17 Confidence × magnitude joint gating

**思路**：用 quantile regression (#4) 算 |ŷ| / σ̂ 作为 confidence；联合 |ŷ| 排序决策：
```python
score = sign(ŷ) * |ŷ| * f(|ŷ|/σ̂)    # f 是 sigmoid
trade if |score| > threshold
```
**为何颠覆**：iter_013 EV-gate 单独看 |ŷ|，不看 confidence；联合度量更稳。
**预期 LOSO Δ**：与 #4 重叠（实质上是 #4 的 decision rule 形式）
**成本**：1h（基于 #4 的 quantile 之上）
**优先级**：与 #4 配套。

---

### #18 Per-feature-percentile-based vol bucket（无需统计 train baseline）

**思路**：从 100-tick window 内**自己**算 vol，不依赖 train 集 baseline。直接用 `vol_rank = (current_vol - min_window_vol) / (max - min)`，归一到 [0, 1]。
**为何颠覆**：完全 sym-agnostic + stateless + 无 hyperparam（baseline = 0 自动学）。
**预期 LOSO Δ**：与 #5 类似（+0.3 ~ +0.8）
**成本**：1h
**优先级**：#5 的"无超参"版本，二选一。

---

## F. Direct PnL Gradient Training (重做 T60 但用 LightGBM)

### #12 LightGBM with custom PnL objective ⭐

**思路**：T60 NN with -E[reward] loss 失败（loss landscape 非凸 + NN OOD）；**LightGBM 自定义 objective 没试过**。

LightGBM 接受 `objective=callable` 返回 (grad, hess)。设计 differentiable PnL approximation：

```python
def pnl_objective(preds, train_data):
    """
    preds: (N,) regression output (proxy for Δmid)
    label: y_regr = Δmid_norm
    Smooth approximation of EV-gate PnL:
      pnl(ŷ, y) = (sigmoid((ŷ-thr_up)/τ) - sigmoid((-ŷ-thr_dn)/τ)) * y - fee * |action|
    grad/hess = autograd analytical (closed form possible due to sigmoid + linear y)
    """
    ŷ = preds
    y = train_data.label
    thr_up, thr_dn = 3.7e-4, 1.6e-4
    fee = 1e-4
    τ = 1e-4   # smoothness
    p_long = 1 / (1 + np.exp(-(ŷ - thr_up) / τ))
    p_short = 1 / (1 + np.exp(-(-ŷ - thr_dn) / τ))
    action_signal = p_long - p_short
    # PnL approx = action_signal * y - fee * (p_long + p_short)
    # gradient w.r.t. ŷ: ∂PnL/∂ŷ = (∂p_long/∂ŷ - ∂p_short/∂ŷ) * y - fee * (∂p_long/∂ŷ + ∂p_short/∂ŷ)
    # gradient lift: GBDT MINIMIZES, so return -∂PnL/∂ŷ as grad
    grad = -(p_long*(1-p_long) - p_short*(1-p_short)) * y / τ
    grad += fee * (p_long*(1-p_long) + p_short*(1-p_short)) / τ
    hess = ...   # diagonal 2nd derivative
    return grad, hess
```

**为何颠覆**：
- LightGBM custom obj 比 NN 训练稳定（GBDT 自带正则）
- T60 失败是 NN 不稳，**objective formulation 本身可能是正确的**
- 完全 differentiable
- R10 报告里列了 S2 (LGBM custom multiclass PnL) 但未执行；这里做 regression 版

**预期 LOSO Δ**：+0.5 ~ +2.5（如果 grad/hess 推得对）
**成本**：4h（含 grad 推导 + numerical check + warm-start from L2 50 round）
**契约**：✅
**风险**：自定义 obj 可能 overfit smoothness τ；用 V4 val 严格 early-stop。

---

### #13 LightGBM custom multiclass PnL objective (已在 R10 列出)

**思路**：R10 报告里 S2，softmax + per-class reward + entropy reg。
**预期**：与 #12 重叠，但 multiclass output；3-class softmax 形式更接近原始 iter_002 而非 iter_013。
**优先级**：低于 #12（regression formulation 更适合现有 EV-gate 习惯）。

---

## G. Stacking / Meta-learner（NEW iter_013 + 老 horizon model 没试）

### #24 Ridge meta over (iter_013-regr) + (iter_002 multi-horizon prob) ⭐

**思路**：
- iter_013 是 regression scalar (442k,)
- iter_002 5 个 horizon × 3 class prob (442k, 15)
- 共 16-d feature → Ridge meta-learn → final score → threshold

**为何颠覆**：
- T47 multi-horizon stacking 失败用的是 iter_002 内部 5 horizon CE prob（多个 CE 互相重复）
- **NEW combo: iter_013 regression scalar + iter_002 5-horizon CE prob** 互补性强：
  - regression 给"未来 +60 tick 的 Δmid 估计"
  - 5-horizon prob 给"短期 momentum 在 5/10/20/40/60 tick 上的 sign 一致性"
  - 二者 view 不重叠

**实施**：
```python
# OOF:
# X_meta = [ŷ_lgb_regr, p0_h5, p1_h5, p2_h5, p0_h10, ..., p2_h60]   # 16-d
# y = Δmid_norm
# from sklearn.linear_model import Ridge
# meta = Ridge(alpha=1.0).fit(X_meta_oof, y_oof)
# final ŷ = meta.predict(X_meta_test) → EV-gate
```

**预期 LOSO Δ**：+0.3 ~ +1.5
**成本**：3h（OOF prep + Ridge fit + DE）
**契约**：✅（Ridge weight 是常量矩阵）
**关键差异**：T47 用了**复杂 meta-learner**（小 NN 或 LightGBM）；Ridge 简单且 L2 reg 防 overfit。

---

### #25 Stacked NN meta on combined predictions

**思路**：同 #24 但 meta-learner 是 2-layer MLP。
**为何颠覆**：捕捉 nonlinear 交互（如 "h_5 prob_up high AND ŷ_regr > 0 → 双重确认"）。
**预期 LOSO Δ**：+0.4 ~ +1.8（vs Ridge）
**成本**：4h
**风险**：meta NN 可能 overfit OOF；先 Ridge 再 NN。

---

## H. 评分总表

| Rank | Idea | 颠覆性 | 成本 | 预期 LOSO Δ | Top 5? |
|---|---|---|---|---|---|
| 1 | #1 MLP regression on Δmid_norm + EV-gate | ★★★★★ | 3-4h | +1.0 ~ +5.0 | ✅ |
| 2 | #2 CatBoost regression + EV-gate | ★★★★ | 2.5h | +0.5 ~ +2.5 | ✅ |
| 3 | #3 XGBoost regression + EV-gate | ★★★ | 2h | +0.3 ~ +1.5 | ✅ |
| 4 | #4 LightGBM Quantile regression (5-q) + dist EV | ★★★★ | 3h | +0.5 ~ +2.0 | ✅ |
| 5 | #5 Volatility-conditional EV thresholds | ★★★ | 1h | +0.3 ~ +1.0 | ✅ |
| 6 | #6 1D-CNN regression on raw window | ★★★★★ | 4-6h | 0 ~ +5 (单)，+1~+3 ens | — |
| 7 | #7 Transformer encoder regression on raw window | ★★★ | 6h | 0 ~ +3 | — |
| 8 | #8 LightGBM Huber regression | ★★ | 1h | +0.2 ~ +0.8 | — |
| 9 | #9 LightGBM L1 (MAE) regression | ★★ | 1h | +0.1 ~ +0.5 | — |
| 10 | #10 LightGBM Tweedie regression | ★★ | 1.5h | 0 ~ +1.0 | — |
| 11 | #11 Mixture Density Network + prob-EV | ★★★★ | 4h | +0.5 ~ +2.5 | — |
| 12 | #12 LightGBM custom PnL objective (regression) | ★★★★★ | 4h | +0.5 ~ +2.5 | — |
| 13 | #13 LGBM custom PnL multiclass | ★★★ | 4h | (overlap with #12) | — |
| 14 | #14 FT-Transformer regression on 359-d | ★★★ | 5h | 0 ~ +2 | — |
| 15 | #15 Multi-objective ensemble L1+L2+Huber LGBM | ★★ | 3h | +0.3 ~ +1.0 | — |
| 16 | #16 Asymmetric Bayesian thresh with prior | ★ | 0.5h | +0.1 ~ +0.5 | — |
| 17 | #17 Confidence × magnitude joint gating | ★★ | 1h | (overlap with #4) | — |
| 18 | #18 Vol-bucket from window (no baseline) | ★★ | 1h | (overlap with #5) | — |
| 19 | #19 Multi-head NN regr+CE aux | ★★★ | 4h | +0.3 ~ +1 | — |
| 20 | #20 LGBM-regr + NN residual correction | ★★★★ | 5h | +0.5 ~ +2.5 | — |
| 21 | #21 NN embedding → LGBM stage-2 | ★★★ | 6h | +0.3 ~ +1.5 | — |
| 22 | #22 LGBM-regr + NN-regr 简单加权 ens | ★★ | 0.5h (after #1) | (= #1) | — |
| 23 | #23 LGB+CB+XGB tri-GBDT regression ens | ★★★ | 5-6h | +1.0 ~ +3.5 | — |
| 24 | #24 Ridge meta over iter_013 + iter_002 multi-h | ★★★ | 3h | +0.3 ~ +1.5 | — |
| 25 | #25 NN meta on combined predictions | ★★★ | 4h | +0.4 ~ +1.8 | — |

总计 **25 个 idea**（远超 12 要求）。

---

## I. Top 5 实施序列（≤ 4h each）

```
Step 1 [1h]:    #5 Volatility-conditional EV thresholds (decision rule, no retrain — 最 fast)
Step 2 [2h]:    #3 XGBoost regression on Δmid_norm + ensemble 与 LGBM-regr
Step 3 [2.5h]:  #2 CatBoost regression on Δmid_norm + tri-GBDT ensemble
Step 4 [3h]:    #4 LightGBM Quantile regression (5-q heads) + distribution-aware EV-gate
Step 5 [3-4h]:  #1 MLP regression on Δmid_norm + ensemble (LGBM-regr + MLP-regr)
```

**预期累计**：+1 ~ +5 LOSO（最低 ~+1，乐观 ~+5）
**Acceptance bar**：单 lever 必须 > +0.3 LOSO 才进 iter_014（DE noise floor）

---

## J. 关键 takeaway / 颠覆总结

1. **"NN 路线 4 次失败"是被 3-class CE 误导的判定** — NN regression on Δmid_norm + EV-gate **从未试过**，是当前最大盲区。MLP-regr (#1) 期望 +1 ~ +5 LOSO 单 lever。

2. **"CatBoost / XGBoost classification 已试" 不等于"regression 已试"** — Optiver 2023 1st 用的是 **CatBoost regression on continuous return**，我们 T46 / T18 都是 classification。CatBoost-regr (#2) 期望 +0.5 ~ +2.5。

3. **iter_013 是 LightGBM L2 一种 objective; LightGBM 还有 7 种其他 objective** — Quantile (#4)、Huber (#8)、L1 (#9)、Tweedie (#10) 全没试，加 5-q + distribution-aware EV-gate (#4) 期望 +0.5 ~ +2.0。

4. **Decision rule 在 iter_013 是 "1 对全局 thr_up/thr_dn"** — Vol-conditional (#5) 是最 cheap (1h) lever，期望 +0.3 ~ +1.0；不需重训。

5. **T60 "direct PnL NN" 的失败 = NN 不稳，不是 objective 错** — 同 objective 在 LightGBM custom obj 下 (#12) 应稳；GBDT 自带正则。

6. **Stacking 不是死路** — T47 失败是因为用了**iter_002 内部 5-horizon CE prob 互相重复**；**iter_013 regression + iter_002 5-horizon prob (#24) 互补**，Ridge meta 期望 +0.3 ~ +1.5。

7. **Ensemble multiplier 极强** — 单 lever +1 看起来不大，但 (LGBM-regr + CB-regr + XGB-regr + MLP-regr + Quantile-regr + Vol-thr) 6 lever 配合，**理论上能 +3 ~ +8 LOSO ≈ +2 ~ +6 平台**（透传 0.70），从 iter_013 +19.23 推到 ~+23-26。

---

## K. 风险与已知失败 mode

- **NN OOD 跨 sym 历史不稳** — #1 必须做 5-fold LOSO ablation，不能只看 LOSO-equiv；如果 fold variance > 5，不进 iter_014。
- **CatBoost Ordered + GPU + MultiClass 不支持** (T49 教训) — 但 **Regression 不是 MultiClass，应该可以 Ordered + GPU**；先 pilot 1 fold 验证。
- **Quantile regression 5-q ensemble 训练慢** — GPU 5-quantile × 5-seed = 25 boosters；需要 ~1.5h GPU 时间预留。
- **Vol-conditional thresh α DE 可能 overfit** — V4 walk-forward val DE，再 442k test ablate；对比 single-bucket baseline。

---

## L. 与 R36 / R37 的差异

| Worker | Focus | 重心 |
|---|---|---|
| R36 | "简单 + 颠覆" | time encoding / multiclassova / snapshot ensemble (训练 trick) |
| R37 | "leaderboard gap" | regression / multi-horizon vote / time-bucket thresh (decision-side) |
| **R40** (本文) | **"model arch + decision rule"** | **NN-regression / CatBoost-regression / XGBoost-regression / Quantile-regression / Vol-conditional EV / LGBM-custom-PnL-obj** |

R36/R37 已覆盖 time / multiclassova / regression 的 LightGBM 版；R40 **填的是：把 regression target 从 LightGBM 推广到所有其他模型类型 + 把 EV-gate 从 1 对全局阈值推广到 distribution-aware / vol-conditional**。

---

RESULT: task=r40_arch metrics={n_ideas=25, top_5=mlp_regr/catboost_regr/xgb_regr/lgb_quantile/vol_cond_ev} notes=最大盲区是 NN with regression target on Δmid_norm + EV-gate（NN 4 次失败全用 3-class CE 或 direct PnL loss）；CatBoost / XGBoost regression 同样从未试；LGBM 还有 quantile / huber / L1 / tweedie 4 个 objective 全没试；decision rule 可加 vol-conditional thresh / distribution-aware gate / Bayesian prior 三种增强；T60 direct PnL NN 失败可在 LGBM custom obj (#12) 重做
