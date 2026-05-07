# R32 Top-10 Ideas — OOD Generalization for Our LightGBM HFT Pipeline

> Date: 2026-05-07
> Source: `r32_ood_tabular_gbdt.md`（详细笔记 + 25-paper survey）
> 当前 SOTA: iter_006 LOSO h_60 = +13.61。**目标: 找到下一个让 LOSO 显著突破的 lever**。
>
> ⚠️ 必须遵守 3 条硬约束：
> 1. `date` 评测置 0 — 不可用
> 2. Predictor 不可跨 predict() 维护 state
> 3. **sym-agnostic at inference**（训练时用 sym 当 group 是 OK 的，inference 不能依赖 sym ID）

---

## 排名摘要表

| Rank | Idea | 期望增益 LOSO h_60 | 工时 | GPU | 硬约束 OK | 风险 | 优先级 |
|-----|------|---------------------|------|-----|-----------|------|--------|
| **1** | DART boosting + 强 L2 + 长 boost-round | **+1 ~ +4** | 2 h | 0 h | ✅ | 极低 | 🔥🔥🔥🔥🔥 |
| **2** | Group DRO LightGBM (sym=group, dynamic sample_weight) | **+2 ~ +6** | 6 h | 1 h | ✅ | 低 | 🔥🔥🔥🔥🔥 |
| **3** | Sym-invariant feature pack: integrated OFI + VPIN + tick-aware | **+1 ~ +3** | 8 h | 0 h | ✅ | 低 | 🔥🔥🔥🔥 |
| **4** | Cross-Sym Mixup augmentation (R11 B 节，未做) | **+1 ~ +4** | 6 h | 1 h | ✅ | 中 | 🔥🔥🔥🔥 |
| **5** | Stable Feature Boosting 2-stage（stable head + residual head） | **+1 ~ +5** | 1 day | 1 h | ✅ | 中 | 🔥🔥🔥🔥 |
| **6** | Linear backbone (Ridge/OLS) + LightGBM-on-residual | **+1 ~ +3** | 6 h | 0 h | ✅ | 低 | 🔥🔥🔥 |
| **7** | Asymmetric reweighting: 2-3x weight 难 sym | **+0.5 ~ +3** | 3 h | 0 h | ✅ | 中 | 🔥🔥🔥 |
| **8** | Scale-invariant feature audit + 强制 ratio 化（226→226） | **+0.5 ~ +3** | 1 day | 0 h | ✅ | 低 | 🔥🔥🔥 |
| **9** | Quantile head with ε-Insensitive Pinball / Huber loss | **+0 ~ +3** | 1 day | 1 h | ✅ | 中-高 | 🔥🔥 |
| **10** | LightMIRM-style IRM penalty via per-sym loss replaying | **+0 ~ +3** | 1.5 day | 1 h | ✅ | 中-高 | 🔥🔥 |

**总览说明:**
- 1+2+3 是 **"first wave"**：合计 ~16 h，预期合计 +4 ~ +13 (assuming 部分 overlap)，**应 1-2 天内全部上**
- 4+5 是 **"second wave"**：依赖 first wave 结果定方向
- 6-10 是 backup pool，根据 first wave 结果选 1-2 个深入

---

## 1. DART Boosting + Heavier L2 ⭐ Top Pick

**一句话描述:** 把 LightGBM 的 `boosting_type='gbdt'` 改成 `'dart'`，drop_rate=0.1, max_drop=50, skip_drop=0.5；同时把 lambda_l2 从默认提到 1.0-5.0。

**理论支持:** [DART (Vinayak & Gilad-Bachrach 2015)](https://arxiv.org/abs/1505.01866)；[LightGBM doc](https://lightgbm.readthedocs.io/en/latest/Parameters.html)；[D.A.R.T 实证 — Medium](https://medium.com/@meir412_37692/d-a-r-t-your-new-weapon-against-overfitting-in-boosting-models-9ea4e6aa435b)

**为什么 work:**
- 我们诊断 **best_iter ~120 << num_boost_round 600**，标准的 over-specialization signal — 后期 trees 只对少数 train samples 起作用，对 OOD sym 一样
- DART 让每轮新 tree 必须 compensate 一组随机 dropped trees 之和 → 强制每棵 tree 学更通用的 pattern，不允许"少数样本修补"

**预期 LOSO h_60 提升:** **+1 ~ +4**（保守 +1，乐观 +4 因为我们 over-specialization 诊断很明确）

**实施成本:** 2 h（配置改动 + 跑 5-seed ensemble，无 GPU 需求 — 也可 GPU 加速训练）

**绕开 sym-agnostic 硬约束?** ✅ 完全 — DART 与 sym 无关

**风险:** 极低。最坏情况 = 持平 GBDT，无 PnL 倒退（DART 一般不会变更糟，最多平）

**实验设计:**
```python
# 单点改动 ± 5-seed
params = {
    'boosting_type': 'dart',          # was 'gbdt'
    'drop_rate': 0.1,
    'max_drop': 50,
    'skip_drop': 0.5,                 # 50% rounds skip dropout (退回 GBDT)
    'uniform_drop': False,            # 用 importance-weighted drop
    'lambda_l2': 2.0,                 # was 0.0 or 0.1
    'num_boost_round': 1500,          # DART 收敛慢, 加倍
    'learning_rate': 0.04,            # 略调小
    # 其他不变
}
```
**对比基线:** iter_006 配置（aug_a 5-seed LightGBM + DE 4D thresh）。LOSO h_60 +13.61 → 看新 LOSO 是否 >= +14.5。

---

## 2. Group DRO LightGBM via Dynamic Sample Weight ⭐⭐ Top Pick

**一句话描述:** 把 sym ∈ {0,1,2,3,4} 当 5 个 environments，每 K=200 boosting rounds 暂停训练，计算 per-sym training loss；对 worst-sym 样本加权（softmax(-loss/τ)），继续训。

**理论支持:** [Sagawa 2019 (arXiv 1911.08731)](https://arxiv.org/abs/1911.08731)；[Group DRO under uncertainty 2024 (arXiv 2509.08942)](https://arxiv.org/html/2509.08942v1)；R11 A 节已设计未实施。

**为什么 work:**
- 当前 LOSO 失败的本质：sym=2 fold 模型 acc < baseline → 模型在 train 时让 sym=2 的 loss "随便高" 反正其他 sym loss 低，平均 loss 优化得不错
- Group DRO 强迫模型把 worst-sym loss 也压低，**训练时就 anticipates LOSO 评测**
- 软加权（τ-sharpened）vs 硬 worst：sym=2 outlier 极端，硬选会被 dragging；用 τ=0.5 让 worst 比 ERM 多关心，但不全押

**预期 LOSO h_60 提升:** **+2 ~ +6**（直接打 LOSO 失败的根因）

**实施成本:** 6 h（custom training loop wrapping `lgb.train`，每 K rounds callback reweight）

**绕开 sym-agnostic 硬约束?** ✅ — sym 仅在**训练时**用作 group label；inference 时 LightGBM model 不需要 sym 输入

**风险:**
- **低-中**：LightGBM 上正确实现 Group DRO 是 routine（callback 模式）
- **τ 调参**: τ=0.5 起步，可能要试 [0.2, 0.5, 1.0, 2.0]
- **K 调参**: K=200 起步

**实验设计:**
```python
# Pseudo
def group_dro_callback(env, sym_array, K=200, tau=0.5):
    if env.iteration % K != 0 or env.iteration == 0:
        return
    # 对当前训练集每个样本算 loss
    pred = env.model.predict(env.train_set.data, raw_score=True)
    losses = compute_per_sample_loss(pred, env.train_set.label)
    # per-sym mean loss
    sym_loss = {s: losses[sym_array == s].mean() for s in range(5)}
    # softmax weight
    weights = np.exp(np.array(list(sym_loss.values())) / tau)
    weights /= weights.sum()
    # 把 sample_weight 设为 weights[sym]
    new_sw = np.array([weights[s] for s in sym_array])
    env.train_set.set_weight(new_sw)
```
**对比基线:** iter_006。LOSO h_60 +13.61 → 看 +15.5 ~ +19。**特别看 sym=2 fold acc 是否过 baseline (0.608)**。

---

## 3. Sym-Invariant Feature Pack ⭐ Cheap Wins

**一句话描述:** 加 5-8 条 universal microstructure features：
- **iOFI_{1,3,5}** (integrated OFI 各 horizon, Cont-Cucuringu 2023)
- **VPIN_50bucket** (volume-clock toxicity, Easley/López de Prado/O'Hara)
- **spread_per_tick** (spread / inferred_tick_size)
- **queue_concentration** (queue_at_best / total_queue_top5)
- **realized_skew_intra-window** (3rd moment of returns within 100 ticks)

**理论支持:**
- [Cont, Cucuringu, Zhang (Quantitative Finance 2023)](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159) — iOFI cross-stock universal
- [Sirignano & Cont 2019](https://arxiv.org/abs/1803.06917) — universal features in pooled training
- [Easley/López de Prado/O'Hara VPIN](https://www.quantresearch.org/VPIN.pdf)；2024 [Easley crypto extension](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- [Briola et al. 2024 LOB benchmark](https://link.springer.com/article/10.1007/s10462-024-10715-4) — tick-size aware features

**为什么 work:** 这 5-8 条都是**理论上 sym-invariant** 的 — 它们的 distribution 跨 sym 应近似一致，因此 LightGBM split 在它们上不会绑死某 sym 的 scale。**先验地补上 universal feature 子集**，让 model 优先在它们上做 split，留更少的 OOD 风险给其他 226 features。

**预期 LOSO h_60 提升:** **+1 ~ +3**

**实施成本:** 8 h（5-8 feature × ~1 h 实现 + 单元测试 + retrain），无 GPU

**绕开 sym-agnostic?** ✅ 全部 stateless 100-tick 窗口内计算，无 sym 依赖

**风险:** 低 — 加 feature 是 monotonic improvement（除非 collinear，retrain 时让 LightGBM 自动选）

**实验设计:**
1. 在 build_cache 里加 5-8 列新 feature
2. retrain iter_006 配置 + 新 feature column
3. **对比 OFF / ON 5-seed LOSO**，看是否 >= +14.5
4. 顺带看 feature_importance 是否 iOFI / VPIN 进 top-30

---

## 4. Cross-Sym Mixup Data Augmentation

**一句话描述:** 训练时按概率 p=0.3 把同一 batch 内不同 sym 的两个 100-tick 窗口做 mixup（α ~ Beta(0.4, 0.4)）：x' = λx_i + (1-λ)x_j，y' soft-label。LightGBM 路径：预生成 N_aug = 0.5×N_train 条 mixup 样本拼到训练集。

**理论支持:**
- R11 B 节已设计；[Embarrassingly Simple MixUp for TS (arXiv 2304.04271)](https://arxiv.org/abs/2304.04271)
- [Jane Street 1st place writeup](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348) — mixup "fills empty space, soften overfitting"
- 2024 ResearchGate "Data augmentation for stock return prediction" — mixup 在 GBDT stock prediction 全部 case 都有 accuracy 提升

**为什么 work:**
- 强制 LightGBM 在 sym_i 与 sym_j 之间 interpolated input 上仍输出合理 prediction → 学到 sym-invariant decision boundary
- 与 N1（DART）和 N2（Group DRO）orthogonal — 一个改 data, 一个改 loss landscape, 一个改 boosting strategy

**预期 LOSO h_60 提升:** **+1 ~ +4**

**实施成本:** 6 h（pre-generate mixup samples + soft-label LightGBM custom objective）

**绕开 sym-agnostic?** ✅

**风险:**
- **中**：100-tick 窗口的时序 coherence 可能被 mixup 打散
- **对策**：试 "static-only mixup" 先 mix 静态 features（mid_price, spread, OFI summary 等），保留 raw LOB 时序部分原样

**实验设计:**
```python
# Pre-generate
N_aug = int(0.5 * len(X_train))
sym_pairs = []
for _ in range(N_aug):
    s_i, s_j = np.random.choice([0,1,2,3,4], 2, replace=False)
    idx_i = np.random.choice(np.where(sym_train==s_i)[0])
    idx_j = np.random.choice(np.where(sym_train==s_j)[0])
    lam = np.random.beta(0.4, 0.4)
    X_aug = lam * X_train[idx_i] + (1-lam) * X_train[idx_j]
    y_aug = lam * y_train[idx_i] + (1-lam) * y_train[idx_j]  # soft label
    # ...
```

---

## 5. Stable Feature Boosting (SFB) — 2-Stage LightGBM

**一句话描述:** 两阶段训练：
- Stage A: train LightGBM-A 在**仅 sym-invariant features** 上 → 输出 ŷ_stable
- Stage B: train LightGBM-B 在**全部 features**, target = (y_true − α·ŷ_stable)
- Inference: y_pred = ŷ_stable + ŷ_residual_B

**理论支持:** [Eastwood et al. NeurIPS 2023 — SFB (arXiv 2307.09933)](https://arxiv.org/abs/2307.09933)

**为什么 work:**
- LightGBM-A 学跨 sym universal 的因果（OOD 无虞）
- LightGBM-B 学剩余 spurious 的 fine-grain（OOD 风险大但 magnitude 小）
- 在 (stable ⊥ unstable) | y 假设下证明 asymptotically optimal
- **关键巧妙:** 用 Stage A 的 ŷ_stable 作为 pseudo-label "guidance"，让 Stage B 只在 stable 信号已对的样本上学剩余 wrinkle

**预期 LOSO h_60 提升:** **+1 ~ +5**

**实施成本:** 1 day（feature 子集人工或 IC variance 自动选 + 双训 + 调 α）

**绕开 sym-agnostic?** ✅ 两个 LightGBM 都不需要 sym ID

**风险:**
- **中**：sym-invariant feature 子集选错（太多 / 太少）会失效
- **对策**: 用 N3 的 5-8 features + 自动添加：在 train data 上算每条 feature 的 per-sym IC，取 IC variance 低的 top-30 加入

**实验设计:**
1. 选 50 条 candidate sym-invariant features (low per-sym IC variance)
2. Train A on these 50, Train B on full 226
3. α grid: [0.5, 0.7, 1.0, 1.3] (在 val 上选)
4. 5-seed ensemble + DE thresh

---

## 6. Linear Backbone + LightGBM-on-Residual

**一句话描述:** Stage 1: Ridge regression on 5-10 维核心 OFI lag features → ŷ_lin。Stage 2: LightGBM target = y − ŷ_lin。Inference: y_pred = ŷ_lin + ŷ_LightGBM。

**理论支持:** [R31 P14 — Hybrid VAR + FNN (arXiv 2411.08382)](https://arxiv.org/abs/2411.08382)；general residual stacking。

**为什么 work:**
- Linear cross-sym 通用性强（OFI 线性效应在 LOB 是 universal 的，Cont 等多数文献支持）
- LightGBM 仅学 ~30% 残差非线性，OOD 风险只在该 30% — 总 OOD brittleness 减少
- 与 N5 (SFB) 区别：N5 用 stable feature 的 LightGBM，N6 用 stable feature 的 linear（更弱但更 invariant）

**预期 LOSO h_60 提升:** **+1 ~ +3**

**实施成本:** 6 h

**绕开 sym-agnostic?** ✅

**风险:** 低

---

## 7. Asymmetric Reweighting Based on Sym Difficulty

**一句话描述:** 训练时给"最难学"的 sym（sym=2）2-3x sample weight，借鉴 Asymmetric Generalization 论文（train on hard, transfer to easy）。

**理论支持:** [Orderbook Feature Learning Asymmetric Generalization (arXiv 2510.12685)](https://arxiv.org/abs/2510.12685) — train on more challenging market generalizes well, reverse fails (12x degradation)

**为什么 work:** 我们 sym=2 fold acc 跌得最狠（0.279 vs 0.608 baseline），可能因为 sym=2 是分布最 atypical / illiquid。如果训练时让模型 prioritize sym=2，转移到其他 sym 会更稳。

**预期 LOSO h_60 提升:** **+0.5 ~ +3**

**实施成本:** 3 h（fixed-weight 设置 + retrain）

**绕开 sym-agnostic?** ✅（仅训练用 sym 加权）

**风险:** 中 — 可能反而 hurt sym=0,1,3,4 fold

**注意:** 与 N2 Group DRO 概念上**类似但更简单**。如果做 N2 就不必额外做 N7（N2 自动找到最难的 sym 加权）。N7 是 N2 的"廉价替代"。

---

## 8. Scale-Invariant Feature Audit & Refactor

**一句话描述:** 审查现有 226 features，把所有 absolute-scale features 改成 ratio / log 形式：
- `price_t` → `log(price_t / mid_window_mean)`
- `volume_t` → `volume_t / EMA_volume_window`
- `tick_count` → `tick_count / window_length`
- 等等

**理论支持:** [Sirignano & Cont 2019](https://arxiv.org/abs/1803.06917)；[TradeFM 2026](https://arxiv.org/abs/2602.23784) "scale-invariant features eliminate asset-specific calibration"；[Briola 2024](https://link.springer.com/article/10.1007/s10462-024-10715-4) tick-size group dependence

**为什么 work:** LightGBM split 阈值是绝对值。如果 sym=0 价格在 [10, 12] 区间，sym=1 在 [50, 60]，则 "price > 11" 这个 split 对 sym=1 永远 True → 看似 informative，OOD 灾难。**Ratio / log 化让所有 sym 在同一 numerical scale**。

**预期 LOSO h_60 提升:** **+0.5 ~ +3**（如果当前 features 已大部分归一化，则下限低；如果还有大量 absolute-scale，则上限高）

**实施成本:** 1 day（226 features 逐条审查 + 改造 + retrain）

**绕开 sym-agnostic?** ✅

**风险:** 低

**优先实施 trigger:** 在做 N3 时顺便做 — N3 的 universal feature 本身已经 scale-invariant，做完后 dump feature stats 看看现有 226 哪些不是。

---

## 9. Quantile Head with ε-Insensitive Pinball / Huber Loss

**一句话描述:** 把 multi-class softmax classifier 换成 quantile regressor（10 个 quantile head, q={0.05, 0.15, ..., 0.95}），用 ε-Insensitive Pinball 或 Huberized Pinball loss 训练；directional 信号 = sign(median - threshold) × confidence(IQR_size)。

**理论支持:**
- [Pinball boosting CSDA 2024](https://www.sciencedirect.com/science/article/pii/S0167947324001117)
- [Robust Quantile Huber 2024 (arXiv 2401.02325)](https://arxiv.org/html/2401.02325v2)

**为什么 work:**
- Multi-class 在 OOD 时容易在某 class 收 spurious correlation；quantile 直接回归 return distribution，OOD 时只是 quantile 估计偏差大但形状对
- ε-insensitive 处理 GBDT 上 pinball 缺 curvature 的问题
- IQR_size 作 confidence → 可与 trade gate 结合（IQR 大 = uncertain → 不交易）

**预期 LOSO h_60 提升:** **+0 ~ +3**

**实施成本:** 1 day（Custom LightGBM objective + 多 quantile head + decoder）

**绕开 sym-agnostic?** ✅

**风险:** 中-高
- **风险 1:** 完全 reframe 任务，回退路径长
- **风险 2:** Custom obj 在 LightGBM 调试坑多
- **建议:** 先 Stage 1-2 做完看 LOSO 状况再决定是否上

---

## 10. LightMIRM-Style IRM Penalty via Per-Sym Loss Replaying

**一句话描述:** LightMIRM (ICDE 2023) 的 LightGBM-IRM 嫁接思路 — 每 K boosting rounds 算 per-sym loss + IRM penalty `‖∇_w R_e(w)‖²|_{w=1}` 的近似（用一阶差分）→ softmax 反向加权 sample。

**理论支持:** [LightMIRM (ICDE 2023)](https://blacksingular.github.io/papers/icde23-LightMIRM.pdf)；原 [IRM (arXiv 1907.02893)](https://arxiv.org/abs/1907.02893)

**为什么 work:** Group DRO 是 worst-case 化；IRM 是要求**所有 env 共享同一 optimal classifier**。理论上更强的 invariance constraint。

**预期 LOSO h_60 提升:** **+0 ~ +3**

**实施成本:** 1.5 day（IRM penalty 在 LightGBM 上的近似比较 hacky）

**绕开 sym-agnostic?** ✅（只训练用 sym）

**风险:** 中-高
- **风险:** IRM 在 covariate shift 下不稳定（Krueger 2021 V-REx 取代 IRM 的主因）；如果做 IRM 应同时考虑 V-REx 替代
- **建议:** 在做完 N2 Group DRO 后看是否需要更强的 invariance constraint。如 LOSO 卡在 sym=2 fold 上 N2 也救不了，再试 N10。

---

## 推荐执行顺序

### Sprint 1（1-2 天，cheap & high-confidence）
1. **N3 Sym-Invariant Features** — 加 5-8 条 (+1 ~ +3)
2. **N1 DART Boosting** — 配置改动 (+1 ~ +4)
3. **N8 Scale-Invariant Audit** — 顺手做 (+0.5 ~ +3)

预期 Sprint 1 后 LOSO h_60 = **+15 ~ +20**（部分 overlap 必然有）

### Sprint 2（2-4 天，medium）
4. **N2 Group DRO** — sym=group dynamic weight (+2 ~ +6)
5. **N4 Cross-Sym Mixup** — pre-generate aug data (+1 ~ +4)

预期 Sprint 2 后 LOSO h_60 = **+18 ~ +28**

### Sprint 3（4-7 天，medium-heavy，conditional）
6. **N5 Stable Feature Boosting** — 2-stage training (+1 ~ +5)
7. **N6 Linear backbone + LightGBM residual** — (+1 ~ +3)

### Sprint 4（backup, on-demand）
8-10. N7 / N9 / N10 — 仅在 LOSO 进展卡住时上

---

## 一些 Anti-Pattern（不要做）

- ❌ Asset embedding / sector embedding — 违反 sym-agnostic
- ❌ Foundation model zero-shot (Chronos / TimesFM) — Re(Visiting) TSFM (R31 P3) 已证伪
- ❌ Bidirectional Mamba / Transformer — inference-time future leak
- ❌ Test-time adaptation (TENT/SHOT) — 评测 shuffle 破坏所有 batch statistics
- ❌ Conformal prediction across syms — 没法保证 cal/val 分隔
- ❌ Per-sym normalization 统计量 — 训练外 sym 没 stat
- ❌ 再调阈值 / 再做校准 / 再做 NN — T37/T38/T40/T1/T19/T32 已证饱和

---

## 关键 references

- N1 DART: [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)；[D.A.R.T Medium](https://medium.com/@meir412_37692/d-a-r-t-your-new-weapon-against-overfitting-in-boosting-models-9ea4e6aa435b)
- N2 Group DRO: [Sagawa 2019 (arXiv 1911.08731)](https://arxiv.org/abs/1911.08731)；[2024 follow-up (arXiv 2509.08942)](https://arxiv.org/html/2509.08942v1)
- N3 Universal Features: [Sirignano & Cont 2019 (arXiv 1803.06917)](https://arxiv.org/abs/1803.06917)；[Cont, Cucuringu, Zhang 2023](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159)；[VPIN](https://www.quantresearch.org/VPIN.pdf)
- N4 Mixup: [Embarrassingly Simple MixUp TS (arXiv 2304.04271)](https://arxiv.org/abs/2304.04271)
- N5 SFB: [Eastwood NeurIPS 2023 (arXiv 2307.09933)](https://arxiv.org/abs/2307.09933)
- N6 Linear+GBDT: [R31 P14 Hybrid VAR+FNN (arXiv 2411.08382)](https://arxiv.org/abs/2411.08382)
- N7 Asymmetric: [Orderbook Asymmetric Generalization (arXiv 2510.12685)](https://arxiv.org/abs/2510.12685)
- N8 Scale-invariant: [TradeFM 2026 (arXiv 2602.23784)](https://arxiv.org/abs/2602.23784)；[Briola 2024](https://link.springer.com/article/10.1007/s10462-024-10715-4)
- N9 Quantile Pinball: [Pinball boosting CSDA 2024](https://www.sciencedirect.com/science/article/pii/S0167947324001117)
- N10 LightMIRM: [LightMIRM ICDE 2023](https://blacksingular.github.io/papers/icde23-LightMIRM.pdf)
