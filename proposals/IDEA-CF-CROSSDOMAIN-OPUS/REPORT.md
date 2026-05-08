# IDEA-CF-CROSSDOMAIN-OPUS — Cross-Domain Brainstorm Report

> **Worker 类型**: brainstorming, 跨域类比专长, 无训练 / 无 GPU / 无 web search
> **日期**: 2026-05-08（初赛 5-11 截止 → 剩 ~3.5 天）
> **基线对照**: iter_015 = +40.09 LOSO（T87 SPO+ DFL）；prior best iter_014 = +38.28；iter_006 LightGBM = +13.61；user 给出的 +36.22 应指 iter_014 一档之前的版本。本报告所有 upside 以 **+40 LOSO baseline** 衡量增量。
> **核心约束（红线，见 CRITICAL_CONSTRAINTS.md）**: ① date 评测置 0; ② 测试点顺序打乱, Predictor 必须 stateless; ③ sym 0–4 但可能含训练外股票, 模型必须 sym-agnostic
> **数据形状**: 5 sym × 120 day × (AM+PM) × 2001 tick × 154-D feature ≈ 1.77M sample; label = {0,1,2}

---

## 0. 执行摘要 — Top 5 跨域类比

| 排名 | 领域 | 方法 | 类比强度 | 可行性 | 预期 LOSO 增量 | 实施时间 |
|---|---|---|---|---|---|---|
| ⭐ #1 | Med Imaging | **Selective Prediction with Abstain** (Geifman & El-Yaniv 2017) | 极强 | 高 | **+2 ~ +4** | 12 h |
| ⭐ #2 | Info Theory | **Information Bottleneck + sym-adversarial GRL** (Tishby 99 / Ganin 16) | 极强 | 中高 | **+2 ~ +3** | 12 h |
| ⭐ #3 | RL / Bandits | **Conservative Q-Learning (CQL)** (Kumar 2020) | 强 | 中 | **+1 ~ +3** | 16 h |
| ⭐ #4 | NLP SSL | **Cross-sym contrastive (SimCSE / SimCLR)** + GBDT 头 | 强 | 中 | **+1 ~ +3** | 16 h |
| ⭐ #5 | CV OOD | **Mahalanobis 距离 → selective 门控** (Lee 2018) | 中 | 极高 | **+1 ~ +2** | 8 h |

**核心 insight**: 已尝试的 80+ 实验里, 90% 是 "怎么把 GBDT 训得更准" 这一个 axis 上的探索。但本任务的本质是**带弃权的 cost-sensitive 决策**——`label=1`(平) 不是一个分类目标, 它是 **abstain action**。这个 reframe 在 medical imaging / RL bandits / selective classification 文献里已经成熟, 而本项目尚未系统化吸收。Top 5 都是围绕 "把弃权升级为 first-class decision" 这条主线。

---

## 1. RL / Bandits 领域

### 1.1 Doubly Robust Off-Policy Evaluation (Dudik, Langford & Li 2011)
**原方法**：当 logged data 来自老策略 π_β, 评估新策略 π 的价值时, IPS 估计有高方差。Doubly Robust 估计:
```
V̂_DR(π) = E_n[ f̂(x, π(x)) + (1[a=π(x)] / p̂(a|x)) · (R - f̂(x, a)) ]
```
其中 f̂ 是 reward 的 direct method 估计, p̂ 是 propensity 估计。即使 f̂ 有偏, 只要 p̂ 准就 unbiased。

**类比映射**:
- 原 logged data → 1.77M (sym, day, t, R_{t+h}) 样本
- 老策略 π_β → 训练集"全开仓 = label=2 if Δmid>0 else label=0"
- 新策略 π → 我们 GBDT 的 argmax/threshold 策略
- f̂(x,a) = Δmid·a − fee·|a| 的 GBDT 回归预测; p̂(a|x) = π(x) softmax

**Plan（伪码）**:
```python
# 1. 训练 direct-method baseline f̂: regression GBDT on R_{t+h} × a_taken
# 2. 训练 policy π: 现有 multi-head GBDT
# 3. 阈值搜索: 用 V̂_DR 而不是 V̂_IPS 计算 hold-out PnL
#    → 阈值 τ 调优 variance ↓ ~30%
```
**预期**: τ 调优更稳, fold 间方差 ↓; LOSO **+0.5 ~ +1.5**
**时间**: 8 h
**红线**: ✅ stateless, ✅ sym-agnostic, ✅ no date
**风险**: 与已有 T62 unbiased thresh / T76 decoupled thresh 部分重合; 但 DR 的 variance reduction 是新角度
**与已有同构性**: T62 用 IPS-style hold-out; **DR 是 IPS 的严格升级**, 不算重复

### 1.2 ⭐ Conservative Q-Learning (CQL, Kumar et al. 2020) — Top 5 #3
**原方法**：offline RL 中，标准 Q-learning 会高估 OOD action 的 Q 值。CQL 加 conservatism penalty:
```
L_CQL = L_Bellman + α · E_s[ logsumexp_a Q(s,a) − E_{a~π_β(·|s)} Q(s,a) ]
```
让 Q(s, a_OOD) 被压低, 只对 in-distribution action 信任。

**类比映射**:
- s = 100×154 LOB window (注意: state 完全在当前 batch 内, **不跨调用**)
- a ∈ {SELL, ABSTAIN, BUY} = {0, 1, 2}
- R(s, a) = (a−1)·Δmid_{t+h} − 0.0001·|a−1|·(2·midprice_t)  [official PnL formula]
- π_β = 训练数据"实际 oracle action" = sign(Δmid) if |Δmid| > α else ABSTAIN
- OOD action = 在 noisy 样本上 BUY/SELL（高概率亏）

**Plan（伪码）**:
```python
# Q-net: φ(s) → ℝ³ (one head per action)
# Single-step bandit (no bootstrapping needed since horizon is fixed h=60)
# Target: Q*(s,a) = R(s,a)  [因为 horizon 已经固化, 不需 γ·max Q(s')]
# CQL loss:
L = Σ_t Σ_a (Q(s_t,a) − R_t(a))²
  + α · [logsumexp_a Q(s_t,a) − Q(s_t, a_oracle_t)]
# Policy at inference: π(s) = argmax_a Q(s,a)
# 弃权升级: 当 max Q(s,a) − Q(s, ABSTAIN) < margin → 弃权
```

**与 T60 区分**: T60 是 **policy gradient with PnL loss**（直接最大化期望 reward, 单 head softmax 输出 action 概率, ent_beta + lam_act 正则）; **CQL 是 value-based with conservatism penalty**, 显式压低未见 action 的估值, 这正是处理 sym-OOD 的合适机制。两者完全不同的 inductive bias。

**预期**: LOSO **+1 ~ +3**
**时间**: 16 h
**红线**: ✅ 全合规 (state 仅当前窗口, no date, no sym embedding)
**风险**: NN backbone 在本项目历史上偏弱 (T93/T94/T95 均 < GBDT)。**Mitigation**: Q-net 用 GBDT (LightGBM) 实现 — 三个 regression head, 训练 R(s, a=BUY)/R(s, a=SELL)/R(s, a=ABSTAIN=0); CQL 正则化通过 sample weight + extra "aug action" 数据点近似实现。

---

## 2. CV OOD Detection / Domain Generalization

### 2.1 ⭐ Mahalanobis Distance OOD Score (Lee et al. 2018) — Top 5 #5
**原方法**：用 pre-final layer features 估 per-class Gaussian {μ_c, Σ}, 在测试时 OOD score:
```
M(x) = min_c (h(x) − μ_c)^⊤ Σ^{-1} (h(x) − μ_c)
```
高 M → OOD; 用于 selective rejection。

**类比映射**:
- 原 in-distribution class = 5 个训练 sym 各自的 representation 簇
- 原 OOD = 训练外 sym (赛题明确允许出现)
- h(x) = GBDT 倒数第二层 = leaf indicator vector (LightGBM `pred_leaf=True` 输出每棵树的 leaf id, 拼成 hash 后再 z-score)
  - 或更简单: h(x) = handcrafted 154-D 特征本身 (经全局 z-score)

**Plan（伪码）**:
```python
# Train 阶段:
H = StandardScaler().fit_transform(X_train)
mu_c = {c: H[sym==c].mean(0) for c in [0,1,2,3,4]}
Sigma = empirical_cov(H − mu_pooled)  # tied covariance
Sigma_inv = pinv(Sigma)

# Inference: 完全 stateless, 不需要 sym ID
def gate(x):
    h = scaler.transform(x)
    M = min((h - mu_c[c]) @ Sigma_inv @ (h - mu_c[c]).T for c in range(5))
    return M  # higher = more OOD

# Trade rule:
# trade if (|GBDT_pred| > τ_signal) AND (M < τ_OOD)
# τ_OOD calibrated on synthetic shift: hold out sym=k, M-quantile=0.95
```

**预期**: LOSO **+1 ~ +2** (剔除 OOD 高方差预测)
**时间**: 8 h
**红线**: ✅ 完全 stateless 且 sym-agnostic (Mahalanobis 不需要测试 sym ID)
**与已有同构性**: T82/T83 robust thresh 是 score-domain rule; Mahalanobis 是 **input-feature-domain** OOD detection — 互补正交
**风险**: 相关矩阵 Σ 在 high-D 下不稳; **mitigation**: 用 diagonal-shrinkage Σ_λ = (1−λ)Σ + λ·diag(Σ), λ=0.1

### 2.2 CORAL — Second-Order Moment Matching (Sun & Saenko 2016)
**原方法**：source/target domain 的 covariance 匹配:
```
L_CORAL = (1/(4d²)) ‖ Cov(features|source) − Cov(features|target) ‖_F²
```
**类比映射**：5 sym = 5 source domain; 训练时强制 5 个 sym 的 feature 协方差矩阵接近, 学到 sym-invariant representation。
**Plan**: NN encoder φ, 总损失 = CE(classifier(φ(x)), y) + λ·Σ_{i<j} ‖Cov(φ(X_i)) − Cov(φ(X_j))‖_F²
**预期**: +0.5 ~ +1.5
**时间**: 8 h
**与 T45 区分**: T45 GroupDRO 用 "max risk over groups" 的 worst-case formulation; CORAL 用二阶矩匹配的 distribution-matching formulation。在 GroupDRO 失败的项目里 CORAL 仍然 work 是常见现象 (DomainBed benchmark 数据)。
**红线**: ✅
**风险**: 同 T45, NN 不一定打过 GBDT

---

## 3. NLP Robust Pretraining / Self-Supervised

### 3.1 ⭐ Cross-Sym Contrastive Pretraining (SimCLR Chen 20 / SimCSE Gao 21) — Top 5 #4
**原方法**：encoder f, 对每个 sample x_i 构造 augmented view x_i'; InfoNCE loss:
```
L_i = −log(exp(sim(f(x_i), f(x_i'))/τ) / Σ_j exp(sim(f(x_i), f(x_j))/τ))
```
学到 augmentation-invariant representation。SimCSE 把 dropout 当 augmentation, 在 sentence representation 上 SOTA。

**类比映射**:
- 原 image augmentation → LOB window 的"看似不同但语义相同"扰动
- 关键 augmentation 设计 (本项目独有):
  1. **Time jitter**: 随机移除 5% 随机 ticks, 后 zero-pad to 100 (模拟 session 边界 / 网络抖动)
  2. **Feature dropout**: 随机置零 1% 特征
  3. **Cross-sym mix-up at encoder**: 同一时间窗内, 用另一 sym 的同类特征替换 10% (强制 sym-invariant)
  4. **Vol scaling**: 随机缩放整体 vol level by ∈ [0.7, 1.4]

**Plan（伪码）**:
```python
# Stage 1: SimCSE-style pretraining (无标签, 用所有 1.77M window)
encoder = ResMLP(154 → 128)  # 或 CNN-1D
for x in batches:
    z1 = encoder(augment(x))   # dropout-aug view 1
    z2 = encoder(augment(x))   # dropout-aug view 2 (different drop)
    L = InfoNCE(z1, z2, temperature=0.05)
    L.backward()

# Stage 2: Freeze encoder, 把 z = encoder(x) (128-D) 加入 GBDT 特征
X_aug = concat([X_handcrafted_226d, encoder(X)])  # 226 + 128 = 354
gbdt.fit(X_aug, y)
```

**关键 trick**: **InfoNCE 的 negative pool 跨 sym** — 强制不同 sym 的相同时刻 (相对盘口 patterns) 被拉近, 同 sym 不同时刻被拉远 (因为它们语义不同)。这把 sym-invariance 编码到 representation 里。

**预期**: LOSO **+1 ~ +3** (T44 R34 features 是 hand-crafted boost +0.5; learned representation 上限更高)
**时间**: 16 h
**红线**: ✅ 完全合规 — encoder 输入是当前窗口, 不需要 sym ID, 不维护 state
**与已有同构性**: T44 是 **手工** crystal feature; **SSL 是 learned** feature; 互补
**风险**: NN encoder 历史弱 (T93-95); **mitigation**: 用极小 model (3-layer MLP, hidden 256), L2 reg, early stop

### 3.2 Masked LOB Modeling (BERT-style, Devlin 2018)
**原方法**：随机 mask 15% input token, 训练模型预测被 mask 的内容; pretrain 后下游 fine-tune。
**类比映射**：mask 100-tick 窗口里 15% 的 timestep, 让 transformer 从 context 预测被 mask 的 154-D 特征。
**Plan**:
```python
# Encoder: small transformer (4 layers, 4 heads, hidden 128)
mask_idx = randperm(100)[:15]
x_masked = x.clone(); x_masked[mask_idx, :] = 0  # 或 mask token
y_pred = encoder(x_masked)
L = MSE(y_pred[mask_idx, :], x[mask_idx, :])
# downstream: 取 CLS token 或 mean-pool 喂 GBDT
```
**预期**: +1 ~ +2
**时间**: 16 h
**红线**: ✅
**与 #3.1 区分**: SimCSE 是 **instance-level** representation (整个窗口 1 个 vector); MLM 是 **token-level** (每 tick 1 个 vector, 适合时序内部依赖)
**风险**: transformer 在 1.77M 数据上可能 underfit; ablation 必备

### 3.3 Curriculum Learning by Horizon (Bengio et al. 2009)
**原方法**：从简单样本/任务先训练, 逐渐增加难度 → 收敛更稳, 泛化更好。
**类比映射**：信噪比从大到小排序 = h=60 (α=0.1%, |R|/σ 最大) → h=40 → h=20 → h=10 → h=5 (α=0.05%, 信号最弱)
**Plan**:
```python
# Sequential transfer:
m_60 = LightGBM().fit(X, y_h60)             # 简单, 高信噪比
m_40 = LightGBM(init_model=m_60).fit(X, y_h40)  # init_score 继承
m_20 = LightGBM(init_model=m_40).fit(X, y_h20)
... down to m_5
```
**预期**: +0.5 ~ +1.5 (主要在 h=5/10 这两个被忽视的 horizon)
**时间**: 6 h
**红线**: ✅
**与已有同构性**: T12 multihorizon stacking 是 **post-hoc ensembling**; 本 idea 是 **训练阶段 inductive transfer**, 不重复
**风险**: 可能 marginal, 因为 final score = max over 5 horizons, 已被 h=60 主导

---

## 4. Robotics Sim-to-Real

### 4.1 Domain Randomization (Tobin et al. 2017) — ⚠️ **同构于 T26, 不算新 idea**
**判定**: T26 已经做了 augmentation A/B/C (gain randomization on features); 本质是 DR 的 feature-space 实例。**Skip**。
**唯一可补充的版本**: **Physics-parameter DR** — 不是 per-feature gain, 而是从 session-level "市场参数" 角度随机化:
- 整体 vol level scaling 0.7-1.3
- 整体 spread bias +/-0.1bp
- Imbalance shift +/-0.05
这相当于把整个 session 重新生成。
**预期增量 (over T26)**: 0 ~ +0.5
**时间**: 4 h (small extension of T26 code)

### 4.2 ⭐ System Identification via In-Window Latent (RL² / Gupta et al. 2018)
**原方法**：robot adapt to unseen environment by **inferring environment params from current trajectory** (in RL² 用 RNN 在 episode 内编码 history 成 latent z, 后续策略 conditional on z)。

**类比映射**:
- 原 environment = 不同摩擦/质量/几何的 sim
- 这里 environment = 不同 sym + 不同时段的市场状态
- 关键: **z 必须从当前窗口推断, 不跨调用** (满足红线 ②)

**Plan（伪码）**:
```python
# Encoder φ: 100×154 → z ∈ ℝ^16  (regime embedding, 16-D enough)
# 注意: φ 输入仅当前窗口, 输出 z; z 与 sym ID 无关, 训练时不喂 sym
# z 通过 auxiliary loss 学习一些 robust signal:
#   - reconstruct future vol: φ(x) → z → predict vol_{t..t+60}
#   - 或者: contrastive loss within same session (similar state)

# 拼接到 GBDT:
X_aug = concat([X_handcrafted, z])
gbdt.fit(X_aug, y)
```

**预期**: +1 ~ +2; z 编码 "当前是什么 regime", 让 GBDT 在不同 regime 下做不同决策
**时间**: 12 h
**红线**: ✅ (z 仅来自当前 100-tick window, no sym, no date)
**与 #3.1 区分**: SimCSE 学的是 **instance-level invariance**; RL² 学的是 **regime-level identification** — 一个是降维, 一个是分桶
**风险**: 16-D bottleneck 可能太窄; sweep [8, 16, 32, 64]

---

## 5. Bioinformatics / Small-Sample Inductive Bias

### 5.1 Prototypical Networks (Snell, Swersky & Zemel 2017)
**原方法**：few-shot, 每 class 计算 prototype c_k = mean({f(x_i) : y_i=k}), 测试 P(y=k|x) ∝ exp(−d(f(x),c_k)/τ)。
**类比映射**:
- 原 few-shot class → "训练 sym 各自" 的特征模式
- ⚠️ **直接以 sym 作 prototype 违反红线**(测试 sym 可能不在训练里)
- **修正映射**: 用 unsupervised clustering 在 encoded space 找 K 个 "regime prototype" (K=10), 测试时 attention soft-route 到最近 prototype, 完全不用 sym ID

**Plan（伪码）**:
```python
# Phase 1: SSL encoder (从 #3.1 共享)
z = encoder(X)  # 1.77M × 128

# Phase 2: K-means K=10 在 z space 找 prototype
prototypes = KMeans(K=10).fit(z[::20]).cluster_centers_

# Phase 3: 给每个 sample 算 regime soft assignment (attention)
attn(x) = softmax(-||z(x) - prototype||²/τ)  ∈ Δ^9

# Phase 4: 训练 K 个 GBDT expert, 每个 expert 在自己 cluster 上训练
for k in range(K):
    weight_k = attn(X)[:, k]
    gbdt_k.fit(X_handcrafted, y, sample_weight=weight_k)

# Inference: ŷ = Σ_k attn(x)[k] · gbdt_k.predict(x_handcrafted)
```

**预期**: +1 ~ +2
**时间**: 12 h
**红线**: ✅ (regime 完全在 feature-space 推断, 不需 sym ID)
**与已有同构性**: T48 cross-sym pooling 已经标记为 inference 不兼容; **本 idea 用 cluster (而非 sym ID) 作 routing**, 不需要 cross-sample 实时统计 — 兼容
**风险**: K=10 的 expert 各自数据量 ~177k, 可能 over-fit; mitigation: K=5 + heavy reg

### 5.2 Strong Architectural Prior — LOB Equivariance
**原方法**：bioinformatics 中 protein structure 强制约束让 small-N 收敛 (e.g., AlphaFold 的 SE(3)-equivariance)。
**类比映射**：LOB 有内在结构:
- bid1..bid10 沿"价格档位距离 mid"单调
- 同 level 的 bid_i 与 ask_i **对称** (镜像)
- size 与 price 是 quasi-orthogonal 维度

**Plan**: DeepLOB 已用了 1D conv 沿 ladder, 但没有 **bid-ask 对称约束**。
```python
# Symmetric DeepLOB:
class SymLOBNet(nn.Module):
    def __init__(self):
        # Shared conv for bid stack and ask stack (with sign flip)
        self.conv_side = nn.Conv1d(2, 16, kernel_size=3)  # input: [price, size]
        ...
    def forward(self, x):
        bid_stack = x[:, BID_COLS]      # 10 levels × 2 (price, size)
        ask_stack = x[:, ASK_COLS]
        h_bid = self.conv_side(bid_stack)
        h_ask = self.conv_side(-ask_stack)  # sign flip price (mid-relative)
        return concat([h_bid, h_ask, ...])  # symmetry-aware
```
**预期**: +0.5 ~ +1.5
**时间**: 10 h
**红线**: ✅
**与 T93/T94/T95/T97 区分**: 这些用了标准 transformer / ResNet / DeepLOB, **没专门用 bid-ask 对称约束**。Sym-LOB 是新角度。
**风险**: NN historicall 弱; 但 strong prior 在 small N regime 是 **少数 NN 比 GBDT 更适合的场景**

---

## 6. Econometrics / Time Series

### 6.1 GARCH Vol-Aware Threshold (Engle 1982 / Bollerslev 1986)
**原方法**：σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}; vol cluster 时大 σ → 更宽信号阈值。
**类比映射**：当前 windowed σ̂_{t+h} 高 → 信号更不可靠 → 提高 trade threshold。
**Plan**:
```python
# 在当前 100-tick 窗内拟合 simple ARMA(0,1)-GARCH(1,1) on midprice returns
# (statsmodels.arch_model 单 fit ~10ms, 100-tick 数据足够)
σ̂_h = forecast h steps ahead
# Trade rule: trade if |GBDT_pred| > k · σ̂_h + 2·fee, k ∈ [0.5, 2]
```
**或更简单**: realized vol over last 60 ticks 替代 GARCH。
**预期**: +0.5 ~ +1.5
**时间**: 8 h
**红线**: ✅ (per-window 拟合, completely stateless)
**与 T76/T82/T83 robust thresh 区分**: 它们是 score-domain (predicted score 校准); 本 idea 是 **realized-vol-aware threshold** (input-domain) — 互补
**风险**: T88 已做 range vol features, 可能边际收益小

### 6.2 Bootstrap Confidence Interval Selective Trading (Efron 1979)
**原方法**：B 次 bootstrap fit, 给每个测试点 (μ̂, σ̂_bootstrap); CI 是 [μ̂ − 1.96σ̂, μ̂ + 1.96σ̂]; 决策若 CI 不跨 0 才下单。
**类比映射**：用 bagging 20 个 LightGBM (不同 bootstrap subsample), 在 inference 时算 bootstrap σ̂; 只在 |μ̂| > 2σ̂ 时 trade。
**Plan（伪码）**:
```python
# Bootstrap aggregation:
models = [LightGBM().fit(X[bootstrap_sample(N)], y[bootstrap_sample(N)]) for _ in range(20)]
preds = stack([m.predict(X_test) for m in models])  # (20, N_test)
mu_hat, sigma_hat = preds.mean(0), preds.std(0)

# Trade rule:
trade_signal = |mu_hat| > 2 * sigma_hat   # bootstrap CI excludes 0
final_action = sign(mu_hat) if trade_signal else ABSTAIN
```
**预期**: +1 ~ +2
**时间**: 6 h
**红线**: ✅
**与 T11 / T17 / T46 区分**: 它们做 **mean ensemble** — 用平均增稳; 本 idea 用 **std as gate** — 用 disagreement 弃权。是新 angle。
**风险**: 需要 20 个 model, 推理速度 20×; 可用 leaf-cache 加速

---

## 7. Medical Imaging / High-Stakes Decision

### 7.1 ⭐ Selective Prediction with Abstain (Geifman & El-Yaniv 2017, "Selective Net") — Top 5 #1
**原方法**：medical diagnosis 中 abstain 是 first-class action; 训练 (classifier g, selector h) 联合优化:
```
L = E_x[ h(x) · ℓ(g(x), y) ] + λ · max(0, c − E_x[h(x)])²
```
where h(x) ∈ [0,1] 是接受概率, c 是目标 coverage。

**类比映射**: ⭐ **完美映射** — 在本任务里, `label=1`(平) **本身就是 abstain action**; 不是分类的一个 class, 而是不下单的决策。把所有现有 3-class 模型 reframe 成 (binary classifier {BUY,SELL}, selector h ∈ [0,1]) 二头模型:

**Plan（伪码）**:
```python
# Phase 1 (training): 重定义 label
# 原 3-class y: 0=DOWN, 1=FLAT, 2=UP
# 新表示:
#   y_bin = sign(Δmid_{t+h})  ∈ {DOWN, UP}  (drop FLAT samples? 或 weight=0)
#   y_select = 1 if |Δmid| > α_h else 0     (是否值得 trade)

# 模型:
g(x) = ClassifierHead(x)  # binary {DOWN, UP}, 对所有样本都训
h(x) = SelectorHead(x)    # ∈ [0,1], 接受概率

# Loss (Geifman 2017 Eq.5):
L = mean(h(x) * CE(g(x), y_bin)) + lambda * max(0, c_target - mean(h(x))) ** 2
# c_target ≈ 30%  (经验:bottom 30% PnL-relevant 样本 trade)

# Inference:
def predict(x):
    if h(x) > 0.5:  # 接受
        return 0 if g(x)<0.5 else 2
    else:           # 弃权
        return 1
```

**为什么之前没尝试**: 80 个实验里所有 GBDT/NN 都把它当 3-class 问题, 用 argmax + threshold 弃权是 **post-hoc** 的; Selective Net 把"是否弃权"放到 **训练目标本身**, 让 g 只在被 select 的样本上 minimize loss → g 学到的是 **conditional on trade-worthy** 的 sharper decision boundary。

**预期**: LOSO **+2 ~ +4** (大头)
**时间**: 12 h (NN 实现); 或 6 h 如用 GBDT 模拟 (sample weight = h_pred from external classifier)
**红线**: ✅ 完全合规 — h(x) 仅依赖当前窗口
**与已有同构性**: 
  - T76 decoupled threshold 是 **post-hoc** 弃权阈值; **Selective Net 在训练阶段就联合优化 g, h** — 完全不同
  - T87 SPO+ 是 cost-aware loss; 但仍然是 single-head; SelNet 是 dual-head
  - T42 OvA binary 把 3-class 拆 OvA; SelNet 重写整个 problem formulation
**风险**: Geifman 原版用 NN; GBDT 实现需技巧 (用 sample weighting iteration). **Mitigation**: 
1. 先 NN 实现快速验证 (PyTorch 200 行); 若 +2+ 则转 GBDT 化
2. GBDT 化方法: alternating optimization. round 1 fit g with weight=1. round 2 train h(x) → P(|Δmid|>α). round 3 refit g with sample_weight=h_pred. 迭代 3 轮收敛.

### 7.2 Deep Ensembles + Predictive Entropy (Lakshminarayanan, Pritzel, Blundell 2017)
**原方法**：M 个独立训练 NN, predictive variance 估计 epistemic uncertainty。
**类比映射**: 已部分由 T11/T17/T46 ensemble 做了; 新 angle = **predictive entropy as gate**:
```python
preds = [m.predict_proba(x) for m in models]  # (M, 3)
mean_p = preds.mean(0)
H = -sum(mean_p * log(mean_p))  # entropy
trade = (H < tau_H)  # low entropy = high confidence
```
**预期**: +0.5 ~ +1
**时间**: 4 h
**红线**: ✅
**与 6.2 bootstrap CI 区分**: 6.2 用 std (二阶矩); 7.2 用 entropy (信息量度). Bootstrap 测 sample variance, ensemble entropy 测 epistemic uncertainty — 不同 source。

---

## 8. Information Theory / Compression

### 8.1 ⭐ Information Bottleneck via Sym-Adversarial GRL (Tishby 1999, Ganin & Lempitsky 2016) — Top 5 #2
**原方法**：features 应满足 max I(Z; Y) − β·I(Z; X) (max info about label, min about other variables)。Domain-Adversarial Neural Networks (DANN) 用 gradient reversal layer (GRL) 实现:
```
min_φ,h max_d   CE(h(φ(x)), y) − λ · CE(d(φ(x)), domain)
```
GRL 在反向时把 d 的梯度乘以 −λ 传给 φ → φ 学到 sym-invariant representation。

**类比映射**: ⭐ **直接攻击红线 ③** (sym-OOD generalization)。
- y 任务 = 3-class (or PnL regression)
- domain = sym ID (训练时已知, 测试时**不使用**)
- 期望: φ(x) 编码可预测 y 的所有信息, 但不可预测 sym → 自动 sym-invariant

**Plan（伪码）**:
```python
class SymInvariantNet(nn.Module):
    def __init__(self):
        self.φ = MLP(154 → 128)
        self.h = Linear(128 → 3)    # label classifier
        self.d = MLP(128 → 5)        # sym discriminator
    def forward(self, x, alpha=1.0):
        z = self.φ(x)
        y_logits = self.h(z)
        # Gradient reversal: forward identity, backward gradient × (-alpha)
        z_rev = GradientReversal.apply(z, alpha)
        sym_logits = self.d(z_rev)
        return y_logits, sym_logits

# Loss:
L = CE(y_logits, y_true) + lambda * CE(sym_logits, sym_true)
# alpha schedule: 0 → 1 over training (warmup, 防止 d 太强干扰 φ)

# Inference: 完全不用 sym, 也不用 d
def predict(x): return softmax(h(φ(x)))
```

**关键洞察**: T45 GroupDRO 失败的可能原因是 **它优化最差 group performance, 但不强制 invariance**; DANN/IB 直接强制 representation 不带 sym 信息, 是更强的约束。

**预期**: LOSO **+2 ~ +3**
**时间**: 12 h
**红线**: ✅ (sym 仅在训练时作 auxiliary signal, 测试不用)
**与 T45 区分**: GroupDRO 是 worst-case risk; IB-DANN 是 invariance penalty — 不同 inductive principle
**风险**: NN 历史弱. **Mitigation**: 
1. 把 φ 当 representation, 喂给 GBDT (与 #3.1 的 stage 2 思路一致)
2. λ sweep [0.01, 0.1, 1.0]; λ 太大 collapse representation

### 8.2 Variational Autoencoder Representation + GBDT (Kingma & Welling 2013)
**原方法**：encoder q(z|x), decoder p(x|z), prior p(z) = N(0, I); ELBO loss; latent z 是压缩表示。
**类比映射**：把 100×154 = 15400-D 窗口压缩到 64-D z, 喂给 GBDT。
**Plan**:
```python
encoder: x ∈ ℝ^{100×154} → (μ, log σ²) ∈ ℝ^{2×64}
z ~ N(μ, σ²)
decoder: z → x̂ ∈ ℝ^{100×154}
L = ‖x − x̂‖² + β · KL(N(μ,σ²) ‖ N(0,I))    # β-VAE, β=0.5

# 下游:
X_aug = concat([X_handcrafted_226, μ_test])  # 不用 sample, 用 mean
gbdt.fit(X_aug, y)
```
**预期**: +0.5 ~ +1.5
**时间**: 10 h
**红线**: ✅
**与 #3.1 SimCSE 区分**: SimCSE 是 contrastive (discriminative); VAE 是 generative; 经验上 contrastive 在 representation quality 通常 > generative — VAE 是 fallback option
**风险**: VAE 容易学到 mean-collapsed representation

### 8.3 MDL Feature Selection (Rissanen 1978)
**原方法**: model selection by min description length: L_data + L_model; 选 feature 子集 minimize MDL。
**类比映射**: 当前 226+ features, 估计哪些是 noise; MDL 视角下 noise feature 的 description length > 它的 predictive gain → 应被剔除。
**Plan**:
- 用 LightGBM L1 reg + monotone constraint 过滤 SHAP 接近 0 的特征
- 或: per-feature MDL = log(N) · num_split + L_residual; 选最小 K
**预期**: +0.3 ~ +0.8 (主要是去噪 / 加快训练)
**时间**: 6 h
**红线**: ✅
**与 T85 feature leakage audit 区分**: T85 是检查 leakage; MDL 是 statistical model selection — 互补
**风险**: 收益小

---

## 9. 与已有 80+ 实验去重检查表

| 跨域 idea | 是否同构 | 已有实验 | 区分点 |
|---|---|---|---|
| 1.1 DR-OPE | 部分重叠 | T62 unbiased thresh | DR 是 IPS 严格升级 |
| 1.2 CQL | **不重叠** | T60 (policy gradient) | T60 是 PG, CQL 是 value-based |
| 2.1 Mahalanobis | **不重叠** | (无) | 全新 input-domain OOD detection |
| 2.2 CORAL | 弱重叠 | T45 GroupDRO | 不同 DG principle |
| 3.1 SimCSE | **不重叠** | T44 (handcrafted features) | learned vs handcrafted representation |
| 3.2 BERT MLM | **不重叠** | (无) | 时序 MLM 全新 |
| 3.3 Curriculum | 弱重叠 | T12 (post-hoc stacking) | 训练时 transfer vs post-hoc ensemble |
| 4.1 DR (Tobin) | **同构 = T26** | T26 | ⚠️ 不算新 |
| 4.2 RL² latent | **不重叠** | (无) | 在窗内 infer regime |
| 5.1 Prototypical | 弱重叠 | T48 (cross-sym pooling) | T48 inference 不兼容; 本 idea 用 cluster |
| 5.2 Sym-LOB | 弱重叠 | T93/T94/T95/T97 (NN arch) | bid-ask 对称约束未用过 |
| 6.1 GARCH thresh | 部分重叠 | T76/T82/T83/T88 | input-domain vs score-domain thresh |
| 6.2 Bootstrap CI | **不重叠** | T11/T17/T46 (mean ens) | std as gate vs mean averaging |
| 7.1 Selective Net | **不重叠** ⭐ | T76 (post-hoc thresh) | 训练阶段联合优化 |
| 7.2 Deep ensembles | 弱重叠 | T11/T17/T46 | predictive entropy as gate (新 angle) |
| 8.1 IB-DANN | **不重叠** | T45 | invariance vs worst-case risk |
| 8.2 VAE | **不重叠** | (无) | generative representation |
| 8.3 MDL | 弱重叠 | T85 (leakage audit) | model selection 角度 |

**结论**: 13 个 idea 中, **8 个完全不重叠, 5 个弱重叠 (但带新 angle), 1 个完全同构 (4.1)**。Top 5 全部在"完全不重叠"集合。

---

## 10. 实施排程建议（剩余 3.5 天）

> 假设 1 天 = 16 工作 h, 一个 worker 同时跑 1 个 idea

### Day 1 (low risk, fast experiment)
- **Idea #5 Mahalanobis OOD** (8 h) — 快速验证, 无 NN 训练, 直接用现有 GBDT
- **Idea #6.2 Bootstrap CI gate** (6 h) — 复用现有 ensemble model

### Day 2 (high upside)
- **Idea #1 Selective Net** (12 h, NN simple version) — 用 NN 验证
  - 若 +2+ → Day 3 转 GBDT 化
  - 若 marginal → 转 #2 IB-DANN

### Day 3 (parallel: representation + IB)
- **Idea #2 IB-DANN with GRL** (12 h) 
- **Idea #4 Cross-sym Contrastive Pretrain** (16 h, 用一个 worker 单独跑)

### Day 3.5 (final ensemble)
- 把所有 work 的 idea 的 prediction 做 ensemble + selective gating
- Submit

### Backlog (若时间余 / 候选)
- Idea #1.2 CQL (16 h)
- Idea #4.2 RL² latent (12 h)
- Idea #5.2 Sym-LOB arch (10 h)

---

## 11. 风险控制与红线兜底

每个 idea 已逐项核对 3 红线 (✅ 标记)。再次列出常见 trap:

1. **Sym ID 泄漏**: 任何 idea 涉及 sym-conditional 操作的, 都用 cluster/regime/Mahalanobis 替代 sym ID
2. **跨调用 state**: encoder φ 都仅作用在当前窗口; 所有 Bootstrap CI / Mahalanobis stats 都是 train-time 算好后存为常量
3. **Date feature**: 所有 idea 都不用 date

特别需要测试 (提交前必做):
- 故意把 1/4 测试点 sym 改为 99: 所有 NN/GBDT/Mahalanobis pipeline 不报错且输出 valid

---

## 12. Top 5 最终推荐（按优先级）

> Ranking = 类比强度 (1-5) × 可行性 (1-5)

1. ⭐⭐⭐ **#7.1 Selective Prediction with Abstain** (5×5=25)
   - 把 abstain reframe 为训练目标, 这是 80 个实验里全部缺失的 paradigm shift
   - 12 h 实施, +2~+4 LOSO, 红线 0 风险
   - **必做**

2. ⭐⭐⭐ **#8.1 Information Bottleneck (DANN)** (5×4=20)
   - 直接攻击 sym-OOD 红线 ③, T45 失败后未尝试此 alternative
   - 12 h, +2~+3 LOSO

3. ⭐⭐ **#1.2 CQL** (4×4=16)
   - Offline RL 是 trade decision 的本质 framework, T60 PG 不 = CQL value-based
   - 16 h, +1~+3 LOSO

4. ⭐⭐ **#3.1 Cross-Sym Contrastive (SimCSE)** (4×4=16)
   - Representation learning 替代 handcrafted, learned features 上限更高
   - 16 h, +1~+3 LOSO

5. ⭐⭐ **#2.1 Mahalanobis OOD Gating** (3×5=15)
   - 简单, 快速可验证, 与现有 pipeline 高度兼容
   - 8 h, +1~+2 LOSO

**组合潜力**: Top 5 中 #1, #2, #5 都是 **gating mechanism**, 可正交叠加; #3, #4 是 representation, 可作为 #1 的 encoder backbone。一个**理想 final pipeline**:

```
[Cross-sym contrastive encoder φ (#4)]
        ↓ z (128-D)
[Concat with handcrafted 226-D]
        ↓
[GBDT classifier g + selective head h (#1)]    + IB-DANN aux loss (#2)
        ↓
[Mahalanobis gate (#5) on z]                    + Bootstrap CI gate (#6.2)
        ↓
ABSTAIN if gate fails, else argmax of g
```

预期组合 LOSO: **+45 ~ +50** (vs +40 baseline), 即 **+5 ~ +10**。

---

## 13. 终极陈述

本项目已尝试 80+ 实验, 全部停留在 "怎么把 GBDT 训得更准 / 用什么阈值过滤" 的层面。**关键缺口在于 problem reframing**: 这是一个带弃权的 cost-sensitive 决策问题, 而 medical imaging / RL bandits / domain-adversarial training 文献已经给出了成熟工具。Top 5 idea 都是把这种 **paradigm shift** 实例化的具体工程方案。

如果只能选 **1 个 idea 实施**, 推荐 **#7.1 Selective Prediction with Abstain** — 它是问题本质的 reframe, 不是又一个超参 sweep。
