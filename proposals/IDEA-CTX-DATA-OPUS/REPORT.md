# IDEA-CTX-DATA-OPUS — 数据/监督信号/训练目标 Brainstorm

> 生成日期：2026-05-08
> 评测截止：2026-05-11
> Baseline：iter_014 = 5-seed LightGBM regression on Δmid_norm，feat_dim=359，**LOSO-equiv +36.22**
> 立场：first-principles，尽量不重复 T1-T98 已尝试过的方案

---

## 0. 核心矛盾（必须先讲清楚）

当前 iter_014 的训练监督信号有 **4 个结构性问题**，决定了下一步突破的方向：

| 问题 | 数据证据 | 含义 |
|---|---|---|
| **target ≠ scoring metric** | y = Δmid_norm, but PnL = signed·Δmid - 2·fee·\|side\|；fee 项被丢进 inference threshold 而非进 loss | 模型在 fee 噪声带 \|δ\|<2·fee 内还在学，但这部分对 PnL 评分**永远是 0**——模型在浪费容量学不可能盈利的样本 |
| **noise-band 贡献大量梯度** | fee=1bp，单边\|δ\|>2·fee 的样本约 35-50%；剩余 50-65% 是无效梯度 | T92 八个 magnitude weight 变体都失败，原因可能是它们**不够极端**——线性/sqrt 衰减仍给噪声带 nonzero 权重；fee-clip 才是硬截断 |
| **uncertainty 没建模** | 49.6% 错过的盈利信号是"模型不敢开仓"；3.6% capture rate；但模型只输出 point estimate，没有 σ | model 不知道自己什么时候 confident vs 模糊；门限只能用绝对值大小判断"敢不敢开仓"，没用上分布信息 |
| **未标注数据全废** | LOSO 训练只用 4 sym × 96 days 数据 ≈ 1.5M ticks；剩 0.9M ticks (held-out sym + val/test region) 完全未参与训练 | SSL pre-training 是合规的（无 label，无 future leak），但项目历史上没尝试过——这是显著的"免费数据" |

**结论**：下一个 +2 不来自更多 features 或更复杂的 NN 架构，来自**让模型学的目标更接近 PnL，并且让模型知道自己的 uncertainty**。

---

## 1. Idea A1：Fee-Truncated PnL Target（Angle A，深论证）⭐⭐⭐⭐⭐

### 1.1 数学推导（why this is the "right" target）

平台 PnL 公式（已知）：
```
PnL(t) = (side · (mp_{t+60} - mp_t) - fee · |side| · ((mp_{t+60}+1) + (mp_t+1))) / (mp_t + 1)
```

设 δ = (mp_{t+60} - mp_t) / (mp_t + 1) = Δmid_norm（当前 iter_014 的 target）。
近似 ε = ((mp_{t+60}+1) + (mp_t+1)) / (mp_t+1) ≈ 2（因为 mp_{t+60} ≈ mp_t，相对小）。

则 PnL(t) ≈ side · δ - 2·fee · |side|。

**Optimal achievable PnL given know-future-δ**:
```
PnL*(t) = max(0, |δ(t)| - 2·fee)   ≥ 0
side*(t) = sign(δ(t)) · 𝟙{|δ(t)| > 2·fee}
```

**新 target（signed fee-truncated PnL）**:
```
y* = sign(δ) · max(0, |δ| - 2·fee)
```

性质：
1. 在 |δ| ≤ 2·fee 时 y* = 0（噪声带强制零）
2. 在 |δ| > 2·fee 时 y* = sign(δ)·(|δ| - 2·fee)（线性 above threshold）
3. 连续，几乎处处可微
4. ∂L/∂ŷ 在噪声带恒为 0（MSE → 当 y=0 且 ŷ=0 时梯度=0）

### 1.2 为什么这是颠覆而非 incremental

| | **iter_014（现状）** | **A1（fee-truncated）** |
|---|---|---|
| Target | y = δ | y = sign(δ)·max(0, \|δ\|-2·fee) |
| Noise-band 行为 | 模型还在学预测一个微小数值 | 模型被强制学预测 0 |
| Gradient on \|δ\|<2·fee | nonzero（O(δ)） | **exactly 0** |
| 模型容量分配 | 50-65% 容量花在不可盈利样本 | 100% 容量花在可盈利样本 |
| Inference threshold | τ ≈ 2·fee（DE-tuned） | τ ≈ 0（target 已经吸收 fee） |

**vs T87 SPO+**: SPO+ 是 PnL-aware **loss** surrogate，target 仍是 δ。A1 改的是 **target** —— 是 prediction 本身的几何。两者**正交**，可叠加：fee-truncated target + SPO+ surrogate loss。

**vs T92 (failed magnitude weights)**: T92 用 |δ|、sqrt(|δ|) 等当 sample weight，但 target 不变。模型仍被要求"在噪声带预测 δ"，只是这部分 loss 被加权降低。A1 直接让 target 在噪声带 = 0，**模型从根本上不被要求学噪声**。这是 sample-weight 与 target-geometry 的本质区别。

**vs iter_014 baseline**: 唯一不同就是 y 的定义。一个 `y = np.sign(δ) * np.maximum(np.abs(δ) - 2*fee, 0)` 替换。

### 1.3 实施 plan（极轻量）

1. 在现有 T75/iter_014 训练 pipeline 中替换 target 计算（5 行代码改动）
2. LightGBM regression L2 loss 不变
3. Inference threshold gate：原来 τ ≈ 2·fee；新 τ 重新 DE-tune（应该接近 0 或略 > 0）
4. 5-seed LightGBM 重训，跑 LOSO 5-fold

**时间**：4-6 小时（包括 threshold re-tune）。

### 1.4 Upside 估计

- **Best case**: +3 到 +5 LOSO（参考 T87 SPO+ +1.81 是 loss-side surrogate；target-side 改动应至少同等量级，且**正交可叠加**）
- **Conservative**: +1 到 +2 LOSO
- **Worst case**: 0 to -0.5（如果 fold 0/3 噪声带样本本身有微弱信号，硬截断会丢掉）

### 1.5 风险 / 红线检查

- ✅ 不用 date：target 计算只用 mp_t、mp_{t+60}
- ✅ 无跨调用 state：训练阶段计算 y，inference 阶段只输出 prediction
- ✅ Sym-agnostic：fee 全平台一致，与 sym 无关
- ⚠️ **fold 0/3 加重**：fold 0/3 已经 80% no-trade，加 fee-truncate 后可能 90%+ y=0，导致回归任务退化为"几乎全是 0"
  - **缓解 1**：与 B2（drop subset of zero rows）组合
  - **缓解 2**：用 Tweedie 回归（zero-inflated regression）替代 L2
- ⚠️ 不再叠加 T31 augmentation 的 [0.8, 1.2] scaling：augmented y 也要重新算 fee-truncate

---

## 2. Idea B1：Fee-Clipped Sample Weight（Angle B，PM 已建议，深论证）⭐⭐⭐⭐

### 2.1 公式
```
w(t) = max(0, |δ(t)| - 2·fee)
```

target 仍是 δ（不像 A1 改 target）。

### 2.2 为什么 T92 8 个变体失败但 B1 可能成功

T92 测过 linear (|δ|)、sqrt、sqrt_offset、log、clipped 等。**关键缺陷**：所有变体都给噪声带 nonzero 权重，只是衰减不同。
- linear: w = |δ|，噪声带 w 仍 > 0
- sqrt: w = sqrt(|δ|)，更慢衰减，噪声带 w 更大
- log: w = log(1 + |δ|)，log(1+ε) ≈ ε，噪声带 w ≈ |δ|
- clipped: w = min(|δ|, c)，仍 nonzero

**B1 是 ReLU**：硬零截断在噪声带。理论上：
- 噪声样本的 σ²(δ|x) 与 噪声样本的 |δ| 是**正相关**（小信号 = 易被噪声主导）。给小 |δ| 高权重 = 学噪声分布。
- 给小 |δ| 零权重 = 模型只对**清晰信号**学习，参数反映"什么 LOB 状态产生大 drift"。

### 2.3 vs A1 的区别（重要）

| | **A1 (truncated target)** | **B1 (clipped weight)** |
|---|---|---|
| Loss | (ŷ - sign(δ)·max(0,\|δ\|-2·fee))² | max(0,\|δ\|-2·fee) · (ŷ - δ)² |
| 模型学什么 | E[fee-truncated PnL \| x] | E[δ \| x] (但只在 \|δ\|>2·fee 区域) |
| ŷ 的尺度 | shrunk（噪声带预测 0） | 接近 δ scale |
| Noise-band 预测 | 趋近 0 | 任意（loss 0，gradient 0） |
| Inference threshold | τ ≈ 0 | τ ≈ 2·fee |

**A1+B1 组合**：既改 target 又改 weight。loss = w·(ŷ - y_truncated)²。极端 noise rejection。这是 **double-trunc**，hypothesis：模型容量、loss landscape、threshold gate 三者一致 fee-aware。

### 2.4 实施 plan
- 1 行代码改动：在 LightGBM Dataset 里加 `weight=np.maximum(np.abs(delta) - 2*fee, 0)`
- 5-seed 重训
- LOSO 5-fold

**时间**：3-5 小时。

### 2.5 Upside
- **Single B1**: +0.5 to +2.5 LOSO
- **A1 + B1 combo**: +1.5 to +4 LOSO

### 2.6 风险
- ✅ 红线：与 baseline 完全一致，无新风险
- ⚠️ 信号样本被进一步稀释（fold 0/3 严重），可能 underfit

---

## 3. Idea C1：Masked LOB Self-Supervised Pre-training（Angle C，深论证）⭐⭐⭐⭐

### 3.1 数据规模 audit

| 数据集 | 样本数（约） | 有 supervision | LOSO 训练用了 |
|---|---|---|---|
| 5 sym × 120 days × 2 sessions × 2001 ticks | 2.4M | 是（labels 全有） | - |
| 4 sym × 96 days × 2 sessions × 2001 ticks (LOSO train) | 1.5M | 是 | ✅ |
| Held-out 1 sym × 96 days × 2 sessions × 2001 ticks | 384k | 是（但 LOSO 不允许用 label） | ❌ |
| 5 sym × 24 val/test days × 2 sessions × 2001 ticks | 480k | 是（但 OOD test) | ❌ |

**关键洞察**：384k held-out sym ticks 的 **features** 与 LOSO 训练 sym 的 features 来自**同一分布**（5 只股票），但 label 不能用。**SSL 不需要 label，只用 features**——所以这 384k 是合规的 free data。

更狠：**生产提交时用全 5 sym 数据 SSL 完全合法**——平台只要求 sym-agnostic inference，没限制训练数据来源。

### 3.2 SSL 任务设计

**Task A：Masked Feature Reconstruction (MAE-style)**
```
Input: 100-tick × 154-feature 窗口
Mask: 随机 mask 30% 的 (tick, feature) cells (设为均值 0)
Target: 预测被 mask 的原始值
Loss: MSE on masked positions
```

Encoder：6-layer Transformer / CNN-MAE，hidden=128，~1M parameters。

**Task B：Next-Tick Forecasting (auxiliary)**
```
Input: ticks t-99 to t (100 ticks)
Target: features at t+1...t+5 (5 ticks ahead)
Loss: MSE on next 5 ticks
```

捕获时间动力学（与 single-tick masked reconstruction 互补）。

### 3.3 Downstream 集成

**两种用法**：
1. **Embedding-as-feature**：SSL encoder 输出 128-d 全局 embedding per 100-tick window，concat 到原 359-d hand features → 487-d，重训 LightGBM
2. **End-to-end NN downstream**：SSL encoder 冻结 / 部分解冻，加 regression head，end-to-end fine-tune 在 supervised data 上（与 T81 NN regression baseline 对比）

推荐 **(1) embedding-as-feature**——保留 LightGBM inductive bias，最小破坏现有 stack。

### 3.4 为什么这是颠覆

- 项目历史 **从未做过 SSL**（T1, T19, T32, T81, T93-T97 都是 supervised NN）
- 384k 未标注数据被丢弃；用 SSL 后**多 25% 训练数据**
- SSL backbone 学到的 representation 比 hand features 更"抽象"（捕捉 LOB 高阶结构如 microstructural patterns）
- T81 NN/LGB 0.77 cross-corr → diversity 不够；SSL backbone 与 LGB 应该相关性更低（不同 inductive bias）

### 3.5 实施 plan

阶段 1（4 hours）: Encoder 架构设计 + masked reconstruction 训练 → SSL ckpt
阶段 2（2 hours）: 用 SSL encoder 在所有 sym/days 上 forward，提取 128-d embedding
阶段 3（3 hours）: 重训 LightGBM regression on 359+128=487-d，5-seed
阶段 4（1 hour）: LOSO eval + threshold tune

**时间**：10-12 小时（两天工作量）。

### 3.6 Upside
- **Best**: +2 to +5 LOSO（如果 SSL embedding 显著降低 NN/LGB cross-corr 到 < 0.5）
- **Realistic**: +0.5 to +2 LOSO（tabular SSL 历史成绩混合）
- **Worst**: 0（embedding 无信息或冗余）

### 3.7 风险 / 红线
- ✅ Date：SSL 不用 date
- ✅ Stateless：encoder forward 是 pure function，每次 predict 独立
- ✅ Sym-agnostic：SSL 不用 sym ID 输入
- ⚠️ Encoder 大小：~1M params × 4 bytes = 4MB，加上 LightGBM 总在 2GB 限制内
- ⚠️ **LOSO 模拟严谨性**：为了 OOF metric 不污染，SSL 也应只在 LOSO train sym 上训。但生产提交可用全 5 sym SSL → 与 LOSO eval gap 会大。需要双轨 evaluation。

---

## 4. Idea G3：Heteroscedastic Regression + Sharpe Gate（Angle G，深论证）⭐⭐⭐⭐

### 4.1 动机

iter_014 输出 point estimate ŷ。但模型对不同样本的 uncertainty 是不同的——high-vol regime 时 |δ| 大但 noise 也大；low-vol regime 时 |δ| 小但 signal-to-noise 高。**point estimate 不够**。

T86 (quantile dual-gate) 用 q=0.3, q=0.7 两个 quantile 间接估 σ；mid = (q30+q70)/2。但 q30/q70 间距是固定 confidence interval；不直接是 σ。

**Heteroscedastic NLL**: 直接预测 (μ, log σ²)：
```
L(y, μ, σ) = (y - μ)² / (2σ²) + log σ + const
```

在 LightGBM 中通过 custom objective 实现：
- 两个 LightGBM models 串联：M1 fit μ on δ；M2 fit log_var on residuals² (after M1)
- 或 NN 双头版本（T81 已有 NN 框架，加一个 σ head）

### 4.2 推理时的 Sharpe Gate

```
score = ŷ_μ / ŷ_σ
trade if |score| > τ
```

**为什么 score = μ/σ 比 score = μ 好**:
- 同样 |μ|=0.1bp，σ=0.01 时 score=10（高 Sharpe，敢开仓）
- 同样 |μ|=0.1bp，σ=0.5 时 score=0.2（噪声主导，不开仓）

这直接对应"开仓置信度"，与 PnL（确定性）vs 期望 PnL（不确定）的差别。

### 4.3 为什么这能解决 49.6% 错过

49.6% 错过样本里：
- 一部分是真盈利但模型 |ŷ| 小 → 模型不敢开仓
- 但其中可能有 |ŷ| 中等 + σ 也小 → high-Sharpe → 应该开仓但 |ŷ| 不够过门限

T86 dual-gate 部分捕获了这点（IQR-mid +1.94），但是**一阶近似**。Heteroscedastic 是**二阶**：每个 sample 独立 σ。

### 4.4 实施 plan

**两阶段 LightGBM**:
1. Train M1 = LGB regression on δ (= iter_014 baseline)
2. Compute residuals r(t) = δ(t) - M1(x(t)) (OOF predictions)
3. Train M2 = LGB regression on log(r² + ε) (with floor ε=1e-8)
4. At inference: ŷ_μ = M1(x), ŷ_σ = sqrt(exp(M2(x))), score = ŷ_μ / ŷ_σ
5. DE-search τ on OOF Sharpe scores

**时间**：6-8 小时（M1 即 iter_014 baseline；M2 是新增）。

### 4.5 Upside
- **Best**: +2 to +4 LOSO（如果 σ 估计准且与 fold 0/3 噪声相关性高）
- **Realistic**: +0.5 to +2 LOSO

### 4.6 风险
- ⚠️ Two-stage 失败模式：M2 训练数据 r² 是 OOF 残差，分布很 noisy，M2 容易过拟合或 underfit
- ⚠️ ŷ_σ → 0 时 score 爆炸；需要 ŷ_σ 下界 clip
- ✅ 红线：所有计算都在 single-window，无跨调用 state

---

## 5. Idea A3：Sign-Magnitude Decoupled Heads（Angle A，深论证）⭐⭐⭐

### 5.1 核心 hypothesis

**方向预测和幅度预测是两个不同的任务，需要不同的归纳偏好和不同的训练数据子集。**

- **方向（direction）**：sign(δ)。在 |δ| < 2·fee 时方向几乎是随机噪声（无信号）。所以训练方向时**只用 |δ| > 2·fee 的样本**。
- **幅度（magnitude）**：|δ|。所有样本都有意义（vol regime 的指示）。

### 5.2 模型架构（双 head）

```
M_dir: 3-class LightGBM classifier on {-1, 0, +1} 
       trained on |δ| > 2·fee subset only (~40% of data)
M_mag: regression LightGBM on |δ| (or log(1 + |δ|/fee))
       trained on all data
```

### 5.3 推理时融合

```
score = (P_dir(+) - P_dir(-)) · ŷ_mag
trade if |score| > τ
side = sign(score)
```

P_dir(+) - P_dir(-) ∈ [-1, +1] 是方向 confidence；乘以 magnitude 得 expected signed magnitude。

### 5.4 为什么 ≠ T72 binary cascade

T72：cascade（先 binary trade-or-not，再 binary buy-or-sell），sequential。
A3：parallel decoupling，sub-tasks **独立训练，独立优化**。
原理：cascade 上游错误传给下游；parallel 各自最优。

### 5.5 实施 plan

1. 准备两个数据集：
   - D_dir = {(x, sign(δ)) : |δ| > 2·fee}
   - D_mag = {(x, |δ|) : all}
2. Train M_dir as 3-class LGB（label 3 类很少 because |δ|<2·fee 已被过滤）—— wait, 实际上 |δ|>2·fee 时 sign(δ) ∈ {-1, +1}，所以是 **2-class**
3. Train M_mag as regression
4. Inference fusion + DE threshold

**时间**：5-6 小时。

### 5.6 Upside
- **Best**: +1 to +3
- **Realistic**: 0 to +1.5

### 5.7 风险
- ⚠️ M_dir 训练样本变少（~40%），泛化可能更差
- ⚠️ 复合误差：P_dir × ŷ_mag，两个误差相乘
- ✅ 红线无问题

---

## 6. Idea D3 / 待讨论：Multi-Target Joint Regression（Angle D）⭐⭐⭐

LightGBM XGBoost 现在支持 multi-output。同时回归 K 个目标：

```
y = [δ_60, σ_60_intra, |δ_max| over [t,t+60], time_to_max_drift, OFI_60_realized]
```

5 个 head 共享 boosting splits，但叶子值 K-d。

**Why useful**：辅助 head 强制 backbone 学更"全面"的 LOB 表征——不只是 mean drift，还包括 intra-window vol、drawdown、信息流等。这种多任务正则在 NN 上常见，LGB 上罕见。

**Implementation**：用 XGBoost `multi_strategy="multi_output_tree"` (XGB 2.0+)；或自己写 chained loss。

**Time**: 6-8h. **Upside**: +0.5 to +2.

风险：multi-output GBDT 不成熟，可能比 single-output 差。

---

## 7. Idea H1 / 待讨论：Curriculum Learning by Signal Quality（Angle H）⭐⭐⭐

阶段化 LightGBM 训练（warm-start）：

```
Stage 1 (rounds 0-200): train on |δ| > 4·fee subset (~25%) → M1
Stage 2 (rounds 200-500): warm-start from M1, train on |δ| > 2·fee (~40%) → M2
Stage 3 (rounds 500-1000): warm-start from M2, train on all data (with fee-clip weight) → M3
```

**与 T92 静态权重的区别**：T92 是单次训练；H1 是 sequential。学习顺序：先建立"clean signal 形态"，再泛化。

**Implementation**：LightGBM `init_model` 参数支持 warm-start。

**Time**: 5-7h. **Upside**: +0.3 to +1.5. 

风险：Stage 1 数据少（25%），可能 overfit 到 stage 1 子集模式。

---

## 8. Idea E1 / 待讨论：Velocity Auxiliary Targets（Angle E）⭐⭐

辅助 target：drift 的 time derivative。

```
v(t) = δ(t, h=10) - δ(t, h=5)  // drift acceleration
a(t) = (v(t+5) - v(t))         // drift jerk
```

Multi-task：joint regression of (δ_60, v, a)，但只用 δ_60 推理。

**为什么**：模型被强制理解"drift 是怎么来的"——不仅 magnitude，还有 time-progression。

**Time**: 5h. **Upside**: +0.3 to +1.

风险：辅助 target 在 GBDT 上正则化效果小（GBDT 不像 NN 那样需要 implicit regularization）。

---

## 9. Top 5 推荐（按 upside × 可行性 / 时间 排序）

| # | Idea | Angle | Target/data 改动 | Upside (LOSO) | 时间 | 重训 | 红线 |
|---|---|---|---|---|---|---|---|
| **1** | **A1+B1 combo: fee-truncated target + fee-clipped weight** | A+B | y = sign(δ)·max(0,\|δ\|-2·fee), w = max(0,\|δ\|-2·fee) | **+1.5 to +4** | 6-8h | 5-seed | ✅ |
| **2** | **G3: Heteroscedastic regression + Sharpe gate** | G | 加 M2 fit residual variance, score = μ/σ | +1 to +3 | 6-8h | 2-stage | ✅ |
| **3** | **C1: Masked LOB SSL pre-training** | C | 加 SSL encoder, 384k unlabeled data 入场, embedding 当 feature | +1 to +5（高方差） | 10-12h | 5-seed + SSL | ⚠️ 需 LOSO/prod 双轨 |
| **4** | **A3: Sign-magnitude decoupled heads** | A | M_dir on \|δ\|>2·fee subset + M_mag on all, multiplicative fusion | +0.5 to +2 | 5-6h | 双模型 | ✅ |
| **5** | **A4 hybrid: best-PnL-reachability auxiliary regression** | A+D | 加 aux head 预测 max-PnL across 5 horizons, 当 confidence filter | +0.5 to +2 | 6h | 重训 | ✅ |

### 推荐执行顺序

**Day 1 (今日剩余)**: A1+B1 combo（最便宜的"PnL-aware target"改动；与现 pipeline 兼容；同时跑 G3 二阶段）
**Day 2**: A1+B1 + G3 结果分析；启动 C1 SSL pre-training（长任务）
**Day 3**: C1 集成 + A3 双 head 实验
**Day 4**: 最佳模型集成 + 提交

---

## 10. 明确否决（Rejected）

| Idea | 否决原因 |
|---|---|
| **Test-time fine-tuning / online adaptation** | 用户明确 ban，违反红线 2（stateless inference） |
| **Per-sym normalization / sym embedding** | 违反红线 3（sym-agnostic） |
| **Date-conditioned features (周一/周五效应等)** | 违反红线 1（date 置 0） |
| **T92-style static magnitude weights (linear/sqrt/log/clipped variants)** | 8 个变体已实验，全部失败；fee-clipped 是 ReLU 硬截断，是不同 family |
| **KNN-target retrieval (FAISS)** | IDEA-CONTRARIAN A 已覆盖；不属于 data/supervision angle |
| **Influence-function reweighting (Pruthi 2020)** | 计算复杂度高（per-sample Hessian-free LiSSA），4 天内 ROI 太低 |
| **Active learning (offline)** | 数据集已固定，"主动选样本"和静态 weighting 等价，不是新方法 |
| **MAML / Prototypical Networks for sym adaptation** | 违反 stateless inference（无法 test-time fine-tune） |
| **GRL (Gradient Reversal) sym adversarial** | 是 architecture angle，不是 data/supervision；且 IDEA-CONTRARIAN H 已覆盖 |
| **Pseudo-labeling on held-out sym** | T65 已尝试，效果有限；本提案的 C1 SSL 是 superior strategy |
| **Frequency-domain (FFT) auxiliary targets** | 实施复杂度高，理论 upside 小（HFT 周期性弱） |

---

## 11. 关键 take-aways

1. **当前最大的"免费午餐"是 A1+B1**：4 行代码、6 小时、显式对齐 fee；这应该是首选。
2. **G3 heteroscedastic 是被低估的方向**：T86 dual-gate 已证明 uncertainty signal 有 +1.94 收益；G3 是更彻底的版本。
3. **C1 SSL 是项目历史空白**：384k 未标注数据被浪费；SSL 是合规且 unprecedented 的尝试。
4. **A1 与 G3 完全正交，可叠加**；A1+B1+G3 组合理论上线 +5 LOSO。
5. **T92 的失败教训关键**：sample weight 必须是**ReLU 硬截断**才有意义；线性/sqrt 衰减全是错觉。fee-clipped 是 ReLU，T92 测试的全是 smooth 函数。

---

## 12. 数学推导备份（供 worker 实现时参考）

### 12.1 Fee-truncated target 的梯度

L = (ŷ - y_truncated)²
y_truncated = sign(δ)·max(0, |δ|-2·fee)

∂L/∂ŷ = 2·(ŷ - y_truncated)

当 |δ| ≤ 2·fee 时 y_truncated = 0，所以 ∂L/∂ŷ = 2·ŷ
- 这鼓励模型在噪声带预测 ŷ → 0
- 但模型并没有被罚"如果预测了 ε > 0"——它只是被告诉"y 在这是 0"

当 |δ| > 2·fee 时，∂L/∂ŷ = 2·(ŷ - sign(δ)·(|δ|-2·fee))
- 模型学预测 fee-truncated PnL

### 12.2 Heteroscedastic NLL 的两阶段实现

Stage 1（与 iter_014 一致）：
```
M1 = LGB(L=L2, target=δ)
ŷ_μ(x) = M1(x)
```

Stage 2（新增）：
```
r(x) = δ(x) - ŷ_μ_oof(x)  # OOF residual (避免 in-sample overfit)
log_var_target(x) = log(r(x)² + 1e-8)
M2 = LGB(L=L2, target=log_var_target)
ŷ_σ(x) = sqrt(exp(M2(x)))
```

Inference score：
```
sharpe(x) = ŷ_μ(x) / max(ŷ_σ(x), σ_floor)  # σ_floor = 5e-4 防爆
trade if |sharpe| > τ_sharpe
```

τ_sharpe 用 DE 在 OOF Sharpe scores 上搜，与原 iter_014 的 τ_abs 比较 LOSO PnL。

### 12.3 SSL encoder 的输入/输出形状

Input: (B, 100, 154)  # 100 ticks, 154 raw features
Mask: 30% of (tick, feat) cells, replace with [MASK] embedding (= mean over training)
Encoder: Transformer 6L × 256d → output (B, 100, 256)
Pooling: mean over 100 → (B, 256)
Reconstruction head (training only): (B, 100, 154) ← unmask predictions
Loss: MSE on masked positions only

Downstream embedding extractor: encoder + mean pool → 256-d feature per window
LightGBM: input = 359 hand + 256 SSL = 615-d
