# W_paradigm_architecture_blinded — 颠覆性 model architecture paradigm

**Worker**: Opus 4.7 xhigh, blinded research mode
**Date**: 2026-05-08
**Time spent**: ~50 min research + 30 min write-up
**Constraint guard**: respects CRITICAL_CONSTRAINTS.md §1 (`date`=0 at eval, no cross-call state, sym-agnostic)

---

## PART 1 — 题目理解：我们是不是选错了 paradigm？

### 重新审视题目本质

| 维度 | 原始描述 | Paradigm 含义 |
|---|---|---|
| **输入** | 5 sym × 100-tick × 154 dim | 单次 predict 只见 100 tick = ~5 min 数据 |
| **输出** | 5 horizon 3-class label (down/flat/up) | 但分类不直接 = PnL |
| **评分** | cumulative PnL，1bp double-side fee, max-over-h | 真实目标 = **expected utility** under fee threshold |
| **OOD** | sym 0-4 但可能含训练外股票 | 必须 sym-agnostic（function of LOB snapshot only） |
| **Shuffle** | 测试点顺序乱 | Predictor 是 **stateless function** |

### 两个根本性 paradigm 错配（hypothesis）

#### 错配 A：所有 model 都是 discriminative point predictor，**忽视决策结构**

**LightGBM/CatBoost 优化 L2/Huber/CE，NN 多数也是 CE**——它们学的是 `P(direction | snapshot)`。
但题目的最优策略不是 "predict direction"，而是：
> 在 5 个 horizon × {long, short, no-trade} 中，选 maximizes expected PnL after fee 的 (h, action)。

只有 T87 SPO+ NN 通过 SPO+ loss 显式 align loss 到 decision，它是唯一 NN-side win——这强烈提示 **decision-aware paradigm 是有效的方向**。但 T87 只改了 loss，**架构和输出空间还是普通 NN**。下一步应当 push 到 architecture/output level。

具体 paradigm 候选：
- **Distributional output**: 输出 quantile 而非 class prob → 直接计算 `P(|Δm| > 2bp+fee)` → 决策
- **Conformal / selective**: 把任何 base model 套上 calibrated selective trading layer → 在 fee 门槛下选择性交易
- **Hierarchical decision**: 显式分解 "should trade" × "which direction" 两阶段
- **RL agent**: 直接 optimize PnL，不走 supervised label

#### 错配 B：所有 model 都把 LOB 当 generic tabular/sequence，**忽视微结构 physics**

LOB dynamics 不是 arbitrary 的——它是 self-exciting point process (Hawkes)，是 Markov queueing system (queue-reactive)，midprice 在 fair value 周围 mean-revert (OU process)。
- LightGBM treats 154 dim as arbitrary tabular features
- GRU/Transformer treat 100-tick as generic sequence with no inductive bias
- 这些架构有 **strong learning power but zero microstructure prior**

**与之对比**：state-space filter 在 100-tick 内 fit fair value，输出 filter velocity；Hawkes 在 100-tick 内估计 6 类订单 intensity & excitation，输出 18 个 microstructure parameters。这些是 **inductive-biased low-dim summary**，自带 sym-agnostic 性质（physics 是 universal 的）。

### 100-tick 窗口对 paradigm 的关键约束

100 tick = ~5 min = **microstructure timescale**, not macro。这意味着：
- LSTM/Transformer 100 步 sequence learning 收益 marginal（信息论上窗口太短，稀疏 NN 反不如 hand-crafted physics features）
- ✅ 合适：state-space filter（少量参数，online estimation OK）、Hawkes（6 维 intensity 估计 OK）
- ❌ 不合适：foundation model fine-tuning（数据太少）、深 LSTM（pattern memorization 不 transfer 到 OOD sym）

### 架构判决

**最佳 paradigm 类**：`physics-grounded representation` → `decision-aware downstream` → `selective trading wrapper`。
当前 ensemble 是 `tabular features` → `discriminative model` → `threshold tuning`，留下了至少 3 层未触及的高 ROI 空间。

---

## PART 2 — 8 个颠覆性 paradigm

### Paradigm 1: Online Kalman / state-space LOB filter (in-window)

**描述**: 在 100-tick 窗口内跑 Kalman filter；state = latent fair value `p*_t`，observation = midprice + microprice + best bid/ask。输出 filter velocity `μ̂_T`、innovation magnitude、posterior variance。
**决策规则**: trade direction = sign(μ̂_T) when |μ̂_T| > c·√Var; horizon 选择基于 velocity scale。
**文献**:
- Adaptive model for security prices driven by latent values (Tandfonline QF 2022)
- Kalman filter as innovation/velocity signal (Quantreo 2024, Quantneuraledge 2026)
- Particle filter extension for nonlinear LOB (Adedimeji Medium 2026)

**为什么 fundamentally 不同**: 不是 ML，是 SDE inference。模型只有 ~5 个 noise covariance 参数，**可以 within-window 在线估计或用 grid search 离线 fit**。Sym-agnostic by construction（同一组 noise covariance 应用到任何 sym）。
**实施成本**: 1-2 day。CPU 推理 cheap (Cython KF 几十 us/window)。
**预估 platform 改进**: +3 to +6（直接 trade）；+2 to +4（作为 LightGBM feature）。
**风险**: Linear Gaussian 假设可能太弱；particle filter 替代更强但 inference cost 高 100x。

### Paradigm 2: Compound Hawkes / Queue-reactive process parameter regression

**描述**: 在 100-tick 窗口拟合 multivariate Hawkes process（6 event types: lb/la/mb/ma/cb/ca）。输出 baseline μ + excitation matrix α + decay β ≈ 18-30 参数。这些参数本身 **就是 prediction**：drift = μ + cumulative excitation。
**决策规则**: predicted future return ∝ (buy intensity – sell intensity) × time-to-horizon；direction = sign。
**文献**:
- General Compound Hawkes for Mid-Price (arxiv 2110.07075, MDPI 2020)
- Hawkes-COE BTC return sign forecasting (arxiv 2312.16190, 2024)
- Forecasting OFI using Hawkes (arxiv 2408.03594, 2024)
- Queue-reactive (Huang-Lehalle 2015 JASA, multivariate extension arxiv 2405.18594, 2024)

**为什么 fundamentally 不同**: 编码 self-exciting order arrival physics。输出是 **calibrated SDE 参数**，不是 learned function。Sym-agnostic 因为 self-excitation 机制是 universal microstructure 现象。
**实施成本**: 3-5 day（MLE/EM 需小心数值，但有现成 PyHawkes/tick 库）。
**预估 platform 改进**: +4 to +10。**特别 promising**: T87 SPO+ NN 的胜利说明 microstructure features 有 alpha；Hawkes 直接给出物理 grounded 18 features。
**风险**: 100 tick (300s) for Hawkes calibration 可能 noisy。Mitigation: prior pre-fit 离线，only adapt 快变 components in-window。

### Paradigm 3: Distributional / quantile regression with PnL-aware decision rule

**描述**: 弃 3-class CE，改 quantile regression 直接预测 `Δmidprice_h` 的 5 个 quantile（10/30/50/70/90）× 5 horizon = 25 outputs。Inference 时计算 `P(Δm_h > +2bp)` 和 `P(Δm_h < -2bp)`，选 (h, direction) maximizes E[PnL after fee]。
**决策规则**: argmax_{h, dir} E[PnL] = P_dir · E[|Δm| | trade] – 2bp · 1[trade]; abstain if all negative.
**文献**:
- Quantile NN for stock return distributions (arxiv 2408.07497, 2024)
- Quantile Regression with LLMs for prediction (ACL Findings 2025)
- Distributional RL for futures trading (arxiv 2501.04421, 2025)
- LightGBM `objective='quantile'` 原生支持

**为什么 fundamentally 不同**: 把 prediction 问题转化为 **expected utility maximization**。直接把 fee 结构集成到决策规则——T87 SPO+ 改 loss 已 +1.81，这个改 output space 应该 +更多。
**实施成本**: 2-3 day（LightGBM-Quantile sweep + decision rule + threshold tune）。
**预估 platform 改进**: +5 to +12。**最高 ROI 之一**。
**风险**: tail quantile (10/90) 估计 noisy（决策 边界恰在尾部）。Mitigation: monotone constraint，ordinal coupling，CRPS loss。

### Paradigm 4: Conformal selective trading wrapper

**描述**: 取任何已有 base model（LightGBM, T87 NN, ensemble），用 holdout calibration set 算 nonconformity score；eval 时只在 conformal prediction set 排除 "no-trade" 时交易。等价：trade only if `P(direction) > calibrated_threshold` 调到 target false-trade rate (FDR)。
**决策规则**: Mondrian conformal stratified by spread / volatility bucket（sym-agnostic stratification）。Trade when `1 - max(p_up, p_down) < α_calibrated`；abstain otherwise。
**文献**:
- Trading via Selective Classification (Chalkidis arxiv 2110.14914, ACM ICAIF 2021——核心 reference)
- Selective Conformal Risk Control (arxiv 2512.12844, 2025)
- Plugin estimators for selective classification with OOD detection (OpenReview DASh 2024)
- Conformal Prediction for risk management (BBVA 2024)

**为什么 fundamentally 不同**: 这是 **decision layer**，绕开 model 改进——把任何 53% directional model 转成 70%-on-traded model。和 2bp fee 结构 极度协同（不交易 = 不亏 fee）。
**实施成本**: 1-2 day。MAPIE / sklearn 现成。
**预估 platform 改进**: +5 to +15。**最高 expected gain per dev hour**。
**风险**: 校准集→eval 分布漂移（OOD sym）。Mitigation: Mondrian conformal stratified by within-window features（spread, vol），不依赖 sym ID。

### Paradigm 5: Mixture of Experts gated by within-window regime

**描述**: 3-5 个 experts（每个 LightGBM/小 NN）专精于 regime（low-vol/high-vol × narrow/wide spread × calm/active OFI）。Gate: small NN/decision tree on within-window stats（rolling vol, spread mean, OFI sign, intensity ratios）。**No sym ID input**——gating 完全依赖 within-window physics features。
**决策规则**: weighted mix or hard route; gate output是 P(regime | within-window stats)。
**文献**:
- Adaptive MoE for Volatility-Sensitive Forecasting (arxiv 2508.02686, 2025)
- Time-MoE foundation model (NeurIPS 2024)
- MoE survey for finance (preprints.org 2025.05.1603)
- Generative-discriminative HMM-SVM regime classifier (Springer 2025)

**为什么 fundamentally 不同**: 每个 regime 有 fundamentally 不同的 dynamics（small-tick vs large-tick；trending vs mean-reverting）。Single model is forced into ensemble averaging across heterogeneous regimes，dilution 严重。MoE 让每个 expert 专精。
**实施成本**: 4-6 day（regime labeling + experts + gate）。
**预估 platform 改进**: +3 to +8。
**风险**: 100-tick gating noisy；regime boundary fuzzy。Mitigation: soft gate + temperature; train end-to-end with PnL signal。

### Paradigm 6: Symbolic regression / formula alpha mining

**描述**: 用 PySR (genetic programming) 或 RL-based 公式搜索，自动 mine 50-200 个短 formula 形式 `f(features_t-k) > c → direction`。每个公式 ~5-10 ops，人 readable。Boost ensemble or majority vote。
**决策规则**: ensemble of formula predictions weighted by holdout PnL.
**文献**:
- PySR: high-performance symbolic regression (Cranmer review 2024, GitHub MilesCranmer/PySR)
- AlphaEval framework (arxiv 2508.13174, 2025)
- Tree-structured thoughts for alpha mining (arxiv 2508.16334, 2025)
- AlphaPortfolio / formula alpha lineage

**为什么 fundamentally 不同**: NO neural net, NO tree learner——纯 explicit interpretable expressions。**OOD generalization 期望 强**：每个公式编码 structural relationship (e.g. OFI / spread ratio)，不依赖 sym-specific pattern memorization。
**实施成本**: 5-7 day（mining 慢，~天级 compute）。
**预估 platform 改进**: +2 to +6 单独；+5 to +10 stacked with LightGBM as features。
**风险**: 公式 in-sample 过拟合；CV 设计要严谨（time-blocked + sym-blocked）。

### Paradigm 7: Two-stage hierarchical "trade-or-not then direction"

**描述**: Stage 1 = binary classifier `should_trade ∈ {0,1}` trained on `1[|Δm_h| > 2bp+fee]`。Stage 2 = directional classifier trained ONLY on examples Stage 1 says trade。Cascade thresholds。
**决策规则**: trade if Stage1.prob > τ1; direction if so = Stage2.argmax; abstain otherwise。
**文献**:
- Trading via Selective Classification (Chalkidis 2021——直接 reference)
- Selective classification with abstention (Mao MLR 2024)
- SelectLLM calibration coverage/risk balance (OpenReview 2025)

**为什么 fundamentally 不同**: 显式 factorize decision 问题 into "should I trade" × "which direction"——matches actual payoff structure。不像 3-class CE 把 abstention 当一个类（这把 size 信号丢了）。
**实施成本**: 1-2 day（两个 LightGBM）。
**预估 platform 改进**: +3 to +8。
**风险**: Stage 1 不能 throw away easy-direction examples；threshold tuning 微妙。Mitigation: joint training with pinball / focal loss。

### Paradigm 8: RL agent (PPO/SAC on PnL)

**描述**: 训 small policy net on 训练数据 trajectories。State = 100-tick LOB tensor; action = (none, long_h5, long_h10, …, short_h60); reward = realized PnL minus fee. PPO/SAC update。
**决策规则**: policy(state) directly outputs action distribution → sample / argmax.
**文献**:
- DRL for OFI-based HFT (ResearchGate 391292844, 2024)
- DRL market making with Hawkes LOB (arxiv 2207.09951)
- Self-rewarding DRL trading (MDPI Mathematics 12.4020, 2024)
- Bilevel RL with conservative TD ensemble (OpenReview zaDU4vMAUr)

**为什么 fundamentally 不同**: 直接 optimize platform's actual scoring。没 supervised label engineering——agent 学自己的 decision rule。理论上是最对齐目标的方法。
**实施成本**: 7-10 day（高 engineering cost：sim env、reward shaping、HP sweep）。
**预估 platform 改进**: -5 to +12（HIGH variance——RL is hard to tune；可能 worse than ensemble baseline）。
**风险**: sample inefficient；sim-eval gap；2bp fee 创造 reward sparsity（多数 transition 不交易，very slow learning）。

---

## PART 3 — TOP 3 推荐

### 排序标准
(a) ROI per dev hour，(b) low risk of platform regression，(c) compatibility with 现有 iter_017 ensemble。

### 🥇 #1: Conformal selective trading wrapper (Paradigm 4)

**Why #1**: ZERO 风险 to existing models（pure decision layer on top）。1-2 day 实施。直接打 fee leak（accuracy-focused models 在 flat 区域被 fee 慢性失血）。Compatible with current LightGBM + T87 SPO+ NN ensemble——无须重训。

**Implementation sketch**:
1. Hold out 15% of training as calibration set，stratified by within-window vol/spread bucket
2. 对 each (model, horizon) 算 nonconformity score `s = 1 - P(predicted_class)`
3. Eval: trade only if `1 - max(p_up, p_down) < q_α` where q_α 是 calibration set 上的 α-quantile
4. Tune α (~0.3-0.4 trade rate). Mondrian-stratify by within-window vol bucket（不用 sym）
5. Compose with max-over-h: among horizons that pass selective gate, pick max E[PnL] horizon

**Expected platform gain**: +5 to +15. **Single highest ROI move on the board.**

**Cross-paradigm note**: This is **orthogonal** to base architecture. 应该和 Paradigm 2 / 3 stack——先训 distributional model，再套 conformal layer。

### 🥈 #2: Distributional / quantile regression with PnL-aware decision rule (Paradigm 3)

**Why #2**: Generalizes T87 SPO+ insight (decision-aware loss → +1.81). 改 output space 而非只改 loss，理论上 gain 更大。Modest cost (2-3 day)。Compatible with LightGBM (`objective='quantile'`).

**Implementation sketch**:
1. Train 5 horizons × 5 quantiles (10/30/50/70/90) = 25 LightGBM models（or single multi-output NN with pinball loss）
2. At inference: 从 quantile estimate 算 approx `P(Δm_h > +2bp)` 和 `P(Δm_h < -2bp)` (linear interp between quantiles)
3. Decision: argmax_{h, dir} `P_dir × Ê[|Δm| | trade] – 2bp × 1[trade]`; abstain if all negative
4. Add monotonic constraint across quantiles to avoid crossing

**Expected platform gain**: +5 to +12.

**Risk mitigation**: tail quantile (10/90) 是 noisy 但决策恰在 tail。用 CRPS loss / monotone NN 缓解。或先用 5 quantile 起步，验证后再加密。

### 🥉 #3: Hawkes/Kalman microstructure features fusion (Paradigm 1+2 fused)

**Why #3**: Orthogonal to existing tabular features（physics features ≠ LOB raw features）。Sym-agnostic 天然成立。Microstructure features 已被 T87 NN 验证有 alpha；Kalman/Hawkes 系统化提取它们。

**Implementation sketch**:
1. Per 100-tick window 跑 Kalman: state-space (linear-Gaussian) on midprice + microprice → 输出 `velocity_T`, `innovation_T`, `residual_var`, `KF_velocity_normalized` (4 features)
2. Per 100-tick window 跑 lightweight Hawkes parameter fit (closed-form moments method, no MLE): 6 类 intensity ratios + 6 类 excitation magnitude ≈ 12 features (faster than full MLE，结果 noisy 但 unbiased)
3. 把 16 features 加到 LightGBM input。 Re-train ensemble。
4. Bonus: standalone Kalman-velocity baseline 当 sanity check（直接 sign(velocity_T) 当 prediction）。

**Expected platform gain**: +3 to +8.

**Risk**: Hawkes 100-tick noisy。如果 features 没贡献，至少 Kalman velocity 是稳的物理信号。

### 综合 stacking 策略

#### Phase 1（快胜，2-3 day）: Paradigm 4（conformal）单独套在现有 ensemble 上
- Risk-free，预期 +5 to +15 platform。
- Validate 后：iter_017 → iter_018_conformal

#### Phase 2（4-6 day）: Paradigm 3（distributional）替换或并联现有 LightGBM
- 在 conformal layer 之下，model 输出 quantile→PnL decision
- 累计预期 +8 to +20

#### Phase 3（5-7 day）: Paradigm 1+2 fused 作为 features 注入
- 物理 grounded features，OOD 鲁棒
- 累计预期 +10 to +25

**最终预期**: iter_017 baseline +28 → +38 to +48 platform。落到目标 top +40+ 区间。

---

## PART 4 — 现状 cross-check（已读 PROGRESS.md 后）

读取的 PROGRESS.md 是 project setup 级别（赛题+数据+ Predictor 契约 + 候选路径 A-D），**没披露实际实验路径**。这是 blinded research 想要的状态——下面只对比可见的官方/讲座 hint：

### 与官方/讲座建议的一致性

| PROGRESS.md 建议 | 我的 Top 3 | 一致性 |
|---|---|---|
| 「PnL > F0.5 > acc，trading 侧重 precision」 | Paradigm 4 (selective conformal) 直接打 precision/coverage tradeoff | ✅ **完全契合**（讲座原话支持选 #1） |
| 「α=0.1% 任务更有利可图，初期优先 label_60」 | Paradigm 3 (distributional) 自动按 horizon 选最大 E[PnL] horizon | ✅ 自动 align（model 自己挑 horizon） |
| 「类别极不均衡 label=1 占 60-80%」 | Paradigm 7 (trade-or-not) + #1 conformal 都明确处理 abstention | ✅ 一致 |
| 「测试集可能有训练外 sym」 | Paradigm 1+2+6 都是 sym-agnostic by physics/structure | ✅ 一致 |
| 「单次评测 ≤ 3 小时；模型 ≤ 2GB；FP32」 | 我的 Top 3 全部 CPU-friendly，inference 几 ms/window | ✅ 远在 budget 内 |

### 与 prompt 提到的现状（iter_017 = T87 SPO+ NN + LightGBM + 3 wins）对比

| Prompt 已知现状 | 我的提议 |
|---|---|
| LightGBM/CatBoost regression L2/Huber 已试 | Paradigm 3 升级 LightGBM 为 quantile output → orthogonal 改 output space |
| T87 SPO+ NN 唯一 NN-side win | Paradigm 3 + 4 都是 SPO+ insight 的延伸（output space + decision layer） |
| GRU/Transformer-style tabular fail | 我**没**推荐 Transformer/sequence DL——契合"sequence DL 没收益"的现状 |
| Multi-head NN 试过 | 没和 Top 3 冲突——Top 3 都不是 multi-head NN |

### 可能的隐藏冲突 (caveat)

下列项 PROGRESS.md 没披露但**可能**已尝试（无法 verify）：
- **Quantile regression** 可能已在某 experiments/ 试过——如果是的话，需聚焦 *PnL-aware 决策规则* 而非 quantile head 本身
- **Conformal/selective classification** 可能已被 threshold tuning 隐式做了——但 Mondrian conformal + within-window stratification 是显式版本
- **Hawkes parameters as features** 也许已用 simpler proxy（例如 OFI、imbalance）——但 Hawkes 是 self-exciting 系统的完整 parameterization，OFI 只是 first-moment proxy

### 结论：Top 3 是否还成立？

**Yes**——基于可见信息（PROGRESS.md + prompt 现状）：
- #1 Conformal 与官方"precision > recall"建议**直接 aligned**，是最低风险高 ROI 路径
- #2 Distributional 是 SPO+ insight 的**架构级延伸**（已知方向 valid）
- #3 Hawkes/Kalman features 是**目前 paradigm 没覆盖的 microstructure physics 方向**（无重复风险）

如果 PM 后续读了 experiments/ 发现 Top 3 中任一项已被试过：
- conformal 已试 → fall back 到 quantile + decision rule（#2）
- quantile 已试 → fall back 到 hierarchical (Paradigm 7) 或 Hawkes (#3)
- 都试过 → 推 RL agent (Paradigm 8) 或 symbolic regression (Paradigm 6)

---

## Sources

- [General Compound Hawkes Processes for Mid-Price Prediction (arxiv 2110.07075)](https://arxiv.org/abs/2110.07075)
- [Hawkes-based cryptocurrency forecasting via LOB data (arxiv 2312.16190)](https://arxiv.org/html/2312.16190v1)
- [Compound Hawkes Process LOB order sizes (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1544612324011863)
- [Forecasting OFI using Hawkes (arxiv 2408.03594, 2024)](https://arxiv.org/html/2408.03594v1)
- [Adaptive model for security prices driven by latent values (Quantitative Finance 2022)](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2021.2023753)
- [Kalman Filter Trading: Noise Reduction (Quantneuraledge 2026)](https://quantneuraledge.com/blog/kalman-filter-trading-noise-reduction)
- [Adaptive Market Intelligence MoE Volatility Forecasting (arxiv 2508.02686, 2025)](https://arxiv.org/abs/2508.02686)
- [MoE survey for business and finance (preprints 2025)](https://www.preprints.org/manuscript/202505.1603/v1/download)
- [Time-MoE Billion-Scale Time Series Foundation Models (NeurIPS 2024)](https://openreview.net/forum?id=e1wDDFmlVu)
- [Trading via Selective Classification (Chalkidis arxiv 2110.14914 / ACM ICAIF 2021)](https://arxiv.org/abs/2110.14914)
- [Selective Conformal Risk Control (arxiv 2512.12844, 2025)](https://arxiv.org/html/2512.12844v1)
- [Conformal Prediction Wikipedia](https://en.wikipedia.org/wiki/Conformal_prediction)
- [Quantile NN for stock return distributions (arxiv 2408.07497, 2024)](https://arxiv.org/html/2408.07497v2)
- [Distributional RL for futures trading (arxiv 2501.04421, 2025)](https://arxiv.org/html/2501.04421)
- [Plugin estimators for selective classification with OOD detection (OpenReview 2024)](https://openreview.net/forum?id=DASh78rJ7g)
- [PySR symbolic regression review (Cranmer 2024)](https://link.springer.com/article/10.1007/s10710-024-09503-4)
- [AlphaEval framework for formula alpha mining (arxiv 2508.13174, 2025)](https://arxiv.org/html/2508.13174v1)
- [Tree-structured thoughts for alpha mining (arxiv 2508.16334, 2025)](https://arxiv.org/html/2508.16334v1)
- [Queue-reactive model JASA (Huang-Lehalle 2015)](https://arxiv.org/abs/1312.0563)
- [Multi-Level OFI in LOB (Cont et al ora.ox.ac.uk)](https://ora.ox.ac.uk/objects/uuid:9b7d0422-4ef1-48e7-a2d4-4eaa8a0a7ec1/files/m89dedb16194e627a2c92d14e3329bd48)
- [Smart Predict-then-Optimize SPO+ (Elmachtoub 1710.08005)](https://arxiv.org/abs/1710.08005)
- [Decision-Focused Learning Foundations (CPAIOR 2024)](https://people.cs.kuleuven.be/~tias.guns/files/CPAIOR24-keynote-DFL.pdf)
- [Online Decision-Focused Learning (arxiv 2505.13564, 2025)](https://arxiv.org/html/2505.13564v3)
- [DRL OFI-based HFT (ResearchGate 391292844, 2024)](https://www.researchgate.net/publication/391292844_Deep_Reinforcement_Learning_for_Optimizing_Order_Book_Imbalance-Based_High-Frequency_Trading_Strategies)
- [DRL market making with Hawkes LOB (arxiv 2207.09951)](https://ideas.repec.org/p/arx/papers/2207.09951.html)
- [Deep LOB forecasting microstructural guide (PMC12315853, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- [HLOB Information persistence in LOBs (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0957417424029452)
