# T114 — External Literature & Blog Scan (online)

> **Worker**: opus xhigh (T114), 2026-05-08
> **Task**: 配合 T110-T113 in-house experiment readers，**重点读外部资料**，覆盖 4 axes (feature / tree models / execution / target loss) — 找出 in-house 没试过的"黄金"。
> **Constraints baseline**: Liangwenbei = 5 sym × 100-tick window × 154→226→359-d LOB+OFI features × 5 horizons (5/10/20/40/60 ticks @ 3s/tick) × PnL scoring × sym-OOD test, **stateless across batches**. iter_016 v3 (current SOTA): +44.71 LOSO (4-way ensemble too heavy).
> **Scope**: 外部 web search / WebFetch only — 不重读 R31/R33/SOTA-SURVEY 已覆盖的内容（去重）。

---

## 0. TL;DR — 8 Gold Tricks Not Yet Tried

| # | Axis | Trick | Source | LOSO Δ est. | Cost |
|---|---|---|---|---|---|
| 🥇 1 | feature | **Hawkes-OFI as engineered feature** (forecasted intensity) | Anantha 2408.03594 | +1.0~+2.5 | 1.5d |
| 🥇 2 | feature | **Pairwise imbalance over ALL price pairs** (not just consecutive); + **stock-level "global features" cached** = empirical quantile per sym computed once at train, frozen at inference | hyd 1st (Optiver Close) + fan2goa1 blog | +1.0~+2.0 | 1d |
| 🥈 3 | tree | **SketchBoost / Py-Boost multi-output GBDT** — joint train h5/10/20/40/60 with shared trees | 2211.12858 NeurIPS22 | +1.0~+3.0 | 2d (high reward, NN-free multi-task) |
| 🥈 4 | tree | **LightGBM monotonic constraints** on OFI/imbalance features (sign-known monotone w.r.t. mid-move) | LightGBM docs + microstructure theory | +0.3~+1.0 | 0.5d |
| 🥉 5 | execution | **BAT (Batch-Adaptive Threshold)** — within-batch quantile gate in `predict()`, 1024-sample empirical conformal | IDEA-CTX-DECISION-OPUS internal + conformal lit | +1.5~+3.0 | 0.5d (predictor-only) |
| 🥉 6 | execution | **Robust DFL losses** (Schutte et al. 2310.04328) — extend T87 SPO+ with aleatoric/epistemic-aware regret surrogates | 2310.04328 | +0.5~+2.0 | 1.5d |
| 🏅 7 | target | **GMADL custom objective** (Michańków 2412.18405) on LGB regression — alternative/companion to T99 Huber | 2412.18405 + crypto verification 2602.00776 | +0.5~+2.0 | 1d |
| 🏅 8 | target | **Numerai-style feature neutralization** of regression predictions against confounding LOB features (sym-proxy variables) | Numerai docs / forum 2016 thread | +0.3~+1.5 | 0.5d (post-hoc) |

> **Combined upside (additive but decorrelated, conservative)**: +4 ~ +8 LOSO over iter_016 v3 (+44.71 → +49 ~ +53), aiming at +30~+35 platform after the 0.7 transmission ratio.

---

## Part 1 — Top Kaggle Writeups: Trick 提炼

> 已在 R33 和 SOTA-SURVEY 中详写 Optiver/Volkova/Yirun/nyanp，本部分**不复述**已记录的 12 条；只列**外部搜索新发现的细节**或在二级 GitHub 复盘中能挖到的实现要点。

### 1.1 Optiver – Trading at the Close 2023, hyd 1st (NEW细节)

来自 [`fan2goa1` blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/) 二级复盘（已 WebFetch 验证）：

| # | Trick | 是否已被我们尝试 | 价值 |
|---|---|---|---|
| 1 | **Triplet imbalance** `(max - mid)/(mid - min)` 用 **numba 并行** | 提议 (R33 §1 Top-Trick #5) 但未实施 | ⚠️ 应该 try |
| 2 | **Pairwise imbalance over ALL combinations** of 6 价格 features（C(6,2)=15 对），不只是相邻档 | ❌ 没试过 — 我们只算 consecutive (bid_i - bid_{i+1}) | **🥇 GOLD** |
| 3 | **Stock-level globals: median, std, peak-to-peak of sizes per stock_id, cached** | ❌ 我们 sym-agnostic 直接 banned per-sym 缓存——但**可用 train-set 全局 quantile 而不是 per-sym** | 半 GOLD |
| 4 | **Synthetic index = weighted_sum(stock_returns)** (linear regression weights), 个股 deviation 作 feature | 部分（cross-sym pooling T48 但 fails on inference）| 修订版可救 |
| 5 | **Lagged shift(1,2,3,5,10) + pct_change(1,2,3,5,10)** on matched_size, imbalance_size, reference_price | ❌ 我们 lag windows 是 EWMA-based，没用过 multi-shift | half GOLD |
| 6 | **Day-of-week (date_id % 5), day-of-month (date_id % 20)** | ❌ banned by C1 (date=0) | ❌ |
| 7 | **Rolling skewness & kurtosis** in addition to mean/std on price/size arrays | ❌ R34 没含高阶矩 | half GOLD（高阶矩可能噪声大） |

**Key takeaway**: **pairwise imbalance ALL pairs + cached stock-globals (replaced by global train quantiles)** 是 in-house 226→359-d 之外最直接可加 ~10-15 个新 features 的 gold mine。

### 1.2 Optiver – Realized Volatility 2021, nyanp 1st (NEW细节)

R33 已详记。**本次 web search 补充**：[91st place blog (Chris Richard Miles)](https://chrisrichardmiles.github.io/chrisrichardmiles/projects/optiver/index_optiver.html) 揭示**主流 top-10 共识做法**：

- **600 个 3s LOB snapshots → 400-700 derived features per (stock, time_id)**（我们的 sample 是 100-tick 窗口，可类比为 100 snapshots → 我们目前 226-359-d，**还有空间**）
- **rolling stats over 30s/60s/120s/300s/600s 多窗口**（我们 EWMA 用了 100/200/600，**没用 [30/60/120/300]**，多窗口拼接低成本）
- **dart booster 比 gbdt 更稳**（T50 已在跑 dart，应等结果）
- **LightGBM + NN ensemble 0.7/0.3 是常见**（我们 iter_016 用的 4-way 是过密的；学一下 weight=0.7/0.3 是否更稳）

### 1.3 Jane Street 2020 + 2024（已在 R33/SOTA-SURVEY 全覆盖）

无新 trick。

### 1.4 G-Research Crypto 2022 (重点提醒：HMA + Fibonacci)

社区共识 [Hull MA](https://www.kaggle.com/competitions/g-research-crypto-forecasting) + Fibonacci windows {55, 210, 340, 890, 3750}（在我们 100-tick 窗口里映射为 **{5, 13, 34, 89}**）。**我们 in-house 还没用过 HMA 和 Fibonacci windows**（仅用了 100/200/600 EWMA）。

### 1.5 IMC Prosperity 2024-2025（差异性大，借鉴有限）

[Prosperity 4](https://prosperity.imc.com/) / [Linear Utility 2nd](https://github.com/ericcccsliu/imc-prosperity-2) / [Alpha Animals 9th Global](https://github.com/CarterT27/imc-prosperity-3)：核心是 **market-making + Black-Scholes + cross-exchange arb**——**与我们的"5 sym 方向预测"任务结构差异大**，trick 难直接迁移。值得借鉴的只有：
- **Game-theory aware threshold tuning**（比赛结构里有人类对手/MM 行为可被博弈）——对我们意义不大
- **Statistical arbitrage between artificially correlated stocks**——我们 5 sym 之间相关性可能非随机，但已被 sym-agnostic 红线封锁

---

## Part 2 — 学术论文核心 idea 提炼

> 已在 R31 P1-P31 + SOTA-SURVEY §1.1-1.15 详读。**本次 web search 新发现**：

### 2.1 Online Decision-Focused Learning (ICLR 2026, [arxiv 2505.13564](https://arxiv.org/abs/2505.13564)) ⭐⭐

- **核心**: 在 dynamic environment 中（objective + data distribution 都变）做 DFL，用 **regularization + perturbation techniques + near-optimal oracle** 处理 non-convex zero/undefined gradient 问题。理论给出 **static + dynamic regret bounds**。
- **vs SPO+**: SPO+ 对 fixed objective + offline 学习；online DFL 处理时变环境。
- **对我们 T87 SPO+ 的启示**:
  - 我们 **不能 online**（C2 ban），但**"perturbation 处理 zero gradient"** 思想可移植：在 SPO+ 的 fee-band 区间内（gradient=0 的死区）注入 Gaussian perturbation σ=fee/3，让 fine-tune 阶段不卡死
  - 对我们的**实验代码 hint**：T87 实现里 `compute_spo_plus_subgrad` 在 |2ĉ-c| ≤ fee_eff 时 grad=0；改为 `grad += ε · sign(c-ĉ)` ε=fee/3 可能解决"训练后期梯度消失"

### 2.2 Robust Losses for Decision-Focused Learning ([arxiv 2310.04328](https://arxiv.org/abs/2310.04328)) ⭐⭐

- **核心**: empirical regret 是 expected regret 的差替代品（"empirical optimal decisions can vary substantially from expected optimal decisions"）；提出 **3 个 robust regret losses** 同时考虑 aleatoric + epistemic uncertainty
- **vs SPO+**: SPO+ 是 empirical regret surrogate；**robust DFL 直接是 expected regret 估计**——"better empirical regret on test samples without increasing computational time per training epoch"
- **对我们 T87 SPO+ 的启示**: T87 SPO+ 在训练时只用 single-realization c，但 test 时 c 实际是分布。如果在 SPO+ 的 c 处加 **bootstrap aleatoric noise**（同 T81 NN regression 残差的 std），则等价于 expected regret 的 plug-in 估计——**对 sym-OOD 应该额外稳**
- **可在 T87 上做 ablation**：损失函数从 `SPO+(ĉ, c)` 改成 `E_{ε~N(0,σ_y)}[SPO+(ĉ, c+ε)]`，σ_y 取 ~2e-3

### 2.3 SketchBoost / Py-Boost (NeurIPS 2022, [arxiv 2211.12858](https://arxiv.org/abs/2211.12858)) ⭐⭐

- **核心**: GBDT for **multi-output**（多 target 同时学），通过 **3 种 sketching 策略**（Top Outputs / Random Sampling / Random Projections）加速 split search **40x**，accuracy 与 LightGBM-OneVsRest 持平或超
- **vs LightGBM 单 target × 5 model**: 我们当前是每个 horizon 单独训 — 5 个独立 LGB；SketchBoost 一次训出 5 个 horizon **共享 splits**，类似 multi-task GBDT
- **对我们的启示**: **5 horizon 是 highly correlated**（h5 是 h60 的 sub-target）；joint 训练理论上能让 trees 学到对 h60 有用、被 h5 用作辅助监督的"中间深度" splits — 类似 Yirun 的 multitask aux target，但 stay in GBDT space
- **风险**: Py-Boost 是 GPU-only，platform CPU inference 怎么 load 模型需要测试（model.txt 兼容性不通用）；可降级用 [XGBoost 2.x multi-output](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html) 替代

### 2.4 GMADL (Michańków et al., [arxiv 2412.18405](https://arxiv.org/html/2412.18405v1)) ⭐⭐⭐

R31 / SOTA-SURVEY P22 已记录，但我们 in-house 还**没真正实施过 LightGBM 版本**。WebFetch 提取了**精确 formula + gradient + hyperparameter recommendation**：

```
GMADL(R, R̂) = -(σ(a·R·R̂) - 0.5) · |R|^b
∂GMADL/∂R̂ = -a · R · σ(a·R·R̂) · (1-σ(a·R·R̂)) · |R|^b
推荐: a=1000 (HFT), b ∈ {1, 2, 5}
```

**LightGBM 单 target regression 实现**（25 行代码）：
```python
def gmadl_objective(y_pred, dataset, a=1000.0, b=1.0):
    y_true = dataset.get_label()
    prod = y_true * y_pred
    sig = 1 / (1 + np.exp(-a * np.clip(prod, -50, 50)))  # numerical safety
    grad = -a * y_true * sig * (1 - sig) * np.abs(y_true)**b
    hess = (a**2 * y_true**2 * sig * (1 - sig) *
            (1 - 2*sig) * np.abs(y_true)**b)
    return grad, np.clip(hess, 1e-6, None)  # hess > 0
```

**vs T99 Huber**: Huber 仍是 magnitude-fitting loss，GMADL 是 **directional + magnitude-weighted loss**——直接对应"PnL = sign · magnitude"的评分公式。可能与 T87 SPO+ 协同（SPO+ 是 NN-side DFL；GMADL 是 GBDT-side DFL；两条腿合奏）。

### 2.5 TabM (ICLR 2025, [arxiv 2410.24210](https://arxiv.org/abs/2410.24210)) ⭐

- **核心**: 改进 MLP 的 parameter-efficient ensembling（BatchEnsemble 类技术），在 [TabArena 2025 benchmark](https://arxiv.org/pdf/2506.16791) 上**与 GBDT 持平、好于 prior tabular DL**
- **median rank vs CatBoost**: TabM=5, CatBoost=5.5, XGBoost=7, LightGBM=7.5
- **对我们的启示**: 当前 NN 多次失败（T1/T19/T32/T81/T93/T94/T95/T97 各种结构）。**TabM 是 MLPLOB 之外另一条 NN 平稳路线**；"parameter-efficient ensembling" 直接是 our 5-seed 的 NN-side 替身，single train run 就出 5-head prediction
- **限制**: GitHub [yandex-research/tabm](https://github.com/yandex-research/tabm) PyTorch only，CPU inference 需要测；优先级低于 GBDT 路线

### 2.6 LOB Cryptocurrency Microstructure ([arxiv 2602.00776](https://arxiv.org/html/2602.00776v1) – 2026-02 Bieganowski/Ślepaczuk) ⭐⭐

SOTA-SURVEY §1.3 已记录核心结论（OFI L1 + spread + VWAP-to-mid 是 top-3 SHAP），**新发现**：

- **Bayesian (TPE/Optuna) hyperparam search + rolling CV with purge window** 是 CatBoost 在 1s 频率上的胜出关键
- **scale-invariant microstructure** 在 5 个 perp 上稳定（BTC/LTC/ETC/ENJ/ROSE 跨 capitalization 一个数量级）——**直接证伪 sym embedding**，加固我们 sym-agnostic 路线
- **关键 missing feature**：VWAP-to-mid deviation 在我们 226→359-d 中**不存在**（SOTA-SURVEY §1.3 已 flag, T54 提议未实施）

### 2.7 Adversarial Validation for Dataset Shift ([arxiv 2112.10078](https://arxiv.org/abs/2112.10078)) ⭐

- **核心**: 用 binary classifier 区分 train / test 样本；如果 ROC-AUC ≫ 0.5 → distributions 不一致 → 选 distribution-closest train 样本做 CV
- **对我们的启示**: T66 已实施 adversarial val；**新角度**：用 adversarial-classifier 学到的样本权重作为 LightGBM `sample_weight` — 对 train 中"接近 test 分布"的样本上权重——**target weighted sample retraining**
- 工具：[hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py) Combinatorial Purged CV (CPCV) 实现

### 2.8 Combinatorial Purged Cross-Validation (CPCV) ⭐

- **2024 review** ([Sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)): CPCV 在 **Probability of Backtest Overfitting (PBO)** 和 **Deflated Sharpe Ratio (DSR)** 上优于 Walk-Forward / K-Fold；新变种 Bagged CPCV / Adaptive CPCV
- **对我们的启示**: 当前 LOSO CV 是"leave-one-sym-out"（5 fold）；CPCV 思想 = **遍历 K-choose-N 的 fold 组合 → 多个 backtest path**（不是单一估计）。我们 sym=5 太少不能直接用；但可在**day-grouped 内**做 CPCV：120 day → C(120,k) 选 k=20 day 作 OOF，扫多 path，更稳的 LOSO 估计

### 2.9 Conformal Prediction for Trading ([arxiv 2502.02561](https://arxiv.org/pdf/2502.02561)) – Decision-Theoretic Foundations ⭐

- **核心**: 风险厌恶 calibration 在 user-specified target quantile 下提供更高 utility
- **对我们 BAT (IDEA-CTX-DECISION) 启示**: BAT 用 batch quantile 是经验 conformal；理论支持来自这条文献 — **finite-sample coverage guarantee under exchangeability** = 我们 1024-sample batch 的 sample mean 误差 ~O(1/√1024) = 3.1%，足够稳

### 2.10 Multi-Class Calibration via Normalization-Aware Isotonic ([arxiv 2512.09054](https://arxiv.org/abs/2512.09054)) ⭐

SOTA-SURVEY P26 已记录。**新搜索补充**: **Beta calibration** [(Trainindata blog)](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/) 是 isotonic 与 Platt 之外**第三条主流路径**——**对 LightGBM 输出概率特别合适**，因为 GBDT 输出常呈现 sigmoid 形状偏差。

---

## Part 3 — Industry best practice 集合

### 3.1 Numerai (forum.numer.ai + docs.numer.ai) ⭐⭐⭐

完全跨 domain 但**思想直接可移植**:

| Numerai 做法 | 对我们的映射 | 已试过? |
|---|---|---|
| **Feature Neutralization** (residualize prediction against features) | 用 **sym-correlated feature 集合** (e.g., spread1, total_volume) 做线性投影，从 prediction 中减掉这些方向的 component → **sym-OOD bias 直接消除** | **❌ GOLD** |
| **Auxiliary targets** ensemble (各种 horizon × residualization)| iter_015 已在用（T87 NN + T75/T99 LGB on Δmid）; **新方向**: 在 LGB-target 里也做 residualization (训 sym-mean-removed Δmid) | 部分 |
| **Era boosting**（按 era 重采样难 era 上权重）| 对应**按 fold 重采样**——但 Group DRO (T45) 已证伪 | ❌ banned |
| **Meta-model neutralization**（自家提交 vs 其他参赛者均值取 orthogonal）| 不适用（无其他 submission 信号） | ❌ |
| **TC (True Contribution)** vs **CORR**：对最终 portfolio 的贡献而非单 feature corr | 对我们 = **per-fold contribution**：T87 在 fold 2 贡献多少？T99 在 fold 3 贡献多少？算 contribution → 重权 | half-试过（CDST stacking 提议） |

> 关键 GOLD: **feature neutralization** 是 in-house 完全空白。code 30 行：
> ```python
> # Y = predictions; X = (n_samples, n_neutralize_features)
> # 投影 Y 到 X 列空间的正交补
> beta = np.linalg.lstsq(X, Y, rcond=None)[0]
> Y_neutral = Y - X @ beta
> ```

### 3.2 Optiver tech blog & 公开 talks

- [Optiver Realized Vol intro](https://www.kaggle.com/code/lucasmorin/optiver-realized-volatility-introduction) (Lucas Morin Optiver staff): **"realized volatility per minute bucket"** 多窗口公式细节（30s/60s/120s/300s/600s）以及**EWMA decay weighted aggregation**
- 我们已用 EWMA windows 100/200/600，但**短窗口 30/60 没用过**

### 3.3 Two Sigma & Jane Street tech blog

- 公开内容偏向 infra（Rust / Python pipelines / latency 优化），**alpha 内容很少公开**
- [Jane Street blog](https://blog.janestreet.com/) 主要是 OCaml / functional programming，无金融 ML 配方
- **不必深挖**——alpha-side 黑盒

### 3.4 HFT 社区共识 (mansoor-mamnoon repo + [emergentmind topics](https://www.emergentmind.com/topics/order-flow-imbalance))

- **Microprice as best mid estimator**: 已实施
- **Hawkes process for OFI forecasting** ([Anantha 2408.03594](https://arxiv.org/abs/2408.03594)): **直接从过去 60 tick 推未来 30 tick OFI 强度** — 作为 feature 加入，**未实施**
- **L1-L10 cumulative imbalance ratios**: 已实施
- **Trade Imbalance + Kyle's λ + OFI 三件套**: Kyle's λ 我们没显式算 (`λ ~ Δmid / Δvolume_signed`)，**未实施**

### 3.5 awesome-deep-trading 仓库 ([cbailes/awesome-deep-trading](https://github.com/cbailes/awesome-deep-trading))

- 主要是 **DRL-based trading systems** (FinRL, PyBroker)——**与我们 stateless predictor 不兼容**
- 个别有意思: **Featuretools** 自动 deep feature synthesis (entity-set + DFS) — 我们 226-d 完全 hand-engineered，从未用过 automated feature engineering 工具
  - 风险：可能 explode 到 5000+ features 之后被 LGB 选不到信号

### 3.6 fan2goa1 mkdocs blog ([Optiver Trading Close 详细复盘](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/))

唯一可读到 hyd 1st 的中文/英文二级解释 (WebFetch 已提取 §1.1)。

---

## Part 4 — 4-axis 整合 top 推荐 (8 total)

> 每个推荐: 出处 / 1 句描述 / in-house 状态 / 实施成本 / 预估 LOSO 改进
> 排序按 (LOSO Δ × feasibility) 从高到低

### Axis 1 — Feature engineering for HFT LOB

#### 🥇 F1: Hawkes-process OFI as engineered feature

- **出处**: [Anantha & Jain 2408.03594](https://arxiv.org/abs/2408.03594) (Aug 2024) + 实践基础 [Anantha et al. 2507.22712 OB Filtration](https://arxiv.org/abs/2507.22712)
- **描述**: 用双变量 Hawkes (buy-self + sell-self + cross excitation, exponential kernel) 拟合过去 60 tick 的 OFI events，monte-carlo 推 next-30 tick OFI 期望强度作为 1 个或多个 lookahead-style feature（**stateless 的关键**：每个 sample 内拟合 + 推断，不跨 sample state）
- **In-house 状态**: ❌ **未试**（仅 EWMA-OFI/MLOFI 30-d 已用；Hawkes filtered 思路在 SOTA-SURVEY P16 提议但只做了 OB filtration，没做 forecasting）
- **实施成本**: 1.5 day（用 [`tick`](https://github.com/X-DataInitiative/tick) 库 hawkes 实现；单 sample 拟合 ≈10ms × 1.47M sample → 4h CPU；可降级用 OLS 自激估计 ε-near closed form）
- **预估 LOSO Δ**: +1.0 ~ +2.5
- **风险**: Hawkes 拟合不稳；可备选 fallback：用 (∂OFI/∂t)·ROC + (OFI_lag - OFI_ema) 近似自激成分

#### 🥇 F2: Pairwise imbalance ALL price pairs + train-set globals as constants

- **出处**: hyd 1st place Optiver Close ([writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)) + [fan2goa1 blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/) §"pairwise imbalance"
- **描述**:
  - **Pairwise**: 对所有 价格 features 集合 P = {bid1..10, ask1..10, midprice, WAP, microprice} 算 C(|P|, 2) 个 `(p_a - p_b)/(p_a + p_b)`（≈ 200-450 对，再用 IC 筛选保留 top 30-50）
  - **Train globals**: 对每 feature 在 train 全集（1.47M 行）算 quantile {0.05, 0.25, 0.5, 0.75, 0.95}，**hardcode 进 Predictor**作为常量；inference 时新 sample 的 feature 减 q50 / 除 (q95-q05) → **sym-agnostic 替代版 stock-globals**
- **In-house 状态**: 部分（已有 cumspread/imbalance 等基础 imbalance，**没做 cross-pair 全展开 + 全局 quantile 锚点**）
- **实施成本**: 1 day（特征生成 4h + 5-seed CV 4h）
- **预估 LOSO Δ**: +1.0 ~ +2.0
- **风险**: 200+ feature → 高维 + 共线；先 IC-rank 后取 top-30，按 LightGBM `feature_importance_gain` 过滤

### Axis 2 — Tree models advanced

#### 🥈 M1: SketchBoost / Py-Boost multi-output GBDT

- **出处**: [Iosipoi & Vakhrushev 2211.12858](https://arxiv.org/abs/2211.12858) NeurIPS 2022 (SBER AI Lab) + [Py-Boost GitHub](https://github.com/sb-ai-lab/Py-Boost)
- **描述**: 一次训练所有 5 horizon (h5/10/20/40/60) joint targets，3 种 sketching (Top Outputs / Random Sampling / Random Projections) 加速 40x；**树 split 共享于 5 horizon → 中层 splits 是 multi-task 监督，比单独 5 LGB 更 generalize**
- **In-house 状态**: ❌ **未试**（仅尝试过 LightGBM 多 horizon stacking T47 = -1.25 vs iter_006，理由是 equal-weight 后处理；joint training 是另一套思路）
- **实施成本**: 2 day（learning curve + GPU train pipeline + CPU inference 兼容性测试）
- **预估 LOSO Δ**: +1.0 ~ +3.0（最大不确定性来自 inference compatibility — Py-Boost saved model 在 CPU-only platform 是否能 load）
- **降级**: [XGBoost 2.x multi_strategy='multi_output_tree'](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html) 是 100% CPU 兼容的，Py-Boost 不可用时直接换 XGB

#### 🥈 M2: LightGBM monotonic constraints on OFI / imbalance / spread

- **出处**: [LightGBM docs - monotone_constraints_method](https://lightgbm.readthedocs.io/en/latest/Parameters.html) ('intermediate' or 'advanced' for less restrictive); microstructure 文献先验 (OFI > 0 → mid 上涨概率 > 0.5)
- **描述**: 对 226-359-d feature 中**已知方向单调** 的 ~10-20 个 OFI / imbalance / spread features 加 monotonic constraint(+1 / -1)；剩余特征 unconstrained
  - 例: `imbalance` (= (bsize-asize)/(bsize+asize)) → +1 wrt `predicted Δmid`
  - `spread1` → +1 wrt `predicted vol of |Δmid|`（**这个是 abs target，需要 abs target head 才能用 — 与 T78/T82 abs-prediction 模型对接**）
- **In-house 状态**: ❌ **未试**（226-d LGB 全 unconstrained）
- **实施成本**: 0.5 day（先验列表 + LGB params + 跑 5-seed → 比对）
- **预估 LOSO Δ**: +0.3 ~ +1.0（小但稳）
- **风险**: 错误先验导致 underfit；用 'advanced' method 减少损失

### Axis 3 — Execution / decision rule under PnL scoring

#### 🥉 D1: BAT (Batch-Adaptive Threshold)

- **出处**: in-house IDEA-CTX-DECISION-OPUS Top-1（理论支持 [Decision-Theoretic Conformal 2502.02561](https://arxiv.org/pdf/2502.02561)）
- **描述**: 在 `Predictor.predict(List[DataFrame])` 内，用本次 batch (N=1024) 的 prediction 经验 α-quantile 作为 gate，动态相对 threshold（与绝对 fee threshold 取 max/min 兜底）
- **In-house 状态**: ❌ **未试**（仅 IDEA-CTX-DECISION 完成 brainstorm，尚未 ablate）
- **实施成本**: 0.5 day（修改 Predictor 50 行 + 在 OOF 上 calibrate α_up / α_dn / fallback_thr）
- **预估 LOSO Δ**: +1.5 ~ +3.0（49.6% 的 missed-trade gap 是单一最大 lever）
- **风险**: batch homogeneity（如全 sym=0）时 quantile 退化；需 audit batch composition + 兜底机制

#### 🥉 D2: Robust DFL Regret Loss extensions to T87 SPO+

- **出处**: [Schutte et al. 2310.04328 ICLR 2024](https://arxiv.org/abs/2310.04328) (Robust DFL Losses)
- **描述**: T87 是 empirical SPO+；**robust DFL** 用 expected regret = E[SPO+(ĉ, c+ε)] (over aleatoric noise ε ~ N(0, σ_y))，相当于 T87 训练时 c 加 σ_y · 标准噪声（σ_y = 2e-3，从 T75 / T99 OOF 残差估）
  - 等价于 **target augmentation + decision-aware loss** 的混合
- **In-house 状态**: ❌ **未试**（T87 SPO+ 是 vanilla 版）
- **实施成本**: 1.5 day（T87 main 训练 loop 加 Gaussian target perturbation + bootstrap σ_y estimate）
- **预估 LOSO Δ**: +0.5 ~ +2.0（在已 +1.85 基础上再加）
- **风险**: σ_y 估值偏差；fallback 用 [-1σ, +1σ] 三点 quadrature 替代 stochastic ε

### Axis 4 — Target / loss for tabular regression with cost

#### 🏅 T1: GMADL custom objective for LightGBM

- **出处**: [Michańków et al. 2412.18405](https://arxiv.org/html/2412.18405v1) + crypto verification [Bieganowski 2602.00776](https://arxiv.org/html/2602.00776v1)
- **描述**: `loss = -(σ(a·R·R̂) - 0.5) · |R|^b`，directional loss with magnitude weight；R̂=LGB regression output, R=Δmid_norm。 a=1000, b=1 是 HFT 推荐
- **In-house 状态**: ❌ **未真正实施**（SOTA-SURVEY §1.2 提议 + Method 1，但 T57 PnL-aware-loss 用的是 sample weight 不是 GMADL；T87 SPO+ 是 NN 侧）
- **实施成本**: 1 day（25 行 LGB fobj 实现 + 5-seed × 6 hyperparam grid {a∈[100, 1000, 10000], b∈[0.5, 1.0]}）
- **预估 LOSO Δ**: +0.5 ~ +2.0（与 T99 Huber 协同；可能 ensemble 加 LGB-GMADL 替代 LGB-Huber 的一份权重）
- **风险**: a=1000 时 sigmoid 易饱和 → numerical issue；用 `np.clip(prod, -50, 50)` + safe hess

#### 🏅 T2: Numerai-style feature neutralization on predictions

- **出处**: [Numerai docs - FNC](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation) + [forum thread "What exactly is neutralization"](https://forum.numer.ai/t/what-exactly-is-neutralization/2016)
- **描述**: 对 OOF 5-seed 预测 Y (n × 1)，选 ~10 个 **可能含 sym-bias 的 confounding features** X (n × 10)（候选: spread1, total_volume, midprice level, vol-realized, time-of-day），做 OLS Y → X，**减去 X 投影**：
  ```python
  beta = np.linalg.lstsq(X, Y, rcond=None)[0]
  Y_neutral = Y - X @ beta
  ```
- **In-house 状态**: ❌ **完全未试**
- **实施成本**: 0.5 day（post-hoc 处理 OOF prediction 后再做 DE thresh + LOSO；如果有效再加进 Predictor.predict 内）
- **预估 LOSO Δ**: +0.3 ~ +1.5（主要修复 fold 2 sym=2 ETF spread bias）
- **风险**: 神经化太强 → 总 variance 减小，信号同时消失。先小步：去掉 1 个 feature 再 incremental

---

## 5. 与 4 axis worker (T110-T113) 协同建议

| 4-axis worker | 我建议他们重点 | 我覆盖到的外部资料 |
|---|---|---|
| T110 (feature) | 优先 F1+F2 (Hawkes OFI + Pairwise ALL) | hyd Optiver, Anantha Hawkes 2024 |
| T111 (model) | 优先 M1 (SketchBoost / Py-Boost) | 2211.12858, Py-Boost docs |
| T112 (execution) | 优先 D1 (BAT) — 最大 lever 49.6% missed | IDEA-CTX-DECISION-OPUS, Conformal 2502.02561 |
| T113 (target/loss) | 优先 T1 (GMADL) + T2 (Numerai neutralization) | 2412.18405, Numerai docs |

---

## 6. Negative results / Don't-do list (来自外部 lit + 自身 audit)

| 已知坑 | 来源 |
|---|---|
| **Online learning** (Volkova +0.008 R²)：违反 C2 | R33 |
| **Sym embedding**：违反 C3，且 Volkova 实测"didn't work" | R33, SOTA-SURVEY |
| **iTransformer / PatchTST / TimesFM zero-shot**：金融上 R² < CatBoost | R31 P3 (Re(Visiting) TSFM) |
| **Per-feature z-score normalization**：破坏 LOB 价格档位单调性 | LOBench 2505.02139 |
| **Group DRO (5 sym 太少)**：T45 实证 -5 LOSO | SOTA-SURVEY §2.1 |
| **Multi-horizon equal-weight stacking**：T47 -1.25 LOSO | SOTA-SURVEY §2.2 |
| **Cross-sym pooling (实时统计)**：T48 推理不兼容 | SOTA-SURVEY §1.1 |
| **iTransformer + 双向 Mamba**：违反 stateless / 因果 | R31 P28-P30 |
| **Featuretools auto-FE**：5000+ features 在 LGB 上信号被稀释（社区经验） | awesome-deep-trading repo |
| **IMC Prosperity strategies**：MM + Black-Scholes 与我们方向预测任务不同构 | IMC repos |

---

## 7. 调研统计

- **WebSearch 次数**: 14
- **WebFetch 次数**: 4 (2 paper full-text, 1 fan2goa1 blog, 1 ICLR 2026 paper)
- **新论文 (R31/R33/SOTA-SURVEY 之外的)**: 8 (online DFL, robust DFL, SketchBoost, TabM, decision-theoretic conformal, multi-class isotonic, GMADL precise formula extraction, adversarial validation 2024)
- **新比赛细节**: 5 (hyd Optiver pairwise ALL, Optiver RV multi-window, IMC Prosperity 4 baseline, Numerai feature neutralization, JS 2024 trimmed-mean)
- **GOLD 候选 (in-house 未试)**: 8 (8 推荐全部未试)
- **半 GOLD (部分试过)**: 4 (multi-shift lag, 30/60s short window, sample-weight via adversarial val, beta calibration)

---

## 8. 关键链接一览

### Kaggle writeups
- [hyd 1st Optiver Trading at Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
- [nyanp 1st Optiver Realized Vol](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- [fan2goa1 blog Optiver](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/)
- [Volkova solution.md JS 2024](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md)
- [chrisrichardmiles 91st Optiver](https://chrisrichardmiles.github.io/chrisrichardmiles/projects/optiver/index_optiver.html)

### Academic 2024-2026 (R31/R33/SOTA-SURVEY 外的新发现)
- [Online DFL ICLR 2026](https://arxiv.org/abs/2505.13564)
- [Robust DFL ICLR 2024](https://arxiv.org/abs/2310.04328)
- [SketchBoost NeurIPS 2022](https://arxiv.org/abs/2211.12858)
- [TabM ICLR 2025](https://arxiv.org/abs/2410.24210)
- [TabArena 2025 benchmark](https://arxiv.org/abs/2506.16791)
- [GMADL 2412.18405](https://arxiv.org/html/2412.18405v1)
- [Hawkes OFI 2408.03594](https://arxiv.org/abs/2408.03594)
- [Decision-Theoretic Conformal 2502.02561](https://arxiv.org/pdf/2502.02561)
- [Crypto Microstructure 2602.00776](https://arxiv.org/html/2602.00776v1)
- [Adversarial Validation 2112.10078](https://arxiv.org/abs/2112.10078)
- [LightGBM monotone constraints](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [XGBoost multi_output_tree](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)

### Industry & tools
- [Numerai docs - FNC](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation)
- [Numerai forum - what exactly is neutralization](https://forum.numer.ai/t/what-exactly-is-neutralization/2016)
- [Py-Boost GitHub](https://github.com/sb-ai-lab/Py-Boost)
- [yandex-research/tabm](https://github.com/yandex-research/tabm)
- [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab)
- [pytabkit](https://github.com/dholzmueller/pytabkit) (RealMLP / TabM unified API)
- [Optiver intro Kaggle Lucas Morin](https://www.kaggle.com/code/lucasmorin/optiver-realized-volatility-introduction)
- [Two Sigma Insights](https://www.twosigma.com/insights/) (mostly infra, low alpha-FE content)

### Awesome lists
- [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant)
- [grananqvist/Awesome-Quant-Machine-Learning-Trading](https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading)
- [cbailes/awesome-deep-trading](https://github.com/cbailes/awesome-deep-trading)
- [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)

---

RESULT: task=literature_scan top_8_per_axis=[F1_Hawkes_OFI_+1.0~+2.5, F2_Pairwise_ALL_pairs+train_globals_+1.0~+2.0, M1_SketchBoost_multi_output_+1.0~+3.0, M2_LightGBM_monotonic_+0.3~+1.0, D1_BAT_batch_quantile_gate_+1.5~+3.0, D2_Robust_DFL_extension_T87_+0.5~+2.0, T1_GMADL_LGB_obj_+0.5~+2.0, T2_Numerai_FNC_+0.3~+1.5] notes=14 web search + 4 webfetch; 8 GOLD recommendations all in-house untried; combined upside +4~+8 LOSO if decorrelated; key new sources beyond R31/R33/SOTA-SURVEY are online_DFL_ICLR2026, robust_DFL_ICLR24, SketchBoost_NeurIPS22, GMADL_precise_formula, Numerai_feature_neutralization, Hawkes_OFI_2408, hyd_pairwise_ALL_pairs (not just consecutive)
