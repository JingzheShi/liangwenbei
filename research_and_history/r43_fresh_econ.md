# R43 — Fresh-eye 视角：计量经济学 / 微结构理论的根本性 ideas

> 写作准则：**不看** experiments / iter_*。只看 data_schema、RULES、ENV、CONSTRAINTS、SUBMISSION_LOG 速览表。
> 视角：经济理论第一性原理，**理论根基性 > 工程技巧**。
> 任务：5 sym × 100-tick × 154 features → 5-horizon 三分类 → 取 max cum_pnl。
> 约束 hardware：fee=2×0.0001 round-trip ≈ 0.02%；α 阈值 0.05%/0.1%；不能用 sym/date；shuffle batches。
> 当前 SOTA：**iter_013** = LightGBM 5-seed regression-on-Δmid + EV-gate → 平台 +19.23 (label_60)。

---

## §0 速读：SUBMISSION_LOG 已做事项摘要

从 SUBMISSION_LOG 速览表 + iter_007/iter_013 详情段落看到：
- **特征**：LightGBM Scheme A→D（最多 327-d / 359-d），R34 Stage 1+2 = 113 sym-invariant 特征：
  dual-zscore, signed RV, Kyle's λ rolling, GOFI/MLOFI, window quantile rank, realized skewness,
  vol-burst ratio, EWMA-OFI；T78 加 6 time features。
- **模型**：LightGBM (5-seed), CatBoost (T46), 多 horizon stacking (T47, T77 vote)。
- **决策**：单 threshold sweep → DE 4D thresh → EV gate (regression + asymmetric thr_up/thr_dn)。
- **失败案**：iTransformer/PatchTST/TimesNet/DeepLOB-CNN h_60 全 -16+；
  Platt/Isotonic/Temperature 校准全摧毁信号；alpha101+191 collinear；
  cross-sym pooling/mixup inference 不兼容；CatBoost Ordered + GPU 不支持；
  group DRO 修 fold2 但毁 fold4；class-weight 全失败（饱和）。
- **本质 lever**：**iter_013 关键不是模型，而是 loss 形式 + EV 决策**——
  regression on Δmid 而非 3-class CE，加 EV gate → 平台 +19.23 vs iter_002 +4.07（**+15.16 提升**）。
  核心信号已被特征捕捉到了，**未来 lever 在 loss / decision 框架，不在更多特征**。

---

## §1 微结构经济学第一性原理 — 12+ ideas

下表是从理论根源出发列出的 idea，每条标注：
- **理论锚**（出处）
- **机制**（为什么有 alpha）
- **状态**：✅ 已做 / ❌ 不可行 / ⚠️ 部分做 / 🆕 未做

---

### Idea 1 ⚠️ — Cont (2014) OFI 单变量回归 ↔ Δmid 的"理论锚"

**理论锚**：Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events".
对 50 只 S&P 500 股票，OLS Δmid ~ OFI（best-quote）→ R² ≈ 65%。
> "Models relying solely on trade imbalance yield substantially lower explanatory power (~32%)."

**机制**：OFI = 净限价委托流（best level），不是简单 imbalance 而是**事件级 net delta**。
- ΔOFI_bid = +bid_size if Δbid_price ≥ 0 else -bid_size_prev (canceled / under-cut)
- ΔOFI_ask 对称定义
- 关键：OFI 是 **signed flow**（区分新增 vs 撤单 vs 价格 step），imbalance 只是 (bsize-asize)/(bsize+asize)

**对比**：log 中提到 GOFI / MLOFI / EWMA-OFI / dual-zscore — 但 GOFI 是 multi-level 加权，
**Cont 原始版本（best-quote signed event flow with Δbid sign + Δsize ）可能没单独验证**。

**状态**：⚠️ MLOFI 用了，但**单纯 best-level Cont OFI 作为"理论 R²-65% baseline"对照可能没跑过**。
检查方式：算 OFI_t = ΔBid_size·1{Δbid_price≥0} − ΔBid_size_prev·1{Δbid_price<0} - 对称 ask side。

**实施**：W=20/50/100 cumulative OFI，作为 regression on Δmid 的单 feature OLS R² 报告——
**应得到 in-sample R² ≥ 0.50**，否则我们的 OFI 实现有 bug。

---

### Idea 2 🆕 — Cont-de Larrard (2010) Queue-Imbalance Closed-Form 概率

**理论锚**：Cont & de Larrard (2010) "Price Dynamics in a Markovian Limit Order Market".
Bid/ask 队列大小 → 分析式给出 **P(下一个 mid move 是上行 | q_b, q_a)**：

```
P(up | q_b, q_a) = q_a / (q_a + q_b)   (在最简 M/M/1 假设下)
```

更精细版本带 λ_b (limit order 到达率), λ_a, μ (market order 到达率)：
```
P(up | state) = f(q_b, q_a, λ_b, λ_a, μ_b, μ_a)
```

**机制**：理论给出一个**单维标量 probability** —— 不需要训练，不需要 fitting。
直接给 LightGBM 当 feature。我们用的 imbalance = (bsize-asize)/(bsize+asize) 与
queue-imbalance ratio q_a/(q_a+q_b) 数学上等价但**带正负号语义**不同（imbalance 居中 0，
queue ratio 居中 0.5）。

**对比 SUBMISSION_LOG**：LightGBM 知道 imbalance；但 imbalance 与 P(up) 的**单调对齐性可能不直接**，
模型自己学的 mapping 不一定收敛到理论解（特别是 LOSO 跨 sym 时）。

**状态**：🆕 **未做**——把 Cont-de Larrard 公式作为**显式单 feature**注入。
进一步：把 q_b·λ_a − q_a·λ_b（drift 方向的理论锚）作为 feature。

**实施**：3 个 features
- `cont_queue_p_up = bsize1 / (bsize1 + asize1)`（top level）
- `cont_queue_p_up_5lvl = Σ bsize_k / (Σ bsize_k + Σ asize_k)`（cumulative top 5）
- `cont_drift = bsize1·la_intst − asize1·lb_intst`（理论 drift）

---

### Idea 3 🆕 — Hawkes 自激发 / 互激发强度（Bacry-Muzy / Bowsher）

**理论锚**：Bacry et al. (2013), Bowsher (2007), Bacry-Muzy "Hawkes processes in finance" (2015)；
最近 Toke (2024) / Lalor-Swishchuk (2025) 用 multivariate Hawkes 做 LOB simulation/forecast.

**机制**：trade arrival 不是 Poisson 而是 self-exciting：
```
λ(t) = μ + Σ_{t_i < t} α · exp(-β·(t − t_i))
```
self-excitation：买单刚来更可能再来买单（informed flow 集群）；
cross-excitation：买市价单 → 紧接买限价单（bid 重建）。

**对比 SUBMISSION_LOG**：我们有 `*_intst`（瞬时强度）与 `*_acc`（变化率）。
但**没有 EWMA-decayed 版本以多个时间尺度 (β=1/5, 1/20, 1/60)**，
没有 cross-excitation features（mb_intst → la_intst 短期 lag 上升的"自相关 1 步"）。

**状态**：🆕 部分——`*_acc` 是一阶差分，**Hawkes 指数衰减 kernel 是积分**，根本不同。
我们 EWMA-OFI 是 EWMA 应用在 OFI 上，**不是 Hawkes 强度本身**。

**实施**：6 类订单 × 3 时间尺度 = 18 features
```python
for evt in ['lb','la','mb','ma','cb','ca']:
    for tau in [5, 20, 60]:
        # exponentially-decayed cumulative intensity
        feat[f'hawkes_{evt}_t{tau}'] = ewma(intst[evt], halflife=tau)
```
另外加 6 cross-excitation features：`hawkes_mb_then_la = corr(mb[t-5:t], la[t-1:t])`。

---

### Idea 4 🆕 — Bipower Variation 与 Jump Detection（Barndorff-Nielsen-Shephard）

**理论锚**：Barndorff-Nielsen & Shephard (2004) "Power and bipower variation with stochastic volatility and jumps".
HAR-RV-J (Andersen-Bollerslev-Diebold 2007).

**机制**：把已实现波动率分解为 **continuous + jump**：
```
RV_t = Σ r_i²
BV_t = (π/2) · Σ |r_i| · |r_{i-1}|   (jumps-robust)
J_t  = max(RV_t − BV_t, 0)             (jump component)
Z_t  = (RV_t − BV_t) / √(varTQ_t)      (BNS jump test statistic)
```

**意义**：jump regime 与 continuous regime 信号特性根本不同。
- Jump 后通常 mean-reversion（overreaction unwinds）
- Continuous 振动后通常 trend continuation（informed flow）

我们的模型若不区分，是把两种场景的信号平均了。

**对比 SUBMISSION_LOG**：Stage 2 有 realized skewness（三阶矩），但**没有 BV/RV 分解**，
没有显式 jump indicator。Stage 2 的 vol-burst ratio 是 max-rolling，**不是 jumps-robust 的**。

**状态**：🆕 **未做**——需要 BV / RV 分解。

**实施**：3+3 features
- `bv_W` for W=20/50/100（jumps-robust vol）
- `jump_share_W = (RV - BV) / RV`（jump proportion，0~1）
- `bns_jump_z_W`（jump test statistic, robust under H0：no jump）

可以叠加：`gate = (bns_jump_z > 3)` 当作"刚刚有 jump"的 binary indicator。

---

### Idea 5 🆕 — Cointegration of bid1/ask1 / VECM 残差作 mean-revert 信号

**理论锚**：Engle & Granger (1987); Hasbrouck (1995, 2002) information share；
Engle-Russell ACD 模型（autoregressive conditional duration）。

**机制**：bid1 与 ask1 在长期是 cointegrated（spread 是平稳的），所以 **bid 与 ask 的相对水平偏离均衡 spread → mean-revert**。
具体：
- 当 bid1 上行快于 ask1 → 下一刻 ask1 需追平 → mid 上行
- 当 spread 异常扩大 → 信息事件刚发生 → 接下来要么 jump 完成、要么修复
- VECM 残差 ε_t 是平稳的 → 它的符号是短期预测信号

**对比 SUBMISSION_LOG**：我们有 spread1, midprice 等，但**没有显式 VECM 拟合 + 残差**。
window-zscore 在 spread 上跑，更接近"是否异常"，但没有 cointegration 框架的"该回到哪里"信息。

**状态**：🆕 **未做**——需要 VECM 拟合 (in-sample) + apply at inference。

**实施**：单 feature
- 训练时：在每个 sym × 训练日上 OLS bid1_t = α + β·ask1_t + ε_t，存 (α, β)
- 但**评测 sym 可能是新的** → 不能用 per-sym (α,β) → **改用 100-tick window 内回归**：
  `vecm_resid = bid1[-1] − (α̂ + β̂·ask1[-1])`，其中 (α̂, β̂) 在窗口内 OLS。
- 该残差的符号 + magnitude 是"应该往哪走"的方向信号。

---

### Idea 6 🆕 — Lee-Ready trade-direction inference + signed volume

**理论锚**：Lee & Ready (1991); Hasbrouck (1991) signed VAR.
Cont (2014) 已显示 signed flow 比 raw imbalance 高 R²。

**机制**：成交方向（buyer-initiated vs seller-initiated）是经典 alpha：
- 价 above mid → buyer aggressor，价 below mid → seller aggressor
- aggressor side 的 volume 是 **signed informed flow**

**对比**：我们有 `mb_intst` (market-buy 强度)，`ma_intst` (market-sell)，比例分得清。
但 **`amount_delta` 没有方向标签**——Lee-Ready 把它转化为 `signed_amount_delta = amount_delta · sign(close − mid_prev)`。

**状态**：⚠️ 部分——`mb_intst − ma_intst` 已被 OFI 类特征覆盖。但
**signed_amount_delta** 把"实际成交金额"分了方向，这是 mb/ma 没有的 size-weighted 信号。

**实施**：1 个 base + W=20/50 累加 = 3 features
```python
tick_dir = sign(close[t] - midprice[t-1])    # +1 buy, -1 sell, 0 unknown
signed_amount = amount_delta * tick_dir
signed_amount_cum_W = rolling_sum(signed_amount, W)
```

---

### Idea 7 🆕 — VPIN（Easley-de Prado-O'Hara 2012）order flow toxicity

**理论锚**：Easley, de Prado, O'Hara (2012) "Flow Toxicity and Liquidity in a High-Frequency World".
预测 2010 Flash Crash。

**机制**：
- 用 volume bucket（不是 time bucket）划分
- 在每个 bucket 内 bulk-classify volume 为 buy/sell（用价格变化分布）
- VPIN = E[|V_buy − V_sell| / (V_buy + V_sell)]

**意义**：toxicity 高 → 信息不对称大 → 当前 trade 信号噪比高 → **减少出手**或**反向交易**。
不是 alpha 方向信号，而是 **置信度/regime 信号**。

**对比 SUBMISSION_LOG**：我们有 mb/ma intensity，但 VPIN 是**时间-体积同步规整**后的标准化值。
没有显式 VPIN feature。

**状态**：🆕 **未做**。

**实施**：1 feature + 1 indicator
```python
vpin_W = mean over rolling W of |mb_intst - ma_intst| / (mb_intst + ma_intst + ε)
high_toxicity = (vpin_W > quantile_window(0.8))
```
然后 EV gate 的 threshold 可以**动态调高**当 toxicity 高时。

---

### Idea 8 🆕 — Adaptive Threshold = f(spread, vol)（regime-aware decision）

**理论锚**：Glosten-Milgrom (1985)；Bollerslev IGARCH intraday seasonality.
**Kelly-with-frictions** (At what Frequency Should the Kelly Bettor Bet? arxiv 2018).

**机制**：fee 是固定的（0.02% round-trip）。但**期望 PnL 标准差是变化的**：
- spread 大 / 隐含成本大 → 真实摩擦 > 显式 fee → threshold 应升高
- vol 小 / 信号细 → break-even 阈值升高
- vol 大 → 信号粗 → break-even 降低（但 single-trade variance 大）

理论 break-even：
```
threshold(t) = max( fee + α·spread(t), k · σ_pred(t) )
```
其中 σ_pred(t) 是模型预测 Δmid 的标准差（regression 残差）。

**对比 SUBMISSION_LOG**：iter_013 用 **固定** asymmetric (thr_up=3.7e-4, thr_dn=1.6e-4)，**全局**。
没有 t-conditional adaptive threshold。

**状态**：🆕 **未做**——这是 iter_013 后的自然下一步。

**实施**：
- 每个测试点：计算 `spread_t = ask1 − bid1`, `vol_t = std(midprice[-20:])`
- 阈值：`thr_t = thr_base + α·spread_t + β·vol_t`
- 在 LOSO OOF 上 sweep (α, β)；预期在大 spread / 大 vol 区域避免出手

⚠️ **风险**：iter_013 已经过 17pt LOSO→Platform gap，再 fit 任何条件阈值容易 overfit。
推荐：**只做单调 monotonic 修正**（thr 随 spread 单调升），不做自由参数空间。

---

### Idea 9 🆕 — Direct PnL-as-Loss / Differentiable Decision Surrogate

**理论锚**：Kelly criterion (Kelly 1956); Markowitz mean-variance; Optimal stopping (Snell envelope).
现代：differentiable decision layers (Amos 2017 OptNet; Wilder et al. 2019 decision-focused learning).

**机制**：iter_013 是 2-step：
1. 训 regressor predict Δmid
2. post-hoc EV gate 决策

**问题**：regressor 不知道 fee，不知道 argmax decision；EV gate 不知道每个 sample 的不确定性。
**End-to-end**：让 loss 直接近似 cum_pnl：

```python
def soft_pnl(pred_logits, true_dmid, fee=2e-4):
    # pred_logits: (B, 3) for [down, flat, up]
    p = softmax(pred_logits)   # smooth argmax
    # signed action = p[2] - p[0] (continuous in [-1, 1])
    action = p[2] - p[0]
    # PnL ≈ action * dmid - fee * |action|
    return -mean(action * true_dmid - fee * |action|)
```

**对比 SUBMISSION_LOG**：iter_013 用 regression_l2 on Δmid_norm；T78 ablation; 没有 PnL-aware loss。

**状态**：🆕 **未做**——根本性切换 loss 形态。

**实施**：
- LightGBM 不支持自定义 loss 直接做 multi-class——但**用 LightGBM `regression` + custom obj/grad** 可以
- 或者用 PyTorch MLP（小模型，sym-agnostic），训 PnL surrogate loss
- 优势：模型直接优化目标；劣势：PnL 信号噪声大，容易 collapse 到全 0

⚠️ **关键提示**：iter_013 之所以从 +4 跳到 +19，正是因为 **regression-on-Δmid 比 3-class CE 更接近 PnL**。
direct PnL loss 是这条路径的**自然延伸**——值得尝试，但要注意 collapse 风险（已观察到 mmpc_demo 的 collapse）。

---

### Idea 10 🆕 — Two-Scale / Multi-Scale Realized Volatility (TSRV / MSRV)

**理论锚**：Zhang, Mykland, Aït-Sahalia (2005) "A Tale of Two Time Scales".
Microstructure noise → naive RV 估计偏 → TSRV 修正。

**机制**：
```
RV_naive = Σ r_i²            (over-estimates due to micro noise)
RV_subsample_K = RV computed on every K-th tick   (under uses data)
TSRV = RV_subsample - (n/n_K) · (RV_naive_truncated)   # bias correction
```

**意义**：naive RV 在 tick 级 dominated by 微观噪声，**真实 vol 被高估** → 信号去噪后更纯净。

**对比 SUBMISSION_LOG**：我们有 rolling std，但是 **naive RV-style，不是 microstructure-corrected**。

**状态**：🆕 **未做**。

**实施**：
- `rv_naive_W = sum(r²)` for r = log(midprice) diff
- `rv_sub5_W = sum over r computed every 5 ticks`
- `tsrv_W = rv_sub5 - 0.2·rv_naive`  (W=100 only worth, since K=5)

---

### Idea 11 🆕 — Range-based Volatility (Garman-Klass / Parkinson)

**理论锚**：Parkinson (1980); Garman-Klass (1980); Rogers-Satchell (1991).
高效估计器：用 high/low/open/close → 比单纯 close-close 方差**信息利用率高 5×**。

**机制**：`high − low` 在区间内的 range 直接反映 vol。
```
parkinson_W = (1/(4·ln 2)) · mean over W of (ln(high/low))²
GK_W = mean over W of [0.5·ln²(high/low) − (2·ln 2 − 1)·ln²(close/open)]
```

**对比 SUBMISSION_LOG**：data_schema 中 `high`/`low`/`open` 都有，
但 SUBMISSION_LOG 提到的 features 都是 close-based / size-based / rate-based。
**range-based vol 大概率没在 stage 1/2 里**。

**状态**：🆕 **未做（高置信度）**。

**实施**：3 features
- `parkinson_vol = (ln(high) − ln(low))² / (4·ln 2)`（per-tick）
- `gk_vol = 0.5·(ln(high) − ln(low))² − (2·ln 2 − 1)·(ln(close) − ln(open))²`
- `range_to_close = (high − low) / |close − open + ε|`（intra-tick noise ratio）

---

### Idea 12 🆕 — Optimal Stopping / Multi-horizon Coherence

**理论锚**：Snell envelope; Almgren-Chriss optimal execution.

**机制**：5 个 horizon 的预测可看作"对未来不同时间段的概率视图"。
若 **h_5 says up, h_10 says up, h_60 says down** → 内部矛盾，置信度低，不出手。
若 **5/5 horizons 一致** → 高置信度，加大出手。

**对比 SUBMISSION_LOG**：T77 multi-horizon vote ensemble +5.98 LOSO-equiv（vs iter_012）。
但 vote 是用于**辅助**（不是 gate）。**没有把"horizon 内部一致性" 用作 EV gate 的 threshold-降低条件**。

**状态**：⚠️ T77 已经做了一部分。但**结构上**还有改进：

**实施**：
- 训 5 个 regressor (per horizon Δmid_h)
- 计算 `coherence = sign(Δmid_5) · sign(Δmid_10) · sign(Δmid_20)` etc.（−1/+1/0）
- 当 coherence ≥ 4/5 → threshold 降低 30%
- 当 coherence ≤ 1/5 → threshold 升高 50%（or 不出手）

✅ 注意：**评测仅看 max horizon**——但若 h_60 是公榜得分，且 h_60 的预测被 h_5/10/20 的内部矛盾否决，那是降低出手数 → 单笔收益升高 → cum_pnl 可能升或降；要看实测。

---

### Idea 13 🆕 — Glosten-Milgrom Adverse-Selection Spread Decomposition

**理论锚**：Glosten-Milgrom (1985); Stoll (1989); Madhavan-Smidt (1991).

**机制**：bid-ask spread 由三部分组成：
- order processing cost（mostly constant）
- inventory cost（与 mm position 相关）
- **adverse selection cost**（与 informed flow 相关）

理论：
```
adverse_selection_share = Cov(Δmid_t, Δmid_{t+1}) / Var(Δmid_t)
```
（Hasbrouck-Sofianos）

高 adverse selection → mm 知道现在交易者是 informed → spread 拉大 → 我们的信号噪比变化。

**对比 SUBMISSION_LOG**：未见。

**状态**：🆕 **未做**。

**实施**：1 feature
- `adverse_selection_W = cov(dmid[-W:], dmid[-W+1:]) / (var(dmid[-W:]) + ε)`
- 高 adverse selection → 减少出手（同 VPIN 思路）

---

### Idea 14 🆕 — Roll's Effective Spread Estimator

**理论锚**：Roll (1984) "A Simple Implicit Measure of the Effective Bid-Ask Spread".

**机制**：在没有 quote 数据时，spread 可由 returns 自相关估计：
```
roll_spread = 2·√(−Cov(r_t, r_{t-1}))   if cov < 0
            = 0                            otherwise
```

**意义**：与 quoted spread 比较 → quoted 高 / Roll 低 = orderbook 表面深，实际成交浅 → 信号弱。
quoted 低 / Roll 高 = 表面紧凑但成交摩擦大 → 流动性陷阱 regime。

**对比 SUBMISSION_LOG**：未见。spread1 用了，但**未与 implied effective spread 对比**。

**状态**：🆕 **未做**。

---

### Idea 15 ⚠️ — Time-of-day Effect（开盘 / 收盘特殊 dynamics）

**理论锚**：Andersen-Bollerslev (1997); Wood-McInish-Ord (1985) U-shape vol.

**机制**：开盘前 30min / 收盘前 30min vol 高、信号噪比变化大。
中午前后 vol 低 / mean-reversion 主导。

**对比 SUBMISSION_LOG**：T78 提到 "regression + EV-gate + 6 time features" 但 LOSO -0.72 vs iter_013，
说明 6 time features 没帮助或反噬。

**状态**：⚠️ T78 试过——但**只有 6 个时间特征**可能不够 / 编码方式不对。

**实施改进**：
- `minutes_since_open = parse_time(time)`（要在 build_cache 阶段加，否则 `time` 不在 DataFrame）
- `is_first_30min` / `is_last_30min` / `is_lunch_quiet` 三个 binary
- `tod_vol_normalized = vol_t / median_vol_at_this_tod`（按时段标准化）→ 时段内偏离

⚠️ 关键：T78 已试 → 收益 -0.72。**不强烈推荐再试，除非编码方式根本不同**。

---

### Idea 16 🆕 — Cancel-to-Limit Ratio（"Spoofing/Fishing" Indicator）

**理论锚**：Eisler-Bouchaud-Kockelkoren (2012) "The price impact of order book events".
Cancel orders 是 informed flow 的**反面信号**——大量取消说明前面挂错了 / 信息变化。

**机制**：
- `cancel_ratio_b = cb_intst / (lb_intst + ε)`
- `cancel_ratio_a = ca_intst / (la_intst + ε)`
- 高 cancel_ratio_b 且低 cancel_ratio_a → bid 在快速消失 → 下行信号

**对比 SUBMISSION_LOG**：cb/ca intst 直接进了 raw 特征，但**比例形式可能没作为标量进**。

**状态**：⚠️ 部分。

**实施**：4 features
- `cancel_pressure_b = cb_intst / (lb_intst + 1e-8)` (rolling W=20/100)
- `cancel_pressure_a = ca_intst / (la_intst + 1e-8)`
- `cancel_asymmetry = (cb_intst − ca_intst) / (cb_intst + ca_intst + ε)`

---

### Idea 17 🆕 — Slippage Curve / Depth-Weighted VWAP

**理论锚**：Almgren-Chriss (2000) "Optimal execution of portfolio transactions";
Obizhaeva-Wang (2013) market impact.

**机制**：把 LOB 看作"对一个假设大单的成交曲线"。
对买 1 单位（1% 换手率），slippage = bid VWAP − mid。
对买 5 单位，slippage 单调上升。**slippage 曲线的斜率**是流动性深度的代用变量。

**对比 SUBMISSION_LOG**：我们有 `cumspread`, `imbalance` 等十档汇总，但
**没有"假设 X 单位成交的 slippage"计算**。

**状态**：🆕 **未做**。

**实施**：2 features
- `slippage_buy_1 = (Σ_{1st level filled to 1 unit} ask_k · asize_k) / 1 − mid` (assume buy 1 unit)
- `slippage_sell_1 = mid − (Σ_{1st level filled to 1 unit} bid_k · bsize_k) / 1`
- `slip_asymmetry = slippage_buy_1 − slippage_sell_1`（正 = 买更难 = 资金紧）

---

## §2 决策框架视角（model vs decision）

### Idea 18 🆕 — Bayesian Posterior Decision (Cost-Sensitive)

**理论锚**：Statistical Decision Theory (Wald 1950); Bayesian risk minimization.

**机制**：**评分公式 = 决策**。给定 P(label=0,1,2 | x)，最优决策是：
```
EV(action=0) = P(0)·E[Δmid|0]·(−1) + P(2)·E[Δmid|2]·(−1) − fee
EV(action=1) = 0
EV(action=2) = P(0)·E[Δmid|0]·(+1) + P(2)·E[Δmid|2]·(+1) − fee
y* = argmax EV(action)
```

不是基于 max P(class) 选 action，而是基于 **posterior expected PnL**。

**对比**：iter_013 是 regression+EV gate（已经接近这个）。但 EV gate 用的是
**单点 Δmid 估计**，不是 P(class)·E[Δmid|class] 加权。

**状态**：🆕 **未做（与 iter_013 在概念上接近，但实现不同）**。

**实施**：
- 训两个模型：classifier P(label=k) + per-class regressor E[Δmid | label=k]
- inference: argmax_y EV(y) — 完整 Bayes 决策

---

### Idea 19 ⚠️ — Kelly-optimal Threshold from Edge / Variance ratio

**理论锚**：Kelly (1956); Thorp (2006) practical Kelly.

**机制**：阈值不应是单纯 EV>fee 而是 **Kelly fraction**：
```
f* = (E[Δmid] − fee) / Var(Δmid)
```
当 f* > 0 出手；f* 越大 size 越大。即使 EV>0，若 Var 巨大，仍可能不该交易。

**对比**：iter_013 EV gate **没考虑 variance**，只看 Δmid_pred 大小。

**状态**：🆕 **未做完整版**。

**实施**：
- 用 LightGBM 同时输出 mean 和 variance（quantile regression: q05, q50, q95）
- Kelly score = (q50 − fee) / ((q95 − q05)/3.92)²
- threshold on Kelly score 而非 on q50

---

### Idea 20 🆕 — CVaR / Tail-aware Loss

**理论锚**：Rockafellar-Uryasev (2000); Robust Portfolio (Cornuejols & Tütüncü).

**机制**：cum_pnl 是 sum，所以**少数极端坏 trades 主导**。CVaR loss 训模型时显式惩罚 tail:
```
L = mean(loss_i) + λ · CVaR_α(loss_i)
```

**对比 SUBMISSION_LOG**：未见 CVaR / robust loss 用法。

**状态**：🆕 **未做**。

---

## §3 Top 5 Fresh-Eye Ideas（**未做** + 理论根基强）

按 expected uplift × novelty × 可实施性排序：

### 🥇 #1 — **Idea 9: Differentiable PnL-as-Loss Surrogate**

**为什么**：
- iter_013 关键 lever 是 loss-shape change（CE → regression）→ +15 PnL
- 这条路径的下一步是 **end-to-end PnL surrogate**：模型直接优化决策目标
- 理论锚：decision-focused learning，明确表明 mismatch 越大 lever 越大

**怎么实施**：
- 用 LightGBM custom objective（`fobj`/`feval`），实现 PnL surrogate gradient
- 或 PyTorch 小 MLP（200d → 256 → 256 → 3）训 3-class soft action
- Loss: `L = -mean( (p[2]-p[0]) · Δmid_norm − fee · |p[2]-p[0]| )`
- 5-seed → ensemble（与 iter_007 风格一致）
- 风险：collapse（mmpc_demo 已踩过）→ 加 KL regularization to ground-truth label distribution

**预期**：+3~+7 LOSO（在 iter_013 基础上）；**但平台 gap 风险高**——若 LOSO+5 → 平台可能 +1~+3。

---

### 🥈 #2 — **Idea 4: Bipower Variation + Jump Detection**

**为什么**：
- Stage 1+2 已用了一堆 RV-based features，但**全部是 jumps-混入版**
- BV (jumps-robust) + RV-BV gap (jumps) → 两个 regime 分开建模
- 单独的 BNS jump z-stat 是文献中证实的强 regime indicator
- 理论锚硬：BNS 的渐近分布已证明（Annals of Statistics 2006）

**怎么实施**：
- 每个 W=20/50/100 加 3 features：BV, jump_share, BNS_z
- 共 9 个新 features，加进当前 359-d 特征 → 368-d
- 其他不变，5-seed regression + EV gate
- 实施成本：~30 min coding + 1 train round

**预期**：+1~+3 LOSO uplift；适合作为 incremental 改进。

---

### 🥉 #3 — **Idea 8: Spread-Adaptive EV Threshold**

**为什么**：
- iter_013 是 fixed asymmetric threshold；理论上**摩擦 ∝ spread**
- LOSO→Platform gap = -17 主因可能就是"全局 threshold 在不同 sym/regime 不通用"
- 简单一阶修正：threshold = base + α·spread；只引入 1 个新参数

**怎么实施**：
- 在 LOSO OOF 上 sweep α ∈ [0, 0.5, 1.0, 1.5, 2.0]
- 选 LOSO-equiv 不下降的最大 α
- 期望 platform 上**降低过度交易的 fold**（low-spread regime 出手不变，high-spread regime 减少出手）

**预期**：LOSO 略降但 platform 改善（-17 gap 可能缩到 -10）；**主要赌 robustness**。

⚠️ 已知 risk：若 spread 与 vol 强相关，可能与已有 features 共线性大导致无效。

---

### #4 — **Idea 2: Cont-de Larrard Closed-Form P(up)**

**为什么**：
- 是**理论解析解**，不是经验 feature
- 即使 LightGBM 已学到 imbalance↔direction 的关系，封闭式概率特征
  对 LOSO 跨 sym 时**保持单调性**——不会因为某 sym 训练数据少而 mapping 学歪
- 实施成本：3 features 加进去就行

**怎么实施**：
- `cont_q_p_up = bsize1 / (bsize1 + asize1 + ε)`
- `cont_q_p_up_5 = sum(bsize[1..5]) / (sum(bsize[1..5]) + sum(asize[1..5]) + ε)`
- `cont_drift = bsize1 · la_intst − asize1 · lb_intst`

**预期**：+0.5~+2 LOSO；**主要做 robustness 而非 magnitude**。

---

### #5 — **Idea 11: Range-based Vol (Parkinson + Garman-Klass)**

**为什么**：
- `high`/`low`/`open` 几乎肯定**没用过**（log 中 features 全是 close-based）
- 理论上比 close-only RV 信息利用率高 5×
- 在 100-tick 短窗口内 RV 估计噪声大，**range-based 估计更稳**
- 与现有 vol features 互信息低（来源不同的统计量）

**怎么实施**：
- 3 features：`parkinson_per_tick`, `gk_per_tick`, `range_close_ratio`
- W=20/50/100 取均值 = 9 features 总
- 加进现有 feature set，retrain

**预期**：+0.5~+2 LOSO；**最低风险高 ROI**。

---

## §4 提交策略建议（在迭代约束下）

每天 2 次提交。Top 5 实施顺序建议：

| 优先 | Idea | 实施时间 | 风险 |
|------|------|---------|------|
| 1 | #5 Range-based vol | 30 min coding + 1 train | 极低（incremental） |
| 2 | #4 Cont-de Larrard | 20 min coding + 1 train | 极低 |
| 3 | #2 Bipower / Jump | 1h + 1 train | 低 |
| 4 | #3 Spread-adaptive thr | 1h sweep | 低（不动模型） |
| 5 | #1 PnL surrogate | 4h + 充分验证 | 中（collapse 风险） |

**1+2+3 可以打包为单次 retrain**（特征叠加），用 1 次提交检验。
**4 是 inference-time 改动**，不需要 retrain。
**5 是独立路径**，单独提交检验。

---

## §5 Summary 数字

- Ideas 列出：**20 条**（含 idea 1-17 微结构本体 + 18-20 决策侧）
- 状态分布：✅ 0 / ⚠️ 5 / ❌ 0 / 🆕 15
- Top 5 全部 🆕 + 理论根基硬 + 实施 < 6h

返回：`RESULT: task=r43_fresh_econ metrics={n_ideas=20, n_undone=15, top5=[pnl_surrogate_loss, bipower_jumps, spread_adaptive_thr, cont_de_larrard_closed_form, range_based_vol_parkinson_gk]} notes=[lever 在 loss/decision 框架，不在更多 features；range_based 与 cont_de_larrard 是低风险 quick wins]`
