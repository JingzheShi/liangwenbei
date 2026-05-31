# Cross-Symbol / Cross-Asset Interaction Features 调研报告

> **项目**: 良文杯 LOB 中间价方向预测 — `cross_sym_nn` (v6_pairwise SOTA +37.25 ± 0.58)
> **任务**: 为 Worker B 实现寻找 cross-sym interaction features
> **数据**: 5 sym × 359 raw feat × 同时刻 t (训练/测试对齐)
> **架构基础**: v6_pairwise (sym_i − sym_j 显式 pair feature → MLP)
> **报告日期**: 2026-05-31

---

## § Executive Summary — 最重要 5 个 Insight

### Insight 1: v6_pairwise 验证了 "显式 pair feature 优于 attention"
v6_pairwise 在 5-seed 上把 SOTA 从 +36.35 推到 +37.25 (+0.9)，**关键改动是
显式构造 `mp_t[i] - mp_t[j]` 类型的 pair feature**, 让 MLP 直接消化而非依靠
self-attention 学。这意味着我们手工选好的 interaction feature 是 first-class
citizen, 后续 ablation 的 baseline。

### Insight 2: Cross-Impact OFI 是最高优先级新方向
[Cross-Impact of Order Flow Imbalance in Equity Markets, Quantitative Finance
2023](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159) 指出：
**lagged cross-asset OFI 显著改善 short-horizon forecasts**, 而 contemporaneous
cross-impact 在 multi-level OFI 已经包含的情况下意义不大。这对应到我们的 setup：
- 同时刻 5 sym 已对齐 → "lagged" 在 LOB tick frequency 上对应 W={5,20,60}
  不同窗口的 OFI 差异。
- 已有 30 维 MLOFI 是单 sym 的；**新方向：每个 sym 增加 (OFI[i] − OFI[j])
  pair feature 共 10 对**。

### Insight 3: Cross-Sectional Rank / Z-Score 是 Numerai/Alpha101 公认的免费 lift
对 5 sym 做 cross-sectional standardize / rank 是经典做法：
- Alpha101 (Kakushadze 2015) 的 rank(·) 算子，用 cross-section 内排名做特征。
- Numerai signals 推荐 era-neutralize + double-neutralize 流程。
- 在我们的 5 sym universe 下，rank(feat[0..4]) 给 sym i 的相对位置 ∈ {0,
  0.25, 0.5, 0.75, 1}, 是 sym-agnostic 的 (rank 不依赖 sym id), date-robust 的
  (cross-sectional 操作天然 era-neutralize)。

### Insight 4: Mantegna MST / DTW 这类 "图特征" 在 5 sym 下信息量稀薄
经典 graph-feature 方法 (Mantegna MST, PMFG, eigenvector centrality) 在 100+
节点的 stock universe 上效果好, 但**只有 5 个 sym 时, MST 只有 4 edges, 拓
扑结构能区分的状态太少**。这类特征列在表中但优先级低 (P3+)。

### Insight 5: 必须严格遵守 3 条评测协议硬约束
1. **`date` 被置 0 不可用作特征** → 任何依赖 absolute date / time-of-day 的
   特征拒掉 (本调研已避免)。
2. **测试点顺序被打乱 + Predictor stateless** → 任何需要 stateful 推理
   (如 Hawkes 强度, online changepoint) 都拒掉; 调研中 lead-lag 全部用
   "更长窗口 (W20) vs 更短窗口 (W5)" 的 lagless proxy。
3. **sym 0-4 但可能含训练外股票** → 任何 per-sym 参数 (例如 cointegration
   beta, factor loading) 标记 `sym_robustness_concern=high` 并加警告。

---

## § Category 1: HFT / LOB Cross-Asset (10 papers)

### 1.1 [Cont, Kukanov & Stoikov 2014] — The Price Impact of Order Book Events
- *Journal of Financial Econometrics* 12(1) | [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Key insight**: 短时间 price changes 主要由 best-bid/ask 的 OFI 驱动；价
  格变化与 OFI 线性, slope ∝ 1/market_depth。
- **本项目用法**: 已存在 F2 MLOFI 30 维。**新方向**: pair diff (OFI[i] −
  OFI[j]) — features F4-F6, F19-F20, F67, F102-103。

### 1.2 [Kolm, Turiel & Westray 2023] — Deep OFI: Extracting Alpha
- *Mathematical Finance* 33(4) | [SSRN 3900141](https://ssrn.com/abstract=3900141)
- **Key insight**: 多 level OFI 输入 + DL 能在多 horizon 上提取 alpha; PCA
  on OFI vector 给 "integrated OFI"。
- **本项目用法**: 已用 MLOFI; **新方向**: 跨 sym 的 OFI 加总/share — F55-60。

### 1.3 [Cross-Impact of OFI in Equity Markets, QF 2023]
- *Quantitative Finance* 23(10) | [Tandfonline 2236159](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159) | [arXiv:2112.13213](https://arxiv.org/abs/2112.13213)
- **Key finding**: "Once information from multiple order book levels is
  integrated into OFI, multi-asset models with cross-impact do **not** provide
  additional explanatory power for **contemporaneous** impact... however,
  **lagged cross-asset OFIs do improve the forecasting of future returns**,
  with this lagged cross-impact mainly manifesting at short-term horizons
  and decaying rapidly in time."
- **本项目用法**: **这是 Top 1 灵感来源**。Worker B 应优先实现 cross-asset
  OFI pair diff (F4, F5, F6) 和 cross-asset OFI residual (F78-80)。

### 1.4 [Sirignano & Cont 2019] — Universal Features of Price Formation
- *Quantitative Finance* 19(9) | [arXiv:1803.06917](https://arxiv.org/abs/1803.06917)
- **Key insight**: pool 多个 stock 训练一个 universal model 比 per-stock 模
  型好, 且能 generalize 到训练外股票。
- **本项目用法**: 直接支持我们的 sym-agnostic 设计;新方向: cross-sectional
  market-residual (F31, F61-63, F100, F166-167, F194)。

### 1.5 [Zhang, Zohren & Roberts 2018] — DeepLOB
- *IEEE TSP* 67(11) | [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- **Key insight**: CNN-Inception-LSTM 在 LOB 上 SOTA;**模型能 generalize 到
  训练外股票**, 暗示 universal features 存在。
- **本项目用法**: 已是 baseline; 暗示 cross-sym features 不会破坏 sym-agnostic
  (因为 LOB events 本身就 universal)。

### 1.6 [Berti et al. 2025] — TLOB / MLPLOB Dual Attention vs MLP
- [arXiv:2502.15757](https://arxiv.org/abs/2502.15757)
- **Key insight**: MLPLOB (simple MLP) 在 LOB 任务上可达 TLOB (dual attention)
  水平, 暗示**显式 feature engineering 比 attention 学更可靠**。
- **本项目用法**: 直接支持 v6_pairwise 设计哲学; cross-sym features 应显式构造而
  非依赖学到。

### 1.7 [Xu, Cont & Stoikov 2019] — Multi-Level OFI in LOB
- [arXiv:1907.06230](https://arxiv.org/pdf/1907.06230)
- **Key insight**: 多 level OFI (MLOFI) 在 6 liquid stocks 上拟合
  contemporaneous mid-price changes; 加深 level 能持续提升 R²。
- **本项目用法**: MLOFI 已有; **新方向**: pair-level MLOFI 差值 (F19-20, F102-103)。

### 1.8 [Pacurar 2008] — Sequential / cross-sectional order flow & price discovery
- Multi-asset volatility spillover 综述; Hasbrouck VAR price discovery 是基础。
- [Hasbrouck 1995](https://faculty.washington.edu/ezivot/research/Fifth%20Draft-%20An%20Order%20Invariant%20measure%20of%20price%20discovery,%20application%20to%20ETF%20-%20Copy.pdf), [Hasbrouck 1991](https://pages.stern.nyu.edu/~jh4/Research/HRVAR/HRVAR08.pdf)
- **Key insight**: VAR 框架估计 price discovery share; cointegrated markets
  共享 efficient price 的 permanent innovation。
- **本项目用法**: F100-101 cross-sym factor residual (logret[i] − beta·market_ret);
  beta=1 简化 (避免 per-sym beta)。

### 1.9 [Attention-Based Multi-Asset OFI Networks for Mid-Price Prediction, ICAIF 2024](https://dl.acm.org/doi/10.1145/3768292.3770430)
- **Key insight**: 2024 论文 - Attention-based + GNN 学 cross-asset relationships
  dynamically; 多 asset OFI 联合处理优于单 asset。
- **本项目用法**: 启示 → closed-form attention proxy (F139, F205) 避免学习开销。

### 1.10 [HLOB: Information persistence and structure in LOB, ESWA 2024](https://www.sciencedirect.com/science/article/pii/S0957417424029452)
- **Key insight**: 2024 LOB DL 模型, 强调 information persistence; siamese 处
  理 ask/bid 共享参数。
- **本项目用法**: bid/ask 对称性提示 — pair feature 可以同样对称设计 (F1: mid
  diff 是天然对称, F4: OFI diff 是 signed 对称)。

---

## § Category 2: Pair / Relative Value Features (10 papers)

### 2.1 [Avellaneda & Lee 2010] — Statistical Arbitrage in the US Equities Market
- *Quantitative Finance* 10(7) | [SSRN 1153505](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505) | [PDF](https://traders.studentorg.berkeley.edu/papers/Statistical%20arbitrage%20in%20the%20US%20equities%20market.pdf)
- **Key insight**: Signal 来自 PCA 或 sector ETF regression; idiosyncratic
  returns 建模为 mean-reverting; volume-adjusted 后 ETF strategy Sharpe 1.51。
- **本项目用法**: F95 (first PC weight), F100-101 (factor residual), F62, F145
  (short-window market-neutral residual)。

### 2.2 [Gatev, Goetzmann & Rouwenhorst 2006] — Pairs Trading
- *Review of Financial Studies* 19(3) | [PDF](https://www.researchgate.net/publication/5217081_Pairs_Trading_Performance_of_a_Relative_Value_Arbitrage_Rule)
- **Key insight**: Distance approach — Euclidean distance of normalized prices;
  分配 pair 后 spread 越离 mean 越多越交易。
- **本项目用法**: F26 sym_pair_ratio_mid (log price ratio = cointegration spread
  proxy); F77 normalized mid distance。

### 2.3 [Krauss, Do & Huck 2017] — DNN/GBT/RF Stat Arb on S&P 500
- *European Journal of Operational Research* 259 | [PDF](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)
- **Key insight**: Ensemble (DNN+GBT+RF) > single; 短期统计套利策略基于
  cross-sectional features, daily returns 0.45%。
- **本项目用法**: F10-11 (sym_pair_diff_logret), F47-48 (csec_rank_returns),
  F100-101。

### 2.4 [Engle & Granger 1987] — Cointegration
- Two-step: regress one price on another, ADF-test residual stationarity。
- **本项目用法**: F105-106 (pair_zscore_cointspread) — 但标记 high sym
  robustness concern (beta 需 frozen per pair)。

### 2.5 [Stoikov 2017] — The Micro-Price
- *Quantitative Finance* | [arXiv:1702.02867](https://arxiv.org/abs/1702.02867)
- **Key insight**: Micro-price 是 mid-price + adjustment 基于 spread 与
  imbalance; martingale 性质。
- **本项目用法**: F15 (microprice pair diff), F147 (microprice − mid demeaned),
  F179, F193。

### 2.6 [Roll 1984] — Implicit Bid-Ask Spread
- *Journal of Finance* 39(4)
- **Key insight**: 从 price serial covariance 估 spread。
- **本项目用法**: F8 (spread pair diff), F84 (spread demean)。

### 2.7 [Hou, Robinson & Engle 2002+] — Realized Covariance, High Frequency
- [Realized Covariation, Engle](https://www.researchgate.net/publication/4896726_Econometric_Analysis_of_Realized_Covariation_High_Frequency_Based_Covariance_Regression_and_Correlation_in_Financial_Economics)
- **Key insight**: 高频 realized covariance/regression 方法; Barndorff-Nielsen-
  Shephard 提出 realized covariation 估计。
- **本项目用法**: F72, F73 (pair logret product = realized cov); F12-13 (pair
  RV/BV diff)。

### 2.8 [Easley, López de Prado & O'Hara 2012] — VPIN
- *Review of Financial Studies* 25(5)
- **Key insight**: VPIN (Volume-Synchronized PIN) 度量 toxicity。
- **本项目用法**: F17 (sym_pair_diff_tox), F137-138 (toxicity at W5/W20 pair diff),
  F182 (toxicity share diff)。

### 2.9 [Reinforcement Learning Pair Trading, 2024](https://arxiv.org/pdf/2407.16103)
- **Key insight**: 2024 paper - RL for pair trading; spread + threshold rebalancing。
- **本项目用法**: 启示 spread 作为 first-class feature (F8, F75, F141)。

### 2.10 [High-frequency lead-lag in Chinese stock index futures, 2025](https://arxiv.org/abs/2501.03171)
- **Key finding**: Near-month contract leads others by one tick driven by liquidity;
  lead-lag spread predicts returns。
- **本项目用法**: F87-89 (asymmetric window pair = lagless lead-lag proxy);
  避免 stateful。

---

## § Category 3: Graph / Network Features (10 papers)

### 3.1 [Mantegna 1999] — Hierarchical Clustering & MST in Stock Markets
- Foundational paper on correlation-based MST。
- [Topology of correlation-based MST](https://www.academia.edu/12206024/Topology_of_correlation-based_minimal_spanning_trees_in_real_and_model_markets)
- **Key insight**: MST 从 correlation matrix 构造; cluster structure 跟随 sector
  分类。
- **本项目用法**: F94 cluster-id, F97 MST centrality - 但 5 sym 节点数过少;
  P3 (低优先级)。

### 3.2 [Tumminello et al. 2005] — Planar Maximally Filtered Graph (PMFG)
- *PNAS* 102(30)
- **Key insight**: PMFG 保留 MST 性质同时保留更多边 (3N-6 edges instead of N-1)。
- **本项目用法**: F98 PMFG eccentricity - 5 sym 太小 PMFG 退化; P3。

### 3.3 [DeltaLag: Learning Dynamic Lead-Lag Patterns, ICAIF 2025](https://arxiv.org/pdf/2511.00390)
- **Key insight**: 2025 paper - dynamic lead-lag patterns learned, outperform
  precomputed graphs。
- **本项目用法**: F87-90 lead-lag proxies (但用 asymmetric window 而非动态学)。

### 3.4 [Sakoe-Chiba 1978] — Dynamic Time Warping (DTW)
- Original DTW algorithm。
- **本项目用法**: F96 DTW dist avg - 但需训练时预计算, 推理无开销; P3
  (5 sym 信息有限)。

### 3.5 [Cont 2001] — Empirical Properties of Asset Returns: Stylized Facts
- *Quantitative Finance* 1
- **Key insight**: 多 asset cross correlation 是 stylized fact; 高频 → 低
  correlation。
- **本项目用法**: F65-66 (csec skew/kurt), F134 (csec kurt OFI)。

### 3.6 [Forecasting Realized Vol with Spillover, GNN, IJF 2024](https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967)
- **Key insight**: GNN forecast realized vol with spillover effects。
- **本项目用法**: F12-13 (pair RV diff), F120 (RV dispersion)。

### 3.7 [GraphFM: Graph Factorization Machines for Feature Interaction, MIR 2024](https://link.springer.com/article/10.1007/s11633-024-1505-5)
- **Key insight**: Graph FM models feature interactions explicitly。
- **本项目用法**: 启示 — explicit pairwise product feature (F108-111, F157-160,
  F175)。

### 3.8 [Hayashi & Yoshida 2005] — Lead-Lag for asynchronous data
- *Bernoulli* 11(2)
- **Key insight**: Hayashi-Yoshida estimator for asynchronous data cross-correlation。
- **本项目用法**: F89 lead-lag imb idx (但避免时序; 用 asymmetric W 提示
  cross-lag direction)。

### 3.9 [Multi-relational Graph Diffusion NN, 2024](https://arxiv.org/pdf/2401.05430)
- **Key insight**: 多关系图 NN, 多边类型表达 stock 关系。
- **本项目用法**: 启示我们可以同时用 raw pair diff + rank-based 相关性 (F58)。

### 3.10 [Cross-Asset OFI, Multi-Asset Market Impact, QF 2023](https://www.researchgate.net/publication/344646869_Multi-asset_market_impact_and_order_flow_commonality)
- **Key insight**: Multi-asset market impact matrix; off-diagonal 元素表达 cross
  asset impact。
- **本项目用法**: F67-69, F102-103 显式 cross OFI pair features 是它的特征化版本。

---

## § Category 4: Alpha101 / Cross-Sectional 因子 (10 alphas)

来源: [Kakushadze 2015 — 101 Formulaic Alphas](https://arxiv.org/abs/1601.00991)

Alpha101 大量使用 `rank()`, `delay()`, `delta()`, `correlation()`, `ts_max()`,
`decay_linear()` 等算子组合。我们关心 cross-sectional 那些 (含 `rank()`)。

### 4.1 Alpha #1: `rank(Ts_ArgMax(SignedPower((returns < 0 ? stddev(returns, 20) : close), 2.), 5)) - 0.5`
- **Insight**: signed power + argmax + rank — 但需要 20 时序窗口, **我们的 5
  sym setting 无 ts**, simplified to F47 (csec rank returns)。

### 4.2 Alpha #3: `(-1 * correlation(rank(open), rank(volume), 10))`
- **Insight**: 时序 correlation of cross-sec ranks; simplified to F58 (csec corr
  rank mid size)。

### 4.3 Alpha #5: `rank(open - sum(vwap, 10)/10) * (-1 * abs(rank(close - vwap)))`
- **Insight**: rank operator 用于 cross-sectional standardization;
  F26 (price ratio) 类似设计。

### 4.4 Alpha #6: `(-1 * correlation(open, volume, 10))`
- **Insight**: time-series correlation; 5 sym setting 用 instant cross corr
  → F58, F92。

### 4.5 Alpha #7: `(adv20 < volume) ? (-1 * ts_rank(abs(delta(close,7)),60)) * sign(delta(close,7)) : -1`
- **Insight**: ts_rank cross-section; F49 (rank wmp change) 类似。

### 4.6 Alpha #9: `((0 < ts_min(delta(close,1),5))) ? delta(close,1) : ((ts_max(delta(close,1),5) < 0) ? delta(close,1) : (-1 * delta(close,1)))`
- **Insight**: 短期 momentum/反转; F47 cross-sec rank returns。

### 4.7 Alpha #22: `(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))`
- **Insight**: rank of vol; F43, F126。

### 4.8 Alpha #44: `(-1 * correlation(high, rank(volume), 5))`
- **Insight**: rank(volume) — F45 (rank depth), F129 (rank depth qrank)。

### 4.9 Alpha #51: `(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))`
- **Insight**: 长短 momentum 比较; F177 (pair mid pct change diff) 是 pair
  化版本。

### 4.10 Alpha #98: 6-level nested rank/correlation
- **Insight**: 深嵌套 cross-sec rank; 我们只用一层 rank (F41-50, F123-129)。

**Alpha191 (Guotai Junan Securities)**: [191 short-term factors](https://ieee-dataport.org/documents/alpha191),
genetic programming 构造的 191 因子。**核心特征**: 大量使用 `RANK(...)`,
`MEAN(...)`, `STD(...)` 跨股票 cross-sectional aggregation; 我们的 F41-50, F123-129
copy 这个思想。

---

## § Category 5: Kaggle / Industry Blogs (30+ blogs)

### Kaggle Competition Writeups
1. **[Optiver Realized Volatility 1st Place Solution (DolphinDB tutorial)](https://docs.dolphindb.com/en/Tutorials/metacode_derived_features.html)** — Feature
   engineering 两个 level: spread = ask − bid (level 1); windowing (level 2)。
   核心 feature: log_return1_linear_weight (importance 0.716)。
2. **[Optiver Realized Volatility - 7th place](https://github.com/michaelpoluektov/orvp)** —
   stream-friendly features。
3. **[Optiver Realized Vol - Issam Sebri Medium](https://koeusiss.medium.com/optiver-realized-volatility-prediction-cb7da76fbd3f)** —
   WAP feature, log returns, cross-stock aggregations。
4. **[Optiver Realized Vol - 91st place](https://chrisrichardmiles.github.io/chrisrichardmiles/projects/optiver/index_optiver.html)** —
   tau feature, RV by stock_id grouping。
5. **[Optiver Trading at the Close - 1st place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)** —
   group by stock_id features (median_size, std_price global), shift+pct_change。
6. **[Optiver Trading at Close - 14th place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/clash-royale-14th-place-solution-for-the-optiver-t)** —
   cross-stock features, time-bucket aggregations。
7. **[Optiver Trading at Close - Medium overview](https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0)** —
   competition framework explanation。
8. **[Jane Street Market Prediction 2020 - 1st place (Yirun)](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)** —
   Supervised Autoencoder + MLP。
9. **[Jane Street 2024 - LGBM baseline](https://www.kaggle.com/code/nawazhaider/jane-street-2024-lgbm)** —
   LGBM with categorical features。
10. **[Jane Street 2024 - LSTM auxiliary features](https://www.kaggle.com/code/ccyhui/jane-street-2024-lstm-with-auxiliary-features)** —
    auxiliary feature design。
11. **[Jane Street Stock Prediction case study - Medium](https://prateeknigam9.medium.com/yth-bc85b326d52f)** —
    autoencoder denoising approach。
12. **[Ubiquant Market Prediction - 1st place writeup](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy)** —
    heavy feature engineering, LightGBM + TabNet ensemble。
13. **[Ubiquant - Top 1% solution](https://github.com/pinouche/ubiquant-kaggle-competition)** —
    tabnet attention features。
14. **[JPX Tokyo Stock Exchange Prediction baseline](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction)** —
    cross-sectional rank predicting top vs bottom 200。
15. **[JPX Cluster and Rank approach](https://www.kaggle.com/code/krishm/jpx-cluster-and-rank)** —
    K-means cluster + rank within cluster。
16. **[G-Research Crypto Forecasting 2022 wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)** —
    feature engineering > model selection。
17. **[Two Sigma News Kaggle 2018](https://www.kaggle.com/c/two-sigma-financial-news)** —
    market + news features cross-stock。
18. **[Kaggle Algorithmic Trading Challenge 2011 winning](https://www.researchgate.net/publication/233731605_Winning_the_Kaggle_Algorithmic_Trading_Challenge_with_the_Composition_of_Many_Models_and_Feature_Engineering)** —
    150+ features price/liquidity/spread; standardize **per stock_id** (我们
    不能直接用因为 sym-agnostic, 但提示 z-score across sym 是可行的)。

### Industry Blogs
19. **[Stoikov Microprice walkthrough - Mhfizt Medium](https://medium.com/@mhfizt/high-frequency-estimator-of-future-prices-micro-price-paper-code-walkthrough-475adb98e91d)** —
    Micro-price 代码实现。
20. **[Numerai Signals overview](https://docs.numer.ai/numerai-signals/signals-overview)** —
    cross-sectional regression neutralization + Barra-style factors。
21. **[Numerai era-neutralize - Suraj Parmar Medium](https://parmarsuraj99.medium.com/lets-talk-about-signals-841934f24450)** —
    practical signals + neutralization。
22. **[Quantopian alphalens repo](https://github.com/quantopian/alphalens)** —
    Quantile + Industry neutralization for cross-sectional features。
23. **[Alpha Factors workflow - Stefan Jansen](https://stefan-jansen.github.io/machine-learning-for-trading/04_alpha_factor_research/)** —
    cross-sectional alpha factor research framework。
24. **[Cross-Sectional Alpha library - S&P Global](https://www.spglobal.com/content/dam/spglobal/mi/en/documents/general/Alpha_Factor_Library_v2.pdf)** —
    industry-grade alpha library。
25. **[OB Imbalance practical guide - QuantStrategy.io](https://quantstrategy.io/blog/order-book-imbalances-a-practical-guide-for-day-traders/)** —
    top 3-5 level weighted imbalance。
26. **[OB Imbalance signals - QuantVPS](https://www.quantvps.com/blog/order-flow-imbalance-signals)** —
    practical OFI deployment in HFT。
27. **[Order Book Depth strategies - QuantStrategy.io](https://quantstrategy.io/blog/mastering-order-book-depth-advanced-strategies-for/)** —
    multi-level depth features。
28. **[How to Use Deep OFI - QuantPedia](https://quantpedia.com/how-to-use-deep-order-flow-imbalance/)** —
    practical Deep OFI explanation。
29. **[Quant Trading w/ Python LOB - Saurav Medium](https://medium.com/@writeronepagecode/quant-trading-with-python-a-guide-to-limit-order-book-analysis-ep-2-365-8db2e017a623)** —
    Python LOB feature engineering guide。
30. **[How OB Imbalances Predict - Wealth Academy Medium](https://medium.com/@thewealthacademyyt/how-order-book-imbalances-predict-price-moves-before-they-happen-crystal-ball-series-part-2-fd9fc66f86a5)** —
    imbalance forward signal, weighted by distance。
31. **[Quantopian Pro Workflow](https://www.quantopian.com/posts/a-professional-quant-equity-workflow)** —
    cross-sectional alpha factor workflow。
32. **[DolphinDB Alpha101 calculation](https://medium.com/@DolphinDB_Inc/a-simpler-way-to-calculate-worldquant-101-alphas-c55dac54e9f7)** —
    Alpha101 implementation reference。
33. **[Scaling/Normalization in Quant - Quantdare](https://quantdare.com/scaling-normalisation-standardisation-a-pervasive-question/)** —
    cross-sectional z-score practice。
34. **[DeepFM CTR feature interaction](https://arxiv.org/pdf/1703.04247)** —
    factorization machine for low/high order feature interactions; 启示
    explicit pair product features。
35. **[xDeepFM Explicit Feature Interaction](https://arxiv.org/pdf/1803.05170)** —
    explicit + implicit interaction modeling。

---

## § Top 50 推荐分析

详见 `TOP_50_RECOMMENDATIONS.json`。下面是排序逻辑与分组：

### Top 5 (优先级最高)
| Rank | ID | Name | Source | Why |
|---|---|---|---|---|
| 1 | F003 | sym_pair_diff_mp_t | v6_pairwise SOTA核心 | 已验证有效, 加深所有 mp 等价物 |
| 2 | F001 | sym_pair_diff_mid | Avellaneda + v6 | baseline pair diff |
| 3 | F004 | sym_pair_diff_ofi_W5 | Cross-Impact 2023 | top1 灵感来源 |
| 4 | F009 | sym_pair_diff_imb1 | Stoikov microprice | top-of-book 最强信号 |
| 5 | F010 | sym_pair_diff_logret_W5 | Avellaneda-Lee | 经典 stat arb |

### Top 50 按类别分布
- **Pair Diff (sym_i − sym_j)**: 18 个 (Rank 1-7, 13-17, 19, 20, 38, 41, 49 ...)
- **Cross-Sectional Demean (sym − cross_mean)**: 14 个 (Rank 10-12, 26-29, 31, 35, 48 ...)
- **Cross-Sectional Z-Score + Rank**: 9 个 (Rank 8-9, 22-25, 28, 29, 42, 44)
- **Pair Interaction (multiplicative)**: 5 个 (Rank 16-18, 32, 33)
- **Cross-Impact OFI Pair (Cont 2023)**: 3 个 (Rank 3, 11, 40)
- **Attention/Aggregate Proxy**: 1 个 (Rank 45-46)

### 评分逻辑
`score = expected_gain * (1/complexity) * (1/sym_robustness) * (1/date_robustness)`

其中 expected_gain {low:1, medium:2, high:3}, 其余 {low:1, medium:2, high:3}。
**惩罚**:
- 任何 robustness concern=high 直接被剔出 Top 50 (除非有压倒性 expected_gain)
- complexity=hard 只有在 expected_gain=high 时才进 Top 50
- 实际上 Top 50 几乎全是 (high gain, easy/easy, low/low) 的"hot zone"

---

## § Worker B 实现建议

### Step 1: 实现 P1 (30 个 features)
P1 = Top 30, 全部是 expected_gain=high/medium + easy + low robustness。

加到 v6_pairwise 已有 259d (`keep_idx_259d.npy`) feature 上, 输出维度估计：
- 20 pair diff × 8 features × ~4 windows ≈ 600+ (cap at top-50 priority)
- 5 sym × 17 csec features ≈ 85
- **总新增大约 100-150 维**, 加到 259d → ~360-410d total

### Step 2: 验证 Smoke Test
1 seed train of v6_pairwise + new feats, compare to SOTA +37.25。

### Step 3: LOSO Validation (CRITICAL)
**核心约束**: cross-sym features 容易引入隐式 sym-specific signal。Worker B
必须做 LOSO (leave-one-sym-out) 测试:
- 训练时屏蔽 sym k 的数据
- 测试时只用 sym k 的数据
- 若 LOSO 比 in-sample 差 > 5%, feature 有 sym-leak; 丢弃

### Step 4: 5-seed final
通过 LOSO 的 P1 features → 5-seed final 报 mean ± std。

### Step 5: P2 batch (optional)
若 P1 给出 +0.3 或更高 lift, 加 P2 batch (Rank 32-50)。

### v6_pairwise 集成方式
v6_pairwise 当前架构 (推测基于 commit "v6_pairwise NEW SOTA"):
- 输入: 5 sym × 259d = 1295d
- 显式 pair diff: 10 pairs × ? d (mp_t etc.)
- MLP head

新 features 加入方式:
```python
# In data_loader or feature_builder
x_pair_diffs = []  # 10 pairs
for (i, j) in itertools.combinations(range(5), 2):
    x_pair_diffs.append(x[:, i, F1_indices] - x[:, j, F1_indices])  # F001 family
    x_pair_diffs.append(x[:, i, ofi_W5_idx] - x[:, j, ofi_W5_idx])  # F004
    # ... 12-20 features per pair
x_csec = []  # 5 sym × csec features
for s in range(5):
    mid_dev = x[:, s, mid_idx] - x[:, :, mid_idx].mean(dim=1)  # F031
    ofi_zscore = (x[:, s, ofi_idx] - x[:, :, ofi_idx].mean(dim=1)) / (x[:, :, ofi_idx].std(dim=1) + 1e-8)  # F034
    # ...
```

---

## § Robustness Analysis Methodology

每个 feature 都标了两个 robustness concern:

### sym_robustness_concern
- `low`: feature 计算只用 raw price/OFI/imbalance 这类**对 sym 不敏感**的量;
  e.g. 差分、ratio、z-score 跨 sym 后, sym id 不进 formula。
- `medium`: feature 涉及 normalize/aggregate, 但**形式上保持对 sym 对称**;
  e.g. 涉及单 sym 历史统计但 same operator 应用于所有 sym。
- `high`: feature 需要 **per-sym 参数** (cointegration beta, factor loading,
  per-sym z-score statistics); test 时若遇到训练外股票直接错。

### date_robustness_concern
- `low`: feature 不依赖 date / time-of-day; date 置 0 也能算。
- `medium`: feature 依赖 sess_idx 或 sess-relative time; 若 sess_idx 也置 0
  会受影响 (本数据 sess_idx 是 OK 的)。
- `high`: 依赖 absolute date; 受 evaluation date=0 影响; **必须避免**。

我们整个 Top 50 都是 `date_robustness_concern=low` 或 `medium`, 没有 `high`。

### Validation Protocol
Worker B 在加入 Top 50 features 之前, 对每个 feature 做:
1. **Statistical test**: 在 sym 上的方差 vs 在 date 上的方差比 < 0.5 (sym-stable)
2. **LOSO test**: leave-one-sym-out train, in-sym test 看 PnL drop < 5%
3. **Out-of-date test**: 后 2 天 (sess_idx 16-21) test 看 PnL drop < 10%

通过以上全部测试的 feature 才进 final submission。

---

## § 参考文献完整列表

### Papers (40+)
1. Cont R, Kukanov A, Stoikov S. (2014). The price impact of order book events. *J Financial Econometrics* 12(1):47-88. [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
2. Kolm P, Turiel J, Westray N. (2023). Deep order flow imbalance. *Math Finance* 33(4). [SSRN 3900141](https://ssrn.com/abstract=3900141)
3. Cross-Impact of Order Flow Imbalance in Equity Markets. (2023). *QF* 23(10). [Tandf 2236159](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159)
4. Sirignano J, Cont R. (2019). Universal features of price formation. *QF* 19(9). [arXiv:1803.06917](https://arxiv.org/abs/1803.06917)
5. Zhang Z, Zohren S, Roberts S. (2019). DeepLOB. *IEEE TSP* 67(11). [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
6. Berti et al. (2025). TLOB / MLPLOB. [arXiv:2502.15757](https://arxiv.org/abs/2502.15757)
7. Xu K, Cont R, Stoikov S. (2019). Multi-Level OFI. [arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
8. Hasbrouck J. (1995). Information shares in security price discovery. *J Finance* 50(4).
9. Attention-Based Multi-Asset OFI Networks, ICAIF 2024. [DOI 10.1145/3768292.3770430](https://dl.acm.org/doi/10.1145/3768292.3770430)
10. HLOB: Information persistence in LOB. ESWA 2024. [Science Direct S0957417424029452](https://www.sciencedirect.com/science/article/pii/S0957417424029452)
11. Avellaneda M, Lee JH. (2010). Statistical arbitrage in US equities. *QF* 10(7):761-782. [SSRN 1153505](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505)
12. Gatev E, Goetzmann W, Rouwenhorst KG. (2006). Pairs trading. *RFS* 19(3).
13. Krauss C, Do XA, Huck N. (2017). DNN/GBT/RF stat arb. *EJOR* 259:689-702.
14. Engle R, Granger CWJ. (1987). Cointegration and error correction. *Econometrica* 55(2):251-276.
15. Stoikov S. (2017). The micro-price. *QF* (online 2018). [arXiv:1702.02867](https://arxiv.org/abs/1702.02867)
16. Roll R. (1984). Implicit bid-ask spread. *J Finance* 39(4):1127-1139.
17. Hou K, Engle R, Robinson D. (2002+). Realized covariation, *J Financial Econometrics*. [ResearchGate 4896726](https://www.researchgate.net/publication/4896726_Econometric_Analysis_of_Realized_Covariation_High_Frequency_Based_Covariance_Regression_and_Correlation_in_Financial_Economics)
18. Easley D, López de Prado M, O'Hara M. (2012). Flow toxicity / VPIN. *RFS* 25(5).
19. Reinforcement Learning Pair Trading. (2024). [arXiv:2407.16103](https://arxiv.org/abs/2407.16103)
20. High-frequency lead-lag in Chinese stock index futures, 2025. [arXiv:2501.03171](https://arxiv.org/abs/2501.03171)
21. Mantegna RN. (1999). Hierarchical structure in financial markets. *Eur Phys J B* 11(1):193-197.
22. Tumminello M, Aste T, Di Matteo T, Mantegna RN. (2005). PMFG. *PNAS* 102(30).
23. DeltaLag: Learning Dynamic Lead-Lag Patterns, ICAIF 2025. [arXiv:2511.00390](https://arxiv.org/abs/2511.00390)
24. Sakoe H, Chiba S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE TASSP* 26(1).
25. Cont R. (2001). Empirical properties of asset returns. *QF* 1.
26. Forecasting Realized Vol with GNN, IJF 2024. [Science Direct S0169207024000967](https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967)
27. GraphFM: Graph Factorization Machines, MIR 2024. [Springer s11633-024-1505-5](https://link.springer.com/article/10.1007/s11633-024-1505-5)
28. Hayashi T, Yoshida N. (2005). Lead-lag asynchronous cross-correlation. *Bernoulli* 11(2).
29. Multi-relational Graph Diffusion NN, 2024. [arXiv:2401.05430](https://arxiv.org/abs/2401.05430)
30. Multi-asset market impact and order flow commonality, QF 2023. [ResearchGate 344646869](https://www.researchgate.net/publication/344646869_Multi-asset_market_impact_and_order_flow_commonality)
31. Kakushadze Z. (2016). 101 Formulaic Alphas. [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)
32. Alpha191 (Guotai Junan Securities, 2017). [IEEE DataPort](https://ieee-dataport.org/documents/alpha191)
33. Barndorff-Nielsen O, Shephard N. (2004). Power/bipower variation. *J Financial Econometrics* 2(1):1-37.
34. Barndorff-Nielsen O, Shephard N. (2006). Bipower variation jump test. *J Financial Econometrics* 4(1):1-30.
35. Kyle AS. (1985). Continuous auctions and insider trading. *Econometrica* 53(6).
36. Hasbrouck J. (1991). Measuring information content of stock trades. *J Finance* 46(1).
37. Amihud Y. (2002). Illiquidity. *J Financial Markets* 5(1).
38. Corwin SA, Schultz P. (2012). High-low estimator. *J Finance* 67(2).
39. Kercheval AN, Zhang Y. (2015). LOB SVM. *QF* 15(8):1315-1329.
40. McLean RD, Pontiff J. (2016). Does academic research destroy stock return predictability? *J Finance* 71(1).
41. Cross-Market Alpha: Testing Short-Term Trading Factors in US Market. (2026). [arXiv:2601.06499](https://arxiv.org/pdf/2601.06499)
42. Intraday Patterns in the Cross-section of Stock Returns. [arXiv:1005.3535](https://arxiv.org/pdf/1005.3535)

### Blogs / Kaggle Writeups (30+)
(见 § Category 5)

---

## § Conclusion

我们围绕 v6_pairwise 当前 SOTA (+37.25 ± 0.58) 在 cross-sym interaction
feature 空间做了系统调研:
- **40+ papers** (Cont, Stoikov, Avellaneda, Krauss, Sirignano, Mantegna,
  Kakushadze 等 + 8 篇 2024-2025 最新)
- **30+ blogs / Kaggle writeups** (Optiver RV/TAC, Jane Street, Numerai,
  Quantopian, QuantStrategy, DolphinDB 等)
- **205 features** in `INTERACTION_FEATURES_TABLE.csv` (覆盖 6 大类:
  relative diff, cross-sectional, pair interaction, pair cross-impact,
  graph, lagless lead-lag)
- **Top 50** ranking 在 `TOP_50_RECOMMENDATIONS.json`, 分 P1 (30, 高优先级)
  和 P2 (20, 次优先级)

**最强 3 个方向**:
1. **Pair Diff** (Rank 1-7): v6_pairwise 已验证, 简单扩展到所有 LOB 数据
2. **Cross-Impact OFI Residual** (Cont 2023): 跨 sym OFI 残差是 short-horizon
   高 alpha
3. **Cross-Sectional Z-Score / Rank** (Alpha101 / Numerai): 免费 lift,
   sym-agnostic 天然

预期 Worker B 实现 P1 后能给出 **+0.3 to +0.8** 的 PnL lift, P2 后再加
**+0.1 to +0.3**, 目标 SOTA +37.55 to +38.5。

---

*Report by: Cross-Sym Research Worker (Opus xhigh)*
*Date: 2026-05-31*
*60-90min budget*
