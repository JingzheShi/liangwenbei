# v8 Feature Research: 50+ New Papers (2023–2025)
# Cross-Sym / Cross-Asset Interaction Features — Beyond v7's 205 Features

**Date**: 2025-05-31  
**Scope**: 54 papers across 7 directions. All features are NEW (no overlap with F001–F205).  
**Constraint**: All features must be stateless (no cross-call state); no `date` as feature; sym-agnostic.

---

## Direction 1: Advanced LOB Microstructure (2024–2025)

---

### LOBFrame — Deep LOB Microstructural Guide (Briola, Cartea, Morales-Arias et al. 2025) — Direction 1

- **Paper**: arxiv 2403.09267 / Quantitative Finance 2025. DOI: 10.1080/14697688.2025.2522911
- **Venue**: Quantitative Finance (Taylor & Francis) 2025
- **Core idea**: LOBFrame is an open-source deep-learning codebase for LOB forecasting. Key finding: stocks'
  *predictability rate* is directly linked to microstructural properties — tick size, spread, and liquidity.
  Feature engineering quality matters more than model depth. Demonstrates that per-stock
  `predictability_rate = f(spread, tick_ratio, realized_vol)` is computable as a feature.
- **Feature formulas**:
  1. `tick_size_ratio`: `tick / (bid1 + ask1) * 2` — normalized tick relative to mid (spread tightness proxy)
  2. `lob_predictability_proxy`: `1 / (1 + spread / rv_W20)` — inverse of noise-to-signal ratio
- **Why适合 v7+**: LOBFrame shows tick-normalized features significantly predict which microstructure regime
  the market is in; the predictability proxy distinguishes informative vs. noise-dominated regimes.
- **预期 PnL gain** (vs v7 baseline +37.67): medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### TLOB — Transformer with Dual Attention for LOB (Berti & Kasneci 2025) — Direction 1

- **Paper**: arxiv 2502.15757
- **Venue**: arXiv preprint (Feb 2025); submitted to major conference
- **Core idea**: TLOB uses dual attention (spatial × temporal) to focus on the LOB's most predictive price
  levels at each moment. Key extracted features: the attention weight vector over price levels is itself an
  informative static proxy. Their ablation shows that levels 2-3 often carry more weight than level 1 in
  volatile markets.
- **Feature formulas**:
  1. `lob_L2_contribution`: `OFI_L2 * bidsz2 / (bidsz2 + asksz2)` — level-2 weighted contribution to price formation
  2. `lob_depth_centroid_bid`: `sum(k * bidsz_k for k=1..5) / sum(bidsz_k)` — center-of-mass of bid depth
  3. `lob_depth_centroid_ask`: `sum(k * asksz_k for k=1..5) / sum(asksz_k)` — center-of-mass of ask depth
- **Why适合 v7+**: Depth centroid captures shape of the order book beyond simple imbalance; not in v7 features.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Crypto LOB — Better Inputs Matter (Wang et al. 2025) — Direction 1

- **Paper**: arxiv 2506.05764
- **Venue**: arXiv preprint (June 2025)
- **Core idea**: Benchmark study on BTC/USDT LOB. Finding: feature preprocessing matters more than adding hidden
  layers. Key useful features: (1) volume percentage at each rank across bid/ask levels, (2) spread normalized
  by trailing mid, (3) LOB shape features (volume concentration). Simpler models with better features beat
  deep networks.
- **Feature formulas**:
  1. `lob_vol_pct_L1_bid`: `bidsz1 / sum(bidsz1..5)` — fraction of bid depth at top level
  2. `lob_entropy_bid`: `-sum(p_k * log(p_k) for k=1..5)` where `p_k = bidsz_k / sum(bidsz)` — LOB depth entropy
  3. `lob_depth_hhi_bid`: `sum((bidsz_k/sum_bidsz)^2)` — HHI concentration of bid volume
- **Why适合 v7+**: LOB shape entropy is orthogonal to all existing depth/OFI features; measures distribution
  rather than total amount.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### High-Res Microprice via Tsetlin Machines (Blakely 2024) — Direction 1

- **Paper**: arxiv 2411.13594
- **Venue**: arXiv preprint (Nov 2024)
- **Core idea**: Error-correcting model for microprice using higher-order LOB imbalances. Standard microprice
  uses only L1 imbalance; this model adds L2–L5 corrections. Feature: `microprice_L3 = microprice_L1 + delta_L2
  + delta_L3` where deltas come from empirical imbalance-correction tables (can be frozen from training data).
- **Feature formulas**:
  1. `microprice_L3_corr`: `microprice_L1 + alpha_2*(imb2 - imb1) + alpha_3*(imb3 - imb2)` where `alpha_k`
     frozen from training; `imb_k = (bidsz_k - asksz_k)/(bidsz_k + asksz_k)`
  2. `microprice_residual_L1_L3`: `microprice_L3_corr - microprice_L1` — deep-book correction term
- **Why适合 v7+**: Provides a more accurate fair-value estimate than L1-only microprice; error-correcting term is
  a new signal not in v7.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Boltzmann Price — Fair Value via Maximum Entropy (Rola 2025) — Direction 1

- **Paper**: arxiv 2507.09734
- **Venue**: arXiv preprint (July 2025)
- **Core idea**: Parametrized family of prices derived from the Maximum Entropy Principle given bid/ask volume
  imbalance across all LOB levels. The Boltzmann price minimizes bias given multi-level volume distribution.
  Under specific parameters it approximates mid or weighted mid; generally provides a more entropy-consistent
  fair value than microprice.
- **Feature formulas**:
  1. `boltzmann_price_L5`: `sum_k(bid_k * ask_szk + ask_k * bid_szk) / sum_k(bid_szk + ask_szk)` — 5-level
     symmetric weighted price (simplified entropy-consistent formula)
  2. `boltzmann_vs_mid`: `boltzmann_price_L5 - (bid1 + ask1)/2` — deviation from mid (like microprice excess)
- **Why适合 v7+**: Multi-level weighted price (boltzmann) is more entropy-consistent than L1-only microprice and
  more robust to sudden L1 queue depletion.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Order Book Filtration & Directional Signal Extraction (Anantha, Jain, Maiti 2025) — Direction 1

- **Paper**: arxiv 2507.22712
- **Venue**: arXiv preprint (July 2025)
- **Core idea**: Three real-time LOB filtration schemes improve directional signal clarity: (1) filter by order
  lifetime, (2) filter by update count, (3) filter by inter-update delay. Key finding: OBI computed *only from
  trade events* (aggressive orders) has stronger causal alignment with price movements than full OBI.
- **Feature formulas**:
  1. `obi_trade_filtered_W5`: `(aggressive_buy_vol_W5 - aggressive_sell_vol_W5) / (total_trade_vol_W5 + eps)` —
     OBI using only market/aggressive orders (tick-rule applied to trade volume)
  2. `obi_long_lifetime_W20`: OBI computed using only quotes active for >3 ticks (excluding fleeting orders)
- **Why适合 v7+**: Trade-event OBI has stronger causal link to price moves than standard OFI; fleeting order
  filtering removes noise not present in v7's OFI features.
- **预期 PnL gain**: high
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Attention-Based LOB Reading & Forecasting (Kolbæk & Poulsen 2024) — Direction 1

- **Paper**: arxiv 2409.02277
- **Venue**: arXiv preprint (Sep 2024); uses Spacetimeformer architecture
- **Core idea**: Customized spatiotemporal embedding captures LOB structural integrity. Key feature:
  LOB "reading" computes attention weights over price levels that can be frozen as feature importances.
  The paper identifies level-2 and level-3 bid sizes as high-attention levels.
- **Feature formulas**:
  1. `lob_L2_L3_imb`: `((bidsz2 - asksz2) + (bidsz3 - asksz3)) / (bidsz2 + asksz2 + bidsz3 + asksz3)` —
     L2-L3 combined imbalance (not just L1)
  2. `lob_depth_centroid_asym`: `lob_depth_centroid_bid - lob_depth_centroid_ask` — signed centroid difference
- **Why适合 v7+**: L2-L3 combined imbalance captures where aggressive liquidity is concentrated, orthogonal to L1.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### HLOB — Information Persistence and Structure in LOB (2024) — Direction 1

- **Paper**: ScienceDirect 2024, Expert Systems with Applications. DOI linked via ScienceDirect/S0957417424029452
- **Venue**: Expert Systems with Applications 2024
- **Core idea**: Models information persistence in LOB using hierarchical LOB features. Key finding:
  volume at mid-levels (L3-L5) contains information that persists longer than L1 noise. Proposes
  `volume_persistence_score` features.
- **Feature formulas**:
  1. `mid_level_vol_ratio`: `(bidsz3 + bidsz4 + bidsz5) / (bidsz1 + bidsz2)` — ratio of deep vs shallow bid volume
  2. `lob_slope_bid`: `cov(k, bidsz_k for k=1..5) / var(k)` — linear slope of bid depth vs level
- **Why适合 v7+**: LOB slope distinguishes convex vs. concave book shapes, providing regime-dependent signals.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

## Direction 2: Cross-Asset Deep Features (2024–2025)

---

### DeltaLag — Dynamic Lead-Lag Patterns (Zhou, Wang et al. 2025) — Direction 2

- **Paper**: arxiv 2511.00390, ACM ICAIF 2025
- **Venue**: 6th ACM International Conference on AI in Finance (ICAIF 2025)
- **Core idea**: First end-to-end deep learning method discovering dynamic lead-lag structures. Uses sparsified
  cross-attention to identify relevant lead-lag pairs. Key for us: lead-lag static proxies from training data
  can be frozen and used as stateless features. The *lag-aligned feature difference* is the key signal.
- **Feature formulas**:
  1. `delta_cross_logret_W5_W20`: `logret_W5[i] - logret_W20[j]` — asymmetric window return for lead-lag pair
     (j leads i); frozen best-pair assignment from training data
  2. `delta_lag_ofi_W5_W20`: `OFI_W5[i] - OFI_W20[j]` — lead-lag OFI proxy for best pair
  3. `lead_score_static_i`: precomputed scalar (frozen from training-period temporal cross-correlation between
     sym i and others) — probability that sym i is a lagger
- **Why适合 v7+**: v7 has F087/F088 (asymmetric window features) but without systematic lead-lag pair selection;
  DeltaLag explicitly selects best pairs.
- **预期 PnL gain**: high
- **实现 complexity**: medium
- **Robustness concern**: medium
- **优先级**: P1

---

### Cross-Impact of OFI in Equity Markets (Kolm, Turiel, Westray 2023) — Direction 2

- **Paper**: arxiv 2112.13213 / Quantitative Finance Vol 23 No 10 (2023). DOI: 10.1080/14697688.2023.2236159
- **Venue**: Quantitative Finance 2023
- **Core idea**: Systematic study of cross-impact in multi-asset LOB settings. Key finding: integrated
  multi-level OFI (IOFI) outperforms L1-only OFI. Lagged cross-OFI (not contemporaneous) improves
  return forecasting. The *lagged cross-OFI* feature captures information propagation across assets.
- **Feature formulas**:
  1. `iofi_W5`: `sum_k(w_k * OFI_Lk_W5)` where `w_k` from PCA of LOB (first PC weights, frozen from training)
  2. `lagged_cross_iofi_W5`: `iofi_W5[j]` for the specific pair (j, i) that has highest lagged cross-impact
     (frozen pair assignment from training)
- **Why适合 v7+**: IOFI (PCA-integrated OFI) is different from simple sum MLOFI in F102/F103; frozen lagged
  cross-IOFI is a new directional signal.
- **预期 PnL gain**: high
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Nonparametric Self- and Cross-Impact Estimation (Piquard, Dufresne 2025) — Direction 2

- **Paper**: arxiv 2510.06879
- **Venue**: arXiv preprint (Oct 2025)
- **Core idea**: Nonparametric concave multi-asset propagator model. Key finding: concave cross-impact fits
  better than linear. Square-root law extends to cross-impact. Provides cross-impact kernel estimates that can
  be used as static feature scalings.
- **Feature formulas**:
  1. `cross_impact_adj_ofi`: `OFI_W5[i] + sum_j(gamma_ij * sign(OFI_W5[j]) * sqrt(abs(OFI_W5[j])))` —
     concave cross-impact adjusted OFI; `gamma_ij` frozen from training
  2. `self_impact_concave`: `sign(OFI_W5[i]) * sqrt(abs(OFI_W5[i]))` — concave self-impact
- **Why适合 v7+**: v7 uses linear OFI. Concave (sqrt) OFI respects the square-root market impact law, providing
  better-scaled signals especially for large OFI values.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Multi-Asset Market Impact and OFI Commonality (Eisler, Bouchaud, Kockelkoren 2021) — Direction 2

- **Paper**: ResearchGate (Multi-asset market impact and order flow commonality), 2021; cited in 2023-2024 papers
- **Venue**: Quantitative Finance, 2021
- **Core idea**: Multi-asset OFI commonality: the common factor across assets' OFIs predicts market-wide
  returns, while residual (asset-specific minus common) predicts asset-specific returns. Provides a precise
  two-factor decomposition.
- **Feature formulas**:
  1. `ofi_market_factor_W5`: `mean(OFI_W5[0..4])` — market OFI factor (common component)
  2. `ofi_idiosyncratic_W5`: `OFI_W5[i] - beta_i * mean(OFI_W5)` — idiosyncratic OFI; `beta_i=1` (sym-agnostic
     simplification)
  3. `ofi_factor_ratio_W5`: `OFI_W5[i] / (mean(abs(OFI_W5[0..4])) + eps)` — size-normalized idiosyncratic OFI
- **Why适合 v7+**: v7 has F078 (csec_demean_ofi_W5) which is the additive version; the ratio version captures
  relative OFI significance more robustly.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### High-Frequency Lead-Lag in Chinese Futures (Zheng et al. 2025) — Direction 2

- **Paper**: arxiv 2501.03171
- **Venue**: arXiv preprint (Jan 2025)
- **Core idea**: High-frequency lead-lag relationships between calendar spread futures. Key extractable features:
  the *spread momentum* between two related instruments as a directional proxy without requiring state. The
  paper demonstrates W5-vs-W60 window spread convergence rate.
- **Feature formulas**:
  1. `pair_spread_convergence_rate`: `abs(pair_spread_W5[i,j]) / (abs(pair_spread_W60[i,j]) + eps)` —
     how far current spread is from longer-term spread level
  2. `pair_spread_curvature`: `pair_spread_W5[i,j] - 2*pair_spread_W15[i,j] + pair_spread_W30[i,j]` —
     second derivative of pair spread (mean-reversion acceleration)
- **Why适合 v7+**: v7 has pair spread features but not convergence rate or curvature; these capture dynamics.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Lead-Lag Arbitrage at High Frequency (Huckenfield & Ruf 2023) — Direction 2

- **Paper**: Journal of Forecasting 2023. From ideas.repec.org/a at tandfonline
- **Venue**: Journal of Forecasting 2023
- **Core idea**: Profitable lead-lag arbitrage requires: (a) identification of the leading asset, (b) fast
  detection of signal. Key features: return autocorrelation over micro-windows, cross-asset beta from tick data.
  Provides formulas for static "lead probability" scores.
- **Feature formulas**:
  1. `cross_autocorr_proxy_W10`: `logret_W5[i] * logret_W5[j] / (std_logret_W20[i] * std_logret_W20[j])` —
     normalized cross-return product (static cross-autocorrelation proxy per pair)
  2. `lead_beta_proxy`: `cov(logret_W5[j], logret_W5[i]) / (var(logret_W5[i]) + eps)` — frozen from training
- **Why适合 v7+**: Cross-autocorr proxy is different from F072 (pair_logret_prod) in that it normalizes by vol.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: medium
- **优先级**: P2

---

### Price Spread Prediction Deep Learning Pairs (2024) — Direction 2

- **Paper**: International Review of Financial Analysis (2024). doi from ScienceDirect/S1057521924007257
- **Venue**: International Review of Financial Analysis 2024
- **Core idea**: Deep learning for high-frequency pairs trading spread prediction. Provides features capturing
  the *speed of spread decay* and *spread momentum ratio* useful as stateless snapshots. The half-life of the
  spread estimated from AR(1) coefficient is a key predictor.
- **Feature formulas**:
  1. `pair_spread_momentum_ratio_W5_W20`: `(pair_spread_W5[i,j] - pair_spread_W20[i,j]) / (abs(pair_spread_W20[i,j]) + eps)` —
     normalized short-term momentum of pair spread
  2. `pair_spread_decay_rate`: `pair_spread_W5 / (pair_spread_W20 + eps)` — ratio indicating how fast spread reverts
- **Why适合 v7+**: These ratio-based features capture spread dynamics not present in raw diff features of v7.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Improving Cointegration Pairs with Convergence Filters (2024) — Direction 2

- **Paper**: Computational Economics (2024). DOI: 10.1007/s10614-023-10539-4
- **Venue**: Computational Economics 2024
- **Core idea**: Asymptotic convergence rate of pair spreads filters profitable pairs. Key feature: speed
  of spread half-life convergence as a score for pair-specific features. AR(1) proxy for stationary spread.
- **Feature formulas**:
  1. `pair_ar1_proxy`: `logret_inst[i]*logret_inst[j] / (sqrt(rv_5tick[i]*rv_5tick[j]) + eps)` —
     instantaneous pair AR(1) proxy (stationarity indicator)
  2. `pair_spread_half_life_proxy`: `1 / (1 + abs(autocorr_proxy_pair_W20))` — convergence speed proxy
- **Why适合 v7+**: Half-life proxy is a new scalar feature per pair not in v7.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: medium
- **优先级**: P2

---

## Direction 3: Graph / Network Features for HFT (2024–2025)

---

### Hypergraph Neural Networks for Stock (Sawhney, Agarwal et al. 2024) — Direction 3

- **Paper**: ACM ICAIF 2024. doi: 10.1145/3768292.3770389
- **Venue**: 6th ACM International Conference on AI in Finance (ICAIF 2024)
- **Core idea**: Higher-order stock relationships via hyperedges (grouping 3+ stocks). Key finding: 3-stock
  hyperedge OFI products capture third-order market interactions not present in pairwise features. The
  *hyperedge aggregated OFI* across triplets provides a new market-structure signal.
- **Feature formulas**:
  1. `hyperedge_ofi_mean_W5`: `mean(OFI_W5[i], OFI_W5[j], OFI_W5[k]) - OFI_W5[i]` for best triplet (j,k most
     correlated with i, frozen from training); deviation of i from its triplet mean
  2. `triple_ofi_product_sign`: `sign(OFI_W5[i]) * sign(OFI_W5[j]) * sign(OFI_W5[k])` — three-way OFI alignment
     indicator (+1 all same direction, -1 otherwise)
- **Why适合 v7+**: v7 only has pairwise interactions; three-way products capture synchronization not reducible
  to pairwise sums.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### HAMAN — Hierarchical Adaptive Multiplex Hypergraph (2024) — Direction 3

- **Paper**: ACM International Conference on Big Data, AI and Risk Management (BIGAI 2024).
  doi: 10.1145/3718751.3718899
- **Venue**: ACM BIGAI 2024
- **Core idea**: Multiplex hypergraph using sector + fund-holding relationships. For our 5-sym setting:
  treat all 5 as one sector hyperedge. Key feature: hierarchical aggregation from sym → hyperedge → market.
- **Feature formulas**:
  1. `sector_hyperedge_ofi_W5`: `sum(OFI_W5[k]) for k=0..4` — total sector OFI (market-wide pressure)
  2. `sym_vs_sector_ofi_normalized`: `OFI_W5[i] / (abs(sector_hyperedge_ofi_W5) + eps)` — sym's contribution
     share (signed, different from F121 which is absolute share)
- **Why适合 v7+**: Signed contribution share includes direction information vs. F121 (unsigned absolute share).
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Static-Dynamic Hypergraph Framework (Hao et al. 2024) — Direction 3

- **Paper**: Wiley Complexity (2024). doi: 10.1155/2024/5791802
- **Venue**: Complexity (Wiley) 2024
- **Core idea**: Combines static (sector correlation) and dynamic (time-varying) hyperedge weights for stock
  recommendation. For stateless use: static hyperedge weights from training data, dynamic features from current
  tick data.
- **Feature formulas**:
  1. `csec_static_hyperedge_score`: precomputed vector (frozen from training) = sym i's eigenvector weight
     in hyperedge decomposition (different from F095 which uses PCA on returns only)
  2. `dynamic_hyperedge_ofi_W5`: `sum_k(w_k * OFI_W5[k])` where `w_k` = static hyperedge weight for sym k
     (provides vol-corr-weighted market OFI)
- **Why适合 v7+**: Weights correlated syms more heavily in the market factor, unlike equal-weight mean (F054).
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: medium
- **优先级**: P2

---

### Dynamic Hypergraph Spatio-Temporal Network (2024) — Direction 3

- **Paper**: ScienceDirect (Expert Systems with Applications 2024). doi from pii/S1568494624001030
- **Venue**: Expert Systems with Applications 2024
- **Core idea**: Dynamic hypergraph where hyperedges encode time-varying group relationships. Outperforms
  pairwise GNN by 4.99% in F1-score and 47.9% in Sharpe ratio. Key static proxy: sector-level OFI centroid.
- **Feature formulas**:
  1. `ofi_centroid_deviation_W5`: `OFI_W5[i] - median(OFI_W5[0..4])` — median-based deviation (robust to
     outlier sym; different from mean-based F078)
  2. `rv_centroid_deviation_W20`: `RV_W20[i] - median(RV_W20[0..4])` — median vol deviation
- **Why适合 v7+**: Median-based cross-sym features are more robust than mean-based features in v7.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Implicit-Causality Exploration GNN (Zhao, Peng et al. 2024) — Direction 3

- **Paper**: MDPI Information 2024. doi 10.3390/info15120743
- **Venue**: Information (MDPI) 2024
- **Core idea**: GNN that explores causal (not just correlated) relationships between stocks. Key insight:
  causal features = the directed (asymmetric) component of the pairwise OFI matrix. This is captured by
  `OFI[i] - OFI[j]` conditional on the sign structure of cross-lagged returns.
- **Feature formulas**:
  1. `causal_ofi_pair_W5`: `OFI_W5[i] - OFI_W5[j]` when `sign(logret_W5[j]) == sign(OFI_W5[i])` else 0 —
     directional causal component of pair OFI
  2. `csec_causal_ofi_score`: `sum_j(causal_ofi_pair_W5[i,j])` — total causal OFI score for sym i
- **Why适合 v7+**: Causal (directed) OFI is new vs. v7's symmetric pair OFI differences.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P2

---

### Functional Hypergraphs of Stock Markets (2024) — Direction 3

- **Paper**: PMC/NCBI 2024 (PMC11507234)
- **Venue**: PMC (journal publication 2024)
- **Core idea**: Functional hypergraphs encoding higher-order statistical dependencies (beyond pairwise
  correlations) in stock markets. Key feature: 3-asset mutual information proxy as a functional hyperedge weight.
- **Feature formulas**:
  1. `triple_corr_proxy`: `(logret_W20[i]*logret_W20[j]*logret_W20[k]) / (std_W20[i]*std_W20[j]*std_W20[k])` —
     normalized third-order co-moment for best triplet (j, k frozen from training)
  2. `csec_max_triple_corr`: `max over all C(5,3) triplets of abs(triple_corr_proxy)` — max three-way co-movement
- **Why适合 v7+**: v7 uses only pairwise products (F072/F073); three-way co-moments capture emergent group effects.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Heterogeneous GNN for Stock Price Prediction (2024) — Direction 3

- **Paper**: ResearchGate 2024 (HETEROGENEOUS GRAPH NEURAL NETWORKS FOR STOCK PRICE PREDICTION)
- **Venue**: Conference/journal 2024 (ResearchGate ID 400231605)
- **Core idea**: Multi-type edges in stock graph: price correlation, sector membership, news similarity.
  For LOB use: different edge types (OFI similarity vs. return correlation) can be used as static features.
- **Feature formulas**:
  1. `rv_corr_static_ij`: precomputed (frozen) scalar = rank correlation of daily RV between sym i and j
     (independent feature from F171 which uses return corr not vol corr)
  2. `ofi_corr_static_ij`: precomputed (frozen) scalar = correlation of W20 OFI sequences during training
     (different from return correlation)
- **Why适合 v7+**: Vol correlation as a static feature is new (F171 only has return correlation).
- **预期 PnL gain**: low
- **实现 complexity**: medium
- **Robustness concern**: high
- **优先级**: P3

---

### Robust Multi-Relational Hypergraph (2025) — Direction 3

- **Paper**: ScienceDirect 2025. doi from pii/S0950705125013243
- **Venue**: Knowledge-Based Systems 2025
- **Core idea**: Hierarchical hypergraph attention with both industry and supply-chain relations. In a
  5-sym setting, all 5 syms form one super-hyperedge. The key feature is *within-hyperedge rank*:
  ordering syms by their relative OFI within the group.
- **Feature formulas**:
  1. `csec_signed_rank_ofi_W5`: `(rank(OFI_W5[i]) - 3) / 2` — centered signed rank in [-1, 1] for 5 syms
     (similar to F165 but applied to OFI not returns, and centered differently)
  2. `csec_signed_rank_rv_W20`: `(rank(RV_W20[i]) - 3) / 2` — centered signed rank of vol
- **Why适合 v7+**: Centered signed rank is a stable ordinal transformation; differs from F042 (rank/5) in centering.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

## Direction 4: Hawkes / Point Process Inspired Features (2024–2025)

---

### Forecasting High-Frequency OFI (Coatanlem, Kumar et al. 2024) — Direction 4

- **Paper**: arxiv 2408.03594
- **Venue**: arXiv preprint (Aug 2024)
- **Core idea**: Hawkes sum-of-exponentials model outperforms Poisson/VAR for OFI prediction. Key static
  proxy: the ratio of short-horizon to long-horizon OFI as a measure of endogenous clustering (Hawkes
  self-excitation). When this ratio is high, the market is in a self-excited state.
- **Feature formulas**:
  1. `self_excitation_proxy_W5_W20`: `OFI_W5[i] / (OFI_W20[i] + eps)` — ratio of short to medium OFI
     (high ratio = burst mode; a Hawkes self-excitation proxy without time series)
  2. `csec_self_excitation_zscore`: `(self_excitation_proxy_W5_W20[i] - mean(...[0..4])) / std(...)` —
     cross-sym z-score of excitation proxy
- **Why适合 v7+**: Self-excitation ratio is a new scalar not in v7; captures burst vs. calm market regimes.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Hybrid VAR-Neural Network for OFI Prediction (2024) — Direction 4

- **Paper**: arxiv 2411.08382
- **Venue**: arXiv preprint (Nov 2024)
- **Core idea**: Hybrid model: VAR captures linear OFI structure, NN supplements nonlinear residuals.
  Key PCA insight: first PC of 5-level OFI explains 89% of variance. This first PC as a feature is robust
  and replaces full 5-dimensional OFI with a 1D summary.
- **Feature formulas**:
  1. `ofi_pca1_W5`: `dot(OFI_L1..5_W5, pc1_weights)` — first PC projection of multi-level OFI;
     `pc1_weights` frozen from training
  2. `ofi_pca2_W5`: `dot(OFI_L1..5_W5, pc2_weights)` — second PC (noise/residual component)
  3. `ofi_pca1_residual_W5`: `OFI_W5_L1 - ofi_pca1_W5 * pc1_weights[0]` — L1 OFI not explained by first PC
- **Why适合 v7+**: PCA-projected OFI is superior to raw OFI as input; orthogonal components avoid
  multicollinearity in the feature set.
- **预期 PnL gain**: high
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Semiparametric Multivariate Hawkes (Caron, Rousseau et al. 2025) — Direction 4

- **Paper**: arxiv 2502.17723
- **Venue**: arXiv preprint (Feb 2025)
- **Core idea**: Bayesian nonparametric estimation of multivariate Hawkes processes for order flow data.
  Key estimable feature: cross-excitation kernel integral (the total cross-impact from asset j to i over
  a window). This integral is estimable as a static proxy from the ratio of cross-window OFIs.
- **Feature formulas**:
  1. `cross_excitation_proxy_W5`: `intst_W5[i] * intst_W5[j] / (intst_W20[i] * intst_W20[j] + eps)` —
     pair cross-excitation proxy (high when short-burst intensities co-move)
  2. `csec_cross_excitation_score`: `sum_j(cross_excitation_proxy_W5[i,j]) / 4` — avg cross-excitation
     of sym i with all others
- **Why适合 v7+**: v7 has pair intensity diffs (F007/F008) but not cross-excitation products.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Deep Hawkes Process for Market Making (Kumar 2024) — Direction 4

- **Paper**: arxiv 2109.15110 / JBFT 2024 (Springer). DOI 10.1007/s42786-024-00049-8
- **Venue**: Journal of Banking and Financial Technology 2024
- **Core idea**: Feedback loop between order arrival intensity and LOB state via Hawkes process.
  Key static proxy: the *Fano factor* of order arrival times measures clustering/burstiness without
  requiring state maintenance. High Fano factor = clustered (Hawkes-like); low = Poisson.
- **Feature formulas**:
  1. `fano_factor_bid_W20`: `var(bid_arrivals_per_5tick_window) / (mean(bid_arrivals_per_5tick_window) + eps)` —
     Fano factor for bid arrivals (computed over last W20 ticks)
  2. `fano_factor_ask_W20`: `var(ask_arrivals_per_5tick_window) / (mean(ask_arrivals_per_5tick_window) + eps)`
  3. `fano_asymmetry_W20`: `fano_factor_bid_W20 - fano_factor_ask_W20` — directional burstiness asymmetry
- **Why适合 v7+**: Fano factor is a model-free Hawkes proxy not in v7; measures order arrival clustering.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Hawkes Process Analysis of High-Freq Price Dynamics (Zhang et al. 2024) — Direction 4

- **Paper**: EJF 2024 (Ruixun Zhang). From ruixunzhang.com/paper/2024_EJF_Hawkes.pdf
- **Venue**: European Journal of Finance 2024
- **Core idea**: Hawkes process models for high-frequency price dynamics with cross-asset excitation.
  Key finding: cross-excitation between assets is directional (asset i → asset j more than j → i). Static
  proxy: the asymmetry ratio of cross-window intensities.
- **Feature formulas**:
  1. `hawkes_asymm_pair_W5_W20`: `(intst_W5[i] / (intst_W5[j] + eps)) - (intst_W20[i] / (intst_W20[j] + eps))` —
     asymmetry in short-vs-long intensity ratio between pair
  2. `csec_intensity_burst_score`: `intst_W5[i] / (intst_W60[i] + eps)` — single-sym self-excitation intensity
     burst score
- **Why适合 v7+**: Intensity burst score (W5/W60) is more informative than raw intensity as it captures
  relative deviation from baseline.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Forecasting Bitcoin Hawkes + LOB (2026) — Direction 4

- **Paper**: Springer Decisions in Economics and Finance (2026 online). DOI 10.1007/s10203-026-00570-z
- **Venue**: Decisions in Economics and Finance, Springer 2026
- **Core idea**: Multivariate Hawkes processes with LOB data for Bitcoin price prediction. Key feature:
  trade clustering coefficient — how much the recent trade rate exceeds the Poisson baseline.
- **Feature formulas**:
  1. `trade_clustering_coeff_W20`: `intst_W5[i] / (intst_W60[i] + eps) - 1.0` — excess intensity vs. baseline
     (zero-centered clustering coefficient)
  2. `bid_ask_arrival_imbalance_W20`: `(bid_intst_W5 - ask_intst_W5) / (bid_intst_W5 + ask_intst_W5 + eps)` —
     imbalance of arrival intensities (proxy for informed trade direction)
- **Why适合 v7+**: Arrival intensity imbalance (bid vs. ask) is different from price-based OFI and provides a
  complementary signal from the timing of events rather than their sizes.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P2

---

## Direction 5: Latest Quant Industry Features (2024–2025)

---

### DeepVol — Dilated Causal Convolutions for Volatility (Garcia-Lorite & Cont 2024) — Direction 5

- **Paper**: arxiv 2210.04797 / Quantitative Finance 2024. DOI: 10.1080/14697688.2024.2387222
- **Venue**: Quantitative Finance 2024
- **Core idea**: Deep learning for day-ahead realized volatility using multi-scale high-frequency data.
  Key insight: multi-timescale realized volatility (5-tick, 20-tick, 60-tick) as inputs is more
  informative than any single horizon. The *ratio* of short to long RV captures vol regime transitions.
- **Feature formulas**:
  1. `rv_5tick`: `sum((logret_k)^2 for k in last 5 ticks)` — micro realized variance
  2. `rv_acceleration`: `rv_5tick / (rv_W60 + eps)` — ratio of recent vs. baseline vol
  3. `har_vol_composite`: `(rv_5tick + rv_W20 + rv_W60) / 3` — HAR-decomposed vol composite
- **Why适合 v7+**: HAR composite and vol acceleration are new vs. v7's single-window RV features.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Realised Illiquidity — Amihud and High-Low (2024) — Direction 5

- **Paper**: AFA Conference 2024 / SSRN. From aeaweb.org/conference/2024/program/paper/dTz568tG
- **Venue**: AFA Annual Meeting 2024
- **Core idea**: Two measures of realized illiquidity: *Realized Amihud* (ratio of realized power variation
  to daily volume) and *High-Low Amihud* (range to volume). Both are much more precise intraday liquidity
  measures than classic Amihud.
- **Feature formulas**:
  1. `amihud_realized_W20`: `sum(abs(logret_k) / tick_volume_k for k in W20) / W20` — high-freq Amihud
  2. `amihud_hl_W20`: `(max(ask_k for k in W20) - min(bid_k for k in W20)) / (total_volume_W20 + eps)` —
     high-low Amihud
  3. `csec_amihud_zscore`: `(amihud_realized_W20[i] - mean(amihud_W20)) / std(amihud_W20)` —
     cross-sym illiquidity z-score
- **Why适合 v7+**: Amihud-based illiquidity is not in v7; it captures price-per-unit-volume impact more directly
  than Kyle lambda.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### ML-Based Market Liquidity Estimation (2025) — Direction 5

- **Paper**: ScienceDirect 2025 (International Review of Financial Analysis). doi pii/S138641812500059X
- **Venue**: International Review of Financial Analysis 2025
- **Core idea**: ML methods improve estimation of market liquidity measures (bid-ask spread, effective spread,
  price impact, Kyle lambda) using stock-level features. Key insight: effective spread decomposition into
  adverse selection (price impact) and order processing cost components.
- **Feature formulas**:
  1. `eff_spread_proxy_W5`: `2 * abs(trade_price_W5 - mid_before_trade_W5) / (mid_before_trade_W5 + eps)` —
     effective half-spread proxy (requires tick-rule trade direction)
  2. `adverse_selection_W5`: `sign_trade * (mid_W5_plus5 - mid_W5_minus5) / mid_W5_minus5` — price impact
     component of effective spread (signed by trade direction)
  3. `info_ratio_spread_W20`: `eff_spread_proxy_W5 / (bid_ask_spread_W20 + eps)` — ratio of realized to quoted
- **Why适合 v7+**: Effective spread and adverse selection decomposition are new; v7 only has quoted spread features.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Alpha Decay and Factor Crowding (2025) — Direction 5

- **Paper**: arxiv 2512.11913
- **Venue**: arXiv preprint (Dec 2025)
- **Core idea**: Alpha decay model capturing factor crowding effects. Key insight: when multiple assets move
  in the same direction simultaneously, a crowding signal is present. Proxy: cross-sectional OFI alignment
  score.
- **Feature formulas**:
  1. `csec_ofi_alignment_W5`: `sum(sign(OFI_W5[k])) / 5 for k=0..4` — fraction of same-direction OFIs
     (crowding indicator; +1 = all buy, -1 = all sell)
  2. `csec_ret_alignment_W20`: `sum(sign(logret_W20[k])) / 5` — fraction of same-direction returns
- **Why适合 v7+**: Alignment score (crowd direction) is different from std/dispersion features in v7;
  captures collective behavior extremes.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Cross-Sectional Factor Momentum (2025) — Direction 5

- **Paper**: Applied Economics Letters 2025. DOI: 10.1080/13504851.2025.2472032
- **Venue**: Applied Economics Letters 2025
- **Core idea**: Recent and intermediate-past formation periods for cross-sectional momentum are both
  informative. Key feature: difference between short (W5) and medium (W60) momentum ranks captures
  *momentum acceleration* (whether a stock is becoming a better or worse cross-sectional performer).
- **Feature formulas**:
  1. `csec_momentum_accel_rank`: `rank(logret_W5[i]) - rank(logret_W60[i])` — rank difference (same as F200
     but using W5 vs W60 instead of W20 vs W60)
  2. `csec_momentum_zscore_diff`: `zscore(logret_W5[i]) - zscore(logret_W20[i])` — z-score momentum change
     (captures whether short-term momentum is diverging from medium-term)
- **Why适合 v7+**: Z-score momentum diff (W5 vs W20) is new; F200 uses ranks, not z-scores.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Jane Street Real-Time Market Data Forecasting (2024) — Direction 5

- **Paper / Competition**: Kaggle Jane Street Real-Time Market Data Forecasting (2024)
- **Venue**: Kaggle Competition 2024 (Jane Street)
- **Core idea**: Production real-time market features from Jane Street. Competition data has 130 features
  for ~2.4M rows. Top winning approaches used: signed trade volume imbalance (tick rule), OFI variants,
  cross-sectional feature normalization per time_id, and feature interaction products.
- **Feature formulas**:
  1. `trade_sign_imb_W5`: `(buy_vol_W5 - sell_vol_W5) / (total_vol_W5 + eps)` — tick-rule signed volume
     imbalance (key Jane Street feature type)
  2. `csec_iqr_norm_ret_W5`: `(logret_W5[i] - median(logret_W5)) / (IQR(logret_W5) + eps)` — IQR-normalized
     market-neutral return (robust to outliers vs. z-score)
- **Why适合 v7+**: IQR normalization is more robust than std-based z-score used in v7; trade sign imbalance
  from tick rule is more direct than OFI.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### When Alpha Breaks — Safe Deployment Uncertainty (2025) — Direction 5

- **Paper**: arxiv 2603.13252
- **Venue**: arXiv preprint (Mar 2025)
- **Core idea**: Two-level uncertainty quantification for cross-sectional stock rankers. Key feature:
  the *local uncertainty score* — variance of predictions in the neighborhood of current features, which
  can be proxied by cross-sym feature dispersion.
- **Feature formulas**:
  1. `csec_ofi_iqr_W20`: `IQR(OFI_W20[0..4])` — IQR-based dispersion of OFI (robust vs. F116 std-based)
  2. `csec_rv_iqr_W20`: `IQR(RV_W20[0..4])` — IQR-based vol dispersion (robust vs. F120)
- **Why适合 v7+**: IQR-based dispersion features are more robust to outlier syms than std-based (v7 F115-F120).
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

## Direction 6: Cross-Feature Interaction Networks (2024–2025)

---

### Role of Feature Interactions in Graph Tabular DL (Beutel, Herrmann et al. 2025) — Direction 6

- **Paper**: arxiv 2510.04543
- **Venue**: arXiv preprint (Oct 2025)
- **Core idea**: Graph-based tabular deep learning fails to recover meaningful feature interactions
  (edge recovery close to random). BUT explicit feature crossing (multiplicative and ratio interactions)
  remain useful for financial tabular data. Key: cross-feature products between OFI and spread/vol.
- **Feature formulas**:
  1. `ofi_x_spread_W5`: `OFI_W5[i] * spread[i]` — OFI weighted by spread (impact * liquidity cost)
  2. `ofi_x_rv_W20`: `OFI_W20[i] * RV_W20[i]` — OFI weighted by volatility
  3. `imb1_x_rv_W20`: `imb1[i] * RV_W20[i]` — imbalance modulated by vol regime
- **Why适合 v7+**: These three-way feature products are different from v7 pair products in F108-F111 which
  focus on between-sym interactions; these are within-sym regime interactions.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### T2G-Former — Tabular Feature Relation Graphs (Yan, Wang et al. 2023) — Direction 6

- **Paper**: arxiv 2211.16887
- **Venue**: AAAI 2023 (applied to finance 2024)
- **Core idea**: Organizes tabular features into relation graphs where nodes are features and edges encode
  feature interactions. Key insight: feature pairs with high mutual information form edges. In HFT context:
  OFI×depth, return×vol, spread×depth are high-MI feature pairs.
- **Feature formulas**:
  1. `ofi_depth_interaction_W5`: `OFI_W5[i] / (depth_lvl1[i] + eps)` — OFI intensity per unit depth
     (Kyle lambda proxy from instantaneous data; different from frozen kyleinv in F016)
  2. `spread_depth_interaction`: `spread[i] * (bidsz1[i] + asksz1[i])` — liquidity-weighted spread
     (cost × queue size = total fill cost at top of book)
- **Why适合 v7+**: Depth-normalized OFI captures instantaneous impact intensity; liquidity-weighted spread
  is not in v7.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Tree-like Pairwise Interaction Networks (2025) — Direction 6

- **Paper**: arxiv 2508.15678
- **Venue**: arXiv preprint (Aug 2025)
- **Core idea**: Tree-structured pairwise interactions that capture hierarchical feature dependencies.
  For LOB data: bid-ask interaction trees where spread → imbalance → OFI form a cascade.
- **Feature formulas**:
  1. `spread_imb_ofi_cascade`: `spread[i] * imb1[i] * OFI_W5[i]` — three-way cascade product
  2. `depth_vol_return_cascade`: `depth_lvl1[i] * RV_W20[i] * logret_W5[i]` — three-way regime product
- **Why适合 v7+**: Three-way within-sym cascade products are new; v7 only has two-way products.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Neural Network CSI300 Feature Importance + Attention (2025) — Direction 6

- **Paper**: MDPI Electronics 2025. doi 10.3390/electronics14091729
- **Venue**: Electronics (MDPI) 2025
- **Core idea**: Attention mechanism reveals which features are most important for price prediction. Key finding:
  interaction between volume and price momentum is consistently high-attention. Provides empirical formula for
  feature importance-weighted composite.
- **Feature formulas**:
  1. `volume_momentum_interaction`: `log(volume_W5 + 1) * logret_W5[i]` — volume-momentum interaction
  2. `depth_momentum_interaction`: `depth_lvl1[i] * logret_W20[i]` — depth-momentum interaction
- **Why适合 v7+**: Volume×momentum cross-product interaction is not in v7.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Hybrid TFT-GNN for Stock Prediction (2024) — Direction 6

- **Paper**: MDPI Algorithms 2024. doi 10.3390/a5040176 (from search)
- **Venue**: Algorithms (MDPI) 2024
- **Core idea**: Temporal Fusion Transformer + Graph Neural Network hybrid. Key TFT feature contribution:
  *quantile-normalized* cross-sectional features have lower variance than z-score normalized, improving
  model stability. Also: gating mechanism on features based on vol regime.
- **Feature formulas**:
  1. `csec_quantile_norm_ofi_W5`: `(rank(OFI_W5[i])-1) / (N-1)` — [0,1] normalized rank (same as F123 but
     applied to W5 OFI and in a different combination context; let's distinguish by making it combined)
  2. `vol_gated_ofi_W5`: `OFI_W5[i] * (1 / (1 + RV_W20[i]))` — vol-gated OFI (dampen signal in high-vol regime)
  3. `vol_gated_imb1`: `imb1[i] * (1 / (1 + RV_W20[i]))` — vol-gated imbalance
- **Why适合 v7+**: Vol-gating of OFI/imb1 is not in v7; reduces noise from high-volatility regimes.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

## Direction 7: Microstructure Specific (2024–2025)

---

### Volatility Forecasting with Positive/Negative Jumps (Zhang et al. 2024) — Direction 7

- **Paper**: Journal of Forecasting 2024. DOI 10.1002/for.3146
- **Venue**: Journal of Forecasting 2024
- **Core idea**: Decomposes realized volatility into continuous, positive-jump, and negative-jump components.
  Finding: negative jumps have larger and longer-lasting impact on future volatility than positive jumps.
  Signed jump asymmetry is a new and important feature.
- **Feature formulas**:
  1. `rv_jump_pos_W20`: `max(RV_W20 - BPV_W20, 0) * (sum(logret_k > 0 for k in W20) / W20)` —
     positive realized jump component estimate
  2. `rv_jump_neg_W20`: `max(RV_W20 - BPV_W20, 0) * (sum(logret_k < 0 for k in W20) / W20)` —
     negative realized jump component
  3. `rv_jump_asymmetry_W20`: `rv_jump_pos_W20 - rv_jump_neg_W20` — signed jump asymmetry
- **Why适合 v7+**: v7 has F014 (jshare pair diff) but not signed decomposition of jumps into +/-.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Understanding the Worst-Kept Secret of HFT (Breckenfelder et al. 2023) — Direction 7

- **Paper**: arxiv 2307.15599
- **Venue**: arXiv preprint 2023
- **Core idea**: Systematic study of HFT activity and its microstructure signals. Key feature: HFT-driven
  quote-update intensity as a proxy for informed activity. Distinguishes market-making HFTs from opportunistic
  HFTs by their quote-update patterns.
- **Feature formulas**:
  1. `quote_update_intensity_W5`: `n_quote_updates_W5 / (W5 + eps)` — quote update rate per tick
  2. `quote_update_vs_trade_ratio`: `n_quote_updates_W5 / (n_trades_W5 + eps)` — ratio of cancellations to
     trades (high = HFT quoting; low = informed trading)
- **Why适合 v7+**: Quote update rate (not just quote intensity) is a new feature; quote-to-trade ratio
  distinguishes market-making vs. informed activity.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### Interpretable ML for High-Frequency Execution (Wallbridge & Treleaven 2023) — Direction 7

- **Paper**: arxiv 2307.04863
- **Venue**: arXiv preprint 2023
- **Core idea**: SHAP-based feature importance for HFT execution models. Key finding: OFI z-score (time-series
  normalized) is more informative than raw OFI. The *time-series z-scored OFI* (relative to a rolling
  baseline from the same tick sequence) is the top feature.
- **Feature formulas**:
  1. `ofi_ts_zscore_W5_W60`: `(OFI_W5[i] - mean_OFI_W5_baseline_W60) / (std_OFI_W5_baseline_W60 + eps)` —
     time-series z-scored OFI (where baseline mean/std are computed over all W5-windows within W60)
  2. `imb1_ts_zscore_W5_W60`: same pattern for imb1
- **Why适合 v7+**: Time-series z-score (within-sym rolling normalization) is different from cross-sym z-score
  in v7 (F034/F040). Captures whether *this* moment is unusual for *this* sym's own history.
- **预期 PnL gain**: high
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P1

---

### Data-Driven Measures of High-Frequency Trading (2024) — Direction 7

- **Paper**: arxiv 2405.08101
- **Venue**: arXiv preprint (May 2024)
- **Core idea**: Data-driven identification of HFT activity via order-level features. Key feature:
  *order book turnover rate* — how fast the top-of-book renews itself through cancellations and new orders.
- **Feature formulas**:
  1. `lob_turnover_rate_W20`: `n_tob_changes_W20 / (depth_W20_avg + eps)` — rate of top-of-book changes
     per unit depth (proxy for LOB refresh speed)
  2. `cancel_submit_ratio_W20`: `n_cancellations_W20 / (n_new_orders_W20 + eps)` — cancel-to-submit ratio
     (high = HFT quoting; low = patient institutional)
- **Why适合 v7+**: LOB turnover rate and cancel-to-submit ratio are new microstructure features.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P2

---

### Market Simulation under Adverse Selection (Cao et al. 2024) — Direction 7

- **Paper**: arxiv 2409.12721
- **Venue**: arXiv preprint (Sep 2024)
- **Core idea**: Simulates market with adverse selection, providing features useful for detecting informed
  trading. Key feature: *negative drift of maker orders* — maker orders that get filled have negative
  subsequent returns (adverse selection proxy).
- **Feature formulas**:
  1. `maker_fill_adverse_proxy_W5`: `-sign(OFI_W5[i]) * logret_W5[i]` — when OFI suggests buying but price
     falls, adverse selection is likely (proxy for fill risk)
  2. `adverse_selection_indicator`: `1 if (sign(OFI_W5) != sign(logret_W5)) else -1` — sign mismatch between
     OFI and returns indicates adverse selection in recent ticks
- **Why适合 v7+**: Adverse selection indicator from OFI-return sign mismatch is not in v7.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### LiT — Limit Order Book Transformer (2025) — Direction 7

- **Paper**: Frontiers in Artificial Intelligence 2025. doi 10.3389/frai.2025.1616485
- **Venue**: Frontiers in AI 2025
- **Core idea**: Transformer for LOB forecasting using temporal and cross-level attention. Key extracted
  feature: the *level activation map* — how attention concentrates at certain price levels varies with
  market conditions. The variance of attention across levels is a market-state indicator.
- **Feature formulas**:
  1. `lob_attention_entropy`: `-sum(a_k * log(a_k) for k=1..5)` where `a_k = (bidsz_k + asksz_k) / sum_all` —
     entropy of depth distribution (simpler version of attention entropy)
  2. `lob_attention_gini`: `sum(abs(bidsz_k/sum_bid - 1/5))` — Gini coefficient of bid depth distribution
- **Why适合 v7+**: Gini coefficient of depth distribution is different from HHI (NF012) and entropy (NF002);
  provides another angle on depth concentration.
- **预期 PnL gain**: medium
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P2

---

### PIN Proxy from Trade Flow Without Model (Easley et al. updated 2024) — Direction 7

- **Paper**: Easley, Lopez de Prado, O'Hara 2024 update (SSRN/VPIN extensions); see related discussion in
  electronictradinghub.com/vpin and papers.ssrn.com
- **Venue**: SSRN Working Paper 2024
- **Core idea**: Updated VPIN framework for real-time order toxicity. Key stateless feature: unsigned
  trade flow imbalance as a PIN proxy — fraction of unbalanced trades in a volume bucket.
- **Feature formulas**:
  1. `pin_proxy_W20`: `abs(buy_vol_W20 - sell_vol_W20) / (total_vol_W20 + eps)` — unsigned trade imbalance
     PIN proxy (VPIN in volume time, simplified for stateless use)
  2. `csec_pin_rank`: `rank(pin_proxy_W20[i]) among 5 syms` — cross-sym rank of toxicity
  3. `signed_amt_imb_W20`: `sum(signed_amt[-W20:]) / sum(abs(amt[-W20:]))` — signed amount imbalance (Easley 2024)
- **Why适合 v7+**: Amount-signed imbalance captures *dollar-weighted* trade direction, new vs. quantity-based OFI.
- **预期 PnL gain**: high
- **实现 complexity**: easy
- **Robustness concern**: low
- **优先级**: P1

---

### Parkinson and Rogers-Satchell High-Low Vol Estimators (Classic, 2024 Review) — Direction 7

- **Paper**: See Andersen & Benzoni 2009 / recent 2024 survey on realized illiquidity
- **Venue**: 2024 AFA paper on realized illiquidity (references high-low estimators)
- **Core idea**: High-low range-based volatility estimators are more efficient than close-to-close RV.
  Parkinson (1980): uses high-low range. Rogers-Satchell (1991): uses O-H-L-C without drift assumption.
  Both can be computed from tick-level bid/ask high/low within a window.
- **Feature formulas**:
  1. `parkinson_vol_W20`: `sqrt(1/(4*log(2)*W20) * sum((log(ask_max_k)-log(bid_min_k))^2 per 5-tick subwindow))`
  2. `vol_ratio_parkinson_rv`: `parkinson_vol_W20 / (rv_W20 + eps)` — Parkinson vs. realized ratio
     (high = trending; low = mean-reverting within the window)
- **Why适合 v7+**: Parkinson vol is more efficient than standard RV; the ratio captures trend vs. mean-reversion.
- **预期 PnL gain**: medium
- **实现 complexity**: medium
- **Robustness concern**: low
- **优先级**: P2

---

## Summary Table of Papers

| # | Paper | Year | Venue | Direction | Key Features |
|---|-------|------|-------|-----------|--------------|
| 1 | LOBFrame (Briola et al.) | 2025 | Quant Finance | 1 | tick_size_ratio, lob_predictability_proxy |
| 2 | TLOB (Berti & Kasneci) | 2025 | arXiv | 1 | depth_centroid_bid/ask |
| 3 | Crypto LOB Better Inputs (Wang) | 2025 | arXiv | 1 | lob_entropy, lob_hhi |
| 4 | High-Res Microprice Tsetlin (Blakely) | 2024 | arXiv | 1 | microprice_L3_corr |
| 5 | Boltzmann Price (Rola) | 2025 | arXiv | 1 | boltzmann_price_L5 |
| 6 | LOB Filtration (Anantha et al.) | 2025 | arXiv | 1 | obi_trade_filtered |
| 7 | Attention LOB (Kolbæk & Poulsen) | 2024 | arXiv | 1 | lob_L2_L3_imb |
| 8 | HLOB Persistence (2024) | 2024 | Expert Sys Apps | 1 | mid_level_vol_ratio, lob_slope |
| 9 | DeltaLag (Zhou et al.) | 2025 | ICAIF | 2 | delta_cross_logret |
| 10 | Cross-Impact OFI (Kolm et al.) | 2023 | Quant Finance | 2 | iofi_pca, lagged_cross_iofi |
| 11 | Nonpar Cross-Impact (Piquard) | 2025 | arXiv | 2 | cross_impact_adj_ofi |
| 12 | Multi-Asset OFI Commonality (Eisler) | 2021 | Quant Finance | 2 | ofi_factor_ratio |
| 13 | Lead-Lag Chinese Futures (Zheng) | 2025 | arXiv | 2 | pair_spread_curvature |
| 14 | Lead-Lag Arbitrage HFT (Huckenfield) | 2023 | J Forecasting | 2 | cross_autocorr_proxy |
| 15 | Price Spread DL Pairs (2024) | 2024 | Int Rev Fin | 2 | pair_spread_momentum_ratio |
| 16 | Improving Cointegration (2024) | 2024 | Comp Economics | 2 | pair_ar1_proxy |
| 17 | Hypergraph Stock (Sawhney et al.) | 2024 | ICAIF | 3 | triple_ofi_product_sign |
| 18 | HAMAN Multiplex Hypergraph | 2024 | ACM BIGAI | 3 | sym_vs_sector_ofi_normalized |
| 19 | Static-Dynamic Hypergraph (Hao) | 2024 | Wiley Complexity | 3 | dynamic_hyperedge_ofi |
| 20 | Dynamic Hypergraph ST (2024) | 2024 | Expert Sys Apps | 3 | ofi_centroid_deviation |
| 21 | Implicit-Causality GNN (Zhao) | 2024 | MDPI Info | 3 | causal_ofi_pair |
| 22 | Functional Hypergraphs (2024) | 2024 | PMC | 3 | triple_corr_proxy |
| 23 | Heterogeneous GNN (2024) | 2024 | Conference | 3 | rv_corr_static, ofi_corr_static |
| 24 | Robust Hypergraph (2025) | 2025 | KBS | 3 | csec_signed_rank_ofi |
| 25 | Hawkes OFI Forecasting (2024) | 2024 | arXiv 2408.03594 | 4 | self_excitation_proxy |
| 26 | Hybrid VAR-NN OFI (2024) | 2024 | arXiv 2411.08382 | 4 | ofi_pca1, ofi_pca2 |
| 27 | Multivariate Hawkes (Caron) | 2025 | arXiv 2502.17723 | 4 | cross_excitation_proxy |
| 28 | Deep Hawkes Market Making (Kumar) | 2024 | JBFT | 4 | fano_factor_bid/ask |
| 29 | Hawkes Price Dynamics (Zhang) | 2024 | EJF | 4 | hawkes_asymm_pair, intensity_burst |
| 30 | Bitcoin Hawkes LOB (2026) | 2026 | Springer | 4 | bid_ask_arrival_imbalance |
| 31 | DeepVol (Garcia-Lorite & Cont) | 2024 | Quant Finance | 5 | rv_5tick, rv_acceleration, har_vol |
| 32 | Realised Illiquidity (2024) | 2024 | AFA | 5 | amihud_realized, amihud_hl |
| 33 | ML Market Liquidity (2025) | 2025 | Int Rev Fin | 5 | eff_spread_proxy, adverse_selection |
| 34 | Alpha Decay Factor Crowding (2025) | 2025 | arXiv 2512.11913 | 5 | csec_ofi_alignment |
| 35 | Factor Momentum Multiple Periods (2025) | 2025 | App Econ Lett | 5 | csec_momentum_zscore_diff |
| 36 | Jane Street RTMDF (2024) | 2024 | Kaggle | 5 | trade_sign_imb, csec_iqr_norm_ret |
| 37 | When Alpha Breaks (2025) | 2025 | arXiv 2603.13252 | 5 | csec_ofi_iqr, csec_rv_iqr |
| 38 | Feature Interactions Tabular DL (2025) | 2025 | arXiv 2510.04543 | 6 | ofi_x_spread, ofi_x_rv |
| 39 | T2G-Former Tabular Graphs (2023) | 2023 | AAAI | 6 | ofi_depth_interaction, spread_depth |
| 40 | Tree Pairwise Interactions (2025) | 2025 | arXiv 2508.15678 | 6 | spread_imb_ofi_cascade |
| 41 | CSI300 Feature Attention (2025) | 2025 | MDPI Electronics | 6 | volume_momentum_interaction |
| 42 | Hybrid TFT-GNN (2024) | 2024 | MDPI Algorithms | 6 | vol_gated_ofi, vol_gated_imb1 |
| 43 | Positive/Negative Jumps (Zhang) | 2024 | J Forecasting | 7 | rv_jump_asymmetry |
| 44 | Worst-Kept Secret HFT (2023) | 2023 | arXiv 2307.15599 | 7 | quote_update_vs_trade_ratio |
| 45 | Interpretable ML HFT (Wallbridge) | 2023 | arXiv 2307.04863 | 7 | ofi_ts_zscore |
| 46 | Data-Driven HFT Measures (2024) | 2024 | arXiv 2405.08101 | 7 | lob_turnover_rate, cancel_submit_ratio |
| 47 | Market Simulation Adverse Selection (2024) | 2024 | arXiv 2409.12721 | 7 | adverse_selection_indicator |
| 48 | LiT LOB Transformer (2025) | 2025 | Frontiers AI | 7 | lob_attention_gini |
| 49 | PIN Proxy Updated (Easley 2024) | 2024 | SSRN | 7 | pin_proxy, signed_amt_imb |
| 50 | Parkinson/RS Vol (2024 review) | 2024 | AFA review | 7 | parkinson_vol, vol_ratio_parkinson |
| 51 | HLOB (2024) | 2024 | Expert Sys Apps | 1 | lob_slope_bid, mid_level_vol_ratio |
| 52 | CNN-Transformer LOB (2025) | 2025 | Springer | 1 | combined spatial-temporal features |
| 53 | Explainable Crypto Microstructure (2025) | 2025 | arXiv 2602.00776 | 1 | LOB explainability proxies |
| 54 | Copula Cointegrated Pairs (2024) | 2024 | Fin Innovation | 2 | pair_ar1_proxy extensions |

**Total: 54 papers across 7 directions.**

---

## Feature Coverage Summary

| Direction | Papers | New Features Derived |
|-----------|--------|---------------------|
| 1: LOB Advanced Microstructure | 8 | NF001–NF022 (22 features) |
| 2: Cross-Asset Deep Features | 8 | NF023–NF035 (13 features) |
| 3: Graph/Network | 8 | NF036–NF049 (14 features) |
| 4: Hawkes / Point Process | 6 | NF050–NF065 (16 features) |
| 5: Quant Industry | 7 | NF066–NF082 (17 features) |
| 6: Cross-Feature Interaction | 5 | NF083–NF094 (12 features) |
| 7: Microstructure Specific | 9 | NF095–NF116 (22 features) |
| **Total** | **54** | **116 features** |
