# T110 — Basic-Yet-Disruptive Feature Proposals (Worker R-T110)

> **Date**: 2026-05-08
> **Worker**: opus xhigh, research-only (no model training)
> **Hypothesis driving the search**: leaderboard top is **CPU LightGBM single
> model finishing in minutes** → the lever isn't model complexity, it's a small
> set of **basic but high-quality features we never explicitly added**.
> **Source rule**: every formula here is computable in a 100-tick window using
> raw schema columns only; sym-agnostic; no `date`; no cross-call state.

---

## Part 1 — Diagnostic of current 370-d schemeP (+stage4 +T74 +T88)

### 1.1 Family-by-family inventory

The current pipeline ships **370 schemeP cols** (= 154 raw last-tick + 196 T3/S1/S2/S3 extras + 20 stage5) plus **47 stage4** (T55) extras + **2 time** (T74) + **12 range-vol** (T88). Total ≈ **431 d** in iter_016.

| Family | Dims | Origin | What it captures |
|---|---:|---|---|
| Raw last-tick LOB | 154 | parquet | b/a price+size 10-lvls, intst×6, acc×6, OHLC, derived (midprice, spread, bid_diff, etc.) |
| **MLOFI** (signed queue change) | 30 | T3 | level k=1..10 × W∈{5,20,60} |
| **WMP** (Stoikov weighted mid) per level + balance12 | 11 | T3 | wmp_k for k=1..10, plus wmp1−wmp2 |
| **Realized vol unsigned** | 4 | T3 | RV at W∈{5,10,20,50} |
| **EWMA intensities** (4α × 6 channels) | 24 | T3 | exponential smoothing of *_intst |
| **Dual window z-score** (z_short−z_long) | 37 | S1 | within-window normalization on 37 raw cols |
| **Signed RV** (asymmetry) | 3 | S1 | (rv_pos−rv_neg)/(rv_pos+rv_neg) for W∈{20,50,100} |
| **Kyle bet-size invariant** (cube-root norm) | 2 | S1 | amount/(|amount|·σ)^{1/3}, W∈{50,100} |
| **EWMA-OFI** (3 lvls × 4α) | 12 | S1 | smoothed multi-α MLOFI |
| **Quantile rank** (W=100) | 20 | S2 | rank of last value within 100-tick window |
| **Realized skewness** (3rd moment) | 3 | S2 | rskew of log-ret W∈{20,50,100} |
| **GOFI** (generalized OFI w/ passive size adj) | 30 | S2 | levels 1..10 × W∈{5,20,60} |
| **Kyle's λ** (Δp ~ signed_dvol slope) | 2 | S2 | rolling W∈{50,100} |
| **Vol burst** (vol_t / mean_W) | 4 | S2 | volume burst + amount burst W∈{20,50} |
| **EWMA-resid mid** (innovation) | 1 | S3 | (mid − ewma_α=0.05)/ewma |
| **RV ratio** (short/long) | 3 | S3 | rv_5/rv_50, rv_20/rv_100, rv_50/rv_100 |
| **J-share** (jump fraction = (rv−bv)/rv) | 4 | S3 | W∈{20,30,50,100} |
| **Cancel imbalance** (cb-share − ca-share) | 3 | S3 | W∈{20,50,100} |
| **Roll's effective spread** | 3 | S3 | 2·sqrt(−cov(Δp,Δp+1)) ratio |
| **Adaptive momentum** | 3 | T68 | (mid_t−mid_{t-W})/std for W∈{20,50,100} |
| **OFI toxicity** | 4 | T68 | corr(OFI_lvl, Δmid) |
| **Multi-scale signed bipower** | 3 | T68 | signed BV / RV |
| **Spread regime** | 3 | T68 | (spr−median)/IQR, W∈{20,50,100} |
| **Trade-direction persistence** | 3 | T68 | autocorr of sign(Δmid), W∈{20,50,100} |
| **Liquidity asymmetry top-5** | 4 | T68 | log(Σbsize/(Σasize)), W∈{5,20,50,100} |
| **Triplet imbalance** | 24 | T55 | bounded form (hi−mid)/(hi−lo), 24 hand-picked triplets |
| **HMA Fibonacci residual** | 20 | T55 | Hull MA residual on mid/log_ret/spread/imb, W∈{5,13,34,55,89} |
| **Spread regression residual** | 3 | T55 | rolling OLS resid of mid_diff~spread |
| **Time of day** (sin/cos) | 2 | T74 | within-session phase encoding |
| **Range-based vol** (Park/GK/RS/YZ) | 12 | T88 | derived OHLC range estimators + ratios |

### 1.2 What is **structurally missing** vs known-strong references

After scanning HYD-1st (Optiver Trading at Close 2023), G-Research Crypto, Optiver Vol 2021, and Cont/Stoikov microstructure literature, the following families have **no representation in our 431-d set**:

| Missing family | Why we lack it | Why top-K teams almost certainly have it |
|---|---|---|
| **HYD interaction "physics" features** (price pressure, market urgency, depth pressure, spread-depth ratio) | Schema has each ingredient (`spread1`, `imbalance`, `bid_diff*`, sizes), but nobody wrote the **products**. | These are HYD's named features — directly cited as the most important new factors by the 1st-place writeup. |
| **Pairwise price imbalance (p_a−p_b)/(p_a+p_b)** for all useful price pairs | We have *triplet* imb (T55) but never the **pairwise** form, which HYD lists as a separate family. | Trivial 2-arg formula; HYD made a sweep over all price-pair combinations. |
| **Multi-scale fixed-lag log returns** log(mid_t/mid_{t−k}) for k∈{1,2,5,10,20,50} | We have RV (squared sum) and adaptive-momentum (W=20/50/100 only), but **no signed return at fixed lags**. The single most basic momentum primitive — astonishingly absent. | G-Research crypto winner explicitly cited multi-scale return as the highest-importance signal. |
| **Mid-price 2nd derivative (acceleration)** | HMA gives smoothed 1st derivative; we have no explicit (Δmid_W − Δmid_{2W}). | "Curvature" signals turning points before they appear in slope. |
| **Book-slope (depth-decay)** — OLS slope of log(bsize_k) vs level k | Schema gives 10 levels of size, we never aggregated their **shape**. | Standard microstructure depth elasticity (Næs-Skjeltorp 2006); cheap. |
| **Order-flow ratio panel** (mb−ma)/(mb+ma), (lb−la)/(lb+la), (cb−ca)/(cb+ca), TFI = (lb+mb+ca)−(la+ma+cb) over W∈{5,20,50} | We have raw `*_intst` and EWMA of them, but never the directly **ratio'd / netted** form. | HYD: "Liquidity Imbalance / Matched Imbalance" series. |
| **EWMA crossover (MACD-style)** on mid, OFI, imbalance | We have single-α EWMA per channel; differences across α are missing. | Classic momentum primitive. |
| **Cross-window stability** — feature(first 50 ticks) vs feature(last 50 ticks) | We have within-window stats but never **"is this regime changing?"** across two halves. | Detects mid-window regime shifts that smoothed stats hide. |
| **Realized kurtosis** (4th moment of returns) | We have rskew (3rd) and jshare (jump fraction) but no kurt — fat-tail signature missing. | Cheap; orthogonal to skew. |
| **Classic L1 triplet (a1, wmp_lvl1, b1)** | T55 explicitly skipped (a1, mid, b1) because mid=(a1+b1)/2 makes the bounded form degenerate (=1). It did NOT try `wmp_lvl1` in place of `mid`, which is non-degenerate. | The single most fundamental triplet in HYD's recipe. |

These 10 gaps are the proposal targets in Part 2. The thesis: each is a *single-product, single-ratio, or single-OLS-slope* — **basic primitives** — but they bring information channels currently absent in our 431-d.

### 1.3 What we shouldn't propose (already saturated)

- More multi-window MLOFI / GOFI / WMP variants → diminishing returns (T20 schemeG +400d failed).
- Alpha101/191 zoo → tested, collinear (T22 +21.80 ≈ baseline).
- ReVol+SG with NN smoothness → tested, OOF -0.67 (T35).
- Auto-ML factor mining → no signal beyond hand-engineered (T22).
- Cross-sym features (rank/mean across syms at same t) → **violates inference-time constraint** (Predictor sees one window at a time, can't see other syms' windows). Hard NO.

---

## Part 2 — 10 proposed feature tricks (with formulas)

For each: **F**ormula, why **B**asic, why **D**isruptive (i.e. what new info channel), implementation **C**ost, expected **L**OSO Δ, **R**isk, redundancy with existing 431-d.

### Idea-1 — HYD interaction quartet (price pressure / market urgency / depth pressure / spread-depth ratio)

**F** (per row at last tick, also W-rolling at W∈{5,20,50}):
```
liq_imb         = (bs1 - as1) / (bs1 + as1)                # have as 'imbalance'
price_pressure  = (bs1 - as1) * (a1 - b1)                  # imbalance_size · spread
market_urgency  = (a1 - b1) * liq_imb                      # spread × signed liq_imb
depth_pressure  = (as1 - bs1) * ((a10 - b10) - (a1 - b1))  # near-far spread differential
spread_depth    = (a1 - b1) / (bs1 + as1 + EPS)            # cost per unit liquidity
```
4 instantaneous + 4×3 W-rolling means = **16 dims**.

**B** Each is a 2-term product/ratio of existing schema columns. ~10 lines numpy. Sym-agnostic by construction (all are scale-invariant ratios or signed products of comparable units).

**D** HYD-1st **explicitly** lists these by name as the most important new factors over their LightGBM baseline. Our 431-d has NO product form between size-imbalance and price-spread; LightGBM cannot recover a multiplicative interaction from a single split.

**C** 2 h. **L** **+0.5 ~ +1.5** (HYD reports these as core; we are missing all four). **R** Some redundancy with existing `imbalance` and `spread1`, but the multiplicative form is genuinely new. **Redundant with**: none (T55/T68 do not include products).

---

### Idea-2 — Pairwise price imbalance

**F** For every pair (p_i, p_j) in the price-anchor set `P = {b1, b5, b10, a1, a5, a10, mid, avgbid, avgask, wmp_lvl1, wmp_lvl5}`:
```
pair_imb(p_i, p_j) = (p_i - p_j) / (p_i + p_j + EPS)
```
Compute at last tick. Pick the 12 **most informative** pairs (others are linearly dependent or trivially redundant) — e.g.
```
(b1, a1), (b1, a10), (a1, b10), (b5, a5),
(avgbid, avgask), (b1, mid), (a1, mid),
(wmp_lvl1, mid), (wmp_lvl1, wmp_lvl5),
(b1, b10), (a1, a10), (avgbid, b1)
```
**12 dims**.

**B** Trivial elementwise vector op, no rolling.

**D** HYD's own write-up names "pairwise price imbalance" alongside "triplet imbalance" as separate features. We implemented triplet (T55) but never the pairwise. Pair captures the *direction and depth-shape information at level boundaries*; triplet only captures three-point relative position.

**C** 0.5 h. **L** **+0.3 ~ +1.0**. **R** Some redundancy with `bid_diff*` and `spread*` (which are just differences without normalization), but the ratio form is dimensionless and works under aug_a price scaling. **Redundant with**: T55 triplet (different geometry — pair vs three-point).

---

### Idea-3 — Multi-scale fixed-lag log returns

**F** Let `mid = midprice1`. For each k∈{1, 2, 5, 10, 20, 50}:
```
log_ret_k = log( (mid_t + 1) / (mid_{t-k} + 1) )       # the "+1" is the schema's relative-to-prev-close base
```
**6 dims**. Optionally clip to ±0.05 to bound tails.

**B** One-line subtraction in log space, no window stats.

**D** This is the single most basic momentum primitive in time-series ML; **it is shockingly absent from our 431-d**. We have RV (squared sum, throws away sign), adaptive momentum (W=20/50/100 normalized by std), and HMA (smoothed). None of those is the raw signed return at a *fixed* lag. G-Research Crypto winner explicitly named multi-scale return as the highest-importance feature. The reason it works: at very different lags (k=1 vs k=50), the autocorrelation structure of returns flips sign in HFT — short-lag has reversal (bid-ask bounce), long-lag has momentum. LightGBM can only model the regime via splits if the lag is exposed.

**C** 0.3 h. **L** **+0.4 ~ +1.0**. **R** Tiny — k=1 will be noisy (single-tick), but LightGBM can handle. **Redundant with**: Stage4 HMA log-return (smoothed), but smoothing kills the high-freq channel. The raw form is orthogonal.

---

### Idea-4 — Book-slope (depth decay) per side + asymmetry

**F** For the last tick, with `levels = [1..10]`:
```
log_bs_k = log1p(bsize_k)         k=1..10           # bid sizes
log_as_k = log1p(asize_k)         k=1..10           # ask sizes

slope_b, intercept_b = polyfit(levels, log_bs_k, deg=1)    # OLS slope (closed form)
slope_a, intercept_a = polyfit(levels, log_as_k, deg=1)
slope_diff           = slope_b - slope_a
slope_avg            = 0.5 * (slope_b + slope_a)

# R²-style fit quality
ss_tot_b = sum((log_bs_k - mean)^2);  ss_res_b = sum((log_bs_k - (intercept_b + slope_b·k))^2)
fit_b = 1 - ss_res_b / (ss_tot_b + EPS)
fit_a = 1 - ss_res_a / (ss_tot_a + EPS)
```
**6 dims**: `slope_b, slope_a, slope_diff, slope_avg, fit_b, fit_a`.

**B** Closed-form OLS slope = `sum((k − k̄)(y − ȳ)) / sum((k − k̄)^2)`. Pure numpy on (N, 10) arrays.

**D** Encodes **depth elasticity** — how fast cumulative liquidity grows with level. A flat slope means deep, supportive book; a steep slope means thin/illiquid. Næs-Skjeltorp 2006 is the canonical reference. We currently have only sums (`totalbsize`, `totalasize`) and per-level raw sizes; the *shape* parameter is missing.

**C** 1 h. **L** **+0.3 ~ +0.8**. **R** None. **Redundant with**: stage4 (no), stage5 liq-asym (only top-5 sum, no shape).

---

### Idea-5 — Mid-price acceleration (2nd derivative)

**F** For W∈{5, 10, 20}:
```
delta_now  = mid_t          - mid_{t-W}
delta_prev = mid_{t-W}      - mid_{t-2W}                # need 2W ≤ 100 → W ≤ 50
accel_W    = delta_now - delta_prev                     # 2nd difference at scale W
accel_norm_W = accel_W / (std(mid[-2W:]) + EPS)         # std-normalized for sym-invariance
```
**6 dims** (3 W × {raw, normalized}).

**B** Two subtractions per W; pure last-tick op.

**D** Curvature signals **trend exhaustion / inflection**, e.g. price was rising fast and is now decelerating → mean-reversion likely. HMA smooths out this very signal. Our adaptive_mom (T68) captures 1st-derivative speed but not 2nd. In quant trading, "jerk" features are well-known but rarely shipped without explicit construction.

**C** 0.5 h. **L** **+0.2 ~ +0.6**. **R** Mid-price granularity is small in HF — could be near-noise at W=5. Mitigation: use **midprice1** + **wmp_lvl1** parallel versions and let LightGBM decide. **Redundant with**: HMA Fibonacci residual provides smoothed momentum, but explicit 2nd-derivative is orthogonal.

---

### Idea-6 — Order-flow ratio + TFI panel

**F** For each side-pair (ratios in [-1, 1]) and each W∈{5, 20, 50}:
```
mb_ma_imb_W = sum_W(mb_intst - ma_intst) / (sum_W(mb_intst + ma_intst) + EPS)
lb_la_imb_W = sum_W(lb_intst - la_intst) / (sum_W(lb_intst + la_intst) + EPS)
cb_ca_imb_W = sum_W(cb_intst - ca_intst) / (sum_W(cb_intst + ca_intst) + EPS)
TFI_W       = sum_W( (mb + lb + ca) - (ma + la + cb) )   # bull pressure − bear pressure
TFI_norm_W  = TFI_W / (sum_W(all 6) + EPS)               # normalized to [-1, 1]
```
**5 ratios × 3 W = 15 dims**.

**B** `np.add.reduce` on 100-tick window slices. Pure ratio → sym-agnostic.

**D** Schema gives `*_intst` raw; T3 EWMAs them. Neither path computes **directional ratios** — EWMA loses the netted-direction information. HYD calls these "Liquidity Imbalance" / "Matched Imbalance" and they are top-importance there. The TFI is the canonical Cont (2014) "Trade Flow Imbalance" which is well-known to dominate raw OFI for short-horizon return prediction.

**C** 1 h. **L** **+0.3 ~ +1.0**. **R** Partial overlap with `cancel_imb_W` (S3, only cb vs ca). **Redundant with**: cancel_imb (1 of 5, different family).

---

### Idea-7 — EWMA crossover (MACD-style) on mid, imbalance, OFI

**F** For each base series x ∈ {mid, imbalance, mlofi_W20_lvl1, totalbsize-totalasize}, use two EWMAs with α_short / α_long:
```
ewma_s = EWMA(x, α=0.3)        # half-life ~2 ticks
ewma_l = EWMA(x, α=0.05)       # half-life ~14 ticks
macd   = ewma_s - ewma_l       # signed crossover at last tick
```
4 series × 1 crossover = **4 dims**.

**B** Two `lfilter` calls, last value subtract. Reuses existing EWMA infrastructure.

**D** Existing schemeP has `ewma_a*_*_intst` (24 dims) for intensities and `ewma_resid_a0.05` for mid. None of them contain the **difference between two α**, which is the canonical MACD signal in TA — often the single highest-importance feature in retail quant systems. Differences between α decompose smoothing scales and amplify medium-frequency oscillation while damping short noise.

**C** 0.5 h. **L** **+0.2 ~ +0.6**. **R** Some collinearity with `ewma_resid_a0.05` (which is mid-ewma_l, related but different baseline). **Redundant with**: ewma_resid_a0.05 (partial — only mid).

---

### Idea-8 — Cross-window stability (first-50 vs last-50)

**F** For each base series x ∈ {mid_diff, OFI_lvl1, imbalance, spread1, log_ret}, compute over the 100-tick window:
```
m_first = mean(x[t-99 : t-50])
m_last  = mean(x[t-49 : t])
s_first = std(x[t-99 : t-50])
s_last  = std(x[t-49 : t])

shift_W100      = m_last - m_first                                 # mean-shift
vol_change_W100 = log((s_last + EPS) / (s_first + EPS))            # vol-regime change
```
5 series × 2 stats = **10 dims**.

**B** Two slicing-mean / std calls per series.

**D** All our existing rolling stats compute statistics over the whole window. None of them detects **regime change *within* the window**. A series that's drifting up then crashing down may have mean=0 and rv=high (looks neutral), but `m_last − m_first` reveals the swing direction. This is a classical "early-warning" feature in change-point detection. Sym-agnostic since it's a difference of two means in the same window.

**C** 1 h. **L** **+0.2 ~ +0.6**. **R** Some collinearity with adapt_mom (T68) which is `mid_t − mid_{t-W}` normalized — but adapt_mom uses only endpoints, not the **mean of two halves**. The latter is robust against single-tick spikes at the boundaries. **Redundant with**: T68 adapt_mom (partial — different geometry).

---

### Idea-9 — Realized kurtosis + Garman-Klass-style fat-tail signal

**F** Let `r = log(mid_t / mid_{t-1})` per tick, computed in batch. For W∈{20, 50, 100}:
```
mu  = mean(r[-W:])
sd  = std(r[-W:]) + EPS
r4  = mean( ((r[-W:] - mu) / sd) ** 4 )         # excess kurtosis = r4 - 3
rkurt_W = clip(r4 - 3.0, -10, 10)

# tail-share: fraction of |r| in the upper 5% of |r| within window
abs_r   = abs(r[-W:])
q95     = quantile(abs_r, 0.95)
tail_share_W = mean(abs_r >= q95) * (abs_r * (abs_r >= q95)).sum() / (abs_r.sum() + EPS)
```
3 W × 2 features = **6 dims**.

**B** One mean of 4th power per W; one quantile per W.

**D** schemeP has `rskew` (3rd moment, 3 dims) and `jshare` (jump fraction via bipower, 4 dims) but **no kurtosis**. Kurtosis is orthogonal to skew — high-skew distribution can be platykurtic; high-kurt ones can be symmetric. Both Amaya 2015 (RSkew predicts cross-section returns) and Bollerslev 2020 (RKurt predicts variance risk premium) prove independent predictive content. The tail-share complements via fraction-of-energy concentration.

**C** 0.5 h. **L** **+0.1 ~ +0.4**. **R** Some collinearity with `signed_rv` (which is signed-vol asymmetry, different family). **Redundant with**: rskew (different moment order).

---

### Idea-10 — L1 triplet imbalance using wmp_lvl1 (T55 missed this)

**F** T55's bounded triplet `(hi - mid) / (hi - lo)` was applied to many triplets but **explicitly skipped (b1, mid, a1)** because `mid = (a1 + b1) / 2` makes that triplet degenerate (the bounded form is constant). However, `wmp_lvl1` ≠ `mid` because WMP weights by size:
```
wmp_lvl1 = (a1 · bsize1 + b1 · asize1) / (bsize1 + asize1)
```
which deviates from mid by the imbalance direction. Therefore the triplet `(b1, wmp_lvl1, a1)` is **non-degenerate**: when the book is balanced, triplet ≈ 0.5; when bid-heavy → wmp moves to a1 → triplet → 0; when ask-heavy → wmp → b1 → triplet → 1.
```
sorted   = sort([b1, wmp_lvl1, a1])
trip_L1  = (sorted[2] - sorted[1]) / (sorted[2] - sorted[0] + EPS)    # bounded [0, 1]

# also do (avgbid, mid, avgask) which is also non-degenerate (avgbid != b1)
trip_avg = sort([avgbid, mid, avgask]) ; bounded form
```
**2 dims** (cheap, but the most fundamental L1 triplet that we currently lack).

**B** 6 lines of numpy.

**D** This is the **single most fundamental triplet in HYD's recipe**. T55 left it out for a defensible reason but never tried the wmp-substitution. The triplet captures the **direction in which trades are likely to push price next** based on micro-imbalance — a stronger signal than `imbalance` alone because it conditions on the *sign-direction* of the book pressure rather than just its magnitude.

**C** 0.3 h. **L** **+0.1 ~ +0.4**. **R** Highly correlated with `imbalance` (sign indicator) but provides bounded non-linear transform. **Redundant with**: T55 triplet family (same family, but new geometry).

---

### Summary of Part 2

| # | Idea | Dims | Cost (h) | Expected LOSO Δ | Risk |
|---|---|---:|---:|---|---|
| 1 | HYD interaction quartet (PP/MU/DP/SDR) | 16 | 2.0 | **+0.5 ~ +1.5** | ★ |
| 2 | Pairwise price imbalance | 12 | 0.5 | **+0.3 ~ +1.0** | ★ |
| 3 | Multi-scale fixed-lag log returns | 6 | 0.3 | **+0.4 ~ +1.0** | ★ |
| 4 | Book-slope per side + asymmetry | 6 | 1.0 | +0.3 ~ +0.8 | ★★ |
| 5 | Mid-price acceleration | 6 | 0.5 | +0.2 ~ +0.6 | ★★ |
| 6 | Order-flow ratio + TFI panel | 15 | 1.0 | **+0.3 ~ +1.0** | ★★ |
| 7 | EWMA crossover (MACD) | 4 | 0.5 | +0.2 ~ +0.6 | ★ |
| 8 | Cross-window stability (first50 vs last50) | 10 | 1.0 | +0.2 ~ +0.6 | ★★ |
| 9 | Realized kurt + tail-share | 6 | 0.5 | +0.1 ~ +0.4 | ★ |
| 10 | L1 triplet via wmp_lvl1 + (avgbid, mid, avgask) | 2 | 0.3 | +0.1 ~ +0.4 | ★ |
| **All** | combined | **83 d** | **7.6 h** | **+1.5 ~ +5.0** (assuming 30-50% ablation) | — |

(LOSO Δ entries in **bold** = high-confidence top-3 picks.)

---

## Part 3 — Top 3 recommendations

The ranking below is **(impact × feasibility) / risk**, with strong preference for *truly missing primitives* (not refinements of what's already in 431-d).

### 🥇 #1 — HYD interaction quartet (Idea-1)

**Why first**: HYD-1st explicitly names price_pressure, market_urgency, depth_pressure, spread-depth ratio as core. We have **none** of these as products. LightGBM cannot recover an `A·B` from a single split — the pre-computation is necessary. They are dirt-cheap to compute (last-tick + a few W-rolling means).

**Step-by-step (2 h)**:
1. Add a new module `experiments/T110_basic_features/hyd_interactions.py` with `compute_hyd_interactions_batch(X3d, col_idx) -> (N, 16)`.
2. Inside: 4 instantaneous formulas at the last tick + 4 W-rolling means at W∈{5,20,50}.
3. Sym-invariance unit test: KS-test the 16 columns across 5 syms on a random 50-session sample, expect worst KS < 0.45 (matches T68 stage5 threshold).
4. Concat to schemeP cache (370 → 386 d), retrain 1 seed h=60 on V4 walk-forward, single fold (held=2 — historically the hardest sym).
5. Compare DE-tuned LOSO sum against latest baseline; if ≥ +0.3 LOSO single-fold, expand to full 5-seed × 5-fold.
6. Target: per-fold gain of **+0.4–0.7** held=2; full LOSO sum +0.7–1.5 vs current best.

**Why basic but disruptive**: every ingredient is in schema; the multiplication itself is the missing piece.

**Risk**: moderate redundancy with `imbalance` and `spread1`, but the multiplicative form provides genuinely orthogonal info (a sum-form feature × a difference-form feature).

---

### 🥈 #2 — Multi-scale fixed-lag log returns (Idea-3)

**Why second**: This is the **simplest-looking idea on the list** (one log subtraction per row), yet it's fundamentally absent. Every momentum-style trading system has explicit lag-k returns. Our 431-d has RV (squared, no sign), adaptive_mom (only W=20/50/100, normalized), and HMA (smoothed), but **no raw signed return at fixed lag k**. Implementation cost is < 30 min and there is essentially no failure mode.

**Step-by-step (0.3 h)**:
1. Add to a `T110_basic_features.py` builder:
   ```python
   for k in (1, 2, 5, 10, 20, 50):
       mid_now  = X3d[:, -1, col_idx['midprice1']]
       mid_lag  = X3d[:, -1-k, col_idx['midprice1']]   # k <= 50, fits in 100-tick window
       log_ret_k = np.log((mid_now + 1.0) / (mid_lag + 1.0))
       feat[:, c] = np.clip(log_ret_k, -0.05, 0.05)
       c += 1
   ```
2. Apply same on `wmp_lvl1` for parallel 6 dims (12 total). (Optional: do both — costs nothing.)
3. Sym-invariance check (log-ratio of price → unitless, near-Gaussian zero-mean across syms; expect KS < 0.30).
4. Cache + retrain h=60 single fold; confirm per-fold gain ≥ +0.2.
5. Run full 5-seed × 5-fold; target full LOSO Δ **+0.4–1.0**.

**Why basic but disruptive**: the single most fundamental momentum primitive in time-series ML; absurdly missing in our 431-d. G-Research 2022 winner: top-importance feature.

**Risk**: at k=1 the signal is noisy (single-tick jitter), but LightGBM can split around it and the larger k=10/20/50 carry the bulk of signal.

---

### 🥉 #3 — Order-flow ratio + TFI panel (Idea-6)

**Why third**: order-flow signals (mb-ma, lb-la, cb-ca, TFI) directly model "bull pressure − bear pressure" in trade arrivals, which is what predicts mid-direction at short horizons (Cont 2014; Lopez de Prado VPIN). Our 431-d has raw `*_intst` and EWMAed forms, but neither path produces the direct **ratio**. Ratios are dimensionless, sym-agnostic, and bounded — exactly what works under aug_a robust training.

**Step-by-step (1 h)**:
1. Add `compute_orderflow_ratios_batch(X3d, col_idx) -> (N, 15)`:
   ```python
   for W in (5, 20, 50):
       mb = X3d[:, -W:, col_idx['mb_intst']].sum(-1); ma = ...sum(-1)
       feat[:, c] = (mb - ma) / (mb + ma + EPS); c += 1
       # similarly for (lb, la), (cb, ca)
       # TFI: sum_W( (mb+lb+ca) - (ma+la+cb) ) and its normalized form
   ```
2. KS-test sym-invariance across 5 syms (ratios always in [-1, 1] → expect KS < 0.20).
3. Cache + 1-seed pilot fold=2; +0.2 per-fold acceptance.
4. Full retrain target **+0.3–1.0 LOSO**.

**Why basic but disruptive**: the *netted*, *ratio'd* form is structurally absent; HYD's entire "Liquidity Imbalance" family lives on this primitive.

**Risk**: partial overlap with `cancel_imb` (covers cb vs ca only). The other 4 channels (mb-ma, lb-la, TFI, TFI_norm) are entirely new.

---

### Bundle plan

If time allows for one **single retrain cycle** (cheapest path to ship): combine Ideas **1+3+6 → +33 dims, 3.3 h coding** before training. Expected combined LOSO Δ **+1.0 ~ +3.0** (assuming 50–70% feature-additivity given moderate redundancy). This bundle is my recommended "T111 schemeQ" candidate.

If ablation budget exists, add Ideas 2 and 10 (also cheap, +14 dims, 0.8 h) for a secondary validation pass; both are shallow risk.

Ideas 4, 5, 7, 8, 9 are good but smaller-impact and should be deferred to a follow-up `schemeR` only if Ideas 1-3-6 cleared their threshold.

---

## Implementation contract checklist (all 10 ideas)

| Constraint | Verification |
|---|---|
| ✅ `date` not used | All formulas reference only `mid`, `b/a`, `bsize/asize`, `*_intst`. No `date` or session indicator. |
| ✅ `time` only via T74 (already shipped) | None of these 10 reference `time`; they are pure intra-window features. |
| ✅ `sym` not used | All ratios/products/log-returns are within-window or last-tick from a single 100-tick array. |
| ✅ Stateless across `predict()` calls | All operations depend solely on the input (N, 100, K) tensor; no buffers. |
| ✅ Causal | All `[t-W:t]` slicing; max lookback ≤ 100 (i.e. ≤ window length). |
| ✅ Sym-agnostic distribution | All formulas are ratios/log-returns/within-window products. KS-test in unit test required to confirm worst-pair KS < 0.45 (T68 stage5 acceptance threshold). |
| ✅ Vectorized | All formulas expressible as numpy operations on (N, T, K) tensors; no Python for-loops over N. |
| ✅ Memory | 83 new dims × 1.47M rows × float32 = 488 MB (manageable). |

---

RESULT: task=feature_basic top_3=[hyd_interaction_quartet, multi_scale_fixed_lag_log_returns, orderflow_ratio_TFI_panel] notes=schemeP 431-d has structural gaps in (a) HYD multiplicative interactions, (b) raw signed lag returns, (c) netted/ratio'd order-flow; all three are 1-line basic primitives, sym-agnostic, untested in this codebase; bundle 1+3+6 = +33 dims / 3.3 h / expected +1.0~+3.0 LOSO
