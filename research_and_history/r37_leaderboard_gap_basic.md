# R37 — Leaderboard Gap: 15 Basic-Yet-Disruptive Ideas We Haven't Tried

> Date: 2026-05-07
> SOTA: iter_012 = LOSO-equiv +26.44; public team = +29.18 → gap +2.74
> Goal: surface **simple** ideas (< 4h each, no NN, no exotic loss) that have **NOT been tried** and could close the gap.

---

## 0. Re-statement of the Gap

| metric | us (iter_012) | public-leader | ratio |
|---|---|---|---|
| accuracy | ~0.330 | 0.330 | 1.0× |
| recall | ~0.20 | 0.302 | **1.5×** |
| F0.5 | ~0.30 | 0.324 | 1.08× |
| n_active | 88,553 | ~174,000 | **2.0×** |
| single_pnl | 0.000299 | 0.000167 | **0.56×** |
| cum_pnl | +26.44 | +29.18 | 1.10× |

**Identity check**: 88,553 × 0.000299 = 26.48 ≈ 26.44 ✓; 174k × 0.000167 = 29.06 ≈ 29.18 ✓.

**One-line diagnosis**: at the **same accuracy**, they trade **2× more** with **half** the per-trade alpha. They are accepting many marginal signals our 4D DE thresh refuses to take. Volume×alpha = +10% PnL.

Implication: the lever is NOT "find a single magic feature that doubles per-trade alpha"; it is **"reach further down the probability distribution without hurting accuracy"**. That's a **decision-rule + calibration + small-signal modeling** problem, not a deep-feature problem.

---

## 1. What we've already done (do NOT re-propose)

Already tried and ruled out (positive or negative — see history_and_important_notes.md):

- ❌ NN h_60 (DeepLOB CNN, GRU): T1, T19, T32, T60
- ❌ Pseudo-labeling: T65
- ❌ Focal loss / class weight: T41
- ❌ Group DRO multi-seed: T45
- ❌ Calibration (Platt / Isotonic / Temperature): T40
- ❌ Cross-sym pooling: T48 (inference incompat)
- ❌ Cross-sym mixup: T52
- ❌ alpha101 + alpha191 factor zoo: T22 (collinear collapse)
- ❌ ReVol+SG (T35 — gain rank #1 but OOF -0.67)
- ❌ Multi-horizon stacking ridge meta: T12, T47 (-1+ LOSO)
- ❌ PnL-aware loss (linear / sqrt / capped weight): T57 (best +11 << +13.6 baseline)
- ❌ Adversarial reweighting: T66 (AUC=1.0 → no useful overlap)
- ✅ Triplet imbalance + HMA Fibonacci + spread-resid: T55 stage 4 (in iter_012)
- ✅ Adaptive momentum + OFI toxicity + bipower + spread regime + trade-direction persistence + liquidity asymmetry: T68 stage 5 (in iter_012)
- ✅ Walk-forward CV split (date 0-75 train / 76-79 val): T64 → +25.94
- ✅ Multi-seed (5) ensemble: yes
- ✅ DE 4D (T_up, T_dn, d_up, d_dn) thresh on full 442k test: yes
- ✅ Multi-split bagging K=20: T67 (no improvement over 5-seed)
- ✅ Binary cascade stage A: T72 (in progress)

Bottom line: the **feature side** is largely tapped out (200+ engineered dims, sym-invariance audited). The **decision rule + small-signal recovery** side is **largely untouched**.

---

## 2. The 15 Ideas — Prioritized by ROI (gain / time)

### 🥇 P0 — top 5 (highest ROI, all < 4h, all untried)

| # | Idea | Expected +LOSO | Cost | Why disruptive |
|---|------|----------------|------|----------------|
| **1** | **Regression on signed Δmid + EV-gated decision** | **+0.5 ~ +2.0** | 3h | Replaces 3-class CE with continuous return target → naturally accepts marginal signals (the gap's root cause) |
| **2** | **Multi-horizon vote ensemble (we have h_5/10/20/40/60 from iter_002 already!)** | **+0.5 ~ +1.5** | 2h | We've trained 5 horizon models but only ship h_60. Use h_5/10/20 prob as side-info in h_60 decision (NOT stacking — direct vote) |
| **3** | **Final retrain on date 0-79 (train + val combined)** with `n_iter = round(best_iter × 80/76)` | **+0.2 ~ +0.6** | 30 min | We are throwing away **5% of data** (date 76-79 = 73,680 rows). Free win. |
| **4** | **Time-of-day cyclic encoding (sin/cos of t_local)** as a feature | **+0.2 ~ +0.5** | 30 min | We've never explicitly added intraday position; t_local IS available; strong intraday vol/spread cycle exists in A-stocks |
| **5** | **Decoupled threshold per intraday time bucket** (3-4 buckets: open, mid-am, mid-pm, close) | **+0.3 ~ +0.8** | 1.5h | Same model, different DE thresh per bucket. The model's prob distribution shifts intraday; one global thresh leaves money on the table |

### 🥈 P1 — middle 5 (good ROI, < 4h)

| # | Idea | Expected +LOSO | Cost | Why disruptive |
|---|------|----------------|------|----------------|
| **6** | **Two-stage binary classifiers (UP-vs-rest, DOWN-vs-rest)** trained independently → trade only when one side fires | +0.3 ~ +1.0 | 3h | T72 stage A is partial. **Full** two-binary decoupling avoids 3-class softmax mass-stealing issue, is well-known in HFT; we never finished it |
| **7** | **Order-flow ratio features (mb-ma)/(mb+ma), (lb-la)/(lb+la), (cb-ca)/(cb+ca) over W=5/20/50** | +0.2 ~ +0.6 | 1h | Schema gives raw `*_intst` but we mostly use them raw or as EWMA. Pure ratios are sym-invariant, scale-free, cheap, surprisingly missing |
| **8** | **Lee-Ready last-trade direction** (sign of last mid_diff bucketed UP/DOWN/UNCH) over W=5/20/50 + share | +0.2 ~ +0.5 | 1h | Microstructure standard; extremely simple (1-line per W); gives directional persistence signal that's orthogonal to OFI |
| **9** | **Past-return-at-lag-k features**: log(mid_t/mid_{t-k}) for k = 1, 2, 5, 10, 20, 50 | +0.2 ~ +0.5 | 30 min | We have `mid_diff` and rolling stats but never explicit fixed-lag returns. Sym-invariant (log-ratio). Direct momentum proxy |
| **10** | **Order-book slope** = OLS slope of (cumsum_bsize, bid_price) for top-5 levels (and ask side) | +0.2 ~ +0.5 | 1h | Encodes price impact / depth elasticity. Sym-invariant since slope of normalized prices. Public team likely has this |

### 🥉 P2 — last 5 (lower expectation OR slightly higher cost; all still simple)

| # | Idea | Expected +LOSO | Cost | Why |
|---|------|----------------|------|-----|
| **11** | **Multi-α EWMA snapshot for Δmid** (α ∈ {0.05, 0.1, 0.3, 0.5}) on existing key features (mid, OFI, intst) | +0.1 ~ +0.4 | 1h | We have window-based stats but α-EWMA gives different smoothing kernel → low-correlation extra channels |
| **12** | **KNN-target retrieval (Optiver 2021 nyanp 1st)** — FAISS over 226-d on train; at inference query top-K nearest train rows, use mean target as 1 feature | +0.3 ~ +1.0 | 4h | Genuinely different signal source; sym-agnostic safe; **never tried**. Cost slightly higher; might be too slow at inference |
| **13** | **Margin-score gating with single-axis threshold**: `s = max(p_up,p_dn) − p_flat`; trade if s > T1 and side has |p_up − p_dn| > T2 | +0.1 ~ +0.4 | 1h | DE 4D may overfit OOF noise (T62 showed split-train gap +0.28 LOSO). 2D simpler thresh = less overfit, may generalize better |
| **14** | **Snapshot ensemble: drop early stopping; train 5 seeds × {500, 600, 700} rounds, average all 15** | +0.1 ~ +0.4 | 3h | Early stopping on tiny val (4 days) is noisy; multi-snapshot averaging is variance-reduction free win |
| **15** | **Class-1 (flat) downsampling** during training: keep 50% flat samples, all directional | +0.1 ~ +0.3 | 2h | Class-balanced weighting was tried; **explicit downsampling** is different (changes the bagging distribution, not just gradient weight). May shift prob distribution toward more directional confidence |

---

## 3. Top 5 — Specific Next Actions

### #1 Regression on signed Δmid + EV-gated decision  ⭐⭐⭐
**Hypothesis**: The 3-class CE loss + 4D thresh is two layers of indirection over what we actually want, which is `argmax_{action ∈ {-1,0,+1}} E[r·action − fee·|action|]`. By regressing `Δmid` directly and using EV decision rule, we **automatically** accept marginal signals (small predicted moves still beat fee) without needing per-class probability mass.

**Why basic but disruptive**:
- Hyd Optiver 2023 1st place used **MAE regression on continuous return**, not classification.
- We have 359 features and 5 LightGBM seeds — can swap loss objective in one line.
- Decision rule becomes 1 hyperparam (`fee_threshold = 2 × 0.0001`) instead of 4D DE → less overfit.

**Implementation**:
1. New target: `y = Δmid_h60 = midprice_{t+60} − midprice_t` (raw signed value, NOT divided by α).
2. Train 5-seed LightGBM with `objective='regression'`, `metric='mae'` or `huber`, on existing 359 features. Same data split (V4 walk-forward).
3. Inference: predict `ŷ`. Decide `action = +1 if ŷ > +0.0002, −1 if ŷ < −0.0002, 0 otherwise` (the 0.0002 ≈ 2× fee, can DE-tune as 1D threshold).
4. Eval on 442k test: compute cum_pnl per sym, compare to iter_012.
5. If marginal: ensemble = 0.5 × cls_prob + 0.5 × regression-derived prob.

**Cost**: 3h. **Risk**: regression model may underweight tail (rare big moves). Mitigation: try `objective='huber', alpha=0.95` quantile.

---

### #2 Multi-horizon Vote (cheap, we already have inputs)  ⭐⭐⭐
**Hypothesis**: We trained h_5/10/20/40/60 models (iter_002). They all see the same window but predict different futures. h_5 capturing immediate direction is a **strong prior** for h_60. Public team likely uses cross-horizon agreement to filter false positives **and** find more true positives.

**Why basic but disruptive**:
- We already have all 5 model.txt files trained.
- "Multi-horizon stacking" T47 was negative because it tried complex meta-learning. **Simple vote** (e.g. UP if k of 5 horizons predict UP) is simpler and may work where stacking didn't.

**Implementation**:
1. Load h_5/10/20/40 models from iter_002 + h_60 model from iter_012 (different feature set; need re-extract h_5/10/20/40 on 359-d, OR use 226-d versions and accept feature mismatch).
2. For each test window, compute 5 prob vectors (each 3-class).
3. Two strategies to test:
   - **Vote**: UP if `(# horizons s.t. argmax==2) ≥ k` and `# DOWN < j`. Sweep (k, j).
   - **Weighted geometric mean** of (prob_up / prob_dn) across horizons, then threshold.
4. DE-tune the new decision rule on V4 val OOF.

**Cost**: 2h (mostly running existing models on test cache). **Risk**: h_5/10/20 are noisy (T2 +21 LOSO but platform -8) — might dilute signal. Try only `h_60 + h_40` first as cheapest version.

---

### #3 Train+val combined final retrain  ⭐⭐⭐
**Hypothesis**: V4 walk-forward holds out date 76-79 (5%) for early stopping. Once `best_iter` is selected, retrain on date 0-79 with `n_iter ≈ best_iter × (80/76) = best_iter × 1.053`. Free 5% more data.

**Why basic but disruptive**:
- Standard Kaggle final-model recipe; we just never did it.
- 5% data on 1.47M rows = 73k extra rows = potentially +0.1-0.3 LOSO.

**Implementation**:
1. Read `best_iter` from each of 5 iter_012 seed model `train.log`.
2. Re-train each seed with `num_boost_round = round(best_iter × 1.053)`, `early_stopping=None`, on date 0-79 combined.
3. Re-evaluate on full 442k test with iter_012's DE thresh (probably need to re-tune slightly). Compare.

**Cost**: 30 min (5 seeds × ~3 min train + DE re-tune). **Risk**: tiny (worst case re-tune thresh unchanged).

---

### #4 Time-of-day cyclic encoding (sin/cos t_local)  ⭐⭐
**Hypothesis**: `t_local` is the only intraday position info platform allows (date masked). A-stock LOB shows strong intraday seasonality (open spike, lunch lull, close volatility). Adding 2 features `sin(2π·t_local/2001), cos(2π·t_local/2001)` lets every tree split on intraday position cheaply.

**Why basic but disruptive**:
- Schema includes `time` as a real timestamp; can derive minute-of-day or sin/cos.
- Almost free (2 features), max W = 0 (uses only current row).
- We've **never** added it — verified no `time_of_day` / `tod` / `sin*time` in any T44/T68 builder.

**Implementation**:
1. Add to feature builder: extract `t_local` (=row index in session), compute `sin/cos(2π·t_local/2001)`. Or extract `time` column → minute-of-day → sin/cos.
2. Re-train 1 seed (seed=42) as pilot on date 0-75, eval on 76-79 → if val PnL improves +0.1+, run full 5-seed.
3. Re-tune DE thresh on test, compare.

**Cost**: 30 min pilot, 1h full. **Risk**: if no intraday signal exists in our schema (unlikely), zero gain.

---

### #5 Decoupled threshold per intraday time bucket  ⭐⭐⭐
**Hypothesis**: The model's probability distribution shifts intraday (more vol at open/close, less at midday). One global DE 4D thresh is **average-optimal**, not regime-optimal. Slicing by 3-4 intraday buckets and DE-tuning per bucket should net +0.3-0.8.

**Why basic but disruptive**:
- Hyd Optiver 2023 1st explicitly used 3 intraday buckets (0-290s / 300-470s / 480-540s) for **separate decisions**.
- No retraining needed; only modify threshold logic in Predictor.
- T62 showed our DE thresh has +0.28 train-eval gap → bucketed thresh should help more.

**Implementation**:
1. Use existing iter_012 5-seed avg probs on test (already have these).
2. Split test rows by `t_local` into 3 buckets: `[0, 700)`, `[700, 1400)`, `[1400, 2001)` (or 4 buckets).
3. DE-tune (T_up, T_dn, d_up, d_dn) per bucket independently on tune-half + eval-half (use T62's split protocol to avoid overfit).
4. Sum cum_pnl across buckets, compare to global thresh.
5. Predictor: at inference, pick bucket from t_local then apply matching thresh.

**Cost**: 1.5h (DE × 3 buckets is faster than full DE since each bucket is smaller). **Risk**: per-bucket DE may overfit further; mitigate with T62-style train/eval split per bucket.

---

## 4. Why These 5 Beat the Other 10

The top-5 attack the **gap mechanism** directly:
- #1 (regression) and #6 (binary cascade) reframe the **prediction unit** to one that naturally captures marginal signals.
- #2 (multi-horizon vote) and #5 (per-bucket thresh) reuse **information we already have** in a different decision rule.
- #3 (train+val) and #4 (time-of-day) are **free wins** (≤ 30 min).

The middle 5 (P1) are good engineering hygiene but more incremental.

The last 5 (P2) are speculative or higher-cost.

---

## 5. Suggested 1-Day Sprint Order

If we have 8 hours (one worker, sequential), do:

```
Hour 0-0.5:  #3 Final retrain on train+val            (free baseline lift)
Hour 0.5-1:  #4 Time-of-day cyclic encoding pilot     (1 seed, decide go/no-go)
Hour 1-2.5:  #5 Decoupled threshold per intraday bucket
Hour 2.5-4.5: #2 Multi-horizon vote (using existing h_5/10/20/40/60 models)
Hour 4.5-7.5: #1 Regression on signed Δmid + EV gate
Hour 7.5-8:  Compare all 4 deltas, decide which combine for iter_013
```

**Acceptance bar**: any individual lever needs to clear DE noise floor of ±0.10 LOSO; bundle target +0.5 LOSO over iter_012's +26.44 → iter_013 target ≈ +27.0.

If #1 (regression) clears +1.0 alone, **stop further work and ship iter_013 = #1 + #3**. Regression head is the most likely "basic but disruptive" answer to the 2x-trades-same-acc gap.

---

## 6. Negative Predictions (won't work — explanations)

Things I considered but excluded from the 15:

- **More boost rounds + smaller LR** (variant of #14): T63 reg sweep already explored hyperparam space; gains < noise.
- **More seeds (10 instead of 5)**: T67 K=20 multi-split bagging found no lift over 5-seed. Diversity is saturated.
- **Different gradient-boosting library (XGBoost / CatBoost)**: T54, T56 tried — within DE noise.
- **More R34-style hand-crafted features**: 5 stages of stage features brought us +13 → +26; stage 6 likely <+0.1 and risks overfit.
- **Mixup-style augmentation (other than T52)**: cross-sym mixup tried, gentle within-sym mixup unlikely to add over current feature-jitter aug_a.
- **Online / incremental updates**: hard-violates C2 (Predictor state).
- **Cross-sym rank features**: T48 NO-GO due to inference-time other-sym-window unavailability.

---

## 7. References

- `r33_kaggle_hft_winners.md` (this report's parent — gap analysis and Kaggle winner trick distillation)
- `r33_recall_tricks.md` (16 trick list — confirms which we already implemented)
- `r34_top15_features.md` (feature side — confirms feature catalog is saturated)
- `history_and_important_notes.md` §10 (R31 SOTA survey, validates regression-on-return + EV-gate are actively-used 2024-25 alpha sources)
- `experiments/T57_pnl_aware_loss/` (our previous PnL-loss attempt — was at sample_weight level; #1 here is at objective level — different)
- `experiments/T70b_iter012/REPORT.md` (current SOTA breakdown)
- `experiments/T62_unbiased_thresh/results.json` (DE 4D overfit evidence: +0.28 train-eval gap supports per-bucket regularization)
