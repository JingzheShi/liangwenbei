# T23 — Robust Cross-Validation + Platform Predictor

**Goal**: Move beyond a single LOSO scalar to a richer estimate of platform PnL with confidence intervals and OOD-stress structure. Calibrate against the platform truth available now: **2 anchors** — `mmpc_demo` (task 2340) and `iter_002` (task 2435).

## TL;DR

**Critical update**: While this analysis was being assembled, iter_002's platform results landed (task 2435), giving us a **2nd calibration anchor**. With 2 anchors per horizon we can fit an exact linear map `platform = a × local + b`:

| horizon | a (slope) | b (intercept) | local needed for platform = 0 |
|---|---|---|---|
| label_5 | 0.10 | -9.51 | +91 (effectively unreachable) |
| label_10 | 0.30 | -15.15 | +51 (very hard) |
| label_20 | 0.12 | -7.39 | +63 (very hard) |
| label_40 | **3.33** | -36.94 | **+11.1** (achievable) |
| label_60 | **9.97** | -58.76 | **+5.9** (already there) |

**Headline findings**:
1. **h_60 has the highest leverage by a wide margin** (slope ≈ 10). Each +1 local h_60 PnL → +10 platform. iter_002 already crossed breakeven at h_60 (local +6.30 → platform +4.07).
2. **Short horizons (h_5/10/20) are a trap**: very low slope (0.1–0.3). Even "great" local h_10 LOSO of +30 would only give platform 0. They mostly track noise around -8 to -10. **The +21.86 LOSO h_10 of iter_002 mapped to -8.64 on platform** — a 30-point gap.
3. **iter_003 (5-seed h_10 ensemble) predicted platform h_10 = -8.53 [-11.87, -5.19]**. Since iter_003 inherits iter_002's h_40/h_60, the platform **max-of-5 ≈ +4.07 (h_60)** — identical to iter_002 within noise. **iter_003 will not improve over iter_002**.
4. **Going forward**: focus on improving h_60 alpha. Every +1 local h_60 LOSO → +10 platform. Stop chasing h_10.

---

## 1. Setup

### 1.1 Test set
240 sessions = 5 sym × 24 dates (96–119) × 2 (am/pm).

Per-(sym, date, session, t) OOF predictions reused (no retraining):
- `iter_002` (Scheme C 226d single seed): T5b/loso_pred_h{5,10,20,40,60}_held{0..4}.parquet — 5 horizons × 5 LOSO folds = 25 files
- `iter_003` (Scheme C 5-seed ensemble): T11/loso_pred_ensemble_h10_held{0..4}.parquet — h_10 only, 5 folds
- `mmpc_demo`: per-session cum_pnl from `experiments/audit_pipeline_per_session.json`

Threshold parameters (asymmetric "side_max ≥ T and side_max > p1+δ → trade") taken from T5b sweep:
| h | T | δ |
|---|---|---|
| 5 | 0.60 | 0.10 |
| 10 | 0.55 | 0.10 |
| 20 | 0.50 | 0.05 |
| 40 | 0.50 | 0.00 |
| 60 | 0.50 | 0.20 |

### 1.2 Sanity checks (Scheme A LOSO sums reproduced)

| model | h | local sum | reference |
|---|---|---|---|
| iter_002 | 5 | +17.492 | T5b: +17.492 ✓ |
| iter_002 | 10 | **+21.860** | T5b: +21.860 ✓ |
| iter_002 | 20 | +19.700 | T5b: +19.700 ✓ |
| iter_002 | 40 | +11.705 | T5b: +11.705 ✓ |
| iter_002 | 60 | +6.300 | T5b: +6.300 ✓ |
| iter_003 | 10 | **+22.224** | T11: +22.224 ✓ |
| mmpc_demo | 5/10/20/40/60 | +4.09/+6.15/+6.33/+6.27/+3.57 | audit ✓ |

> **Correction to task brief**: brief stated "mmpc_demo Scheme A LOSO label_60 = -22.10". That number is from `T4_loso_validate/loso_results_schemeA.json` — a **trained LightGBM** baseline, not mmpc_demo. mmpc_demo's actual local label_60 sum is **+3.573** (240 sessions). Platform truth (-23.13) is correct.

---

## 2. Five CV Schemes (iter_002 h_10 detail)

All schemes operate on per-(sym, date, session) cum_pnl after applying per-horizon thresholds. See `scheme_results.json` for the full output across all (model, horizon).

### Scheme A — Standard 5-fold LOSO

| held sym | cum_pnl |
|---|---|
| 0 | +1.94 |
| 1 | +5.13 |
| 2 | +4.35 |
| 3 | +2.11 |
| 4 | **+8.33** |

sum = +21.86, std across folds = 2.61. **Sym 4 contributes 38% of total!**

### Scheme B — LO2SO (10-pair regrouping)

10 pair sums, mean = +8.74, std = 3.01, range [+4.06, +13.45].
Pairs containing sym 4 always > mean; pair (0,3) is weakest (+4.06).
*Caveat*: regrouped from existing 5-fold OOF, not retrained holding 2 syms.

### Scheme C — Time-rolling within test region (3 windows of 8 dates)

| window | dates | sum |
|---|---|---|
| 0 | 96–103 | +9.10 |
| 1 | 104–111 | +7.51 |
| 2 | 112–119 | +5.25 |

**Monotonic decay (-43% from window 0 to 2)** — model degrades on dates further from training. If platform's test region is yet further forward in time, edge erodes more.

### Scheme D — Cross sym × cross time (5×3 = 15 cells)

| | w0 | w1 | w2 |
|---|---|---|---|
| sym 0 | +1.18 | +0.42 | +0.35 |
| sym 1 | +2.28 | +1.76 | +1.09 |
| sym 2 | +0.94 | +1.82 | +1.58 |
| sym 3 | +0.77 | +0.66 | +0.68 |
| sym 4 | +3.93 | +2.84 | +1.56 |

Sym variance (row-sum std): 2.61. Time variance (col-sum std): 1.93.
Sym 4 dominates and decays sharpest. Sym 0 stays consistently lowest.

### Scheme E — Bootstrap CI (n=2000)

| level | observed | std | 95% CI |
|---|---|---|---|
| **session** (n=240) | +21.86 | 1.89 | [+18.30, +25.91] |
| **cluster-sym** (n=5) | +21.86 | **5.26** | **[+12.47, +32.22]** |
| cluster-date (n=24) | +21.86 | 2.33 | [+17.80, +26.65] |

Cluster-sym CI is 3× wider than session — captures the OOD-sym risk (which dominates platform shock for h_10).

---

## 3. Two-anchor linear calibration (CRITICAL UPDATE)

### 3.1 Anchors

| horizon | mmpc local | **mmpc plat** | iter_002 local | **iter_002 plat** |
|---|---|---|---|---|
| label_5 | +4.09 | -9.08 | +17.49 | -7.68 |
| label_10 | +6.15 | -13.32 | +21.86 | **-8.64** |
| label_20 | +6.33 | -6.65 | +19.70 | -5.09 |
| label_40 | +6.27 | -16.07 | +11.71 | **+2.02** |
| label_60 | +3.57 | -23.13 | +6.30 | **+4.07** |

### 3.2 Linear fit `platform = a × local + b` (exact, 2 points)

| horizon | a | b | platform @ local=0 | breakeven local | interpretation |
|---|---|---|---|---|---|
| label_5 | 0.105 | -9.51 | -9.51 | +91 | local barely matters; platform always near -9 |
| label_10 | 0.298 | -15.15 | -15.15 | +51 | weak slope, ceiling ~ -8 even at very high local |
| label_20 | 0.117 | -7.39 | -7.39 | +63 | similar to h_5 |
| label_40 | **3.33** | -36.94 | -36.94 | **+11.1** | steep — every +1 local → +3.3 platform |
| label_60 | **9.97** | -58.76 | -58.76 | **+5.9** | **massive leverage — every +1 local → +10 platform** |

### 3.3 Why are short and long horizons so different?

The slope `a` reveals **how much of the local PnL signal survives platform OOD shift**:
- **h_5/10/20 (small a)**: The short-horizon LightGBM is mostly fitting noise patterns specific to the local test draw (date 96-119 5-sym). On platform's broader/different test, those patterns don't generalize. Platform PnL is bounded near a *negative* constant ≈ -8 to -15.
- **h_40 (a=3.3)**: Some genuine alpha — model partially generalizes. iter_002 already positive on platform (+2.02 vs +11.7 local).
- **h_60 (a=10)**: **Strong amplification** — local-positive signal lands on platform almost free of cost. iter_002 +6.3 → +4.07. mmpc_demo +3.6 → -23 (random near-zero local destroys at h_60 because higher activation rate trades more noise).

The high a at h_60 is partly because **the two anchors span a small local range** (3.57 vs 6.30) yet a wide platform range (-23.13 vs +4.07). With 2 points, the slope is mathematically pinned but with high *uncertainty* about extrapolation. Caveats below.

### 3.4 iter_002 fit verification (must reproduce exact platform truth)

| h | predicted | truth | err |
|---|---|---|---|
| 5 | -7.68 | -7.68 | 0 |
| 10 | -8.64 | -8.64 | 0 |
| 20 | -5.09 | -5.09 | 0 |
| 40 | +2.02 | +2.02 | 0 |
| 60 | +4.07 | +4.07 | 0 |

Exact (degenerate fit through both anchor points).

---

## 4. iter_003 platform prediction (2-anchor fit)

**Only h_10 OOF available for iter_003** (the 5-seed ensemble was h_10-only). h_5/20/40/60 in iter_003 submission **reuse iter_002's models verbatim** (per submission/SUBMISSION_LOG.md).

### 4.1 h_10 prediction

| local A | slope a | intercept b | platform mean | session CI95 | cluster-sym CI95 |
|---|---|---|---|---|---|
| +22.22 | 0.298 | -15.15 | **-8.53** | [-9.67, -7.39] | [-11.87, -5.19] |

iter_003's h_10 LOSO improvement (+0.36 over iter_002) translates to **+0.11 platform** (since slope = 0.30). Within bootstrap noise.

### 4.2 Max-of-5 prediction

iter_003 submission's per-horizon platform predictions:
| h | source | predicted platform |
|---|---|---|
| 5 | iter_002 model | -7.68 |
| 10 | iter_003 5-seed ensemble | **-8.53** |
| 20 | iter_002 model | -5.09 |
| 40 | iter_002 model | +2.02 |
| 60 | iter_002 model | **+4.07** |

**max = +4.07 (h_60)** — *identical to iter_002 max*. **iter_003 will not improve over iter_002 on platform.**

This is a critical finding: **the 5-seed h_10 ensemble was wasted effort** — even if it perfectly improved h_10 LOSO by 1.0 (instead of 0.36), platform h_10 = 0.298 × (+22.86) - 15.15 = -8.34 vs baseline -8.64, still well below h_60's +4.07.

---

## 5. iter_002 detailed predictions per horizon

For completeness — the 1-anchor (mmpc-only) analysis vs 2-anchor truth. Big mistake the 1-anchor would have made:

| h | local A | 1-anchor predicted | 2-anchor truth | 1-anchor error |
|---|---|---|---|---|
| 5 | +17.49 | +4.32 | -7.68 | **-12.0** |
| 10 | +21.86 | +2.39 | -8.64 | **-11.0** |
| 20 | +19.70 | +6.72 | -5.09 | **-11.8** |
| 40 | +11.71 | -10.63 | +2.02 | **+12.7** |
| 60 | +6.30 | -20.40 | +4.07 | **+24.5** |

**The 1-anchor "constant additive gap" model failed catastrophically at long horizons** — predicted -20 for h_60, actual +4. Going forward, **1-anchor calibration with mmpc_demo alone is unreliable for trained models**. Always include at least one trained-model anchor.

---

## 6. Stress test (iter_002 h_10)

Add Gaussian noise σ to OOF probabilities (renormalize, re-threshold):

| σ | cum_pnl | n_active |
|---|---|---|
| 0.00 | +21.86 | 97k |
| 0.05 | +21.07 | 100k |
| 0.10 | +18.56 | 109k |
| 0.20 | +9.44 | 136k |
| **0.30** | **-1.85** | **164k** |
| 0.50 | -17.15 | 195k |
| 1.00 | -32.22 | 217k |

iter_002 h_10 collapses fast: σ=0.3 already negative. The model has shallow alpha — a tiny perturbation broadens the high-confidence prob region, more "meh" trades enter, alpha vanishes. Consistent with the **2-anchor finding that platform h_10 is already in the "no real alpha" zone (a=0.3)**: most of LOSO h_10 PnL is local-test-specific overfitting.

**Implication**: future iter_004 should run this stress test as a sanity gate — only ship models whose h_10 stays positive at σ ≥ 0.2.

---

## 7. Recommendations for next iterations

### 7.1 Stop optimizing h_10 LOSO
- Slope a = 0.30. Each +1 LOSO h_10 → +0.30 platform. To beat iter_002's +4.07 platform max, would need LOSO h_10 → 0.30L - 15.15 > 4.07 → L > 64. Currently best is +22, would need 3× improvement. Not a realistic path.

### 7.2 Optimize h_60 LOSO instead
- Slope a = 9.97. Each +1 LOSO h_60 → +10 platform. iter_002's +6.30 → +4.07. Even +0.5 LOSO h_60 improvement → +5 platform.
- h_60 has a strict acceptance criterion: each ΔLOSO matters ~30× more than h_10.

### 7.3 Threshold gating to keep h_60 active and h_10 quiet
- The current iter_002 thresholds: T_60=0.50, δ_60=0.20; T_10=0.55, δ_10=0.10.
- Idea: tighten short-horizon thresholds further (T=0.70+ for h_5/10/20) to make those horizons predict mostly 1 (no trade), reducing platform losses to ~0 instead of -8.
- Caveat: platform takes max, so a noisy short-horizon could randomly outperform h_60 on a different test draw. But variance bounds the upside (best case h_10 ≈ -5).

### 7.4 Calibration plan
- Submit one more model targeting h_60 specifically (e.g., features specifically engineered for long-horizon mean reversion). Use iter_004 slot.
- Once we have a 3rd anchor with high local h_60, we can validate the slope-9.97 extrapolation and detect any non-linearity.

### 7.5 What CV scheme to use going forward
- **Scheme A (5-fold LOSO) sum is still useful as the headline number** (it's how we generated all the OOF data).
- **Scheme E cluster-sym bootstrap** is the best CI for "what range of local PnL would I see on a different sym mix" — use as the lower bound.
- **Scheme C (time-rolling)** — must run before submitting. Monotonic decay in time windows = the model is brittle to time drift; expect platform under-performance.
- **2-anchor linear extrapolation** — refresh this calibration every time a new platform result arrives. By iter_005 we should have 3+ anchors.

---

## 8. Hard limits / honest caveats

1. **2 anchors per horizon → exact linear fit, no residual to estimate slope uncertainty**. The slope-9.97 at h_60 is pinned by 2 points (mmpc local +3.57 and iter_002 local +6.30) — only 2.7 wide. Any new model with local h_60 > +6.30 ventures into extrapolation. The true relationship may not be linear.
2. **Cluster-sym CI is the only honest CI**, but it's a within-anchor variability — does NOT include calibration-line uncertainty.
3. **All 240 local sessions are date 96-119 5-sym**. Platform's test set is unknown — if it's a different date range or includes OOD syms, the linear fit could mis-extrapolate.
4. **iter_003 prediction (-8.53 h_10) assumes the slope a=0.298 is correct**. If iter_003 is a different "kind" of model and slope differs, prediction is wrong.

---

## 9. Files

- `cv_schemes.py` — 5 CV schemes + 1-anchor + 2-anchor calibration + max-over-horizons + stress test
- `per_cell_pnl.parquet` — long-form (model, horizon, sym, date, session) cum_pnl table (2,640 rows)
- `scheme_results.json` — per-(model, horizon) Schemes A/B/C/D/E summaries
- `calibration.json` — 1-anchor + 2-anchor calibrations + predictions
- `results.json` — final task summary
- `run.log` — stdout from main run

---

## 10. Headline numbers (for quoting)

- **iter_002 platform = +4.07 (h_60)** — already known from task 2435
- **iter_003 predicted platform max = +4.07 (h_60)** — will not improve
- **Future iter_004 target**: local h_60 LOSO ≥ +7 → predicted platform ≥ +11
- **Slope per horizon (each +1 LOSO → +X platform)**: 0.10 / 0.30 / 0.12 / 3.33 / **9.97**
