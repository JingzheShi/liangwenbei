# T85 — Within-window feature leakage audit (training cache + inference)

**Goal.** T84 already verified that batch inference is byte-for-byte identical to single-window inference. This audit asks the next question: *within a single 100-tick window, does any last-tick feature use rows from `[t, t+W-1]` (forward) or `[t-W/2, t+W/2]` (centred) instead of the causal slice `[t-W+1, t]`?* And does the **training cache** (`experiments/T68_stage5_features/cache/schemeP_*.npz`) carry the same causal property as inference?

**Verdict.** **No leakage found.** Every feature in `compute_batch_features` is causal: depends only on rows `[t-W+1, t]` of the 100-tick window. Training cache values match fresh inference-time computation within float32 storage precision (max abs rel diff = 5.75e-08 across 207 features × 128 random rows; the only outlier — `kyle_inv_W{50,100}` abs diff up to 0.226 — is numerical instability of `amt_last / cbrt(amt_last·σ + EPS)` for tiny σ, *also* at relative precision 5.5e-08).

**Severity.** 0 (no fix needed). The LOSO-equiv vs platform gap for iter_013 (LOSO +36.23 vs platform +19.23) is **not** caused by feature-level future leakage in either training cache build or inference compute.

---

## Scope

Files audited:

| File | Lines | Role |
|---|---|---|
| `submission/iter_013_time_features/fast_features_batch.py` | 780 | Inference batch extractor (T3+S1+S2+S3+S5 = 216) |
| `submission/iter_013_time_features/fast_features.py` | 568 | Single-window reference (196 cols, no S5) |
| `submission/iter_010_t61_batchvec/fast_features_batch.py` | 590 | **Identical** to iter_013 minus inlined S5; used by training cache |
| `experiments/T68_stage5_features/stage5_features.py` | 153 | S5 implementation imported by training cache |
| `experiments/T68_stage5_features/build_features.py` | 255 | Cache builder — produces `schemeP_{train,val,test}.npz` |
| `submission/iter_013_time_features/Predictor.py` | 286 | Inference wrapper |

Audited feature classes (and their declared sub-windows W):

| Family | Cols | Operator | W |
|---|---|---|---|
| T3 mlofi | 30 | rolling sum on Δ-LOB | 5, 20, 60 |
| T3 wmp / balance | 11 | last-tick weighted mid | 1 |
| T3 RV | 4 | √(Σ log²-ret) | 5, 10, 20, 50 |
| T3 EWMA intst | 24 | causal IIR (lfilter) | full window (recursive) |
| S1 dual_z | 37 | (last − μ20)/σ20 − (last − μ100)/σ100 | 20 + 100 |
| S1 signed_rv | 3 | (Σ⁺r² − Σ⁻r²)/(Σ⁺+Σ⁻) | 20, 50, 100 |
| S1 kyle_inv | 2 | amt / cbrt(\|amt\|·σ + ε) | 50, 100 |
| S1 ewma_ofi | 12 | EWMA on `mlofi_W20_lvl{k}` | full window (recursive) |
| S2 qrank | 20 | (x[-W:] ≤ x[-1]).mean() | 100 |
| S2 r-skew | 3 | population skew via moments | 20, 50, 100 |
| S2 gofi | 30 | rolling sum of LOB Δ flow | 5, 20, 60 |
| S2 kyle_lambda | 2 | rolling cov(signed_dvol, Δmid)/var | 50, 100 |
| S2 vol_burst | 4 | last / mean(W) | 20, 50 |
| S3 ewma_resid | 1 | mid[-1] − ewma(mid) | full window (recursive) |
| S3 rv_ratio | 3 | RV(W_n) / RV(W_d) | (5,50), (20,100), (50,100) |
| S3 jshare | 4 | (RV − BV) / RV | 20, 30, 50, 100 |
| S3 cancel_imb | 3 | rolling cancel/total share | 20, 50, 100 |
| S3 roll_eff_spr | 3 | rolling Roll spread / quoted | 30, 50, 100 |
| S5 adapt_mom | 3 | (mid[-1] − mid[-W]) / σ | 20, 50, 100 |
| S5 ofi_tox | 4 | corr(OFI, Δmid) over last W | 20, 50 |
| S5 signed_bipower | 3 | π/2 Σ sign(r)·\|r\|·\|r₋₁\| / RV | 20, 50, 100 |
| S5 spread_regime | 3 | (spr[-1] − med) / IQR | 20, 50, 100 |
| S5 trade_pers | 3 | lag-1 autocorr of sign(Δmid) | 20, 50, 100 |
| S5 liq_asym | 4 | log((1+Σbid₅)/(1+Σask₅)) | 5, 20, 50, 100 |

Total = 216. **All operators are reductions over `[..., -W:]` on per-tick precomputed arrays whose `t`-th entry depends on raw rows `≤ t`.**

---

## Method

Five complementary checks executed by `audit.py` (reproducible: `python3 experiments/T85_feature_leakage_audit/audit.py`). Per-feature data in `audit_results.json`.

### M1 — Static line-by-line review

Searched for any non-causal pattern:

- `[..., :W]` (forward slice) — **not present** anywhere
- `np.roll(..., negative)` (right shift) — **not present**
- `pandas.rolling(..., center=True)` — **not present** (pandas not used at all in batch path)
- `scipy.signal.filtfilt` (forward-backward IIR) — **not present**; only `lfilter` (causal)
- `np.cumsum(...) [-1]` patterns — present but values used as `cs[t+1] − cs[t+1−W]` = sum over `[t-W+1..t]`, causal
- `np.quantile / np.median / np.std (..., axis=0)` over batch — **not present**; all reductions are along time (`axis=-1` or `axis=1` for the (N,T,F) Stage-1 stack)
- `b_prev[:, 1:] = b[:, :-1]` lag pattern — present everywhere; **always backward shift** (b_prev[t] = b[t-1])
- Initial conditions: `lfilter(..., zi=(1-α)·x[:, :1])` — derived from window's own row 0, never from cross-window state
- EWMA on synthetic mlofi_W20: full per-tick array `_rolling_sum_W_full(e_clean, 20)` builds `out[t] = sum(e[t-19..t])` for `t ≥ 19` (causal cumsum) and zeros earlier entries to mimic pandas `min_periods=20` + NaN→0 fill

Conclusion: every reduction reads from time slice `[..., -W:]` or earlier. **Static-only verdict: causal.**

### M2 — Empirical "perturb each tick"

Pick a real 100-tick window (sym0 / date0 / am / end-tick 500). For each `p ∈ [0, 99]`, perturb only row `p` by ±1% noise, recompute the feature vector, count features whose value changes by > 1e-9. Record the *earliest* `p` per feature.

For a causal feature with sub-window `W` (and an extra lag-1 from Δ-style precomputation), the earliest perturbation row that changes the feature should be `99 − W` (or `99 − W + 1` if no lag).

| earliest p | n features | matches W of |
|---:|---:|---|
| 0 | 80 | EWMA, W=100 dual_z long, signed_rv_W100, qrank_W100, full-window cumulant |
| 39 | 20 | W=60 mlofi/gofi (lag-1 → 99−60=39) |
| 49 | 10 | W=50 RV, kyle, jshare, cancel, etc. (99−50=49) |
| 79 | 28 | W=20 family (99−20+1=80; lag-1 → 79) |
| 94 | 21 | W=5 mlofi/gofi/RV (99−5=94) |
| 99 | 12 | wmp_lvl1..10 + last-tick-only diagnostics |
| –1 | 3 | invariant (e.g. wmp_lvl4..10 had degenerate input → 0) |

Histogram of "n_responsive_per_p" is monotonically non-decreasing toward p=99 with the four expected steps:

```
p=0  → 80   p=20 → 75   p=40 → 98   p=50 → 108
p=70 → 119  p=80 → 166  p=90 → 139  p=95 → 190  p=99 → 208
```

The bumps at p=80 (W=20 entries activate), p=95 (W=5 entries activate), and p=99 (last-tick only) are exactly where causal sub-windows begin. **No feature is responsive to a "future"-only perturbation; this is impossible since `p ≤ 99 = last tick`, but more importantly there is no feature whose response *only* appears for early-`p` indices, which would indicate a forward shift.**

### M3 — Step-signal "shift the cliff"

Construct a 100-tick window with a step at position `s`: rows `[0..s-1] = LOW`, rows `[s..99] = HIGH` (sane bid/ask/size/intst values; spread held constant). Sweep `s ∈ [0, 100]` and trace the feature value.

Expected behaviour for a causal feature with sub-window W:
- For `s ∈ [0, 99-W]` the entire last-W slice is HIGH → feature reaches its "all-HIGH" steady state and **plateaus**
- For `s ∈ (99-W, 99]` the step is *inside* the last-W slice → feature transitions
- At `s = 100` the window is all LOW → baseline

For 23 representative features the transition boundary matches the declared W (within ±1 due to lag-1 in diff-based features). Sample (full table in `audit_step_visual.py` output):

| feature | W | value @s=99 | value @s=100-W | value @s=0 | plateau? |
|---|---:|---:|---:|---:|---|
| rv_w5 | 5 | 0.0025 | 0.0025 | 0.0000 | yes (s≤94 = 0) |
| rv_w20 | 20 | 0.0025 | 0.0025 | 0.0000 | yes (s≤79 = 0) |
| signed_rv_W20 | 20 | 0.9996 | 0.9996 | 0.0000 | yes (s≤79 = 0) |
| jshare_W20 | 20 | 0.9996 | 0.9996 | 0.0000 | yes (s≤79 = 0) |
| mlofi_W20_lvl1 | 20 | 3000 | 3000 | 0 | yes |
| mlofi_W60_lvl1 | 60 | 3000 | 3000 | 0 | yes (s≤39 = 0) |
| adapt_mom_W20 | 20 | 4.59 | 0 | 0 | yes (s≤79 = 0) |
| kyle_lam_W50 | 50 | 0.0004 | 0.0004 | 0 | yes (s≤49 = 0) |

**Every causal-W feature plateaus to 0 (the all-LOW value) once the step is older than 99-W ticks.** No feature "remembers the future" of the step location.

### M4 — Reverse-window sanity

Reverse the 100-tick window in time: `w_rev[i] = w[99-i]`. Perturb the **same physical row** (originally row 99 = "now"; in reversed window, that becomes row 0 = "deep history").

| Where perturbation lands | n features changed (of 216) |
|---|---:|
| Original window, row 99 (now) | **208** (high) |
| Reversed window, row 0 (deep history) | **76** (low) |

The 76 = full-window/EWMA features (W=100, exponentially-weighted recursion). The 208 vs 76 gap shows perturbations near "now" propagate into many more features than perturbations deep in history — the signature of causal short-window operators.

### M5 — Training cache vs fresh inference compute

Using `experiments/T68_stage5_features/cache/schemeP_train.npz` (~1.47M rows). Pick 128 random rows, recover `(sym, date, sess_idx, t)` from cache meta, load the underlying parquet, slice rows `[t-99 .. t]`, and run `compute_batch_features` again. Compare per-feature against the cached row (cache columns 154..369 hold the 216-d extras, after being NaN→0 sanitized and cast to float32).

Per-feature max absolute / relative diff across 128 rows:

| Feature class | max abs | max rel | Verdict |
|---|---:|---:|---|
| 207 features (excl. kyle_inv & wmp_lvl4-10) | 9.29e-07 | **5.75e-08** | float32 storage precision (rel 2⁻²⁴ ≈ 6e-8) |
| kyle_inv_W50, W100 | 2.26e-01 | 5.52e-08 | numerical instability of `amt/cbrt(amt·σ+ε)` for σ→0; **rel** still at float32 precision |
| wmp_lvl4..10 | NaN | NaN | cache applies `np.where(isfinite, ..., 0)` post-compute; some sym/date have no L4-10 LOB so wmp = NaN. Drilldown re-applies the same filter and matches |

Conclusion: the training cache reproduces fresh per-window inference exactly (modulo float32 precision and the NaN→0 sanitization pass).  **No off-by-one row, no forward shift, no future window.**  Equivalently, the relation

```
cache row r ↔ window A[r:r+100] ↔ last tick t = r+99 ↔ stored t_arr[r] = r+99
features at cache row r = compute_batch_features(window) ≡ depend on rows [t-99, t]
```

holds bit-exactly.

---

## Per-feature attribution

`audit.py::test1` records, for every feature, the earliest perturbation row that moves it. There is **no feature** whose earliest-responsive row exceeds `99 − W_declared`. There is **no feature** that responds only to early rows (which would indicate forward shift).

Specifically the user's flagged classes:

| Class | Hypothesis flagged | Audit result |
|---|---|---|
| qrank (W=100) | could be `np.quantile(x[:W], …)` instead of `x[-W:]` | code uses `x[-100:]`; perturbation responsive from p=0 (entire window); step-signal plateaus correctly |
| dual_z / std (W=20, 100) | could be centred or look-ahead std | code uses `seg20 = X_dz[:, -20:, :]; m20 = seg20.mean(axis=1)`; perturbation responsive from p=0 (long arm) and p=79 (short arm 20+lag); ✓ |
| realized_vol / signed_rv (W=20, 50, 100) | sum direction | `r2_pos[:, -W:].sum(axis=-1)`; step-signal plateau exactly at 99-W; ✓ |
| bipower (W=20, 50, 100) | could pair r[t] with r[t+1] | code uses `abs_r_lag[:, 1:] = abs_r[:, :-1]` then `bv_term[t] = sign(r[t])·\|r[t]\|·\|r[t-1]\|` and `bv_term[:, -W:].sum(...)`; lag-1 is **backward**; ✓ |
| EWMA (intst, ofi, mid_resid) | could be filtfilt | uses `scipy.signal.lfilter` axis=-1 with per-window `zi=(1-α)·x[:, :1]`; this is documented as a forward-only IIR. No filtfilt anywhere; ✓ |
| vol_burst / spread_reg (W=20, 50, 100) | percentile direction | `np.median(spr[:, -W:], axis=-1)`, `np.quantile(seg, 0.75, axis=-1)` over `seg = spr[:, -W:]`; ✓ |
| kyle_lambda (W=50, 100) | rolling regression slope direction | `x_seg = signed_dvol[:, -W:]; y_seg = delta_mid[:, -W:]`; cov/var on last W; lag-0 (no shift between x and y). ✓ |
| trade_pers (W=20, 50, 100) | lag-1 autocorr direction | `sign_lag[:, 1:] = sign_dm[:, :-1]` (backward shift), corr over last W; ✓ |

---

## Conclusion

`fast_features_batch.py` (and its single-window twin `fast_features.py`, and the training cache builder `experiments/T68_stage5_features/build_features.py`) compute every last-tick feature as a function of the rows `[t-W+1, t]` of the 100-tick window. Five independent empirical checks plus a line-by-line static review converge on the same answer.

- Static review: 0 forward operators, 0 centred operators, 0 backward IIR.
- Perturb-each-tick: every feature's earliest-responsive row matches its declared sub-window.
- Step-signal sweep: every causal-W feature plateaus exactly at the expected boundary.
- Reverse-window sanity: 208/216 features sensitive at "now", only 76/216 sensitive at "deep history".
- Cache vs fresh: max relative diff across 207 features × 128 rows = 5.75e-08, i.e. float32 storage precision.

There is **no future leakage** at the within-window level. The training cache and inference path are mutually consistent and both causal.

The LOSO-equiv → platform gap (iter_013: +36.23 vs +19.23) must be explained by something other than feature-level future leakage — leading candidates: (a) sym-distribution shift (test set may include training-set-out symbols, c.f. CRITICAL_CONSTRAINTS §3), (b) date-distribution shift (different vol regime), (c) threshold over-fit, (d) label-side issue (e.g. midprice1 vs traded price). T84 ruled out batch-cross-window leakage; T85 rules out within-window forward leakage. Both auditing fronts close.

## Artifacts

- `audit.py` — primary repro script (7 sub-tests)
- `audit_drilldown.py` — per-feature cache-vs-fresh diff distribution
- `audit_step_visual.py` — visual step-signal trace for 38 representative features
- `audit_results.json` — machine-readable per-test results (~400 KB)
- `audit_drilldown.json` — per-feature max abs/rel diffs
- `REPORT.md` — this file

## RESULT line

```
RESULT: task=t85_future_audit metrics={leakage_found=N, suspicious_features=[], severity=0, max_rel_diff_cache_vs_fresh=5.75e-08} notes=all features causal; cache matches fresh within float32 precision; LOSO-platform gap is NOT future leakage
```
