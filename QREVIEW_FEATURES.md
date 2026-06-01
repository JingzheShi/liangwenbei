# QREVIEW — Feature Construction Code (final_submission_code)

Reviewer: critical, white-board, code-only.

Files in scope:
- `01_build_features/build_schemeP_cache.py`
- `01_build_features/fast_features_batch.py`
- `01_build_features/stage5_features.py`
- `04_build_pkg/fast_features.py`        (single-window reference)
- `04_build_pkg/fast_features_batch.py`  (= 01 version byte-identical, confirmed via `diff`)

Conventions in this doc:
- **(B)** = `fast_features_batch.py` (`01_build_features/`)
- **(S)** = `fast_features.py` single-window (`04_build_pkg/`)
- **(Stage5)** = `stage5_features.py`
- **(Cache)** = `build_schemeP_cache.py`

---

## §0. Macro / dimension / split sanity

**Q1.** (Cache:18) Header says "370-d = 154 + 196 + 20" but line 19 then says "fast_features_batch already integrates Stage5 (216 = 196 baseline + 20 Stage5)". Final concat is `raw_last + extras = 154 + 216 = 370`. Where in the code is it ever verified that `compute_batch_features` actually returns 216 (not 215, not 217)? There is no assert on the second dim before the npz is written.

**Q2.** (B:5, B:763, B:780) Header comment says "(N, 216): T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14) + Stage5(20)". 69+54+59+14+20 = 216 ✓. But: is anywhere checking column counts at runtime? If somebody bumps `T3_RV_WINDOWS` from 4 to 5 entries, the docstring stays "216" but every downstream index breaks silently.

**Q3.** (Cache:18 vs Predictor:68) README/docstring claims 370 features. Predictor drops 11 named features (`FAIL_NAMES`), leaving 154 + (216-11) = 359. Where is `359` derived from `370 - 11`? Why exactly 11 — by which statistical criterion ("KS-fail" is mentioned in fast_features.py:10 docstring but nowhere defined)? Is the threshold 0.05? 0.1? Bonferroni-adjusted? Done on which val/test split? Random seed? On which horizon's labels?

**Q4.** (Predictor:68-75) `FAIL_NAMES` contains: `dualz_ask_diff1`, `dualz_bid_diff5`, `dualz_ask_diff5`, four `qrank_W100_spread*`+`cumspread`, `kyle_lam_W50/W100`, `roll_eff_spr_ratio_W100`, `liq_asym_top5_W5`. Why was `dualz_bid_diff1` *kept* but `dualz_ask_diff1` dropped — what asymmetry causes only the ask side to fail KS? Mistake?

**Q5.** (Predictor:68-75) `liq_asym_top5_W5` is dropped but `W20/50/100` are kept. Is W=5 just too noisy, or is it actually anti-correlated? Was the rejection based on a single CV fold?

**Q6.** (Predictor:68-75) All three `kyle_lam_W` features get dropped — meaning Stage2's "kyle_lam" family contributes literally zero after KS. Then why is it computed at all? Pure waste of compute on every inference call.

**Q7.** (Predictor:68-75) Same for `qrank_W100_spread1/5/10/cumspread` — 4 of the 20 qrank features dropped, but `qrank_W100_spread1` (the most basic micro spread rank) is one of them. Is this evidence that qrank quantization on spread is degenerate (most ticks have same min-spread = "1 tick"), making rank ≡ rank of last_val noise?

**Q8.** (Cache:73-84) Split is hard-coded date 0-79 train / 80-95 val / 96-119 test. Why this exact 80/16/24 split? Why not 80/20/20 or rolling? Is the eval platform's blind test in date range 96-119? If yes, "test" here is in-distribution; if no, val=80-95 is the only true held-out — and 16 days = ~30 sessions of (5 sym × 2 sess × 16) is very thin for hyperparameter selection.

**Q9.** (Cache:81-83) Splits are date-based but training will see all sym×date×sess concatenated. For LightGBM/NN ensembles, does the training pipeline shuffle rows from different sym/dates together? If so, are dates inside a session treated as i.i.d., violating temporal block structure?

**Q10.** (Cache:88-90) `valid_lo = WINDOW - 1 = 99` and `valid_hi = T - 1 - MAX_HORIZON = 2000 - 60 = 1940`. So per session you keep rows 99..1940 = 1842 examples; rows 0..98 and 1941..2000 are dropped. For h=5 predictions you discard 55 perfectly-usable trailing rows (rows 1941..1995 have valid labels for h=5). Why throw away ~50 rows per session × 1200 sessions = ~60k samples?

**Q11.** (Cache:117-122) Returns `t = np.arange(valid_lo, valid_hi+1, dtype=np.int16)` — but `t` is then never used for any model feature (just emitted as identifier). Confirm: this is purely for traceability / debugging, not for any leakage?

**Q12.** (Cache:46-47) `HORIZONS = (5, 10, 20, 40, 60)`. Why these specific horizons? Why is there a gap (no h=15, 25, 30)? Why h=40 and not h=30?

**Q13.** (Cache:131) `out[f"mp_t{h}"] = mid_arr[rows + h]` — uses `df["midprice1"]` not the windowed mid. Confirm midprice1 has no NaN / always > 0 in raw parquet, else mp_t{h} becomes invalid label-side metric.

**Q14.** (Cache:165) `sess_idx` stored as `int8`, value 0 or 1. Never used by Predictor — purely meta. Is the model implicitly relying on sess_idx during training (e.g. as a feature)? If yes, that would violate **sym-agnostic + date-agnostic** — sess_idx is technically per-session metadata that may correlate with sym population.

**Q15.** (Cache:198-238) The CLI flag `--which train,val,test` — what happens when only train is built? Predictor's threshold tuning would silently use stale val cache. No version stamp on npz.

---

## §1. RAW_COLS (154-dim raw block) — semantics & ordering

**Q16.** (Cache:53-69) `RAW_COLS` has 154 entries. The full list comes hard-coded — but `assert len(RAW_COLS) == 154` only catches *length*, not *ordering*. If the raw parquet schema changes column names (e.g. `bsize1` → `bid_size_1`), there is no schema validation here.

**Q17.** (Cache:107) `raw_last = X3d[:, -1, :].astype(np.float32)` — the entire 154-column raw row at t=last is dumped into the feature vector. Is every column meaningful as a raw feature? Specifically:
   - `bid1..bid10` (prices) at one tick are tiny, noisy and only meaningful relative to mid — using raw bid is essentially leaking absolute price.
   - `midprice1..midprice10` are 10 redundant copies of mid level.
   - `bid_diff{k}`, `ask_diff{k}`, `bid_rate{k}`, `ask_rate{k}` — are these derived columns precomputed by the upstream data team? What is their definition? Without their formula we can't audit them.

**Q18.** (Cache:108-110) Special-cases only `amount_delta` → `sign(v) * log1p(|v|)`. But `volume_delta` is in the same units family — why not log-transform it? Same applies to `totalbsize`, `totalasize`, `bsize{k}`, `asize{k}` which span orders of magnitude. Inconsistent treatment.

**Q19.** (Cache:109-110) `np.log1p(|v|)` — uses log base e. Why not log10 (more interpretable bucket)? Why log1p (assumes small values matter near 0) when amount_delta is presumably non-negative in absolute value and >> 1?

**Q20.** (Cache:108-110) Is this transform applied to `raw_last` *only* (one column out of 154)? What about the 100-tick window used by all derived features — `amount_delta` *inside* X3d is **not** log-transformed (line 311 in B uses raw amt for kyle_inv, signed_dvol). So features are computed on raw scale, but the column dumped to raw_last is log-transformed. Inconsistency → kyle_inv etc. see raw amount, predictor sees log(amount).

**Q21.** (Cache:104-105) `extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)`. Cast to float32 silently. Any extreme finite values (e.g. 1e30) survive the finite check but become inf in float32. Is there pre-clipping before downcast?

**Q22.** (Cache:99-101) `col_idx["midprice"] = col_idx["midprice1"]` — aliasing midprice1 as "midprice". This means QRANK's `"midprice"` entry (B:357-364) and Stage1's signed_rv (B:294) and Stage3's mid use midprice1, not midprice (i.e. not WMP, not vwap). Why pick the level-1 mid over midprice5/midprice10 (less noisy)?

**Q23.** (Cache:55-56) Columns `open, high, low, close` are included in RAW_COLS — these are bar-aggregated stats. At 3-second tick they are basically equal to mid most of the time. Are they really worth a feature slot each? Or pure leak/noise?

**Q24.** (Cache:55-69) `RAW_COLS` order has been manually curated to put price-like, size-like, intensity-like, midprice/spread/diff, rate columns each in their own block. But the model treats this as a flat 154-dim vector. Was column order verified to match the cache produced before training, every time models are re-fit? Any chance the column order drifted between runs?

**Q25.** (Cache) `lb_ind / la_ind / mb_ind / ma_ind / cb_ind / ca_ind` (6 columns, the `*_ind` variants) — these are kept as raw features but **never** used in any derived feature. Are they binary indicators? Continuous? What information do they carry beyond their `*_intst`/`*_acc` siblings? Why include them in raw if they're not signal-rich?

---

## §2. T3 no-time block (69 dims) — MLOFI + WMP + RV + EWMA intensities

### MLOFI

**Q26.** (B:159-174) MLOFI per-level definition:
```
e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * asz + ind_a_up * as_prev
```
This deviates from Cont/Kukanov/Stoikov (2014) standard OFI. Standard form:
```
e = (bid_up * bs - bid_dn * bs_prev - bid_eq * (bs - bs_prev))_BID
  - (ask_up * as_prev - ask_dn * asz - ask_eq * (asz - as_prev))_ASK
```
Notice missing `bid_eq` and `ask_eq` terms here. **Bug or intentional**? See also Stage2's `gofi` (B:411-418) which DOES include `eq` terms — internal inconsistency.

**Q27.** (B:169-172) `ind_b_up = (b >= b_prev).astype(float64)` and `ind_b_dn = (b <= b_prev).astype(float64)`. When `b == b_prev` BOTH indicators fire → contribution becomes `bs - bs_prev`. That matches the "eq" case for bid. But on the ask side: `ind_a_dn = (a <= a_prev)` and `ind_a_up = (a >= a_prev)` — when equal, `-asz + as_prev = -(asz - as_prev)`. So the equality case for ask gives `as_prev - asz`. Sign of the "equal" branch:
   - bid side: `bs - bs_prev` (positive when size grows)
   - ask side: `as_prev - asz` (negative when ask size grows)
   These have **opposite sign conventions** for the same "Δsize" event. Is that the OFI definition you wanted?

**Q28.** (B:165-168) `b_prev[:, 0] = np.nan`. Why NaN sentinel for t=0 (then later cleaned) instead of just using `b[:, 0]` (treat the first tick as no-change)? The Stage5 OFI (B:641-644) uses the latter convention (`b_prev[:, 0] = b[:, 0]`). **Inconsistent between MLOFI and Stage5 OFI within the same file.**

**Q29.** (B:165-174) Once `b_prev` has NaN at t=0, *every* comparison `b >= b_prev` returns False at t=0 (since NaN compares False in NumPy). So `e[:, 0]` becomes `0*bs - 0*nan - 0*asz + 0*nan = NaN - NaN`. Trace through: `0 * NaN = NaN` in float64. So `e_per_lvl[k][:, 0]` is always NaN. Then `_rolling_sum_W_full(e_clean, 20)` (B:196) uses `e_clean = np.where(np.isnan(e), 0.0, e)` → OK. But then line 198 zeros out `rolled_full[:, :20] = 0.0` — wait, that means **the first 20 ticks of the rolling-mlofi are always 0**, even though for t=19 you'd normally have a 20-tick window from t=0..19 (all valid except for t=0). Why throw away another full row? Looks like a kludge to "match pandas behavior" but is the pandas behavior the right behavior?

**Q30.** (B:177-184) `T3_MLOFI_WINDOWS = (5, 20, 60)`. Why exactly 5, 20, 60? Why not 10, 30, 100? Is the 60-tick (= 3 min) MLOFI signal documented somewhere as the empirical sweet spot? Did anyone do a grid sweep?

**Q31.** (B:177-184) Output is 30 features (3 windows × 10 levels). The level-10 MLOFI (deepest book) is unlikely to carry actionable info on most stocks — has anyone checked that all 10 levels are non-degenerate (bid10 ≠ ask10 ≠ flat for >90% of ticks)?

**Q32.** (B:182) `window = np.where(np.isnan(window), 0.0, window)` inside the sum loop runs for **every** (W, k) pair: 3 × 10 = 30 redundant NaN-cleaning passes on overlapping data. Inefficient.

**Q33.** (B:177-184) For W=5, `window = e[:, -5:]` — does NOT include t=0 (since T=100, -5: → indices 95..99). So the NaN-at-t=0 issue is irrelevant for W=5,20,60. Then the `np.where(np.isnan(window), 0.0, window)` line (B:182) is dead code for these windows. **Comment+code disagrees.**

**Q34.** (B:193-199) Only levels 1, 5, 10 of `mlofi_W20` are kept in `derived` for EWMA-OFI downstream. Why these three (out of 10)? Why not 1,3,5? Why W=20 specifically?

### WMP

**Q35.** (B:202-216) WMP = `(a * bs + b * asz) / (bs + asz)`. **This is bid/ask sizes swapped relative to the Avellaneda/Stoikov micro-price definition.** Standard micro-price weights are `(bid*ask_size + ask*bid_size) / (bid_size + ask_size)` — the **bid price weighted by ask size** (because more sellers → price drifts up to ask). Here the code has it reversed. **Likely sign bug**: positive mid-skew imbalance flips meaning.

Re-checking: `wmp_normal = (a * bs + b * asz) / (bs + asz)`. When ask_size >> bid_size, denominator dominated by `bs`, numerator dominated by `b * asz`, so wmp ≈ b (the bid). But intuitively when ask_size >> bid_size, there's excess supply, price should move toward bid = correct direction. Hmm — actually this matches Avellaneda definition. Let me re-derive: micro = (B·Sa + A·Sb)/(Sa+Sb) where bid_size pulls toward ask and ask_size pulls toward bid. Yes the formula here is **correct**. But still worth flagging for explicit comment.

**Q36.** (B:208-211) `denom = bs + asz`; `wmp_fallback = (a + b) / 2.0` when `denom == 0`. Are zero-size top-of-book situations frequent enough to need this fallback (suggesting LOB gaps), or so rare it never fires (dead code)?

**Q37.** (B:213-214) Only `wmp[:, -1]` (last tick) is emitted per level — 10 features. Then balance_12 = `wmp_lvl1[-1] - wmp_lvl2[-1]`. Why just 1 vs 2 difference? Why not also 1 vs 5, 1 vs 10? Why a single "balance_12" feature?

**Q38.** (B:217) `derived["wmp_lvl1"] = wmp_per_level[0]` — full N×T array kept, **only** used for RV computation (lines 220-230). Could have been replaced by direct mid-price-based RV; using WMP-based RV is an unusual choice. Why?

### RV from WMP

**Q39.** (B:220-230) `safe = wmp + 1.0`, then `np.where(safe > 0, safe, np.nan)`. Why `+ 1.0`? WMP is a price ~ tens-of-yuan; adding 1 is meaningless absolute-scale shift. Is this a relic of code where WMP could be a negative return? **Suspicious magic constant.**

**Q40.** (B:222-223) Log return computed as `log(safe[t] / safe[t-1])`. With `safe = wmp + 1`, this is `log((wmp_t + 1)/(wmp_{t-1} + 1))`. Approximately equals `(wmp_t - wmp_{t-1}) / (wmp + 1)` for small returns — i.e. price-shift normalized by `wmp + 1`. Why `+1`? Why not just `log(wmp_t / wmp_{t-1})` ?

**Q41.** (B:226-230) `T3_RV_WINDOWS = (5, 10, 20, 50)`. Note: 10 is here but `T3_MLOFI_WINDOWS` is (5,20,60). Different sets of windows for similar concepts → suggests no unified design. Why these 4 windows for RV?

**Q42.** (B:227-229) `s = sq.sum(); s = max(s, 0)` — applies a clip because float sum of squares could be slightly negative due to fp error. Reasonable. But then `sqrt(s)` is taken — **this is √RV (realized vol), not RV**. Half the literature uses RV (variance), half uses √RV. Which convention is downstream code expecting? Inconsistent with Stage3's `rv_W[W] = r2.sum()` (B:514) which uses RV (variance, no sqrt).

### EWMA intensities

**Q43.** (B:235-241) `T3_EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)`. EWMA half-life: `t_{1/2} = ln(2)/ln(1/(1-α))`. So α=0.05 → HL ≈ 13.5 ticks; α=0.1 → HL ≈ 6.6; α=0.3 → HL ≈ 1.9; α=0.5 → HL ≈ 1.0. Are α=0.5 and α=0.3 different enough to add information? Have they been redundancy-checked?

**Q44.** (B:235-241) 6 intensity columns × 4 alphas = 24 features just here. Together with EWMA-OFI (12 in Stage1) it's 36 EWMA features. Are the intensity-EWMAs actually distinct from a constant + small noise? Intensities (`lb_intst` etc.) are typically smooth Poisson rates — heavily auto-correlated. The α=0.5 EWMA is basically `x[t]` itself.

**Q45.** (B:82-100) `_ewma_last_batch` uses scipy.signal.lfilter with `zi = (1 - α) * x[:, :1]`. This implements `y_0 = α*x_0 + (1-α)*x_init`, where `x_init = x_0` due to the zi choice. That yields `y_0 = x_0`, matching pandas `ewm(adjust=False).mean()`. OK. But: zi must be `(1-α)*y_init` for filter state; calling it `(1-α)*x_0` makes the implicit assumption `y_init == x_0`. Confirmed correct? Did anyone unit-test this against pandas?

**Q46.** (B:235-238) `x_safe = np.where(np.isfinite(x), x, 0.0)`. NaN→0 fill **inside** the input to EWMA. This is **wrong** for genuinely missing data — it tells EWMA "the intensity was 0", which biases the smoothed value downward whenever data is missing. Should be NaN-skipping EWMA (forward-fill). Acceptable if intensity columns never have NaN (claim made in docstring B:226-227 of S file) — but does that hold for *all* parquets?

**Q47.** (B:235-238) `T3_INTST_COLS` order is `("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")` — limit/market/cancel for bid then ask, alphabetically grouped by side: limit-bid, limit-ask, market-bid, market-ask, cancel-bid, cancel-ask. Why this order specifically? It affects feature index → if model.feat_mean was trained with a different order (e.g. all bids then all asks), index mismatch breaks predictions silently because vector dimensions agree.

---

## §3. Stage 1 (54 dims): dualz + signed_rv + kyle_inv + ewma_ofi

### dualz (37)

**Q48.** (B:276-291, B:35-46) `DUAL_Z_COLS` has 37 entries. Why exactly these 37? Curated by what process? Why include `bid1, ask1, bid_mean, ask_mean` (price levels — absolutely-scaled), `midprice1, midprice2, midprice5, midprice10` (4 different mid levels — highly correlated), `cumspread` AND `spread1, spread5, spread10` (cumspread is essentially derivable from spread1..10)?

**Q49.** (B:286-289) Dual-z: `z_s = (last - m20)/s20`, `z_l = (last - m100)/s100`, `dual = z_s - z_l`. The dual-z is a "regime change" detector. But for cols like `bid1` (a level in absolute price), the standardization windows are 20-tick and 100-tick segments — both tiny relative to intra-day price drift. The denominator `s20` (20-tick std of `bid1`) could legitimately be 0 if bid1 hasn't moved. Then `(last - m20)/EPS ≈ huge number`, then `dual = huge_z_s - huge_z_l`. Are extreme values clipped? Line B:289 `np.where(np.isfinite(dual), dual, 0.0)` — but huge finite numbers ≠ inf, so they survive.

**Q50.** (B:286, EPS = 1e-8) `EPS = 1e-8` added to std denominator. But many of these columns (bid sizes, intensities) have raw magnitudes 10^3..10^6. A 1e-8 epsilon is **8 orders of magnitude smaller** than typical std → essentially zero protection. If std actually IS small, the resulting z explodes. Should use a *fractional* epsilon (e.g. `EPS * |last|`).

**Q51.** (B:280-285) `seg20 = X_dz[:, -20:, :]`; `m20 = seg20.mean(axis=1)`. Population (ddof=0) std. Fine. But for a `spread1` col that's constant at 1 tick for 95% of ticks, `s20 ≈ 0` virtually always. So dual-z on spread1 is essentially a "did spread move?" binary signal — was that the intent?

**Q52.** (B:289) `dual = np.where(np.isfinite(dual), dual, 0.0)` — masks inf/NaN but does **not** clip large finite values. A z-score of 1e6 (from near-zero denom) goes straight into the model. LightGBM doesn't care (tree split), but the NN does — feeds into standardization which uses train-time `feat_mean/feat_std` (Predictor:158). At inference an unseen huge z passes through and becomes an extreme normalized input. Test for this in val data?

**Q53.** (B:43-44) `DUAL_Z_COLS` includes `lb_acc, la_acc, mb_acc, ma_acc, cb_acc, ca_acc` (6 accumulator cols). Accumulators are monotonic (cumulative sums) → std over a 20-tick window is just the magnitude of recent increment, std over 100 ticks is roughly 5x larger → `z_s - z_l` is basically a recency indicator. Was this intended, or accidental?

**Q54.** (B:35-46) Why 20 and 100 specifically as the short/long? Why not 50 vs 200? Did anyone optimize this?

### signed_rv (3)

**Q55.** (B:293-308) Uses mid from `derived["midprice"]` = midprice1. Computes signed RV as `(rv_pos - rv_neg) / (rv_pos + rv_neg + EPS)`. This is the "RV-skew" or "signed-RV ratio". 
   - Bounded in [-1, 1] by construction → OK.
   - But: when both rv_pos and rv_neg are tiny (flat regime), `EPS` saves you from division by 0 but result becomes `(rp-rn)/EPS` which can be 0 (if rp=rn=0) or arbitrarily large if rp,rn ≠ 0 but ≈ EPS. Numerator can dominate denominator. **Possible explosion**.

**Q56.** (B:304-308) `SIGNED_RV_WINDOWS = (20, 50, 100)`. So same 100-tick base as the cache window. The W=100 variant uses the *entire* window — no point in calling it "rolling". Why not (10, 30, 60) for shorter horizons?

**Q57.** (B:300-303) `r_pos_mask = (r > 0).astype(float64)`. Returns 0 when `r == 0` — i.e. ticks with zero return contribute to neither pos nor neg, but they are in the denominator? No actually they contribute 0 to both. OK, the ratio still makes sense, but lots of zero-return ticks (likely common in 3s data with limit-up regimes) deflate both terms.

### kyle_inv (2)

**Q58.** (B:310-321) Kyle's inverse defined as:
```
sigma = std(r[-W:]) + EPS
activity = |amt_last| * sigma + EPS
a13 = cbrt(activity)
v = amt_last / a13
```
This is **not** Kyle's lambda. Kyle's λ = covariance(price_change, signed_order_flow) / variance(signed_order_flow). What's actually computed is "amount_delta / (|amount| × σ)^(1/3)" — closer to a normalized return per √vol per √dollar trade. **What is the theoretical motivation?**

**Q59.** (B:316) `a13 = np.cbrt(activity)`. Cube root specifically? Why not square root (square-root impact model)? Cube root corresponds to Almgren's impact model, but does it apply to 3-second tick data with this specific data generator?

**Q60.** (B:312-314) `amt_last = amt[:, -1]`; `abs_amt_last = abs(amt_last)`. Both stripped from same value. So `v = amt_last / (|amt_last| * σ)^(1/3)`. Algebraically this simplifies to `sign(amt_last) * |amt_last|^(2/3) / σ^(1/3)`. Why express it the convoluted way?

**Q61.** (B:314-320) `KYLE_WINDOWS = (50, 100)` — sigma computed over 50 or 100 ticks of the **same r** for both features. Two scales picked, but result depends only on `r[-50:]` vs `r[-100:]` std → only sigma differs. The result is `amt_last/cbrt(|amt_last|*σ)`. Are these two features just correlated 0.95+? If so, drop one.

**Q62.** (B:319) `v = np.where(np.isfinite(v), v, 0.0)`. No clip. If `amt_last = 1e10` (rare big trade), `v ≈ 1e10 / cbrt(1e10 * σ)` — still huge. Why no clip here when other features get clipped to [-10, 10]?

### ewma_ofi (12)

**Q63.** (B:323-328) `EWMA_OFI_LEVELS = (1, 5, 10)`, `EWMA_OFI_ALPHAS = (0.05, 0.1, 0.3, 0.5)`. So 12 features from EWMA-of-rolling-mlofi. But `derived["mlofi_W20_lvl1"]` has first 20 ticks forced to 0 (Q29). So EWMA of "0,0,...,0, real, real, ..., real" — the leading zeros bias the smoothed value low for the first ~50 ticks. Last-tick value is dominated by recent inputs (especially for α=0.5), so probably fine — but for α=0.05 (HL=13), the influence of early zeros is non-trivial.

**Q64.** (B:325) EWMA of `mlofi_W20`: this is "smoothed rolling MLOFI". Information bandwidth: `α=0.5` on a series with W=20 inner aggregation already has effective horizon ≈ 20-30 ticks (60-90s). What is α=0.05 trying to capture (HL ≈ 13 ticks) on top of an already 20-tick-averaged signal? Smoothing the smoothed.

---

## §4. Stage 2 (59 dims): qrank + rskew + gofi + kyle_lam + vol_burst

### qrank (20)

**Q65.** (B:355-372, B:55-62) `QRANK_COLS` (20 columns). Why these 20? Mostly different from `DUAL_Z_COLS` — e.g. duals have `bid1,ask1,bid_mean,ask_mean,midprice2,5,10`, qrank has none. Conversely, qrank has `lb_acc..ma_acc` but only 4 of them (no `cb_acc,ca_acc`). Why?

**Q66.** (B:368) `x32 = x.astype(np.float32, copy=False)`. Cast to float32 specifically for "match reference dtype". For a 100-tick window of LOB sizes (1e3..1e6), float32 has 7 significant digits — enough for rank comparison. But the comment "match reference dtype" suggests this is a backwards-compatibility patch — i.e. earlier versions used float32 here. If consistency with old training matters, OK. But the rest of B uses float64; why the one-off downcast?

**Q67.** (B:369-372) `out = (seg <= last_val).mean(axis=-1)`. So `qrank ∈ [1/W, 1]` (always at least 1 because last_val ≤ last_val). The min is 1/100 = 0.01, not 0 — slight bias. Tied values inflate rank (every tick with the same value scores 1.0). For spread1 which is mostly 1-tick, **qrank ≈ 1.0 always** — adding zero info. (Confirms why `qrank_W100_spread1` is in `FAIL_NAMES`.)

**Q68.** (B:359-365) For `"midprice"`, looks up `derived["midprice"]`. But the qrank loop iterates `QRANK_COLS` in order — if `"midprice"` appears mid-list (it's last in QRANK_COLS), the order-by-name in `qr_idx` (B:357-361) builds a sentinel `-1` then later iterates again checking `c == "midprice"`. Why a sentinel? Why not just `x = derived["midprice"] if c == "midprice" else X3d[:, :, col_idx[c]]` directly?

**Q69.** (B:365-371) Only the last value's rank is computed — i.e. "how unusual is right now compared to last 100 ticks". Why not also include earlier-window ranks (e.g. rank of mean of last 20 ticks vs last 100)?

### rskew (3)

**Q70.** (B:380-393) `SKEW_WINDOWS = (20, 50, 100)`. Population skewness formula `m3 / sd^3` then clipped to [-10, 10]. Skewness on 20-tick returns has huge variance — for a window with one outlier, skew can be ±√W. Is the clip enough?

**Q71.** (B:388) `m3 = m_r3 - 3.0 * mu * m_r2 + 2.0 * mu^3`. This is the centered third moment formula. For mid log-returns at tick scale, `mu` is ~1e-5 and `m_r2` is ~1e-8, so the correction terms are O(1e-13) and `m_r3` is the dominant term. Effectively skew == raw third moment ratio. OK but fragile to numerical precision in float64? Probably fine.

**Q72.** (B:380-385) Uses log-returns of midprice1. But mid in 3s data is often constant for many ticks → r has many zeros → skew of mostly-zero distribution is dominated by the few non-zero ticks → very noisy estimator.

### gofi (30)

**Q73.** (B:395-429) GOFI = Generalized OFI. Definition:
```
bid_cont = bid_up * bs_now  -  bid_dn * bs_lag  +  bid_eq * (bs_now - bs_lag)
ask_cont = ask_up * as_lag  -  ask_dn * as_now  -  ask_eq * (as_now - as_lag)
e = bid_cont + ask_cont
```
The ask side has `- ask_eq * (as_now - as_lag)` — i.e. when ask price unchanged, growth in ask size *decreases* OFI (more sellers waiting = bearish). The bid side: `+ bid_eq * (bs_now - bs_lag)` — growth in bid size *increases* OFI (more buyers waiting = bullish). Sign-consistent.

But look at the **non-eq** terms:
- bid_up uses `bsnow` (the *new* bid size at the new price level)
- bid_dn uses `bs_lag` (the *old* bid size at the old level)
- ask_up uses `as_lag` (price moved up: ask retreated, ask size at old level is "consumed")
- ask_dn uses `asnow` (price moved down: aggressive seller pressure visible at the new level)

This matches Cont-Kukanov definition. OK. **But compare to MLOFI in B:169-174** — totally different formula. Two OFIs in the same file using different conventions. Why?

**Q74.** (B:415-418) `ask_eq = (anow == a_lag).astype(float64) * (asnow - as_lag)`; `ask_cont = ask_up - ask_dn - ask_eq`. The minus sign on ask_eq is intended (ask growth ⇒ negative OFI). But `bid_cont = bid_up - bid_dn + bid_eq` has a PLUS on bid_eq. Sign asymmetry is correct only if you trust the convention. Worth a comment.

**Q75.** (B:419-420) `e = np.zeros((N, T)); e[:, 1:] = bid_cont + ask_cont`. So `e[:, 0] = 0` by construction. Then summing over windows including index 0 (e.g. W=100 starting at -100: → 0..99) includes a known-zero. Slight bias for the W=large case.

**Q76.** (B:421-429) GOFI for 10 levels × 3 windows (5, 20, 60) = 30 features. As discussed in Q31, top-N (N>3) levels rarely move on liquid 3s tick data. Many gofi_W*_lvl{6..10} features are probably ≈ 0 most of the time.

**Q77.** (B:411-413) `bid_up = (bnow > b_lag).astype(float64) * bsnow`. Strict `>`. Vs MLOFI's `>=` (B:169). Inconsistent equality treatment between MLOFI and GOFI.

### kyle_lambda (2)

**Q78.** (B:431-453) `signed_dvol = sign(Δmid) * sqrt(|amt| + EPS)`. Why square root of dollar volume? Why not raw dollar volume? Almgren impact model uses √V for permanent impact; Kyle's original lambda uses V linearly.

**Q79.** (B:435-437) `sign_dm = np.sign(delta_mid)`. When `Δmid == 0` (common in 3s ticks), `sign = 0` → `signed_dvol = 0`. So all the "no-change" ticks vote 0 into both numerator and denominator — biases λ toward the active subset of ticks.

**Q80.** (B:441-452) Cov/var-ratio formula:
```
cov = E[X·Y] - E[X]·E[Y]
var = E[X²] - E[X]²
lam = cov / (var + EPS)
clip(-1, 1)
```
- Standard formula but no `n` correction → using biased estimators. For W=50, ddof=0 means MLE estimates. OK for our scale.
- The clip to [-1, 1] is **harsh**: real Kyle λ can be much larger than 1 in absolute value depending on units. Why clip to 1? Implies they pre-knew λ rarely exceeds 1 (which it doesn't with the sqrt-amt scaling).

**Q81.** (B:431-453) Output is "linear regression slope of Δmid on signed √-dvol". Result ∈ [-1,1] after clip. Pretty noisy — variance proportional to 1/√W. With W=50 the std error of λ is huge. KS-dropped by FAIL_NAMES (Q6). **Confirmed**: this whole family is wasted.

### vol_burst (4)

**Q82.** (B:455-468) `vol_burst = vol[-1] / mean(vol[-W:])`. Last-tick volume / W-tick average. "Spike" detector. But clipped to [-100, 100] for `vol_burst` (B:465) and [0, 100] for `amt_burst` (B:467). Why allow negative for vol but not amt? `volume_delta` should be non-negative (volume traded in last 3s) but might be reported as 0 with cumulative... is `volume_delta` ever negative? If never, why clip to negative range at all?

**Q83.** (B:458-468) `vol_burst_W20` and `vol_burst_W50` — both. With overlapping windows of size 20 and 50, these two are highly correlated. Why not pick one?

**Q84.** (B:459-460) `mean_v = vol[-W:].mean() + EPS`. Using EPS=1e-8 in the denominator of a count-volume ratio is meaningless when typical vol is ≥ 1. But when window is all zeros (no trades for last 20 ticks, possible at lunch close), `mean_v = EPS` and `rv = vol[-1] / EPS = vol[-1]*1e8` — instantly hits the clip-100 boundary. So "all-zero window, then a trade" gets a constant 100. Is that the intended sentinel?

---

## §5. Stage 3 (14 dims): ewma_resid + rv_ratio + jshare + cancel_imb + roll_eff_spr

### ewma_resid (1)

**Q85.** (B:499-504) `EWMA_RES_ALPHA = 0.05` (HL ≈ 13.5 ticks ≈ 40s). `resid = mid[-1] - ewma_mid`. Clipped to [-0.05, 0.05]. **What are the units of mid?** If mid is in yuan (e.g. 50), then 0.05 = ¥0.05 = ~5 ticks = ~10 bp on a ¥50 stock. So the clip is "the mid deviates by at most 10bp from its 40s EWMA". For high-vol regimes, this clip activates frequently → loss of info. For low-vol it's never hit. Either way, an absolute clip on a value that scales with price is sketchy. Should be relative.

**Q86.** (B:499-500) `_ewma_last_batch(mid, (0.05,))[:, 0]` — EWMA over the *raw* midprice, not log-mid. So the residual is in price units, not return units. Mid drifts over the session → ewma_mid lags mid → resid is non-stationary (has a drift). Bad feature.

### rv_ratio (3)

**Q87.** (B:506-519) `RV_RATIO_PAIRS = ((5,50), (20,100), (50,100))`. Each ratio is `Σ r²(W_n) / Σ r²(W_d)`. By construction `rv_W5 ≤ rv_W50` (smaller window has less variance), so `rv_ratio_W5_W50 ∈ [0, 1]` typically, but with EPS division it can exceed 1 numerically. Clip [0, 100].
   - For `(5,50)`: short-term / mid-term — burstiness indicator. OK.
   - For `(50, 100)`: ratio of bottom-half RV to whole window — by construction `rv50/rv100 ∈ [0, 1]` (since rv50 ⊆ rv100 components). Then clip [0, 100] does nothing. Redundant clip.

**Q88.** (B:513-514) `rv_W[W] = r2[:, -W:].sum(axis=-1)`. No normalization by W → these are sums, not means. Then ratio sum_W1/sum_W2 ≈ (W1/W2)*(avg_r²_W1/avg_r²_W2). So the W1/W2 ratio is mostly the ratio of window-lengths times an actual vol ratio. Did anyone realize the constant baseline?

### jshare (4)

**Q89.** (B:521-533) Jump-share: `(RV - BV) / RV` where `BV = (π/2) Σ |r_t|·|r_{t-1}|` is bipower variation. `JSHARE_WINDOWS = (20, 30, 50, 100)`. Why include 30 but not 40 or 60? Inconsistent with other window sets ({20,50,100} or {5,20,60}).

**Q90.** (B:523-525) `abs_r_lag[:, 1:] = abs_r[:, :-1]; prod = abs_r * abs_r_lag`. Standard BV. For W=20, very few "jumps" expected → jshare ≈ 0. Estimator variance huge at small W → noisy. Clip to [0, 1].

**Q91.** (B:530) `j = (rv_w - bv) / (rv_w + EPS)`. Negative when no jumps (BV slightly > RV due to sampling noise). Clipped to [0, 1] — so half the distribution is censored at 0. Result is mostly 0 with occasional spikes. Probably zero predictive info on most ticks.

### cancel_imb (3)

**Q92.** (B:535-554) `cancel_imbalance = cb_share - ca_share`, where each share is `cancel_intensity / (limit+market+cancel)_intensity`. **Question**: is `*_intst` per-tick rate (events/sec) or count in last 3s? If rate, summing over W gives weighted sum; if count, summing gives totals. Without raw data spec we can't say.

**Q93.** (B:544-554) `CANCEL_WINDOWS = (20, 50, 100)`. Why these? Same as adapt_mom and several others.

**Q94.** (B:549-551) `cb_share = cb_sum / (bt_sum + EPS)`; `ca_share = ca_sum / (at_sum + EPS)`. Both shares in [0,1]; difference in [-1,1]. Clip [-1,1]. OK.

### roll_eff_spr (3)

**Q95.** (B:556-576) Roll (1984) effective spread: `eff = 2 * √(-cov(Δp_t, Δp_{t-1}))` if cov<0, else 0. Uses `close` not midprice! Why? close-to-close returns include bid-ask bounce; midprice doesn't. Roll's estimator specifically relies on bid-ask bounce → need raw trade price. So `close` is the right choice if close == last trade price. But often `close` in LOB snapshots is actually last mid-price-as-of-snapshot. **What is `close` in this dataset?**

**Q96.** (B:563) `qspr_last = X3d[:, -1, spread1] + 1.0`. The "+1.0" again. As denominator. Why? Spread is already non-negative; if spread=0, division by `EPS+1.0 ≈ 1` is fine. Adding 1 makes the ratio always smaller than `eff` for any spread<1. **Looks like another fudge constant.**

**Q97.** (B:565-575) `ROLL_WINDOWS = (30, 50, 100)`. Why include 30? Same comment as Q89.

**Q98.** (B:572) `eff = 2.0 * np.sqrt(np.maximum(-cov, 0.0))`. Standard, but: about half of empirical -cov(Δp_t, Δp_{t-1}) is positive due to momentum (not noise). For ticks with positive autocorr in price changes, eff = 0 — feature truncates to zero in half the cases.

**Q99.** (B:573) `ratio = eff / (qspr_last + EPS)`. `eff` is in price units, `qspr_last` is spread+1 in price units → ratio dimensionless. Clip [0, 10].

**Q100.** (Q66+B:566) Same vectorized rolling cov formula as kyle_lambda — both implementations could share a helper. They don't. Minor.

---

## §6. Stage 5 (20 dims): adapt_mom + ofi_tox + signed_bv + spread_reg + trade_pers + liq_asym

### adapt_mom (3)

**Q101.** (B:614-622, Stage5:60-68) `adapt_mom = (mid[-1] - mid[-W]) / (std(mid[-W:]) + EPS)`. Compares last to mid W ticks ago, normalized by W-window vol. Clip [-10, 10]. 
   - Cherry-pick risk: depends on the single value `mid[-W]`. If that one tick was an outlier, the feature spikes.
   - `std(mid[-W:])` uses raw prices, not returns. For a drifting price series, std is dominated by trend, not micro-fluctuation. So adapt_mom can saturate easily.

**Q102.** (B:614-616) For W=100 (full window), `mid[-100] = mid[0]`. So `adapt_mom_W100 = (mid[-1] - mid[0]) / std(mid)`. This is "where did mid end vs start over the window, scaled by window std". For a constant-drift series this becomes ~(rate*W)/std ~ W/N(0,1) → huge values, often clipped at ±10.

**Q103.** (B:617) `ref = mid[:, -W]`. For W=20, this is `mid[:, -20]` = 20 ticks back = 60 seconds. For W=100 it's the **window start**. So `adapt_mom_W100` uses the boundary tick — extremely sensitive to that one value.

### ofi_tox (4)

**Q104.** (B:634-669) OFI Toxicity = Pearson correlation between OFI and Δmid over W. `OFI_TOX_PAIRS = ((1,20),(1,50),(5,20),(5,50))`. Why not also (1, 100) or (5, 100)?

**Q105.** (B:641-644) `b_prev[:, 0] = b[:, 0]` (NOT NaN like in MLOFI). So the Stage5 OFI uses a "no-change at t=0" convention. **Inconsistent with the MLOFI in same file (Q28).**

**Q106.** (B:645-651) Sign convention:
```
bid contribution: + (bid_up * bs - bid_dn * bs_prev + bid_eq * (bs - bs_prev))
ask contribution: - (ask_up * as_prev - ask_dn * asz - ask_eq * (asz - as_prev))
e = bid - ask
```
Note `ask_eq` has a minus sign inside the parens, so the whole ask contribution is `-(ask_up - ask_dn - ask_eq) = -ask_up + ask_dn + ask_eq`. Translated: ask price unchanged but ask size grew → +ask_eq → reduces `e` (bearish). Bid price unchanged but bid size grew → +bid_eq → increases `e` (bullish). Consistent.

**But**: in MLOFI (B:169-173), there is no `eq` term at all → MLOFI ignores size-change-at-fixed-price events, Stage5 OFI counts them. **Different definitions of "OFI" in same file.** Documentation needed.

**Q107.** (B:655-669) Pearson corr over 20 or 50 ticks — std error of corr estimator is ~1/√W → for W=20 the 95% CI is ±0.45. **The correlation values are mostly noise.** And then it's clipped to [-1, 1] — meaningless clip given values are bounded.

**Q108.** (B:660-665) `sxy = (x_c * y_c).sum`, `sxx = ...`. Two-pass implementation but recomputed manually. NumPy has `np.corrcoef` but vectorizing across batch dim requires manual code — OK, but no comment explaining why.

### signed_bv (3)

**Q109.** (B:671-685) `signed_bv = (π/2) Σ sign(r_t) * |r_t| * |r_{t-1}|`, normalized by RV. Effectively "BV with sign of current return". Clip [-10, 10].

**Q110.** (B:675-676) `sign_r = np.sign(r)`. When r=0, sign=0. Most ticks have r=0 (3s, slow mover) → contributes 0 → estimator dominated by few non-zero ticks → noisy.

**Q111.** (B:680-685) Normalization by `rv = r².sum() + EPS`. So signed_bv/rv ∈ [-π/2, π/2] approximately (since |r||r_{-1}| ≤ (r²+r_{-1}²)/2). But the clip is [-10, 10] — 5x wider than expected range. Either over-cautious or the normalization sometimes blows up.

### spread_reg (3)

**Q112.** (B:687-699) `spread_regime = (spr_last - median(spr_W)) / IQR(spr_W)`. Robust z-score on spread. Clip [-10, 10]. Sensible.
   - But for W=20: 20 samples to compute median + IQR is very coarse. Quantile interpolation matters: numpy default is linear → can yield biased estimates with discrete spread values.
   - When IQR=0 (constant spread, common!), `denom = inf` → feature = 0.

**Q113.** (B:692-695) `np.quantile(seg, 0.75, axis=-1)` etc. uses default linear interpolation. For integer-tick spreads, this gives non-integer values. Not necessarily wrong, but means even "spread is always 1" gives q75=1, q25=1, IQR=0 → feature = 0. So most ticks → 0 → feature mostly dead.

### trade_pers (3)

**Q114.** (B:701-719) Lag-1 autocorr of `sign(Δmid)`. With Δmid often 0, sign often 0, autocorr is mostly dominated by ticks where mid moves. For W=20 with maybe 5 non-zero Δmid ticks, the autocorr is computed over essentially 4 lag pairs. **Extremely noisy.**

**Q115.** (B:704-705) `sign_lag[:, 1:] = sign_dm[:, :-1]; sign_lag[:, 0] = 0`. Lag with zero-padding (not NaN). Means the first element of `sign_lag` is 0, biasing the corr.

**Q116.** (B:702-718) The two-pass Pearson formula again, identical to ofi_tox. Helper not extracted.

### liq_asym (4)

**Q117.** (B:721-733) `liq_asym = log((1 + Σbid_top5) / (1 + Σask_top5))`. Aggregates top-5 size both sides over W ticks, takes log ratio. Why top-5 and not top-1 or top-10? Why W=5 included (W=5 means only 5 ticks of history)?

**Q118.** (B:722-726) `bid_top5 = sum of bsize1..5` per tick. Then summed across W ticks. So feature = `log((1 + W*<avg bid top5>) / (1 + W*<avg ask top5>))`. The `+1` is dwarfed by the `W*...` for W ≥ 20 and typical sizes 1e3+. So `+1` is dead code for W=20/50/100 and only matters when both sides are tiny (W=5 with extreme regime). Pointless epsilon.

**Q119.** (B:727-732) For W=5, summing 5 ticks of top-5 size on both sides = very small window. Feature should be ~stationary and basically `log(bid/ask)`. Was its KS-drop (FAIL_NAMES: `liq_asym_top5_W5`) because it correlates 0.99 with W=20 of the same?

**Q120.** (B:730) `np.log` — `log(0/x) = -inf` and `log(x/0) = +inf`. But the `1+` saves against both being 0. So safe. But never clipped to a sensible range; just `np.clip(v, -10, 10)`. For a healthy book, the log-ratio is ~±0.5 → clip is mostly inactive, but for one-sided liquidity event, the clip activates. Why ±10?

---

## §7. Cross-stage / global concerns

**Q121.** (B vs S file) Stage5 logic in batch (B:602-736) is identical to standalone `stage5_features.py`. Why two copies? Both maintained? If they drift, which wins?

**Q122.** (B vs S) Single-window `fast_features.py` (S) does NOT include Stage5 (its `all_feature_names()` at S:554-556 returns 196 = 69+54+59+14, NOT 216). But `04_build_pkg/fast_features.py` is shipped in the same pkg as `fast_features_batch.py` (which DOES include Stage5). So **single-window code is stale** w.r.t. Stage5. Is single-window code ever called at inference? Predictor.py imports `fast_features_batch.py` (Predictor:191) — so single-window is dead code in production. Why ship it then?

**Q123.** (Predictor:191, B & S agree on first 196 cols) The first 196 dims of `compute_batch_features` are *defined* identically to S. But the *implementations* differ:
   - dual_z m100 in S uses entire window mean (`X.mean(axis=0)`, S:269), but `s100` uses `X.std(axis=0)` with default ddof=1 (NumPy default!) → biased differently.
   - dual_z in B (B:284) uses `ddof=0` explicitly.
   **BUG**: S uses `ddof=1` by NumPy default for `.std()`, but B uses `ddof=0`. So train/inference numerics differ. Confirm via S:269-270.

**Q124.** Actually re-check: NumPy `ndarray.std()` default is `ddof=0`. So both match. (NumPy stats default differ from pandas, which defaults to ddof=1.) Verify by reading the numpy version intended? `np.std` ddof default = 0; `pd.Series.std` ddof default = 1. If anywhere in the *training* pipeline pandas was used, train ≠ inference.

**Q125.** (S:267-270) `X[-DUAL_Z_W_SHORT:].mean(axis=0)` — `axis=0` since X is (T, 37). OK. Same in B (axis=1 because (N, T, 37)). Match.

**Q126.** (S:295) Kyle_inv `sigma = r[-W:].std() + EPS` — uses default numpy ddof=0. But `r` includes index 0 (which is 0 in our impl since `log_ret[0] = 0`). So `r[-100:]` includes a synthetic 0 at front for W=100 case → std slightly biased. Same for B. Same in any code computing returns over the full window.

**Q127.** (B:117-121) `_rolling_sum_W_full` uses `np.cumsum` — float64 cumulative on N×T arrays. Numerical stability: for T=100 and bounded inputs, fine. For T=10000+ would need Kahan compensation; not relevant here.

**Q128.** (B:25, S:27) `EPS = 1e-8`. Same in B, S, Stage5. Used in 20+ places. All applications of EPS are "denominator + EPS" or "value + EPS" — purpose to avoid div-by-0. Issue: for values with magnitude 1e-5 (typical mid log-return), `EPS = 1e-8` is 1000x smaller — no perturbation. For values with magnitude 1e-12 (rare), EPS is 1e4x bigger — bias. **Single global EPS for all scales** is wrong.

**Q129.** Why `EPS = 1e-8` and not 1e-12 (float64 native precision) or 1e-6 (float32 native)? Picked arbitrary?

**Q130.** (B:766-780) `compute_batch_features` returns concatenation of t3+s1+s2+s3+s5. No `assert out.shape[1] == 216`. If any stage returns wrong dim, you only catch it downstream.

**Q131.** (B:107-121) Rolling-sum helper uses `dtype=x.dtype` for `cs`, preserving float64. OK.

**Q132.** (B:125-132) `_rolling_mean_std_lastW` is **defined but never called** in this file. Dead code.

**Q133.** (B:765-780 vs Cache:103-105) Cache code does `extras = np.where(np.isfinite(extras), extras, 0.0)`. So NaN/inf in any of the 216 features becomes 0. This silently masks bugs — a buggy feature returning `inf` everywhere shows up as zero everywhere, which would still train but underperform invisibly. **No alerting on % NaN per feature.**

**Q134.** (Cache:170-175) Progress print only every 20th session. No memory tracking. For 100 sym×date×sess sessions × 1842 rows × 370 cols × 4 bytes = ~273MB just for one chunk; concatenation peak could hit several GB. No streaming write.

**Q135.** (B:766, Cache:104) `compute_batch_features` is called once per session (with N = ~1842 windows). All 216 features computed in float64 — that's 1842 × 216 × 8 bytes = ~3MB per session in extras alone. With 1200 sessions in train, ~3.6GB. Memory fine for modern boxes, but: the intermediate `derived` arrays in t3 add (N, T) per derived key — 1842 × 100 × 8 = 1.4MB per derived. With 5 derived keys, another 7MB. Fine.

**Q136.** (B:766) The whole pipeline assumes T=100. Many constants (`-W:` where W=100, hardcoded 20 in B:198) implicitly require T=100. **No early assertion** `assert T == 100`. If a caller passes T=50 windows, dual_z_W_long would silently use entire window mean — different from designed behavior. Add `assert T == 100, ...` at top of `compute_batch_features`.

**Q137.** (B:202-216) WMP loop computes wmp for all 10 levels but emits only `wmp[:, -1]` for each. The full WMP array per level is allocated but only level 1 is kept in derived. Memory: 10 × (N, T) → 10 × 1.4MB = 14MB per session. Could be optimized to compute only level 1 with full T-array.

**Q138.** (B:269-291 dual_z stack) `np.column_stack` equivalent via `X3d[:, :, dz_idx]` — gather along axis=2 via fancy indexing — generates a copy (37 cols × 100 × N float64 = ~5MB per call). For 1842 windows per session × 1200 sessions: not the bottleneck but worth profiling.

**Q139.** (Cache:88-90) `valid_lo = 99` assumes the first 99 ticks have **not enough history** for the 100-tick window. But many features use shorter sub-windows (W=5,20,50). For the first 99 ticks, you could still compute W=20 features with valid data. By insisting on full 100-tick history before emitting any sample, you discard `99 × N_sess` rows where most features would be valid. Why?

**Q140.** (Cache:131) `mp_t{h} = mid_arr[rows + h]`. So label = midprice1[t+h] for label_h regression. Discrete classification labels are `label_{h}` columns. **What is the relationship between `mp_t{h}` and `label_{h}`?** Presumably label is `sign(mp_t{h} - mp_t) > thr`. Without inspecting upstream label generator we can't audit consistency.

**Q141.** (Predictor:283) Predictor inference uses `np.sign(v) * np.log1p(np.abs(v))` on `amount_delta` at the last tick of the raw input. Same transform as training (Cache:108-110). **Confirm same; verify scale** — but recall the raw features fed to derived feature computation in Predictor (line 285 `compute_batch_features(X3d, ...)`) use raw `amount_delta`, NOT log-transformed. Same as training. So train/inference symmetric. OK.

**Q142.** (Cache:104, Predictor:286) Both apply `np.where(np.isfinite(extras), extras, 0.0).astype(float32)`. Symmetric NaN handling. OK.

---

## §8. Naming / order / IDs

**Q143.** (B:248-261) `t3_no_time_feature_names()` returns names in order: mlofi → wmp → balance → rv → ewma. **Must match physical output order** in `compute_t3_no_time_batch` (B:177-241). Visually checked: ✓. But no unit test asserts this.

**Q144.** (B:333-340, S:311-318) `stage1_feature_names()` matches `compute_stage1_batch` output order. Same as above.

**Q145.** (B:473-483) `stage2_feature_names`: qrank(20) → rskew(3) → gofi(30) → kyle_lam(2) → vol_burst(4). 20+3+30+2+4 = 59. ✓.

**Q146.** (Predictor:68-75 vs B:474) `qrank_W{QRANK_W}_{c}` uses `QRANK_W=100` so prefix is `qrank_W100_`. FAIL_NAMES has `qrank_W100_spread1` etc. — name matches. But if anyone bumps `QRANK_W` to 50, the FAIL_NAMES name no longer matches → bad features survive silently. **Brittle hardcoded names.**

**Q147.** (B:339, S:317) EWMA-OFI name: `f"ewma_ofi_a{a}_lvl{k}"` where `a` is float like `0.05`. Float-to-string is locale-sensitive and `f"{0.1}"` = `"0.1"` but `f"{0.30000000000001}"` could differ. Risk of name mismatch across machines.

**Q148.** (B:582-587) `stage3_feature_names()`: 1+3+4+3+3 = 14 ✓.

**Q149.** (B:739-753) `stage5_feature_names()`: 3+4+3+3+3+4 = 20 ✓.

**Q150.** Total = 69 + 54 + 59 + 14 + 20 = 216 ✓. Matches `compute_batch_features` output dim (Q1).

---

## §9. Missing / suspicious / dead

**Q151.** No `stage4_features.py`. The naming jumps stage1 → stage2 → stage3 → stage5. **Where is stage4?** Was it tried, failed, and silently dropped? Are stage4 features somewhere else, included by mistake / left out by mistake?

**Q152.** Comments at top of stage5_features.py (Stage5:1-23) say "T68 Stage 5 features (batch-vectorized)". "T68" implies an iteration label. Yet `fast_features_batch.py` says "iter_012" (B:1), `fast_features.py` says "iter_010" (S:1), and Predictor.py imports as `iter_018_ffb` (Predictor:191). **Three different iteration numbers in one pkg.** Which is authoritative?

**Q153.** (B:1) "iter_012 batch-vectorized" — but Predictor labels it as iter_018. Either header is stale or pkg dir is misaligned.

**Q154.** (B:768-774) `compute_batch_features` allows X3d to be any K-wide array as long as `col_idx` indexes it. No validation that all referenced column names exist in col_idx. KeyError would surface at runtime — silent unless tested.

**Q155.** (Cache:104) The pipeline NEVER explicitly checks that any feature is in a "sane" range. If a bug pushes a feature to 1e15, it ends up clipped via float32 saturation (becomes ~3.4e38) or just stays huge. **No per-feature value-range monitoring.**

**Q156.** (B:765-780, Stage5:46-188) All Stage5 features either clip to [-10, 10] or [-1, 1]. No reason given for these specific bounds. Were they chosen by inspection of train data distribution? Documented anywhere?

**Q157.** (S:1-15) The docstring of single-window says it returns 350 dims (T3+S1+S2+S3+raw_last = 69+54+59+14+154). But `compute_window_features` returns 196 dims (no raw_last). **Doc/code disagree.**

**Q158.** (S:9-11) Says "the caller drops 10 KS-fail extras to get 340 final features". But the production pkg has FAIL_NAMES = 11 names dropped, not 10. **Stale comment.** Was originally 10, became 11 with stage5 addition?

**Q159.** (B:1-7) Header says "T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14) + Stage5(20)" — matches code. But "(69) + Stage1(54)" ordering means features in the 216-vec are grouped by stage. Yet `FAIL_NAMES` removes features from across stages → final 205-vec is interleaved gaps. Did anyone verify keep_idx is computed correctly (Predictor:200-203)?

**Q160.** (Predictor:200-203) `_extra_keep_idx = [i for i, n in enumerate(extra_names) if n not in FAIL_NAMES]`. Order-preserving filter. Length = 216 - len(FAIL_NAMES set intersection). What if a FAIL_NAMES entry isn't in extra_names (typo)? Silent — no warning that "kyle_lam_W50" was supposed to be dropped but wasn't found.

**Q161.** Should validate: `assert len(set(FAIL_NAMES) - set(extra_names)) == 0`. Currently not done.

---

## §10. Constraint compliance & micro-leakage

**Q162.** (B:13-17 header claims compliance) "sym never used, date never used, each window independent". 
   - sym: no `sym` index used in feature computation ✓.
   - date: no `date` index used ✓.
   - But: window concatenation across sessions might inherit ordering. Looking at Cache:160-168 — Xs are appended per-session in iteration order (sym, date, sess). The NPZ doesn't shuffle. **If downstream training reads sequentially without shuffling, session-block effects could leak.** Are training scripts using random shuffle?

**Q163.** (Cache:160-165) Stores `sym`, `date`, `sess_idx` arrays alongside features in NPZ. These ARE used at training time (for stratification, splits, weighting?). Confirm they are NOT used as input features.

**Q164.** (Predictor:225-227) `_col_idx` aliases `midprice = midprice1`. Defensive. But what if upstream raw schema renames midprice1 to midprice? No fallback.

**Q165.** (Predictor:291-304) `_extract_band_per_row` reads `df["sym"].iloc[-1]`. So **sym IS used** at inference time — for the conformal abstain band lookup, not for model input. The header (Predictor:20-21) explicitly allows this. But: is `iloc[-1]` always defined for the last row of the input frame? What if the input is sorted differently? **The contract: sym at last tick = the prediction target sym.** Verified upstream?

**Q166.** (Predictor:295-303) Falls back to `_cw_default_band` if sym not in 0..4 or sym column missing. Default β=0.16, σ=4e-4 → band = 6.4e-5. Where do these defaults come from? Median of in-sample bands?

**Q167.** (Predictor:282) `if self._amount_delta_idx >= 0: ...`. Always >= 0 since `index()` returns valid or raises. The guard is dead. Bug: should be `if self._amount_delta_idx is not None`?

**Q168.** (Predictor:281-283) Applies log1p to `raw_last[:, amount_delta_idx]` AFTER copying `raw_last` from X3d. So X3d itself is unchanged → downstream `compute_batch_features` sees raw amount_delta. Correct, but easy to misread. Comment would help.

---

## §11. Performance / numerical edge cases

**Q169.** (B:159-174) Per-level MLOFI loop creates 4 `_prev` arrays per level × 10 levels = 40 allocations. Could batch all levels in one tensor. Minor.

**Q170.** (B:235-241) EWMA outer loop is alpha × col instead of vectorized — but `_ewma_last_batch` handles all alphas in one call. Why not pass all `T3_EWMA_ALPHAS` at once for each col?

**Q171.** (B:82-100) `_ewma_last_batch` iterates over alphas with separate `lfilter` calls. For 4 alphas × 6 intst × N windows, that's 24 filter calls. Could be a single batched IIR.

**Q172.** (B:286-289) `EPS=1e-8` in std denominator. For `bid_diff5` or `cumspread` (could be 0 for many ticks if spread doesn't move), `s20=0` → `(last-m20)/EPS` = `(0-0)/1e-8 = 0` if last == m20, or large if not. Sensible if EPS is much smaller than typical std. Not necessarily.

**Q173.** (B:506-519) RV computation done **separately in Stage1** (for signed_rv, line 296-299) and **again in Stage3** (line 508-511). The `r` series is recomputed instead of cached via `derived`. Inefficient.

**Q174.** (B:498) `derived` dict is passed into stage1/2/3 but **stage5** is called without `derived` (B:602). Stage5 recomputes `mid`, log returns, etc. Three implementations of the same log-ret formula across stages.

**Q175.** (B:294, 374-379, 508-510) `mid_safe = np.where(mid > 0, mid, np.nan); r[1:] = log(safe[1:]/safe[:-1])` — exact same idiom repeated 3 times (stage1, stage2, stage3). Refactor target.

**Q176.** (B:294-299, S:277-281) Single-window version computes r exactly the same. Numerics should match — but small float ordering differences arise from `(N, T)` slicing vs `(T,)` indexing. Has anyone verified train cache (batch) == predictor inference (also batch in this pkg) bit-exact?

**Q177.** (B:209-211) `wmp_normal = (a*bs + b*asz) / np.where(denom == 0, 1.0, denom)`. The `np.where` rewrites the denom to 1 when zero — but `wmp_normal` is still computed (gets nonsense value), then `wmp = np.where(denom == 0, fallback, wmp_normal)`. So the nonsense value is computed and thrown away — minor compute waste, more importantly: doesn't trigger fp warning.

**Q178.** (B:441-452) `kyle_lambda` uses biased mean/var. For W=50, `cov = mean(xy) - mean(x)*mean(y)` — biased but for symmetric distribution OK. For OFI which is left-skewed in down-trends, bias is non-negligible. Was bias considered?

**Q179.** (B:565-575) `roll_eff_spr` uses same biased cov formula. Same comment.

**Q180.** (B:660-665) Stage5 corr formula uses centered-and-summed (not means), avoiding the cov bias. **Different cov formula style** than Stage2/3 — inconsistent.

---

## §12. Hidden assumptions & "intuition gone wrong"

**Q181.** (B:215) `wmp_balance_12 = wmp_lvl1 - wmp_lvl2`. wmp_lvl1 ≥ best bid, wmp_lvl2 has wider levels → wmp_lvl2 is typically *very* close to wmp_lvl1 (~1bp). The difference is dominated by tick-size noise. **Useful signal or just tick-grid artifact?**

**Q182.** (B:563) `qspr_last = X3d[:, -1, spread1] + 1.0`. Adding 1 to spread before using as denominator: if spread is in raw price units (1 = 1 fen on a yuan stock), `+1` is a unit of price; if in basis points, `+1` = +1bp. **Unit-dependent magic constant.**

**Q183.** (B:218, 220) `wmp + 1.0` for the RV computation. Same `+1` mystery. Either both come from a shared "epsilon to avoid log(0)" intent (but EPS exists for that), or someone made a typo and never caught it. The wmp itself is a price > 0 always (if denom > 0). **Why add 1?**

**Q184.** (B:222-223) `log(safe[1:]/safe[:-1])` after `+1` shift. `log((wmp_t+1)/(wmp_{t-1}+1))` ≠ log return of WMP. For wmp ~50, `(wmp+1)/wmp ≈ 1.02` → an artificial 2% noise added to the log ratio per tick? Let me re-derive:
   - `log((50.001 + 1)/(50 + 1)) = log(51.001/51) ≈ 0.0000196`
   - actual log return = `log(50.001/50) ≈ 0.00002`
   So the `+1` doesn't add noise per tick; it *reduces* the apparent return ratio by factor `wmp/(wmp+1)` ≈ 0.98 (for wmp=50). So the RV is ~4% smaller than true RV. **Constant proportional bias** — model can learn around it, but it's still a hidden distortion.

**Q185.** Was the `+1` originally introduced because wmp could be negative on some debug data, and never removed? Code archaeology required.

**Q186.** (B:213-216) `wmp_per_level[0][:, -1] - wmp_per_level[1][:, -1]` = wmp1 - wmp2. wmp1 uses bid1/ask1/sizes; wmp2 uses bid2/ask2/sizes. Difference depends on whether levels 1-2 are stacked or scattered. For a tight book (1-tick spread, sizes at every level), wmp1 ≈ mid+ε, wmp2 ≈ mid+δ where ε,δ small → diff ≈ ε-δ ≈ tick-size noise. Useful?

**Q187.** (B:436-438) `signed_dvol = sign(Δmid) * sqrt(|amt| + EPS)`. **Issue**: `sign(Δmid)` is the price-change sign, but in market microstructure, "signed volume" usually means signed by trade direction (buy-initiated +, sell-initiated −), inferred via Lee-Ready or BVC. Using Δmid sign here is a **proxy for trade direction** which is well-known to be biased near the spread. Was that thought through?

**Q188.** (Stage5:154-170, B:702-718) `trade_pers` autocorrelates sign(Δmid) — but Δmid sign on 3s ticks is mostly 0 (no change). Effective sample size is much less than W. Power of estimator is poor for small W.

**Q189.** (B:633-645) Stage5 OFI sign convention vs MLOFI sign convention — already raised in Q73. Confirmed: **two different OFI formulas in one codebase, both named "OFI"**. Anyone reading this code blind would be confused.

**Q190.** (B:104-105) `_rolling_sum_last_W` — declared but never called in this module. Dead code (like _rolling_mean_std_lastW).

**Q191.** (B:323-328) EWMA-OFI loop iterates levels then computes 4 alphas in one call. Output: `out[:, col:col+4]`. **Ordering of alphas**: for each level, alphas in order (0.05, 0.1, 0.3, 0.5). Names emitted as `ewma_ofi_a0.05_lvl1, ewma_ofi_a0.1_lvl1, ...`. Match between names and output? Verified by visual inspection: ✓.

**Q192.** (B:235-241) EWMA-intst ordering: outer loop is **alpha**, inner is **column**. So output order is: `a=0.05 × (lb, la, mb, ma, cb, ca), a=0.1 × ..., ...`. Names in `t3_no_time_feature_names` (B:258-260) iterate same order ✓.

**Q193.** (B:233-234) "iter_009 EWMA path does NaN→0 fill before EWMA, so:" — reference to "iter_009" implies a previous iteration's behavior. Are we maintaining bug-for-bug compatibility with iter_009? If yes, why not just fix the bug?

**Q194.** (B:154-155, 178-185) Comments inside `compute_t3_no_time_batch` reference "iter_009" repeatedly. Suggest substantial rewrite was done but kept faithful to old quirks. **Pure technical debt.**

---

## §13. Schema / type / contract

**Q195.** (Predictor:46-66, Cache:53-69) `RAW_COLS_TRAIN_ORDER` in Predictor vs `RAW_COLS` in Cache. Are they byte-equal? Visual: order matches. But: there are TWO source-of-truth lists. **Move to a shared constants module.**

**Q196.** (Predictor:194-203) `_extra_names = list(self._ffb.all_feature_names())`. Imports the names from the batch module at init time. Consistent.

**Q197.** (Cache:155) `pd.read_parquet(path)`. No engine specified — defaults to pyarrow or fastparquet depending on installation. Could yield different float precision (pyarrow→float64 always, fastparquet may downcast). Risk of train/eval discrepancy.

**Q198.** (Cache:94) `A = df[RAW_COLS].to_numpy(dtype=np.float64, copy=False)`. Forces float64 even if source is float32. Inflates memory 2x but ensures consistent precision. OK.

**Q199.** (Cache:128) `labels = df[col].to_numpy()` — dtype unspecified. If labels are int8 in parquet, becomes int8 numpy; otherwise int64. Then cast to int8 (line 129). If label values exceed int8 range → silent overflow. Probably fine since labels are 0/1/2.

**Q200.** (Cache:130-131) `mid_arr = df["midprice1"].to_numpy(dtype=np.float32)` — cast to float32. Then `mp_t{h}` is float32. But the `mp_t` (current) at line 115 is also float32. So labels and mp arrays are float32 throughout. OK.

---

## §14. End-to-end open questions

**Q201.** No code anywhere computes feature importance, KS stats, or any feature-quality metric. The `FAIL_NAMES` decision happened elsewhere — where? Reproducible? Versioned?

**Q202.** For each stage1/2/3/5 feature, what is its empirical Spearman correlation with label_60? If <0.005 (often the case for noisy features), it's adding only noise to LGB and (worse) noise to the NN.

**Q203.** Are any of these 216 extras pairwise correlated >0.95? Almost certainly: dual_z_midprice1 vs dual_z_midprice2 vs dual_z_midprice5, the multiple MLOFI windows on adjacent levels, etc. **Redundancy without explicit decorrelation**.

**Q204.** `T3_RV_WINDOWS = (5,10,20,50)` vs `T3_MLOFI_WINDOWS = (5,20,60)` vs `SIGNED_RV_WINDOWS = (20,50,100)` vs `RV_RATIO`'s windows (5,20,50,100) — every family has its own window set. Why no unified set?

**Q205.** Window 60 appears only in MLOFI/GOFI; window 30 appears only in JSHARE/ROLL. **Idiosyncratic** — not tuned, just picked.

**Q206.** Have any of these features been ablated? I.e. train with vs without each feature family. If not, you don't know which add value.

**Q207.** (B:212-214) `wmp` includes `np.where(denom == 0, fallback, normal)` — but downstream code uses wmp without checking for fallback events. If denom=0 is rare, fallback effect minor; if common, you have a chunk of "(a+b)/2 = mid" entries diluting your wmp_balance_12 to 0.

**Q208.** (B:455-468) `vol_burst` clipped at -100/+100. With `mean_v + EPS` denominator, the only way to hit ±100 is `vol[-1] / mean_v > 100` i.e. last vol is 100x the W-avg. Rare event detector. But for stocks with bursts of activity at e.g. half-hour marks, this fires consistently — does this leak time-of-day info via volume regime patterns?

**Q209.** (Stage5:182, B:730) `log((1+sb)/(1+sa))` — `sb, sa` are sums of sizes over W ticks (could be 1e7+ for W=100 on liquid stocks). Then `log(big/big)` ≈ `log(1 + (sb-sa)/sa)` ≈ `(sb-sa)/sa`. The `+1` is dead. Why use log at all then? Could just compute `(sb-sa)/(sa+1)`.

**Q210.** (B:766-780) Feature ordering is **stage-grouped**, but FAIL_NAMES drops features mid-stage → final feature vector has gaps in stage boundaries. NN feat_mean/feat_std were trained on the kept-only vector — order must match. Predictor.py:200-203 builds keep_idx → applies at inference (Predictor:287). Confirm: the NN's `feat_mean[:, kept_idx]` was actually trained on data in the same kept order. **No assertion** in Predictor or training code verifies feat_mean dim == len(kept_idx) at load time.

**Q211.** (Predictor:148-160) `if X_in.shape[1] != self.in_dim: X = X_in[:, self.keep_idx]`. So the BatchedMLPEnsemble also applies its own keep_idx. **Two levels of keep_idx**: Predictor applies _extra_keep_idx (drops FAIL_NAMES), then BatchedMLPEnsemble applies its keep_idx if dims still don't match. Could double-drop. The NN's `keep_idx` was set during training; if NN was trained on the post-FAIL_NAMES vector, then `keep_idx == identity`. Verify.

---

## §15. Things that look like "I forgot"

**Q212.** (B:155) `# Note: fast_features_batch already integrates Stage5` — yet `stage5_features.py` is still maintained as a separate file. **One is dead.**

**Q213.** (B:43-44) `DUAL_Z_COLS` includes `lb_acc..ca_acc` (6 cols, the accumulator versions of intensities). These are monotonic → see Q53. As features they're nearly useless (every increment is just the current intensity). **Should have used `*_intst` rate diffs, not `*_acc` levels.**

**Q214.** (B:36) `DUAL_Z_COLS` first entries are `spread1, spread5, spread10, cumspread`. With spread quantized to ticks, `dualz_spread1` is binary-ish, **likely uninformative for fine prediction**.

**Q215.** (Predictor:131-133) `target_scale` is per-NN — each NN has its own scaling factor for its output. Why? Implies each NN was trained with different label normalization. Then `mean(over NNs)` of differently-scaled predictions is **incoherent** unless target_scale is applied first (which it is: line 176 `out / self.target_scale`). OK, but worth a comment.

**Q216.** (Predictor:130) `self.clip = float(max(clips))` — uses the MAX clip across NNs. So if NN_3 has clip=10 and NN_7 has clip=5, all NNs use clip=10. Inconsistent with per-NN training behavior. Should be per-NN clip.

**Q217.** (Cache:43) `# Note: fast_features_batch already integrates Stage5 (216 = 196 + 20).` Implies single-window doesn't. Confirmed (S:554-556 returns 196). **Single-window cannot be used for inference** if model expects 216-d (or 205-d post-drop) vector. Yet build_pkg ships single-window. Why?

---

## §16. Final list of suspects

**Q218.** The single hottest "magic" numbers worth tracing back to a notebook:
   - `+1.0` in `wmp + 1.0` for RV (B:220)
   - `+1.0` in `spread1 + 1.0` for roll_eff_spr (B:563)
   - `EPS = 1e-8` everywhere
   - `clip(-10, 10)` in 7+ features
   - `clip(-1, 1)` in 4+ features
   - `clip(-0.05, 0.05)` in ewma_resid
   - `clip(-100, 100)` in vol_burst
   - `0.16` and `4.0e-4` as conformal OOD defaults (Predictor:222-223)

   None of these have a paper citation or experiment justification in the code. **Each should be tuned or removed.**

**Q219.** The two `OFI` definitions (Q73/Q189) AND the two equality conventions (Q77) AND the two boundary conventions (Q28/Q105) suggest this code is the result of merging multiple branches without harmonization. **Reproducibility risk** if branches were trained separately.

**Q220.** `liq_asym_top5_W5` failed KS but `liq_asym_top5_W{20,50,100}` didn't. The `W=5` value uses only 5 ticks of history → wildly noisy. Could be a sanity check: if W=5 failed and W>=20 didn't, the *signal exists* but you need ≥20 ticks. Fine. But why include W=5 at all if you suspect it'll fail?

---

## §17. Sym-agnostic / date-agnostic spot checks

**Q221.** All B and Stage5 features are computed per-window using only `X3d[n]` data → no cross-window contamination ✓.
**Q222.** No global statistics fit on train data are baked into B/Stage5 — all stats are window-local ✓.
**Q223.** Cache emits `sym, date, sess_idx` columns separately from X (no leak into X) ✓.
**Q224.** Predictor reads sym only for conformal band, never feeds to model — flagged with comment (Predictor:20-21) ✓ but **trust requires running an integration test on sym=99 (OOD) — Predictor's `__main__` (line 383) does this. Confirm OOD handling.**
**Q225.** No feature uses absolute timestamp, time-of-day, or session index. ✓ but `vol_burst` may indirectly proxy time-of-day via volume regime (Q208).

---

# SUMMARY (for the agent log)

Approx feature families examined:
- Raw 154-d block: 154 cols (~10 distinct families: open/high/low/close, vol/amt, bid/ask×10, bsize/asize×10, midprice×10, spread×10, bid_diff×10, ask_diff×10, intst×6, ind×6, acc×6, rate×40, mean/sum aggregates ×6)
- MLOFI: 30 cols (3 W × 10 lvl)
- WMP: 11 cols (10 wmp + 1 balance)
- RV: 4 cols
- EWMA-intst: 24 cols (4 α × 6 col)
- DualZ: 37 cols
- Signed RV: 3 cols
- Kyle inverse: 2 cols
- EWMA-OFI: 12 cols
- QRank: 20 cols
- RSkew: 3 cols
- GOFI: 30 cols
- Kyle lambda: 2 cols
- Vol burst: 4 cols (2 vol + 2 amt)
- EWMA residual: 1 col
- RV ratio: 3 cols
- JShare: 4 cols
- Cancel imbalance: 3 cols
- Roll spread: 3 cols
- Adapt momentum: 3 cols
- OFI toxicity: 4 cols
- Signed BV: 3 cols
- Spread regime: 3 cols
- Trade persistence: 3 cols
- Liquidity asymmetry: 4 cols

Total = 370 (154 raw + 216 derived). FAIL_NAMES drops 11 → 359 fed to models.

n_questions = 225
n_features_examined ≈ 25 families (216 derived + 154 raw cols across ~10 families)

RESULT: task=[features review] metrics={n_questions=225, n_features_examined=25_families_370_cols} notes=[225 questions in 17 sections covering every magic constant, window choice, formula, sign convention, eps, clip range, stage division, dead code, name-ordering risk, schema brittleness, and constraint-compliance. Highlighted bugs/concerns: (1) two different OFI definitions in same file [Q26 vs Q73 vs Q106]; (2) mystery `+1.0` constants in RV/WMP/spread denominators with no documented reason [Q39, Q96, Q183-185]; (3) `EPS=1e-8` global across features with values spanning 1e-12 to 1e7 [Q50, Q128]; (4) Stage5 OFI vs MLOFI use different boundary/equality conventions [Q28-29, Q77, Q105-106]; (5) dropped 11 features but kept the families that produce them — wasted compute on kyle_lambda etc. [Q6]; (6) single-window file (fast_features.py) doesn't include Stage5, yet shipped in build_pkg [Q122, Q217]; (7) header docstrings disagree with code (says 350-dim, returns 196; says drops 10, drops 11) [Q157-158]; (8) feature names depend on float formatting (a=0.1) — locale risk [Q147]; (9) iteration labels in file headers disagree (iter_010 vs iter_012 vs iter_018) [Q152-153]; (10) magic clip bounds (-10, -1, -100, -0.05) without justification [Q218]; (11) NaN→0 fill inside EWMA biases smoothed values [Q46, Q63]; (12) signed_dvol uses sign(Δmid) instead of trade direction — Lee-Ready proxy bias [Q187]; (13) dead helper functions and dead code [Q132, Q190]; (14) FAIL_NAMES not asserted to exist in feature names [Q160-161]; (15) no early assertion T==100 [Q136]]
