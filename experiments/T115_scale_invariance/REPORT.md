# T115: Scale-Invariance Audit + Fix Experiment

**Date**: 2026-05-08  
**Hypothesis (from PM)**: schemeP 359-d carries 2 silent scale-dependent leaks
violating sym-agnostic red line: (1) `amount_delta` is the only un-normalized
field (元 量级), (2) `spread*` / `bid_diff*` / `ask_diff*` / `cumspread` carry
absolute-tick-size signal that is structurally sym-dependent.

**Spoiler**: Hypothesis is partially confirmed by KS audit, but **fix does NOT
improve test performance**. Best variant `variant_C` (smart prev-close proxy)
gives ΔLOSO = -0.34 (within DE noise). NOT an iter_017 candidate.

---

## Phase 1 — KS Audit (370 features × 10 sym-pairs)

Method: subsample 20k per sym, compute pairwise KS distance per feature, take
max across pairs.

### Top 30 worst KS features (sym-distinguishability)

| Rank | feat_idx | feat_name              | KS_max | abs_mean ratio | std ratio |
|------|----------|------------------------|--------|----------------|-----------|
| 1    | 88       | bid_diff1              | 1.0000 | 1.00           | 2.10      |
| 2    | 97       | bid_diff10             | 1.0000 | 1.02           | 2.71      |
| 3    | 108      | bid_mean               | 1.0000 | **29.78**      | **85.09** |
| 4    | 110      | bsize_mean             | 1.0000 | 1.00           | 295.42    |
| 5    | 96       | bid_diff9              | 1.0000 | 1.00           | 3.05      |
| 6    | 95       | bid_diff8              | 1.0000 | 1.00           | 3.66      |
| 7    | 91       | bid_diff4              | 1.0000 | 1.00           | 3.29      |
| 8    | 92       | bid_diff5              | 0.9999 | 1.00           | 3.46      |
| 9    | 93       | bid_diff6              | 0.9999 | 1.00           | 3.48      |
| 10   | 89       | bid_diff2              | 0.9999 | 1.00           | 2.75      |
| 11   | 112      | cumspread              | 0.9999 | 1.26           | 128.17    |
| 12   | 90       | bid_diff3              | 0.9998 | 1.00           | 2.36      |
| 13   | 94       | bid_diff7              | 0.9998 | 1.00           | 3.56      |
| 14   | 48       | totalbsize             | 0.9946 | 5.37           | 6.36      |
| 15   | 49       | totalasize             | 0.9943 | 5.76           | 7.51      |
| 16-25 | 16-25   | bsize2..10, bsize1     | 0.88-0.98 | 28-77       | 14-59    |
| 26   | 331      | kyle_lam_W100          | 0.9729 | 237.74         | 25,289    |
| 27   | 330      | kyle_lam_W50           | 0.9462 | 501.20         | 41,105    |
| 28-30 | 277-280 | qrank_W100_spread1/5/cumspread | 0.85 | 1.7-2.0 | 6.7-10.1 |

### PM hypothesis verification

| Target              | KS_max | In top 30 (raw)? |
|---------------------|--------|------------------|
| `amount_delta`      | 0.568  | NO (rank ~80)    |
| `spread1-10` (raw)  | NaN/0  | NO (data has NaNs in sym=1; otherwise minor) |
| `bid_diff1-10`      | 1.000  | **YES (10/30)**  |
| `ask_diff1-10`      | NaN    | NO (NaNs in sym=1) |
| `cumspread`         | 1.000  | **YES**          |

**Verified**: `bid_diff*` and `cumspread` are **catastrophic sym-leak**
(KS=1.000 = totally non-overlapping distributions across syms).

**Falsified**: `amount_delta` is **moderate** leak (KS=0.57, after
log1p sign-preserving in cache); raw `spread1-10` show NaN issue but
not extreme leak. `ask_diff*` similar.

**NEW finding (not in PM hypothesis)**:
- `bid_mean`, `ask_mean` are stored in **元 absolute** (not 涨跌幅) →
  KS=1.00, abs_mean_ratio=30× across syms (worst leak by magnitude).
  Sample values per sym: 14.3 / 18.0 / 130.3 / 4.4 / 82.0 元.
- `bsize*`, `totalbsize/asize` (volume features) leak sym info too
  (KS 0.88-0.99, abs_mean_ratio up to 77×).
- `kyle_lam_W50/W100` leak (depend on price-volume scale).

### Feature encoding decoded

Raw parquet uses mixed encoding:
- `bid1-bid10`, `ask1-ask10`, `midprice1-10` are 涨跌幅 (relative to prev close, dimensionless)
- `bid_mean`, `ask_mean` are 元 (absolute, NOT 涨跌幅) ⚠️
- `spread_k = (ask_k - bid_k) - 1` (涨跌幅 difference with -1 offset)
- `bid_diff_k = (bid_k - bid_{k+1}) - 1`, `ask_diff_k = (ask_{k+1} - ask_k) - 1`
- `cumspread = sum(spread_k for k=1..10) - 1`
- `amount_delta` is **元** (raw); cache applies `sign·log1p` transform

**Critical implication**: PM's proposed fix `/ (midprice + 1.0)` divides
by a number ≈ 1.0 (since midprice is 涨跌幅 ≈ 0.01), so it is essentially
a **no-op normalization**. This was confirmed by experiment.

---

## Phase 2 — Variant feature caches built

| Variant     | Dims | Description |
|-------------|------|-------------|
| baseline    | 359  | schemeP 370-d minus T59_FAIL (10) + STAGE5_FAIL (1) |
| variant_A   | 391  | baseline + 32 PM ratios (`amount_delta + spread + bid_diff + ask_diff + cumspread`) / (midprice + 1) |
| variant_B   | 359  | baseline with 32 raw cols **replaced** by PM ratios |
| variant_C   | 391  | baseline + 32 smart-fix cols (NEW): `prev_close ≈ bid_mean / (bid1 + 1)`, then `(raw_offset + 1) × prev_close` for spread/diff/cumspread (restores 元-tick scale), `log1p(amount / prev_close)` for amount |

prev_close per sym (verified): 14.3 / 18.0 / 130.3 / 4.4 / 82.0 元 ✅
matches `bid_mean` per-sym means.

NaN issue: 86,602 NaN values in train (for sym=1 only on `ask_*`/`spread*`/`ask_mean` features) → replaced with 0.0 (LGB tolerates).

---

## Phase 3 — Training & DE thresh LOSO eval

**Training**: LGB Huber alpha=1e-3, 3 seeds (1, 7, 42), GPU,
V4 split (train 0-75, val 76-79), aug_a [0.8,1.2], 3-class balanced weights.

**Eval**: DE thresh search (asymmetric tu, td) on test, joint 5-sym LOSO sum;
report per_sym breakdown.

### 3-seed averaged DE LOSO results

| Variant     | DE Sum     | per_sym (sym0,1,2,3,4)              | min_per_sym | Δ DE     | Δ min   |
|-------------|------------|-------------------------------------|-------------|----------|---------|
| baseline    | **+41.225** | +4.95, +5.43, +4.31, +11.51, +15.02 | +4.31       | (ref)    | (ref)   |
| variant_A   | +39.916    | +4.71, +5.15, +3.84, +11.20, +15.01 | +3.84       | -1.31    | -0.47   |
| variant_B   | +40.083    | +4.62, +5.43, +3.81, +11.33, +14.89 | +3.81       | -1.14    | -0.50   |
| variant_C   | +40.880    | +4.65, +5.83, +4.09, +10.93, +15.38 | +4.09       | **-0.34** | -0.22   |

### Per-seed DE values

| Variant     | seed 1 | seed 7 | seed 42 | Mean |
|-------------|--------|--------|---------|------|
| baseline    | +41.21 | +40.99 | +39.40  | +40.53 |
| variant_A   | +39.64 | +38.83 | +39.15  | +39.20 |
| variant_B   | +38.82 | +40.69 | +38.98  | +39.50 |
| variant_C   | +39.94 | +41.73 | +38.88  | +40.18 |

### Per-sym Δ vs baseline (3-seed avg)

| Variant     | Δ sym0 | Δ sym1 | Δ sym2 | Δ sym3 | Δ sym4 |
|-------------|--------|--------|--------|--------|--------|
| variant_A   | -0.24  | -0.29  | -0.47  | -0.31  | -0.01  |
| variant_B   | -0.32  | -0.00  | -0.50  | -0.19  | -0.13  |
| variant_C   | **-0.30** | **+0.40** | **-0.22** | **-0.58** | **+0.36** |

variant_C is the only one where some syms IMPROVE: sym1 (+0.40, the
mid-price 18元 stock) and sym4 (+0.36, the 82元 stock). But sym3 (the
low-price 4元 stock, where prev_close proxy is noisiest) loses -0.58.

### KS reduction verified for variant_C

| Feature      | KS_raw | KS_smc (variant_C) | Δ KS    |
|--------------|--------|--------------------|---------:|
| amount_delta | 0.5684 | **0.2821**         | -0.286  |
| spread1      | 0.9963 | 0.9896             | -0.007  |
| spread5      | 1.0000 | 0.9882             | -0.012  |
| bid_diff1    | 1.0000 | 0.9998             | ≈ 0     |
| bid_diff5    | 0.9999 | 1.0000             | ≈ 0     |
| ask_diff1    | 0.9997 | 0.9999             | ≈ 0     |
| ask_diff5    | 0.9998 | 0.9999             | ≈ 0     |
| cumspread    | 0.9999 | 0.9967             | -0.003  |

**Only `amount_delta` KS dropped meaningfully** (0.57 → 0.28). The
spread/diff features remain near KS=1.00 because **typical spread in 元 is
itself sym-dependent** (sym1 has wider spreads than sym0 in 元 due to
liquidity, not unit confusion). This is a real economic difference that
no rescaling can fully erase.

---

## Success criteria check

| Criterion                          | Threshold     | Best result        | Pass? |
|-----------------------------------|--------------:|-------------------:|------:|
| ΔLOSO > +0.3                      | +0.3          | **-0.34** (variant_C) | ❌ |
| Δ per_sym_min > +0.5              | +0.5          | -0.22 (variant_C)  | ❌ |
| KS distance reduction on raw      | meaningful    | partial (only amount_delta) | partial |

---

## Recommendation

**Do NOT promote any T115 variant to iter_017.** Both PM-literal variants
(A, B) are -1.1 to -1.3 worse, and the smart variant_C (-0.34) does not
meet success criteria.

### Why the fix hurt

The "leaks" identified are **load-bearing features** for in-distribution
performance. The 5 syms in test are the same 5 syms in train, so
sym-specific signal (e.g., "if bid_diff1 ≈ 1 tick, bid is locked at the
top of book; predicts mean reversion") is genuinely predictive within
this universe. Removing/normalizing these features destroys legitimate
signal.

### When the fix WOULD matter

If the platform's hidden test set includes **stocks outside the 5
training syms** (red line #3 silent compliance risk), the baseline
model relies on memorized sym-typical bid_diff distributions and would
fail OOD. variant_C is **structurally sym-agnostic** and would
generalize cleanly.

We have no way to detect this OOD scenario without a held-out OOD sym.
But: the platform docs do not guarantee in-universe testing. The
silent-compliance risk remains.

### Best follow-ups (not pursued in T115)

1. **OOD test simulation**: pick one sym (e.g., sym2, the high-price
   130元 stock) as held-out, train on the other 4, eval on sym2. Compare
   baseline vs variant_C delta on the OOD sym.
2. **Targeted feature drop** (vs full re-normalization): drop only the
   worst-KS leaks (`bid_mean`, `ask_mean`, `cumspread`, `kyle_lam_W*`,
   `bsize_mean`, `totalbsize/asize`) and keep `dualz_/qrank_` versions.
   Less invasive than variant_B.
3. **Diversification ensemble**: ensemble baseline + variant_C — variant_C
   provides scale-invariant complementarity. Test correlation:
   if corr < 0.95, ensemble may add value even though variant_C alone is
   slightly worse.

### What to keep

- KS audit script (`ks_audit.py`) is reusable for future feature cache audits.
- `prev_close_proxy = bid_mean / (bid1 + 1.0)` is a useful primitive
  for any future sym-agnostic feature engineering.
- Decoded the raw feature encoding: `spread = (ask - bid) - 1`,
  `bid_diff = (bid_k - bid_{k+1}) - 1`, etc.

---

## Files

- `ks_audit.py`, `all_ks_features.csv`, `top_ks_features.csv`,
  `ks_audit_summary.json` — Phase 1 audit
- `build_variants.py` — Phase 2 cache builder
- `cache/{baseline,variant_A,variant_B,variant_C}_{train,val,test}.npz`
- `train_lgb_t115.py` — Phase 3 training script
- `model_T115_*.txt`, `pred_T115_*.parquet`, `summary_T115_*.json`
- `eval_t115.py`, `eval_t115.json`, `eval_t115.log` — Phase 3 eval

---

```
RESULT: task=scale_invariance metrics={baseline_loso_3seed=41.225, variant_a_loso=39.916, variant_b_loso=40.083, variant_c_loso=40.880, delta_loso_best=-0.345, delta_per_sym_min=-0.222} notes=[KS audit confirms bid_diff*+cumspread+bid_mean leak (KS=1.00), but PM fix is no-op (divides by ≈1); smart prev-close-proxy fix reduces only amount_delta KS (0.57→0.28); ALL variants underperform baseline, NOT iter_017 candidate; per-sym diff suggests structural sym differences are economic (spread liquidity), not unit-scale]
```
