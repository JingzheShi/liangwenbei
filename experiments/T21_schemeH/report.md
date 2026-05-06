# T21b Scheme H 291-d single seed (seed=42) LOSO h=10 — Report

## TL;DR

| Metric | T11 Scheme C 226d seed=42 | **T21 Scheme H 291d seed=42** | iter_002 (Scheme C) | iter_003 (5-seed ensemble) |
|--------|---------------------------|--------------------------------|---------------------|----------------------------|
| best (T,δ) | 0.55 / 0.15 | **0.55 / 0.15** | 0.55 / 0.10 | 0.55 / 0.10 |
| LOSO sum (h=10) | +21.715 | **+21.698** | +21.86 | +22.22 |
| diff vs iter_002 | -0.16 | **-0.16** | — | +0.36 |

Decision: **NO iter_004 build** (+21.70 < +22.5 threshold, also < iter_002 +21.86).

Adding 65 R20 Top-15-paper factors to the 226d Scheme C base gave **essentially zero
improvement on a single-seed model** (same number to 0.02 cum_pnl). The 5-seed ensemble
path that produced iter_003 (+22.22) remains the only configuration that beats iter_002.

But — feature importance shows R20 features *are* useful (10/30 top, 16/50 top, 0 dead).
The bottleneck is model-side variance, not feature quality.

---

## 1. LOSO 5-fold raw argmax (h=10, seed=42)

| Fold | held sym | n_active | accuracy | raw cum_pnl |
|------|----------|---------:|---------:|------------:|
| 0 | 0 | 74,075 | 0.293 | -5.7303 |
| 1 | 1 | 76,891 | 0.404 | -0.0486 |
| 2 | 2 | 38,352 | 0.631 | +2.8211 |
| 3 | 3 | 10,880 | 0.864 | +2.2272 |
| 4 | 4 | 63,996 | 0.479 | +6.6046 |
| **sum** | | 264,194 | — | **+5.8741** |

Compare T11 seed_42 raw argmax sum: +7.220. Schema H raw is slightly worse.

## 2. Threshold sweep (T_grid × δ_grid = 8 × 6 = 48)

Best: **T=0.55, δ=0.15 → sum=+21.6981**, 5/5 folds positive, mean_acc=0.7044, n_active=98,691.

Per-fold thresholded PnL: `[1.638, 5.183, 4.355, 1.735, 8.788]`.

The runner-ups (T=0.55, δ ∈ {0.0, 0.05, 0.10}) all sit at +21.6979 — virtually flat plateau.

### top-10 grid (h=10):

```
   T  delta  sum_cum_pnl   per-fold-pnl
0.55  0.15      21.6981   [1.638, 5.183, 4.355, 1.735, 8.788]
0.55  0.05      21.6979   [1.638, 5.183, 4.357, 1.733, 8.787]
0.55  0.10      21.6979   [1.638, 5.183, 4.357, 1.733, 8.787]
0.55  0.00      21.6979   [1.638, 5.183, 4.357, 1.733, 8.787]
0.55  0.20      21.5723   [1.638, 5.173, 4.323, 1.651, 8.788]
0.55  0.25      21.3656   [1.636, 5.174, 4.315, 1.496, 8.743]
0.50  0.00      20.6962
0.50  0.05      20.6874
0.50  0.10      20.6197
0.50  0.15      20.5064
```

## 3. Feature importance (mean over 5 fold boosters by gain)

10/30 of the top-30 features and 16/50 of the top-50 are R20 NEW. Zero dead features.

### Top-30 by gain (NEW = R20 Top-15 factor):

| rank | feature | gain | NEW? |
|---|---|---:|---|
|  1 | mb_intst | 271713 | |
|  2 | ma_intst | 251056 | |
|  3 | rv_pos_W50 | 240564 | **NEW** |
|  4 | bsize1 | 220210 | |
|  5 | rv_w20 | 205309 | |
|  6 | asize1 | 196691 | |
|  7 | book_slope_bid | 194011 | **NEW** |
|  8 | ewma_a0.5_mb_intst | 172298 | |
|  9 | ask_mean | 158030 | |
| 10 | rv_w50 | 134658 | |
| 11 | ewma_a0.5_la_intst | 113939 | |
| 12 | book_slope_ask | 99767 | **NEW** |
| 13 | ewma_a0.5_lb_intst | 98252 | |
| 14 | la_intst | 96580 | |
| 15 | pair_close_mid | 89639 | **NEW** |
| 16 | ewma_a0.5_ma_intst | 88628 | |
| 17 | pair_a1_wmp | 88588 | **NEW** |
| 18 | bv_W50 | 81113 | **NEW** |
| 19 | lb_intst | 79909 | |
| 20 | ewma_a0.3_lb_intst | 73299 | |
| 21 | ewma_a0.3_mb_intst | 69660 | |
| 22 | stoikov_micro_gap_mid | 69446 | **NEW** |
| 23 | ewma_a0.05_mb_intst | 67123 | |
| 24 | pair_wmp_mid | 61867 | **NEW** |
| 25 | pair_b1_wmp | 61352 | **NEW** |
| 26 | mlofi_W5_lvl1 | 59665 | |
| 27 | ewma_a0.3_ma_intst | 52704 | |
| 28 | lb_ind | 51834 | |
| 29 | bsize_mean | 51098 | |
| 30 | rv_neg_W50 | 50235 | **NEW** |

### Useful R20 buckets (rank ≤ 50):

- **Realised vol asymmetry**: `rv_pos_W50`(#3), `rv_neg_W50`(#30), `bv_W50`(#18) — strong
- **Book slope**: `book_slope_bid`(#7), `book_slope_ask`(#12) — strong
- **Pair ratios** (spread compression vs reference): `pair_close_mid`(#15), `pair_a1_wmp`(#17), `pair_wmp_mid`(#24), `pair_b1_wmp`(#25)
- **Stoikov micro gap**: `stoikov_micro_gap_mid`(#22)

### Weak R20 (rank > 100, 33 features — candidates to prune for T22):

- All deeper imb_lvl (`imb_lvl2`, 5, 6, 9, 10) — book imbalance at deeper levels redundant with lvl1
- All `ewma_ofi_total_a*` (0.05/0.1/0.3/0.5) — redundant with existing ewma_*_intst
- `pair_b{1,5}_b{5,10}`, `pair_a*_a*` — within-side pair ratios less informative than cross-side
- `tick_imb_W{10,20}`, `rvs_signed_W{20,50}`, `roll_spread_W{20,30,50}`, `gofi_W20`, `r_skew_W{10,20}`
- `stoikov_micro`(#240, gain=1279) — barely active

Overall pattern: **~half the R20 set added value, half was redundant noise. T22 candidate: drop the 33 rank-100+ R20 features, keeping a leaner 258-d feature set.**

## 4. Why didn't 65 new features lift the model?

- `iter_003` (5-seed ensemble of 226d Scheme C) reached +22.22 on the same OOF protocol;
  the gain from 226d → 291d is ~0 single-seed, so:
  - the R20 features are correlated with existing `ewma_*` / `mb_intst` family
  - LightGBM with `feature_fraction=0.8` already saturates information at a smaller feature set
- per-fold thresholded PnL is **almost identical** to T11 seed_42 (-0.02 in fold 0, -0.04 in fold 4, ±0.01 elsewhere) → not a significant difference, just sampling noise

## 5. What to do next (suggestions, not committed)

a) **T22 = lean Scheme H** — drop 33 weak R20 features → 258-d, retrain 5-seed ensemble.
   Hypothesis: less noise → ensemble could push past +22.22.

b) **Don't bother retraining single-seed Scheme H — it never beats iter_002**.

c) Best near-term path to >+22.5 is still **more model variance** (different algos: cross-algo
   ensemble of LGBM + XGB + CatBoost on Scheme C 226d) — already partially explored in T17/T18.

## Verdict

- LOSO sum +21.6981 single-seed Scheme H 291d is **virtually identical** to single-seed
  Scheme C 226d (+21.715), so 65 R20 features alone don't move the needle.
- 10/30 R20 features rank in top 30 → R20 idea is sound, but ensemble (not features) is the
  gating factor for >+22.
- **No iter_004 built.** iter_003 (Scheme C 5-seed h_10 ensemble, +22.22) remains best.
