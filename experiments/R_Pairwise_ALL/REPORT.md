# R_Pairwise_ALL — All-pair price imbalances + train-set globals

**Date:** 2026-05-08
**Status:** ✅ Trick wins LOSO. **Recommend as iter_017 candidate.**
(Pivoted to local RTX 3090 — remote r7 ssh8.vast.ai:17438 was Connection refused.)

## TL;DR

| arm | n_features | LOSO (3-seed ensemble, DE 2D) | Δ vs baseline |
|---|---|---:|---:|
| baseline (schemeP, 359-d) | 359 | **39.82** | — |
| trick (schemeP + 30 pairs + 7 globals) | 396 | **42.15** | **+2.33** |

- **All 3 seeds positive**: Δ = {seed1: +2.33, seed7: +2.60, seed42: +2.03}, **mean +2.32, std 0.29**, min +2.03
- **Per-sym min Δ (ensemble): −0.030** (sym2; effectively zero — within noise)
- **Per-sym max Δ**: +0.948 (sym4); +0.71 sym0, +0.49 sym1, +0.22 sym3
- **LOSO criterion +0.5: PASSED** by ~5×; **per-sym min ≥ −0.5: PASSED** (−0.03 ≫ −0.5)

## Hypothesis tested (T114 #2)

HYD 1st-place team computed pairwise imbalances over **all 10 price levels** of LOB, not just adjacent levels. With 20 prices (bid1..bid10, ask1..ask10), C(20,2)=190 pairs. The schemeP 359-d already has midprices, spreads, depth-imbalance, cumspread, and dualz/qrank stats — but no explicit **deep-level price ratio** features. Hypothesis: deep-level pairs (e.g., bid9 vs bid10) capture LOB shape information not encoded in level-1 imbalance or aggregated cumspread.

Plus: train-set globals (mean/std/percentiles of mid-price across the entire training distribution) as constant-broadcast features. These act as priors LGB might use to localize splits.

## Method

### Step 1: Build all 190 pairwise imbalances (last tick of window)

For each 100-tick sliding window, compute at last tick:
```
pair_<a>_<b> = (price_a − price_b) / (price_a + price_b + 1e-12)
```
across all C(20,2) = 190 pairs of prices in {bid1..bid10, ask1..ask10}.

CRITICAL_CONSTRAINTS-safe:
- sym/date free
- single-tick computation (last tick of window) — stateless
- ratio scale-invariant — sym-agnostic

Cache: `cache/pairwise_{train,val,test}.npy` (1.47M × 190, 1.12 GB train).

### Step 2: 7 train-set global mid-price statistics

Computed once on full schemeP train mp_t (1.47M rows), broadcast as constants to every row:
```
g_mp_mean=−1.475e-3, g_mp_std=1.973e-2,
g_mp_p10=−2.380e-2, g_mp_p25=−1.333e-2, g_mp_p50=−2.864e-3,
g_mp_p75=+8.121e-3, g_mp_p90=+2.179e-2
```

### Step 3: Filter top-30 pairs by LGB gain importance

One LGB screening pass on (schemeP-359 + 190 pairs + 7 globals = 556-d), Huber α=1e-3, GPU, feature_fraction=1.0, 500 rounds early-stop 40, seed=0. Best iter=264.

**Importance distribution:**
| group | total gain | total splits |
|---|---:|---:|
| schemeP (359 feats) | 3.903 | 30,043 |
| pairs (190 feats) | 0.351 (~9.0% of total) | 3,221 |
| globals (7 feats) | **0.000** | **0** |

**Globals get zero gain** — broadcast constants cannot create a tree split (all data goes to one side, gain=0). They are kept in the trick anyway per spec, but contribute nothing. In future, replace with "broadcast train-percentile of *current row*" — i.e., a per-row rank vs train distribution — which would carry actual signal.

**Top-30 pairs by gain** (saved to `cache/top_pairs_names.txt`, see Appendix below).

### Step 4: Train baseline (359-d) and trick (396-d) — 3 seeds × 2 arms

T117/R3 recipe: Huber α=1e-3, GPU LightGBM, 600 rounds, early-stop 40, lr=0.05,
aug_a per-feature scale [0.8, 1.2] ratio=1.0, class-balanced sample weight,
V4 train/val (dates 0–75 / 76–79).

Per-seed: seed1 nl=127 ff=0.6 bf=0.7 l2=1.0; seed7 nl=63 ff=0.7 bf=0.85 l2=2.0; seed42 nl=127 ff=0.8 bf=0.8 l2=1.0.

### Step 5: DE 2D LOSO-equiv evaluation

Per-arm 3-seed mean(pred_dmid_norm), DE-optimized 2D threshold (T_up, T_dn) shared across all 5 syms (5 DE seeds {0,1,2,7,42} × 80 iter × pop24, sobol init), summed PnL across syms = LOSO-equivalent.

## Per-seed train metrics

| seed | arm | best_iter | val_corr | val_l1 | test_corr | EV-gate(k=1) PnL |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 328 | 0.1804 | 0.001521 | 0.1652 | 32.87 |
| 1 | trick    | 264 | 0.1813 | 0.001521 | **0.1670** | **36.95** |
| 7 | baseline | 338 | 0.1798 | 0.001518 | 0.1652 | 34.33 |
| 7 | trick    | 244 | 0.1764 | 0.001514 | **0.1686** | **38.00** |
| 42 | baseline | 251 | 0.1859 | 0.001515 | 0.1651 | 34.30 |
| 42 | trick    | 246 | 0.1796 | 0.001516 | 0.1655 | 35.19 |

Pattern: trick val_corr is mixed vs baseline (some up, some down) but **test_corr improves on 2/3 seeds and EV-gate cum_pnl on all 3**. Trick reaches best_iter ~50–80 rounds earlier — the 30 extra split candidates each iteration accelerate convergence without hurting generalization.

## DE thresholds and per-sym LOSO (3-seed ensemble)

| arm | T_up | T_dn | sym0 | sym1 | sym2 | sym3 | sym4 | LOSO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.539e-4 | 3.789e-4 | 4.485 | 5.681 | 4.139 | 11.177 | 14.343 | **39.824** |
| trick    | 3.154e-4 | 3.517e-4 | 5.190 | 6.168 | 4.109 | 11.392 | 15.291 | **42.150** |
| **delta** | | | **+0.705** | **+0.487** | −0.030 | +0.215 | **+0.948** | **+2.326** |

Trick learns lower (more aggressive) thresholds → **more trades, more profit**. 4/5 syms improve (sym0, sym1, sym3, sym4 all positive); sym2 marginal regression of −0.03 is well within seed noise.

## Per-seed delta (independent runs, no ensemble)

```
seed=1   baseline=38.8964  trick=41.2256  delta=+2.3292
seed=7   baseline=39.5633  trick=42.1615  delta=+2.5983
seed=42  baseline=39.1142  trick=41.1396  delta=+2.0254
mean delta=+2.3176   std=0.2866   min=+2.0254   max=+2.5983
```

3/3 seeds positive, very tight spread (std 0.29). Per-seed mean (+2.32) ≈ ensemble delta (+2.33), so the gain is **not** an ensembling artefact — it's a true feature signal.

## Where the signal comes from

The top-30 selected pairs cluster into 3 patterns:

1. **Deep-level same-side spreads** (most slots: ~24/30): e.g., `bid9_bid10`, `ask8_ask10`, `ask6_ask10`, `bid8_bid9`. These capture LOB-tail compression/expansion that single-level spreads (which schemeP has) miss. When tail price levels are spaced widely vs midprice, it signals slow markets / less informed flow.

2. **Cross-side spread anchors** (1–2 slots): `bid10_ask10` (rank 26), `bid9_ask1` (rank 22). These are deep-spread analogs.

3. **Mid-tail same-side** (~5 slots): `bid4_bid6`, `bid5_bid10`, `bid1_bid6`, `bid5_bid9`. Capture mid-of-book shape.

The 7 broadcast globals are a no-op for LGB (constant ⇒ zero gain). The +2.33 LOSO is **purely** from the top-30 pairs.

## Verdict

- **PASS**: +2.33 LOSO ≥ +0.5 threshold (4.6× over bar) → **iter_017 candidate**.
- **Robustness**: per-sym min Δ = −0.030 (sym2; basically zero). 4/5 syms strictly improve.
- **Seed stability**: all 3 seeds Δ ≥ +2.0; std 0.29 << mean 2.32.
- **Compared to R3 HYD-quartet (+1.42)** and **R4 lag-returns (+1.60)**: this trick is **stronger** in both LOSO delta and per-sym min stability.
- **Compared to T117/T99 baseline ladder**: schemeP baseline reproduces R3's 39.82 exactly, confirming protocol consistency.

## CRITICAL_CONSTRAINTS audit

- ✅ No `date` use (date placeholder is just a slicer in train/val split, never a feature)
- ✅ No `sym` embedding / sym-specific normalization (pairs are ratio-based, sym-agnostic)
- ✅ Stateless within-window pair computation (only last-tick of 100-tick window used)
- ✅ Globals are train-derived constants, not test-leaky

## Files

- `build_pairwise.py` — produces `cache/pairwise_{train,val,test}.npy` (190 dims) from raw parquets
- `build_globals.py` — computes 7 train-mp_t stats → `cache/globals_consts.npz`
- `screen_topk.py` — single-LGB screening, ranks 190 pairs by gain, picks top 30 → `cache/top_pairs_idx.npy`, `top_pairs_names.txt`, `screening_full.json`
- `train.py` — main 6-run trainer (3 seeds × {baseline, trick})
- `eval_de.py` — DE 2D ensemble + per-seed eval → `results.json`
- `model_RP_{baseline,trick}_seed{1,7,42}.txt` — saved boosters
- `pred_RP_{baseline,trick}_seed{1,7,42}.parquet` — test predictions
- `summary_RP_*.json` — per-train summaries
- `results.json` — final 3-seed eval

## Appendix: top-30 pairs (gain importance from screening LGB)

| rank | pair | gain |
|---:|---|---:|
| 1 | pair_bid9_bid10 | 0.00881 |
| 2 | pair_ask8_ask10 | 0.00767 |
| 3 | pair_ask6_ask10 | 0.00766 |
| 4 | pair_bid8_bid9 | 0.00634 |
| 5 | pair_ask5_ask7 | 0.00595 |
| 6 | pair_bid4_bid6 | 0.00519 |
| 7 | pair_bid7_bid9 | 0.00482 |
| 8 | pair_ask9_ask10 | 0.00466 |
| 9 | pair_bid8_bid10 | 0.00466 |
| 10 | pair_ask5_ask9 | 0.00440 |
| 11 | pair_bid7_bid10 | 0.00437 |
| 12 | pair_ask2_ask10 | 0.00433 |
| 13 | pair_ask7_ask10 | 0.00414 |
| 14 | pair_ask8_ask9 | 0.00412 |
| 15 | pair_ask7_ask9 | 0.00406 |
| 16 | pair_ask4_ask8 | 0.00387 |
| 17 | pair_ask7_ask8 | 0.00379 |
| 18 | pair_ask5_ask8 | 0.00360 |
| 19 | pair_ask6_ask7 | 0.00352 |
| 20 | pair_ask4_ask10 | 0.00344 |
| 21 | pair_ask3_ask5 | 0.00336 |
| 22 | pair_bid9_ask1 | 0.00333 |
| 23 | pair_ask5_ask10 | 0.00333 |
| 24 | pair_ask6_ask9 | 0.00332 |
| 25 | pair_ask1_ask10 | 0.00332 |
| 26 | pair_bid10_ask10 | 0.00328 |
| 27 | pair_bid5_bid10 | 0.00322 |
| 28 | pair_ask3_ask10 | 0.00317 |
| 29 | pair_bid1_bid6 | 0.00309 |
| 30 | pair_bid5_bid9 | 0.00301 |

20+/30 pairs are deep-level (≥3 levels apart), confirming the hypothesis that schemeP under-encodes LOB-tail shape. Ask-side pairs dominate (20/30) over bid-side (8/30), with 2 cross-side anchors.

RESULT: task=pairwise_all metrics={baseline=39.8243, trick=42.1496, delta=+2.3253, per_seed_delta_mean=+2.3176, per_seed_delta_std=0.2866, per_sym_min_delta=-0.0302, n_pairwise_kept=30, n_globals_kept=7_but_zero_gain, n_features_baseline=359, n_features_trick=396} notes=[All 3 seeds positive; 4/5 syms improve; deep-level same-side pairs (bid9-bid10, ask8-ask10) carry main signal; broadcast globals are no-op for LGB.]
