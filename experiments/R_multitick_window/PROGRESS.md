# R_multitick_window — Multi-tick LOB snapshots ABORT

## Hypothesis
T100 W3 "cheap fix" said: schemeP 359-d uses last-tick (t=-1) raw LOB only;
the W=100 window middle structure (e.g. t=-50) is unused. Adding raw
snapshots at t∈{-5,-20,-50,-90} should add +1~+3 LOSO over T75 L2 baseline.

## Setup
- 4 lookbacks × 20 features (bid1-5, bsize1-5, ask1-5, asize1-5) = +80 features
- baseline = schemeP 359-d, trick = 439-d
- 3-seed (1, 7, 42), LGB regression_l2 GPU, num_boost=600, ES=40 on val MSE
- V4 walk-forward (train 0-75, val 76-79, test 96-119)
- DE 2D + DE 4D session thresh, LOSO-equiv (sum per-sym)

## Result (3-seed avg ensemble)

| metric | baseline | trick | Δ |
|---|---|---|---|
| DE 2D LOSO | +36.52 | +35.80 | **-0.72** |
| DE 4D LOSO | +38.05 | +36.90 | **-1.15** |
| Test corr  | 0.1567 | 0.1533 | -0.0034 |
| best_iter avg | 88.3 | 90.3 | +2.0 |

**Per-sym Δ (4D):** sym0=-0.24, sym1=-0.22, sym2=+0.22, sym3=+0.29, sym4=**-1.19**

3 of 5 syms regress; sym 4 (biggest baseline contributor at +12) loses 1.19.

## Diagnostic — Feature importance (gain rank)
Top-50 features per seed contain **only 1 multi-tick feature**, ranked 46-49.
The model essentially ignores the new dimensions.

| seed | top-50 multi-tick count | best multi-tick rank |
|---|---|---|
| 1  | 1 | 46 (`bid4_lb90`) |
| 7  | 1 | 48 (`ask5_lb90`)  |
| 42 | 1 | 49 (`ask5_lb90`)  |

## Why it failed
schemeP 359-d already has rolling W=20/50/100 stats (mean, std, skew, kurt
of bid/ask/sizes), plus signed_bv_W100, OFI tox lvl1/5, EWMA, qrank,
rv_w50, etc. These richer aggregates already encode the intra-window
structure that raw lookback snapshots would have captured (and more
robustly — single-tick LOB values are noisy).

Multi-tick raw snapshots are therefore **redundant** to existing aggregates
and contribute mostly noise → mild generalization regression (-0.0034 corr,
-1.15 LOSO-4D).

## Verdict
**ABORT.** NOT iter_018 candidate. T123 iter_017 (+44.72 LOSO-equiv,
HYD+monotone+date-decay+SPO+T75-LGB-L2) remains leader.

## Lesson for the broader rethink
The "schemeP only uses last-tick raw" framing is misleading. Even though
the 0-44 prefix names look like last-tick raw, the rest of the 359 dims
are heavy on rolling-window summaries that already span the W=100 window.
"Multi-tick snapshots" doesn't add new info on top of that.

If a future trick wants to exploit intra-window structure, it should be
something that **summary stats can't capture**, e.g.:
- Cross-tick contraction patterns (e.g. fast spread tightening velocity)
- Toxicity ramp shapes (not just rolling mean of OFI but the SHAPE)
- Phase transitions (regime changes within the 100-tick window)
- Order-book TICK pattern (e.g. "5 cancels in last 5 ticks of L1")

Plain raw value at t=-5 is too redundant.
