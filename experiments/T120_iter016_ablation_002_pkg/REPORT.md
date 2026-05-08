# T120 — iter_016 Ablation Submission 002

**Submission file**: `submission_050816_iter016_ablation_002.zip` (~12.8 MB)

## Strategy

User can submit at any time (no 12h cooldown), so we run a strategic A/B
test against T119 (ablation_001). The single delta vs T119:

> **Replace T99 baseline LGB Huber 5-seed ensemble** with **T117/T120 monotone-constrained
> LGB Huber 5-seed ensemble.**

Everything else identical to T119: same NN ensemble, same CB Huber ensemble,
same weights `(NN=1.0, CB_Huber=0.7, LGB_Huber=1.0)`, same conservative
thresholds `thr_up=4.21e-4 / thr_dn=1.86e-4` (no DE re-search), short
horizons disabled.

## Why monotone constraints

T117 added LightGBM `monotone_constraints={ +1 / -1 }` on 18 microstructure
features whose signed direction is dictated by Cont-2014 / standard
order-flow theory:

| Sign | Features |
|---|---|
| +1 (up-monotone) | `imbalance`, `wmp_balance_12`, `lb_intst`, `mb_intst`, `ca_intst`, `ewma_ofi_a{0.05,0.1,0.3,0.5}_lvl1`, `signed_rv_W{20,50,100}`, `signed_bv_W{20,50,100}` |
| -1 (down-monotone) | `la_intst`, `ma_intst`, `cb_intst` |

Trained with `monotone_constraints_method='advanced'` for tightest enforcement.
Identical training setup to T99 (Huber objective α=1e-3, GPU LightGBM, V4
walk-forward split dates 0-75 train / 76-79 val, augmentation, 5-seed
configs).

Standalone T117 ablation showed **+1.87 LOSO** (3-seed) — best single trick
in the iter_015 → iter_016 audit.

## Models used (5 seeds: 1, 7, 13, 42, 100)

| Stream | T119 (ablation_001) | T120 (ablation_002) |
|---|---|---|
| NN h60 | T87 SPO+ MLP `nn_h60_seed*.npz` | (unchanged) |
| CB h60 | T99 CB Huber α=1e-3 `model_cb_huber_seed*.cbm` | (unchanged) |
| LGB h60 | T99 LGB Huber α=1e-3 `model_T99_huber_a0.001_seed*.txt` | **T120 LGB Huber α=1e-3 monotone `model_T120_huber_mono_seed*.txt`** |

Seeds 1, 7, 42 of T120 reuse the T117 monotone training run. Seeds 13, 100
were freshly trained for T120 (`train_t117.py --seed {13,100} --horizon 60
--alpha 1e-3`, ~60-90s each on RTX 3090).

## Local 442k full-test verification

```
TOTAL cum_pnl = +40.7538  n_active=219,814/442,080  active_rate=49.72%
thr_up=4.21e-4  thr_dn=1.86e-4
weights = (NN=1.0, CB_Huber=0.7, LGB_Huber=1.0)

per-sym:
  sym=0: +1.8488   (n_act 40,938)
  sym=1: +4.9851   (n_act 45,657)
  sym=2: +4.2304   (n_act 44,982)
  sym=3: +12.5740  (n_act 40,944)
  sym=4: +17.1154  (n_act 47,293)
sum_per_sym (LOSO-equiv) = +40.7538
```

## Comparison

| Submission | Local LOSO | Platform |
|---|---|---|
| iter_015 v1 (packaged) | +40.13 | **+28.16** |
| **T119 ablation_001** | +39.98 | (just submitted) |
| **T120 ablation_002 (this)** | **+40.75** | (to submit) |
| T109 v3 | +44.72 | +25.08 |

T120 = T119 **+0.78 local LOSO** lift, all of which comes from swapping
the LGB Huber stream. Active count nearly identical (219,814 vs 220,916).
Per-sym lift comes mostly from sym=0,3,4.

The ratio +0.78 / 5-seed ensemble vs +1.87 standalone-LGB-only lift is
expected — the LGB Huber stream is only 1/2.7 ≈ 37% of the h=60 weighted
average in the 3-way stack, so a monotone-only +1.87 lift gets diluted to
roughly +0.7 in the ensemble — matching observation.

## Pkg integrity (T120 pkg)

End-to-end (4096 real LOB top5 + synthetic non-LOB):
- `_h_short` active = `[]` (all 4 short horizons disabled, asserted all-flat ✓)
- `_h_long` active = True
- h=5/10/20/40 all 4096 flat ✓
- h=60 short=2008, flat=1013, long=1075
- **stateless** ✓ (call#2 byte-identical to call#1)
- **shuffle-invariant** ✓ (permutation + invert == original)
- timing 1.54 ms/sample → 442k = 681s (≪ 3h budget)

Smoke (synthetic 100×154 frames, 3 batches): runs, returns (3, 5) actions,
all flat as expected for low-magnitude inputs.

## CRITICAL_CONSTRAINTS compliance

- ✓ `sym` / `date` never enter feature vector or any model input
- ✓ Predictor stateless across calls (no cross-batch state)
- ✓ Every component sym-agnostic (no sym embedding / sym-specific norm)
- ✓ requirements.txt has no torch (5 packages)

## What this A/B test isolates

Since T120 = T119 with **only** the LGB-Huber stream swapped (same NN, same
CB, same weights, same thresholds), the platform Δ between submissions
**directly measures whether monotone constraints transmit from local to
platform**. Three possible outcomes:

| Platform Δ vs T119 | Interpretation |
|---|---|
| Δ ≈ +0.7 (matches local) | Monotone is robust signal — bake into iter_017 |
| 0 < Δ < +0.7 | Partial transmission — keep but watch other tricks |
| Δ ≤ 0 | Monotone over-fit local 442k — drop or re-think |

## Files

```
pkg/  (12.8 MB zip / 26 MB unzipped, 21 files)
  Predictor.py            — h60 LGB stream now loads model_T120_huber_mono_seed*.txt
  config.json             — unchanged 154-col feature schema
  thresholds.json         — h60 only active, conservative thr (unchanged from T119)
  requirements.txt        — 5 packages (no torch)
  fast_features.py        — unchanged
  fast_features_batch.py  — unchanged
  nn_h60_seed*.npz                  (5)  — T87 SPO+ NN              (unchanged)
  model_cb_huber_seed*.cbm          (5)  — T99 CB Huber α=1e-3      (unchanged)
  model_T120_huber_mono_seed*.txt   (5)  — T120 LGB Huber monotone  (NEW)
```
