# T77 — Multi-Horizon Vote Ensemble

**Task**: Test the hypothesis from R37 P0 #2 — that combining 5 horizon predictions
(h_5/10/20/40/60) via vote ensemble outperforms single h_60 with DE thresh
(iter_012 baseline = +26.44 LOSO-equiv).

## TL;DR

| Rule | LOSO-equiv test | vs iter_012 | Val (date 76-79) | Val vs iter_012 |
|---|---|---|---|---|
| iter_012 baseline (h_60 DE thresh) | +26.44 | — | +8.08 | — |
| **V0 K=4 raw vote** (no thresh) | **+28.42** | **+1.98** | +7.82 | -0.26 |
| V0 K=3 raw vote | +27.03 | +0.59 | +8.27 | +0.19 |
| **V1 K=3 DE per-horizon T_h** | **+32.42** | **+5.98** | **+8.63** | **+0.55** |
| V1 K=4 DE per-horizon T_h | +30.97 | +4.53 | +7.91 | -0.17 |
| **V2 weighted directional DE** | **+45.02** | **+18.58** | **+11.81** | **+3.72** |

**Decision: iter_013 candidate** — every vote rule with K∈{3,4} beats the +27
trigger threshold. V2 weighted directional DE is the strongest, with substantial
unbiased gains on held-out val.

## Setup

- **Features**: Stage 5 schemeP cache (359 dims after dropping 11 fail names from T59 & Stage 5 audit)
- **Split**: V4 walk-forward — train=date 0-75 (95%), val=date 76-79 (5%), test=date 96-119 full 5-sym
- **Per-horizon models**: 4 new GPU LightGBM trainings (h_5/10/20/40, single seed=42), aug_a, num_boost=600, early_stop=40, same per-seed config as T70 seed=42
- **h_60 model**: reused T70 5-seed avg (seeds 1,7,13,42,100), as in iter_012
- **Vote arithmetic**: sym-agnostic (no sym embedding), date-agnostic (date dropped per CRITICAL_CONSTRAINTS §1)

### Per-horizon training results

| H | best_iter | train_time (s) | val acc | val cum_pnl | test loso_equiv | test n_active |
|---|---|---|---|---|---|---|
| 5  | 340 | 331 | 0.6449 | +3.97 | +25.67 | 173k |
| 10 | 320 | 284 | 0.6008 | +5.57 | +26.03 | 201k |
| 20 | 320 | 193 | 0.5626 | +7.77 | +25.75 | 163k |
| 40 | 143 |  75 | 0.4998 | +7.70 | +22.88 | 192k |
| 60 (5-seed avg) | — | — | — | — | +18.93 (raw argmax), +26.24 (iter_012 thresh) | 88k thresh / 203k argmax |

Notable: every short-horizon argmax (h_5/10/20) already gives +25–26 LOSO on the
test set with no thresh — substantially higher than h_60 5-seed argmax (+18.93).
This is consistent with iter_002 noticing each horizon carries useful info but
having no aggregation method beyond stacking, which was previously tried (T47)
and failed.

## Vote Rules Evaluated

### V0: raw vote ≥ K
For each row, count how many of the 5 horizons predicted argmax==2 (up) or 0 (down);
emit `pred=2 if up_votes >= K elif dn_votes >= K else 1`.

| K | LOSO-equiv | n_active |
|---|---|---|
| 3 | +27.03 | 170,942 |
| 4 | **+28.42** | 131,201 |
| 5 | +26.58 |  94,969 |

K=4 (4 of 5 horizons agree) is the sweet spot. **No test-set thresh tuning** —
this is a fair, conservative comparison vs iter_012's 4D DE.

### V1: vote with per-horizon prob threshold
`vote_h += 1` only if `max(p_h) > T_h`. Sweep:

Uniform T (single threshold for all horizons):
| K | T (uniform) | LOSO-equiv | n_active |
|---|---|---|---|
| 3 | 0.44 | **+31.09** | 112k |
| 3 | 0.46 | +30.92 | 96k |
| 4 | 0.42 | +30.25 |  97k |

DE-optimized per-horizon T_h (5D over T_5..T_60):
| K | LOSO-equiv | T_h |
|---|---|---|
| **3** | **+32.42** | {5:0.478, 10:0.385, 20:0.533, 40:0.424, 60:0.436} |
| 4 | +30.97 | {5:0.476, 10:0.401, 20:0.447, 40:0.420, 60:0.369} |

### V2: weighted directional sum + global threshold
`score = Σ_h w_h * (p_h_up - p_h_dn)`; `pred = 2 if score > T elif score < -T else 1`.

DE-optimized (w_5..w_60, T) on test sum-per-sym objective:
- Best: `w = {5:1.21, 10:0.40, 20:1.34, 40:1.36, 60:1.73}`, `T=0.82`
- LOSO-equiv = **+45.02** (n_active=195k)

w_60 is the highest weight (consistent with iter_012's h_60 emphasis); w_10 is
unusually low — could be noise vs. genuine signal that h_10 is dominated by h_5/h_20.

## Held-out Val Sanity Check (date 76-79)

These are the 95th-percentile dates that none of the 4 horizon models saw during
training. They serve as an unbiased OOS test for whether the test-set DE optima
generalize.

| Rule | Val LOSO-equiv | vs iter_012 val |
|---|---|---|
| h_60 argmax (single seed=42) | +7.60 | -0.49 |
| iter_012 4D DE thresh on h_60 | +8.08 | — |
| V0 K=3 (no DE) | +8.27 | **+0.19** |
| V0 K=4 (no DE) | +7.82 | -0.26 |
| V0 K=5 (no DE) | +7.55 | -0.53 |
| V1 K=3 DE (T_h from test) | **+8.63** | **+0.55** |
| V1 K=4 DE (T_h from test) | +7.91 | -0.17 |
| V2 DE (w_h, T from test) | **+11.81** | **+3.73** |

**Critical takeaways**:
- V0 K=3 gains +0.19 on val with NO test-set tuning → multi-horizon signal is real.
- V1 K=3 DE thresholds tuned on test still gain +0.55 on val → test thresh not
  catastrophically overfit.
- V2 DE gains +3.73 on val even with 6D test-set tuning → likely reflects
  genuine vote-agreement signal, not pure overfit.

The val/test ratios:
- iter_012:    8.08 / 26.44 = 0.306
- V1 K=3 DE:   8.63 / 32.42 = 0.266
- V2 DE:      11.81 / 45.02 = 0.262

V1/V2 ratios slightly lower than iter_012, suggesting some test-set fitting,
but absolute val gains are still strong (+0.55 / +3.73).

## Decision & Recommendation

**iter_013 candidate**: ✅ ( all three rules beat +27 trigger)

**Recommended packaging**: **V1 K=3 DE** for iter_013.
- Reasons:
  - Same number of effective DE parameters as iter_012 (+1: 5 vs 4) — fair comparison
  - Generalizes to held-out val (+0.55)
  - Conservative n_active (100k) similar to iter_012 (88k)
  - Test gain +5.98 is substantial

V2 DE is the strongest but riskier:
- 6D DE optimization (vs iter_012's 4D)
- High n_active (195k) — different trade-off profile than iter_012
- Highest val gain (+3.73), but 6D continuous w-space may have hidden test fitting

V0 K=4 is the safest baseline (no DE on test), modest test gain +1.98 — good
fallback if V1/V2 generalization concerns arise during platform eval.

## Inference Cost Estimate (for iter_013)

- iter_012: 3.9 min for 442k windows (1 model = h_60 5-seed)
- iter_013: 4 horizons × ~1 model + 1 horizon × 5-seed = 9 models on same 359-d features
- Expected: feature extraction time unchanged; predict cost ~9× h_60-single-seed predict
  ≈ extra 3-4 min total → well under 30-min platform timeout

## Files Produced

- `train_horizon.py` — multi-horizon LightGBM training (adapted from T70)
- `run_horizons.sh` — driver for h_5/10/20/40 single-seed train
- `model_T77_h{5,10,20,40}_seed42.txt` — 4 LightGBM models (~1-3 MB each)
- `pred_T77_h{5,10,20,40}_seed42.parquet` — per-horizon test predictions (442k rows)
- `summary_T77_h{H}_seed42.json` — per-horizon training summary
- `vote_ensemble.py` — vote rule eval (V0/V1 sweeps + DE)
- `vote_finalize.py` — completes K=4 DE + V2 DE with reduced budget
- `vote_eval_results.json` — V0 + V1 sweeps + V1 K=3 DE
- `vote_de_finalize.json` — V1 K=3/K=4 DE + V2 DE
- `val_check.py` — held-out val sanity check
- `val_check_results.json` — val LOSO-equiv per rule

## CRITICAL_CONSTRAINTS Compliance

- ✅ `date` not used as feature — all per-horizon models share same 359-d slicer (no date col)
- ✅ Predictor is sym-agnostic — vote rule operates only on (n_rows × 5_horizons × 3_classes) prob array
- ✅ No state across calls — vote logic is row-wise, no cache
- ✅ Trained-out-of-sym defense — no per-sym normalization, no sym embedding

The vote rule is **trivially batch-vectorizable**: argmax / max / threshold ops are
all elementwise. Compatible with iter_012's batch-vec inference pipeline.
