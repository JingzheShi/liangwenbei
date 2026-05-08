# T119 — iter_016 Ablation Submission

**Submission file**: `submission_050816_iter016_ablation_001.zip`

## Hypothesis

iter_016 v3 (T109) **+44.72** local LOSO but only **+25.08** platform.
iter_015 v1 **+40.13** local but **+28.16** platform.
**Local-platform reverse signal** = v3 over-tuned the local 442k 26d window.

The two main local-only tuning vectors added to v3 vs iter_015 were:

1. **T95 GRU** ensemble — cross-corr to GBDT preds = 0.34 (weakest of all components, suggesting it captures local-specific noise rather than the underlying signal).
2. **5-horizon DE thresh search** — every additional active horizon (h=5/10/20/40) introduces a new (thr_up, thr_dn) pair tuned on the local 442k. Platform short horizons came in -3 to -5.
3. **h=60 thresh re-tuned aggressively** — v3 went from iter_015 v1's (3.58e-4, 2.16e-4) to v3's (2.88e-4, 2.19e-4), pushing active rate up.

T119 ablates all three.

## Changes vs T109 v3

| Component | v3 (T109) | T119 ablation |
|---|---|---|
| h=60 stack | NN + CB_Huber + LGB_Huber + **GRU** (4-way) | NN + CB_Huber + LGB_Huber (3-way, GRU dropped) |
| h=60 weights | 1.0 / 0.7 / 1.0 / 1.0 | 1.0 / 0.7 / 1.0 (same non-GRU ratio) |
| h=60 thr_up | 2.883e-4 (aggressive DE) | **4.21e-4** (iter_015 v1-style conservative) |
| h=60 thr_dn | 2.186e-4 | **1.86e-4** |
| h=5/10/20/40 | active, DE-tuned thresh | **disabled** (always action=1 / flat) |
| `requirements.txt` | includes `torch==2.5.1` | torch removed |

## Local 442k full-test verification

```
TOTAL cum_pnl = +39.9761  n_active=220,916/442,080  active_rate=49.97%
thr_up=4.21e-4  thr_dn=1.86e-4
weights = (NN=1.0, CB_Huber=0.7, LGB_Huber=1.0)

per-sym:
  sym=0: +1.6703   (n_act 41,497/88,416)
  sym=1: +4.7654   (n_act 45,710/88,416)
  sym=2: +4.1081   (n_act 45,049/88,416)
  sym=3: +12.4448  (n_act 41,366/88,416)
  sym=4: +16.9874  (n_act 47,294/88,416)
sum_per_sym (LOSO-equiv) = +39.9761
```

## Comparison

| Baseline | Local LOSO | Platform | Δ T119 (local) |
|---|---|---|---|
| iter_014 | +38.28 | +25.86 | +1.70 |
| iter_015 v1 | +40.13 | **+28.16** | -0.15 |
| iter_015 v2 (T106) | +43.20 | n/a | -3.22 |
| **iter_016 v3 (T109)** | **+44.72** | **+25.08** | -4.74 |
| **T119 ablation** | **+39.98** | **(submitted)** | — |

T119 sits **almost exactly at iter_015 v1 local performance**. Crucially, the
underlying h=60 stack (NN + CB_Huber + LGB_Huber) is the T99-winner non-GRU
3-way, which is structurally stronger than iter_015 v1's NN + LGB_T75 2-way —
but the conservative thresh and the absence of GRU/short-horizon over-tuning
should keep platform performance close to iter_015 v1 (+28.16) or higher.

## Pkg integrity checks

- **Smoke** (synthetic 100×154 frames, 3 batches): runs, returns (3, 5) actions, all flat as expected for low-magnitude inputs.
- **End-to-end** (4096 real LOB top5 windows + synthetic non-LOB cols):
  - `_h_short` active = `[]` (all 4 short horizons disabled)
  - `_h_long` active = True
  - h=5/10/20/40 all 4096 flat ✓ (assert)
  - h=60 short=1979, flat=1050, long=1067
  - stateless ✓ (call#2 byte-identical to call#1)
  - shuffle-invariant ✓ (permutation + invert == original)
  - timing: 0.65 ms/sample → 442k = **285s** (well under 3h limit, much faster than v3 which had GRU at ~600s)

## CRITICAL_CONSTRAINTS compliance

- ✓ sym / date never enter feature vector or any model input
- ✓ Predictor is stateless across calls (no cross-batch state held on `self` beyond loaded models)
- ✓ Every component is sym-agnostic (no sym embedding, no sym-specific norms)
- ✓ requirements.txt no longer includes torch (not needed without GRU)
- ✓ Inference time 285s ≪ 3h budget

## Files

```
pkg/
  Predictor.py           — 3-way stack, GRU classes removed
  config.json            — unchanged 154-col feature schema
  thresholds.json        — h=60 only active, 3-way no_gru, conservative thresh
  requirements.txt       — torch removed (5 packages)
  fast_features.py       — unchanged
  fast_features_batch.py — unchanged
  nn_h60_seed*.npz       (5 files)  — T87 SPO+ NN
  model_cb_huber_seed*.cbm (5 files) — T99 CB Huber α=1e-3
  model_T99_huber_a0.001_seed*.txt (5 files) — T99 LGB Huber α=1e-3
```

Total pkg ≈ 26 MB (vs T109 v3 ≈ 102 MB pre-cleanup). Short-horizon
models removed since `active=false` means Predictor never loads them
(saves ~76 MB).
