# T123 / iter_017 v1 Package Report

**Stack:** `5×T87 SPO+ NN + 5×T123 LGB L2 (HYD + monotone + date-decay)`
**Submission file:** `submission_050817_iter017_v1.zip` (5.4 MB, 16 files)
**Date:** 2026-05-08

## Recipe

| Component | Variant | Source |
|---|---|---|
| NN | T87 SPO+ DFL (5 seeds, λ=30, lr=3e-5) | `experiments/T87_spo_dfl/iter015_pkg/nn_h60_seed*.npz` (reused, identical to iter_015 v1) |
| LGB L2 | T123 (HYD 16 dims + monotone 18 feats + date-decay sw) | newly trained 5 seeds (1, 7, 13, 42, 100) |
| Ensemble | `(w_nn·pred_nn + w_lgb·pred_lgb) / (w_nn + w_lgb)`, w_nn=1.0, w_lgb=1.5 | iter_015 v1 ratio |
| Decision | Asymmetric EV gate, thr_up=0.000421, thr_dn=0.000186 | **iter_015 v1 conservative — NOT retuned for iter_017** |

## Three Wins Applied to T75 LGB L2 Base

1. **HYD interaction quartet (16 dims)** — `pp, mu, dp, sdr` × `{last, W5, W20, W50}`
   - `pp = (bsize1 - asize1) · (ask1 - bid1)` (price pressure)
   - `mu = (ask1 - bid1) · liq_imb` (market urgency)
   - `dp = (totalasize - totalbsize) · (avgask - ask1)` (depth pressure)
   - `sdr = (ask1 - bid1) / (totalbsize + totalasize)` (spread/depth ratio)
   - Computed at runtime from the 100×154 raw window in `Predictor._compute_batch_features` (causal, single-window).

2. **Monotone constraints** on 18 features (T117 list, `monotone_constraints_method='advanced'`)
   - +1: `imbalance, wmp_balance_12, lb_intst, mb_intst, ewma_ofi_a*_lvl1, signed_rv_W*, signed_bv_W*`
   - -1: `la_intst, ma_intst, cb_intst`
   - +1: `ca_intst` (sign matches "more cancel-ask → mid up")

3. **Date-decay sample weight** — `sw = sw_class_balanced × (1.0 + 1.0 · date / 79)`
   - Linear weight increase from 1.00 (date 0) to 1.949 (date 76, last train date).
   - Applied during training only — `date` does NOT enter the model as a feature.
   - Mean combined sw ≈ 1.47 on the 1.4M train+aug rows.

## Per-Seed Training Results (T123 LGB only, regression_l2 with all 3 wins)

| seed | best_iter | val MSE | val corr | TEST corr | EV-gate k=1 cum_pnl | per_sym (sym=0..4) |
|---|---|---|---|---|---|---|
| 1   | 93 | 5.04e-6 | 0.156 | 0.162 | +33.08 | [4.99, 3.19, 3.58, 8.13, 13.19] |
| 7   | 78 | 5.04e-6 | 0.155 | 0.162 | +33.59 | [4.25, 3.43, 3.39, 10.14, 12.38] |
| 13  | 79 | 5.06e-6 | 0.147 | 0.158 | +31.15 | [4.50, 3.25, 2.41, 7.28, 13.70] |
| 42  | 89 | 5.04e-6 | 0.151 | 0.151 | +28.77 | [3.98, 2.22, 2.84, 8.18, 11.55] |
| 100 | 75 | 5.05e-6 | 0.158 | 0.162 | +32.57 | [4.54, 2.90, 3.60, 8.40, 13.12] |

Each model trained on 1.4M (1× aug) rows in 20-40s on RTX 3090, 375 input dims.

## Local 442k Verification (full_test_verify.py)

| Stack | thr | cum_pnl | per_sym (0,1,2,3,4) | Δ vs iter_015 v1 |
|---|---|---|---|---|
| iter_015 v1 (5 NN + 5 T75) DE-LOSO opt | 3.58e-4 / 2.16e-4 | **+40.13** (published) | [4.31, 4.78, 4.86, 11.89, 15.88] | — |
| iter_015 v1 same conservative thr  | 4.21e-4 / 1.86e-4 | +39.08 | [2.98, 5.59, 4.66, 11.93, 13.92] | — |
| **iter_017 v1** (5 NN + 5 T123)        | **4.21e-4 / 1.86e-4** | **+40.70** | **[3.45, 5.06, 4.86, 11.78, 15.55]** | **+0.57 vs DE / +1.62 vs same thr** |
| iter_017 v1 DE-LOSO opt (informational) | 3.43e-4 / 2.17e-4 | +41.72 | [4.31, 4.78, 4.86, 11.89, 15.88] | +1.59 |

Per-sym min for iter_017 v1 at the conservative threshold = +3.45 (sym=0), all positive.

## Compliance Checks

- `Predictor.py` smoke test: ✅
- `end_to_end_check.py` (10 files × 8 test points): ✅
  - shuffle invariance: predictions are identical after shuffling input order
  - sym=99 injection: no crash; predictions identical to sym=0 inputs (sym not used)
- `date` and `sym` not in feature list (lgb feat_names assertion in train script)
- HYD computation is causal (only uses `[t-99..t]`) and stateless

## Files in submission_050817_iter017_v1.zip (5.4 MB)

```
Predictor.py              13 KB   (numpy NN inference + lightgbm LGB + runtime HYD computation)
fast_features.py          22 KB   (single-window reference; not actually called)
fast_features_batch.py    28 KB   (216-d schemeP extras, batch-vectorized)
config.json                3 KB   (154 raw feature names + 5 horizon labels)
thresholds.json          0.9 KB   (h=60 active, w_nn=1.0/w_lgb=1.5, thr_up=4.21e-4/thr_dn=1.86e-4)
requirements.txt        0.06 KB   (numpy 2.4.4, pandas 2.3.3, lightgbm 4.6.0, scipy 1.17.1)
model_h60_seed{1,7,13,42,100}.txt    1.1-2.4 MB each   (T123 LGB L2 + HYD + monotone + date-decay)
nn_h60_seed{1,7,13,42,100}.npz       0.55 MB each      (T87 SPO+ DFL NN, copied from iter_015)
```

## Next Steps / Considerations

- **Submission reuse risk**: The 5 NN .npz are byte-identical copies from iter_015 v1's `experiments/T87_spo_dfl/iter015_pkg/`. No retraining of NN.
- **Threshold caution**: Used iter_015 v1's conservative (4.21e-4, 1.86e-4) per task spec. DE-LOSO optimum on iter_017 v1 stack would yield +41.72 (vs +40.70), but retuning thresholds adds platform overfit risk that was specifically called out in the task.
- **HYD runtime cost**: ~negligible. HYD ops are pure numpy on the (N, 100, 7) raw cols slice. ~100µs per batch of 1024.
