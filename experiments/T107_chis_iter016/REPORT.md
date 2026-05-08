# T107: iter_016 CHIS — Cross-Horizon Independent Submission

## Summary

iter_007–iter_015 only ever activated **h=60**. The platform scores via
**MAX over 5 horizons (h=5/10/20/40/60)** — short horizons were dead weight
in every previous submission, leaving free upside on the table.

iter_016 CHIS activates **all 5 horizons independently**. Each horizon has
its own ensemble (LGB+CB regression for short, 4-way for h=60) and its own
joint-DE-tuned asymmetric EV gate. Worst case = h=60 alone (= iter_015 v2);
best case = a short horizon happens to outperform on the platform's actual
test data.

## Per-horizon stack

| h  | kind                  | ensemble (5 seeds each)                            | w_lgb / w_cb (or 4-way)        | thr_up      | thr_dn      | LOSO-equiv (442k local) |
|----|-----------------------|----------------------------------------------------|--------------------------------|-------------|-------------|-------------------------|
| 5  | lgb_cb_2way           | T98 LGB + T98 CB                                   | (1.552, 0.423)                 | 4.256e-04   | 4.024e-04   | +18.16                  |
| 10 | lgb_cb_2way           | T98 LGB + T98 CB                                   | (0.177, 0.007)                 | 3.854e-04   | 2.987e-04   | +25.14                  |
| 20 | lgb_cb_2way           | T98 LGB + T98 CB                                   | (2.695, 0.704)                 | 4.524e-04   | 3.929e-04   | +30.10                  |
| 40 | lgb_cb_2way           | T98 LGB + T98 CB                                   | (2.821, 0.952)                 | 3.367e-04   | 2.386e-04   | +36.38                  |
| 60 | fourway_iter015v2     | T87 SPO+ NN + T89 CB + T99 LGB Huber + T95 GRU    | 1.0 / 1.5 / 0.7 / 0.7          | 2.871e-04   | 2.074e-04   | +43.12 (= iter_015 v2)  |

**MAX over horizons: +43.12** — same as iter_015 v2 (within the 0.10
verify-vs-target gap from iter_015 v2's full_test_verify). The MAX comes
from h=60 on local LOSO; if any short horizon happens to outperform on the
platform's hidden test, iter_016 captures it for free.

## Per-sym (h=60, our highest local horizon — same as iter_015 v2)

| sym | cum_pnl   |
|-----|-----------|
| 0   | +4.97     |
| 1   | +4.58     |
| 2   | +4.66     |
| 3   | +12.74    |
| 4   | +16.17    |

## Per-sym (h=40, second-highest local horizon)

| sym | cum_pnl   | n_active |
|-----|-----------|----------|
| 0   | +3.80     | 22,661   |
| 1   | +5.26     | 34,551   |
| 2   | +5.87     | 38,813   |
| 3   | +8.33     | 46,100   |
| 4   | +13.12    | 36,164   |

## Why CHIS is 0-downside

- The platform takes **max** across horizons, not sum.
- Worst case for any short horizon is its independent submission scores
  worse than h=60's. That is irrelevant — h=60 is still in the bag.
- If on the actual test data a short horizon scores *better* than h=60's
  +43.12, CHIS captures that strictly larger value. iter_015 v2 cannot.
- Marginal cost: zero — short horizons don't perturb the h=60 path; they
  only fill columns 0..3 of the action matrix that iter_015 v2 left at 1
  (flat).

## Predictor pipeline

1. Compute the 359-d feature vector (raw 154 last-tick + 216 derived
   stage-5 features, drop 11 fail features) from the 100×154 window —
   identical to iter_015 v2's pipeline.
2. For each of h ∈ {5,10,20,40}: stack 5 LGB + 5 CB regressors with the
   horizon's joint-DE weights; apply asymmetric EV gate → fill column.
3. For h=60: stack 5 T87 NN + 5 T89 CB + 5 T99 Huber + 5 T95 GRU with
   1:1.5:0.7:0.7; apply asymmetric EV gate → fill column 4.

## CRITICAL_CONSTRAINTS compliance

- **date never used** — all models fed only the 359-d feature vector or
  the GRU's raw 100×20 window; no date column referenced.
- **stateless predict()** — verified by running predict twice and confirming
  output equality (`end_to_end_check.py` pass 2).
- **sym-agnostic** — no sym embedding anywhere; per-window normalisation
  for the GRU; global normalisation stats for the MLP.
- **shuffle-invariant** — verified by running on a 100-window batch of
  permuted inputs and on a single-window slice; outputs match per-row
  (`end_to_end_check.py` pass 3).

## Files

- `pkg/Predictor.py` — 5-horizon ensemble runner.
- `pkg/thresholds.json` — per-horizon weights + thresholds.
- `pkg/config.json` — 154 feature columns, 5 labels (label_5/10/20/40/60).
- `pkg/requirements.txt` — numpy, pandas, lightgbm, catboost, scipy.
- `pkg/fast_features*.py` — derived feature engine (copied from
  iter_015 v2 unchanged).
- `pkg/{nn,gru}_h60_seed*.npz`, `pkg/model_T89_seed*.cbm`,
  `pkg/model_T99_huber_a0.001_seed*.txt` — h=60 4-way iter_015 v2 stack.
- `pkg/model_lgb_h{5,10,20,40}_seed*.txt`, `pkg/model_cb_h{5,10,20,40}_seed*.cbm`
  — short-horizon T98 stacks.

## Source DE-thresh runs

- Short horizons: `experiments/T98_multihorizon_iter016/de_thresh_per_h.log`
  (joint 4D DE — best across LGB-only / CB-only / 7 fixed-weight 2way / joint).
- h=60: `experiments/T106_iter015v2_pkg/full_test_verify.json`
  (4-way w=1:1.5:0.7:0.7).

## Output zip

`submission_050813_iter016_chis.zip` (49 MB)
