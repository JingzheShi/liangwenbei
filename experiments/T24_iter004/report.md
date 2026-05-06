# T24 — iter_004 (a/b) — short-horizon threshold tightening

## Motivation

iter_002 platform results (task 2435) revealed a hard calibration gap:

| label | platform | per-trade | LOSO sum | calibration |
|---|---|---|---|---|
| label_5  | -7.68 | -0.000099 (~ -fee) | +17.49 | severely overoptimistic |
| label_10 | -8.64 | -0.000105 (~ -fee) | +21.86 | severely overoptimistic |
| label_20 | -5.09 | -0.000053          | +19.70 | overoptimistic |
| label_40 | +2.02 | +0.000034          | +11.71 | OK |
| label_60 | **+4.07** | +0.000117      | +6.30  | best (calibrated) |
| best of 5 | **+4.07 (h_60)** | — | — | — |

Reading: short horizons trade so often (~80-130K active) at iter_002
thresholds that per-trade PnL collapses to ≈ -fee on platform, i.e. the
model has near-zero edge at high-frequency thresholds in OOD evaluation.
Long horizons (40 / 60) survive with a real per-trade > 0 because they
trade less and signal-to-noise is higher.

The reverse-rank between LOSO and platform on short horizons (LOSO best
came from h_10 = +21.9 → platform −8.6) means the LOSO ranking we used
to pick thresholds in iter_002 is not predictive of platform PnL on
short horizons. We can not trust LOSO sum as a target there.

## What we tried

Two sibling submissions, no retraining (same 5 LightGBM model files
from iter_002 / T5b). Only `thresholds.json` differs.

### iter_004a — "safe": floor short horizons at 0

`thresholds.json` marks `h_5`, `h_10`, `h_20` as `active: false`. The
existing Predictor.predict path leaves columns at 1 (no trade) when a
horizon is inactive, which gives 0 platform PnL on those horizons.

Expected platform:
- h_5/10/20: 0 (no trades emitted)
- h_40: +2.02 (same as iter_002, threshold unchanged)
- h_60: +4.07 (same as iter_002, threshold unchanged)
- **best of 5: +4.07** — strict floor; equal to iter_002 best.

This is a one-way bet: it costs nothing relative to iter_002 best, and
removes the −5 to −9 tail on short horizons.

### iter_004b — "aggressive": only act on high-confidence short-h signals

Re-ran the per-horizon threshold sweep with a finer grid that extends
`T` up to 0.85 and `delta` up to 0.40 (`sweep_aggressive.py`), then for
each short horizon picked the (T, δ) with the highest **per-trade** PnL
that still has all 5 LOSO folds positive and ≥ 5K active trades:

| Horizon | iter_002 (T, δ) | iter_004b (T, δ) | LOSO sum | per-trade  | n_active | n_pos folds |
|---|---|---|---|---|---|---|
| h_5  | (0.60, 0.10) | **(0.85, 0.20)** | +7.55  | +0.000643 | 11.7K (vs 80.5K) | 5/5 |
| h_10 | (0.55, 0.10) | **(0.80, 0.20)** | +6.46  | +0.000702 |  9.2K (vs 97.1K) | 5/5 |
| h_20 | (0.50, 0.05) | **(0.70, 0.20)** | +6.71  | +0.000647 | 10.4K (vs 88.1K) | 5/5 |
| h_40 | (0.50, 0.00) | (0.50, 0.00)     | +11.71 | (iter_002) | (iter_002) | 5/5 |
| h_60 | (0.50, 0.20) | (0.50, 0.20)     | +6.30  | (iter_002) | (iter_002) | 3/5 |

Per-trade PnL on LOSO is now ~5-7× iter_002 thresholds — about +6e-4
versus +1e-4. Even if the LOSO → platform per-trade drift on short
horizons is the same magnitude (~ −7e-4) we saw in iter_002, the
post-drift platform per-trade should land roughly neutral instead of −1e-4.

Expected platform (best case):
- h_5/10/20: −1 to +2 each (vs −5 to −9 in iter_002)
- h_40: +2.02
- h_60: +4.07
- **best of 5: +4.07 floor; potential +5 to +6 if a tightened short
  horizon clears +1**

Risk: iter_002's per-trade drift might not actually be a constant offset.
If platform OOD also wipes the ~+6e-4 per-trade we see in LOSO, short
horizons stay slightly negative, but each is small in magnitude (~−1)
because n_active is now 10-12K not 80-100K. Worst case best of 5 is
still +4.07 from h_60, same as iter_002 / iter_004a.

### iter_004c — skipped

Submitting 5× h_60 (with h_60's threshold) is identical-in-floor to
iter_004a (+4.07) but strictly weaker on upside, so it adds no
information. Not built.

## Recommendation

**Submit iter_004a first** for the floor (+4.07). Then submit iter_004b
to test the aggressive-threshold hypothesis — if either short horizon
clears +1 on platform, best of 5 will exceed +4.07 and we have a new
ceiling. If iter_004b regresses, we have learned that LOSO per-trade
also doesn't transfer to platform (even after 6× tightening), which is
useful for iteration design.

## Validation

Both packages:
- pass `sanity_check.py`: 22/22 (no warnings)
- import + smoke-test Predictor cleanly
- defensive: order-shuffle invariant; sym/date not in features
- model files identical to iter_002 (sym-agnostic, stateless,
  date-independent), so all CRITICAL_CONSTRAINTS §1 holds

## Artifacts

- `experiments/T24_iter004/build_iter004a.py`
- `experiments/T24_iter004/build_iter004b.py`
- `experiments/T24_iter004/sweep_aggressive.py`
- `experiments/T24_iter004/sweep_aggressive_summary.json`
- `experiments/T24_iter004/sweep_aggressive_h{5,10,20,40,60}.csv`
- `submission/iter_004a_lgbm_schemeC_safe/`
- `submission/iter_004b_lgbm_schemeC_aggressive/`
- `submission_050622_iter004a_safe.zip`        (15.18 MB)
- `submission_050623_iter004b_aggressive.zip`  (15.18 MB)
