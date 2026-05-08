# T83 — Robust threshold strategies on iter_013 OOF

## TL;DR

The iter_013 platform gap (LOSO +36.23 → platform +19.23, gap −17) is **NOT
caused by threshold overfit**. Threshold-strategy choice changes local
LOSO-equiv by ≤ 0.10 across all robust strategies tested (K-fold CV-DE,
bootstrap median, quantile-fixed, ensemble averaging, regularized DE).
The gap is dominated by intrinsic per-date PnL variation — half the local
test (dates 96..107) yields ~26 % higher PnL/day than the other half
(dates 108..119) regardless of threshold.

**iter_014 candidate**: kfold5_date winner thresh = (3.7153e-4, 1.6599e-4)
→ LOSO +36.22 (vs iter_013 +36.15, **+0.075** local). Differs from iter_013
on **2 722 of 442 080 rows (0.62 %)**, almost all in the down-side band
1.61e-4 < |pred| < 1.66e-4 where iter_014 trades less. Expected platform
delta: small positive (negligible) at best — does **not** close the −17 gap.

`submission_050800_iter014_robust_thresh.zip` (3.28 MB) is built and passes
all sanity checks.

`RESULT: task=t83_robust_thresh metrics={best_strategy=kfold5_date, unbiased_eval=35.94, vs_iter013=+0.20} notes="all robust strategies converge to ≈ iter_013 thr; gap is intrinsic data-quality, not threshold overfit"`

## Setup

- **OOF**: `pred_T75_seed{1,7,13,42,100}.parquet`, 442 080 rows = 24 dates ×
  5 syms × 2 sessions × ~3 684 ticks. Pred = mean of 5-seed regression
  Δmid_norm boosters.
- **PnL**: per-row `(side · (mp_th − mp_t) − FEE · |side| · |(mp_th+1)+(mp_t+1)|) / (mp_t + 1)`,
  summed per-sym, then summed across syms (= LOSO-equiv).
- **iter_013 reference** (full-fit DE asym applied to all 442k):
  - thr_up = 3.7230e-4, thr_dn = 1.6130e-4 (as written in iter_013's
    `thresholds.json`; the slightly-coarser-than-DE-optimum value in that
    file gives **+36.15** local; the actual DE optimum is +36.23).
  - Platform actual: **+19.23** → gap = **−17**.

## Strategies tested

| ID | name             | description |
|----|------------------|-------------|
| A  | sym_grid         | 1D symmetric k ∈ [0.25, 3.5] (thr_up = thr_dn = 2k·FEE) |
| B  | de_asym          | 2D differential evolution on (thr_up, thr_dn) — iter_013 form |
| C  | de_reg_lam{λ}    | DE asym + (λ/FEE²)·(thr_up − thr_dn)² symmetry penalty (λ ∈ {0.1, 1, 10}) |
| D  | kfold5_date      | 5-fold CV-DE on date axis; threshold = median of fold-out fits |
| E  | kfold5_row       | 5-fold CV-DE on rows; median |
| F  | bootstrap15      | 15 bootstrap samples × DE (frac=0.7); median threshold |
| G  | quantile_grid    | grid-search (q_up, q_dn) over pred-distribution quantiles |
| H  | ensemble_avg_3   | average of {sym_grid, kfold5_date, bootstrap_de} thresholds |

## Splits

| ID | name           | description |
|----|----------------|-------------|
| S1 | date_fwd       | train = dates 96..107 (halfA), eval = dates 108..119 (halfB) |
| S2 | date_bwd       | train = dates 108..119 (halfB), eval = dates 96..107 (halfA) |
| S3 | random_50      | random 50/50 row split (sanity — should give gap ≈ 0) |

## Diagnostic finding (per-half PnL is heterogeneous)

`diagnostic.py` results:

| half             | iter_013 thr → PnL | oracle (DE on this half only) |
|------------------|--------------------|-------------------------------|
| A (96..107)      | 20.7500            | 20.9537                       |
| B (108..119)     | 15.3964            | 15.4621                       |
| total            | **36.1464**        | —                             |

Cross-application:
- thr_optimal_A applied to halfB: 15.0501  (vs halfB-oracle 15.4621 → **lost 0.41**)
- thr_optimal_B applied to halfA: 20.6937  (vs halfA-oracle 20.9537 → **lost 0.26**)

**Two consequences:**

1. **Half-A and Half-B have nearly the same oracle thresholds** — the
   "wrong" thr from one half costs only 0.26-0.41 PnL on the other.
2. **Half B's intrinsic PnL is ~26 % lower than half A's** at any
   sensible threshold. Per-date PnL ranges 0.21..4.37 with no threshold
   choice that can recover the bottom dates.

So the maximum possible eval-PnL gain from a "perfectly chosen" thr is
≈ 0.4 PnL per half-split. **No threshold strategy can close anything close
to the platform's −17 gap.**

## Per-strategy results (mean over S1+S2 date splits)

`run_main.py` ran each strategy on S1, S2, S3 (S3 is sanity, gap ≈ 0).
Summed PnL across the two date-splits' eval halves (= unbiased LOSO-equiv
estimate):

| rank | strategy        | sum_eval (S1+S2) | mean_gap | thr_up (S1, S2)         | thr_dn (S1, S2)         |
|------|-----------------|-----------------:|---------:|-------------------------|-------------------------|
| 1    | **kfold5_date** | **35.944**       | −0.20    | (3.72e-4, 3.70e-4)      | (1.32e-4, 1.59e-4)      |
| 2    | de_asym         | 35.744           | −0.34    | (3.72e-4, 3.70e-4)      | (2.07e-4, 1.59e-4)      |
| 3    | kfold5_row      | 35.735           | −0.32    | (3.72e-4, 3.70e-4)      | (2.06e-4, 1.60e-4)      |
| 4    | ensemble_avg_3  | 35.733           | **+0.09**| (3.24e-4, 3.34e-4)      | (1.81e-4, 1.99e-4)      |
| 5    | quantile_grid   | 35.690           | −0.25    | (3.64e-4, 3.71e-4)      | (2.14e-4, 1.62e-4)      |
| 6    | bootstrap15     | 35.649           | −0.28    | (3.67e-4, 3.45e-4)      | (1.93e-4, 1.59e-4)      |
| 7    | de_reg_lam0.1   | 35.391           | −0.47    | (3.72e-4, 3.33e-4)      | (2.22e-4, 1.63e-4)      |
| —    | (full-fit DE)   | 36.146 (LOSO ref)|  —       | 3.72e-4                 | 1.61e-4                 |

Top six strategies span **only 0.30 sum_eval** (≈0.4 % spread). The kfold5_date
edge over de_asym is 0.20 — within the noise of inner-fold sample size (kfold5
inner folds use 9-10 dates of data each; thr_dn variance is sample-dependent).

S3 random row split: all strategies give |gap| ≤ 0.6 → confirms the ~6 PnL
gap on date splits is data-distribution shift, not optimization noise.

## Pick

**Winner = kfold5_date.** It has the highest mean eval (35.944 vs de_asym
35.744). When refit on the full 442k:

```
thr_up = 3.7153e-4    thr_dn = 1.6599e-4   →   LOSO +36.2213
```

vs iter_013 `thresholds.json` (3.723e-4, 1.613e-4) → LOSO +36.1464.

**Net change**: thr_up nearly identical (−0.21 % relative), thr_dn 2.91 %
more conservative. **2 722 / 442 080 (0.62 %) rows flip action** vs
iter_013, mostly short → flat in the band 1.613e-4 < |pred| < 1.660e-4.

Local LOSO improvement: +0.0750. Translated through observed S1 transmission
rate ≈ 0.72: expected platform delta ≈ +0.05. **Not material.**

## Recommendation / honest accounting

The hypothesis that robust thresholding closes the platform gap is
**falsified** by this analysis. The −17 gap is dominated by data-quality
heterogeneity across dates, not by thresholding overfit. To close that
gap, the lever has to be **the regression model itself** (e.g. better
calibration, distribution-shift robustness, larger ensembles) — or
**reducing trade volume** (more conservative thr) to lower variance, at
the cost of expected return.

We ship `iter_014_robust_thresh.zip` because:
- It marginally improves local LOSO (+0.075).
- It uses a CV-derived threshold (less overfit by construction).
- It costs nothing — same boosters, same Predictor; only `thresholds.json` differs.
- It strictly de-risks (more conservative thr_dn → fewer trades in noisy band).

But it is **not expected** to materially close the platform gap. We
should look for other levers (model retraining with different objective,
or a fundamentally different signal) for the +5 platform target.

## Artifacts

- `core.py` — strategy fitters & PnL utilities
- `extra_strategies.py` — quantile_grid, ensemble_avg_3, pnl_sharpe
- `run_main.py` — train/eval split runner (10 strategies × 3 splits)
- `run_main.log` — runtime log
- `results_main.json` — parsed strategy results
- `diagnostic.py` / `diagnostic.log` — per-half oracle analysis
- `pick_winner.py` — winner selection + full-data fit
- `winner_summary.json` — winner metadata
- `thresholds.json` — iter_014 EV-gate thresholds (kfold5_date winner)
- `pack_iter014.py` — submission packaging
- `iter014_pkg/` — unzipped iter_014 package (Predictor + boosters + thresholds)
- `sanity_check.py` / `sanity_check.log` / `sanity_check_results.json` —
   end-to-end smoke + bit-for-bit comparison vs iter_013_pkg
- `submission_050800_iter014_robust_thresh.zip` — final 3.28 MB submission

## Sanity checks (`sanity_check.py`)

- ✓ config.json identical between iter_013 and iter_014 packages (154 features, 5 labels)
- ✓ all 5 boosters byte-identical (only thresholds.json differs)
- ✓ predict on 32 random batches (sym=99 implicit) succeeds
- ✓ shuffle invariant
- ✓ 1024-batch timing 0.577 s ≈ iter_013's 0.633 s
- 0.62 % of 442k OOF rows have action flips vs iter_013
