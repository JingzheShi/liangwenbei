# T50 — LightGBM DART boosting, h_60, 5-seed × 5-fold LOSO

**Status:** PENDING (will be filled after run completes)

## Hypothesis

DART (Dropouts meet Multiple Additive Regression Trees) replaces GBDT in
iter_005b. Each new tree must compensate for a randomly dropped subset of
existing trees → forces redundant / general patterns → expected to reduce
OOD overfit. R32+R35 expectation: +1~+4 LOSO over GBDT (iter_006 = +13.61).

## Setup (single-idea ablation, only boosting_type swapped)

| field | value |
|---|---|
| features | Scheme C 223-d (matches iter_005b/iter_006; drop 3 trailing time-encoding cols) |
| horizon | 60 |
| aug | per-(sample,feat) U[0.80, 1.20], orig+aug 1:1 |
| folds | LOSO over sym ∈ {0,1,2,3,4} |
| seeds | {42, 1, 7, 13, 100} (T27 SEED_CONFIGS for diversity) |
| boosting | DART (drop_rate=0.10, max_drop=50, skip_drop=0.50, uniform_drop=False) |
| num_boost_round | 1500 fixed (DART has no early stopping) |
| learning_rate | 0.04 (DART converges slower than GBDT) |
| device | GPU |
| ensemble | NONE (single idea ablation) |

## Reference baselines (from PROGRESS.md)

| run | OOF LOSO sum | notes |
|---|---|---|
| iter_006 (T26 aug_a, 5-seed) | **+13.61** | current SOTA |
| T44 R34 features | +12.09 | < iter_006 |
| T45 Group DRO | +9.07 | < iter_006 |
| T46 CatBoost Plain h60 | +12.64 | < iter_006 |
| T47 multi-h stacking | +12.36 | < iter_006 |
| T50 DART (this) | TBD | target > +13.61 |

## Results (filled in after run)

(filled in)

## Decision

(filled in)
