# T11 — Scheme C × 5-seed ensemble (multi-horizon)

**Status**: running
**Started**: 2026-05-06 07:05 UTC
**WandB run**: T11-schemeC-multiseed-* (project=liangwenbei)

## Goal
Build a multi-seed ensemble on top of the best Scheme C horizon (h_10, +21.86 LOSO from
iter_002). Force diversity across seeds via different (feature_fraction, bagging_fraction,
num_leaves, lambda_l2) — T6b confirmed that same-hyperparam multi-seed gives no
improvement (low diversity).

## Method

### LOSO setup (matches T5b)
- Train: (sym != held, dates 0..79)
- Val:   (sym != held, dates 80..95) — early stopping
- Test:  (sym == held, dates 96..119) — held-out

### Feature space
Use the **223-d** (Scheme C, dropping 3 time-encoding cols) feature space — same as
iter_002's submission feature space. Cache: T5b's `schemeC_train/val/test.npz` sliced
to first 223 cols.

### Diversity-forcing seed configs
| seed | feature_fraction | bagging_fraction | num_leaves | lambda_l2 |
|---:|---:|---:|---:|---:|
| 42  | 0.80 | 0.80 | 127 | 1.0 |
| 1   | 0.60 | 0.70 | 127 | 1.0 |
| 7   | 0.70 | 0.85 |  63 | 1.0 |
| 13  | 0.60 | 0.70 | 127 | 2.0 |
| 100 | 0.50 | 0.60 | 255 | 1.0 |

Common: lr=0.05, min_data_in_leaf=100, bagging_freq=5, num_threads=18,
num_boost_round=600 + early_stopping=40.

### Aggregation
For each (h, fold), average `prob_0/1/2` across the 5 seed boosters → ensemble OOF.

### Threshold sweep
Per-horizon grid sweep on ensemble OOF:
- T ∈ {0.35, 0.40, …, 0.70} (8 vals)
- δ ∈ {0.00, 0.05, …, 0.25} (6 vals)
- Decision: side_max=max(p0,p2); take iff (side_max ≥ T) ∧ (side_max > p1+δ); else hold.

## Results

### h_10 — per-seed individual sweep + ensemble (TBD)

(filled in by `threshold_sweep.py`)

### Cross-horizon comparison vs iter_002

| | iter_002 (single seed=42, T5b) | iter_004 (5-seed ensemble) |
|---|---|---|
| h_5  best  | +17.49 | TBD |
| h_10 best  | +21.86 | TBD |
| h_20 best  | +19.70 | TBD |
| best_of_3  | +21.86 (h_10) | TBD |

## Submission

If h_10 ensemble best > +23 → build `submission/iter_004_lgbm_schemeC_ensemble/` with:
- 5 seed × N_horizon final-trained 223-d models
- Predictor.py averaging probability across seeds
- thresholds.json with per-horizon (T, δ)

## Verdict

(TBD after results)
