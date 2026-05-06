# T12 — Multi-horizon stacking with meta-learner (Ridge / LR / LightGBM)

**Status: Negative result. Meta does not beat iter_002 baseline. iter_005 NOT built.**

## TL;DR

| Approach | LOSO h_10 sum_cum_pnl | Notes |
|---|---|---|
| iter_002 base h_10 (T=0.55, δ=0.10) | **+21.86** | reference |
| Meta-LR (best per fold) + threshold sweep | +17.97 | best at T=0.35, δ=0.0 (≈ raw argmax) |
| Meta-LGBM (best per fold) + threshold sweep | +17.46 | best at T=0.40, δ=0.0 |

R12's "completely free" multi-horizon stacking does not work for our setup. Threshold-tuned base model is uncatchable by any post-softmax meta. **No iter_005 produced.**

## Setup

- Loaded T5b's 25 LOSO OOF parquets (5 horizons × 5 folds) → 442,080 rows × 15 base prob features (3 classes × 5 horizons).
- For each held-out sym k ∈ {0..4}:
  - meta_train = OOF rows where sym ≠ k (353,664 rows; preds came from held{0..4}\{k} models, valid OOF)
  - meta_val = OOF rows where sym = k (88,416 rows; preds from held{k} model)
- Target = label_10. Sweep:
  - LR with C ∈ {0.5, 1, 2, 5, 10}
  - Ridge-regressor + softmax with α ∈ {0.1, 1, 5, 10, 50}
  - LightGBM with 3 small configs (num_leaves ∈ {7,15,31}, depth 3-5, 150-300 rounds, early stop on val multi_logloss)
- Selection: per-fold val multi_logloss best.

## Per-fold meta selection

| Fold (held sym) | Best (LR/Ridge sweep) | Best LGBM | iter_002 base h10 PnL |
|---|---|---|---|
| 0 | LR C=2 ll=0.5757 | lgbm_7_3_150 ll=0.6035 | +1.94 |
| 1 | RidgeSoftmax α=0.1 ll=1.0274 | lgbm_31_5_300 ll=1.0341 | +5.13 |
| 2 | LR C=2 ll=0.6173 | lgbm_7_3_150 ll=0.6325 | +4.35 |
| 3 | LR C=0.5 ll=0.2745 | lgbm_7_3_150 ll=0.2905 | +2.11 |
| 4 | LR C=1 ll=0.8133 | lgbm_15_4_200 ll=0.8134 | +8.33 |

LR generally beats Ridge (better calibration). LGBM is comparable but not better.

## Per-fold meta PnL after threshold sweep

| Fold | iter_002 base | Meta-LR best | Meta-LGBM best |
|---|---|---|---|
| 0 | +1.94 | +1.27 (raw) | +0.92 |
| 1 | +5.13 | +5.55 | +5.91 |
| 2 | +4.35 | +3.98 | +3.91 |
| 3 | +2.11 | +0.95 | +1.19 |
| 4 | +8.33 | +6.22 | +5.53 |
| **sum** | **+21.86** | **+17.97** | **+17.46** |

Meta is competitive on fold-1 only (matched/slightly better). It loses on the other 4 folds.

## Why does meta lose?

Distribution diagnosis on the full 442k OOF:

| Statistic | Base h_10 | Meta-LR | Meta-LGBM |
|---|---|---|---|
| mean p1 (hold prob) | 0.371 | **0.679** | 0.680 |
| frac side-max ≥ 0.55 | 21.97% | 2.47% | 3.37% |
| take-rate at (T=0.55, δ=0.10) | 21.97% | 2.47% | 3.37% |

The meta-learner is **dramatically more conservative**. It absorbs the 71% class-1 (no-move) prior from the training distribution and pulls all predictions toward p1 ≈ 0.68. The base LightGBM, trained directly on 223-d raw features, can escape this prior on individual confident windows (22% of rows clear the threshold), but the meta — which only sees 15-d post-softmax probs — cannot recover the same confidence:

> Going through softmax + 15-d projection is an **information bottleneck**. Whatever discriminating signal the base model uses to lift p0 or p2 above 0.55 cannot be reconstructed from 15 averaged probabilities.

This is a known stacking failure mode: when the base is already well-calibrated for thresholded decisions and the meta sees only its outputs, the meta either reproduces or smooths the base — it cannot exceed it.

## Feature importance (LGBM meta, held0)

Top features by gain (decoded from Column_i):

| rank | feature | gain |
|---|---|---|
| 1 | p1_h10 | 340k |
| 2 | p2_h5  | 239k |
| 3 | p0_h10 | 174k |
| 4 | p1_h5  | 169k |
| 5 | p0_h5  |  84k |
| 6 | p2_h40 |  41k |
| 7 | p2_h60 |  33k |

h_5 and h_10 dominate. h_20 contributes essentially nothing. The meta does try to use cross-horizon info (h_5 ahead of h_10), but the lift in raw-argmax (+9.9 vs base) is offset by the over-conservative softmax — at any meaningful threshold the base wins.

## What it would take to beat iter_002

The meta cannot win unless:
1. Meta gets richer features than raw probs (e.g., prob differences, max-min spread, agreement across horizons, base h_10 thresholded decision as a side input).
2. Meta is trained on a thresholding-aware objective (custom loss that rewards confident correct sides), not multi-logloss.
3. Or: include the full 223-d feature vector in the meta input → at which point we're just retraining a base model with horizon hints.

These are no longer "completely free"; they require new training pipelines. R12's recommendation framed multi-horizon stacking as a free win — for this dataset and this base model it is not.

## Files

- `train_meta.py`, `train_meta.log` — Ridge / LR sweep (5 LR + 5 Ridge per fold)
- `train_meta_lgbm.py`, `train_meta_lgbm.log` — LightGBM meta (3 configs × 5 folds)
- `threshold_sweep.py`, `threshold_results.json`, `sweep_results_h10_meta.csv` — LR/Ridge meta sweep
- `threshold_sweep_lgbm.py`, `threshold_results_lgbm.json`, `sweep_results_h10_meta_lgbm.csv` — LGBM meta sweep
- `meta_oof_h10.parquet` — LR/Ridge meta OOF (442,080 rows)
- `meta_oof_h10_lgbm.parquet` — LGBM meta OOF
- `meta_lgbm_held{0..4}.txt` — saved LGBM meta boosters
- `fold_summaries.json`, `fold_summaries_lgbm.json` — per-fold model picks

No `iter_005` submission produced (meta < iter_002, threshold not met).
