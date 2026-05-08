# R_50seed_bag — meta-paradigm M2

50-seed LightGBM L2 bag (only seed varies) + isotonic calibration + DE asym threshold,
with bag-size sweep B ∈ {5, 15, 30, 50}.

## Config

- **Single fixed hyperparam config** (= T75 seed-42 canonical):
  `feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0`
- 50 seeds (1..50). Diversity comes only from `seed` field
  (LightGBM `feature_fraction_seed`, `bagging_seed`, `data_random_seed` all derived).
- Train: schemeP date 0-75. Val: 76-79 (early stop + isotonic calib set).
  Test: schemeP test split (date 96-119, 5 syms).
- Aug: aug_a (per-(sample, feat) scale [0.80, 1.20]), class-balanced sample weights.

## Why local (not r7)

Remote vast.ai SSH ports refused at start of run. Pivoted to local RTX 3090:
- 1 seed ≈ 30s LGB GPU training. 50 seeds ≈ 25-30 min total. Fits within budget.

## Pipeline

1. `train_50seed.py` — train 50 LGB models, save per-seed `model_h60_seed{S}.txt`,
   `pred_T75_50seed_seed{S}.parquet` (test), `pred_T75_50seed_val_seed{S}.parquet` (val).
2. `eval_bag.py` — for each B:
   - bag avg test pred (B seeds)
   - **uncal**: DE asym threshold search → LOSO-equiv pnl
   - **isotonic**: fit IsotonicRegression on val (date 76-79), apply to test bag pred,
     then DE asym threshold → LOSO-equiv pnl
3. Compare against canonical 5-seed baseline (T75 = +36.23 LOSO-equiv).

## CRITICAL_CONSTRAINTS compliance

- sym/date NOT in features (drop set + asserts in `train_50seed.py`)
- Predictor stateless (no per-call state — model.predict only)
- Single global threshold pair across all syms (sym-agnostic)
- Calibration uses val date 76-79 only; test (96-119) untouched for calibration
