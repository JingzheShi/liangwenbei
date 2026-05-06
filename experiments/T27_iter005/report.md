# T27 — iter_005a / iter_005b: aug_a-driven h_60 submissions

**Goal:** translate the T26 LOSO breakthrough — `aug_a` (per-feature random
scale [0.8, 1.2]) on h_60 raised LOSO sum from +6.30 (iter_002 baseline) to
**+9.67** — into a platform submission. Two packages produced:

- **iter_005a**: T26 `aug_a` final model (single seed 42), h_60 only
- **iter_005b**: 5-seed `aug_a` ensemble (seeds 42, 1, 7, 13, 100) for h_60

Both keep h_5 / h_10 / h_20 inactive (matches `iter_004a`) and reuse the
iter_002 baseline `model_h40.txt` (T=0.5, δ=0.0, platform +2.02).

## 1. Background

iter_002 platform best-of-5 = **+4.07** (h_60 active, T=0.5, δ=0.2).

T25 + T26 follow-up isolated **per-feature random scale `aug_a`** as the
single biggest LOSO lever for h_60:

| variant       | LOSO best (h_60) | δ from baseline |
|---------------|------------------|-----------------|
| iter_002 baseline (single seed 42) | +6.30  | — |
| T25 5-seed ensemble (no aug)       | +7.56  | +1.26 |
| T26 aug_a (seed 42, T=0.45 δ=0.10) | **+9.67** | +3.37 |
| T26 aug_b / aug_c / aug_abc        | ≤ +7.5 | ≤ +1.2 |

Calibration gap from previous iters: LOSO h_60 +6.30 → platform +4.07 (gap
−2.23). Linear extrapolation predicts:

- iter_005a single-seed aug_a → platform ≈ +9.67 − 2.23 = **+7.44**
- iter_005b 5-seed aug_a ensemble → platform target ≈ +9 to +11 (if the
  ensemble synergy from T25 stacks on top of aug_a)

## 2. iter_005a (single seed aug_a)

**File:** `submission_050624_iter005a.zip` (16.04 MB), bundle dir
`submission/iter_005a_aug_a_h60/`.

Build:
1. `train_final_aug_a.py --seeds 42` retrains a 223-d Scheme C LightGBM on
   train+val (dates 0..95, all 5 syms) with `aug_a` augment; uses test
   (dates 96..119) as the early-stopping val set. 52.4 s on GPU,
   `best_iter=110`. Output: `final_model_h60_aug_a_seed42.txt` (4.6 MB).
2. `build_iter_005a.py` copies iter_002 base + new h_60 model + thresholds:

| h | T | δ | active | source |
|---|---|---|--------|--------|
| 5  | 0.99 | 0.99 | False | iter_004a strategy (platform −7.68) |
| 10 | 0.99 | 0.99 | False | iter_004a strategy (platform −8.64) |
| 20 | 0.99 | 0.99 | False | iter_004a strategy (platform −5.09) |
| 40 | 0.50 | 0.00 | True  | iter_002 (platform +2.02) |
| 60 | 0.45 | 0.10 | True  | T26 aug_a sweep (LOSO +9.67) |

Sanity test: 22/22 batches OK, h_5/10/20 = 1 (inactive) on shuffled OOR-sym
inputs.

**Expected platform best-of-5 ≈ +7.4** (vs iter_002 +4.07 → uplift +3.3).

## 3. iter_005b (5-seed aug_a ensemble)

**File:** `submission_050625_iter005b.zip`, bundle dir
`submission/iter_005b_aug_a_5seed_h60/`.

Pipeline:

1. **LOSO** (`train_5seed_aug_a.py --seeds 1,7,13,100`): for each new seed,
   5 LOSO folds (held-out sym) with `aug_a`. 20 fold trainings on GPU.
   T25 SEED_CONFIGS (varying feature/bagging/leaves/L2 per seed for diversity).
   Outputs: `loso_pred_h60_aug_a_seed{S}_held{K}.parquet` (S∈{1,7,13,100},
   K∈0..4). Combined with the existing T26 seed=42 OOF
   (`experiments/T26_domain_randomization/loso_pred_h60_aug_a_held{K}.parquet`).
2. **5-seed ensemble OOF sweep** (`sweep_5seed_ensemble.py`): per fold,
   ensemble probs = mean over 5 seeds. Sweep (T, δ) on the 442 080-row
   concatenated OOF.
3. **Final retrain** (`train_final_aug_a.py --seeds 1,7,13,100`): each new
   seed retrained on full train+val with `aug_a`. Outputs:
   `final_model_h60_aug_a_seed{S}.txt`.
4. **Build** (`build_iter_005b.py`): bundles 5× h_60 models with the
   ensemble-aware Predictor (`Predictor_iter_005b.py`) that mean-averages
   booster softmax outputs before thresholding. Threshold pulled from step 2.

Predictor compliance: still sym-agnostic, stateless, no date — only change
vs iter_002 is loading a list of boosters per horizon.

### LOSO 5-seed results (raw argmax per seed, h=60)

_(filled in after `train_5seed_aug_a.py` completes — see
`loso_5seed_aug_a_summary.json`)_

### 5-seed ensemble OOF threshold sweep

_(filled in after `sweep_5seed_ensemble.py` completes — see
`sweep_5seed_ensemble_results.json`)_

## 4. Comparison vs prior iterations

| iter   | h_60 source | h_60 (T,δ) | LOSO h_60 sum | Platform best-of-5 |
|--------|-------------|-----------|---------------|---------------------|
| iter_002  | LightGBM C 223-d (seed 42)            | (0.50, 0.20) | +6.30 | +4.07 |
| iter_004a | iter_002 + h_5/10/20 disabled         | (0.50, 0.20) | +6.30 | TBD (≥ iter_002) |
| iter_004b | iter_002 + h_5/10/20 disabled, h_60 (0.45,0.20) | (0.45, 0.20) | TBD   | TBD |
| iter_005a | aug_a single seed 42                  | (0.45, 0.10) | +9.67 (LOSO sweep) | **expected ≈ +7.4** |
| iter_005b | aug_a 5-seed ensemble                 | (TBD)        | TBD             | **expected ≈ +9 to +11** |

## 5. Files

- `train_final_aug_a.py` — full-data aug_a final trainer (1 model per seed)
- `train_5seed_aug_a.py` — LOSO aug_a trainer for 5 diverse seeds
- `sweep_5seed_ensemble.py` — OOF threshold sweep on 5-seed mean probs
- `build_iter_005a.py` / `build_iter_005b.py` — submission builders
- `Predictor_iter_005b.py` — ensemble-aware Predictor template

## 6. Constraints (compliance)

All three `CRITICAL_CONSTRAINTS.md` red lines respected:

1. `date` not in features (config.feature unchanged from iter_002, which
   excluded `date`).
2. Predictor is fully stateless across `predict()` calls — no buffers.
3. Models are sym-agnostic — only the 154 raw + 69 T3 features (Scheme C
   223-d, no `sym`/`date`/`time`). aug_a uses no sym info.
