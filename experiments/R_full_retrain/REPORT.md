# R_full_retrain — iter_019 v1/v2: Full-data LGB retrain (M7 paradigm)

**Status**: TRAINING IN PROGRESS — will be filled after all 5 seeds + LOSO sanity + zip pack done.

## Task

Apply meta-paradigm M7 (standard Kaggle final-submit trick): retrain the 5 T75 LGB
L2 boosters on **full data date 0-119** (train + val + test concatenated),
**no validation, no early stop**, fixed `num_boost_round = 330`
(≈ avg(T75 best_iter) × 1.1).

Rationale: date 80-119 is the most recent platform regime — letting LGB see
that distribution should improve out-of-sample (platform) performance with
trivial cost (+1 to +3 platform PnL).

5 seeds (1, 7, 13, 42, 100) — same per-seed configs as T75_regression_dmid:

| seed | feature_fraction | bagging_fraction | num_leaves | lambda_l2 |
|---|---|---|---|---|
| 1   | 0.6 | 0.70 | 127 | 1.0 |
| 7   | 0.7 | 0.85 |  63 | 2.0 |
| 13  | 0.5 | 0.60 | 255 | 0.5 |
| 42  | 0.8 | 0.80 | 127 | 1.0 |
| 100 | 0.4 | 0.50 | 127 | 3.0 |

## Compute Notes

- **Originally targeted remote r7** (Vast.ai, RTX 3080) but scp transfer rate
  was ~0.8 MB/s; transferring the 3.2 GB cache would have eaten the entire
  90-min budget. Killed scp at 192 MB after 20 min.
- **Pivoted to local 3090.** Initial GPU LGB (OpenCL) attempts hit
  `boost::compute::opencl_error: Memory Object Allocation Failure` at >5
  boost rounds (allocation pattern with num_leaves up to 255 exceeds GPU
  device buffer limit). Fell back to **CPU mode** (18 threads). seed=1 ran
  in 403 s; remaining 4 seeds queued sequentially in CPU.

## Constraints (CRITICAL_CONSTRAINTS)

| Constraint | Status |
|---|---|
| 1. `date` not used as feature | ✓ (drop_idx excludes nothing extra; feat_dim=359, no date/sym/time in feat_names assertion) |
| 2. Stateless predict, no cross-call state | ✓ (Predictor inherited byte-identical from iter_018) |
| 3. sym-agnostic forward (sym not fed to model) | ✓ (LGB trained without sym; sym only used as lookup key for v2 conformal band) |
| L2 (not Huber) | ✓ (`objective=regression_l2`, matches T75 / iter_015 v1 baseline) |
| W ≤ 100 | ✓ (inherited from iter_015 v1) |

## Pipeline

1. Load `schemeP_{train,val,test}.npz` from `T68_stage5_features/cache`
2. Concat → 2,210,400 rows, date 0-119, feat_dim=359 (after dropping 11 fail features)
3. aug_a augmentation (ratio=1.0, lo=0.80, hi=1.20) → 4,420,800 train rows
4. class_balanced_weight (3-class, regression target = (mp_t60 - mp_t) / (mp_t + 1))
5. LGB train (regression_l2, lr=0.05, num_boost_round=330, no validation)
6. Save `model_h60_seed{S}.txt`

## Packaging

Two submission zips:

### iter_019 v1 — `submission_050818_iter019_v1_fullretrain.zip`
- Base: **iter_015 v1** (no conformal wrapper, 154-feat config)
- Predictor.py / fast_features*.py / config.json / requirements.txt: byte-identical to iter_015 v1
- thresholds.json: byte-identical to iter_015 v1 (thr_up=0.000358, thr_dn=0.000216)
- 5 nn_h60_seed{S}.npz: byte-identical (T87 SPO+ NN)
- **5 model_h60_seed{S}.txt: full-retrain (replaces iter_015 v1 LGBs)**

### iter_019 v2 — `submission_050818_iter019_v2_fullretrain_conformal.zip`
- Base: **iter_018 v1** (per-sym β-conformal abstain wrapper, 155-feat config with sym)
- Predictor.py / fast_features*.py / config.json / requirements.txt: byte-identical to iter_018 v1
- thresholds.json: byte-identical to iter_018 v1 (thr_up=2.999e-4, thr_dn=2.159e-4 + per_sym β/σ)
- 5 nn_h60_seed{S}.npz: byte-identical (T87 SPO+ NN)
- **5 model_h60_seed{S}.txt: full-retrain (replaces iter_018 v1 LGBs)**

## Local LOSO sanity (INFLATED — not a decision criterion)

The full-retrain LGBs have **seen the test split (date 96-119)** during
training. Local PnL on test will be in-sample for LGB and is therefore an
inflated upper bound. We compute it anyway only to confirm:
- All 5 LGB models load and predict without error
- Combined stack (NN + LGB) variance is in the expected range (σ ≈ 4e-4)
- Per-sym min ≥ 0 (no obvious failure mode)

**The real eval is platform.** Expected platform PnL improvement: +1 to +3
over iter_015 v1 / iter_018 v1 baselines, per M7 paradigm priors.

(Local LOSO numbers will be inserted after training completes.)

## Files

```
R_full_retrain/
├── train_full.py              — adapted T128 script with num_boost_round=330
├── run_remaining_seeds.sh     — sequential CPU run for seeds 7/13/42/100
├── loso_sanity.py             — INFLATED LOSO eval (sanity only)
├── finalize.sh                — copy LGB models into pkg dirs + zip
├── model_h60_seed{1,7,13,42,100}.txt    — 5 full-retrain LGBs
├── summary_h60_seed{S}.json   — per-seed train summaries
├── pkg_iter019_v1/            — staged contents for v1 zip
├── pkg_iter019_v2/            — staged contents for v2 zip
└── train_remaining.log / logs/  — training logs
```
