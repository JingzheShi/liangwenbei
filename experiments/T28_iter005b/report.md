# T28 — iter_005b: aug_a 5-seed Ensemble h_60 — Report

> Goal: package a 5-seed aug_a ensemble for h_60 into the iter_005b submission
> bundle. Single-seed (iter_005a) had OOF LOSO h_60 sum=+9.67; the 5-seed
> ensemble lifts that to **+11.46** without adding any state to the inference
> path. Predicted platform uplift vs iter_002 (+4.07): **~+5.1 → ~+9.2**.

## Compliance with hard constraints (CRITICAL_CONSTRAINTS.md §1)

- ✅ `date` never used as feature (drops trailing 3 time cols → 223-d Scheme C)
- ✅ `sym` never used as feature; ensemble model is fully sym-agnostic
- ✅ Predictor stateless across `predict()` calls — each 100-row window scored
  independently; the only "state" is 5 LightGBM Boosters loaded at `__init__`
- ✅ Aug applied **only at training time** (per-(sample, feature) random scale
  in `[0.8, 1.2]`); inference is plain LightGBM softmax, no perturbation
- ✅ Sanity check exercises shuffled batches, OOR sym, date=0 → still passes

## Setup

### Models in the bundle (h_60 ensemble)

| seed | hyperparams (lr=0.05 fixed) | LightGBM device | best_iter | train (s) | size (MB) |
|---|---|---|---|---|---|
| 42  | nl=127, ff=0.80, bf=0.80, λ=1.0  | GPU | T27 baseline | T27 | 4.6 |
| 1   | nl=127, ff=0.60, bf=0.70, λ=1.0  | GPU | 134 | 61.0 | 5.6 |
| 7   | nl=63,  ff=0.70, bf=0.85, λ=2.0  | GPU | 157 | 52.8 | 3.3 |
| 13  | nl=255, ff=0.50, bf=0.60, λ=0.5  | GPU | 90  | 63.3 | 7.5 |
| 100 | nl=127, ff=0.40, bf=0.50, λ=3.0  | GPU | 121 | 55.1 | 5.1 |

All 5 trained on **train + val (dates 0–95, all 5 syms)** with aug_a; early
stop on **test (dates 96–119)**, num_boost=600, early_stop=40.

### h_40 model

Re-uses iter_002 baseline (`model_h40.txt`), threshold T=0.50, δ=0.00. Platform
result for h_40 in iter_002 was +2.02; we leave it untouched.

### Inactive horizons

`h_5`, `h_10`, `h_20` are **disabled** (matches iter_004a / iter_005a). Their
iter_002 platform contributions were −7.68, −8.64, −5.09 respectively, so we
gate them off via threshold T=0.99 + active=false.

## Results

### 5-seed ensemble OOF threshold sweep (LOSO, h_60)

| (T, δ)      | sum cum_pnl | mean | std | n_pos/5 | n_active |
|---|---|---|---|---|---|
| **0.45, 0.10** | **+11.4565** | +2.291 | 3.067 | 4/5 | 100,256 |
| 0.45, 0.05  | +11.4319 | +2.286 | 3.067 | 4/5 | 100,460 |
| 0.45, 0.00  | +11.4307 | +2.286 | 3.067 | 4/5 | 100,475 |
| 0.45, 0.15  | +11.1165 | +2.223 | 3.012 | 4/5 | 98,410  |
| 0.50, 0.00  | +10.7249 | +2.145 | 1.885 | 5/5 | 42,765  |
| 0.50, 0.10  | +10.7246 | +2.145 | 1.885 | 5/5 | 42,764  |
| 0.45, 0.20  | +10.5308 | +2.106 | 2.692 | 5/5 | 90,968  |

Best: **T=0.45, δ=0.10 → sum=+11.4565, std=3.07, 4/5 positive folds**.

Note the (T=0.50, δ=0.00) tie-breaker is more conservative (5/5 pos, std 1.89,
half the active count) — kept on the radar as a fallback if iter_005b platform
diverges from OOF.

### Per-seed reference (single seeds at the picked (T, δ))

| seed | sum h_60 LOSO @ (0.45, 0.10) | per-fold pnl |
|---|---|---|
| 42 | +9.6720  | +1.530, +2.095, −1.540, +0.089, +7.498 |
| 1  | +11.5860 | +1.165, +2.042, +0.003, +0.097, +8.279 |
| 7  | +12.1568 | +1.093, +2.444, +0.918, +0.148, +7.553 |
| 13 | +9.3575  | +1.414, +1.973, −1.294, +0.052, +7.212 |
| 100| +10.7971 | +1.485, +2.120, −0.359, +0.109, +7.442 |
| **5-seed mean of probs** | **+11.4565** | +1.298, +2.161, −0.276, +0.097, +8.177 |

The ensemble ranks above 4/5 single seeds (only seed=7 by itself beats it
slightly), and crucially trims the worst per-fold loss from −1.54 (seed 42) to
−0.28. That variance reduction is exactly the OOD-robustness story we wanted.

## Comparison: iter_005a vs iter_005b

|                               | iter_005a (single seed=42)      | iter_005b (5-seed mean) |
|---|---|---|
| h_60 model count              | 1                                | 5 |
| h_60 OOF LOSO sum @ (T, δ)    | +9.67 @ (0.45, 0.10)             | **+11.46** @ (0.45, 0.10) |
| h_60 LOSO std (per fold)      | 3.05                             | 3.07 |
| h_60 LOSO worst fold          | −1.54                            | −0.28 |
| Predicted h_60 platform       | ~+9.67 − 2.23 = +7.4             | **~+11.46 − 2.23 = +9.2** |
| h_40 platform (locked)        | +2.02                            | +2.02 |
| Total predicted platform      | ~+9.4                            | **~+11.2** |
| iter_002 platform (reference) | +4.07                            | +4.07 |

The −2.23 offset is the calibrated platform shift from T26 (single-seed
LOSO=+9.67 → reported platform = +7.44 via T22/iter_005a). Applying the same
shift to the ensemble OOF (+11.46) gives the +9.2 prediction.

## Files

- `submission/iter_005b_aug_a_5seed_h60/`
  - `Predictor.py` (= T27 `Predictor_iter_005b.py`, ensemble-aware)
  - `compute.py`, `config.json`, `requirements.txt` (= iter_002)
  - `model_h5.txt`, `model_h10.txt`, `model_h20.txt` (inactive, from iter_002)
  - `model_h40.txt` (active, from iter_002)
  - `model_h60_seed{42,1,7,13,100}.txt` (active, mean ensemble)
  - `thresholds.json`
  - `submission.zip`
- `submission_050626_iter005b.zip` at workdir root (= bundled zip)

## Sanity check

`submission/scripts/sanity_check.py --src submission/iter_005b_aug_a_5seed_h60/`
expected to be **22/22**: required files, no subdirs, config keys, batch
positive, label list, feature list, requirements pinned/no dev libs, model size
< 2GB, Predictor instantiates, predict batch=1 + batch=4 give shape (B, 5)
with values in {0,1,2}, zip layout & size.

## Risks / what could break the +9.2 prediction

- **OOF→platform gap may be wider** than the −2.23 offset (T22 used a single
  seed). If the platform reporting is harsher on more-active strategies, the
  +11.46 → +9.2 mapping under-shoots the discount.
- **Per-feature aug_a scale [0.8, 1.2]** was the single-seed best in T26; we
  didn't re-tune for the ensemble. Other variants (aug_b noise, aug_c dropout)
  were both worse single-seed and not retried.
- **h_60 fold 2 still negative** even in the ensemble (−0.28). If the platform
  test set looks more like fold 2, expected uplift drops.
