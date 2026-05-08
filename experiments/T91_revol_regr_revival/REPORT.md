# T91 — R44 Top-3 Revival: ReVol+Savgol features under regression

> Worker T91, 2026-05-08. Task: revive T35 (ReVol per-sample log-return + Savgol smoothers,
> 32-d) under T75's L2-regression-on-Δmid_norm regime to see if R44's F-F (false-signal)
> hypothesis holds — i.e. that under L2-regression the τ-noise signal is rejected and the
> true Δmid signal in ReVol shows through.

## Setup
- Feature space: T68 schemeP (370-d) - 11 T59/Stage5-fail drops = **359-d** baseline,
  plus 32-d T35 schemeK_extra cache, evaluated in 3 variants:
  - **v1_all32** = 359 + 32 = **391-d** (full ReVol+Savgol)
  - **v2_revol12** = 359 + 12 = **371-d** (ReVol only, drop Savgol)
  - **v4_revol_only6** = 359 + 6 = **365-d** (just wmp1-ReVol features)
- Target: y_regr = (mp_t60 - mp_t)/(mp_t + 1)
- Protocol: identical to T75 — V4 walk-forward (train 0-75 / val 76-79 / test 96-119),
  LightGBM regression_l2, 5-seed average {1, 7, 13, 42, 100} with class-balanced sample
  weight, aug_a (per-(sample,feat) scale [0.80, 1.20]) at ratio 1.0.
- Evaluation: DE asymmetric threshold over (thr_up, thr_dn) ∈ [0, 0.0040]², LOSO-equiv
  = sum-of-per-sym cum_pnl (5 syms).
- Ensemble: T91-LGB ⊕ T81-NN (5 seeds each) at weight grid
  {(1,0),(0,1),(1,1),(1.5,1),(1,1.5),(2,1),(1,2),(1,0.5),(0.5,1)}.

## Headline numbers

| Variant | n_feat | LGB-only | vs T75 | best ens (best w_nn,w_lgb) | vs iter_014 |
|---|---:|---:|---:|---:|---:|
| **v1_all32** | 391 | +36.10 | -0.12 | +38.05 (1.0, 2.0) | **-0.23** |
| **v2_revol12** | 371 | **+36.38** | **+0.15** | **+38.18** (1.0, 1.5) | **-0.10** |
| **v4_revol_only6** | 365 | +36.28 | +0.06 | +38.09 (1.0, 1.0) | -0.19 |
| Reference: T75 LGB-only | 359 | +36.23 | — | iter_014 = +38.28 | — |

## Per-seed lift over T75 (test EV-gate k=1)

|seed | T75 ev_k1 | v1_all32 Δ | v2_revol12 Δ | v4_revol_only6 Δ |
|---:|---:|---:|---:|---:|
| 42  | +28.28 | +0.75 | +1.74 | **+5.09** |
| 1   | +30.55 | +0.00 | -0.21 | -1.17 |
| 7   | +31.02 | +1.14 | -0.02 | -1.97 |
| 13  | +31.30 | -1.35 | -1.20 | -0.64 |
| 100 | +29.85 | +2.46 | +0.03 | +0.20 |
| **mean** | **+30.20** | **+0.60** | **+0.07** | **+0.30** |

## Verdict — R44 Top-3 (T35 revival): **PARTIALLY DISCONFIRMED**

Two findings:

1. ✅ **R44's F-F (false-signal rejection) half is CONFIRMED**: T35's catastrophic CE
   loss of -0.77 LOSO is gone. Under regression L2, ReVol features no longer mis-fit
   τ-thresholding noise — the worst T91 variant is only -0.12 vs T75 (vs T35 CE -0.77
   from baseline).
2. ❌ **R44's "ReVol becomes a true signal" half is NOT supported**: the best variant
   (v2_revol12) lifts only +0.15 LOSO-equiv standalone, well below R44's predicted
   +0.5 ~ +2.5 range. With NN ensemble: NO gain (-0.10 vs iter_014).

The high per-seed variance — v4 seed 42 +5.09 but seed 7 -1.97, mean +0.30 — is itself
diagnostic: a stable signal would lift consistently across seeds. The variance pattern
is what we'd expect if ReVol is mostly fitting test-window-specific noise.

## Why this matters for iter_015

- T35-revival is **NOT a viable iter_015 candidate** (-0.10 vs iter_014).
- The R44 thesis ("regression unlocks 5 dead CE experiments") holds for the
  F-F-rejection half (no harm), but its *positive* gain prediction overestimates
  what ReVol delivers under regression at the iter_014 ensemble level.
- The reasonable next moves are R44 top-1 (CatBoost-regression — diversity from a
  different boosting algorithm has a stronger theoretical basis: HYD's Optiver-2023
  1st-place recipe used CatBoost-MAE at 50% weight) and R44 top-2 (alpha101+alpha191
  at h=60 + regression — those alphas were designed for signed-return regression,
  cache `schemeI_*.npz` already exists).
- Lower priority: R44 top-4 (T57 magnitude weight, 1-line change) is still worth a
  cheap try; R44 top-5 (pseudo-distill) bears test-leakage risk.

## Files in `experiments/T91_revol_regr_revival/`

| file | role |
|---|---|
| `train_regr.py` | training script (3 variants via `--variant` flag) |
| `ev_gate_eval.py` | DE asymmetric EV-gate sweep + NN ensemble eval |
| `model_T91_<variant>_seed{S}.txt` | LightGBM model artifacts (5 seeds × 3 variants = 15 models) |
| `pred_T91_<variant>_seed{S}.parquet` | per-row test predictions for ensembling |
| `summary_T91_<variant>_seed{S}.json` | per-run metrics |
| `ev_gate_results_<variant>.json` | per-variant DE-EV results + ensemble grid |
| `results.json` | combined T91 final results |
| `logs/` | training + EV-gate eval logs |

## Compliance

- ✅ no `sym`, no `date`, no `time` in features (392 names checked, no leak)
- ✅ ReVol/Savgol cached features use only causal past-window (T35 build was already
  compliant; cache reused untouched)
- ✅ Predictor is stateless and shuffle-invariant (5-seed avg + threshold = pointwise)
- ✅ Out-of-train-sym safe (single global threshold pair across syms)

## RESULT

```
RESULT: task=t91_revol_regr metrics={n_features=371(best=v2_revol12), loso_equiv=36.38(LGB-only), vs_T75=+0.15, best_ensemble_loso_equiv=38.18, vs_iter014=-0.10} notes=R44 top-3 ReVol revival PARTIALLY DISCONFIRMED — F-F false-signal rejection confirmed (no -0.77 CE drop), but "ReVol becomes true signal under regression" not supported (within-noise standalone, no NN-ensemble gain). Recommend deprioritizing T35-revival; pursue R44 top-1 (CatBoost-regression) and top-2 (alpha101 at h60+regr).
```
