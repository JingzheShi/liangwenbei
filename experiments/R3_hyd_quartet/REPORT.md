# R3 — HYD Interaction Quartet (T110 #1)

**Date:** 2026-05-08
**Status:** ✅ Trick wins LOSO. Recommend as iter_017 candidate (with caveat).

## TL;DR

| arm | n_features | LOSO (3-seed ensemble) | Δ vs baseline |
|---|---|---:|---:|
| baseline (schemeP, no HYD) | 359 | **39.82** | — |
| trick (schemeP + HYD-16)   | 375 | **41.24** | **+1.42** |

- **All 3 individual seeds** show positive delta: {seed1: +1.89, seed7: +2.27, seed42: +0.13}, mean = +1.43, std = 0.93
- **Per-sym min delta** (ensemble): −0.859 (sym 1 hurt; sym 4 +1.56 dominates)
- **LOSO criterion +0.5: PASSED** → iter_017 candidate
- Modest improvement; the +1.42 LOSO is well above the +0.5 bar but per-sym risk on sym 1 is real

## Hypothesis tested (T110 #1)

HYD 1st-place team used 4 multiplicative interaction features in their LOB encoder. Our schemeP (359-d) has linear/single-feature signals but no `A·B` cross terms. LightGBM cannot recover `A·B` from a single split, so explicit interaction features should help.

## 16-dim HYD interaction features

For each 100-tick window, 4 base interaction signals computed per tick:
```
pp  = (bsize1 - asize1) * (ask1 - bid1)                 # price_pressure (signed level-1 imbalance × spread)
mu  = (ask1 - bid1) * (totalbsize - totalasize)/depth   # market_urgency (spread × normalized imbalance)
dp  = (totalasize - totalbsize) * (avgask - ask1)       # depth_pressure (depth diff × ask-side slope)
sdr = (ask1 - bid1) / (totalbsize + totalasize + eps)   # spread/depth_ratio
```

Per window we summarize each base signal as: last-tick value, mean over last 5 ticks, mean over last 20 ticks, mean over last 50 ticks → 4×4 = **16 dims**.

All sym-agnostic, stateless within window, no `date`/`sym` use → CRITICAL_CONSTRAINTS compliant.

## Setup

- **Recipe**: T117 with `--no-monotone` (= T99 baseline). LightGBM Huber α=1e-3, GPU, num_boost_round=600, early_stopping=40, learning_rate=0.05, aug_a per-feature scale [0.8, 1.2] ratio=1.0, class-balanced sample weight, V4 train/val (dates 0–75 / 76–79).
- **Per-seed configs** (T117 SEED_CONFIGS): seed1 nl=127 ff=0.6 bf=0.7 l2=1.0; seed7 nl=63 ff=0.7 bf=0.85 l2=2.0; seed42 nl=127 ff=0.8 bf=0.8 l2=1.0.
- **Eval**: per-arm 3-seed mean(pred_dmid_norm), DE-optimized 2D threshold (T_up, T_dn) shared across all syms, summed PnL across 5 syms = LOSO-equivalent.
- **Note**: Originally planned for vast.ai RTX 3080 (ssh6.vast.ai:14816 was Connection refused — instance down). Pivoted to local RTX 3090 with same data (data already local).

## Per-seed details

| seed | arm | best_iter | val_corr | val_l1 | test_corr | ev-gate(k=1) PnL |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 328 | 0.1805 | 0.001521 | 0.1652 | 32.86 |
| 1 | trick    | 242 | 0.1730 | 0.001523 | 0.1677 | 37.33 |
| 7 | baseline | 338 | 0.1798 | 0.001518 | 0.1652 | 34.33 |
| 7 | trick    | 260 | 0.1717 | 0.001518 | 0.1685 | 37.85 |
| 42 | baseline | 251 | 0.1859 | 0.001515 | 0.1651 | 34.30 |
| 42 | trick    | 247 | 0.1733 | 0.001522 | 0.1644 | 34.76 |

Notable pattern: trick has **lower val_corr** but **higher test_corr** (seeds 1, 7) — the HYD interactions seem to help generalization to held-out test (dates 96–119) better than to nearby val (dates 76–79). Could indicate the interactions capture structural microstructure relations that transfer better than overfitting-prone marginal signals.

## DE thresholds and per-sym LOSO

| arm | T_up | T_dn | sym0 | sym1 | sym2 | sym3 | sym4 | LOSO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.539e-4 | 3.789e-4 | 4.485 | 5.681 | 4.139 | 11.177 | 14.343 | **39.82** |
| trick    | 3.012e-4 | 2.989e-4 | 4.478 | 4.822 | 4.231 | 11.811 | 15.903 | **41.24** |
| **delta** | | | −0.007 | **−0.859** | +0.092 | +0.634 | **+1.559** | **+1.42** |

Trick learns lower (more aggressive) thresholds → trades off some sym 1 PnL for larger sym 3/4 gains. Per-sym min delta = **−0.859 (sym 1)**, max = +1.56 (sym 4).

## Per-seed delta (independent runs, no ensemble)

```
seed=1   baseline=38.91  trick=40.80  delta=+1.89
seed=7   baseline=39.56  trick=41.84  delta=+2.27
seed=42  baseline=39.11  trick=39.25  delta=+0.13
mean delta=+1.43   std=0.93   min=+0.13   max=+2.27
```

3/3 seeds positive; seed 42 marginal but no regression. Mean per-seed delta (+1.43) ≈ ensemble delta (+1.42), so the gain is not an ensembling artefact.

## Verdict

- **PASS**: +1.42 LOSO ≥ +0.5 threshold → iter_017 candidate.
- **Caveat**: per-sym min delta = −0.86 (sym 1 is worse). If risk-averse on per-sym OOD, consider:
  - Use fewer HYD dims (drop W=50 means? keep only last+W=5?)
  - Sym-stratified DE threshold (decoupled — T76)
- **Comparison to recent iter candidates**: T99 (Huber win) +44.72 LOSO (full ensemble), T87+SPO+ +40.09. Pure LGB Huber baseline here is 39.82; trick's 41.24 is an ablation-fair comparison. To validate as iter_017, would need to swap trick's LGB Huber into the full T108/T109 ensemble (CB Huber + NN + GRU) and re-run DE — out of scope for this 90-min experiment.

## Artifacts

```
experiments/R3_hyd_quartet/
├── build_hyd.py                       # 16-dim HYD feature builder
├── train.py                            # unified baseline/trick trainer
├── eval.py                             # 3-seed ensemble + DE 2D + LOSO
├── cache/hyd_{train,val,test}.npy     # 16-dim HYD features
├── pred_R3_{baseline,trick}_seed{1,7,42}.parquet  # 6 prediction sets
├── model_R3_{baseline,trick}_seed*.txt           # 6 LGB models
├── summary_R3_*.json                              # per-train summaries
├── loso_results.json                              # ensemble LOSO results
├── per_seed_results.json                          # per-seed LOSO results
└── REPORT.md                                       # this file
```

```
RESULT: task=hyd_quartet metrics={baseline_loso=39.8245, trick_loso=41.2438, delta=+1.4193, delta_per_sym_min=-0.8589} notes=[3/3 seeds positive, mean per-seed delta +1.43; sym 1 hurt -0.86 ensemble; passes +0.5 LOSO bar -> iter_017 candidate (LGB Huber arm only; full ensemble validation needed)]
```
