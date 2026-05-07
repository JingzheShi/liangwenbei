# T45 Group DRO LightGBM — Bug fix + LOSO + DE thresh

## TL;DR

- **Bug fixed**: callback now fires (3-4× per fold). Previous `n_dro_updates: 0` was an off-by-K error — `K=200` with `best_iter ~120 + early_stopping=40` meant the first scheduled fire (`it % 200 == 0`) never happened before training stopped.
- **Group DRO meaningfully helps single-seed h60**: sum_cum_pnl improves from **T44 baseline -5.65 → T45 +3.83** (raw argmax, +9.48 absolute). With per-T45 DE-tuned asymmetric thresholds → **+8.66** (4/5 positive folds).
- **Does NOT beat +13.61 5-seed ensemble baseline**: best T45 (single-seed) is **-4.95 below** the bar. **Negative result** per task gating.

## What was wrong

`experiments/T45_group_dro/train_loso.py` GroupDROCallback (line 144 before fix):

```python
if env.iteration == 0 or env.iteration % self.K != 0:
    return
```

With `K=200`, `early_stopping=40`, models early-stop at `best_iter ~120-165`. First scheduled fire would have been at iter 200 — strictly past every observed `best_iter`. All four tau ablations recorded `n_dro_updates: 0` and were effectively *just the baseline aug_a model*, explaining why the tau ablation showed no signal.

## Fix

`experiments/T45_group_dro/train_loso.py`:

- New CLI flag `--first-update-at INT` (default 20)
- Default `--K` reduced 200 → 50
- Callback fires when `iteration == first_update_at`, then every K rounds thereafter (iter 20, 70, 120, 170, …)
- Verified by unit test (callback object replayed iter 0..200) and live training logs
  (smoke run at `K=50, fua=20`: fires at 20/70/120, `best_iter=120`, `n_dro_updates=3`)

## Smoke test (fold 2 only, tau=0.5, K=50)

```
[DRO@iter20]  sym_loss=[0.92,1.01,—,0.80,1.04]  w=[0.23,0.27,0.04,0.18,0.29]
[DRO@iter70]  sym_loss=[0.87,0.97,—,0.66,1.01]  w=[0.23,0.28,0.04,0.15,0.30]
[DRO@iter120] sym_loss=[0.83,0.93,—,0.58,0.98]  w=[0.23,0.28,0.04,0.14,0.31]
best_iter=120, n_dro_updates=3, fold2 cum_pnl=-3.105
```

`-3.10 > -5` gate → ran full LOSO.

(sym 2 is held out — its weight is residual softmax mass, a few % only.)

## Full LOSO (single seed=42, tau=0.5, K=50, first_update_at=20)

| held | best_iter | n_dro_updates | cum_pnl  | acc    | T44 baseline cum_pnl | Δ vs T44 |
|------|-----------|---------------|----------|--------|----------------------|----------|
| 0    | 162       | 4             | -0.766   | 0.777  | -1.163               | +0.397   |
| 1    | 124       | 3             | +1.824   | 0.464  | -2.736               | +4.560   |
| 2    | 121       | 3             | -2.796   | 0.527  | -8.894               | +6.098   |
| 3    | 159       | 4             | +0.991   | 0.786  | +0.253               | +0.738   |
| 4    | 150       | 4             | +4.581   | 0.497  | +6.894               | -2.313   |
| **sum** |        |               | **+3.834** |      | **-5.648**           | **+9.482** |

Group DRO substantially rescues the worst LOSO fold (sym 2, -8.89 → -2.80). The held-out sym typically has highest cross-entropy on the rest-of-training models, so up-weighting hard syms pushes the model toward features that generalize cross-sym. The cost is fold 4 (-2.3): sym 4 had the highest loss in 4/5 folds (consistently w≈0.30), so up-weighting sym 4 trains a model less specialized for the easier syms.

## DE asymmetric thresholding

Applied two ways:

1. **iter_006 DE optimum unchanged** (`Tu=0.448, Td=0.391, du=0.260, dd=0.024`):
   sum = **+3.06**, all 5 folds nudged toward 0 from raw

2. **Per-T45 coarse grid** (Tu/Td step 0.02, du/dd step 0.04, 4320 combos, 37s):
   - **Best**: `Tu=0.38, Td=0.46, du=0.00, dd=0.00` → sum = **+8.66** (4/5 pos)
   - Per-fold: `[+0.20, +3.17, -1.70, +0.79, +6.20]`
   - Top of grid is a stable ridge — top 20 configs all `Tu∈[0.34,0.38]`, `Td∈[0.44,0.48]`, very small `du/dd` → much less aggressive on `up` than iter_006's `du=0.26`. Single-seed T45 needs a different threshold profile than the 5-seed ensemble.

## Comparison vs +13.61 bar

```
ensemble baseline (iter_006, 5-seed, asym thresh)   = +13.61
T45 single-seed Group DRO + best T45 thresh         =  +8.66    Δ = -4.95
T45 single-seed Group DRO + iter_006 thresh         =  +3.06    Δ = -10.55
T45 single-seed Group DRO raw argmax                =  +3.83    Δ = -9.78
T44 single-seed (no DRO) raw argmax                 =  -5.65    Δ = -19.26
```

→ **Negative result vs the +13.61 spec gate.**

## Interpretation

Group DRO lifts single-seed h60 LOSO by ~+9 PnL (raw) and by an additional ~+5 with per-T45 thresholds, but the gap from a single-seed run to the 5-seed iter_006 ensemble is too large to close with one technique. The "right" follow-up — running 5 seeds with Group DRO and ensembling — was outside the time budget here, but the per-fold profile (folds 1 & 2 lifted from negative to neutral/positive without destroying fold 4) suggests ensembling 5 seeds of T45 could plausibly land closer to or above +13.61. Recommend that as the next experiment if PM wants to pursue.

## Artifacts

- `train_loso.py` (fixed)
- `de_thresh.py` — new sweep + iter_006 application script
- `results_tau_0_5_K50.json` — fold 2 smoke
- `results_tau_0_5_K50_loso.json` — full 5-fold seed 42
- `tau_0_5_K50_loso_pred_seed42_held{0..4}.parquet` — OOF predictions
- `tau_0_5_K50_loso_drohist_seed42_held{0..4}.json` — per-fire sym losses + weights
- `tau_0_5_K50_loso_model_seed42_held{0..4}.txt` — saved boosters
- `de_thresh_grid.csv` — full 4320-config grid
- `de_thresh_results.json` — DE summary
- `smoke_K50.log`, `loso_K50.log` — training logs
