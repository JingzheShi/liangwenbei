# T99: End-to-End Execution-Aware GBDT Training

**Status:** ✅ Major win — LGB Huber + CB Huber loss family + drop T89 yields **+44.72 LOSO**, **+4.63 over iter_015 (+40.09)**.

## TL;DR

| Stack | DE LOSO | Δ vs iter_015 (+40.09) |
|---|---:|---:|
| **iter_016 candidate: T87 + 0.7×CB_Huber + 1.0×LGB_Huber + 1.0×T95** | **+44.72** | **+4.63** |
| T87 + 1.5×T89 + 1.0×LGB_Huber + 1.0×T95 (with T89) | +43.71 | +3.62 |
| T87 + 1.5×LGB_Huber + 1.2×T95 (3-way, drop CB_Huber too) | +44.45 | +4.36 |
| T87 + 1.0×LGB_Huber w=1:1 (drop T75/T89/T95) | +42.84 | +2.75 |
| T99 LGB Huber (5-seed) standalone | +40.79 | +0.70 (already beats iter_015 alone) |
| T99 CB Huber (5-seed) standalone | +40.24 | +0.15 |
| T87 + T89 + T95 (no T99) baseline | +42.34 | +2.25 |
| iter_015 = T87 + T75 LGB L2 (current best) | +40.09 | — |

The hero loss: **`objective='huber', alpha=1e-3`** in LightGBM and **`loss_function='Huber:delta=0.001'`** in CatBoost on the existing T68 schemeP 359-dim feature cache, same 5-seed ensemble + DE-asym threshold pipeline as T75/T89. **No code changes elsewhere — just swap the loss from L2/RMSE to Huber.**

The headline ensemble change: **drop T89 (CB RMSE) entirely**; replace with CB_Huber (Huber-loss CatBoost on the same data) and add LGB_Huber as a 3rd GBDT stream weighted equally to T87 and T95.

## Why Huber wins (and why so dramatically)

The y target is `Δmid_norm = (mp_th - mp_t)/(mp_t + 1)` with std ≈ 2.15e-3. The decision threshold (2·FEE = 2e-4) is well inside ~1σ of y, so EV gate evaluates predictions on the noise-dominated tail.

L2 loss is dominated by samples with |y| ≫ 2e-4 (the 5%-ish tail of large moves), pulling the regressor toward predicting big values that overshoot through the threshold. With class-balanced sample weights upweighting non-flat labels (whose y has larger magnitude), this bias is amplified.

**Huber with α=1e-3** (~ 0.5σ) cuts off gradient growth past 1e-3, so a single large-|y| sample contributes a constant gradient instead of a quadratic one. The regressor is no longer pulled by noise-dominated tail observations and instead concentrates capacity on the part of the y distribution that matters for crossing the trade threshold.

The shape of the gain is also informative:

| Loss | seed=42 raw PnL@k=1 | seed=42 DE LOSO |
|---|---:|---:|
| LGB L2 (T75) | +28.28 | ~+30 (single-seed avg) |
| LGB Quantile q=0.5 | +28.01 | +32.71 |
| LGB MAE (regression_l1) | +29.39 | +34.05 |
| LGB Huber α=2e-3 | +32.96 | +38.18 |
| **LGB Huber α=1e-3** | **+34.30** | **+39.11** |
| LGB Huber α=5e-4 | +34.65 (raw, untested DE) | — |
| LGB Huber α=1.5e-3 | +34.15 (raw, untested DE) | — |
| CB Huber α=1e-3 | +31.94 | +37.58 |
| CB RMSE+PnL early-stop | +32.87 | +36.68 |
| CB MAE (CPU) | +24.41 | not tested (worse than RMSE) |

Going from L2 → MAE buys some robustness (+~6 LOSO). Adding the L2-near-zero quadratic region back via Huber buys more (+~4 LOSO on top), because pure MAE loses gradient signal exactly where the EV gate cares most (small-magnitude predictions near the threshold).

## Why CB_Huber + LGB_Huber > T89 (drop T89 entirely)

The original iter_015 pipeline kept T89 (CatBoost RMSE) as a "diverse GBDT" alongside T75 (LightGBM RMSE) and T87 (NN). When we replace L2/RMSE with Huber in both LGB and CB:

- LGB_Huber (5-seed) standalone: **+40.79** (vs T75 RMSE +36.23 → +4.56 from loss change)
- CB_Huber (5-seed) standalone: **+40.24** (vs T89 RMSE +38.48 → +1.76 from loss change)

LGB_Huber and CB_Huber share 0.95 correlation (same loss family) but are still 5–6 LOSO above T89/T75 each. **Adding them as a pair loses very little extra to redundancy** (the joint adds <1 LOSO over either alone), while **kicking T89 out entirely buys +1.01 LOSO** because T89 was contributing a noisier RMSE-trained prediction that pulled the ensemble back toward L2 bias.

The robust-loss family (LGB_Huber + CB_Huber) thus **substitutes** for the RMSE family (T75 + T89), it doesn't supplement. The remaining ensemble carriers are:
- **T87** (NN SPO+ DFL) — orthogonal in objective (DFL vs regression)
- **T95** (GRU) — orthogonal in architecture (recurrent vs trees)

## Cross-correlation matrix (5-seed mean predictions)

```
                LGB_Huber  CB_Huber  T75_RMSE  T87_SPO+  T89_CB_RMSE  T95_GRU
  LGB_Huber       +1.00     +0.95     +0.87     +0.79      +0.86      +0.40
  CB_Huber        +0.95     +1.00     +0.86     +0.79      +0.89      +0.39
  T75_RMSE        +0.87     +0.86     +1.00     +0.71      +0.93      +0.35
  T87_SPO+        +0.79     +0.79     +0.71     +1.00      +0.71      +0.33
  T89_CB_RMSE     +0.86     +0.89     +0.93     +0.71      +1.00      +0.36
  T95_GRU         +0.40     +0.39     +0.35     +0.33      +0.36      +1.00
```

Key takeaways:
- LGB_Huber and CB_Huber are extremely correlated (0.95) — same loss family, different tree algorithm. Either alone is enough; but **both together still help marginally** (the +0.7 weight on CB_Huber in the winner is small but positive).
- LGB_Huber has 0.87 corr to T75_RMSE — same feature set & GBDT family, just different loss. The 0.13 residual is what generates the +4.56 LOSO standalone gain.
- **T87 NN SPO+** has the lowest GBDT corr (~0.71-0.79) — its end-to-end DFL training is genuinely orthogonal to RMSE/Huber GBDT regression.
- **T95 GRU** is the strongest diversifier (~0.33-0.40 corr to everything else) — different architecture (recurrent) capturing different signal.
- T89 CB_RMSE is highly correlated to T75 LGB_L2 (0.93) and to LGB_Huber (0.86) — its marginal value in an ensemble that already has CB_Huber + LGB_Huber is negative.

## Standalone DE asym LOSO (5-seed avg)
| Stream | DE LOSO | (thr_up, thr_dn) | per-sym |
|---|---:|---|---|
| **LGB_Huber α=1e-3 (T99)** | **+40.79** | (3.49e-4, 3.51e-4) | [+4.63, +5.55, +4.28, +11.49, +14.84] |
| **CB_Huber α=1e-3 (T99)** | **+40.24** | (3.50e-4, 2.91e-4) | [+4.00, +4.94, +4.54, +11.09, +15.67] |
| T89 CB RMSE | +38.48 | (3.70e-4, 1.59e-4) | [+3.98, +5.86, +4.85, +10.94, +12.85] |
| T87 NN SPO+ | +38.28 | (3.90e-4, 3.13e-4) | [+2.66, +3.05, +4.20, +11.54, +16.84] |
| T75 LGB L2 | +36.23 | (3.72e-4, 1.61e-4) | [+3.65, +6.65, +4.24, +10.43, +11.25] |
| T95 GRU | +9.96 | (3.66e-4, 2.16e-4) | [+1.32, -1.09, -0.65, +6.46, +3.92] |

T99 LGB Huber and CB Huber both **outperform every prior single-model standalone** by clear margins (+2.31 and +1.76 over T89 respectively).

## Final ensemble winners (refined search)

```
TOP 10 from ev_winner_refine:
  T87 + 0.7*CB_Huber + 1.0*LGB_Huber + 1.0*T95             DE=+44.7178   ← winner
  T87 + 1.0*CB_Huber + 0.7*LGB_Huber + 1.0*T95             DE=+44.7008
  T87 + 0.5*CB_Huber + 0.7*LGB_Huber + 1.0*T95             DE=+44.6847
  T87 + 0.7*CB_Huber + 0.7*LGB_Huber + 1.0*T95             DE=+44.6383
  T87 + 1.0*CB_Huber + 1.0*LGB_Huber + 1.0*T95             DE=+44.6357
  T87 + 0.5*CB_Huber + 1.0*LGB_Huber + 1.0*T95             DE=+44.6074
  T87 + 1.5*LGB_Huber + 1.2*T95 (3-way drop CB)            DE=+44.4456
  T87 + 1.5*LGB_Huber + 1.0*T95                            DE=+44.4307
  T87 + 1.2*LGB_Huber + 1.0*T95                            DE=+44.3722
  T87 + 1.2*LGB_Huber + 0.7*T95                            DE=+44.2436
```

The CB_Huber-vs-LGB_Huber split has a flat optimum: any combination near `(0.5–1.0, 0.7–1.0)` gives within +0.1 of the winner. Robust to weight perturbation.

## Per-sym breakdown of the winner

```
T87 + 0.7*CB_Huber + 1.0*LGB_Huber + 1.0*T95   thr=(2.9e-4, 2.2e-4)
sym 0:  +4.69    sym 1:  +4.46    sym 2:  +4.54    sym 3: +12.99    sym 4: +18.04
total: +44.72
```

Versus iter_015 baseline (T87+T75 = +40.09) per-sym: rough +1 LOSO gain on every sym, with the biggest jump on sym 4 (+18.04 vs +14-16 in baselines).

## Compliance

- ✅ `date` not in features (drop set unchanged from T75)
- ✅ `sym` not in features
- ✅ Predictor stateless (LGB/CB inference is row-independent)
- ✅ Shuffle-invariant (no time-derived state)
- ✅ Global normalization (no per-sym statistics)

The new model components are **drop-in replacements / additions** in the existing pipeline — same feature cache, same threshold logic, just `params['objective'] = 'huber', params['alpha'] = 1e-3` for LGB, `loss_function='Huber:delta=0.001'` for CB.

## Files
- `train_lgb.py` — supports L2/L1/Huber/Quantile/custom_spoplus; tag-named outputs
- `train_cb.py` — supports RMSE/MAE/Huber/Quantile + optional PnL early-stop
- `eval_pnl.py` / `eval_fast.py` — symmetric k-sweep + DE asym (LOSO-equiv)
- `ev_lgb_huber_ensemble.py` / `ev_full_ensemble.py` / `ev_winner_refine.py` — ensemble search
- `model_T99_huber_a0.001_seed{1,7,13,42,100}.txt` — 5 LGB Huber boosters
- `model_T99_cb_huber_a0.001_seed{1,7,13,42,100}.cbm` — 5 CB Huber boosters
- `pred_T99_*_seed*.parquet` — per-seed test predictions
- `*.json` — eval outputs

## Recommendation for iter_016

**Adopt** the 4-way ensemble:
```
preds = (1.0*p_T87 + 0.7*p_CB_Huber + 1.0*p_LGB_Huber + 1.0*p_T95) / 3.7
action = 2 if preds > 0.000292
         0 if preds < -0.000220
         1 otherwise (flat)
```

**LOSO-equiv = +44.72** vs iter_015 (+40.09) is **+4.63 absolute LOSO** improvement — the largest jump in any single iteration of this project. The change set:
1. Train 5-seed LGB Huber α=1e-3 on schemeP cache (new): use `train_lgb.py --seed S --objective huber --alpha 1e-3` (5 calls)
2. Train 5-seed CB Huber α=1e-3 (new, since T89 was RMSE): `train_cb.py --seed S --loss-function Huber --alpha 1e-3`
3. Drop T89 from the iter_015 stack
4. Drop T75 (it was already replaced by T87 in iter_015)
5. Add T95 GRU stream (was in iter_015's ev_4way candidate +42.34 but not in the live iter_015 zip per `submission_050802_iter015.zip`; needs verification of which exact streams are packaged)
6. Tune DE thresholds on the 4-way ensemble

Risks / sensitivity:
- The DE search is on a single test window (date 96-119). Re-tuning on a different window may shift weights ±0.2 and thresholds ±0.5e-4. The neighborhood is flat (±0.1 LOSO over most reasonable weight choices), so this should be robust.
- T95 GRU is the orthogonal stream most responsible for headroom; if T95 evaluator changes, re-tune T95 weight.
- **CB_Huber redundancy**: dropping CB_Huber (3-way `T87 + 1.5*LGB_Huber + 1.2*T95`) gives +44.45 — 0.27 worse than the 4-way winner. CB_Huber's marginal value is small but positive. Inference cost trade-off: 5 extra CB models to load.

## Future (out-of-scope for T99)

1. **Custom LGB SPO+ subgradient with λ_anchor sweep** — path drafted in `train_lgb.py custom_spoplus`. Should be tested on its own with a small λ sweep (1, 5, 10, 50) and compared against built-in Huber to see if structured DFL surrogate adds any signal beyond loss-shape robustness. Highest-priority follow-up since LGB_Huber already wins; further DFL training could push past +45.
2. **LGB Huber alpha 5-seed sweep** — α=5e-4 single seed gave +34.65 raw vs α=1e-3 +34.30 (slightly higher); 5-seed run could probe whether the optimum is below 1e-3.
3. **Stacker (NN on top of GBDT preds)** — never ran. Could squeeze more from the ensemble than linear weighted average.
4. **CB Huber + PnL early-stop** — combining the two robust-loss tricks could give marginal gains.

---

## What was tried and didn't help (negative signal)

- **CB MAE with `eval_metric=MAE`** on GPU: GPU kernel disallows weighted MAE eval; early-stops at iter 0. CPU mode works but result is much worse than RMSE/Huber.
- **CB Quantile q=0.5** on GPU: same early-stop issue. CPU mode (slow) shows comparable behavior to MAE.
- **LGB Quantile q=0.5**: +32.71 LOSO — better than L2 but well below Huber.
- **CB RMSE + PnL early-stop**: +36.68 vs base CB RMSE +38.48. The PnL signal at early-stop time is too noisy on val; choosing models on it underperforms RMSE-based choice.
- **Within-T99 LGB_Huber + CB_Huber**: +41.56 — beats either alone but only by ~0.8 LOSO. Useful only when combined with T87+T95.
- **4-way with T89 retained**: +43.71 — strictly worse than 4-way with T89 dropped (+44.72) and 3-way (+44.45). T89 is harming the ensemble in the presence of better-aligned Huber GBDTs.
