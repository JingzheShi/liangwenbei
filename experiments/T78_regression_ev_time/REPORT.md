# T78 — Regression on Δmid + EV-gate + 6 time features (ablation)

## TL;DR

**Adding the 6 schemeQ time features to the iter_013 regression-EV stack does NOT compound; LOSO-equiv drops from +36.23 → +35.51 (-0.72).** Same disappointing pattern as T74 (time features added on top of 3-class classification, which gave -0.x). Conclusion: the LOB microstructure features already implicitly carry whatever session-position signal exists at horizon 60. Explicit cyclic-encoding of t/sess_idx primarily adds a noisy extra split dimension that overfits the training-window's intraday-volatility quirks.

**iter_013 remains the SOTA submission.** No iter_014 to package.

## Setup

| | T75 (iter_013, baseline) | T78 (this) |
|---|---|---|
| Cache | T68 schemeP (370 dims, 359 after drop) | T74 schemeQ (376 dims, **365 after drop**) |
| Extra features | — | sec_norm, sin/cos sess pos, is_open_30min, is_close_30min, is_pm |
| Objective | regression_l2 | regression_l2 |
| Target | (mp_t60 - mp_t) / (mp_t + 1) | same |
| Seeds | 5 (42, 1, 7, 13, 100), hyperparam-diverse | same configs |
| aug_a | [0.80, 1.20] per-(sample, feat) scale | same |
| Validation | V4 walk-forward (date 0-75 train / 76-79 val) | same |
| Train rounds | num_boost=600, ES=40 on val MSE | same |
| Backend | LightGBM GPU | same |
| Decision | EV-gate, 2D DE search (thr_up, thr_dn) | same |

## Results (5-seed ensemble, 442 080 test rows, 5 syms)

### Per-seed (single-model k=1 EV-gate)

| seed | T75 | T78 | Δ | T78 best_iter | T78 test corr |
|------|------|------|---------|---------------|---------------|
| 42  | +28.28 | +31.23 | +2.95 | 66 | 0.1529 |
| 1   | +30.55 | +29.31 | -1.25 | 73 | 0.1503 |
| 7   | +31.02 | +29.51 | -1.51 | 78 | 0.1530 |
| 13  | +31.30 | +29.45 | -1.85 | 68 | 0.1511 |
| 100 | +29.85 | +28.62 | -1.23 | 124 | 0.1429 |
| **mean** | **+30.20** | **+29.62** | **-0.58** | — | 0.150 |

Single-seed mean is slightly worse with time features. Test correlation is essentially flat (0.150 vs 0.153).

### 5-seed ensemble · symmetric k sweep

| k | thr | T78 sum_per_sym | T75 sum_per_sym | Δ |
|---|---|---|---|---|
| 0.50 | 1e-4 | +23.65 | +24.26 | -0.61 |
| 1.00 | 2e-4 | +31.99 | +32.48 | -0.49 |
| **1.25** | **2.5e-4** | **+32.75** | +33.72 | -0.97 |
| 1.50 | 3e-4 | +31.94 | +33.28 | -1.34 |
| 2.00 | 4e-4 | +28.56 | +29.89 | -1.33 |

### 5-seed ensemble · DE asymmetric (the real headline)

| | T75 (iter_013) | T78 (time) | Δ |
|---|---|---|---|
| thr_up | 0.000372 | 0.000354 | -0.000018 |
| thr_dn | 0.000161 | 0.000176 | +0.000014 |
| **LOSO-equiv** | **+36.2281** | **+35.5116** | **-0.7165** |
| per-sym sym0 | +3.65 | +4.40 | +0.75 |
| per-sym sym1 | +6.65 | +5.65 | -1.00 |
| per-sym sym2 | +4.24 | +3.71 | -0.53 |
| per-sym sym3 | +10.43 | +10.44 | +0.00 |
| per-sym sym4 | +11.25 | +11.32 | +0.07 |

The optimal thresholds barely move. The pnl shifts within syms: gain +0.75 on sym0, lose ~1.5 on sym1+sym2 combined, sym3/sym4 unchanged.

## Why time features didn't help even for regression

Going in, the hypothesis was: "T74 = time + classification = small loss because classification only needs the *sign* of Δmid, and the LOB already encodes the sign-direction. But regression needs the *magnitude*, and time-of-day strongly modulates volatility — open volatility ↑, mid stable, close volatility ↑ — so T78 = regression + time should compound."

What actually happens, from the numbers:

1. **The LOB carries time-of-day implicitly.** Spread widths, trade intensity bursts, and order-flow imbalance all have intraday seasonalities that a depth-127 tree forest with 359 features picks up without needing an explicit `t / 2000` feature. The marginal information of `sec_norm` etc. is small.
2. **The regression objective is dominated by signal direction × magnitude, but the optimal EV gate trades only when |pred| crosses ~2× FEE, far above the noise floor.** The set of trades that the gate would *change* due to a +ε from a time feature is tiny — most rows have |pred| nowhere near the threshold.
3. **Time features add 6 splits of degrees of freedom that the model uses to memorize date-window-specific quirks.** Training dates 0-75 ≠ test dates 96-119 (gap of 16+ days), and intraday-volatility patterns drift; an `is_open_30min` interaction that helps in train can flip sign in test.
4. **Per-sym redistribution** (+sym0, -sym1/2) is consistent with the model trading off slightly different intraday timing in each sym — there's no universal gain, just shuffling.

Note that single-seed seed=42 went from +28.28 to +31.23 (+2.95). That's the optimistic outlier — the other 4 seeds all regress by ~1.2-1.9, so the ensemble loses. The seed-42 gain didn't translate into ensemble gain.

## Decision

| threshold | result |
|---|---|
| > +37 (clear win) | NO (+35.51) |
| > +36.23 (strict improvement, save OOF) | NO |
| < +36.23 (no help) | **YES** |

→ **iter_013 stays as the SOTA. Time features are neutral-to-slightly-negative for both classification (T74) and regression (T78).** Don't pursue this direction further at horizon 60. If anyone wants to revisit time features, the angle to try is interaction features (e.g. spread_W20 * sin_sess_pos) rather than additive ones — the tree booster can't easily synthesize trigonometric interactions on its own.

## Compliance (no submission packaged, but features are still constraint-safe)

- ✅ `sym` not used as input
- ✅ `date` not used as input
- ✅ time features computed from `t` and `sess_idx` only — both are window-position quantities, derivable from the input batch alone (no cross-batch state, no calendar info)
- ✅ Stateless and shuffle-invariant by construction (per T74's design)

## Artifacts saved

```
experiments/T78_regression_ev_time/
├── REPORT.md
├── results.json
├── ev_gate_results.json
├── ev_gate_eval.log
├── train_regr_q.py        # train script (loads schemeQ; dropped 11 fail features)
├── ev_gate_eval_q.py      # 2D DE search on 5-seed avg pred
├── train_seed{1,7,13,42,100}.log
├── summary_T78_seed{1,7,13,42,100}.json
├── model_T78_seed{1,7,13,42,100}.txt    # 5 boosters, regression_l2, GPU
└── pred_T78_seed{1,7,13,42,100}.parquet # test-set predicted Δmid_norm
```

## Next steps (recommendations to PM)

1. **iter_013 is the production submission**; don't replace it with iter_014.
2. **Stop testing time features** at horizon 60 — both objectives say no.
3. Direction worth more compute: **alternative targets** (e.g. log-return, Sharpe-equivalent, or per-sym scaled return) may unlock magnitude information the current regression target misses. If T78's seed-42 gain was real, it suggests the regressor *can* improve with the right inductive bias.
4. Direction worth more compute: **larger / different model class** (GBM with deeper interactions, or a small MLP head on top of the 365-d features) — within iter_013's regression-EV framework, the LightGBM may be saturated.
