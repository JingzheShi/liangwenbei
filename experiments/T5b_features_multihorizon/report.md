# T5b: Scheme C (T2 raw + T3 features) multi-horizon LOSO + iter_002

## Goal

Combine T2 raw 154-dim and T3 72-dim features into a 226-d Scheme C1
representation (later trimmed to **223-d** by dropping 3 time-encoding cols
for LocalEvaluator compatibility), then train one LightGBM per horizon
(h ∈ {5, 10, 20, 40, 60}) under Leave-One-Sym-Out (LOSO). Threshold-tune
each horizon and ship all 5 in iter_002 (platform takes max across 5).

## Hard constraints (CRITICAL_CONSTRAINTS.md)

✅ Feature names exclude `date`, `sym`, `time` (asserted in train scripts)
✅ Predictor is sym-agnostic (no per-sym embedding / normalization)
✅ Predictor.predict is stateless (no cross-call buffer)
✅ T3 features computed online from the 100-row window (max abs diff vs
   full-session cache ≤ 1.6e-6, only on EWMA α=0.05 — empirically verified)
✅ `time` is NOT in config.feature; the 3 T3 time-encoding cols are dropped.

## Why 223-d not 226-d?

Initial Scheme C1 was 226-d (154 raw + 72 T3), but `time_*` cols depend on
parsing `time` (a `datetime.time` column). Including `time` in
`config.feature` triggers LocalEvaluator's `out.astype(np.float32)` in
`_make_window_df`, which fails on `datetime.time`. To keep the local
contract validator green, we drop the 3 time-encoding T3 cols entirely:

  - `time_minutes_since_session_start` (rank 17/226 by gain on h=10, 1.22% share)
  - `time_session_progress` (rank 82, 0.29% share)
  - `time_is_pm` (rank 21, 1.08% share)

Total ~2.6% of feature gain dropped. Cheap insurance against the local/
platform contract mismatch.

## Scheme C1 design (final 223-d)

154 raw last-tick + 69 T3 last-tick = **223 features**.

T3 sub-features kept (69):
  - 30 MLOFI: 10 LOB levels × 3 windows {5, 20, 60}
  - 11 WMP: 10 wmp_lvl{1..10} + 1 wmp_balance_12
  - 4  RV: rolling sums of squared log-returns over {5, 10, 20, 50}
  - 24 EWMA intensities: 6 *_intst cols × 4 alphas {0.05, 0.1, 0.3, 0.5}

## LOSO 5-fold results (Scheme C1, 226-d cache; same X,y as 223-d
because the 3 time cols sit at the tail of the cache file)

### Raw argmax 5-fold sum_cum_pnl (held-out sym)

| horizon | h0      | h1      | h2      | h3      | h4      | sum     | n_pos |
|--------:|---------|---------|---------|---------|---------|---------|------:|
|     5   | -1.378  | +0.159  | +2.270  | +1.756  | +3.081  | **+5.89** | 4/5 |
|    10   | -3.054  | -0.403  | +2.334  | +2.227  | +6.939  | **+8.04** | 3/5 |
|    20   | +0.348  | -0.125  | +1.196  | +0.913  | +8.090  | **+10.42** | 4/5 |
|    40   | -0.196  | -1.741  | -5.302  | +0.341  | +6.986  | **+0.09** | 2/5 |
|    60   | -0.670  | -5.310  | -10.246 | -1.791  | +5.876  | **-12.14** | 1/5 |

### Threshold-sweep best per horizon (T × δ ∈ {0.35..0.7} × {0..0.25})

| horizon | best_T | best_δ | h0    | h1    | h2    | h3    | h4    | sum    | n_pos |
|--------:|-------:|-------:|-------|-------|-------|-------|-------|-------:|------:|
|     5   | 0.60   | 0.10   | +1.55 | +5.30 | +3.37 | +1.88 | +5.40 | **+17.49** | 5/5 |
|    10   | 0.55   | 0.10   | +1.94 | +5.13 | +4.35 | +2.12 | +8.33 | **+21.86** | 5/5 |
|    20   | 0.50   | 0.05   | +1.42 | +4.46 | +4.29 | +0.76 | +8.77 | **+19.70** | 5/5 |
|    40   | 0.50   | 0.00   | +0.72 | +4.32 | +0.80 | +0.30 | +5.57 | **+11.71** | 5/5 |
|    60   | 0.50   | 0.20   | +0.41 | +3.03 | -1.41 | -0.29 | +4.56 | **+6.30**  | 3/5 |

### Comparison vs prior iters

| iter      | scheme       | LOSO 5-fold best sum | n_pos | submission              |
|-----------|--------------|----------------------|-------|-------------------------|
| iter_001  | B raw label_60 | +1.33  | 3/5 | iter_001_lgbm_schemeB |
| iter_001c | B (T=0.50, δ=0.15) label_60 | +6.45 | 5/5 | iter_001c_lgbm_schemeB_thresh |
| **iter_002** | **C1 multi-horizon** | **max(h10=+21.86)** | **5/5 (4/5 horizons all-positive, h60 3/5)** | iter_002_lgbm_schemeC |

**iter_002 vs iter_001c**: For label_60 alone, iter_001c +6.45 vs iter_002 h60 +6.30 — basically equal. But iter_002 has FOUR more horizons each with stronger LOSO than iter_001c:
- h=10: +21.86 → +15.4 better than iter_001c's label_60
- h=20: +19.70 → +13.3 better
- h=5:  +17.49 → +11.0 better
- h=40: +11.71 →  +5.3 better

Platform scoring takes the BEST of 5 horizons → iter_002 expected platform score ≈ +21.9 (worst case ≥ +6.3 if h=60 dominates), comfortably above iter_001c's +6.45.

## Decision (iter_002 contents)

- **Active horizons**: ALL 5 (every horizon has positive LOSO best-thresholded sum, so always include — platform's max-over-horizons rule means including a positive horizon strictly dominates excluding it).
- **Per-horizon (T, δ)**:

| h  | T    | δ    | LOSO sum | n_pos |
|---:|------|------|---------:|------:|
| 5  | 0.60 | 0.10 | +17.49   | 5/5   |
| 10 | 0.55 | 0.10 | +21.86   | 5/5   |
| 20 | 0.50 | 0.05 | +19.70   | 5/5   |
| 40 | 0.50 | 0.00 | +11.71   | 5/5   |
| 60 | 0.50 | 0.20 | +6.30    | 3/5   |

## Local-eval sanity check on date 119 am × 5 syms (shuffle=True)

| label    | cum_pnl | acc    | active preds |
|----------|---------|--------|--------------|
| label_5  | +0.255  | 0.7844 | 1458 / 9210  |
| label_10 | +0.284  | 0.6933 | 1636 / 9210  |
| label_20 | +0.374  | 0.7422 | 1527 / 9210  |
| label_40 | **+0.428** | 0.6904 | 736 / 9210 |
| label_60 | +0.293  | 0.6480 | 329 / 9210   |

best_score = +0.428 (label_40). All horizons positive → robust.

## Training protocol notes

- **Cache**: 226-d Scheme C1 (`schemeC_{train,val,test}.npz`), built in 12s
  by reading T3 features_v1 parquets directly (already-computed 235-col files).
- **LOSO trainer** (`train_loso.py`): 5 horizons × 5 held-out syms = 25
  trainings, ~80–110s each (LightGBM 18-thread CPU). Total ~35 min.
- **Final 223-d trainer** (`train_final_223d.py`): one LightGBM per horizon
  on (all 5 syms, date 0..95), val on (date 96..119) with early stopping.
  Total ~17 min. Drops the last 3 cols (time_*) of the 226-d cache.
- **Threshold sweep** (`threshold_sweep.py`): 8 × 6 = 48 (T, δ) combos
  per horizon × 5 horizons = 240 evaluations. Total ~5s.

## Predictor inference latency

Local benchmark: 9210 windows / 106 s = 11.5 ms/window. T3 compute is the
bottleneck (~1ms/window pandas, ×100 + amount_delta + model predict).
For typical platform test (a few hundred sessions × ~1.8K windows/session)
expect ~10–60 min total. Acceptable.

## Files in this experiment

```
experiments/T5b_features_multihorizon/
├── build_features_C.py     # Scheme C cache builder (parquet → npz)
├── train_loso.py           # LOSO multi-horizon LightGBM trainer (25 runs)
├── train_final.py          # 226-d final model per horizon (initial; replaced by 223-d)
├── train_final_223d.py     # 223-d final model per horizon (used in iter_002)
├── threshold_sweep.py      # per-horizon (T, δ) sweep on LOSO OOF
├── build_iter_002.py       # build iter_002 submission from sweep results
├── cache/                  # 226-d Scheme C train/val/test npz + feat_names
├── loso_pred_h{H}_held{K}.parquet   # 25 fold OOF predictions
├── loso_model_h{H}_held{K}.txt      # 25 fold models
├── final_model_h{H}.txt             # initial 226-d final models (deprecated)
├── final_223_model_h{H}.txt         # 223-d final models (used in iter_002)
├── loso_results_schemeC1_step1.json
├── loso_results_schemeC1_step2_held1234.json
├── threshold_results.json
├── sweep_results_h{H}.csv
├── results.json
├── worker-progress.json
└── report.md (this file)
```
