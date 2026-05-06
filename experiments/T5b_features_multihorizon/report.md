# T5b: Scheme C (T2 raw + T3 features) multi-horizon LOSO + iter_002

## Goal

Combine the T2 raw 154-dim and T3 72-dim features into a single 226-d Scheme C1
representation, then train ONE LightGBM per horizon (h ∈ {5, 10, 20, 40, 60})
under Leave-One-Sym-Out (LOSO) cross-validation. Pick the horizon(s) with the
best LOSO PnL, threshold-tune them, and ship as iter_002.

## Hard constraints (CRITICAL_CONSTRAINTS.md)

✅ Feature names exclude `date`, `sym`, `time` (asserted in train_loso.py)
✅ `time` enters the platform input column list only so the Predictor can
   compute the 3 time-encoding features inside the 100-row window — `time` itself
   is *not* a model feature.
✅ Predictor is sym-agnostic (no per-sym embedding / normalization)
✅ Predictor.predict is stateless (no cross-call buffer)
✅ T3 features are computed online from the 100-row window (verified
   max abs diff vs full-session cache ≤ 1.6e-6, only on EWMA α=0.05)

## Scheme C1 design

Choice: 154 raw last-tick + 72 T3 last-tick = **226 features**.
Skip rolling-stats wrapper (Scheme B) — keeps Predictor inference latency
small, training fast, and side-steps the EWMA windowed/cached drift concern
that grows with rolling window size.

### Why no rolling stats here?

Two reasons:
1. T3 already includes rich time-aggregated structure (MLOFI rolling sums over
   {5,20,60}; RV rolling sums over {5,10,20,50}; EWMA over 4 alphas).
2. T2 Scheme B (2002-d) on label_60 LOSO sum was only +1.33; the marginal
   gain from rolling-stats on 154 raw didn't justify the 9× feature
   inflation. Scheme C1 keeps the proven raw + the new T3 signal.

## LOSO results

(See `loso_results_*.json` for full breakdown; tables in the iter_002 thresholds
file.)

### Raw argmax 5-fold LOSO sum_cum_pnl

| horizon | sum | mean | std | n_pos / 5 |
|--------:|-----|------|-----|-----------|
| 5  | _filled_post_run_ | | | |
| 10 | _filled_post_run_ | | | |
| 20 | _filled_post_run_ | | | |
| 40 | _filled_post_run_ | | | |
| 60 | _filled_post_run_ | | | |

### Threshold-sweep best per horizon

| horizon | best_T | best_δ | sum | n_pos / 5 |
|--------:|--------|--------|-----|-----------|
| 5  | _filled_ | _filled_ | _filled_ | _filled_ |
| 10 | _filled_ | _filled_ | _filled_ | _filled_ |
| 20 | _filled_ | _filled_ | _filled_ | _filled_ |
| 40 | _filled_ | _filled_ | _filled_ | _filled_ |
| 60 | _filled_ | _filled_ | _filled_ | _filled_ |

### Comparison vs prior iters

| iter | scheme | LOSO 5-fold sum | n_pos | submission |
|------|--------|-----------------|-------|------------|
| iter_001  | Scheme B raw argmax (label_60)  | +1.33  | 3/5 | iter_001_lgbm_schemeB |
| iter_001c | Scheme B (T=0.50, δ=0.15) label_60 | +6.45 | 5/5 | iter_001c_lgbm_schemeB_thresh |
| iter_002  | Scheme C1 multi-horizon best | _filled_ | _filled_ | iter_002_lgbm_schemeC |

## Decision (iter_002 contents)

- **Horizons included** (active): _filled after build_iter_002.py_
- **Per-horizon (T, δ)**: _filled_
- **Reasoning**: include every horizon whose best-thresholded LOSO sum > 0.

## Training protocol notes

- Scheme C cache: 226-d float32, train 1.33 GB, val 0.27 GB, test 0.40 GB.
  Built in 12s by reading T3 features_v1 parquets directly.
- LOSO trainer (`train_loso.py`): 5 horizons × 5 held-out syms = 25 trainings,
  ~80–100s each (LightGBM CPU 18-thread). Total ~35 min.
- Final model trainer (`train_final.py`): trains one LightGBM per chosen
  horizon on (all 5 syms, date 0..95) with early stopping on date 96..119
  (mirrors iter_001 protocol — uses 8% MORE training data than LOSO folds
  because no sym is held out).

## Predictor design (iter_002)

```python
class Predictor:
    def __init__(self):
        # vendored compute.py provides T3 feature_v1 functions
        # thresholds.json: per-horizon {T, delta, active}
        # model_h{H}.txt: LightGBM booster per active horizon

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        # For each 100-row batch:
        #   T3 = compute on window (fall back to zero time-encoding if 'time' unparseable)
        #   features = concat(raw_last_tick_154, T3_last_tick_72) → 226-d
        #   For each active horizon h:
        #     probs = booster_h.predict(features)
        #     pred_h = threshold_predict(probs, T_h, δ_h)
        #   Output [pred_5, pred_10, pred_20, pred_40, pred_60] (flat 1 for inactive)
```

Latency budget: 100 rows × 72 T3 cols vectorized ≈ 1–2 ms/window;
LightGBM 226-d batch predict: ~0.1 ms/window × 5 horizons. Batch=1024 fits
in ~1–2s.

## Files in this experiment

```
experiments/T5b_features_multihorizon/
├── build_features_C.py     # Scheme C cache builder (parquet → npz)
├── train_loso.py           # LOSO multi-horizon LightGBM trainer
├── train_final.py          # Train final per-horizon model on full data
├── threshold_sweep.py      # Per-horizon (T, δ) sweep on LOSO OOF
├── build_iter_002.py       # Build submission/iter_002 from sweep results
├── cache/                  # 226-d Scheme C train/val/test npz
├── loso_pred_h{H}_held{K}.parquet   # 25 fold OOF predictions
├── loso_model_h{H}_held{K}.txt      # 25 fold models
├── final_model_h{H}.txt             # final model per chosen horizon
├── loso_results_*.json
├── threshold_results.json
├── sweep_results_h{H}.csv
├── results.json
├── worker-progress.json
└── report.md (this file)
```
