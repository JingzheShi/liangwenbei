# T61 — iter_010 inference fix via batch vectorization

## Problem

iter_008 / iter_009 (340-d feature × 5-seed LightGBM ensemble for h=60) **timed
out on the platform**. Local profiling on iter_009 showed:

- 1024 windows took **35.3 s** (estimated **254 min** for 442k samples).
- **99.6 %** of the time was in feature extraction; the 5-model LightGBM
  predict was only **0.07 %**.
- Feature extraction breakdown per 200-window batch: Stage 1 dual-z 45.8 %, T3
  31.7 %, Stage 2 13.1 %, Stage 3 5.7 %.
- GPU LightGBM is therefore irrelevant — predict was never the bottleneck.

## Approach (chosen)

User instruction: do **not** drop features or seeds. Same model, same
LOSO-equivalent +24.52. Only optimise the feature extractor.

### What changed
1. **fast_features.py** — already in repo from earlier, replaces pandas
   `.rolling()` / `.ewm()` with NumPy `cumsum` / `scipy.signal.lfilter` for a
   single window (eliminates pandas Series construction + per-call rolling
   overhead).
2. **fast_features_batch.py** — *new*. Operates on `(N, 100, K)` 3-D tensor.
   Every per-feature op is vectorized across the batch axis; EWMA uses
   `scipy.signal.lfilter(axis=-1)`.
3. **Predictor.py** — stacks the list of `pd.DataFrame` into one 3-D NumPy
   array and calls the batch extractor once per `predict()`. Booster
   inference is unchanged.

Stack:
- `numpy==2.4.4`, `pandas==2.3.3`, `scipy==1.17.1`, `lightgbm==4.6.0` (CPU).

## Numerical correctness

Compared on real data (sym0, multiple sessions, 256 random 100-tick windows):

| Compare against              | feat max abs diff | predictions agreement (h=60) |
|------------------------------|-------------------|------------------------------|
| `fast_features.py` single-window | **3.6e-7**     | 100 %                          |
| `iter_009` reference (pandas)    | **7.2e-3**     | 100 %                          |

The 7.2e-3 only appears on `kyle_inv_W{50,100}` whose absolute value is
~2.6 × 10⁵ — that's relative ~3e-8, pure float32 round-off. All 256
predictions match iter_009 byte-for-byte.

## Speed

| Variant                | 1024-win wall | 442k extrapolation |
|------------------------|---------------|--------------------|
| iter_009 (pandas)      | 35.3 s        | **254 min**        |
| iter_010 batch (CPU)   | **0.60 s**    | **4.3 min**        |
| Speedup                | **58×**       |                    |

Breakdown of the 0.60 s for 1024 windows: feature extract 0.555 s, ensemble
predict 0.035 s — the feature side is now ≈ 16× faster than the predict side
*per window*, so further feature-side gains have rapidly diminishing returns.

The 442k inference is now well under the contract budget (< 30 min limit,
even allowing 5× headroom for slower platform CPUs).

## Files (submission/iter_010_t61_batchvec/)

```
Predictor.py            # batch entry point
fast_features_batch.py  # (N, 100, K) → (N, 196) extractor
fast_features.py        # single-window fallback (kept for parity validation)
config.json             # raw feature list (from iter_009)
thresholds.json         # T59 5-seed DE 4D thresh (LOSO-equiv +24.52)
model_h60_seed{1,7,13,42,100}.txt  # 5 LightGBM boosters (unchanged)
requirements.txt        # numpy / pandas / scipy / lightgbm pins
submission.zip          # 15.47 MB
```

## Validation log

- `submission/scripts/sanity_check.py`: **19 pass / 1 warn / 0 fail** (warn is
  for missing zip on first pass; resolved after `prepare.py`).
- `python -m src.submit.validate_contract`: **all asserts pass** — zip
  extracts, Predictor instantiates, run() returns (1842, 5) on a real
  session, all elements ∈ {0,1,2}, label_60 yields 20 down / 1809 flat / 13
  up.
- `pip install -r requirements.txt`: dry-run succeeds in current env.

## Decision summary

- **Did not** reduce seeds or drop features (kept LOSO-equiv +24.52).
- **Did not** introduce numba / cython / GPU LightGBM (predict wasn't the
  bottleneck and adding new build deps is risky on the platform).
- **Did** batch-vectorise across windows. Pure NumPy + SciPy, available on
  any standard Python env.

`RESULT: task=t61_iter010 metrics={old_time_min=254.1, new_time_min=4.32, speedup=58x, vectorized=Y, predictions_match_iter009=100%}`
