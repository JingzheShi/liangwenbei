# T108: iter_016 CHIS — torch GPU GRU repackage

**Goal**: replace the numpy GRU forward (1391s on 442k×5 seeds) with a
torch implementation that uses CUDA when available, while preserving the
exact prediction pipeline of iter_016 v1 (T107 CHIS). Package as a new
zip with torch in requirements.txt.

## TL;DR

- **GRU inference: 1391s → 3.7s on 442k×5 seeds (375x speedup)**.
- **cum_pnl on full 442k = +43.2002** vs T106 numpy baseline +43.1218
  (drift +0.078, well below the ±0.10 fp32 threshold target).
- All 4 model families combined inference on 442k: **35.7s** (vs ≈1424s
  with numpy GRU).
- Per-seed parity test: max |numpy-cuda| = **3.49e-10** on raw GRU
  outputs (effectively bit-identical; the +0.078 cum_pnl drift comes from
  the EV gate flipping ~28 borderline samples between actions).
- All CRITICAL_CONSTRAINTS preserved: stateless, shuffle-invariant,
  sym/date never used, sym-agnostic.
- **Robust fallback**: if torch is unavailable, or `torch.cuda` init fails,
  the Predictor auto-degrades to torch CPU; if torch import fails entirely,
  it degrades to the original numpy GRU. The selected backend is reported
  in `predict()` setup.

## Build artefact

- Zip: `submission_050814_iter016_gpu.zip` (49 MB)
- 66 files, identical model weights to iter_016 v1 (no .pt added — the
  torch GRU is built from the existing `gru_h60_seed*.npz` files).
- requirements.txt:
  ```
  numpy==2.4.4
  pandas==2.3.3
  lightgbm==4.6.0
  catboost==1.2.10
  scipy==1.17.1
  torch==2.5.1
  ```

## What changed in Predictor.py

- Added `_GRUEnsembleTorch` class: builds 5 `torch.nn.GRU(20→64, 1
  layer, batch_first=True)` modules from the existing npz weights, plus
  matching `LayerNorm` + `Linear` heads. Loads them onto the chosen
  device once. `predict_mean()` does:
  1. Per-window mean/std on time axis (axis=1) with `unbiased=False` to
     match numpy's default ddof=0;
  2. Clip ±10.0 (matches all 5 npzs);
  3. Run per-model `LayerNorm → GRU → last-step → Linear → /target_scale`,
     accumulate, divide by 5;
  4. Return numpy float32.
- Predictor `__init__` now picks backend in this order:
  1. torch + CUDA available  → `_GRUEnsembleTorch("cuda")`
  2. torch present, no CUDA  → `_GRUEnsembleTorch("cpu")`
  3. anything else fails     → list of `_GRUNumpy` (legacy fallback)
- The numpy `_GRUNumpy` class is kept verbatim as fallback.

## CRITICAL_CONSTRAINTS compliance

- `sym` / `date` are not in the feature pipeline anywhere (verified via
  `_compute_features`: it only reads RAW_COLS_TRAIN_ORDER from each
  DataFrame and the GRU only sees the LOB top-5 columns).
- **Stateless**: `predict()` reuses no per-sample state. End-to-end test:
  calling `predict()` twice on the same input returns byte-identical
  output (`stateless_ok: true`).
- **Shuffle-invariant**: end-to-end test permutes the input, runs
  `predict()`, then unpermutes — output matches the unshuffled call
  byte-identically (`shuffle_invariant_ok: true`).
- **sym-agnostic**: no sym embedding, no per-sym normalisation; per-window
  normalisation for the GRU is window-internal.

## Verification

### Per-seed numpy-vs-cuda parity (8192 synthetic windows)

| metric                | value    |
|-----------------------|----------|
| numpy GRU 8192 batch  | 4.65 s   |
| torch CPU 8192 batch  | 0.44 s   |
| torch CUDA 8192 batch | 0.014 s  |
| max\|numpy − cuda\|     | 3.49e-10 |
| mean\|numpy − cuda\|    | 5.39e-11 |

(see `scratch/test_parity.py`)

### Full 442k cum_pnl (h=60 4-way, w=1.0:1.5:0.7:0.7)

| component       | T106 numpy GRU | T108 torch CUDA GRU |
|-----------------|----------------|---------------------|
| sum_per_sym     | +43.1218       | **+43.2002**        |
| n_active        | 188,641        | 188,613             |
| T87 NN time     | (in 1424s tot) | 24.5 s              |
| T89 CB time     | —              | 1.3 s               |
| T99 Huber time  | —              | 6.2 s               |
| T95 GRU time    | **1391 s**     | **3.7 s**           |
| total inference | ≈1424 s        | **35.7 s**          |

per-sym (T108):

| sym | cum_pnl  |
|-----|----------|
| 0   | +5.0257  |
| 1   | +4.5797  |
| 2   | +4.6510  |
| 3   | +12.7709 |
| 4   | +16.1729 |

(see `full_test_verify.json`)

### End-to-end Predictor smoke (n=4096 real LOB windows)

| check                      | result   |
|----------------------------|----------|
| GRU backend selected       | torch_cuda |
| stateless (call#2 == call#1) | true   |
| shuffle-invariant          | true     |
| extrapolated 442k time     | 330.8 s (incl. fast_features extras) |
| ms / sample                | 0.75 ms  |

(see `end_to_end_check.json`)

### Packaged zip extraction smoke (n=1024)

- Backend selected: `torch_cuda`
- predict(1024) returns (1024, 5) in 0.79 s (0.77 ms/sample)
- requirements.txt verified inside zip

## Recommendation

iter_016 GPU **strictly dominates** iter_016 v1 on every metric:

- Same model weights, same thresholds, same EV gate → same expected
  cum_pnl on the platform's hidden test (within ±0.10 fp32 drift).
- 35× lower model inference time on 442k samples → far less risk of
  hitting the platform's per-call wall clock.
- Robust auto-fallback: if the platform turns out *not* to support torch
  GPU (or even torch at all), the Predictor still functions (numpy GRU)
  — but at iter_016 v1 speed, not slower.

**Replace iter_016 v1 with iter_016 GPU as the active submission**.
Keep `submission_050813_iter016_chis.zip` as the no-torch fallback in
case the platform rejects torch in requirements.txt.

## Risk: platform compatibility

Two ways this could fail on the platform:

1. **torch wheel rejected at install time** — requirements.txt parsing
   fails because torch==2.5.1 wheel is too large or unavailable.
   *Mitigation*: keep iter_016 v1 (`submission_050813_iter016_chis.zip`)
   as fallback and resubmit if this zip fails to install.

2. **CUDA driver not available at runtime** — torch loads but
   `torch.cuda.is_available()` returns False.
   *Behaviour*: Predictor automatically falls back to torch CPU GRU —
   still ~10× faster than numpy (0.44s vs 4.65s on 8192 batch in our
   parity test). cum_pnl unchanged.

## Files

- `pkg/Predictor.py` — torch-aware Predictor with auto-fallback
- `pkg/requirements.txt` — adds `torch==2.5.1`
- `pkg/*` — all other model files unchanged from iter_016 v1
- `full_test_verify.py` + `.json` + `.log` — 442k cum_pnl verification
- `end_to_end_check.py` + `.json` — stateless + shuffle-invariance check
- `scratch/test_parity.py` — per-seed numpy-vs-cuda parity test
