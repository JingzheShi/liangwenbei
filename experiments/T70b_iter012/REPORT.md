# T70b — iter_012 Packaging Report

**Date**: 2026-05-07
**Tag**: iter_012_v4_stage5
**Origin**: T70 V4 walk-forward + Stage 5 features 5-seed + DE 4D thresh

## Headline metric

| | LOSO-equivalent (sum cum_pnl over sym 0-4) |
|---|---|
| iter_011 (T64 V4 walk-forward, 340 feats) | +25.94 |
| **iter_012 (T70 V4 + Stage 5, 359 feats)** | **+26.44** |
| Δ vs iter_011 | **+0.50** |

Per-sym cum_pnl @ DE-tuned 4D thresh:
`[+2.67, +5.22, +3.73, +3.26, +11.56]` → 5/5 positive.

DE-tuned thresholds (from `experiments/T70_v4_stage5/de_5seed_results.json`):
- `T_up = 0.4380`, `T_dn = 0.4449`
- `d_up = 0.0000810`, `d_dn = 0.0000669`

## What changed vs iter_011

1. **+20 Stage 5 features** (T68): 6 families, all sym-agnostic, all max W = 100.
   - A. Adaptive momentum (3): W ∈ {20, 50, 100}
   - B. OFI toxicity (4): (lvl, W) ∈ {(1,20), (1,50), (5,20), (5,50)}
   - C. Multi-scale signed bipower (3): W ∈ {20, 50, 100}
   - D. Stationary spread regime (3): W ∈ {20, 50, 100}
   - E. Trade-direction persistence (3): W ∈ {20, 50, 100}
   - F. Liquidity asymmetry (4): W ∈ {5, 20, 50, 100}
2. **5 fresh boosters** (T70 5-seed: 1, 7, 13, 42, 100), trained on schemeP cache (370 dims) with 11 FAIL drops → 359 dims.
3. **DE 4D thresh re-tuned** on T70 5-seed argmax+ensemble probs.
4. **`liq_asym_top5_W5` added to FAIL drop set** (sym-invariance test failed in T68 audit).

## Audit: §1 hard constraints (CRITICAL_CONSTRAINTS.md)

- ✅ `date` never enters feature vector. (Predictor reads only the 154 raw + 216 extras computed from 100-tick window.)
- ✅ Stateless: `Predictor.predict` holds no cross-call state on `self`. Verified via shuffled-batch invariance test (sanity #21).
- ✅ Sym-agnostic: no sym ID, sym embedding, or sym-conditional normalization. Stage 5 was validated sym-invariant in T68.
- ✅ All 6 Stage 5 families have max W = 100 — entirely computable from the 100-tick window. **No re-train needed for Stage 5 W audit**.

## Bug fix found during packaging

**Issue**: T70 models were trained on the T68 `schemeP` cache, whose raw-column block uses a *blocked* layout (`bid1..bid10, bsize1..bsize10, ask1..ask10, asize1..asize10`). However, `config.json`'s `feature` list (legacy from iter_011) is *interleaved* (`bid1, bsize1, bid2, bsize2, ...`). LightGBM `Booster.predict` is positional, so a naive port of iter_011's Predictor would have fed columns to the wrong slots.

**Fix**: Added `RAW_COLS_TRAIN_ORDER` constant in `Predictor.py` matching the T68 build_features.py blocked layout, and use it (instead of `config.json["feature"]`) when extracting columns from the platform's DataFrame.

`config.json["feature"]` is left in interleaved order so the platform's input pipeline (which presumably uses it to ensure column presence) is not disturbed; the 154 columns are the same set, just a different index permutation, which is what we need to control inside the Predictor.

## Precision & end-to-end verification

- **Feature pipeline precision**: max abs diff vs T68 schemeP cache = **0.0** across 50 random test rows × 359 features (`experiments/T70b_iter012/precision_check.py`).
- **End-to-end probability match**: max abs diff vs direct `booster.predict` on cache = **0.0** across 256 windows × 3 classes (`experiments/T70b_iter012/end_to_end_check.py`). Argmax agreement: 256/256.

## 22-item sanity check

`experiments/T70b_iter012/sanity_and_timing.py` → 22/22 PASS.

Highlights:
- Predictor instantiates, returns `list[list[int]]` shape `(N, 5)` with values in `{0,1,2}`.
- 359 final features (= 154 raw + 205 extras after 11 FAIL drop).
- `RAW_COLS_TRAIN_ORDER` blocked layout confirmed.
- `extra_keep_idx` length = 205, `FAIL_NAMES` length = 11.
- h=60 active, h=5/10/20/40 inactive.
- Ensemble seeds = `[1, 7, 13, 42, 100]`.
- sym=99 (out-of-train) does not error.
- Shuffled-batch order produces the same per-window prediction (no cross-call state).
- 1024-batch inference: **0.554 s** (~0.54 ms/window).

## Contract validator

`python -m src.submit.validate_contract --src submission/iter_012_v4_stage5 --pkg-name mmpc --n-batches 3` → all assertions PASS.

- zip created (11.5 MB, 11 files) — flat layout, no nested dirs
- run() returns `(1842, 5)` int64 in `{0, 1, 2}` for one full session

## Inference timing

- 1024-batch (synthetic): 0.554 s
- 1842-point full session via LocalEvaluator: 1.07 s (`first_batch_s = 0.61`, `avg_batch_s = 0.54`)
- **Extrapolated full 442k**: ~232 s ≈ **3.9 min** (well within the 30-min platform timeout).

## Package contents

```
submission_050707_iter012.zip  (11.5 MB)
├── Predictor.py
├── config.json
├── fast_features.py        (single-window reference, unused at runtime)
├── fast_features_batch.py  (batch-vectorized, T3+S1+S2+S3+S5 → 216 extras)
├── thresholds.json         (h=60 active; T_up=0.4380, T_dn=0.4449, d_up=8.1e-5, d_dn=6.7e-5; ensemble_seeds=[1,7,13,42,100])
├── requirements.txt        (numpy 2.4.4, pandas 2.3.3, lightgbm 4.6.0, scipy 1.17.1)
└── model_h60_seed{1,7,13,42,100}.txt   (T70 V4+Stage5 boosters)
```

## Decision

**PACKAGED** — `submission_050707_iter012.zip` ready to submit.

Expected platform LOSO-equivalent ≈ +26.44, +0.50 vs iter_011.
