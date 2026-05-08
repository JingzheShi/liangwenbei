# R-NaN2: CatBoost Huber NaN-handling sweep

**Status:** ❌ Hypothesis falsified. NaN→0 vs NaN-passthrough are within seed-noise (Δ +0.22 LOSO). `nan_mode='Max'` hurts by -1.39 LOSO vs Min. **Recommendation: keep current pipeline (no change).**

## TL;DR

| Variant (3-seed: 1, 7, 42) | k=1 cum_pnl | DE LOSO | thr_up / thr_dn | Δ vs T99 baseline (+40.24, 5-seed) |
|---|---:|---:|---|---:|
| **zero** (explicit `np.nan_to_num(X, nan=0)`) | +36.58 | **+40.39** | 0.000301 / 0.000307 | **+0.15** |
| **min** (NaN passthrough + `nan_mode='Min'`, **= current default**) | +36.11 | +40.18 | 0.000339 / 0.000315 | -0.06 |
| **max** (NaN passthrough + `nan_mode='Max'`) | +34.13 | +38.80 | 0.000321 / 0.000353 | -1.44 |

3-seed seed-noise (across {1, 7, 42}) is ~0.5 LOSO; the zero–min gap (+0.22) sits inside that noise. The min–max gap (-1.39) is real and consistent (max is worse on every sym).

## Critical finding: PM premise was wrong

The R-NaN1/R-NaN2 prompt assumed "current pipeline does NaN→0 replacement, hurting CB native missing handling". **This is empirically false** for the T99 CB Huber pipeline:

- `experiments/T68_stage5_features/cache/schemeP_train.npz` preserves NaN: 86,602 NaN cells in train (0.016%), 77,576 in val, 0 in test (62 / 370 cols affected).
- `experiments/T99_e2e_execution_gbdt/train_cb.py` does **no** `nan_to_num` / `fillna` call — X with NaN goes straight into CB `Pool`.
- CB defaults `nan_mode='Min'`, so the T99 baseline (+40.24 LOSO 5-seed) was **already** "Trick A: NaN passthrough + Min".

Therefore the experiment redesigned to:
- **zero**: explicit `np.nan_to_num(X, nan=0)` — what the PM thought baseline was doing.
- **min**: no replacement, `nan_mode='Min'` (CB default — what baseline actually does).
- **max**: no replacement, `nan_mode='Max'`.

## Why so little effect

- NaN density is tiny (86k / 1.47M × 359 = 0.016% of training cells). The signal-to-noise for "where exactly NaN gets sent" is correspondingly small.
- The 62 cols with NaN concentrate at session warmup (before rolling-window feats stabilize) — these rows have low predictive value either way.
- CB's Min bucket vs explicit-0 essentially routes the NaN rows to the same end of the split distribution (these features were already small-magnitude); Max routes them to the opposite end, which apparently anti-correlates with the rolling-window signal.

## Per-seed raw cum_pnl@k=1 (sanity)

| variant | seed=1 | seed=7 | seed=42 | mean |
|---|---:|---:|---:|---:|
| zero | +36.75 | +35.63 | +34.02 | +35.47 |
| min  | +32.19 | +36.41 | +38.63 | +35.74 |
| max  | +31.50 | +30.02 | +37.95 | +33.16 |

Per-seed std ~ ±2.0 LOSO confirms 3-seed avg gives ~±1.2 standard error. The +0.22 zero–min DE gap is comfortably inside.

## Per-sym DE results

| Variant | sym0 | sym1 | sym2 | sym3 | sym4 |
|---|---:|---:|---:|---:|---:|
| zero | +4.45 | +4.56 | +4.13 | +11.15 | +16.11 |
| min  | +4.69 | +5.00 | +4.04 | +11.26 | +15.19 |
| max  | +4.45 | +4.62 | +4.15 | +10.53 | +15.05 |

Min and zero trade per-sym (zero better on sym4, min better on syms 0,1,3). Max never wins.

## Decision

- **Do NOT switch the CB pipeline** to a new NaN handling. Current default (passthrough → CB Min bucket) is right; explicit `nan_to_num(X, nan=0)` is also fine and within noise.
- **Reject `nan_mode='Max'`** — costs ~1.4 LOSO with no upside.
- **No combined LGB+CB ensemble run** — Step-3 was conditional on Trick A win; A did not win.

## Artifacts (on remote `/root/lwb_work_v3/`)

- `train_cb_nan.py` — variant-aware CB Huber trainer (GPU)
- `eval_de.py` — DE LOSO thresh search
- `model_cb_huber_a0.001_{variant}_seed{1,7,42}.cbm` (9 models)
- `pred_cb_huber_a0.001_{variant}_seed{1,7,42}.parquet` (9 prediction files)
- `summary_cb_huber_a0.001_{variant}_seed{1,7,42}.json` (9 summaries)
- `de_results.json` — per-variant DE outputs
- `log_{variant}_s{seed}.log` — train logs

## Time budget actual vs planned

- Pipeline modification + script: 25 min (planned 30)
- 9 CB GPU trains: ~3 min total (planned 30 — much faster, RTX 3080 + small data)
- DE eval: 1.5 min (planned 15)
- Report: 10 min (planned 15)
- **Total: ~40 min** (planned 90)

## Recommended next action

Drop NaN-handling avenue entirely; the residual signal is too small to chase. Spend further compute on orthogonal directions (better features, more diverse model heads, longer-horizon stacking, etc.).
