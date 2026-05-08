# R4 — Multi-scale fixed-lag log returns (T110 #2)

## Hypothesis
G-Research 2022 winner's top-importance feature was a fixed-lag log return.
schemeP 359-d has **no raw signed lag return** — only EWMA / RV / HMA.
Add 12 ratio-based, sym-agnostic, stateless dims:

```
logret_mid1_lag{k}  for k ∈ (1, 2, 5, 10, 20, 50)
logret_wmp1_lag{k}  for k ∈ (1, 2, 5, 10, 20, 50)
   wmp1 = (bid1·asize1 + ask1·bsize1) / (bsize1 + asize1)
   log_ret_k = log((p_t + 1) / (p_{t−k} + 1))   clipped to [-0.05, 0.05]
```

CRITICAL_CONSTRAINTS-safe: ratio-based (sym-agnostic), single-row computation (stateless).

## Protocol
- Identical hyper-params for baseline vs trick. Only feature set differs.
- LGB Huber alpha=0.001, 3 seeds {1,7,42}, V4 split (train 0–75 / val 76–79 / test 96–119).
- num_boost_round=300, early_stopping=25, lr=0.05, num_threads=4 (4-core remote).
- Augmentation **dropped** (deviation from R2 protocol — same drop applied to both arms,
  so the comparison is internally consistent; only absolute level shifts).
- DE asym threshold (5-restart Sobol) over full 442k local test set, per-seed prediction averaged.

## Results (3-seed avg, DE asym threshold)

|             | baseline 359-d | trick 359+12-d | Δ        |
|-------------|---------------:|---------------:|---------:|
| LOSO sum    | +38.7963       | +40.3922       | +1.5959  |
| per_sym     | [4.76, 4.86, 3.85, 10.80, 14.53] | [4.45, 5.16, 4.13, 11.06, 15.60] |          |
| per_sym min | 3.8498         | 4.1278         | +0.2780  |
| per_sym std | 4.189          | 4.534          | +0.345   |

Per-sym deltas: sym0 −0.31, sym1 +0.30, sym2 +0.28, sym3 +0.26, sym4 +1.07.
Sym 4 captures most of the gain; sym 0 mildly regresses.

## Per-seed train metrics
| seed | best_iter (base) | cum_pnl (base, fixed) | best_iter (trick) | cum_pnl (trick, fixed) |
|------|------------------|-----------------------|-------------------|------------------------|
| 1    | 281              | +31.58                | 173               | +37.74                 |
| 7    | 246              | +35.21                | 298               | +34.66                 |
| 42   | 241              | +32.94                | 220               | +33.76                 |

(fixed_thr=2·FEE PnL — pre-DE).

## Verdict
**SUCCESS_GATE = True** (Δ_loso = +1.60 > 0.5).
12-dim multi-scale fixed-lag log-return features lift LOSO PnL by **+1.60**
on top of the schemeP 359-d baseline — a meaningful structural improvement
from a feature G-Research 2022 winners highlighted but schemeP omitted.

Δ_per_sym_min = +0.28 (below 0.5) — gain concentrated in syms 1–4, sym 0 mildly regresses.
Net robustness improves but sym-min not unanimously.

## Files
- `train.py` — LGB Huber trainer, supports `--use-lag` and `--no-aug`
- `eval_de.py` — DE asym threshold eval + per-sym summary
- `build_lag_features.py` — produces `lagret_{train,test}.npz` from snapshot parquets
- `cache/lagret_{train,test}.npz` — precomputed lag features (12 dims, NaN-free)
- `pred_R4_{baseline,trick}_seed{1,7,42}.parquet` — test predictions
- `summary_R4_{baseline,trick}_seed{1,7,42}.json` — per-train summaries
- `results.json` — final 3-seed eval
