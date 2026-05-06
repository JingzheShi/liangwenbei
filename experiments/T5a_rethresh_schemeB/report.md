# T5a — Re-tune threshold on Scheme B's own LOSO OOF

## TL;DR

| Variant | (T, δ) | sum_cum_pnl | n_pos / 5 | sum_n_active | mean_acc |
|---|---|---:|---:|---:|---:|
| baseline raw argmax | — | **+1.33** | 3 | 231,284 | — |
| **iter_001c** (Scheme A's threshold transferred to Scheme B model) | (0.50, 0.15) | **+8.15** | 5 | 32,879 | 0.6075 |
| **iter_001d** (best from Scheme B's own LOSO sweep) | (**0.45, 0.05**) | **+11.11** | 4 | 83,048 | 0.6090 |

Improvement of iter_001d over iter_001c: **+2.95 cum_pnl** on 5-fold LOSO sum (significance threshold 0.5).
Decision: **build iter_001d** and submit alongside iter_001c.

---

## 1. Setup

- **OOF**: 5-fold LOSO from T4 (`experiments/T4_loso_validate/loso_pred_schemeB_held{0..4}.parquet`),
  total 442,080 evaluation points across 5 held-out syms.
- **Decision rule** (same as T4's `apply_threshold`):
  ```
  side_max = max(prob_0, prob_2)
  if side_max >= T and side_max > prob_1 + delta:
      pred = argmax(prob_0, prob_2)   # 0 if prob_0 > prob_2 else 2
  else:
      pred = 1                         # flat / no trade
  ```
  Note: comparator is `>=` for T and `>` for delta; aligned in iter_001d's Predictor.
- **Grid**: T ∈ {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70} × δ ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25} = 48 combos.
- **Metric**: `cum_pnl` from `src/eval/pnl._per_horizon_metrics` with `fee_rate=0.0001` (matches platform).

## 2. Baseline (raw argmax) per fold

| held-out sym | cum_pnl | n_active | accuracy |
|---:|---:|---:|---:|
| 0 | +1.5045 | 9,944 | 0.7783 |
| 1 | -1.3445 | 60,825 | 0.4063 |
| 2 | **-7.0743** | 73,133 | 0.3238 |
| 3 | +0.0399 | 13,646 | 0.7188 |
| 4 | +8.2038 | 73,736 | 0.4036 |
| **sum** | **+1.33** | 231,284 | — |

Sym=2 is the dominant brittleness sink (-7.07). Sym=4 carries the bulk of positive PnL (+8.20).
Default argmax trades on ~52% of all OOF points — way too active, fees + brittleness eat returns.

## 3. Sweep top 10 (sorted by sum_cum_pnl desc)

| T | δ | sum_pnl | pos/5 | n_active | f0 | f1 | f2 | f3 | f4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.45** | **0.05** | **+11.11** | 4 | 83,048 | +1.36 | +3.40 | -0.98 | +0.23 | +7.11 |
| 0.45 | 0.00 | +11.11 | 4 | 83,063 | +1.36 | +3.40 | -0.98 | +0.23 | +7.11 |
| 0.45 | 0.10 | +11.10 | 4 | 82,529 | +1.34 | +3.39 | -0.95 | +0.22 | +7.10 |
| 0.45 | 0.15 | +10.99 | 4 | 79,409 | +1.28 | +3.33 | -0.68 | +0.17 | +6.88 |
| 0.45 | 0.20 | +10.65 | 4 | 71,156 | +1.23 | +3.17 | -0.11 | +0.11 | +6.26 |
| 0.45 | 0.25 |  +8.99 | **5** | 56,865 | +1.11 | +2.81 | +0.40 | +0.03 | +4.65 |
| 0.35 | 0.25 |  +8.50 | 5 | 64,577 | +1.19 | +2.54 | +0.11 | +0.02 | +4.64 |
| 0.40 | 0.25 |  +8.50 | 5 | 64,577 | (same as 0.35,0.25 — both are dominated by δ=0.25 gate) |
| 0.40 | 0.05 |  +8.39 | 4 | 157,121 | +1.70 | +1.21 | -4.15 | +0.13 | +9.49 |
| 0.35 | 0.20 |  +8.37 | 3 | 94,416 | +1.40 | +1.87 | -0.95 | -0.14 | +6.19 |

Observations:

- **T = 0.45 is a sharp peak**. T = 0.40 keeps too many trades (157k → -4.15 on sym=2). T ≥ 0.50 throws away too much edge (the (0.50, *) row is +8.15).
- δ has **almost no effect at T = 0.45** for δ ≤ 0.20 (everything within +10.65..+11.11). The T cutoff alone is doing the heavy lifting.
- **(0.45, 0.05) and (0.45, 0.0) are tied to within 0.001** (+11.108 vs +11.107); we picked δ=0.05 by argmax but functionally either works.

## 4. Best (T*, δ*) = (0.45, 0.05) — per-fold breakdown

| held-out sym | cum_pnl | n_active | accuracy |
|---:|---:|---:|---:|
| 0 | +1.3559 |  1,140 | 0.8178 |
| 1 | +3.4019 | 19,384 | 0.4459 |
| 2 | **-0.9839** | 36,213 | 0.5116 |
| 3 | +0.2262 |  2,833 | 0.7843 |
| 4 | +7.1085 | 23,478 | 0.4856 |
| **sum** | **+11.11** | 83,048 | 0.6090 |

vs baseline raw argmax: **+9.78** improvement (+1.33 → +11.11).
vs iter_001c: **+2.95** improvement.

The tradeoff vs iter_001c: lower T makes us trade more often, sym=2 flips from positive (+0.93) to slightly negative (-0.98), but sym=1 (+0.57) and sym=4 (+3.47) more than make up for it. Sym=0 also gains substantially (+0.81).

## 5. iter_001c (T=0.50, δ=0.15) — per-fold breakdown for reference

| held-out sym | cum_pnl | n_active | accuracy |
|---:|---:|---:|---:|
| 0 | +0.5442 |    249 | 0.8173 |
| 1 | +2.8364 |  6,807 | 0.4415 |
| 2 | +0.9307 | 16,706 | 0.5709 |
| 3 | +0.2017 |    747 | 0.7948 |
| 4 | +3.6416 |  8,370 | 0.4750 |
| **sum** | **+8.15** | 32,879 | 0.6075 |

iter_001c wins on **5/5 positive folds** and is more conservative (only 33k active trades vs 83k).
**Risk profile**: iter_001c offers more consistent positive PnL across syms; iter_001d has higher peak but takes a small loss on sym=2.

## 6. Decision

**Build iter_001d** (replaces iter_001c as primary submission).

Rationale:
- Improvement of +2.95 sum_cum_pnl is well above the 0.5 significance threshold.
- 4/5 positive folds is still a robust pattern (only sym=2 is mildly negative at -0.98, vs -7.07 for raw argmax → still a 6.1 improvement on the worst fold).
- Higher mean accuracy (0.6090 vs 0.6075) and more trading edge captured in the high-conviction regime.
- iter_001c is preserved as a more-conservative backup (5/5 positive, lower variance).

Note: iter_001c was tuned on **Scheme A's** LOSO OOF (sweep showed T=0.50, δ=0.15 was that best), and we transferred to a Scheme B model. T5a confirms that this transfer was suboptimal — Scheme B's calibration peak is at slightly lower T (0.45), so trading the lower-confidence high-conviction tail is profitable on Scheme B but was not on Scheme A.

## 7. Deliverables

- `experiments/T5a_rethresh_schemeB/sweep.py` — sweep script
- `experiments/T5a_rethresh_schemeB/sweep_results.csv` — full 48-row grid
- `experiments/T5a_rethresh_schemeB/best_per_fold.csv` — per-fold @ (0.45, 0.05)
- `experiments/T5a_rethresh_schemeB/iter001c_per_fold.csv` — per-fold @ (0.50, 0.15)
- `experiments/T5a_rethresh_schemeB/results.json` — full JSON summary
- `submission/iter_001d_lgbm_schemeB_rethresh/` — new submission package
  - `Predictor.py` updated: THRESHOLD_T = 0.45, THRESHOLD_DELTA = 0.05
  - `submission.zip` rebuilt
  - sanity_check: 22/22 ✓
- `submission_050603_iter1d.zip` — copy at workdir root for PM hand-off

## 8. WandB

Run: https://wandb.ai/jingzheshi/liangwenbei/runs/m4clzawu (project liangwenbei).

Note: the configured entity `cjxh21-Tsinghua University` rejected the supplied API key
("permission denied" at `wandb.init`). Logged under personal entity `jingzheshi` instead so
that data is at least visible. PM may want to verify whether the API key in the system prompt
actually has access to the team entity, or whether a different key/entity is needed.
