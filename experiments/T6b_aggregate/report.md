# T6b: 5-seed Scheme A Ensemble — OOF Aggregation + Threshold Sweep

## TL;DR

- **Ensemble does NOT improve over single-seed iter_001d.**
- Ensemble best (T*, δ*) on the 48-grid: **(0.45, 0.25) → LOSO sum = +8.87** (4/5 pos folds).
- Single-seed seed=42 + iter_001d (0.45, 0.05) baseline: **+11.11** (4/5).
- **Δ = −2.24** vs single-seed iter_001d → no iter_001e.
- iter_002 (T5b Scheme C multi-horizon h_10) at **+21.86** remains the strongest LOSO candidate.

## Inputs

- 5 seeds × 5 LOSO folds = 25 single-seed prob files for Scheme A.
  - seed=42 from `experiments/T4_loso_validate/loso_pred_schemeA_held{0..4}.parquet`
  - seeds {1, 7, 13, 100} from `experiments/T6_ensemble_multiseed/loso_pred_schemeA_seed{S}_held{0..4}.parquet`
- All 5 seeds matched on (sym, date, session, t, true_label_60) keys — exact alignment after sort.
- 88,416 rows / fold × 5 folds = **442,080 OOF rows**.

## Step 1 — Aggregate OOF

`aggregate_oof.py` averages the 5 seeds' `prob_0/prob_1/prob_2` per row (after `argmax` would lose information; we average **probabilities**, not votes). Outputs 5 files:

```
experiments/T6_ensemble_multiseed/loso_pred_ensemble_held{0..4}.parquet
```

Each file contains: `sym, date, session, t, true_label_60, pred_label_60, prob_0/1/2, midprice_t, midprice_t60`. Row keys verified identical across seeds; row sums of avg probs ≈ 1.0.

## Step 2 — Threshold Sweep (48 combos)

Decision rule (matches T4/T5a/iter_001c/d):
```
side_max = max(prob_0, prob_2)
take = (side_max >= T) and (side_max > prob_1 + delta)
pred = 2 if (take and prob_2 > prob_0) else (0 if take else 1)
```
Grid: T ∈ {0.35..0.70 step 0.05}, δ ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}.

### Top 5 of the sweep

| rank | T    | δ    | sum_cum_pnl | pos folds | sum_n_active |
|------|------|------|-------------|-----------|--------------|
| 1    | 0.45 | 0.25 | +8.8674     | 4/5       | 52,541       |
| 2    | 0.45 | 0.20 | +8.6307     | 4/5       | 81,415       |
| 3    | 0.50 | 0.25 | +8.3899     | 4/5       | 30,132       |
| 4    | 0.50 | 0.20 | +8.2854     | 4/5       | 33,037       |
| 5    | 0.50 | 0.00 | +8.2685     | 4/5       | 33,918       |

Saturation kicks in once T ≥ 0.50 — most rows already pass `prob_1 + δ` so δ has no effect. Below T=0.45 the prediction count blows up (200K+) and PnL collapses (overtrading the noisy signal).

## Step 3 — Variant comparison (single seed vs ensemble)

| Variant                                              | LOSO sum | pos folds | source       |
|------------------------------------------------------|---------:|----------:|--------------|
| Single seed=42 raw argmax                            |  −22.10  | 2/5       | T4 baseline  |
| Single seed=42 + (0.50, 0.15) (= iter_001c)          |   +6.45  | 5/5       | T4 sweep     |
| Single seed=42 + (0.45, 0.05) (= iter_001d)          |  +11.11  | 4/5       | T4 sweep     |
| **Ensemble raw argmax**                              |  −21.95  | 2/5       | this report  |
| **Ensemble + (0.50, 0.15)** (iter_001c thresh)       |   +8.27  | 4/5       | this report  |
| **Ensemble + (0.45, 0.05)** (iter_001d thresh)       |   +5.80  | 4/5       | this report  |
| **Ensemble + (0.45, 0.25) — best**                   | **+8.87**| 4/5       | this report  |
| iter_002 (T5b Scheme C h_10 multi-horizon)           | **+21.86** | 5/5     | T5b          |

Ensemble vs single seed:
- Raw argmax: ensemble +0.15 (essentially identical, both still negative).
- iter_001c threshold: ensemble +1.82 (modest gain, but lost a positive fold: 5/5 → 4/5).
- iter_001d threshold: ensemble **−5.31** (strict δ=0.05 over-trades the smoother ensemble probs).
- Best ensemble (0.45, 0.25): **−2.24 vs single-seed iter_001d**.

### Why averaging hurts the iter_001d threshold

The single-seed seed=42 probs were sharper — `prob_1` was lower more often, so the `(0.45, 0.05)` filter let through more high-conviction trades. After averaging across 5 seeds (which disagree heavily), `prob_1` rises and `prob_0/prob_2` shrink, so fewer rows clear the gate, and the surviving trades have a lower edge. The ensemble's "best" δ=0.25 is partial compensation but can't recover the lost edge.

## Per-fold @ ensemble best (T=0.45, δ=0.25)

| sym | cum_pnl  | n_active | accuracy |
|-----|---------:|---------:|---------:|
| 0   |  +1.366  |  3,149   | 0.801    |
| 1   |  +2.222  | 17,106   | 0.442    |
| 2   |  +1.526  | 12,290   | 0.600    |
| 3   |  −0.014  |     96   | 0.797    |
| 4   |  +3.768  | 19,900   | 0.471    |

Sym 3 (the blue-chip / ETF profile) is the only negative fold and is barely traded (96 active). 4/5 positive.

## Step 4 — Decision

| Threshold gate                              | Ensemble best | Action         |
|---------------------------------------------|---------------|----------------|
| `>= +14`  → build iter_001e                 | +8.87 ✗       | not triggered  |
| `>= +12 and < +14`  → marginal, skip        | +8.87 ✗       | not triggered  |
| `< +12`  → ensemble not significant         | +8.87 ✓       | **selected**   |

**Decision: do NOT build iter_001e.** The ensemble's best LOSO sum is below single-seed iter_001d.

### PM-facing note

iter_002 (T5b Scheme C multi-horizon, h_10) at **LOSO +21.86** is the strongest candidate by a wide margin. Recommend:
1. Hold iter_002 as the current best submission.
2. Consider applying threshold-postproc to iter_002 (multi-horizon may also benefit from a (T, δ) sweep on its own h_10 OOF).
3. Multi-seed ensembling on top of Scheme C (rather than Scheme A) could be a logical next experiment if ensemble averaging is to be explored further.

## Files

- `aggregate_oof.py` — averages 5 seeds' probs per fold; outputs to `experiments/T6_ensemble_multiseed/loso_pred_ensemble_held{0..4}.parquet`.
- `threshold_sweep.py` — 48-combo (T, δ) sweep + variant comparison + decision logic.
- `sweep_results.csv` — full grid sorted by sum_cum_pnl.
- `raw_argmax_per_fold.csv` — ensemble argmax baseline per fold.
- `best_per_fold.csv` — per-fold @ (T=0.45, δ=0.25).
- `iter001c_per_fold.csv` / `iter001d_per_fold.csv` — per-fold @ iter_001c/d thresholds applied to ensemble.
- `results.json` — machine-readable summary.

## Hard constraints check

- `date` was kept only as a metadata column for OOF row identity, not used in any feature path.
- Predictor would be unaffected — this analysis is offline OOF aggregation only; no Predictor or src/ code changed.
- sym-agnostic: averaging is per-row, no per-sym split or sym-conditioned routing.
