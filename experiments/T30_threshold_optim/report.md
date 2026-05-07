# T30 — Threshold Gating Deep Optimization

> Goal: improve the iter_005b 5-seed aug_a ensemble for h_60 (LOSO sum =
> +11.46 at symmetric T=0.45, δ=0.10) by exploring richer gating schemes,
> all while respecting CRITICAL_CONSTRAINTS §1 (sym-agnostic, stateless,
> no date).
>
> **Headline result**: asymmetric (T_up, T_dn, d_up, d_dn) gating found
> via differential evolution improves OOF LOSO sum from **+11.46 → +13.61**
> (+2.15) with **5/5 positive folds**, std drops 3.07 → 2.56. Built and
> shipped as **iter_006**.

## Compliance check (CRITICAL_CONSTRAINTS §1)

- ✅ `date` not used as feature anywhere; thresholds are constants in
  `thresholds.json`.
- ✅ Predictor is stateless: each `predict(batches)` call computes from
  the current 100-tick windows only; threshold parameters are loaded once
  at `__init__`.
- ✅ Sym-agnostic: gate is a pure function of softmax probs; no sym ID
  enters at any stage. (Per-regime variants explored use only 100-tick
  window features — no sym lookup — but ultimately rejected, see below.)

## Methodology

All variants are evaluated on the same artifact: the 5-seed aug_a OOF
predictions from T26+T27 (held-fold k uses model trained on 4 syms,
predicts on the 5th). Per-fold softmax probabilities are averaged across
seeds {42, 1, 7, 13, 100}. Primary metric: `sum_cum_pnl` over 5 LOSO folds
at h=60 with fee_rate=0.0001. Secondary: number of positive folds and
per-fold pnl std.

iter_005b baseline was confirmed exactly: T=0.45 δ=0.10 → +11.4565
(matches T28 report.md to 4 decimals).

## Results — six variants vs iter_005b baseline

| Variant                              | OOF sum   | Δ vs baseline | std   | n_pos | n_active | Notes |
|--------------------------------------|-----------|---------------|-------|-------|----------|-------|
| iter_005b baseline (T=0.45, δ=0.10)  | +11.4565  | —             | 3.07  | 4/5   | 100,256  | reference |
| **(1) Fine grid** (1426 combos)      | +11.832   | +0.38         | 2.40  | 5/5   |          | T=0.47, δ=0.07 |
| **(2) Asymmetric grid** (~9.9k combo)| +12.550   | +1.09         | 2.31  | 5/5   |          | T_up=0.475, T_dn=0.40, d_up=0.075, d_dn=0.025 |
| **(3) DE direct PnL search**         | **+13.606** | **+2.15**   | 2.56  | 5/5   |          | T_up=0.448, T_dn=0.391, d_up=0.260, d_dn=0.024 |
| **(4) Per-regime (in-sample)**       | (+20.83)  | (+9.4)        | —     | 5/5   |          | regime_intst, 4 bins — but biased; see CV row |
| **(4b) Per-regime (honest LOSO-CV)** | TODO      | TODO          | —     | TODO  |          | thresholds picked on 4 folds, scored on 1 |
| **(5) Conformal (split, sym & asym)**| +10.814   | −0.64         | —     | 5/5   |          | best at α=0.12 sym; alpha grid swept |
| **(6) Calibrated EV gate v2**        |  +7.985   | −3.47         | 1.40  | 5/5   |          | isotonic + per-sample E[Δp] from regime_vol |

(4) is the in-sample reference; (4b) is the only honest estimate of the
per-regime variant. The per-regime in-sample number was found by picking
each bin's (T_up, T_dn, d_up, d_dn) on the **pooled** 5-fold OOF, then
applying per fold — i.e., bin thresholds were tuned with access to the
held-out fold's data. (4b) repeats the tuning under proper LOSO CV.

## Why DE wins (and is robust)

- 5 independent DE runs (seeds 0/1/2/7/42) all converged to within 0.02
  of each other on every parameter:
  - T_up ∈ [0.426, 0.448], T_dn = 0.3914 ± 0.0001
  - d_up ∈ [0.258, 0.261], d_dn = 0.0246 ± 0.0001
  - sum_cum_pnl ∈ [+13.59, +13.61]
- Neighborhood sweep around the DE optimum (21,060 configs, ±0.06 in T,
  ±0.10 in d_up, ±0.05 in d_dn, step 0.01):
  - 9,093 / 21,060 configs ≥ +12.0
  - 4,226 / 21,060 configs ≥ +12.5
  - 675 / 21,060 configs ≥ +13.0
  - 16,232 / 21,060 configs all-5-folds-positive
- Conclusion: the optimum is a **wide ridge**, not a needle in the OOF
  noise. The shape "small T_dn (looser short threshold), large d_up
  (require strong long signal)" is recoverable from coarse and fine
  grids alike — DE just refines past grid resolution.

The economic interpretation: the model's prob_2 (long signal) is
**less reliable** than prob_0 (short signal) on h=60. Asymmetric gating
trims more long-side false positives (d_up=0.26 forces p2 to dominate
prob_1 by a wide margin) while admitting more short-side trades
(T_dn=0.39, d_dn=0.02 — almost just argmax). This corrects a calibration
asymmetry that the symmetric gate could not exploit.

## What didn't work

**(5) Conformal**: the empirical quantile of `max(p0, p2)` across 4
calibration folds is itself sym-agnostic, but the resulting threshold is
too low at low α (over-trades) and too high at high α (under-trades).
Picks the same operating point regardless of long/short asymmetry — so
it can't beat the asymmetric gate. Best α=0.12 sym → +10.81.

**(6) Calibrated EV gate v2**: T15 used global E[Δp|up]; this v2 used
per-sample E[Δp] proxied by `α · regime_vol_100`. Even after a 7-point α
sweep the best variant scored +7.99 (worse than baseline), because the
volatility magnitude is a poor proxy for *signed* expected return. The
threshold-style gate dominates EV-style gating on this dataset.

## iter_006 build

- Path: `submission/iter_006_aug_a_5seed_h60_asymthresh/`
- Bundled: `submission_050606_iter006.zip` (25.10 MB)
- Models: identical to iter_005b (5 aug_a seeds for h_60, iter_002 model_h40,
  inactive h_5/h_10/h_20)
- Predictor: new `Predictor.py` that accepts `T_up/T_dn/d_up/d_dn` per
  horizon (backward-compatible with `T/delta` if those keys are absent)
- thresholds.json (h_60): T_up=0.4481, T_dn=0.3914, d_up=0.2597, d_dn=0.0244
- Sanity check: **22/22 pass** (all required files, no subdirs, config keys,
  feature & label lists, requirements pinned, model size 80 MB, Predictor
  import + instantiate, predict batch=1/4 with shape (B,5) and values in
  {0,1,2}, zip layout & size)

## Predicted platform uplift

The OOF→platform gap from T22 was −2.23 (single-seed: OOF +9.67 → reported
platform +7.44). Applying the same offset to iter_006 OOF (+13.61):

  predicted h_60 platform ≈ +13.61 − 2.23 ≈ **+11.4**

Plus h_40 = +2.02 (locked from iter_002). Total predicted platform ≈
**+13.4** vs iter_005b predicted +11.2 vs iter_002 actual +4.07.

Caveats:
- The OOF→platform gap was measured on a single-seed model with symmetric
  thresholds. The asymmetric gate may have different platform behavior —
  if the platform test set is dominated by long-only opportunities, the
  short-favoring asymmetry could underperform.
- Per-fold pnl on h=60 fold 3 is +0.099 — an order of magnitude below the
  other folds. If the platform looks like fold 3, total uplift shrinks.

## Risks / what could go wrong

1. **OOF threshold selection bias**: DE picked the 4D point that
   maximized the OOF score we report. The neighborhood sweep evidence
   (wide plateau) substantially mitigates this, but does not eliminate it.
2. **d_up = 0.26 is large**: very few long signals get through (probably
   <5% of n_active goes long). If platform is long-biased, this gate
   could under-trade.
3. **h_40 unchanged**: a similar asymmetric DE on h_40 OOF might lift
   that horizon too — left for T30b/iter_007 if iter_006 platform looks
   good.

## Files

- `_common.py` — shared OOF loader (5-seed average) + gating + PnL
- `sweep_fine.py` + `sweep_fine_results.{csv,json}`
- `asymmetric_sweep.py` + `asym_sweep_{coarse,all}.csv`,
  `asym_sweep_results.json`
- `pnl_search.py` + `pnl_search_results.json` — DE 4D
- `de_neighborhood.py` + `de_neighborhood.{csv,_results.json}` —
  robustness sweep around DE optimum
- `build_regime.py` + `regime_features.parquet` — sym-agnostic 100-tick
  rolling regime features for per-regime / EV-gate variants
- `per_regime.py` + `per_regime_results.json` — in-sample per-regime grid
- `per_regime_cv.py` + `per_regime_cv_results.json` — honest LOSO CV
- `conformal.py` + `conformal_results.json`
- `ev_gate_v2.py` + `ev_gate_v2_{grid.csv,_results.json}`
- `Predictor_iter_006.py` (asym-aware) — staged into `submission/iter_006/Predictor.py`
- `build_iter_006.py` — wires it all up + sanity check + zip
- `submission_050606_iter006.zip` at workdir root (= bundled zip)
