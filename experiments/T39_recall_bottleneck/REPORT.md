# T39 — Recall Bottleneck Diagnosis on iter_006

**Question** *(from user)*: "我们的 accuracy 和公榜较好选手差不多，但 recall 较差" — is recall really the PnL bottleneck?

**TL;DR**: **No, recall is not the bottleneck — the model is.** We are already at the fee-aware threshold saturation point: the marginal trades we'd activate by lowering T have **<break-even precision** (≈0.20–0.40), so any naive recall increase loses PnL. Meanwhile our `F0.5_macro` is only `0.235` vs iter_002 `0.209` — practically the same model skill; almost all of our recent platform PnL gains came from **threshold engineering, not model quality**. To break out, build a **better classifier** (saturation path), not a more permissive gate.

---

## 1. Setup

- 5-seed (42, 1, 7, 13, 100) `aug_a` h=60 LightGBM ensemble OOF predictions, 5 LOSO folds.
- Probability source: `experiments/T26_domain_randomization/loso_pred_h60_aug_a_held{K}.parquet` + `experiments/T27_iter005/loso_pred_h60_aug_a_seed{1,7,13,100}_held{K}.parquet`. (5-seed prob mean.)
- iter_006 gating: asymmetric 4D `T_up=0.4481, T_dn=0.3914, d_up=0.2597, d_dn=0.0244`.
- Sym-agnostic / stateless / no-date — strict CRITICAL_CONSTRAINTS.md compliance.

## 2. Oracle vs iter_006

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Sum** |
|---|---:|---:|---:|---:|---:|---:|
| Oracle PnL (perfect labels) | +32.58 | +110.64 | +70.05 | +40.98 | +125.95 | **+380.20** |
| iter_006 PnL | +1.82 | +2.64 | +1.48 | +0.10 | +7.57 | **+13.61** |
| Gap to oracle | 30.76 | 108.00 | 68.57 | 40.88 | 118.38 | **366.59** |

We capture only ~3.6% of the oracle PnL pool. But oracle PnL is unrealistic — it's the upper bound under "predict every label perfectly." A more useful question: **how is the gap distributed**?

## 3. Error decomposition (per-row, signed PnL)

For each held-out row with iter_006 prediction, classify into:
- **A — missed profitable**: `iter006 = 1` AND `true_label ∈ {0, 2}` → would-have-been profit (signed by oracle PnL on these rows).
- **B — wrong active**: `iter006 ∈ {0, 2}` AND `iter006 != true_label` → realized loss (fee + adverse direction).
- **C — correct active**: `iter006 ∈ {0, 2}` AND `iter006 = true_label` → realized profit.
- (Idle correct: `iter006 = 1` AND `true_label = 1` → 0 contribution.)

Identity check: `iter_006 PnL = B + C = +13.60` ≈ +13.61 ✓.

| Bucket | Sum PnL | # rows | Per-row mean PnL |
|---|---:|---:|---:|
| A — missed profitable (true≠1, pred=1) | **+181.79** | n_A across folds | small per-row |
| B — wrong active (pred wrong side) | **−101.82** | n_B | sizeable loss |
| C — correct active (pred right side) | **+115.42** | n_C | sizeable gain |

**At face value, A=$182 (~50% of the gap) suggests "we're leaving lots of recall on the table."**
But that interpretation collapses once we ask **at what marginal precision** we'd be activating those trades.

## 4. Threshold sweep (symmetric, δ=0)

For each fold, scan T ∈ [0.30, 0.65] step 0.01:

| Threshold T | Joint sum PnL (5 folds) | Per-fold PnL |
|---|---:|---|
| 0.30 | −1.58 | [+1.57, −3.93, −7.08, +0.12, +7.75] |
| 0.40 | +6.53 | [+2.51, −1.40, −4.07, +0.17, +9.30] |
| 0.45 | +11.43 | [+1.30, +2.13, −0.28, +0.11, +8.18] |
| **0.47** | **+11.83 (max)** | [+0.83, +3.37, +0.89, +0.11, +6.63] |
| 0.50 | +10.72 | [+0.40, +3.71, +1.64, +0.06, +4.92] |
| 0.55 | +6.31 | [−0.01, +2.37, +1.13, +0.03, +2.79] |
| 0.60 | +3.19 | [−0.02, +1.15, +0.62, +0.01, +1.42] |

- Joint best symmetric T (δ=0) = **0.47 → +11.83**. Below iter_006 asymmetric **+13.61** by 1.78 — the asymmetric 4D gate already extracts most available threshold leverage.
- Per-fold T*<sub>PnL</sub>: [0.39, 0.49, 0.51, 0.40, 0.41]. Per-fold T*<sub>F0.5</sub>: [0.39, 0.40, 0.43, 0.30, 0.41]. F0.5 prefers slightly more recall on every fold than PnL does; this gap is the metric–reward tension.

PR-PnL curves per fold: see `pr_pnl_curve_fold{0..4}.png`.

## 5. Marginal precision below break-even — saturation evidence

For each fold, scan T from `T*+0.05` down to `T*−0.05` and measure **precision and PnL of newly-activated trades** at each step.

Random-pred baseline for marginal precision is ≈ 0.33 (3-class with side-argmax). The fee-break-even precision depends on the typical move size; for fee=0.0001 single-leg the analytic break-even is `prec_be = 0.5 + fee / (2 · |move|)` — for typical 5–10 bps moves that lands at **0.55–0.75**.

| Fold | Marginal precision at T = T*−0.02 | Marginal PnL at T = T*−0.02 |
|---|---:|---:|
| 0 | 0.13 | **−0.32** (negative) |
| 1 | 0.35 | **−0.10** |
| 2 | 0.27 | **−0.09** |
| 3 | 0.22 | −0.011 |
| 4 | 0.30 | **−0.18** |

For **every fold**, dropping T 2 ticks below the per-fold optimum already produces **net-negative marginal PnL**. Trades activated by lower thresholds are systematically lower-edge than the fee can cover.

→ **Threshold-lowering cannot recover A.** The probability mass we'd recruit isn't reliably on the right side, and the per-trade edge is too thin to overcome 2× fee.

## 6. F0.5 macro at iter_006 thresholds

| Fold | Recall macro | Precision macro | F0.5 macro |
|---|---:|---:|---:|
| 0 | 0.099 | 0.380 | 0.153 |
| 1 | 0.300 | 0.370 | 0.347 |
| 2 | 0.340 | 0.297 | 0.305 |
| 3 | 0.005 | 0.436 | 0.025 |
| 4 | 0.328 | 0.418 | 0.347 |
| **Mean** | 0.215 | 0.380 | **0.235** |

**iter_006 F0.5 = 0.235 vs iter_002 platform F0.5 = 0.209 → only +0.026 model-skill lift.**
- Folds 0 and 3 are recall-anaemic (0.099 and 0.005). Fold 3 is the OOD sym=3 fold (~0.18% activation rate) — model is almost silent there; that explains the user's recall complaint at the leaderboard level.
- Folds 1, 2, 4 (`held_sym ∈ {1,2,4}`) recall is 0.30-0.34, which is where most of the realized PnL comes from.

## 7. Diagnosis

- **A is large but unrecoverable by threshold changes.** Marginal precision below break-even on every fold confirms we're at saturation.
- **B (wrong active) is also non-trivial (−$102).** Improving model calibration / reducing false-active rate is at least as valuable as improving recall.
- **F0.5 macro 0.235 ≈ iter_002 0.209.** Model classification skill barely moved between iter_002 and iter_006. The platform PnL gain (+4.07 → predicted ~+11.4) is **threshold engineering**, not model improvement.
- The user's observation ("similar accuracy, worse recall vs leaders") is consistent with the leaders having a slightly stronger underlying classifier (better F0.5), not just a more aggressive threshold.

## 8. Recommended path: **SATURATION** (model improvement)

Priority order — biggest expected uplift first:

1. **Push F0.5_macro of the base classifier**. Targets:
   - Stronger feature set on the active-class boundary (T22 alpha101/191 already in; try alpha158 cross-sectional ranks, microstructure imbalance higher-order moments, ReVol/Savitzky already in T35 — check residual gain).
   - Calibrated multitask head (single h=60 + auxiliary horizons) — T12 stacking didn't help much; revisit with proper temperature scaling on h=60 only.
   - Replace LightGBM head with deep tabular (TabPFN / SAINT / FT-Transformer) for the active-class-vs-flat margin only — fee-aware loss can be plugged in.
2. **Per-regime gating with honest LOSO CV.** T30 per-regime in-sample claimed +9.4, but it pooled OOF including held fold (biased). Run the unbiased per-regime LOSO CV (T30 has the script wired but not finalized); even half of the bias-corrected gain would be larger than any threshold tweak.
3. **Targeted boost on fold-0 / fold-3 pattern**. Both folds have very low active-rate (<5% per-fold trade share of label). Cross-sym mixup (T14) targeted at low-volatility regimes, or domain-randomized augmentation focused on quiet sessions, may lift recall *without* lowering threshold.
4. **Reduce B (wrong active)**. Train a small **direction-confidence calibrator** (logistic on `[prob_2 - prob_0, prob_2 - prob_1, prob_0 - prob_1, regime_features]`) that re-weights the gate per regime. Current d_up=0.260 is harsh on up-side; symmetric calibration may sharpen it.

**Explicitly NOT recommended**:
- Lower T below iter_006's asymmetric 4D — saturation, will lose PnL.
- More aggressive ensemble of more LGBM seeds — diminishing return; T28 already showed +1.78 from 1→5 seeds.
- Pure threshold optimization beyond DE_4D — checked exhaustively in T30, near-saturation.

## 9. Numeric anchors for `results.json`

```
oracle_total_pnl       = +380.20
iter006_total_pnl      = +13.61
gap_to_oracle          = +366.59
A_missed_profitable    = +181.79  (49.6% of gap)
B_wrong_active         = -101.82  (-27.8% of gap)
C_correct_active       = +115.42
joint_T_optimal_delta0 = 0.47 → +11.83
iter006_asym_sum_pnl   = +13.61   (vs joint sym +11.83 → asym buys +1.78)
iter006_F0.5_macro     = 0.235
iter002_F0.5_macro     = 0.209  (delta = +0.026)
recommended_path       = SATURATION (model improvement, not threshold)
```

## 10. Files

- `diagnose.py` — driver
- `diagnose.log` — run log
- `results.json` — every numeric result above (per-fold + totals + sweep + marginal band)
- `pr_pnl_curve_fold{0..4}.png` — PR vs PnL curves with iter_006, oracle, and PnL/F0.5 optima marked
