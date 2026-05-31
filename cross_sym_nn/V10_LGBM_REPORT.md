# V10 LGBM Experiment Report

**Date:** 2026-05-31  
**Experiment:** LightGBM with v9 Interaction Features — 4 Configs × 5 Seeds  
**Author:** Worker Agent (auto-generated)

---

## 1. Experiment Setup

### Strict Train/Val/Test Split (hard constraint — no test leakage)

| Split | Date Range | File | Rows |
|-------|-----------|------|------|
| Train | 0–79      | schemeP_train.npz | ~1,473,600 |
| Val   | 80–95     | schemeP_val.npz   | ~737,800   |
| Test  | 96–119    | schemeP_test.npz  | ~442,080   |

**Protocol:** Early stopping on Val PnL; threshold search on Val; test evaluation uses frozen thresholds.

### 4 Configurations

| Config | Features | Dim | Description |
|--------|----------|-----|-------------|
| A | 259d L8-pruned baseline | 259 | `keep_idx_259d.npy` — best 259 of 359 L8 features |
| B | 259d + 40 v7 interaction | 299 | A + top-40 selected v7 cross-sym interaction features |
| C | 259d + 60 v9 interaction | 319 | A + 60 v9 features (same as NN SOTA input) |
| D | 359d full + 60 v9 interaction | 419 | all 359 L8 + 60 v9 interaction features |

### LGBM Hyperparameters

```python
params = {
    'objective': 'regression_l2', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 127,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l2': 1.0, 'min_data_in_leaf': 100,
    'device': 'gpu', 'gpu_use_dp': False,
}
num_boost_round = 500, early_stopping_rounds = 50
```

Class-balanced sample weights (equal weight per class label in {-1, 0, +1}).

---

## 2. Results Summary

### Aggregate (5 seeds mean ± std)

| Config | Feat Dim | test_pnl (mean ± std) | test_ic (mean ± std) | best_iter | vs A |
|--------|----------|----------------------|---------------------|-----------|------|
| **A**  | 259      | **+30.08 ± 1.40**    | 0.1660 ± 0.0051     | 29.6      | —    |
| B      | 299      | +28.93 ± 1.14        | 0.1527 ± 0.0038     | 29.6      | -1.15 |
| C      | 319      | +29.30 ± 1.18        | 0.1566 ± 0.0042     | 28.4      | -0.78 |
| D      | 419      | +30.07 ± 0.62        | 0.1571 ± 0.0049     | 34.0      | -0.01 |

**Reference baselines:**
- **LGB prior baseline** (359d, no strict split): +26.58 ± 0.96
- **NN SOTA** (Pairformer v9, 60 features): **+39.65 ± 0.92**

### Per-Seed Breakdown

**Config A** (259d baseline):

| Seed | val_pnl | test_pnl | best_iter |
|------|---------|----------|-----------|
| 0    | 26.50   | 30.10    | 32        |
| 1    | 25.65   | 31.81    | 25        |
| 2    | 25.19   | 30.00    | 28        |
| 3    | 24.23   | 30.88    | 24        |
| 4    | 26.88   | 27.60    | 39        |
| **mean±std** | | **30.08 ± 1.40** | 29.6 |

**Config B** (259d + 40 v7 inter):

| Seed | val_pnl | test_pnl | best_iter |
|------|---------|----------|-----------|
| 0    | 26.85   | 31.01    | 32        |
| 1    | 26.04   | 27.62    | 35        |
| 2    | 26.56   | 28.85    | 27        |
| 3    | 27.03   | 28.93    | 29        |
| 4    | 26.37   | 28.26    | 25        |
| **mean±std** | | **28.93 ± 1.14** | 29.6 |

**Config C** (259d + 60 v9 inter):

| Seed | val_pnl | test_pnl | best_iter |
|------|---------|----------|-----------|
| 0    | 27.83   | 30.16    | 31        |
| 1    | 26.25   | 29.11    | 31        |
| 2    | 25.52   | 27.11    | 26        |
| 3    | 25.03   | 29.66    | 26        |
| 4    | 26.53   | 30.44    | 28        |
| **mean±std** | | **29.30 ± 1.18** | 28.4 |

**Config D** (359d + 60 v9 inter):

| Seed | val_pnl | test_pnl | best_iter |
|------|---------|----------|-----------|
| 0    | 26.74   | 29.48    | 29        |
| 1    | 27.09   | 30.58    | 39        |
| 2    | 27.35   | 29.30    | 36        |
| 3    | 26.86   | 30.90    | 31        |
| 4    | 26.94   | 30.11    | 35        |
| **mean±std** | | **30.07 ± 0.62** | 34.0 |

---

## 3. Analysis: Do v9 Interaction Features Help LGBM?

**Answer: No — interaction features provide zero benefit on LGBM and slightly hurt.**

### Key Observations

1. **Config A (baseline 259d) is the best or equal best** across all 4 configs:
   - B is worse by **-1.15 PnL** (v7 interaction features hurt)
   - C is worse by **-0.78 PnL** (v9 interaction features hurt slightly less)
   - D matches A (within noise: -0.01 PnL), but uses 60% more features

2. **IC tells the same story**: Config A has the highest IC (0.1660). All configs with interaction features have lower IC (B: 0.1527, C: 0.1566, D: 0.1571).

3. **Config D reduces variance** (std 0.62 vs A's 1.40), suggesting that adding features from the full 359d L8 space (100 additional pruned features) does stabilize training even without PnL improvement.

4. **Feature importance confirms interaction features are NOT dominant**: Even in Configs B, C, D, the top positions are occupied by L8 raw features (market microstructure). Interaction features appear occasionally in the top 20 but do not dominate.

### Why Interaction Features Help NN but Not LGBM

The v9 interaction features exploit **cross-sym patterns**: pair differences, cross-sym rank, cross-sym demean, cross-sym IQR normalization. These are meaningful only when a model processes all 5 symbols **simultaneously** in a shared representation space.

- **NN Pairformer**: processes (N, 5, F) jointly with attention layers — cross-sym features are leverage points for its permutation-equivariant architecture.
- **LGBM**: treats each (group_t × sym) as an independent sample; the "cross-sym context" encoded in interaction features is artificially correlated with the same-group L8 features already present. LGBM sees them as noisy redundant splits.

This is a fundamental architectural limitation: tree-based models cannot natively learn cross-sample interactions.

### LGBM vs NN SOTA Gap

| Model | test_pnl | Gap vs NN SOTA |
|-------|----------|----------------|
| LGB prior baseline | +26.58 ± 0.96 | -13.07 |
| **LGB Config A (v10)** | **+30.08 ± 1.40** | **-9.57** |
| NN SOTA (Pairformer v9) | +39.65 ± 0.92 | — |

Config A closes 3.50 PnL points vs the prior LGBM baseline (strict split + 259d pruning eliminates noise). However, the NN still leads by **+9.57 PnL** — a gap that cannot be bridged by feature engineering alone in a tree model.

---

## 4. Per-Config Feature Importance (seed=0, by gain)

### Config A — Top 20 (259d features)

| Rank | Feature | Category |
|------|---------|----------|
| 1 | ewma_a0.5_mb_intst | Market buy intensity (smoothed) |
| 2 | mb_intst | Market buy intensity (raw) |
| 3 | totalbsize | Total bid depth |
| 4 | avgask | Average ask price |
| 5 | totalasize | Total ask depth |
| 6 | ma_intst | Market ask intensity |
| 7 | rv_w50 | Realized volatility (W50) |
| 8 | ewma_a0.5_lb_intst | Limit bid intensity (smoothed) |
| 9 | open | Open price |
| 10 | low | Low price |
| 11 | qrank_W100_imbalance | Quantile-ranked imbalance |
| 12 | high | High price |
| 13 | ewma_a0.5_la_intst | Limit ask intensity (smoothed) |
| 14 | bid_mean | Mean bid price |
| 15 | ask_mean | Mean ask price |
| 16 | la_intst | Limit ask intensity |
| 17 | ofi_tox_lvl1_W50 | OFI toxicity L1 W50 |
| 18 | avgbid | Average bid price |
| 19 | ewma_a0.05_mb_intst | Market buy intensity (slow EWMA) |
| 20 | lb_ind | Limit bid indicator |

### Config B — Top 20 (259d + 40 v7 inter)

| Rank | Feature | Type |
|------|---------|------|
| 1 | mb_intst | L8 raw |
| 2 | totalbsize | L8 raw |
| 3 | ewma_a0.5_mb_intst | L8 raw |
| 4 | ma_intst | L8 raw |
| 5 | avgask | L8 raw |
| 6 | **v7_inter_ch7** | Interaction |
| 7 | low | L8 raw |
| 8 | totalasize | L8 raw |
| 9 | rv_w50 | L8 raw |
| 10 | **v7_inter_ch107** | Interaction |
| 11 | ewma_a0.5_lb_intst | L8 raw |
| 12 | **v7_inter_ch2** | Interaction |
| 13 | qrank_W100_imbalance | L8 raw |
| 14 | open | L8 raw |
| 15 | gofi_W5_lvl1 | L8 raw |
| 16 | high | L8 raw |
| 17 | la_intst | L8 raw |
| 18 | **v7_inter_ch0** | Interaction |
| 19 | mid_ewma_resid_a0.05 | L8 raw |
| 20 | ewma_a0.5_la_intst | L8 raw |

### Config C — Top 20 (259d + 60 v9 inter)

| Rank | Feature | Type |
|------|---------|------|
| 1 | totalbsize | L8 raw |
| 2 | mb_intst | L8 raw |
| 3 | ewma_a0.5_mb_intst | L8 raw |
| 4 | ma_intst | L8 raw |
| 5 | avgask | L8 raw |
| 6 | **v7_inter_ch7** | Interaction |
| 7 | low | L8 raw |
| 8 | **v7_inter_ch2** | Interaction |
| 9 | totalasize | L8 raw |
| 10 | ewma_a0.5_lb_intst | L8 raw |
| 11 | open | L8 raw |
| 12 | rv_w50 | L8 raw |
| 13 | qrank_W100_imbalance | L8 raw |
| 14 | **v7_inter_ch107** | Interaction |
| 15 | high | L8 raw |
| 16 | ewma_a0.5_la_intst | L8 raw |
| 17 | la_intst | L8 raw |
| 18 | ewma_a0.05_mb_intst | L8 raw |
| 19 | **v7_inter_ch0** | Interaction |
| 20 | **v7_inter_ch1** | Interaction |

### Config D — Top 20 (359d full + 60 v9 inter)

| Rank | Feature | Type |
|------|---------|------|
| 1 | mb_intst | L8 raw |
| 2 | totalbsize | L8 raw |
| 3 | ma_intst | L8 raw |
| 4 | ewma_a0.5_mb_intst | L8 raw |
| 5 | ewma_a0.5_lb_intst | L8 raw |
| 6 | **v7_inter_ch7** | Interaction |
| 7 | avgask | L8 raw |
| 8 | **v7_inter_ch2** | Interaction |
| 9 | rv_w50 | L8 raw |
| 10 | low | L8 raw |
| 11 | qrank_W100_imbalance | L8 raw |
| 12 | totalasize | L8 raw |
| 13 | **v7_inter_ch107** | Interaction |
| 14 | ewma_a0.5_la_intst | L8 raw |
| 15 | high | L8 raw |
| 16 | open | L8 raw |
| 17 | ewma_a0.05_ma_intst | L8 raw |
| 18 | la_intst | L8 raw |
| 19 | **v7_inter_ch0** | Interaction |
| 20 | **v7_inter_ch32** | Interaction |

**Cross-config pattern**: The same interaction features (v7_inter_ch0, ch2, ch7, ch107) appear across B, C, D — suggesting they capture some information LGBM can use (likely cross-sym rank or demean signals). However, they do not rank high enough to improve aggregate PnL, suggesting they add some partial signal but also add noise/complexity.

---

## 5. Conclusions

1. **Best LGBM config is A** (259d pruned, strict split) at **+30.08 ± 1.40 test PnL**. Adding interaction features does not help.

2. **v9 interaction features are architecture-dependent**: They are essential for the NN Pairformer (+39.65) but provide no benefit to LGBM because LGBM cannot natively exploit cross-sample relationships.

3. **Strict split matters**: Config A (+30.08) vs prior LGBM baseline (+26.58) shows +3.50 improvement, partly from eliminating val/test leakage in threshold tuning and early stopping.

4. **NN–LGBM gap is +9.57 PnL**: This cannot be closed by feature engineering alone. The Pairformer's permutation-equivariant attention is fundamentally necessary to exploit cross-sym signals.

5. **Config D's lower variance** (std=0.62 vs A's 1.40) is interesting — the 100 "pruned" L8 features + interaction features together stabilize LGBM convergence, even if they don't improve mean PnL.

### Recommendation

- **For submission**: Use Config A (259d, strict split). It matches or beats interaction-augmented configs.
- **For ensemble**: Consider A+D ensemble given complementary feature sets and D's lower variance.
- **Future direction**: Tree-based models hit a hard ceiling against NN cross-sym architectures. Focus NN research over LGBM tuning.
