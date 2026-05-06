# T18: XGBoost LOSO h_10 (Scheme C 223-d)

**Goal**: Test whether XGBoost (heterogeneous algorithm vs LightGBM) adds diversity
to the iter_003 5-seed LightGBM ensemble (LOSO h_10 = +22.224, T=0.55, δ=0.10).

## Setup

- Cache: `experiments/T5b_features_multihorizon/cache/schemeC_{train,val,test}.npz` (226d → 223d, drop 3 time-encoding cols)
- 5-fold LOSO (held_out_sym 0..4); train sym!=held dates 0..79, val sym!=held dates 80..95, test sym==held dates 96..119
- Class-balanced sample weights (since XGBoost has no class_weight='balanced')
- `objective='multi:softprob', num_class=3, tree_method='hist', eta=0.05, num_boost_round=2000, early_stopping=50`

Per-seed diversity (mirrors T11 LightGBM):
| seed | subsample | colsample_bytree | max_depth | reg_lambda | min_child_weight |
|------|-----------|------------------|-----------|------------|------------------|
| 42   | 0.80      | 0.80             | 7         | 1.0        | 100              |
| 1    | 0.70      | 0.60             | 7         | 1.0        | 100              |
| 7    | 0.85      | 0.70             | 6         | 1.0        | 100              |
| 13   | 0.70      | 0.60             | 7         | 2.0        | 150              |
| 100  | 0.60      | 0.50             | 8         | 1.0        | 80               |

## Results

(populated after training completes — see `loso_train_T18_summary.json`,
`threshold_results_xgb.json`, `combine_results_*.json`)

### Per-fold OOF cum_pnl, seed=42

| held | XGBoost argmax | XGBoost best (T,δ) | LightGBM seed=42 best (T11) |
|------|----------------|--------------------|----------------------------|
| 0    | TBD            | TBD                | 1.934 (0.55, 0.15)         |
| 1    | TBD            | TBD                | 5.030                      |
| 2    | TBD            | TBD                | 4.328                      |
| 3    | TBD            | TBD                | 1.770                      |
| 4    | TBD            | TBD                | 8.653                      |
| sum  | TBD            | TBD                | 21.715                     |

### Algorithm comparison (h=10 LOSO)

| algo | best (T,δ) | sum_cum_pnl | per_fold ≥ 0 |
|------|------------|-------------|--------------|
| LightGBM seed=42 (T11) | (0.55, 0.15) | 21.715 | 5/5 |
| LightGBM 5-seed ensemble (T11) | (0.55, 0.10) | 22.224 | 5/5 |
| XGBoost seed=42 (T18) | TBD | TBD | TBD |

### Heterogeneous combine

| combine | best (T,δ) | sum_cum_pnl | uplift vs iter_003 |
|---------|------------|-------------|--------------------|
| LightGBM 5 + XGBoost 1 | TBD | TBD | TBD |
| LightGBM 5 + XGBoost 5 | TBD | TBD | TBD |

## Conclusion

(filled in after results land)
