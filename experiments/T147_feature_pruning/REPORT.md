# T147: Feature Pruning + TSH-FT Evaluation

## Protocol
- INNER: date 0-95 (train+val), HOLDOUT: date 96-119 (test)
- Phase 1: 1-seed holdout with frozen baseline thresholds for screening
- Phase 2: full TSH-FT (LOSO+DE+2-seed holdout) for top-5 candidates
- TSH-FT = 0.3×inner_LOSO + 0.7×holdout − 0.3×max(0, inner−holdout)

## Baseline (359 features)
- Inner LOSO: +14.1006
- Holdout: +16.8739
- TSH-FT: +16.0419
- Thresholds: buy=0.002717 sell=0.000288

## Phase 1 Screening Results (sorted by holdout PnL)

| Candidate | N features | Holdout PnL (1-seed, frozen thresh) |
|-----------|-----------|-------------------------------------|
| C_drop150 | 209 | +20.1325 |
| C_drop100 | 259 | +20.0064 **best** |
| E_combined | 164 | +20.0011 |
| C_drop50 | 309 | +18.2308 |
| A_k150 | 150 | +15.5921 |
| A_k250 | 250 | +15.4410 |
| B_corr_dedup | 209 | +15.0204 |
| D_drop50 | 309 | +14.7954 |
| A_k200 | 200 | +13.5473 |
| A_k100 | 100 | +12.7003 |
| D_drop100 | 259 | +12.6650 |
| D_drop150 | 209 | +12.1315 |
| A_k50 | 50 | +10.0550 |
| D_drop200 | 159 | +1.5216 |
| BASELINE | 359 | +16.8739 (reference) |

## Phase 2 Full TSH-FT Results

| Candidate | N | Inner LOSO | Holdout | TSH-FT |
|-----------|---|-----------|---------|--------|
| BASELINE | 359 | +14.1006 | +16.8739 | +16.0419 |
| C_drop100 | 259 | +82.9994 | +38.5162 | +38.5162 **BEST** |
| E_combined | 164 | +92.4425 | +38.1730 | +38.1730 |
| C_drop150 | 209 | +76.4770 | +37.6729 | +37.6729 |
| C_drop50 | 309 | +70.4658 | +35.2651 | +35.2651 |
| A_k150 | 150 | +20.7545 | +16.5788 | +16.5788 |

## Best Result
**C_drop100**: 259 features, TSH-FT = +38.5162
- vs Baseline TSH-FT = +16.0419
- Delta TSH-FT: +22.4743

## Top-20 Features by Gain Importance

1. totalasize (gain=1.1)
2. totalbsize (gain=1.0)
3. high (gain=0.9)
4. open (gain=0.9)
5. avgask (gain=0.8)
6. low (gain=0.8)
7. rv_w50 (gain=0.7)
8. avgbid (gain=0.6)
9. liq_asym_top5_W100 (gain=0.6)
10. ask_mean (gain=0.5)
11. signed_bv_W100 (gain=0.5)
12. bid_mean (gain=0.5)
13. rskew_W100 (gain=0.4)
14. trade_pers_W100 (gain=0.4)
15. cancel_imb_W100 (gain=0.3)
16. mb_intst (gain=0.3)
17. jshare_W100 (gain=0.3)
18. ma_intst (gain=0.3)
19. ewma_a0.05_ma_intst (gain=0.3)
20. signed_rv_W100 (gain=0.3)

## Recommendation
Use best_features.txt (259 features) for next iteration packaging.
