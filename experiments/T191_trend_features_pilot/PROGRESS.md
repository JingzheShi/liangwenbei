# T191 Trend Features Pilot — Progress Log

**Hypothesis**: Do 5 signed cumulative log-return / momentum features add residual IC on top of existing 359-d (actually 364-d after drop) schemeP features?

**Why a pilot**: A 4-hour 150+150 retrain is wasteful unless the trend features show independent predictive content. We train a quick 5+5 ensemble on 375-d (370 + 5 trend - 11 drop = 364) and orthogonalize the prediction against T188v2 50+50 to measure residual IC.

## Decisions / Choices

- **Trend features (last-tick aligned, causal)**:
  - mean_logret_W{10,20,50,100} (mean per-tick log return over W)
  - mid_pct_change_W100 (net 100-tick pct change)
- **Feature count**: 370 schemeP + 5 trend = 375 raw, minus 11 DROP_NAMES = **364** model in_dim
- **Seeds**: {1, 7, 13, 42, 100} matching T170 baseline (5 NN + 5 LGB)
- **Protocol**: T188v2 (T81 pretrain + T170 SPO+ M7 finetune, 11ep@λ=30, lr=3e-5)
- **Cache strategy**: save only the 5 new cols as `trend_feat_*.npz` (~45 MB) and concat at load time; do NOT rebuild the entire 3GB schemeP cache (alignment verified)

## Steps

1. ✅ Built `cache/trend_feat_{train,val,test}.npz` from raw parquets (4s total, alignment OK)
2. ✅ Wrote `train_T191_{nn,lgb}_seed.py` with concat at load
3. ⏳ Get remote GPU
   - First tried M2 (36331333, ssh9.vast.ai:11332) — SSH banner-timeout repeatedly even though `vastai show` says `running`. Gave up.
   - Rented fresh: instance 36511889 (RTX 3080, R=99.7, $0.121/hr, ssh1.vast.ai:31888) — booting
4. ⏳ Push cache + scripts; run 5+5 training (~60 min total)
5. ⏳ Pull models back; run `eval_T191_holdout.py`
6. ⏳ Verdict
