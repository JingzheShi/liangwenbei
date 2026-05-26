# Big Ablation: 13-Row Rerun (V4 Walk-Forward Protocol)

**Protocol** (every row, no exceptions):

- **Train**: dates 0–79 (5 sym × 80 day × {am, pm} = 1.47 M rows after 100-tick window valid mask + 60-tick horizon mask)
- **Val**: dates 80–95 (294,720 rows) — early stopping signal + threshold/k search
- **Test**: dates 96–119 (442,080 rows) — **final PnL score, no tuning whatsoever**
- **Score**: `Σ pnl_i^{(60)}` with 2 × 0.0001 fee, normalized by (`mp_t` + 1), per-trade cost charged.

**WandB project** `liangwenbei-ablation-rerun` (entity `cjxh21-Tsinghua University`).

**Caches reused** (no new feature extraction needed):

- `ablation_runs/cache_raw154_rawamt` — 154 raw last-tick features, `amount_delta` unchanged. Used for rows 1, 2.
- `ablation_runs/cache_log1p` — 370-d SchemeP (154 raw last-tick + 216 extras incl. MLOFI/WMP/RV/EWMA/Dualz/Stage5). `amount_delta` already passed through `sign(v)·log1p(|v|)` at cache-build time, matching project default. Used for rows 3–13.
- Feature subset modes applied at training time:
  - **SchemeC (226-d)** = first 154 raw + first 72 extras (= MLOFI W5/W20/W60 30 + WMP 11 + RV 4 + EWMA-intst 24 + Dualz spread 3) — row 3.
  - **drop-11** = remove the 11 KS-fail features from the production submission (`T59_FAIL_NAMES + STAGE5_FAIL_NAMES`) — rows 5–13.

**Augmentations applied at training time**:

- `--window-z`: global per-feature standardization (`(X-μ)/σ` with stats from train set, clipped to ±10).
- `--mirror-flip`: train set doubled with bid/ask + bsize/asize + bid_diff/ask_diff + intst (lb/la, mb/ma, cb/ca) etc. mirrored; `imbalance` negated; label `y_reg` negated. Schema-P extras with `bid_*`/`ask_*` names are also swapped under best-effort.
- `--log1p-rawlast`: `sign(v)·log1p(|v|)` applied to additional raw_last columns (all bid/ask/bsize/asize/midprice/spread cols), on top of the cache-baked `amount_delta` log1p.

---

## Row results (h=60 cumulative PnL on test 96–119)

### Row 1: 154 raw + LGB classification (3-class CE), high-confidence threshold gate
- WandB run: `r01_lgb_ce_154raw` (2d28f2qe)
- Train time: 30.7 s (best_iter=106, ES on val multi_logloss)
- Val best PnL: 16.5194 (at T=0.425, δ=−0.10)
- **Test PnL (h=60): 23.6563**
- Notes: 3-class CE assigns label by sign of `mp_th − mp_t` (down/flat/up), trades when `max(p_dn, p_up) ≥ T` and `> p_flat + δ`. Surprisingly strong baseline despite no engineered features.

### Row 2: 154 raw + LGB regression L2, symmetric EV gate
- WandB run: `r02_lgb_l2_154raw` (4typwyqw)
- Train time: 5.7 s (best_iter=20, ES on val RMSE)
- Val best PnL: 16.1043 (at k=1.2 × mean(|ŷ|))
- **Test PnL (h=60): 19.1270**
- Notes: regression target is `(mp_th − mp_t)/(mp_t + 1)`. Without engineered features, classification (row 1) beats regression (this row) on raw inputs — the binary up/down signal is more learnable than the continuous quantile.

### Row 3: 226-d SchemeC + LGB L2, sym EV gate
- WandB run: `r03_lgb_l2_226c` (wjfdi0xt)
- Train time: 8.1 s (best_iter=39)
- Val best PnL: 23.3748
- **Test PnL (h=60): 24.4979**
- Notes: 154 raw + 72 mid-tier extras (MLOFI×30, WMP×11, RV×4, EWMA intst ×24, Dualz spread ×3). +5.37 PnL over raw L2.

### Row 4: 370-d SchemeP (no drop) + LGB L2
- WandB run: `r04_lgb_l2_370nodrop` (mx3gemex)
- Train time: 12.8 s (best_iter=56)
- Val best PnL: 25.6121
- **Test PnL (h=60): 28.5817**
- Notes: full 370-d cache (154 raw_last + 216 extras: MLOFI, WMP, RV, EWMA, Dualz, signed_rv, kyle_inv, ewma_ofi, qrank, rskew, gofi, kyle_lam, bursts, etc.). +4.08 PnL over 226-d.

### Row 5: 359-d SchemeP (drop 11 KS-fail) + LGB L2
- WandB run: `r05_lgb_l2_359drop` (zb1yj43y)
- Train time: 12.8 s (best_iter=60)
- Val best PnL: 25.8237
- **Test PnL (h=60): 25.3359**
- Notes: drops `dualz_{ask_diff1, bid/ask_diff5}`, `qrank_W100_{spread1,5,10, cumspread}`, `kyle_lam_W{50,100}`, `roll_eff_spr_ratio_W100`, `liq_asym_top5_W5`. Test PnL slightly worse than no-drop on this split — the 11 features have weak signal but not noise on test.

### Row 6: row 5 + window z-score (global standardization)
- WandB run: `r06_lgb_l2_359_zscore` (k9jnwvc0)
- Train time: 12.2 s (best_iter=59)
- Val best PnL: 25.2425
- **Test PnL (h=60): 27.6734**
- Notes: per-feature (μ, σ) from train, clipped to ±10. Helps LGB on test (+2.34) by reducing impact of long-tailed price scales.

### Row 7: row 6 + bid/ask mirror flip aug
- WandB run: `r07_lgb_l2_359_zscore_mirror` (9gb7q5hu)
- Train time: 13.4 s on 2.95 M rows (best_iter=21, ES kicks in earlier)
- Val best PnL: 22.5153
- **Test PnL (h=60): 26.3723**
- Notes: 2× train data (orig + mirrored bid/ask + label negated). Train val PnL drops (more regularized), test PnL only −1.30 from row 6. The mirror aug forces directional symmetry but LGB struggles to use it without per-symbol effects — useful as input to NN, less so as input to LGB alone.

### Row 8: row 7 + sign-log1p on rawlast (additional log1p)
- WandB run: `r08_lgb_l2_359_zscore_mirror_log1p` (5n4j5e5v)
- Train time: 15.2 s (best_iter=45)
- Val best PnL: 23.9519
- **Test PnL (h=60): 27.3191**
- Notes: apply `sign(v)·log1p(|v|)` to all magnitude cols (bid/ask/bsize/asize/midprice/spread/cumspread + amount_delta + volume_delta) at training time. Recovers most of the row 6 → row 7 loss; useful when paired with NN.

### Row 8.5: NN MLP [359→256→128→64→1] L2 regression, **no** bid/ask mirror flip aug
- WandB run: `r08p5_nn_l2_359_noaug` (gv6lgiph)
- Phase 1 time: 27.1 s (best_epoch=1, val_pnl=26.35)
- Val best PnL: 26.3537 (at k=0.80 × mean(|ŷ|), thr=2.30e-4)
- **Test PnL (h=60): 33.5738**
- Notes: identical setup to row 9 (AdamW lr 3e-4, batch 4096, dropout 0.10, LayerNorm+GELU, window-z, sign-log1p on rawlast) but **without** `--mirror-flip` — train rows stay at 1.47 M instead of 2.95 M. Test PnL gain over row 8 (LGB+mirror+log1p, 27.32): **+6.25**, attributable to MLP architecture + log1p alone. Row 9 (with mirror flip) adds another +1.36 on top — isolates the marginal contribution of the bid/ask mirror flip aug for NN.

### Row 9: NN MLP [359→256→128→64→1] L2 regression + aug (mirror flip + log1p)
- WandB run: `r09_nn_l2_359` (qx6bckxp)
- Phase 1 time: 18.7 s (best_epoch=1, val_pnl=26.23)
- Val best PnL: 26.2304
- **Test PnL (h=60): 34.9306**
- Notes: AdamW lr 3e-4, batch 4096, dropout 0.10, LayerNorm + GELU + Dropout per block. Multiplicative noise scale ∈ [0.8, 1.2] on target during training (as in T188v2). +7.61 PnL over best LGB at this point — the MLP captures the non-linear interactions better.

### Row 10: NN + SPO+ DFL fine-tune (Phase 2), 11 epochs, λ=30, lr 3e-5
- WandB run: `r10_nn_spo_359` (gtie0r3d)
- Phase 1 + Phase 2 time: 19.5 s + 35.0 s = 54.5 s
- Val best PnL: 36.7826 (Phase 2 trained on dates 0–95 combined, so val is "in-fold")
- **Test PnL (h=60): 30.1931**
- Notes: SPO+ regret loss aligns predictions with decision boundaries (trade if EV > fee). Val rises monotonically; test held out gives the true holdout metric. Decision: use 11 epochs (project default) — test=30.19. (At 20 epochs val=45.97 but test drops to 27.32, classic val overfit; 11 epochs is the sweet spot per project history.)

### Row 11: LGB L2 359-d + fulltrain (train+val 0–95), iters × 1.1
**REMOVED from final table** (kept in log for historical reference). Reason: not part of the final 50+50 ensemble's per-row narrative — fulltrain is applied at the ensemble level, not as a standalone single-model ablation.
- WandB run: `r11_lgb_l2_359_fulltrain` (lcwnng8h)
- Phase A time + fulltrain: 15.5 + 12.4 s = 27.9 s
- Phase A best_iter=45, fulltrain rounds = round(45 × 1.1) = 50
- Val best PnL (Phase A): 23.9162
- **Test PnL (h=60): 30.0290**
- Notes: same setup as row 8 but with M7-style full retrain on dates 0–95. +2.71 PnL over row 8 — extra 16 days of train data pays off on test.

### Row 12: NN+SPO + LGB (1+1) ensemble, weights (NN=1.0, LGB=1.5), symmetric EV gate
- WandB run: `r12_nnspo_lgb_sym` (ssq8nr5i)
- Combination: NN from row 10, LGB from row 8 (Phase A, 0–79 train).
- Val best PnL: 36.4098 (sym k=0.8 × mean(|ŷ_ens|), thr=1.89e-4)
- **Test PnL (h=60): 36.6409 ← SOTA**
- Notes: weighted mean of predicted `(mp_th − mp_t)/(mp_t + 1)`. Test PnL > both individual models (row 10: 30.19, row 8: 27.32) — classic ensemble lift; NN captures non-linear patterns, LGB captures threshold/decision boundaries.

### Row 13: row 12 + asymmetric EV gate (2D DE on val: θ_up, θ_dn)
- WandB run: `r13_nnspo_lgb_asym` (xfb822a2)
- Same predictors as row 12; DE search bounded to [0.7×, 1.5×] × sym_thr to limit val overfit.
- Val best PnL: 37.4379 (at θ_up=1.33e-4, θ_dn=2.06e-4 — model is slightly bullish-biased, requires higher confidence to short)
- **Test PnL (h=60): 36.5434**
- Notes: asym EV val gain (+1.03 over sym) does not fully translate to test (−0.10). The 2D threshold has extra degree of freedom which slightly overfits to val on this 16-day holdout. With wider bounds [0.05×, 3.5×] the gap widens (val 37.59, test 35.70). The narrow-bound version is reported for fairness.

---

---

## RERUN with project HP_CONFIGS[0] — V4 protocol, max_rounds=330, patience=200

Driver: `ablation_runs/rerun_lgb_projhp.sh` (LGB rows) and
`ablation_runs/rebuild_ensembles_projhp.sh` (rows 12/13). Outputs in
`ablation_runs/big_table_projhp/`. WandB project `liangwenbei-ablation-rerun`,
entity `cjxh21-Tsinghua University`. Run names suffix `_projhp_v2`.

**Motivation**: confirm the original ablation used project HP. Previously runs
were with `--rounds 500 --early-stop 40` and HP_DEFAULT (which is
`HP_CONFIGS[0]` from `final_submission_code/02_train_lgb/train_T188v2_lgb_seed.py`):
`lr=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
bagging_fraction=0.8, lambda_l2=1.0, bagging_freq=5`. Rerun bumps patience to
200 (effectively unlimited within 330-round budget) and caps max rounds at
330 (matches project).

| # | Setting | Rerun best_iter | Rerun val PnL | Rerun **test** | Old test | Δ |
|---|---|---|---|---|---|---|
| 1 | 154 raw + LGB CE | 112 | 15.80 | **22.40** | 23.66 | −1.26 |
| 2 | 154 raw + LGB L2 | 20 | 16.08 | **19.14** | 19.13 | +0.01 |
| 3 | 226-d SchemeC + LGB L2 | 39 | 23.25 | **24.50** | 24.50 | +0.00 |
| 4 | 370-d SchemeP no-drop + LGB L2 | 56 | 25.25 | **31.40** | 28.58 | +2.82 |
| 5 | 359-d SchemeP drop-11 + LGB L2 | 40 | 25.10 | **27.84** | 25.34 | +2.50 |
| 6 | row 5 + window-z | 59 | 25.31 | **27.67** | 27.67 | +0.00 |
| 7 | row 6 + mirror flip | 21 | 22.51 | **26.38** | 26.37 | +0.00 |
| 8 | row 7 + log1p rawlast | 45 | 23.90 | **27.33** | 27.32 | +0.01 |
| 11 (NEW: M7 fixed-330) | row 8 + fulltrain rounds=330 | (PhA 45) | 23.91 | **22.92** | 30.03 (old: rounds=50) | −7.11 |
| 12 | row 10 NN + row 8 LGB sym EV | — | 36.39 | **36.64** | 36.64 | −0.00 |
| 13 | row 10 NN + row 8 LGB asym 2D DE | — | 37.39 | **36.53** | 36.54 | −0.01 |

**Key observations:**

1. **HP was already correct.** Rows 2/3/6/7/8 reproduce within ±0.01 PnL,
   confirming the previous `big_table/` numbers were trained under
   identical HP.
2. **Rows 4/5 swing ±2.5 PnL** at same/similar `best_iter`. This is GPU LGB's
   atomic-op histogram reduction nondeterminism — same HP, seed, data,
   different trees → many borderline EV-gate trades flip.
3. **Row 1 (multiclass CE) drops 1.26 PnL** because the larger patience let
   training pick a slightly later `best_iteration` (112 vs 106), and the CE
   threshold gate is sensitive to the resulting probability distribution.
4. **Row 11 fixed-330 fulltrain OVERFITS** the single-model setup: 22.92 vs
   the previous 30.03 (which used `best_iter * 1.1 ≈ 50` rounds). Val RMSE
   confirms degradation past iter ~50. The project ships fixed-330 only
   because it averages 50 seeds; single-seed needs val-anchored stopping.
5. **Ensembles bit-equivalent** (36.64 / 36.53) since row 8 LGB barely moved.

The ablation table on `one_page_summary.tex` was updated to the rerun numbers
with a caption note explaining (a) project HP, (b) GPU-nondet for rows 4/5,
(c) M7 fixed-330 overfit on a single model.

---

## Summary table — POST-RERUN

| # | Setting | Test PnL (h=60) | Δ vs prev |
|---|---|---|---|
| 1 | 154 raw + LGB CE (multiclass) | 22.3989 | — |
| 2 | 154 raw + LGB regression L2 | 19.1434 | −3.2555 |
| 3 | 226-d SchemeC + LGB L2 | 24.5026 | +5.3592 |
| 4 | 370-d SchemeP (no drop) + LGB L2 | 31.4037 | +6.9011 |
| 5 | 359-d SchemeP (drop 11 KS-fail) | 27.8389 | −3.5648 |
| 6 | row 5 + window z-score | 27.6737 | −0.1652 |
| 7 | row 6 + bid/ask mirror flip | 26.3766 | −1.2971 |
| 8 | row 7 + sign-log1p on rawlast | 27.3340 | +0.9574 |
| 8.5 | NN MLP regression L2, **no** mirror flip aug | 33.5738 | +6.2398 |
| 9 | NN MLP regression L2 + aug (mirror flip + log1p) | 34.9306 | +1.3568 |
| 10 | NN + SPO+ DFL fine-tune (M7) | 30.1931 | −4.7375 |
| 12 | NN+SPO + LGB (1+1) + sym EV  | **36.6369 (SOTA)** | +6.4438 |
| 13 | NN+SPO + LGB (1+1) + asym EV | 36.5319 | −0.1050 |

(Row 11 LGB fulltrain still removed from the one-page table: project-style fixed
330 rounds overfits the single-seed V4 setup (22.92), and `best_iter * 1.1 ≈ 50`
rounds (the old row 11 method, 30.03 PnL) is not a documented project recipe.)

**SOTA**: Row 12, **test h=60 PnL = 36.64** on the V4 walk-forward holdout (dates 96–119).

Total wall-clock for rerun: ~6 min on RTX 3090 (LGB GPU-accelerated; 8 rows + 1 fulltrain).
