# T8: sym=2 brittleness diagnostic

**Goal**: Find why sym=2 is the worst LOSO fold (-7.07 / -8.12 / -0.98 across iters), to guide feature engineering.

## TL;DR

sym=2 is the **highest-volume / tightest-spread / lowest-per-tick-volatility** stock — looks like a large-cap blue-chip / ETF. Models trained LOSO on syms {0,1,3,4} (lower-volume, wider-spread, higher-vol) read sym=2's baseline microstructure as "abnormal accumulation" and over-predict UP at **62% rate**, while sym=2's true UP rate in dates 96–119 is only **19%**. Net effect: model is structurally biased, even after thresholding.

The fix is **NOT** sym-specific features (banned by hard constraint) but **scale-invariant / cross-sectional / ratio features** that make sym=2 look like the same class as other syms.

## 1. Per-sym statistical profile

| metric | sym=0 | sym=1 | sym=2 | sym=3 | sym=4 | sym=2 z vs others |
|---|---|---|---|---|---|---|
| midprice mean (rel close) | -0.0005 | -0.0011 | **-0.0026** | -0.0009 | -0.0016 | -4.24 |
| mid_diff std (per-tick vol) | 2.74e-4 | 3.38e-4 | **2.14e-4** | 3.86e-4 | 3.18e-4 | -2.85 |
| amount_delta mean (CNY) | 39 220 | 30 206 | **154 284** | 22 098 | 63 284 | **+7.49** |
| spread1 q95 (=ask1-bid1-1) | -0.9986 | -0.9984 | **-0.9990** | -0.9976 | -0.9982 | -2.33 |
| label_60 down/flat/up | .20/.62/.18 | .29/.45/.26 | .25/.53/.22 | .18/.67/.16 | .31/.42/.27 | mid-pack |
| big_60_move_rate (>0.1%) | 0.370 | 0.541 | 0.458 | 0.326 | 0.566 | mid-pack |

### Top 10 anomaly dimensions (|z| of sym=2 vs other-4 average)

| # | metric | sym=2 | others mean | z |
|---|---|---|---|---|
| 1 | **amount_delta_raw_mean (CNY)** | 1.54e5 | 3.87e4 | **+7.49** |
| 2 | volume_delta median | 1.23e-4 | 2.52e-5 | **+5.50** |
| 3 | log_amount_delta max | 17.25 | 16.36 | +5.35 |
| 4 | midprice mean (downtrend) | -0.0026 | -0.0010 | -4.24 |
| 5 | log_amount_delta mean | 9.38 | 5.99 | +4.21 |
| 6 | log_amount_delta q95 | 13.31 | 11.94 | +3.64 |
| 7 | mid_diff std (per-tick vol) | 2.14e-4 | 3.29e-4 | **-2.85** |
| 8 | log_amount_delta std | 4.33 | 4.87 | -2.76 |
| 9 | log_amount_delta median | 10.84 | 8.02 | +2.55 |
| 10 | totalbsize min | 0.033 | 0.012 | +2.34 |

### Interpretation: what kind of stock is sym=2?

Composite picture:
- **High trading volume** (~4× others in CNY)
- **Tight spread** (smallest ask-bid difference)
- **Low per-tick volatility** (smallest |Δmid| per tick)
- **Deep books** (totalbsize / totalasize floor higher)

→ Likely a **large-cap blue-chip or ETF**. Other syms (esp. 1, 4) look like more volatile single-name equities.

## 2. T2 Scheme B LOSO held=2 prediction error analysis

OOF predictions (88 416 rows, dates 96–119):

```
            pred=0(down)  pred=1(flat)  pred=2(up)
true=0(down)     0.301        0.107        0.592
true=1(flat)     0.173        0.219        0.608   ← model calls "up" on 60% of flat ticks
true=2(up)       0.208        0.093        0.699
```

**Predicted distribution: 21 % down / 17 % flat / 62 % up**
**True distribution:      21 % down / 61 % flat / 18 % up**

The model is **massively over-confident on UP** (62 % vs 18 % truth).
Mean max_prob = 0.448 — model only weakly committed but systematically tilted up.

### Why the UP bias?

- Trained on syms {0,1,3,4} where average UP rate is ≈ 22 %.
- sym=2 microstructure features (high volume, tight spread, deep books) on training syms typically appear during **directional moves** (esp. accumulation → UP). On sym=2 those are *baseline*.
- Result: the model interprets sym=2's normal state as "buying pressure" and outputs UP probability too often.

### Train vs test distribution shift inside sym=2

| period | down | flat | up |
|---|---|---|---|
| sym=2 dates 0–95 (train) | 0.257 | 0.515 | 0.228 |
| sym=2 dates 96–119 (test) | 0.211 | 0.604 | 0.185 |

Test period is even MORE flat (60 %) than sym=2's own training period — so even per-sym calibration would not fully fix it. But cross-sym generalization is the dominant failure.

### Confidence by date (subset)

Worst PnL dates and their accuracy on directional predictions:
- 111-pm acc=0.17, 112-pm acc=0.17, 107-am acc=0.18, 96-pm acc=0.20 — concentrated in **PM session**, late-period dates.
- Max-prob distribution is symmetric correct vs wrong; the model is poorly calibrated rather than uncertain.

## 3. Hypotheses about sym=2's "identity"

1. **Likely a blue-chip / ETF** based on:
   - Highest CNY-volume per tick (3-7× others)
   - Tightest bid-ask spread
   - Lowest per-tick volatility
   - Deepest order books (totalbsize/totalasize floor 2-3× others)
2. Other syms include at least one **higher-spread, higher-vol single-name** (sym=3 has spread 7× sym=2's; sym=1/4 are most volatile).
3. The dominant "regime" sym=2 lives in (high-volume + tight-spread + low-vol-per-tick) is **rarely sampled** in the LOSO training set ⇒ the model has effectively no analogue.

## 4. Feature engineering recommendations

All proposals respect the 3 hard constraints (no sym, no date, no cross-tick state). They aim to **collapse sym=2 onto the same feature manifold as other syms**.

### F1. Volume / amount **z-score within window** (NOT global, NOT per-sym) — HIGH PRIORITY
- Formula: `amount_delta_z = (amount_delta - rolling_mean_100t) / (rolling_std_100t + ε)` over the same 100-tick window the predictor sees.
- Reason: amount_delta z=+7.49 between sym=2 and others — biggest gap by far. Rolling within-window normalization makes "high volume tick" mean the same thing on any stock.
- Cost: low (single rolling calc per feature per window).
- Expected: cuts much of the high-volume → "accumulation" false signal. Conservative estimate: lift LOSO held=2 by **+3 to +5 pnl** (since the bias is the central failure mode), and modest lift on other folds.

### F2. **Spread-normalized price moves** — HIGH PRIORITY
- Formula: `mid_move_normed = Δmidprice / max(abs(spread1+1), ε)` where `spread1+1 = ask1-bid1` is the actual spread.
- Reason: a 0.1 % move means very different things on a 3 bp spread (sym=2) vs a 24 bp spread (sym=3). Currently the model sees raw moves; spread-relative moves equalize across syms.
- Cost: low.
- Expected: improves cross-sym generalization on label_60 by reducing per-tick volatility heterogeneity. **+1 to +3 pnl** lift on LOSO held=2; small but consistent on other folds.

### F3. **Within-window relative volatility quantile** — MEDIUM PRIORITY
- Formula: `realized_vol_100t = std(Δmidprice over current window)`. Don't use the raw value directly — feed it as ratio: `current_tick_abs_move / realized_vol_100t`.
- Reason: directly measures "how big is this move relative to recent ambient vol on THIS stock right now". Forces sym-agnostic.
- Cost: low.
- Expected: the model can build a "z-scored move" ladder regardless of sym. Combined with F1+F2, this is the core sym-agnostic price-move trio. **+2 pnl** lift expected.

### F4. **Order-flow imbalance ratios** (NOT raw rates) — MEDIUM PRIORITY
- Formula:
  - `mb_over_lb = mb_intst / (lb_intst + mb_intst + ε)`
  - `cancel_ratio = (cb_intst + ca_intst) / (lb_intst + la_intst + cb_intst + ca_intst + ε)`
  - Same for sell side.
- Reason: raw intensities of 6 order types are 1.5× higher on sym=2 (z≈+1.2 each), but **ratios** between them carry the directional signal. Ratios are scale-invariant.
- Cost: low.
- Expected: makes order-flow patterns transferable. **+1 pnl** lift.

### F5. **Class-prior calibration head** (post-hoc) — LOW PRIORITY (trains-time fix)
- Add a *trained* output rebalancing layer that learns p(class | features, batch-prior). At inference compute prior from the 100-tick window's recent label-like proxy (e.g., realized 60-tick move count) and adjust class probabilities by `p_adjusted ∝ p_model × prior_window / prior_train`.
- This isn't strictly a feature, more a softmax recalibration. Useful if F1-F4 don't fully resolve the UP-bias on sym=2.
- Cost: medium (extra training step + a small head).
- Expected: corrects the residual prior mismatch left after sym-agnostic features. **+0.5 to +1.5 pnl**.

### Suggested shipping order
1. **F1 (volume z within window)** — biggest visible bias.
2. **F2 (spread-normalized moves)** — pairs with F1.
3. **F3 (relative vol quantile)** — completes the trio.
4. **F4 (order-flow ratios)** — easy add.
5. **F5** — only if 1–4 don't kill the UP bias.

All five are sym-agnostic by construction and respect the hard constraints (no sym, no date, no cross-call state). Implementations live in `cache_manifest`-style pre-computation per (sym, date, session) over the same 100-tick windows the Predictor uses.

## 5. Files

- `compute_stats.py` — per-sym aggregate stats (5 syms × 30+ dimensions).
- `analyze_oof.py` — T2 Scheme B LOSO held=2 prediction error breakdown.
- `make_figures.py` — 5 PNG plots.
- `figures/sym_midprice_traj.png` — mean midprice over time per sym.
- `figures/sym_label_dist.png` — label_60 distribution per sym.
- `figures/sym_volatility_traj.png` — per-day mean |Δmid| per sym (log-y).
- `figures/sym_amount_delta_hist.png` — log1p(amount) distribution per sym.
- `figures/sym2_oof_diagnosis.png` — OOF max_prob histogram + true/pred dist mismatch.
- `sym_stats_raw.json`, `sym_stats_table.csv`, `zscore_top20.json`, `oof_analysis.json`, `oof_feature_by_category.csv`.
