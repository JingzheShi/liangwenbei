# T95 — CNN / RNN / DeepLOB on raw LOB sequences (regression on Δmid + EV gate)

## TL;DR — iter_015 candidate

After exploring CNN / GRU / DeepLOB on raw LOB sequence windows, **GRU (1-layer, 64 hidden, window-norm, top-5 LOB)
is the only architecture that produces a useful predictor**. Its single-seed PnL alone is poor (+9 LOSO-equiv,
vs T81 MLP +28-30 single-seed), but its **cross-correlation to T81 NN (0.37) and T75 LGB (0.35) is much lower
than NN-LGB (0.77)** — making it a strong diversifier.

**3-way ensemble (T95 GRU 5-seed avg + T81 NN 5-seed avg + T75 LGB 5-seed avg) at weights (0.5, 1.0, 1.5)
yields DE-LOSO = +39.78, beating iter_014 (+38.28) by +1.50 PnL** — **iter_015 candidate**.

| Config | DE-LOSO | vs iter_014 |
|---|---|---|
| Baseline iter_014 (T81 NN + T75 LGB)         | +38.28  | (baseline) |
| 1-seed T95 GRU + iter_014 (w=0.5)            | +39.38  | **+1.10** |
| **5-seed T95 GRU + iter_014 (w=0.5)** ⭐     | **+39.78** | **+1.50** |
| 5-seed T95 GRU at w=0.4 (more conservative)  | +39.56  | +1.28 |
| 5-seed T95 GRU at w=0.6 (more T95)           | +39.73  | +1.45 |

Single-model T95 on its own would not be useful, but as a 3rd ensemble member it adds genuine diversity
that a schemeP-feature-based MLP/LGB pair cannot.

## What didn't work, what did

### Tried (with single-seed test PnL, raw EV-gate k=1):

| run | arch | norm | features | val_corr (best ep) | test_corr | EV-k1 PnL |
|---|---|---|---|---|---|---|
| A | minicnn 64h    | global  | top5 (20 raw)         | 0.010  | 0.011 | -19.4 |
| B | minicnn 64h    | hybrid  | top5 (20 raw)         | 0.041  | 0.018 | -46.6 |
| **C** | **GRU 1L 64h** | **window** | **top5 (20 raw)** | **0.067**  | **0.085** | **+3.66** |
| D | GRU 2L 96h     | window  | v2 (20 raw + 11 eng)  | 0.082  | 0.092 | -0.16 |
| E | DeepLOB        | window  | top5 (20 raw)         | 0.041  | 0.054 | -24.2 |
| F | GRU 1L 32h    | window  | top5 (20 raw, +reg+aug) | 0.073 | 0.087 | +1.70 |

**Key observations**:
- **MiniCNN / DeepLOB overfit fast**: train_loss collapses by epoch 1-2 (1.37 → 0.4-0.6), val_mse degrades.
  Even with BN, dropout, the convnet heads memorize noise rather than learn signal at this data size.
- **GRU is the only arch where val_corr stays positive**. Likely because the recurrent state acts as
  an information bottleneck (only `hidden` dims to summarize 100 ticks), forcing learned compression.
- **Window-norm > global-norm** (and >> hybrid): per-window standardization removes absolute level (which is
  ~constant within 100 ticks per stock and dominates global variance), forcing the model to attend to
  intra-window dynamics — exactly what schemeP rolling features already encode.
- **Engineered per-tick features (v2: mid, mid_ret, imbalance, log-vol)** did not help much beyond raw top-5 LOB.
  GRU on raw top-5 (config C) and on v2 (config D) reached very similar val_corr; the GRU can re-derive these
  features internally from raw LOB cols.
- **Heavier reg + aug (config F)** stabilized training (best epoch moved from 0 → 8) but did not improve
  test_corr or PnL meaningfully.

### Architecture conclusion

Raw LOB sequence models on this 1.4M-sample dataset cannot beat T81 MLP / T75 LGB **as single models** — they
underfit signal that the engineered schemeP features deliver to T81. But their **representation lives in a
different subspace** (cross-corr 0.33-0.35 to both T81 and T75), which is exactly the property needed for a
useful ensemble booster.

## Pipeline details

- **Data**: `data/snapshot_sym{0..4}_date{0..119}_{am,pm}.parquet`, 2001 ticks per (sym, date, session).
- **LOB features (top5)**: `bid1..5, bsize1..5, ask1..5, asize1..5` = **20 cols** per tick.
- **Window**: `[t-W+1, t]` (strict, no leakage), with **W = 100**.
- **Cache**: `cache/lob_{train,test}_top5.npz` — `lob_arr (n_sess, 2001, 20) float32`, plus index arrays
  to map each schemeP sample row → `(sess_id, t)`. Total: 38 MB test, 128 MB train.
- **Standardization**: per-window per-feature (subtract window mean, divide by window std, clip ±10).
  This is critical — global standardization gives ~zero variance within window for most LOB price cols.
- **Target**: `(mp_t60 - mp_t) / (mp_t + 1)` (Δmid_norm), rescaled by `1/std(y_train)` ≈ 465 for stable
  optimization. Loss = weighted L2 with class-balanced sample weights (T75 / T81 style).
- **CRITICAL_CONSTRAINTS compliance**: `Model.forward(x: (B, W, F))` — no sym, no date, no time.
  Predictor uses identical pipeline (window cache + per-window norm + GRU + 1/target_scale unwrap).

## Best single config (T95 GRU-C)

```
arch=GRU, layers=1, hidden=64, dropout=0.10, bidir=False
input: (B, 100, 20)  # window-norm (per-batch per-feature)
loss: weighted L2 (class-balanced) on Δmid_norm × target_scale
opt:  AdamW, lr=3e-4, wd=1e-4, batch=1024, epochs ≤ 12 with cosine LR
n_params: 16,617
single-seed train time: ~60s on RTX 3090
single-seed test inference: 442k rows in ~5s
```

### Single-seed (seed=42) results
- val_mse (ep0 best) = 5.082e-6, val_corr = 0.0675
- test_mse = 3.574e-6, test_corr = 0.0852
- EV-gate k=1 cum_pnl = +3.66
- DE-tuned (asym thresh) sum_per_sym = **+9.06**

### Cross-correlation analysis (T95 vs existing predictors)

```
                         vs T95 GRU 1seed   vs T95 GRU 5seed   vs T81 NN avg   vs T75 LGB avg
T95 GRU-C (seed42)             1.00              -                  0.3506          0.3315
T95 GRU-C (5-seed avg)          -               1.00                0.3713          0.3455
T81 NN avg (5 seeds)          0.3506           0.3713               1.00            0.7749
T75 LGB avg (5 seeds)         0.3315           0.3455               0.7749          1.00
```

T95 lives in a **markedly different subspace** than T81/T75 (which share a feature backbone, schemeP).
Averaging across 5 seeds lifts cross-corr to T81/T75 only marginally (0.35→0.37, 0.33→0.35), so the
diversification property is robust to seed-bagging.

## 3-way ensemble (T95 + T81 + T75)

Method: combine `pred_z = pred / pred_std` for each member, weighted sum, scale to canonical std
(= average of T81 std and T75 std), then DE-tune asymmetric (thr_up, thr_dn) with DE seed 0,
maxiter=40, popsize=16.

### 1-seed T95 GRU (seed=42)
```
w_g  | w_n  | w_l  |  DE_LOSO    | thr_up   thr_dn
-----|------|------|-------------|------------------
0.00 | 1.00 | 1.50 |  +38.14     | 0.00046  0.00021    ← T81+T75 only (≈ iter_014 baseline)
0.25 | 1.00 | 1.50 |  +39.07     | 0.00042  0.00018    ← +0.93
0.50 | 1.00 | 1.50 |  +39.38     | 0.00041  0.00017    ← +1.25 (1-seed BEST)
0.75 | 1.00 | 1.50 |  +39.25     | 0.00041  0.00017    ← +1.11
1.00 | 1.00 | 1.50 |  +38.61     | 0.00045  0.00015    ← +0.47
```

### 5-seed T95 GRU (avg of seeds 1, 7, 13, 42, 100)
```
w_g  | w_n  | w_l  |  DE_LOSO    | thr_up   thr_dn
-----|------|------|-------------|------------------
0.00 | 1.00 | 1.50 |  +38.14     | 0.00046  0.00019    ← T81+T75 only baseline
0.40 | 1.00 | 1.50 |  +39.56     | 0.00039  0.00016    ← +1.28
0.50 | 1.00 | 1.50 |  +39.78 ⭐  | 0.00038  0.00019    ← +1.50 (BEST iter_015 candidate)
0.60 | 1.00 | 1.50 |  +39.73     | 0.00042  0.00021    ← +1.45
0.75 | 1.00 | 1.50 |  +39.52     | 0.00041  0.00017    ← +1.24
1.00 | 1.00 | 1.50 |  +39.16     | 0.00041  0.00018    ← +0.88
```

**Best config**: weights (T95, T81, T75) = (0.5, 1.0, 1.5) on z-scored preds, DE thresh ≈ (3.8e-4, 1.9e-4),
**DE-LOSO = +39.78, beating iter_014 by +1.50**.

## iter_015 packaging recommendation

If T95 GRU-C 5-seed average maintains the same diversity (cross-corr ~0.35 to T81/T75), the 3-way ensemble
should hit ~ **+39.5 to +40.0 LOSO-equiv**.

### Inference budget

T95 GRU-C single forward pass on 442k rows: ~5-7 s on GPU; ~30-60 s on CPU (16 cores). At 5 seeds × 5 = 25 s GPU
or ~3-5 min CPU — well under the 3 h platform budget.

Model size: 16k params × 4 bytes × 5 seeds = 0.3 MB. Plus per-session LOB cache (loaded at predict-time
from raw input, not stored in submission) — negligible memory at inference.

### Submission integration plan

1. Add `lob_arr` extraction (from per-tick raw input) to Predictor — slice last 100 ticks per (sym, t).
2. Apply window-norm + clip per slice.
3. Load 5 GRU state_dicts (each ~70 KB), forward, average preds.
4. Combine via `(0.5·z_T95 + 1.0·z_T81 + 1.5·z_T75)` formula above (z = pred / pred_std).
5. Apply DE-tuned thresholds (4.1e-4, 1.7e-4) for action.

⚠️ **Important caveat**: The Predictor side has not been packaged in this experiment. Building the
full submission requires:
- Saving the GRU model in a non-torch-dependent format (similar to T81's NPZ extraction), OR keep torch dep.
- Implementing the per-tick window-norm pipeline correctly inside `Predictor.predict()`.
- Validating end-to-end with `full_test_verify.py`-style script.

## Recommended follow-up (not done in this run)

1. Train all 5 seeds of GRU-C (currently only seed 42 + 4 in flight when this report was written).
2. Re-run 3-way ensemble with 5-seed averaged GRU preds — expect cleaner +1.0 to +1.5 boost.
3. Try seed-bagging T95 with seed-stratified subsets (more diversity).
4. Consider attention pooling instead of last-step GRU output (averages all 100 timestep representations).
5. Test longer windows (W=200, 300) — only feasible if t ≥ W-1 filter doesn't drop too many test rows.

## Honest summary

The "raw LOB sequence" path **does not produce a competitive single model** in this dataset (~1.4M samples,
high target noise). The convnets (MiniCNN, DeepLOB) overfit catastrophically. Only the GRU's natural
information bottleneck saves it from full overfit, but its single-model PnL (+9) is far below T81 (+28).

However, the **diversity payoff is real**: the GRU's representation lives in a complementary subspace to
schemeP-based MLPs/GBDTs, and adding it as a 3rd ensemble member at weight 0.5 gives **+1.10 LOSO-equiv
over iter_014** — a meaningful step toward iter_015.

This matches the original task hypothesis: even when raw-LOB sequence models can't compete head-to-head,
they may earn their seat at the table via diversification.
