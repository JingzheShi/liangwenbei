# T32 — DeepLOB single-head h_60 + aug_a, NN+LGB ensemble (negative result)

## TL;DR
- 5-fold LOSO single-seed NN h_60: **OOF sum cum_pnl = −16.97** (LOSO sum at T=0.45,δ=0.10).
- Combined with iter_005b LGB 5-seed aug_a (mean-prob ensemble at α∈[0,1]):
  best is **α=0.00 (LGB-only) at +11.46** — same as iter_005b.
  Even α=0.05 already drops the ensemble to +10.71.
- **Decision: do NOT build iter_007. Single-seed DeepLOB h_60 actively hurts the ensemble.**
- Fold 0 NN test +0.32 was within reach of T19 single-fold +3.07, but folds 1/2/4 are deeply
  negative (−10.81, −1.27, −5.29). The OOD val/test gap for NN is much wider than for LGB.

## Setup
- Backbone: DeepLOB CNN (T19 reuse), GroupNorm replacing BatchNorm.
- Single h_60 head (3-class) — drops T19 multi-task heads that overfit on h_10.
- Regularization:
  - Dropout 0.4 (conv) / 0.5 (FC)
  - AdamW weight_decay = 5e-3 (5× T19's 1e-3)
  - Label smoothing ε = 0.1
  - lr = 5e-4 (½ of T19), warmup 100 steps + cosine decay over `epochs * steps_per_epoch`
- aug_a (NN-side): per-sample, per-feature multiplicative scale ~ U[0.8, 1.2]
  applied to the z-scored 100×154 input each train batch. (NB: aug applied
  in z-score space, not raw — a simplification vs. LGB's raw-space aug; this
  saves recomputing log1p+normalize per batch.)
- Early-stop on val h_60 thresholded cum_pnl (T=0.45, δ=0.10), patience=2,
  swa_topk=1 (single best epoch's state, no averaging — first run with topk=3
  averaged a good epoch with a bad one and made it worse).
- 8 epochs max, ~4 min/epoch on RTX 3090; folds early-stopped at 3–5 epochs.

## Per-fold results (h_60)

| held | best epoch | epochs run | val_thr (T=0.45, δ=0.10) | test_argmax | test_thr | LGB iter_005b test_thr |
|------|------------|------------|---------------------------|-------------|----------|------------------------|
| 0    | 1 | 3 | −1.04 | +0.16 | **+0.32** | +1.30 |
| 1    | 3 | 5 | +2.97 | −20.95 | **−10.81** | +2.16 |
| 2    | 2 | 4 | +0.59 | −13.97 | **−1.27** | −0.28 |
| 3    | 1 | 3 | −1.22 | −0.33 | **+0.08** | +0.10 |
| 4    | 1 | 3 | +0.15 | −11.35 | **−5.29** | +8.18 |
| **sum** | – | – | **+1.45** | **−46.45** | **−16.97** | **+11.46** |

Observations:
- Val and test diverge dramatically on folds 1, 2, 4 (val ≥ 0 but test deeply negative).
  Single-seed NN cannot generalize to held-out sym at h_60 horizon as reliably as the
  5-seed LGB aug_a ensemble.
- Argmax-vs-threshold ratio: most folds have test_argmax_pnl ≪ 0, but the
  threshold (T=0.45, δ=0.10) filters down to a very small active fraction
  (NN: 0.3% of rows have max(p₀,p₂)≥0.45 vs. LGB: 3.8%) which dampens losses
  but also dampens any potential gains.
- NN's pred distribution: argmax is 91% class 1 (neutral), only 0.3% class 2 (up),
  vs. true label distribution (held-out fold 0): 9.6% / 81.7% / 8.7%. Heavy
  bias toward neutral — under-confident on the up direction.

## NN + LightGBM 5-seed aug_a ensemble OOF sweep

For each fold K, ensemble probs = α·NN_probs + (1−α)·mean(LGB_seed{42,1,7,13,100}).
Sweep α ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00}
× T_GRID {0.35..0.70} × D_GRID {0.0..0.25}.

### Best (T, δ) per α value (5-fold OOF concat)

| α (NN weight) | best T | best δ | sum_cum_pnl | n_pos_folds | sum_n_active |
|---------------|--------|--------|-------------|-------------|--------------|
| 0.00 (LGB-only) | 0.45 | 0.10 | **+11.46** | 4/5 | 100,256 |
| 0.05 | 0.45 | 0.00 | +10.71 | 4/5 | 95,759 |
| 0.10 | 0.45 | 0.00 | +10.42 | 5/5 | 90,992 |
| 0.15 | 0.45 | 0.05 | +9.68 | 5/5 | 86,172 |
| 0.20 | 0.45 | 0.00 | +9.18 | 5/5 | 81,424 |
| 0.30 | 0.45 | 0.00 | +7.85 | 5/5 | 72,708 |
| 0.40 | 0.45 | 0.05 | +5.33 | 4/5 | 64,691 |
| 0.50 | 0.45 | 0.15 | +2.16 | 4/5 | 57,394 |
| 0.60 | 0.55 | 0.25 | −0.03 | 1/5 | 3,837 |
| 0.70 | 0.65 | 0.25 | −0.06 | 1/5 | 400 |
| 0.80 | 0.70 | 0.00 | −0.14 | 1/5 | 171 |
| 1.00 (NN-only) | 0.70 | 0.00 | −0.17 | 1/5 | 609 |

Monotone: every step away from α=0 strictly hurts the ensemble.
Adding NN even at 5% weight reduces sum_cum_pnl by ~0.75 and removes ~4500 active
positions. The NN's class-1 bias dilutes LGB's directional confidence enough that
fewer trades cross the (T, δ) threshold, and the trades that remain are no better
on average.

## iter_007 decision

Threshold to beat: iter_005b OOF best = +11.46.
Best ensemble found: +11.46 at α=0 (= iter_005b).

**No iter_007 build.** Negative result documented; iter_005b remains the best
submission so far. (No platform call wasted.)

## Why did this fail?

- **NN single-seed has too high OOD variance for h_60.** LGB aug_a achieves
  per-fold pnl of [+1.30, +2.16, −0.28, +0.10, +8.18] (std ≈ 3.07), and 5-seed
  averaging shrinks the noise. NN single-seed achieves [+0.32, −10.81, −1.27,
  +0.08, −5.29] (std ≈ 4.50) — both higher noise floor AND systematically
  negative mean.
- **T19's fold-0 +3.07 was multi-task and not robust.** The T19 single-fold result
  did not generalize: making the architecture single-head and harder-regularized
  did not lift other folds.
- **aug_a in z-score space is not equivalent to LGB raw-space aug.** A more
  faithful NN aug would (a) keep raw features in the dataset, (b) randomly
  rescale before log1p + z-score. We took the simpler path; possible the
  effective regularization was weaker than intended.
- **Patience+SWA tuning did not save it.** patience=1 with topk=3 averaging
  (initial run) blended the best epoch with a strictly worse one and dropped
  test pnl. Switching to patience=2 + topk=1 (single-best state) recovered fold 0
  by ~0.0 → +0.32 but did not change the macro picture.

## Next steps (if pursued further)
1. **NN aug_a in raw space** — recompute log1p+z-score per batch after random
   scale to match LGB's aug semantics.
2. **Multi-seed NN ensemble** — 3-5 NN seeds per fold to reduce variance, then
   re-run combine. Cost: 3-5× training time.
3. **Add a light auxiliary h_40 / h_20 head** — shallow heads (linear) on shared
   backbone may regularize without dominating; T19 5-head may have been the right
   structural choice after all.
4. **Try a smaller NN** — DeepLOB has ~1M params on 1.2M training samples;
   might be over-parameterized for the held-out generalization regime. A
   smaller MLP on hand-crafted T3 features could be a stronger baseline.

## Files
- `model.py` — DeepLOB_H60 (single-head)
- `train_loso.py` — per-fold trainer with aug_a in z-score space
- `sweep_utils.py` — shared threshold helpers
- `combine_with_lgbm.py` — alpha sweep + threshold sweep on combined OOF
- `build_iter_007.py` — submission packager (UNUSED — no iter_007 built)
- `loso_pred_nn_h60_held{K}.parquet` — per-fold OOF predictions (K=0..4)
- `model_h60_held{K}.pt` — per-fold SWA model state (~10 MB each)
- `combine_results.json`, `combine_sweep.csv` — sweep outputs
- `fold{K}.out`, `fold{K}_history.json`, `fold{K}_train.log` — per-fold training logs
- WandB project: `liangwenbei`, runs `T32-NN-h60-fold{K}-...`
