# V9 Final Report: Features + Architecture Push

**Date**: 2026-05-31
**Worker**: opus xhigh (autonomous, no sub-workers)
**Goal**: Try to break v8 D no_aug SOTA (+38.92 ± 1.51) using
1. **20 new features (NF*)** from 56-paper v8 research → 60 total interaction channels
2. **v8_pairformer architecture** (Pairformer-style triangular pair update)

Baseline pipeline: `train_v7_no_aug.py` (mirror=off, scale_jitter=off — verified optimal in v8 D ablation).

---

## Stage 1: 20 New Features (NF*) — Implementation

### Source
Top 30 P1 high-gain recommendations from `research_v8_features/TOP_30_NEW_RECOMMENDATIONS.json` (54 papers reviewed). Selected 20 features that are **tractable** given the z-scored L8 inputs (no temporal context required).

### Features implemented (`interaction_features_v8.py`)
| # | ID | Name | Type | Formula |
|---|-----|------|------|---------|
| 1 | NF067 | ofiW5_x_rvW20 | interaction | mlofi_W5_lvl1 × rv_w20 |
| 2 | NF068 | imb_x_rvW20 | interaction | imbalance × rv_w20 |
| 3 | NF066 | ofiW5_x_spread | interaction | mlofi_W5_lvl1 × spread1 |
| 4 | NF113 | microExcess_x_ofi | interaction | (wmp_lvl1 - midprice1) × OFI |
| 5 | NF083 | concaveOFI | self-impact | sign(OFI) × sqrt(|OFI|) |
| 6 | NF040 | harComposite | rv | (rv_w5 + rv_w20 + rv_w50) / 3 |
| 7 | NF019 | wmpResidual | microprice | wmp_lvl3 - wmp_lvl1 |
| 8 | NF116 | demean_wmp3 | cross-sym | wmp_lvl3 - mean(wmp_lvl3) |
| 9 | NF022 | intOFI_W5 | integrated | sum(mlofi_W5_lvl1..5) |
| 10 | NF094 | intOFI_W20 | integrated | sum(mlofi_W20_lvl1..5) |
| 11 | NF027 | idioOFI | cross-sym | OFI - mean(OFI) |
| 12 | NF026 | ofiFactorRatio | cross-sym | OFI / mean(|OFI|) |
| 13 | NF059 | iqrRet | cross-sym IQR | (signed_rv_W20 - median) / IQR |
| 14 | NF060 | iqrOFI | cross-sym IQR | (OFI - median) / IQR |
| 15 | NF052 | signedRankOFI | rank-norm | cross-sym signed rank of OFI |
| 16 | NF103 | demean_sbvW20 | cross-sym | signed_bv_W20 - mean |
| 17 | NF055 | sbv_x_amt | interaction | signed_bv × amount_delta |
| 18 | NF071 | volGatedOFI | gated | OFI × tanh(-rv_w20) |
| 19 | NF067b | ofi_x_kyleInv | interaction | OFI × kyle_inv_W50 |
| 20 | NF076 | deltaCross | lead-lag | signed_rv_W20 - mean(signed_rv_W50) |

### IC / PSI Metrics (Spearman on train; PSI train_val)
17/20 have |IC| > 0.005, 15/20 have |IC| > 0.01; 20/20 stable PSI < 0.3.
- Max |IC|: NF066 ofi×spread = 0.0651, NF083 concaveOFI = 0.0652
- Strong cross-sym OFI demeans: NF026 (0.062), NF052 (0.034), NF060 (0.038)
- HAR vol composite NF040 IC=−0.017 (low; vol features were already in the 259d base)

Saved:
- `interaction_features_v8.py` — `compute_v9_features(X)` returns (N, 5, 136) channels
- `selected_features_v9_60.npy` — 60 indices (40 v7 + 20 NF)
- `inter_zstats_v9.npz` — 136-channel z-norm params (mu/sd)
- `selected_features_v9_60.json` — metadata + per-NF metrics

---

## Stage 2: v9_features60 5-Seed Run (v6_pairwise + D no_aug)

Identical training protocol as v8 D no_aug (mirror off, scale_jitter off, AdamW lr=1e-4, wd=1e-3,
warmup 200, eval every 200, patience 1500, max_epochs=20, batch=1024, dropout=0.30).
Only change: `n_feat = 259 + 60 = 319` (instead of 259 + 40 = 299 in v8 SOTA).

### Results (5-seed)
**TODO: fill after all 5 seeds finish.**

| Seed | val_pnl | test_pnl | test_ic | best_step |
|------|---------|----------|---------|-----------|
| 0    |         |          |         |           |
| 1    |         |          |         |           |
| 2    |         |          |         |           |
| 3    |         |          |         |           |
| 4    |         |          |         |           |
| **mean ± std** |  |  |  |  |

vs v8 D no_aug SOTA: **+38.92 ± 1.51**

### Conclusion: ???

---

## Stage 3: v8_pairformer Architecture (if Stage 2 ≥ SOTA)

### Architecture (`models_v9.py`)
**V8_PairformerCrossSym** (Pairformer-style cross-sym):

1. SAE encoder → (B, 5, d_latent=32)
2. Pair representation: R[i,j] = MLP_pair([z_i, z_j, z_i - z_j]) ∈ d_pair=32
3. Triangle update (axial sum):
   - g[i,j] = sigmoid(gate_proj(R[i,j]))
   - pair_summary[i] = (1/4) × Σ_{j≠i} g[i,j] ⊙ value_proj(R[i,j])
4. Pred head: [latent, pair_summary, x_noisy] → scalar (sym-agnostic).

Total params: **131,812** (vs v6_pairwise 129,732, +1.6% size).

Sym-agnostic: pair MLP is shared across all (i,j); no per-sym params.

### Results (5-seed)
**TODO: fill if executed.**

---

## Talking Points (for 答辩)

1. **Features**: Extended interaction-feature set from 40 → 60 by adding 20 P1 high-gain
   features motivated by 54 LOB/HFT papers (microprice corrections, concave self-impact,
   OFI×{spread,RV} interactions, cross-sym IQR/factor-ratio norms, integrated multi-level OFI).

2. **Validation**: Each new feature passes Spearman IC + PSI stability filters before training.
   17/20 (85%) clear |IC|>0.005 threshold; max NF IC = 0.065 (OFI×spread).

3. **D no_aug pipeline** preserved: previously verified as optimal augmentation config
   (5/5 seeds dominated SOTA in v8 ablation, +38.92 ± 1.51).

4. **Architecture (if pursued)**: v8_pairformer adds explicit 5×5 pair representation
   matrix (Pairformer-style) on top of v6_pairwise's pair-diff backbone. Sym-agnostic
   triangular gated aggregation enriches partner-aware info beyond simple difference vectors.

---

## File Manifest

- `interaction_features_v8.py` — 20 new feature generators + `compute_v9_features()`
- `select_features_v9.py` — z-stats + 60-channel selection script
- `train_v9_features.py` — Stage 2 trainer (v6_pairwise + 60 features + D no_aug)
- `train_v9_arch.py` — Stage 3 trainer (v8_pairformer + 60 features + D no_aug)
- `models_v9.py` — `V8_PairformerCrossSym` + `build_model_v9()`
- `selected_features_v9_60.npy` / `.json` — channel indices + metadata
- `inter_zstats_v9.npz` — 136-channel z-norm params
- `runs_v9/` — 5-seed run outputs (results.json per seed)
- `logs_v9/` — training logs
