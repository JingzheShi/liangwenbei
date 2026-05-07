# Pipeline Leakage Audit Report
**Project**: 良文杯 LOB Prediction  
**Audit Date**: 2026-05-07  
**Auditor**: Claude Code (AUDIT-LEAKAGE worker)  
**Scope**: iter_006_aug_a_5seed_h60_asymthresh — current best submission

---

## 1. Pipeline DAG

```
Raw Data
  data/snapshot_sym{S}_date{D}_{sess}.parquet
  163 cols × 2001 rows/session
  (date, sym, time, 10-level LOB OHLCV, midprice, 5 labels)
         |
         v [T3_features_v1/build_cache.py]
  compute_all(df, session) on FULL session
  → concat raw (163) + T3 (72) = 235 cols
  data/features_v1/snapshot_*.parquet
         |
         v [T5b_features_multihorizon/build_features_C.py]
  Select 226 cols (154 raw + 72 T3), log1p(amount_delta)
  Slice valid prediction points t ∈ [99, 1940]
  → experiments/T5b_features_multihorizon/cache/schemeC_{train,val,test}.npz
    X:(N,226)  y5/10/20/40/60:(N,)  mp_t/mp_th  sym/date/sess/t
         |
         v [T26/build_aug.py  ←  T27/train_5seed_aug_a.py + train_final_aug_a.py]
  Per LOSO fold / final:
    drop last 3 cols (time_*) → 223d features
    compute feat_mean/feat_std from CURRENT train fold only
    aug_a: X *= U[0.8,1.2] per-feature per-sample (train only)
    LightGBM multiclass, 5 seeds × {5 folds LOSO + 1 final}
  → experiments/T27_iter005/final_model_h60_aug_a_seed{1,7,13,42,100}.txt
         |
         v [T27/build_iter_005b.py  →  T30 DE threshold optimization]
  Pack submission:
    Predictor.py (iter_006 asymmetric threshold variant)
    compute.py   (T3 feature computation)
    config.json  (154-feature list, no sym/date/time)
    thresholds.json (h40: T=0.5,d=0; h60 asymmetric T_up/T_dn/d_up/d_dn from OOF DE)
    model_h40.txt (from T5b iter_002)
    model_h60_seed{1,7,13,42,100}.txt
         |
         v [Platform evaluation]
  Platform calls predict(batch_dfs):
    each df: 100 rows × 154 cols (config["feature"])
    date zeroed, sym 0-4 (possibly OOD), test order shuffled
  Predictor._compute_window_features(df):
    compute_mlofi(df)        → 30 features (100-tick window)
    compute_wmp(df)          → 11 features
    compute_rv(wmp_lvl1)     → 4 features
    compute_ewma_intst(df)   → 24 features   (NO time features)
    raw df.iloc[-1] 154 features + log1p(amount_delta)
    concat → 223d input to 5 LightGBM boosters (h=60) + 1 (h=40)
    ensemble h60 probs → asymmetric threshold → {0,1,2}
```

---

## 2. Leakage Risk Table (per node)

| Node | Leakage Type | Severity | Evidence | Status |
|------|-------------|----------|---------- |--------|
| **Feature: date** | Future info / forbidden field | CRITICAL | `config["feature"]` has no `date`; local_evaluator zeroes date before passing to Predictor; T27 asserts `forbidden = {"date","sym","time"}` not in feat_names | ✅ CLEAN |
| **Feature: sym** | Sym-specific patterns, OOD failure | CRITICAL | sym not in config["feature"]; Predictor never indexes by sym; runtime test with OOD input passes | ✅ CLEAN |
| **Feature: time_*** | Time-of-day = partial session info; not available (date=0 on platform) | HIGH | `_t3_feat_cols` filters `not c.startswith("time_")`; `N_DROP_TAIL=3` drops tail in training; feature count verified: 69 not 72 T3 cols | ✅ CLEAN |
| **T3 MLOFI (rolling W≤60)** | Future-looking rolling sum | HIGH | Rolling W = {5,20,60} ticks; prediction point uses last row of 100-tick window → lookback fully within window (verified analytically and by alignment test) | ✅ CLEAN |
| **T3 RV (rolling W≤50)** | Future-looking realized vol | HIGH | RV W={5,10,20,50}; lookback ≤50 < 100-tick window → all data causal | ✅ CLEAN |
| **T3 EWMA (alpha={0.05,0.1,0.3,0.5})** | Train uses full session; inference uses 100-tick window | LOW | Asymmetry = (1-α)^99 × initial_diff. Measured: alpha=0.05 max 1.23% of signal std; alpha=0.1 max 0.005%; alpha≥0.3 zero. Negligible for LightGBM. | ⚠️ NEGLIGIBLE SKEW |
| **aug_a feat_stats (mean, std)** | Stats contaminated by val/test | MEDIUM | `compute_feat_stats(X_tr_o)` called on train fold only; `X_tr_o` is masked by `m_tr = sym != held` and uses only train dates | ✅ CLEAN |
| **aug_a augmentation** | Applied to val/test | HIGH | Augmentation applied only to `X_tr_o`, never to `X_va`/`X_te`; inference path has no aug | ✅ CLEAN |
| **log1p(amount_delta)** | Train/inference inconsistency | HIGH | Applied in build_features_C.py line 77 (training) and Predictor.py line 108 (inference); identical formula `sign(v)*log1p(|v|)` | ✅ CLEAN |
| **LOSO fold split** | Held sym data in train | CRITICAL | `m_tr = train_full["sym"] != held`; `m_va = val_full["sym"] != held`; held sym only in test fold | ✅ CLEAN |
| **Feature order (train vs inference)** | Silent dimension mismatch | CRITICAL | Automated check: `inference_feats == train_feats` for all 223 features (EXACT MATCH confirmed by Python script) | ✅ CLEAN |
| **Class-balanced weights** | Val/test label distribution exposed | MEDIUM | `class_balanced_weight(y_merged)` uses ONLY train+aug labels; val/test never touch weight computation | ✅ CLEAN |
| **Threshold optimization (T30 DE)** | Threshold tuned on test distribution | LOW | Thresholds optimized on LOSO OOF predictions (held-sym test, dates 96..119). LOSO models didn't see held sym during training. Standard OOF practice. | ⚠️ MILD OPTIMISM |
| **Final model early stopping** | Early stopping on "test" period | LOW | `train_final_aug_a.py`: trained on dates 0..95, early-stopped on dates 96..119. Same period used for threshold OOF tuning → slightly optimistic iteration count | ⚠️ MILD OPTIMISM |
| **Cross-sym normalization** | Per-sym stats leak | CRITICAL | No normalization in SchemeC; aug stats are global-per-fold (not per-sym) | ✅ CLEAN |
| **Stateful Predictor** | Cross-call hidden state | CRITICAL | Runtime test: `predict([a,b])[0] == predict([b,a])[1]` (True). No `self.state`, no LSTM, no buffer | ✅ CLEAN |
| **Submission ZIP contents** | Wrong model files | CRITICAL | Verified: zip contains `Predictor.py`, `compute.py`, `config.json`, `thresholds.json`, `model_h40.txt`, `model_h60_seed{1,7,13,42,100}.txt`; inactive files h5/10/20 present but NOT loaded (Predictor skips `active=false` horizons) | ✅ CLEAN |

---

## 3. Detailed Answers to Audit Questions

### Q1: Feature construction causal?

**CAUSAL. No future-looking.**

All T3 rolling-window features (MLOFI, WMP, RV) have lookback ≤ 60 ticks. Since the prediction window is 100 ticks, all required prior data is strictly within the window. `shift(1)` at row 0 of the window produces NaN for e_k, but only `iloc[-1]` (row 99) is used, where all rolling sums are valid and causal.

EWMA uses `ewm(alpha, adjust=False)`, which starts from the first value of the series. The discrepancy between full-session initialization (training) and 100-tick window initialization (inference) was **empirically measured**:

| α | Max |diff| | Relative to signal std |
|---|----------|------------------------|
| 0.05 | 0.00276 | **1.23%** |
| 0.10 | 0.0000113 | 0.005% |
| 0.30 | ~0 | ~0% |
| 0.50 | ~0 | ~0% |

**Not a critical issue**: LightGBM is robust to 1% noise on a single feature group.

### Q2: iter006 223-feature list

Verified: `schemeC_223d_feat_names.txt` (223 lines) = config.json features (154) + feature_v1_columns() minus time_* (69). Feature names at every position match exactly between training and inference. The 3 dropped "tail" features are `time_minutes_since_session_start`, `time_session_progress`, `time_is_pm`.

### Q3: aug_a augmentation

- **Range**: U[0.8, 1.2] per-(sample, feature), as stated in build_aug.py
- **RNG seed**: `np.random.default_rng(seed * 7919 + held * 17 + 1)` — deterministic, fold-specific, no seed reuse
- **Train-only**: `X_va` and `X_te` are NEVER augmented (code explicitly separates `X_tr_o` augmentation from val/test)
- **Sample-wise**: independent per sample, no cross-sym contamination possible
- **aug_ratio=1.0**: augmented copies = original copies, concatenated → 2× training size
- **Final model**: `seed_rng = np.random.default_rng(s * 7919 + 1)` — same formula, fold=0

### Q4: Normalization leakage

**No normalization in final models.** SchemeC features are RAW (plus log1p for amount_delta). T7 window z-score (iter_001f) was an abandoned experiment — not in the submission.

For the augmentation stats (`feat_mean`, `feat_std`): computed from `X_tr_o` (train fold only, no val/test). Used only for aug_b/aug_c/aug_abc variants — `aug_a` does not use them at all (it only needs `rng.uniform(0.8, 1.2, size=X.shape)`).

### Q5: CRITICAL_CONSTRAINTS compliance

| Constraint | Status | Evidence |
|-----------|--------|----------|
| date=0 at eval | ✅ COMPLIANT | config["feature"] has no "date"; local_evaluator zeroes date; T27 assertion `forbidden = {"date","sym","time"}` passes |
| Order shuffled | ✅ COMPLIANT | Runtime test confirms statefulness: `predict([a,b])[0] == predict([b,a])[1]` = True; no self.buffer |
| sym-agnostic | ✅ COMPLIANT | sym not in features; OOD sym=99 test passes; no per-sym normalization |

### Q6: Train/inference skew

**NO SIGNIFICANT SKEW.** Automated check confirms exact 223-feature alignment. The only non-zero discrepancy is EWMA initialization (max 1.23% for α=0.05), which is negligible for tree-based models.

Specific consistency checks:
- `amount_delta` log1p: identical formula in training cache and Predictor inference
- T3 feature selection: both use `[c for c in feature_v1_columns() if not c.startswith("time_")]`
- Raw features: 154 columns from config.json, in parquet-native order, matching `get_default_feature_cols()`

### Q7: Label-aware features?

**NO LABEL LEAKAGE.** No feature references label values. `class_balanced_weight()` uses training labels only (y_merged = y_orig + y_aug, both from the training fold). No quantile/decile normalization based on label distribution.

### Q8: Cross-sym leakage?

**NO CROSS-SYM LEAKAGE.** All T3 features are computed per-sample from the 100-tick window. The `compute_feat_stats(X_tr_o)` for aug purposes uses the combined training fold (all non-held syms' train data) — this is global-per-fold, not per-sym. No per-sym normalization. No sym embedding. LightGBM never sees `sym` as a feature column.

---

## 4. Critical Bugs

**NONE FOUND.**

Zero bugs that would cause platform score collapse.

---

## 5. Suspect Features

**NONE.**

All 223 features are causal, sym-agnostic, and computed consistently between training and inference.

---

## 6. Minor Concerns (Non-Critical)

### M1: EWMA initialization skew (LOW)
- **What**: Training EWMA uses full-session history; inference uses 100-tick window
- **Magnitude**: max 1.23% for α=0.05; essentially zero for faster alphas
- **Impact**: Tree-based models are robust to this-level noise
- **Fix if desired**: Precompute EWMA from tick t-199 (200-tick warm-up) and only keep last 100 ticks

### M2: Test set used for both early stopping and threshold tuning (LOW)
- **What**: Final models early-stop on dates 96..119; threshold OOF optimization targets the same period via LOSO
- **Impact**: Slight upward bias in LOSO PnL metrics; platform performance may be modestly lower than LOSO suggests
- **Note**: Standard competition practice; not a fundamental flaw

### M3: Inactive model files (COSMETIC)
- `model_h5.txt`, `model_h10.txt`, `model_h20.txt` in submission but never loaded (active=false in thresholds.json)
- **Impact**: Zero functional impact; ~3× unnecessary model load skipped at init

---

## 7. Recommendations (Quick Wins)

None are critical. For defensive engineering if a new submission is built:

1. **Assert no forbidden features** (already done in T27 train script): `assert not ({"date","sym","time"} & set(feat_names))` — keep this in all future training scripts
2. **Add OOD sym test** to build scripts (already exists in build_iter_005b.py sanity test)
3. **Remove inactive model files** from submission to reduce package size and loading time

---

## 8. Audit Methodology

1. Read all source files in `src/data/`, `src/submit/`, `submission/iter_006_aug_a_5seed_h60_asymthresh/`
2. Traced training pipeline: `T3_features_v1/build_cache.py` → `T5b_features_multihorizon/build_features_C.py` → `T26/build_aug.py` → `T27/train_5seed_aug_a.py` + `train_final_aug_a.py`
3. Ran Python verification scripts:
   - Feature order alignment: 223-feature exact match ✓
   - EWMA skew quantification: max 1.23% for α=0.05 ✓
   - Statelessness: predict([a,b])[0] == predict([b,a])[1] ✓
   - Feature vector size: 223 (154+69) ✓
4. Verified submission ZIP contents against known-good file list ✓

---

**Verdict: PIPELINE IS CLEAN. SUBMIT WITH CONFIDENCE.**
