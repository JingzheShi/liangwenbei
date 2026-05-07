# CV Design Audit — 良文杯 LOSO Cross-Validation

**Date**: 2026-05-07  
**Auditor**: AUDIT-CV worker  
**Based on**: T37/T45 train_loso.py, de_thresh.py, src/data/split.py, T23/T34 reports, history_and_important_notes.md

---

## 1. Current CV Design (Diagram)

```
Dates:  [0 ............. 79][80 ...... 95][96 .......... 119]
         TRAIN (80 days)      VAL (16d)    TEST/OOF (24d)
         5 syms × 80d × 2s   5 syms×16d×2 5 syms×24d×2
         = 800 sessions       = 160 sess.  = 240 sessions

───── LOSO outer loop (5 folds) ────────────────────────────────────
Fold k = held-out sym k ∈ {0,1,2,3,4}:

  Training data:  {sym ≠ k} × date[0..79]   → 640 sessions
                  + uniform aug [0.75,1.25]  → ×2 = 1280 samples (T37)
  Val (ES):       {sym ≠ k} × date[80..95]  → 128 sessions (NO aug)
                  ↑ used for early_stopping(patience=40)
  Test (OOF):     {sym = k} × date[96..119] → 48 sessions (NO aug)

  5 seeds × different (num_leaves, ff, bf, l2) hyperparams
  → 25 models per fold set, each producing OOF probabilities

───── Threshold optimization (DE) ──────────────────────────────────
  Input:  average probs across 5 seeds for each of 5 OOF folds
  Search: 4D (T_up, T_dn, d_up, d_dn) coarse grid + DE (maxiter=80,
          popsize=24, 5 DE seeds) + neighborhood check
  Objective: maximize Σ_{k=0..4} pnl(fold_k | T_up,T_dn,d_up,d_dn)
  Output: best 4D threshold → applied to FINAL submission

  ⚠️ SAME OOF used for BOTH evaluation and threshold selection ⚠️

───── Reported metric ──────────────────────────────────────────────
  LOSO sum_pnl = Σ_{k=0..4} pnl(OOF_k | DE_thresh)

  Current SOTA iter_006: sum = +13.61
  Best individual fold: sym=4 +6.23 (contributes 46% of total)
```

---

## 2. Risk Analysis

### Q1 — LOSO OOD Fidelity: Does it mimic the platform's test distribution?

**Severity: HIGH**

**What LOSO tests:**  
Train on 4 known syms → test on 1 known sym. The 5th sym (e.g., sym=2) was present in all training sessions — the model has learned its *sibling distributions* from syms {0,1,3,4}. Within-LOSO, held sym's microstructure is implicitly constrained to "be like the other 4 syms shifted."

**What platform tests (per CRITICAL_CONSTRAINTS.md §3):**  
> "sym 0-4, 可能有不来自于 5 只训练集股票的数据"  

Platform sym IDs (0-4) may map to **completely different stocks** — with foreign microstructure the model has never seen. This is a strictly harder OOD than LOSO.

**Train:test ratio mismatch:**  
- LOSO: 4 known syms train → 1 known sym test (4:1 ratio)  
- Platform: 5 known syms train → unknown mix of known+unseen syms test

**Evidence of calibration gap:**  
From T23 calibration (2 anchors), at h_60:  
- iter_002: LOSO +6.30 → platform +4.07 (36% discount)  
- iter_005b: LOSO +11.46 → predicted platform +9.23 (20% discount)  
- The calibration slope a=9.97 is pinned on just 2 data points spanning only ΔLOSO=2.73 — extrapolation to iter_006 LOSO +13.61 is extremely uncertain

**T8 analysis (sym=2 brittleness)** confirms that even within the 5 training syms, one sym (sym=2, likely large-cap/ETF) is highly OOD relative to the others: amount_delta +7.49 z-scores. If the platform's test includes similar "outlier-stock" substitutions, LOSO systematically underestimates the penalty.

**Consequence:**  
LOSO sum is an **optimistic, biased** estimate of platform performance. The true OOD gap could be larger than the observed 2-4 point gap if platform substitutes truly novel stocks.

---

### Q2 — Temporal Cut: Future-looking? Purged walk-forward needed?

**Severity: LOW-MEDIUM**

**Temporal cut is clean:**  
```
split.py:  train=date[0,80), val=date[80,96), test=date[96,120)
```
Sessions are loaded per-file (`snapshot_sym{s}_date{d}_{sess}.parquet`). Each file is independent; no cross-session state in features. The 100-tick sliding window is entirely within one session. No cross-day feature leakage.

**No future-looking in features:**  
All features use only the last 100 ticks of the current session. No normalization statistics cross the train/val/test boundary (features are pre-normalized as log-returns and turnover ratios per data_schema.md).

**Temporal drift IS real (not a leakage issue, but a distribution shift issue):**  
From T23 Scheme C (rolling windows within test period):  
| Window | Dates | iter_002 h_10 cum_pnl |
|---|---|---|
| w0 | 96–103 | +9.10 |
| w1 | 104–111 | +7.51 |
| w2 | 112–119 | +5.25 |

**-43% decay from w0 to w2** — model performance degrades as dates move further from training end (date 79). If the platform's test set is yet further forward in time (post-date 119), the decay continues.

**No purged walk-forward is used.** The current design uses a single temporal split (80/16/24 days). A rolling walk-forward CV would better quantify this temporal drift, but its absence is not a *leakage* issue — it's a *confidence interval* issue.

**Missing gap analysis:** There is no "purging gap" between train (date 79) and val (date 80). For a 60-tick horizon ≈ 3 minutes, autocorrelation decays fast in LOB data. However, the session boundary between date 79 PM and date 80 AM is a natural break — no autocorrelation crosses this. The gap is unnecessary here.

---

### Q3 — Val Set Usage: Early Stopping Bias?

**Severity: MEDIUM**

**Val set definition:**  
Val = {sym ≠ k, date 80–95} — same 4 training syms, 16 later days.

**What this means for early stopping:**  
The model's best_iteration is selected by minimizing `multi_logloss` on these 128 val sessions. The val distribution is:
- **In-sym**: 4 syms the model was trained on — NOT the held-out sym
- **In-time**: 16 days immediately after training period

**The misalignment:**  
Early stopping optimizes for "perform well on dates 80-95 of training syms." But the actual test is "perform on dates 96-119 of held-out sym." Two sources of distributional mismatch:
1. **Sym mismatch**: Val is training syms; test is held-out sym. Model may stop at a point that's great for training syms but suboptimal for OOD sym.
2. **Time mismatch**: Val is 80-95, test is 96-119. Given temporal decay (Q2), the "optimal" iteration for dates 80-95 may differ from optimal for dates 96-119.

**Alternative considered but rejected:**  
Using val = {sym = k, date 80–95} for early stopping would introduce information leakage: model selection uses data from the same sym being tested. The current design (val on training syms) is the correct LOSO convention.

**Practical severity:**  
With patience=40 rounds and typical best_iter ~120-150 (from T37 logs), the model is reasonably regularized. The mismatch adds noise but is unlikely to be a large bias. **Medium** risk.

---

### Q4 — Augmentation and CV Independence

**Severity: LOW** (no material issue)

**Augmentation design in T37:**
```python
# Per-fold deterministic seed
fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
# Per-(sample, feature) uniform scale, concatenate original + augmented
X_tr, y_tr, sw_tr = aug_uniform_concat(X_tr_o, y_tr_o, lo, hi, fold_rng)
```

**Correctness check:**  
- ✅ Aug only applied to training data (`X_tr_o`), never to val or test
- ✅ Each (seed, held_sym) pair gets a unique deterministic RNG — no cross-fold contamination
- ✅ Augmented labels are identical to originals (label-preserving scale)
- ✅ Val uses `class_balanced_weight(y_va)` computed from val labels only — no aug stats

**One potential concern:**  
`class_balanced_weight` on merged (orig + aug) training set doubles each class equally, so class ratios are unchanged. Weights are identical to original training distribution. No issue.

**Augmentation range impact:**  
T37 used [0.75, 1.25] (wider than T31's [0.80, 1.20]). History shows wider aug improved single-model robustness but reduced 5-seed ensemble diversity (T37 negative result). This is a **model performance issue**, not a CV design issue.

---

### Q5 — LOSO Sum as Platform Proxy: Bias and Variance?

**Severity: HIGH**

**Bias (LOSO optimistic vs platform):**

| iter | LOSO h_60 | Platform h_60 | Gap |
|---|---|---|---|
| mmpc_demo | +3.57 | -23.13 | -26.70 |
| iter_002 | +6.30 | +4.07 | **-2.23** |
| iter_005b | +11.46 | predicted +9.23 | -2.23 (assumed) |
| iter_006 | +13.61 | **UNKNOWN** | ? |

Key concerns:
1. Only **2 trained-model calibration anchors** (mmpc_demo is a random model, not a fair anchor for trained models). The LOSO→platform mapping is extremely uncertain.
2. The gap of -2.23 is assumed constant, but the true relationship may be nonlinear or model-architecture-dependent.
3. **iter_006's LOSO includes DE threshold optimization on OOF (see Q7)**. The true platform PnL will use the same threshold applied to new, unseen stocks. The threshold may not generalize.

**Variance (uncertainty in LOSO sum):**

From T34 bootstrap CI for iter_005b h_60:
- 95% CI: [+5.02, +18.22] — a **±7 point** range at 95% confidence
- Cluster-sym bootstrap std: ±3.3 per fold

For iter_006 (stronger model, same structure): CI width likely similar or wider.

**Structural coverage gap:**  
The LOSO OOF covers date 96-119 of each held sym. Platform's test set:
- May cover different date ranges (post-119?)
- May include genuinely new stocks (not just the 5 training ones)

Neither of these scenarios is captured in LOSO.

**Practical consequence:**  
iter_006 LOSO +13.61 maps to platform estimate of roughly +13.61 - 2.23 = **+11.4** (using constant-shift model), but the true range is approximately **[+5, +18]** at 95% confidence — before accounting for threshold generalization uncertainty.

---

### Q6 — Multi-Seed: Does It Reduce CV Noise?

**Severity: MEDIUM**

**What 5 seeds reduce:**  
Seed-specific model variance within each fold. If a single model's prediction for a sample has variance σ²_model, averaging 5 models reduces it to σ²_model / 5.

**What 5 seeds do NOT reduce:**  
**Sym-level OOD variance** — the dominant noise source.

From T23 Scheme E (cluster-sym bootstrap):
- Session-level bootstrap std: ±1.89
- **Cluster-sym bootstrap std: ±5.26** (nearly 3× larger)

This means: the biggest source of LOSO uncertainty is which sym is held out, not which random seed is used. The 5-seed ensemble reduces model noise but the sym-level variance (~±5 per fold) is unchanged.

**Evidence from seed-level inspection (T37 stepA):**  
```
raw argmax per fold: [+1.93, -3.12, -7.21, +0.15, +7.68]
```
Fold-level variance is enormous (range: ~15 units). Across the 5 seeds, the per-seed LOSO sums varied by <0.5. The dominant variance is OOF-fold-specific, not seed-specific.

**But 5 seeds DO help the threshold optimizer:**  
5-seed averaged probabilities are smoother than single-seed, making the DE threshold optimization more robust and less sensitive to single-model peculiarities. This is a real benefit.

**Honest assessment:**  
The 5-seed ensemble is worth doing for model quality and OOF smoothing, but it does NOT meaningfully reduce the fundamental uncertainty in the LOSO sum. Reporting "5-seed LOSO = +13.61" without a ±6 confidence interval is misleading.

---

### Q7 — Threshold Selection Contaminating OOF: Is There OOF Overfitting?

**Severity: HIGH** (most technically important issue)

**The contamination:**

```
OOF predictions (date 96-119, 5 held syms)
      │
      ├──→ Report: LOSO sum_pnl = f(OOF | thresh*)
      └──→ DE threshold search: thresh* = argmax_thresh Σ pnl(OOF | thresh)
```

**The same OOF data is used to both optimize the threshold AND report the final metric.**

This is circular: the reported +13.61 was maximized by finding the best (T_up, T_dn, d_up, d_dn) on the exact data we're evaluating. The threshold is NOT cross-validated.

**How bad is the overfitting?**  

In de_thresh.py, DE runs:
- Coarse grid: ~50k combinations (9×9×9×9 ≈ 6,500 combos)
- DE: 5 seeds × maxiter=80 × popsize=24 ≈ **9,600+ function evaluations**
- Neighborhood check: additional ~1,000 combos

With **~240k OOF data points** and only **4 parameters**, overfitting risk per-parameter is low in theory. But the landscape is pathological for optimization:
1. PnL is a non-smooth step function of thresholds (changing (T_up, d_up) by ε can switch 1000s of samples in/out)
2. Sym=4 contributes ~46% of OOF PnL and has unusual properties — the optimizer may exploit sym=4's specific data
3. The OOF date range (96-119) may have specific distributional properties the threshold latches onto

**Empirical evidence of overfitting (T38):**  
> "DE 4D thresh 噪声 ±0.05 — 任何 LOSO sum 改进 < 0.1 都不可信，改进必须 > +0.3 才值得提交"

This ±0.05 DE noise estimate is just the optimizer's stochasticity, NOT the out-of-sample generalization gap. The true generalization gap (OOF threshold → platform) is unknown but likely larger.

**From T37 de log — raw argmax vs DE threshold:**
```
raw argmax sum:  -0.57
DE optimized:    +12.94 to +13.61
```

The **threshold adds +13-14 units of PnL** on top of raw argmax. This enormous gap suggests the DE threshold is doing heavy work on OOF-specific patterns. How much of this transfers to platform is unknown.

**The correct fix:** Nested CV:
1. Use **val set (date 80-95)** to select optimal threshold
2. Report OOF test (date 96-119) with that threshold

This would give an unbiased LOSO estimate. The price: val only covers 16 days × 4 syms = 128 sessions (vs 240 in OOF). Threshold optimization on 128 sessions might be noisier.

**Consequence for platform comparison:**  
If the best threshold on OOF (96-119) differs from the best threshold on the platform's test set, the LOSO score is inflated. Given that sym=4 has unusual properties and contributes 46% of DE-optimized PnL, this is a real risk.

---

## 3. Summary Risk Table

| Risk | Severity | Description |
|---|---|---|
| OOF threshold contamination (Q7) | **CRITICAL** | DE on same OOF used for final metric; +13 from threshold may be partially OOF-specific |
| LOSO doesn't simulate platform OOD (Q1) | **HIGH** | Platform tests may include novel stocks; LOSO tests known-sym generalization only |
| LOSO→platform gap unknown for iter_006 (Q5) | **HIGH** | Only 2 calibration anchors, constant-shift assumption unvalidated; 95% CI is ±7 points |
| Multi-seed doesn't reduce dominant variance (Q6) | **MEDIUM** | Sym-level variance (±5/fold) dominates; 5 seeds reduce only model noise |
| Val early stopping on training syms (Q3) | **MEDIUM** | Early stopping optimizes for in-distribution syms, not held-out sym distribution |
| Temporal drift uncaptured (Q2) | **LOW-MED** | No walk-forward CV; T23 shows -43% PnL decay within 24-day test window |
| Augmentation design | **LOW** | Correctly isolated to training; no leakage |

---

## 4. Recommended More Robust CV Scheme

### Option A: Threshold-Clean LOSO (quick fix, 2-4 hours)

```
For each held_sym k:
    TRAIN_DATA: sym≠k, date[0..79]   → aug → LightGBM
    VAL_DATA:   sym≠k, date[80..95]  → early stopping
    OOF_DATA:   sym=k, date[96..119] → OOF predictions

DE threshold selection:
    ← USE VAL_DATA predictions (80-95) to optimize (T_up, T_dn, d_up, d_dn)
    ← NOT OOF_DATA!
    
Evaluation:
    Apply val-selected threshold to OOF_DATA
    Report LOSO sum on OOF_DATA with val-threshold (UNBIASED)
```

Requires saving val predictions (sym≠k, date 80-95) during training. Currently these are discarded. Predicted impact: LOSO sum may drop 1-3 points (removing circular optimization benefit), but the reported number is more honest and closer to platform gap.

### Option B: 3-Window Temporal Walk-Forward (no retraining, 1-2 hours from OOF)

```
For each held_sym k, split OOF test into 3 sub-windows:
    w0: date 96-103  (8 days)
    w1: date 104-111 (8 days)
    w2: date 112-119 (8 days)

Report: per-window cum_pnl, time-decayed metric = weighted(w0×0.5, w1×0.3, w2×0.2)
```

This quantifies temporal drift within the OOF itself. If performance decays from w0→w2, apply a discount to the LOSO sum to estimate platform impact (where test may extend beyond date 119).

### Option C: Nested CV with Purged Sym-Val (best practice, not feasible locally)

```
Outer loop (5-fold LOSO on test):
    held_sym = k
    test: sym=k, date 96-119

Inner loop (for threshold selection, separate val):
    val_k: sym=k, date 80-95   ← uses some of held sym's data for thresh selection
```

**Warning**: This introduces information leakage (val_k and test_k are same sym). A cleaner nested CV would need a 6th "inner holdout" sym which is impossible with only 5 syms.

**The fundamental constraint**: With 5 syms, you cannot do nested CV without either data leakage (using held sym's val) or losing the outermost OOD fold entirely.

### Recommended Practical Scheme (achievable within competition timeline)

```
1. LOSO training: unchanged (correct as-is)
2. Threshold selection: move to val set (sym≠k, date 80-95) — FIXES Q7
3. Reporting: always show "raw argmax LOSO sum" AND "val-thresh LOSO sum"
4. Bootstrap CI: run T34's bootstrap_ci.py on iter_006 OOF — for honest uncertainty
5. LO2SO: run T34's lo2so.py on iter_006 OOF — stress test
6. Time-decay check: split OOF into 3 windows (w0/w1/w2), report decay rate
```

---

## 5. Top 3 Quick-Win Plans

### QW1: Time-Window Decay Check (Severity for risk: LOW-MED → confirm/deny drift)

**What**: Compute OOF cum_pnl for each of 5 LOSO folds broken into 3 sub-windows (96-103, 104-111, 112-119). Check monotonic decay.

**How**: ~30 lines of Python on existing iter_006 OOF parquets. No model retraining.

```python
for k in range(5):
    df = load_oof_parquet(k)  # existing {tag}_pred_seed*_held{k}.parquet averaged
    for window, (d_lo, d_hi) in enumerate([(96,104),(104,112),(112,120)]):
        sub = df[(df.date >= d_lo) & (df.date < d_hi)]
        pnl_w = compute_pnl(sub, thresh)
        print(f"fold={k} window={window} dates={d_lo}-{d_hi} pnl={pnl_w:+.3f}")
```

**Expected time**: 2-3 hours (write + run EDA script on existing OOF parquets)

**Expected benefit**: If decay exists (likely, per T23 Scheme C), this:
1. Gives a discount factor to apply to LOSO sum for platform prediction
2. Motivates either re-weighting training toward recent dates OR walk-forward CV

**Expected LOSO impact**: None directly. Diagnostic only. But might justify a **-15% discount** on the current LOSO→platform prediction.

---

### QW2: Val-Based Threshold Selection (Severity fix for Q7: CRITICAL)

**What**: Retarget DE threshold optimization to use val-set predictions (sym≠k, date 80-95) instead of OOF test predictions. Apply resulting threshold to OOF test for uncontaminated LOSO sum.

**How**: Modify training loop to save val predictions:
```python
# In train_one_fold, after training:
va_prob = predict_proba(booster, X_va)  # val predictions
va_pred = va_prob.argmax(axis=1)
# Save va_df similarly to te_df
```

Then run DE on val predictions across 5 folds (each fold's val = sym≠k, date 80-95). Apply best threshold to OOF.

**Expected time**: 6-12 hours (modify train_loso.py to save val preds, retrain would take 2-3h, run DE on val OOF ~30min)

**However**: Retraining not strictly needed. Current models already exist. We just need to re-run inference on val set using saved models.

```bash
# Smaller version: use existing models, run predict on val set
for k in {0..4}; do
    for seed in 42 1 7 13 100; do
        python3 infer_val.py --model stepA_model_seed${seed}_held${k}.txt --split val --held ${k}
    done
done
```

**Expected benefit**: Removes circularity in LOSO+threshold reporting. True LOSO sum will likely drop 1-4 points (from +13.61 to possibly +9-12), but this number is **more trustworthy** for platform prediction. Avoids submitting overfitted thresholds to platform.

**Expected LOSO change**: -1 to -4 points (losing OOF-specific threshold benefit). But this represents *honest* PnL. Platform gap may shrink too.

---

### QW3: LO2SO + Bootstrap CI for iter_006 (Confidence calibration)

**What**: Run T34's `lo2so.py` and `bootstrap_ci.py` adapted for iter_006 OOF parquets.

**How**: T34 already has the infrastructure. Add iter_006 as a new target in `oof_io.py`, then run:
```bash
cd experiments/T34_cv_v2
python3 lo2so.py       # 10 sym pairs, ~15 min
python3 bootstrap_ci.py # session-level CI, ~5 min
```

**Expected time**: 2-4 hours (adapt oof_io.py for iter_006 layout, run scripts)

**Expected benefit**:
- **LO2SO worst-pair PnL**: If negative (iter_005b had worst-pair -0.18), indicates fragility. For iter_006, if worst-pair goes negative with val-thresh (QW2), flag before submitting.
- **Bootstrap CI**: Quantify whether the +13.61 is in the "reliably positive" zone or just a lucky draw.
- **Platform prediction**: Update M5 anchor prediction for iter_006 based on actual calibration shift.

**Expected LOSO change**: None (diagnostic only). But might flag "DO NOT SUBMIT" if worst-pair < -2 or CI lower bound < 0.

---

## 6. Honest Summary: What Can We Trust?

| Claim | Trustworthiness | Honest bound |
|---|---|---|
| "iter_006 LOSO sum = +13.61" | OPTIMISTIC (Q7 contamination) | True unbiased estimate: likely +9 to +12 |
| "platform PnL ≈ LOSO - 2.23 = +11.38" | LOW confidence | 95% CI roughly [+4, +18] |
| "LOSO is a reliable platform proxy" | PARTIAL | Systematic +2-4 optimistic bias; wrong OOD distribution |
| "5-seed ensemble is stable" | TRUE for model noise | Sym-level variance ±5/fold dominates |
| "augmentation doesn't contaminate CV" | TRUE | Correctly isolated to train |
| "val early stopping is unbiased" | MOSTLY TRUE | Slight sym-mismatch bias, manageable |
