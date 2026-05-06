# Conformal Prediction for Selective Classification (2020-2025)

**Why we read it**: Conformal prediction gives **distribution-free coverage guarantees** for prediction sets. Combined with selective prediction ("abstain when uncertain"), it's a principled alternative to hand-set probability thresholds. Particularly attractive when the test distribution drifts (LOSO is exactly this kind of shift).

---

## Core algorithm — split (inductive) conformal

Given:
- A trained classifier `p̂(y|x)`
- A held-out **calibration set** `(x_i, y_i)_{i=1..n}` (NOT used to train the classifier)
- A user-chosen miscoverage level `α` (e.g. 0.10 for 90% coverage)

Step 1 — **non-conformity scores** on calibration set. Most common choice (APS, Adaptive Prediction Sets):

```
s_i = 1 − p̂(y_i | x_i)         # large s = "model confident in WRONG class"
```

Step 2 — quantile:

```
q̂ = (⌈(n+1)·(1−α)⌉)-th smallest of {s_1, ..., s_n}    # roughly the (1-α) quantile
```

Step 3 — at test time, prediction set:

```
C(x_test) = { y : 1 − p̂(y | x_test) ≤ q̂ }
          = { y : p̂(y | x_test) ≥ 1 − q̂ }
```

So the threshold for "include class y in the prediction set" is `1 − q̂`, derived from data, not hand-chosen.

**Coverage guarantee** (theorem, distribution-free):

```
P(y_test ∈ C(x_test)) ≥ 1 − α
```

provided calibration and test points are exchangeable. (Exchangeability fails under distribution shift — see Mondrian below.)

---

## Selective prediction with conformal sets

Use the prediction set to **decide when to abstain**. For our 3-class problem:

| Prediction set | Action | Interpretation |
|---|---|---|
| `{1}` | `pred = 1` (no trade) | model is confident in flat |
| `{0}` | `pred = 0` (short) | model confident in short |
| `{2}` | `pred = 2` (long) | model confident in long |
| `{0,1}` or `{1,2}` | `pred = 1` (abstain) | model can't pin down direction → don't trade |
| `{0,1,2}` | `pred = 1` (abstain) | model has no idea |
| `{0,2}` | `pred = 1` (abstain — rare but possible) | "either short or long but not flat" — strange, abstain |

This is **automatic, hyperparameter-free** beyond the choice of α. Compared to our T+δ rule it has a clear interpretation: "if the calibrated set has only the trade direction, trade; otherwise abstain."

---

## Why conformal is attractive for our task

1. **Distribution-free**: doesn't assume the model is calibrated; works on raw scores. (Though combining with calibration gives smaller sets.)
2. **Coverage transferable**: if the LOSO held-out sym is exchangeable with calibration, coverage holds. If not exchangeable (it isn't, exactly), we can use **Mondrian/conditional conformal** — separate quantile per group (e.g. per sym).
3. **No threshold to tune**: replace T+δ tuning with α (which has a direct interpretation: "I want to be wrong at most α of the time when I trade").
4. **Adapts to feature complexity**: hard inputs get bigger sets and abstain more; easy inputs get singletons.

---

## Mondrian (group-conditional) conformal — handles distribution shift

Compute `q̂_g` separately per group `g` (e.g. per sym, or per volatility regime):

```
q̂_g = (1−α) quantile of {s_i : group(i) = g}
```

This gives **coverage conditional on group**, e.g. coverage holds within each sym separately. For LOSO: we can pre-segment training data by sym, compute 5 different `q̂` values, and at test time use the test point's sym. This addresses the "OOD sym" issue more directly than a single global threshold.

If we don't trust sym ID at test time (CRITICAL_CONSTRAINTS §3 — sym may be new!), Mondrian on **observable signals** like spread tier or vol regime is a robust alternative.

---

## Practical recipe for our project

```python
# After LightGBM training on train days:
val_X, val_y = held_out_val_data        # ~10% of train days, NOT seen during fit
val_p = lgb_model.predict(val_X)        # softmax outputs (n_val × 3)
s = 1 - val_p[np.arange(n), val_y]      # non-conformity scores
alpha = 0.10                            # we want 90% coverage of the true class
q_hat = np.quantile(s, 1 - alpha, method='higher')

# At inference time:
def predict_conformal(x):
    p = lgb_model.predict(x)            # shape (3,)
    pred_set = np.where(p >= 1 - q_hat)[0]
    if pred_set.tolist() == [0]:   return 0   # short
    elif pred_set.tolist() == [2]: return 2   # long
    else:                          return 1   # abstain
```

**Expected improvement**: a coverage-controlled selective rule should be more robust than a hand-tuned T because it explicitly accounts for over-confidence. If the model is over-confident, `q̂` ends up smaller (looser threshold) → trades less; vice versa.

**Combining with calibration**: conformal works on any score, but calibrated `p` gives **smaller** sets (more efficient). So the recommended pipeline is:
1. Train LightGBM
2. Calibrate (temperature / beta) on val set 1
3. Compute conformal `q̂` on val set 2 using calibrated probs
4. At test: calibrate → conformal set → action

---

## Limitations / pitfalls

- **Exchangeability is required**. Pure LOSO violates this. Mondrian helps but is not perfect.
- **Coverage ≠ predictive performance**. A trivially-large set covers everything; conformal trades coverage for set size. For us, an over-large set → "always abstain" → 0 PnL (still better than negative).
- **n_calibration matters**. Smaller calibration set → noisier `q̂`. With our LOSO ~50k held-out rows, this is fine.
- **Multi-class with K=3 has only 3 set sizes that matter**: `{singleton}`, `{2-element}`, `{full}`. Most efficiency gains come from making singletons happen often.

---

## Sources

- [BBVA AI Factory — Conformal Prediction Intro](https://www.bbvaaifactory.com/conformal-prediction-an-introduction-to-measuring-uncertainty/)
- [Selective Conformal Risk Control (arXiv 2512.12844)](https://www.arxiv.org/pdf/2512.12844)
- [Conformal Selective Prediction with General Risk Control (arXiv 2603.24704)](https://arxiv.org/abs/2603.24704)
- [Calibrated Selective Classification (OpenReview)](https://openreview.net/pdf?id=zFhNBs8GaV)
- [Optimal strategies for reject option classifiers (arXiv 2101.12523)](https://arxiv.org/abs/2101.12523)
