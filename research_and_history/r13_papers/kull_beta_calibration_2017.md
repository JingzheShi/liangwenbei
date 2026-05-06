# Kull, Silva Filho, Flach (2017) — Beta Calibration

**Citation**: Kull, M., Silva Filho, T., Flach, P. (2017). Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration. *Electronic Journal of Statistics* 11, 5052–5080. AISTATS 2017 short version.
**Why we read it**: A drop-in replacement for Platt scaling that handles **non-sigmoid distortions** in calibration curves — exactly what GBDTs (which can have S-shaped or inverse-S-shaped reliability) tend to produce.

---

## Problem with Platt scaling

Platt scaling fits a 2-parameter sigmoid to `(score, label)`:

```
p̂(s) = 1 / (1 + exp(−(a·s + b)))
```

This is the right family **only when per-class scores are normally distributed with equal variance** (Platt's original derivation assumed Gaussian-mixture decision scores). For:
- **Boosted trees** (LightGBM/XGBoost): scores tend to push toward 0/1 → calibration curve has an S-shape that Platt can fit OK but not optimally
- **Naive Bayes / random forests**: heavily skewed scores → Platt can produce calibrations *worse* than uncalibrated scores
- **GBDT with class-imbalance reweighting**: bimodal score distribution → Platt fits poorly

---

## Beta calibration formula

Three-parameter map (a, b, c):

```
p̂(s) = 1 / (1 + 1/F(s))
```

where:

```
F(s) = exp(c) · s^a · (1 − s)^(−b)
```

so equivalently in logit form:

```
log(p̂(s) / (1 − p̂(s))) = c + a · log(s) − b · log(1 − s)
```

This is a **logistic regression on two features**: `log(s)` and `−log(1−s)`. Fit:
1. Compute `x1 = log(s)`, `x2 = −log(1−s)` from raw classifier scores `s ∈ (0,1)` on held-out val set
2. Fit logistic regression `y ~ x1 + x2` with intercept (using sklearn `LogisticRegression`)
3. The fitted `(a, b, c)` give the calibration map

The constraint `a, b > 0` ensures the calibration map is monotone increasing. Kull recommends a constrained optimizer or a 2-step fit: relax + clip.

---

## Why it generalizes Platt

- `a = b` and rename `α = a` gives a sigmoid-on-logit-of-score (close to Platt)
- `a = 0` or `b = 0` allows asymmetric distortions where one side of the calibration curve is sigmoid-like and the other is flat
- Identity (no recalibration) corresponds to `a = b = 1, c = 0`

So beta calibration = **a strictly larger family** that includes Platt as a special case. If Platt is optimal, beta scaling will recover it; otherwise beta does strictly better.

---

## Empirical results

Kull et al. compared on UCI datasets across 6 classifier types (NB, AdaBoost, RF, LR, SVM, MLP) and found beta calibration outperforms Platt scaling with consistent margins (5–15% Brier-score improvement). It is roughly tied with isotonic regression but uses far less data (3 params vs hundreds of breakpoints) → less overfitting on small calibration sets.

---

## Multi-class extension

Same one-vs-rest as Platt/isotonic in sklearn:
1. Fit beta calibration per class against `y == k` binary labels
2. Get K calibrated probabilities `p_k`
3. Renormalize: `p_k' = p_k / Σ_j p_j`

For our 3-class case: 3 beta calibrators → 3 logistic regressions on 2 features each → almost free.

---

## Application to our setting

| Method | Params | Min cal samples | Multi-class? | LightGBM fit? |
|---|---|---|---|---|
| Platt | 2 | ~100 | OvR | OK |
| **Beta** | **3** | **~200** | **OvR** | **Better than Platt for GBDT-style scores** |
| Isotonic | many | ~1000 | OvR | Best with lots of data, prone to overfit |
| Temperature | 1 | ~100 | Native | Best 1-param baseline |

We have ~50k LOSO training-set rows when we hold out 1 sym → enough for any of these. **Beta calibration is the recommended pre-step before temperature scaling** (or instead of, if we want a per-class form).

---

## Sources

- [Kull 2017 PMLR paper](https://proceedings.mlr.press/v54/kull17a.html)
- [Kull 2017 EJS (extended version)](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-2/Beyond-sigmoids--How-to-obtain-well-calibrated-probabilities-from/10.1214/17-EJS1338SI.pdf)
- [betacal Python package + GitHub](https://betacal.github.io/)
- [Abzu intro to calibration part II](https://www.abzu.ai/data-science/calibration-introduction-part-2/)
