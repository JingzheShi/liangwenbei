# Guo, Pleiss, Sun, Weinberger (2017) — On Calibration of Modern Neural Networks

**Citation**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. ICML 2017.
**arXiv**: 1706.04599
**Why we read it**: This is the standard reference for **temperature scaling** — a 1-parameter softmax post-fit that calibrates multi-class probabilities without changing argmax. Best calibration baseline for our LightGBM 3-class outputs.

---

## Core finding

Modern deep nets (and modern boosted trees with high capacity) are **systematically over-confident**: the average top-class probability is well above the empirical accuracy. This is measured by Expected Calibration Error.

**ECE definition** (binning approach, M bins on confidence axis):

```
ECE = Σ_{m=1..M} (|B_m| / N) * |acc(B_m) - conf(B_m)|
```

where:
- `B_m` = samples whose top-class prob falls in m-th bin (e.g. M=15 equally spaced bins on [0,1])
- `acc(B_m)` = empirical fraction correct in bin m
- `conf(B_m)` = average top-class prob in bin m

A perfectly calibrated model has `acc(B_m) = conf(B_m)` for every bin → ECE = 0.

**Reliability diagram**: plot `acc(B_m)` vs `conf(B_m)`. Diagonal = perfect; below-diagonal = over-confident; above-diagonal = under-confident.

---

## Temperature scaling

Given uncalibrated logits `z ∈ R^K` and softmax `σ(z) = exp(z) / Σ exp(z_k)`:

```
p_calibrated = softmax(z / T)
```

Single scalar `T > 0`. Fit by minimizing **NLL on a held-out validation set**:

```
T* = argmin_T  − Σ_i log [softmax(z_i / T)]_{y_i}
```

Easy convex 1-D optimization (e.g. scipy `minimize_scalar` or LBFGS).

**Why it preserves accuracy**: dividing all logits by the same T does not change the argmax. So top-1 accuracy is **identical** before and after; only the probability values change.

**Why it works on LightGBM**: LightGBM with `objective='multiclass'` outputs raw scores per class which are then softmax'd — same shape as NN logits. We can pull the raw scores via `model.predict(X, raw_score=True)` and apply temperature scaling identically.

---

## Comparison to other methods (Guo's results)

On CIFAR-100 / ResNet:
- Histogram binning: ECE 0.045 (loses fine-grained info, hurts log-loss)
- Isotonic regression: ECE 0.044 (good, but K(K-1)/2 separate fits in multi-class)
- Platt (sigmoid) per-class: ECE 0.039
- Matrix scaling: too many params, overfits
- Vector scaling: K params, decent
- **Temperature scaling: ECE 0.030 with 1 param** ← best on most datasets

**Practical takeaway**: temperature scaling is a 5-line implementation that is "surprisingly effective" — Guo's exact word.

---

## Application to our setting

Our LightGBM model outputs softmax probabilities `(p_0, p_1, p_2)` for short / flat / long. Current threshold-gating treats raw `p_0, p_2` as if they were calibrated probabilities, but LightGBM softmax is typically over-confident on the dominant class (here `flat`) and under-confident on extremes after class-imbalance reweighting.

Recipe (assuming we hold out a calibration set, e.g. 20% of train days):
1. Train LightGBM on first 80% of train days
2. Get raw scores `z_i ∈ R^3` on the held-out 20%
3. Fit `T*` minimizing NLL: `argmin_T − Σ log softmax(z_i/T)[y_i]`
4. At eval time: `p = softmax(z / T*)` → feed to threshold-gating
5. **Argmax does not change** — but threshold T=0.55 now corresponds to a real 55% empirical hit rate, not just a model-internal score

**Expected benefit**: tighter relationship between threshold T and actual hit rate → easier to PnL-optimize the threshold via grid search, and the threshold transfers across folds because it's now in calibrated probability space.

---

## Sources

- [arxiv:1706.04599 abstract](https://arxiv.org/abs/1706.04599)
- [PMLR proceedings PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
- [Geoff Pleiss reference implementation (PyTorch, 30 lines)](https://github.com/gpleiss/temperature_scaling)
- [AWS Prescriptive Guidance: Temperature Scaling](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/temp-scaling.html)
