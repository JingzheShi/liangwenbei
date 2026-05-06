"""T26: Domain Randomization augments for LightGBM training.

Three feature-level augments (applied **only at training time**, never at val/test/inference):

  A. Per-feature random scale: X *= U[lo, hi]  (default 0.8..1.2)
     simulates "training-out sym has slightly different feature magnitudes"

  B. Gaussian noise:           X += N(0, sigma * train_std)
     small additive noise to harden against per-sym distributional shifts

  C. Random feature dropout:   X[mask] = train_mean[mask]
     some fraction of features per sample replaced by their training mean
     (zero is unsafe for non-centered features)

  ABC. Combined: apply A, then B, then C with reduced strengths.

Each function returns the augmented X (same shape, same dtype) plus the y
unchanged. Augmented batch is concatenated with the original training batch
so the model sees both clean and perturbed samples (similar to T14 mixup
aug_ratio=1.0 pattern). Sample weights are recomputed for the merged set.

Constraint compliance:
  * No use of `sym` to drive augmentation parameters (sym-agnostic)
  * No use of `date`
  * Augment is per-sample and stateless across calls (predictor never sees this)
"""
from __future__ import annotations

import numpy as np


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X: np.ndarray, rng: np.random.Generator,
                lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Per-(sample, feature) multiplicative scale in [lo, hi]."""
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def aug_b_noise(X: np.ndarray, rng: np.random.Generator,
                feat_std: np.ndarray, sigma_frac: float = 0.05) -> np.ndarray:
    """Additive Gaussian noise scaled by per-feature std."""
    noise = rng.standard_normal(size=X.shape).astype(X.dtype)
    scale = (sigma_frac * feat_std).astype(X.dtype)
    return X + noise * scale  # broadcast over rows


def aug_c_dropout(X: np.ndarray, rng: np.random.Generator,
                  feat_mean: np.ndarray, drop_lo: float = 0.05,
                  drop_hi: float = 0.10) -> np.ndarray:
    """Random feature dropout: mask 5-10% per sample, replace with train mean."""
    drop_rates = rng.uniform(drop_lo, drop_hi, size=(X.shape[0], 1)).astype(np.float32)
    keep_prob = 1.0 - drop_rates  # broadcast across feat axis
    keep_mask = (rng.random(size=X.shape).astype(np.float32) < keep_prob)
    out = np.where(keep_mask, X, feat_mean.astype(X.dtype))
    return out.astype(X.dtype)


def aug_abc(X: np.ndarray, rng: np.random.Generator,
            feat_std: np.ndarray, feat_mean: np.ndarray,
            scale_lo: float = 0.9, scale_hi: float = 1.1,
            sigma_frac: float = 0.03,
            drop_lo: float = 0.03, drop_hi: float = 0.07) -> np.ndarray:
    """Combined A+B+C with reduced per-augment strengths."""
    out = aug_a_scale(X, rng, lo=scale_lo, hi=scale_hi)
    out = aug_b_noise(out, rng, feat_std, sigma_frac=sigma_frac)
    out = aug_c_dropout(out, rng, feat_mean, drop_lo=drop_lo, drop_hi=drop_hi)
    return out


VARIANTS = ("baseline", "aug_a", "aug_b", "aug_c", "aug_abc")


def build_train_for_variant(
    variant: str,
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    feat_std: np.ndarray,
    feat_mean: np.ndarray,
    rng: np.random.Generator,
    aug_ratio: float = 1.0,
    num_class: int = 3,
):
    """Return (X_train, y_train, w_train) for a given variant.

    For non-baseline variants, X_train = concat(X_orig, X_aug) where
    X_aug is sampled from X_orig (with replacement if aug_ratio>1) and
    augmented per `variant`. y is duplicated; weights are class-balanced
    on the merged y.
    """
    assert variant in VARIANTS, f"unknown variant {variant}"
    if variant == "baseline":
        return X_orig, y_orig.astype(np.int64), class_balanced_weight(y_orig, num_class)

    n_aug = int(round(aug_ratio * len(X_orig)))
    if n_aug <= 0:
        return X_orig, y_orig.astype(np.int64), class_balanced_weight(y_orig, num_class)

    if n_aug == len(X_orig):
        idx = np.arange(len(X_orig))
    else:
        idx = rng.integers(0, len(X_orig), size=n_aug)
    X_src = X_orig[idx]
    y_src = y_orig[idx]

    if variant == "aug_a":
        X_aug = aug_a_scale(X_src, rng)
    elif variant == "aug_b":
        X_aug = aug_b_noise(X_src, rng, feat_std)
    elif variant == "aug_c":
        X_aug = aug_c_dropout(X_src, rng, feat_mean)
    elif variant == "aug_abc":
        X_aug = aug_abc(X_src, rng, feat_std, feat_mean)
    else:
        raise ValueError(variant)

    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_src], axis=0).astype(np.int64)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def compute_feat_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) across all rows. std clamped to >=1e-6."""
    feat_mean = X.mean(axis=0).astype(np.float32)
    feat_std = X.std(axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    return feat_mean, feat_std


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10000, 50)).astype(np.float32)
    y = rng.integers(0, 3, size=10000)
    fmean, fstd = compute_feat_stats(X)
    for v in VARIANTS:
        Xn, yn, wn = build_train_for_variant(v, X, y, fstd, fmean, rng)
        print(f"{v:10s}: X={Xn.shape} y={yn.shape} w_sum={wn.sum():.0f} y_unique={np.unique(yn, return_counts=True)}")
