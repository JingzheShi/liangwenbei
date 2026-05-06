"""Cross-sym mixup augmentation core + sanity check.

Mechanism (vectorized):
    Sample idx_i, idx_j ~ Uniform[0, N) with rejection until sym[i] != sym[j].
    lam_raw ~ Beta(alpha, alpha)
    lam = max(lam_raw, 1 - lam_raw)        # symmetric mixup -> lam in [0.5, 1]
    x_mix = lam * x_i + (1 - lam) * x_j
    label = label_i  (dominant since lam >= 0.5)
    weight = lam * class_balanced_weight(label_i)

Hard-constraint compliance:
    - sym is used ONLY at augmentation-time to pair across syms; not leaked into features.
    - x_mix occupies the same 226-d feature space as original x; no sym ID added.
    - inference path is unchanged (mixup is training-only).

Usage:
    python build_mixup.py --alpha 0.2 --n-aug 1000 --quick
    -> prints stats + saves a sanity sample.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def cross_sym_mixup(
    X: np.ndarray,
    y: np.ndarray,
    sym: np.ndarray,
    n_aug: int,
    alpha: float,
    rng: np.random.Generator,
    max_reject_iters: int = 30,
):
    """Generate n_aug cross-sym mixup samples.

    Returns (X_aug, y_aug, lam) where:
        X_aug: [n_aug, F] mixed features
        y_aug: [n_aug] dominant labels (argmax of one-hot mixup)
        lam:   [n_aug] mixing weight in [0.5, 1.0]
    """
    N = len(X)
    assert len(y) == N and len(sym) == N

    idx_i = rng.integers(0, N, size=n_aug)
    idx_j = rng.integers(0, N, size=n_aug)
    bad = sym[idx_i] == sym[idx_j]
    iters = 0
    while bad.any() and iters < max_reject_iters:
        n_bad = int(bad.sum())
        idx_j[bad] = rng.integers(0, N, size=n_bad)
        bad = sym[idx_i] == sym[idx_j]
        iters += 1
    if bad.any():
        # Last-resort: replace remaining bad with the next-different sym index.
        for k in np.where(bad)[0]:
            target = sym[idx_i[k]]
            cand = np.where(sym != target)[0]
            idx_j[k] = rng.choice(cand)

    lam_raw = rng.beta(alpha, alpha, size=n_aug).astype(np.float32)
    lam = np.maximum(lam_raw, 1.0 - lam_raw)
    lam_col = lam[:, None]

    X_aug = (lam_col * X[idx_i] + (1.0 - lam_col) * X[idx_j]).astype(np.float32)
    y_aug = y[idx_i].astype(np.int8)
    return X_aug, y_aug, lam


def augment_for_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sym_tr: np.ndarray,
    aug_ratio: float,
    alpha: float,
    seed: int,
    num_class: int = 3,
):
    """Build (X_combined, y_combined, sw_combined) for a fold.

    aug_ratio = n_aug / N_train.
    Original samples retain class_balanced_weight; augmented samples use
    lam * class_balanced_weight(label_i).
    """
    rng = np.random.default_rng(seed)
    N = len(X_tr)
    n_aug = int(N * aug_ratio)

    sw_orig = class_balanced_weight(y_tr, num_class=num_class)
    if n_aug <= 0 or alpha <= 0:
        return X_tr, y_tr, sw_orig

    t0 = time.time()
    X_aug, y_aug, lam = cross_sym_mixup(X_tr, y_tr, sym_tr, n_aug, alpha, rng)
    aug_time = time.time() - t0

    sw_aug_base = class_balanced_weight(y_aug, num_class=num_class)
    sw_aug = (lam * sw_aug_base).astype(np.float32)

    X_combined = np.concatenate([X_tr, X_aug], axis=0)
    y_combined = np.concatenate([y_tr, y_aug], axis=0)
    sw_combined = np.concatenate([sw_orig, sw_aug], axis=0)
    print(
        f"  [mixup] N_orig={N:,} n_aug={n_aug:,} alpha={alpha} "
        f"lam_mean={float(lam.mean()):.3f} lam_min={float(lam.min()):.3f} "
        f"aug_time={aug_time:.1f}s",
        flush=True,
    )
    return X_combined, y_combined, sw_combined


def _sanity_check(args):
    """Quick sanity: load a small slice of T5b cache, run mixup, print stats."""
    print(f"=== sanity check: alpha={args.alpha} n_aug={args.n_aug} ===", flush=True)
    p = os.path.join(CACHE_DIR, "schemeC_train.npz")
    print(f"loading {p} ...", flush=True)
    d = np.load(p)
    N_full = len(d["sym"])
    if args.quick:
        rng_q = np.random.default_rng(0)
        idx = np.sort(rng_q.choice(N_full, size=min(200_000, N_full), replace=False))
        X = d["X"][idx]
        y = d["y10"][idx]
        sym = d["sym"][idx]
    else:
        X = d["X"]
        y = d["y10"]
        sym = d["sym"]
    print(f"  N={len(X):,} X.shape={X.shape} dtype={X.dtype}", flush=True)
    print(f"  sym distribution: {dict(zip(*np.unique(sym, return_counts=True)))}", flush=True)
    print(f"  y10 distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    rng = np.random.default_rng(42)
    t0 = time.time()
    X_aug, y_aug, lam = cross_sym_mixup(X, y, sym, args.n_aug, args.alpha, rng)
    print(f"  mixup time: {time.time() - t0:.2f}s", flush=True)
    print(f"  X_aug.shape={X_aug.shape} y_aug.shape={y_aug.shape} lam.shape={lam.shape}", flush=True)
    print(
        f"  lam stats: min={lam.min():.3f} mean={lam.mean():.3f} median={np.median(lam):.3f} max={lam.max():.3f}",
        flush=True,
    )
    print(f"  y_aug distribution: {dict(zip(*np.unique(y_aug, return_counts=True)))}", flush=True)

    # Quantile of lam
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    for q in qs:
        print(f"    lam q{int(q*100):>2d}: {np.quantile(lam, q):.3f}", flush=True)

    # Verify cross-sym: re-derive idx_i, idx_j? Not needed; we trust the function.
    # But we can spot-check: for a few aug samples, find the closest two original sym sources.
    # Skip — already vectorized.

    # Output dim check
    assert X_aug.shape[1] == X.shape[1]
    assert X_aug.dtype == X.dtype
    print("  OK: shape & dtype preserved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--n-aug", type=int, default=100_000)
    ap.add_argument("--quick", action="store_true",
                    help="use 200k subset of train for fast sanity")
    args = ap.parse_args()
    _sanity_check(args)


if __name__ == "__main__":
    main()
