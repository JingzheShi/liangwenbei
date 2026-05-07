"""T52 Cross-Sym Mixup augment + aug_a stacking.

Method B (recommended in spec): each cross-sym mixup sample produces TWO
weighted hard-label rows (label_i with weight=lam, label_j with weight=1-lam).
This avoids needing a custom KL objective and works directly with LightGBM
multiclass softmax + per-row sample weights.

Hard-constraint compliance:
  - sym used ONLY at augmentation-time to pair across-sym; never leaked into features.
  - x_mix is in same 226-d feature space as original x (no sym ID added).
  - inference path is unchanged (mixup is training-only).

Usage as module:
    from build_mixup import build_train_with_mixup
"""
from __future__ import annotations

import sys
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402


def cross_sym_mixup_pairs(
    sym: np.ndarray,
    n_aug: int,
    alpha: float,
    rng: np.random.Generator,
    max_reject_iters: int = 30,
):
    """Return (idx_i, idx_j, lam) for n_aug cross-sym pairs.

    lam is symmetrized to [0.5, 1.0] so idx_i is always the "dominant" half;
    that's only a labelling convention — Method B emits weighted rows for
    BOTH labels, so the symmetrization just makes sample weights monotone.
    """
    N = len(sym)
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
        # last-resort: fall back to a guaranteed-different sym index
        for k in np.where(bad)[0]:
            target = sym[idx_i[k]]
            cand = np.where(sym != target)[0]
            idx_j[k] = rng.choice(cand)

    lam_raw = rng.beta(alpha, alpha, size=n_aug).astype(np.float32)
    lam = np.maximum(lam_raw, 1.0 - lam_raw)  # in [0.5, 1.0]
    return idx_i, idx_j, lam


def build_mixup_method_B(
    X: np.ndarray,
    y: np.ndarray,
    sym: np.ndarray,
    n_aug: int,
    alpha: float,
    rng: np.random.Generator,
):
    """Method B: each pair -> 2 rows.
        row1: x_mix, label_i, weight = lam
        row2: x_mix, label_j, weight = 1 - lam

    Returns (X_aug2, y_aug2, w_aug2) of length 2*n_aug.
    """
    if n_aug <= 0:
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )
    idx_i, idx_j, lam = cross_sym_mixup_pairs(sym, n_aug, alpha, rng)
    lam_col = lam[:, None]
    X_mix = (lam_col * X[idx_i] + (1.0 - lam_col) * X[idx_j]).astype(np.float32)

    # Stack 2 rows per pair: (X_mix, y_i, lam) and (X_mix, y_j, 1-lam)
    X_aug2 = np.concatenate([X_mix, X_mix], axis=0)
    y_aug2 = np.concatenate(
        [y[idx_i].astype(np.int64), y[idx_j].astype(np.int64)],
        axis=0,
    )
    w_aug2 = np.concatenate(
        [lam.astype(np.float32), (1.0 - lam).astype(np.float32)],
        axis=0,
    )
    return X_aug2, y_aug2, w_aug2


def build_train_with_mixup(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sym_tr: np.ndarray,
    rng: np.random.Generator,
    use_aug_a: bool,
    aug_a_ratio: float,
    mixup_ratio: float,
    alpha: float,
    num_class: int = 3,
):
    """Build merged training dataset = [orig] + [aug_a copy] + [mixup pairs (Method B)].

    aug_a_ratio: how many aug_a rows per orig (typically 1.0)
    mixup_ratio: how many mixup PAIRS per orig (each emits 2 weighted rows)
    """
    N = len(X_tr)
    pieces_X, pieces_y, pieces_w = [], [], []

    # 1) original
    sw_orig = class_balanced_weight(y_tr, num_class)
    pieces_X.append(X_tr)
    pieces_y.append(y_tr.astype(np.int64))
    pieces_w.append(sw_orig)

    # 2) aug_a (per-(sample, feat) scale uniform[0.8, 1.2]) — applied on a
    # resample of the original
    if use_aug_a and aug_a_ratio > 0:
        n_a = int(round(aug_a_ratio * N))
        idx_a = rng.integers(0, N, size=n_a)
        X_a_src = X_tr[idx_a]
        y_a_src = y_tr[idx_a]
        X_a = aug_a_scale(X_a_src, rng)
        # weight each aug_a row by class-balance of its label, like T26 does
        # via concat-then-rebalance; here we keep weights = sw_orig[idx_a]
        # so total weight per class is roughly preserved.
        pieces_X.append(X_a)
        pieces_y.append(y_a_src.astype(np.int64))
        pieces_w.append(sw_orig[idx_a])

    # 3) cross-sym mixup pairs (Method B) — each pair -> 2 weighted rows
    if mixup_ratio > 0 and alpha > 0:
        n_pairs = int(round(mixup_ratio * N))
        t_mix = time.time()
        X_m, y_m, w_m = build_mixup_method_B(
            X_tr, y_tr, sym_tr, n_pairs, alpha, rng,
        )
        # Multiply mixup weights by class balance of dominant label
        # so rare-class samples retain their relative emphasis.
        cb = class_balanced_weight(y_m, num_class)
        w_m_scaled = (w_m * cb).astype(np.float32)
        pieces_X.append(X_m)
        pieces_y.append(y_m)
        pieces_w.append(w_m_scaled)
        print(
            f"  [mixup] n_pairs={n_pairs:,} (=> 2x rows={2*n_pairs:,}) "
            f"alpha={alpha} mix_time={time.time()-t_mix:.1f}s",
            flush=True,
        )

    X_merged = np.concatenate(pieces_X, axis=0)
    y_merged = np.concatenate(pieces_y, axis=0)
    w_merged = np.concatenate(pieces_w, axis=0)
    return X_merged, y_merged, w_merged


if __name__ == "__main__":
    # quick sanity
    rng = np.random.default_rng(0)
    N = 1000
    F = 226
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.integers(0, 3, size=N)
    sym = rng.integers(0, 5, size=N)
    Xm, ym, wm = build_train_with_mixup(
        X, y, sym, rng,
        use_aug_a=True, aug_a_ratio=1.0,
        mixup_ratio=0.5, alpha=0.4,
    )
    print(f"orig N={N}; aug_a=N; mixup pairs=0.5N -> 2*pairs rows")
    print(f"X={Xm.shape} y={ym.shape} w={wm.shape}")
    print(f"y unique: {dict(zip(*np.unique(ym, return_counts=True)))}")
    print(f"w stats: min={wm.min():.3f} max={wm.max():.3f} mean={wm.mean():.3f}")
