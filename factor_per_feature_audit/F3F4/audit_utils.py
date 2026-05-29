"""Audit utilities for per-feature analysis (F3+F4 worker).

Implements:
 - PSI on quantile bins (matches SPEC.md spec exactly).
 - KS via scipy.stats.ks_2samp.
 - Cache loading helpers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


CACHE_DIR = "/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p"
FEAT_NAMES_PATH = f"{CACHE_DIR}/schemeP_feat_names.txt"


def load_feat_names():
    with open(FEAT_NAMES_PATH) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_cache(split: str, keep_cols_idx):
    """Load split cache; return dict with X[:, keep_cols_idx], sym, date."""
    path = f"{CACHE_DIR}/schemeP_{split}.npz"
    d = np.load(path)
    return {
        "X": d["X"][:, keep_cols_idx].astype(np.float32),
        "sym": d["sym"][:],
        "date": d["date"][:],
    }


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI on quantile bins of `ref`. Matches SPEC.md.

    For (near-)binary features the quantile-bin path collapses (<3 unique edges);
    we then fall back to a 2-category proportion-PSI on the unique values shared
    between ref and cur, so binary _ind features still get a finite PSI value.
    """
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 100 or len(cur) < 100:
        return float("nan")
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        # binary/near-constant fallback: PSI on proportion of positives (and the rest)
        ref_uniq = np.unique(ref)
        if len(ref_uniq) < 2:
            return float("nan")
        # use the highest unique value as the "positive" bin
        thr = float(ref_uniq[-1])
        p_pos_ref = float(np.mean(ref >= thr))
        p_pos_cur = float(np.mean(cur >= thr))
        # 2-bin PSI with smoothing
        p = np.array([1 - p_pos_ref, p_pos_ref]) + 1e-6
        q = np.array([1 - p_pos_cur, p_pos_cur]) + 1e-6
        p = p / p.sum(); q = q / q.sum()
        return float(((p - q) * np.log(p / q)).sum())
    qs[0], qs[-1] = -np.inf, np.inf
    p, _ = np.histogram(ref, bins=qs)
    q, _ = np.histogram(cur, bins=qs)
    p = (p + 1e-6) / (p.sum() + 1e-6 * len(p))
    q = (q + 1e-6) / (q.sum() + 1e-6 * len(q))
    return float(((p - q) * np.log(p / q)).sum())


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100:
        return float("nan")
    return float(ks_2samp(a, b, alternative="two-sided", mode="asymp").statistic)


def nan_pct(x: np.ndarray) -> tuple[int, float]:
    """Return (count_nan, pct_nan*100)."""
    n_nan = int(np.isnan(x).sum())
    return n_nan, 100.0 * n_nan / max(1, len(x))


def verdict(psi_max: float) -> str:
    if not np.isfinite(psi_max):
        return "insufficient_data"
    if psi_max < 0.10:
        return "stable"
    if psi_max < 0.25:
        return "mild_drift"
    return "strong_drift"


def subsample_idx(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform sub-sample indices (without replacement) of size frac*n."""
    k = max(1, int(n * frac))
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)
