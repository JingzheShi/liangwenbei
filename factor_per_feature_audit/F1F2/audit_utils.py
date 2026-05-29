"""Shared audit utilities for per-feature audit (F1+F2 worker).

Implements PSI/KS/loaders following shared/SPEC.md exactly so worker outputs
can be merged by the PM.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import os
import time

import numpy as np
from scipy.stats import ks_2samp


CACHE_DIR = "/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p"
FEAT_NAMES_FILE = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")


# -------------------- PSI / KS --------------------

def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI on quantile bins of ref (per SPEC.md exact)."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 100 or len(cur) < 100:
        return float("nan")
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return float("nan")
    qs[0], qs[-1] = -np.inf, np.inf
    p, _ = np.histogram(ref, bins=qs)
    q, _ = np.histogram(cur, bins=qs)
    p = (p + 1e-6) / (p.sum() + 1e-6 * len(p))
    q = (q + 1e-6) / (q.sum() + 1e-6 * len(q))
    return float(((p - q) * np.log(p / q)).sum())


def ks_stat(ref: np.ndarray, cur: np.ndarray) -> float:
    """Two-sample KS distance (scipy)."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 100 or len(cur) < 100:
        return float("nan")
    try:
        return float(ks_2samp(ref, cur, mode="asymp").statistic)
    except Exception:
        return float("nan")


def verdict_from_psi(psi_max: float) -> str:
    if not np.isfinite(psi_max):
        return "insufficient_data"
    if psi_max < 0.10:
        return "stable"
    if psi_max < 0.25:
        return "mild_drift"
    return "strong_drift"


# -------------------- Cache loading --------------------

def load_feat_names() -> List[str]:
    with open(FEAT_NAMES_FILE) as f:
        return [l.strip() for l in f if l.strip()]


def load_cache_columns(split: str, col_idx: List[int]) -> Dict[str, np.ndarray]:
    """Load only specific columns of X plus sym/date.

    split: 'train' | 'val' | 'test'
    col_idx: list of int column indices to retain from X.
    Returns: {'X': (N, K) float32, 'sym': (N,), 'date': (N,)}
    """
    fp = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    z = np.load(fp, mmap_mode="r")
    # Read sliced columns into RAM
    arr = z["X"]
    # Fancy index is fast enough for our purpose; X is float32
    X_sub = np.ascontiguousarray(arr[:, col_idx])
    sym = np.ascontiguousarray(z["sym"][:])
    date = np.ascontiguousarray(z["date"][:])
    return {"X": X_sub, "sym": sym, "date": date}


# -------------------- Sampling --------------------

def stratified_sample(n_total: int, target: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample target rows out of n_total (capped at n_total)."""
    target = int(min(target, n_total))
    if target == n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=target, replace=False))


def date_bucket(date: np.ndarray, bucket_size: int = 20) -> np.ndarray:
    """Convert date 0..119 → bucket 0..5 (with bucket_size=20)."""
    return (date // bucket_size).astype(np.int8)
