"""Per-feature audit utilities (shared across F5F6 worker).

Implements:
  - psi(ref, cur, n_bins=10)  — population stability index on ref quantiles
  - ks_stat(a, b)             — KS 2-sample test statistic (uses scipy if available else fallback)
  - load_cache(path, cols)    — load .npz, keep only X[:, cols] + sym + date
  - safe_sample(arr, n)       — uniform downsample helper
  - feature_psi_cross_sym(x, sym, ref_idx, n_bins)
  - feature_psi_cross_date_buckets(x_tr, x_val, x_te, date_tr, ref_idx, n_buckets)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence

try:
    from scipy.stats import ks_2samp as _ks  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI on quantile bins of ref. NaN-safe; returns nan if constant or too small."""
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


def _ks_fallback(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    if HAS_SCIPY:
        try:
            res = _ks(a, b)
            return float(res.statistic)
        except Exception:
            return _ks_fallback(a, b)
    return _ks_fallback(a, b)


def safe_sample(arr: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(arr) <= n:
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def load_cache_subset(path: str, col_indices: Sequence[int]):
    """Load .npz and return (X_sub, sym, date) where X_sub = X[:, col_indices] (float32)."""
    z = np.load(path)
    col_indices = np.asarray(col_indices, dtype=np.int64)
    X_sub = np.ascontiguousarray(z["X"][:, col_indices])
    sym = z["sym"][:]
    date = z["date"][:]
    return X_sub, sym, date


def verdict_psi(psi_max: float) -> str:
    if not np.isfinite(psi_max):
        return "insufficient_data"
    if psi_max < 0.10:
        return "stable"
    if psi_max < 0.25:
        return "mild_drift"
    return "strong_drift"


def nan_pct(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean(~np.isfinite(x)) * 100.0)


def cross_sym_stats(x: np.ndarray, sym: np.ndarray,
                    ref_sample_n: int = 200_000,
                    per_sym_sample_n: int = 80_000,
                    n_bins: int = 10,
                    rng: Optional[np.random.Generator] = None):
    """For one feature column x (1D) and sym (1D int): compute psi/ks per sym.

    Returns dict with by_sym list, max, mean (for psi) and similar for ks.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    syms = np.unique(sym)
    # Build ref by sampling globally
    ref = safe_sample(x, ref_sample_n, rng)
    finite_total = np.isfinite(ref).sum()
    if finite_total < 1000:
        return {
            "psi_by_sym": [float("nan")] * len(syms),
            "psi_max": float("nan"),
            "psi_mean": float("nan"),
            "ks_by_sym": [float("nan")] * len(syms),
            "ks_max": float("nan"),
            "note": "insufficient_finite_data",
        }
    psi_list, ks_list = [], []
    for s in syms:
        cur_all = x[sym == s]
        cur = safe_sample(cur_all, per_sym_sample_n, rng)
        psi_list.append(psi(ref, cur, n_bins=n_bins))
        ks_list.append(ks_stat(ref, cur))
    psi_arr = np.array(psi_list, dtype=float)
    ks_arr = np.array(ks_list, dtype=float)
    return {
        "psi_by_sym": [None if not np.isfinite(v) else float(v) for v in psi_arr],
        "psi_max": float(np.nanmax(psi_arr)) if np.any(np.isfinite(psi_arr)) else float("nan"),
        "psi_mean": float(np.nanmean(psi_arr)) if np.any(np.isfinite(psi_arr)) else float("nan"),
        "ks_by_sym": [None if not np.isfinite(v) else float(v) for v in ks_arr],
        "ks_max": float(np.nanmax(ks_arr)) if np.any(np.isfinite(ks_arr)) else float("nan"),
    }


def cross_date_stats(x_tr: np.ndarray, x_val: np.ndarray, x_te: np.ndarray,
                     date_tr: np.ndarray,
                     ref_sample_n: int = 200_000,
                     bucket_sample_n: int = 50_000,
                     n_buckets: int = 6,
                     n_bins: int = 10,
                     rng: Optional[np.random.Generator] = None):
    """Cross-date PSI/KS: train→val, train→test, plus per-bucket PSI on train (date 0-79 in 6 buckets).

    We additionally split date range [0,119] into 6 buckets of ~20 days each, where
    buckets 0-3 cover train (0-79) and buckets 4-5 cover val+test (80-119). Bucket PSI
    is psi(ref, bucket-of-x_tr) for buckets in train, psi(ref, bucket-of-(val+test)) for later.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    ref = safe_sample(x_tr, ref_sample_n, rng)
    finite_total = np.isfinite(ref).sum()
    if finite_total < 1000:
        return {
            "psi_train_vs_val": float("nan"),
            "psi_train_vs_test": float("nan"),
            "psi_by_bucket": [float("nan")] * n_buckets,
            "ks_train_vs_val": float("nan"),
            "ks_train_vs_test": float("nan"),
            "note": "insufficient_finite_data",
        }
    cur_val = safe_sample(x_val, ref_sample_n, rng)
    cur_te = safe_sample(x_te, ref_sample_n, rng)
    psi_vv = psi(ref, cur_val, n_bins=n_bins)
    psi_vt = psi(ref, cur_te, n_bins=n_bins)
    ks_vv = ks_stat(ref, cur_val)
    ks_vt = ks_stat(ref, cur_te)

    # Per-bucket: split full [0,119] domain in 6 equal buckets of 20 days each.
    bucket_psis: List[float] = []
    for bi in range(n_buckets):
        lo, hi = bi * 20, (bi + 1) * 20  # [lo, hi)
        if hi <= 80:
            mask = (date_tr >= lo) & (date_tr < hi)
            slice_x = x_tr[mask]
        else:
            # Approximate: for buckets covering val/test, sample from the corresponding split
            # bucket 4 (80-99): val (80-95) + first 4 days of test (96-99) → just use val
            # bucket 5 (100-119): rest of test (100-119)
            # We don't have date arrays for val/test here; approximate by using full split.
            if bi == 4:
                slice_x = x_val
            else:
                slice_x = x_te
        slice_x = safe_sample(slice_x, bucket_sample_n, rng)
        p = psi(ref, slice_x, n_bins=n_bins)
        bucket_psis.append(p)
    bucket_psis_clean = [None if not np.isfinite(v) else float(v) for v in bucket_psis]
    return {
        "psi_train_vs_val": None if not np.isfinite(psi_vv) else float(psi_vv),
        "psi_train_vs_test": None if not np.isfinite(psi_vt) else float(psi_vt),
        "psi_by_bucket": bucket_psis_clean,
        "ks_train_vs_val": None if not np.isfinite(ks_vv) else float(ks_vv),
        "ks_train_vs_test": None if not np.isfinite(ks_vt) else float(ks_vt),
    }
