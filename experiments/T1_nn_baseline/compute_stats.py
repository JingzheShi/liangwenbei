"""Compute per-feature mean/std on the train split for z-score normalization.

amount_delta is log1p-transformed before stats (per task spec).
Saves stats.npz in this directory.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.dataset import get_default_feature_cols  # noqa: E402
from src.data.split import get_file_path, get_split  # noqa: E402

LOG1P_COLS = ("amount_delta",)


def main():
    feature_cols = get_default_feature_cols()
    print(f"[stats] {len(feature_cols)} feature columns")
    log1p_idx = np.array([feature_cols.index(c) for c in LOG1P_COLS], dtype=np.int64)

    splits = get_split(mode="local_debug")
    train_keys = splits["train"]
    print(f"[stats] {len(train_keys)} train sessions")

    n_per_feat = np.zeros(len(feature_cols), dtype=np.int64)
    s = np.zeros(len(feature_cols), dtype=np.float64)
    ssq = np.zeros(len(feature_cols), dtype=np.float64)
    nan_session_count = 0
    inf_session_count = 0
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(train_keys):
        path = get_file_path(sym, date, sess, data_dir=os.path.join(ROOT, "data"))
        df = pd.read_parquet(path, columns=list(feature_cols))
        x = df.to_numpy(dtype=np.float64, copy=False)  # (T, F)
        # log1p on selected cols
        x[:, log1p_idx] = np.log1p(x[:, log1p_idx])
        # NaN-aware accumulation (some sessions have missing deep ask levels)
        if np.isnan(x).any():
            nan_session_count += 1
        if np.isinf(x).any():
            inf_session_count += 1
        # Treat inf as nan for stats safety
        x = np.where(np.isfinite(x), x, np.nan)
        valid = ~np.isnan(x)
        x_zero = np.where(valid, x, 0.0)
        s += x_zero.sum(axis=0)
        ssq += (x_zero * x_zero).sum(axis=0)
        n_per_feat += valid.sum(axis=0)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_keys)}] elapsed={time.time()-t0:.1f}s")

    n_safe = np.maximum(n_per_feat, 1)
    mean = s / n_safe
    var = ssq / n_safe - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)
    n_total = int(n_per_feat.max())

    print(f"[stats] n_samples={n_total}, nan_sessions={nan_session_count}, inf_sessions={inf_session_count}, time={time.time()-t0:.1f}s")
    print(f"[stats] min n_per_feat={n_per_feat.min()}, max n_per_feat={n_per_feat.max()}")
    print(f"[stats] mean range: [{mean.min():.4g}, {mean.max():.4g}]")
    print(f"[stats] std  range: [{std.min():.4g}, {std.max():.4g}]")
    print(f"[stats] amount_delta (log1p): mean={mean[log1p_idx[0]]:.4g}, std={std[log1p_idx[0]]:.4g}")

    out_path = os.path.join(HERE, "stats.npz")
    np.savez(
        out_path,
        feature_cols=np.array(feature_cols, dtype=object),
        log1p_cols=np.array(LOG1P_COLS, dtype=object),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        n_samples=n_total,
    )
    print(f"[stats] wrote {out_path}")


if __name__ == "__main__":
    main()
