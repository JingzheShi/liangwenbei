"""T7 Scheme D1: window z-score normalization features (sym-agnostic adaptive).

Per-tick, per-feature normalization using the past-100-tick window:
    mean100[t] = mean(X[t-99..t])
    std100[t]  = std(X[t-99..t]) + 1e-8
    zscore[t]  = (X[t] - mean100[t]) / std100[t]

D1 final feature vector at t = raw[t] (154) + zscore[t] (154) = 308 dims.

Why both:
- raw last-tick keeps absolute level info ("price is at 123, not 0")
- zscore last-tick gives sym-agnostic adaptive signal ("current value is +1.3 std
  above local mean") — does not depend on sym scale, robust to OOD syms.

Strict no-leakage: window at row t uses ticks [t-99..t] inclusive (matches
inference contract: Predictor sees the past 100 ticks ending at the current tick).

amount_delta is log1p'd (sign-preserving) before z-score because raw scale e3-e6.

Output (per split): .npz with keys
  X        (N, 308) float32
  y60      (N,)     int8
  mp_t     (N,)     float32
  mp_t60   (N,)     float32
  sym      (N,)     int8
  date     (N,)     int16
  sess_idx (N,)     int8     0=am 1=pm
  t        (N,)     int16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split, get_file_path  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402

WINDOW = 100
HORIZON = 60
EPS = 1e-8
SESS_TO_IDX = {"am": 0, "pm": 1}


def compute_rolling_mean_std(X: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling mean & std over right-closed window of size W.

    X: (T, F) float32. Window at row t uses X[t-W+1..t] inclusive.
    Returns (mean, std) each (T, F); first W-1 rows = NaN.
    """
    sw = np.lib.stride_tricks.sliding_window_view(X, window_shape=W, axis=0)  # (T-W+1, F, W)
    out_mean = sw.mean(axis=-1, dtype=np.float32)
    out_std = sw.std(axis=-1, dtype=np.float32)
    pad = np.full((W - 1, X.shape[1]), np.nan, dtype=np.float32)
    return (
        np.concatenate([pad, out_mean], axis=0),
        np.concatenate([pad, out_std], axis=0),
    )


def build_features_one_session(
    df: pd.DataFrame, feat_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_valid, y60, mp_t, mp_t60, t_valid)."""
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - HORIZON
    if valid_hi < valid_lo:
        F2 = 2 * len(feat_cols)
        return (
            np.zeros((0, F2), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int16),
        )

    X = df[feat_cols].to_numpy(dtype=np.float32, copy=True)  # (T, F)
    if "amount_delta" in feat_cols:
        idx = feat_cols.index("amount_delta")
        col = X[:, idx]
        X[:, idx] = np.sign(col) * np.log1p(np.abs(col))

    midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
    y60 = df["label_60"].to_numpy(dtype=np.int8, copy=False)

    # Rolling mean/std over WINDOW=100 ticks (right-closed, includes current tick).
    mean100, std100 = compute_rolling_mean_std(X, WINDOW)  # (T, F), (T, F)
    zscore = (X - mean100) / (std100 + EPS)  # (T, F)

    feats_full = np.concatenate([X, zscore], axis=1)  # (T, 2F)

    sl = slice(valid_lo, valid_hi + 1)
    X_valid = feats_full[sl]
    y_valid = y60[sl]
    mp_t = midprice[sl]
    mp_t60 = midprice[valid_lo + HORIZON : valid_hi + 1 + HORIZON]
    t_valid = np.arange(valid_lo, valid_hi + 1, dtype=np.int16)

    # Replace any residual nan (shouldn't happen since valid_lo >= W-1)
    if not np.isfinite(X_valid).all():
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    return X_valid, y_valid, mp_t, mp_t60, t_valid


def feature_names(feat_cols: Sequence[str]) -> List[str]:
    out = list(feat_cols)
    out.extend([f"{c}__zscore100" for c in feat_cols])
    return out


def build_split(sym_dates, feat_cols: List[str], data_dir: str) -> dict:
    Xs, ys, mts, mt60s = [], [], [], []
    syms, dates, sess_idxs, ts = [], [], [], []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = get_file_path(sym, date, sess, data_dir=data_dir)
        df = pd.read_parquet(path)
        X, y60, mp_t, mp_t60, t_valid = build_features_one_session(df, feat_cols)
        n = len(X)
        Xs.append(X)
        ys.append(y60)
        mts.append(mp_t)
        mt60s.append(mp_t60)
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(t_valid)
        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_sessions}] sessions, elapsed {elapsed:.1f}s", flush=True)

    out = {
        "X": np.concatenate(Xs, axis=0),
        "y60": np.concatenate(ys, axis=0),
        "mp_t": np.concatenate(mts, axis=0),
        "mp_t60": np.concatenate(mt60s, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    print(
        f"  -> X shape {out['X'].shape} dtype {out['X'].dtype} "
        f"mem {out['X'].nbytes/1e9:.2f}GB",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="experiments/T7_window_zscore/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat_cols = get_default_feature_cols()
    feat_names = feature_names(feat_cols)
    print(f"Scheme D1: {len(feat_names)} features per sample (raw + zscore100)", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], feat_cols, args.data_dir)
        out_path = os.path.join(args.out_dir, f"schemeD1_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeD1_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_names:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
