"""Build flat tabular features for LightGBM from per-session parquets.

Two schemes:
  A: last-tick only -> 154 dims
  B: last-tick + rolling{mean,std,min,max} over W in {5,20,60} -> 154 + 154*3*4 = 2002 dims

Output (per split): .npz with keys
  X        (N, D) float32
  y60      (N,)   int8
  mp_t     (N,)   float32
  mp_t60   (N,)   float32
  sym      (N,)   int8
  date     (N,)   int16
  sess_idx (N,)   int8     0=am 1=pm
  t        (N,)   int16    tick index in session

For label_60 we only need midprice at t and t+60.

Strict no-leakage: rolling window at row t uses ticks [t-W+1 .. t] (inclusive of current tick).
That matches inference where Predictor sees the past 100 ticks ending at the current tick.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# Local imports (run with PYTHONPATH=workdir root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.split import get_split, get_file_path  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402

WINDOW = 100      # past ticks fed at inference (matches Predictor contract)
HORIZON = 60      # label_60
ROLL_WS = (5, 20, 60)
SESS_TO_IDX = {"am": 0, "pm": 1}


def compute_rolling_stats(X: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised rolling stats with right-closed window.

    Returns mean/std/min/max each shape (T, F). First W-1 rows = NaN.
    Window at row t uses X[t-W+1 .. t] inclusive.
    """
    T, F = X.shape
    # sliding_window_view(axis=0, window_shape=W) -> (T-W+1, F, W)
    sw = np.lib.stride_tricks.sliding_window_view(X, window_shape=W, axis=0)
    # Use float32 to keep RAM in check
    out_mean = sw.mean(axis=-1, dtype=np.float32)
    out_std = sw.std(axis=-1, dtype=np.float32)
    out_min = sw.min(axis=-1).astype(np.float32, copy=False)
    out_max = sw.max(axis=-1).astype(np.float32, copy=False)
    pad = np.full((W - 1, F), np.nan, dtype=np.float32)
    return (
        np.concatenate([pad, out_mean]),
        np.concatenate([pad, out_std]),
        np.concatenate([pad, out_min]),
        np.concatenate([pad, out_max]),
    )


def build_features_one_session(
    df: pd.DataFrame, feat_cols: List[str], scheme: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_valid, y60, mp_t, mp_t60, t_valid).

    valid t range: [WINDOW-1, T-1-HORIZON] inclusive.
    """
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - HORIZON  # inclusive
    if valid_hi < valid_lo:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int16),
        )

    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)  # (T, F)
    # log1p amount_delta (large magnitude); preserve sign in case of negatives
    if "amount_delta" in feat_cols:
        idx = feat_cols.index("amount_delta")
        col = X[:, idx]
        X = X.copy()
        X[:, idx] = np.sign(col) * np.log1p(np.abs(col))

    midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
    y60 = df["label_60"].to_numpy(dtype=np.int8, copy=False)

    if scheme == "A":
        feats_full = X  # (T, F)
    elif scheme == "B":
        parts = [X]
        for W in ROLL_WS:
            m, s, mn, mx = compute_rolling_stats(X, W)
            parts.extend([m, s, mn, mx])
        feats_full = np.concatenate(parts, axis=1)  # (T, F * (1 + 3*4)) = (T, F*13)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")

    sl = slice(valid_lo, valid_hi + 1)
    X_valid = feats_full[sl]
    y_valid = y60[sl]
    mp_t = midprice[sl]
    mp_t60 = midprice[valid_lo + HORIZON : valid_hi + 1 + HORIZON]
    t_valid = np.arange(valid_lo, valid_hi + 1, dtype=np.int16)

    return X_valid, y_valid, mp_t, mp_t60, t_valid


def feature_names(feat_cols: Sequence[str], scheme: str) -> List[str]:
    if scheme == "A":
        return list(feat_cols)
    out = list(feat_cols)
    for W in ROLL_WS:
        for stat in ("mean", "std", "min", "max"):
            out.extend([f"{c}__roll{W}_{stat}" for c in feat_cols])
    return out


def build_split(
    sym_dates, feat_cols: List[str], scheme: str, data_dir: str
) -> dict:
    """Process all sessions for one split. Concatenate into big arrays."""
    Xs, ys, mts, mt60s = [], [], [], []
    syms, dates, sess_idxs, ts = [], [], [], []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = get_file_path(sym, date, sess, data_dir=data_dir)
        df = pd.read_parquet(path)
        X, y60, mp_t, mp_t60, t_valid = build_features_one_session(df, feat_cols, scheme)
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
            print(f"  [{i+1}/{n_sessions}] sessions processed, elapsed {elapsed:.1f}s", flush=True)

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
        f"  -> X shape {out['X'].shape} dtype {out['X'].dtype} mem {out['X'].nbytes/1e9:.2f}GB",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("A", "B"), required=True)
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="experiments/T2_gbdt_lgbm/cache")
    ap.add_argument("--which", default="train,val,test", help="comma-separated subset of split keys")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat_cols = get_default_feature_cols()
    feat_names = feature_names(feat_cols, args.scheme)
    print(f"Scheme {args.scheme}: {len(feat_names)} features per sample", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], feat_cols, args.scheme, args.data_dir)
        out_path = os.path.join(args.out_dir, f"scheme{args.scheme}_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    # Save feature names alongside
    names_path = os.path.join(args.out_dir, f"scheme{args.scheme}_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_names:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
