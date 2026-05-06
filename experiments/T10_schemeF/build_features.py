"""Build Scheme F last-tick features from features_v1 cache.

Scheme F = 154 raw last-tick + 154 window-zscore (T7 D1 style) + 72 T3 last-tick
        = 380 features per sample.

For each session, valid t range = [WINDOW-1, T-1-MAX_HORIZON] (so all 5 horizons valid).

Z-score computation (sym-agnostic):
  For each tick t at the valid range, compute mean100[t]/std100[t] over the
  preceding 100-tick window of the 154 raw features. amount_delta is log1p'd
  (sign-preserving) before z-score because raw scale e3-e6.

Output (per split): .npz with keys
  X         (N, 380)  float32
  y5,y10,y20,y40,y60    (N,) int8
  mp_t      (N,)       float32
  mp_t5,mp_t10,mp_t20,mp_t40,mp_t60   (N,) float32
  sym       (N,)       int8
  date      (N,)       int16
  sess_idx  (N,)       int8     0=am 1=pm
  t         (N,)       int16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "T3_features_v1"))

from src.data.split import get_split  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402
from compute import feature_v1_columns  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
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


def build_session(
    df: pd.DataFrame,
    raw_cols: List[str],
    t3_cols: List[str],
    amount_idx: int,
):
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return None

    # Raw 154 — full session for z-score, log1p amount_delta first
    raw_full = df[raw_cols].to_numpy(dtype=np.float32, copy=True)  # (T, 154)
    if amount_idx >= 0:
        col = raw_full[:, amount_idx]
        raw_full[:, amount_idx] = np.sign(col) * np.log1p(np.abs(col))

    # Z-score against 100-tick window
    mean100, std100 = compute_rolling_mean_std(raw_full, WINDOW)
    zscore_full = (raw_full - mean100) / (std100 + EPS)  # (T, 154)

    # T3 72 features (already cached in df)
    t3_full = df[t3_cols].to_numpy(dtype=np.float32, copy=False)  # (T, 72)

    sl = slice(valid_lo, valid_hi + 1)
    raw_last = raw_full[sl]    # (N, 154)
    z_last = zscore_full[sl]   # (N, 154)
    t3_last = t3_full[sl]      # (N, 72)
    X = np.concatenate([raw_last, z_last, t3_last], axis=1)  # (N, 380)

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
    mp_t = midprice[sl]
    out = {
        "X": X,
        "mp_t": mp_t,
        "t": np.arange(valid_lo, valid_hi + 1, dtype=np.int16),
    }
    for h in HORIZONS:
        y = df[f"label_{h}"].iloc[sl].to_numpy(dtype=np.int8, copy=False)
        out[f"y{h}"] = y
        mp_h = midprice[valid_lo + h : valid_hi + 1 + h]
        out[f"mp_t{h}"] = mp_h
    return out


def build_split(sym_dates, raw_cols, t3_cols, cache_dir, amount_idx) -> dict:
    Xs = []
    ys = {h: [] for h in HORIZONS}
    mps = {f"mp_t{h}": [] for h in HORIZONS}
    mp_t = []
    syms, dates, sess_idxs, ts = [], [], [], []

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()

    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        out = build_session(df, raw_cols, t3_cols, amount_idx)
        if out is None:
            continue
        Xs.append(out["X"])
        mp_t.append(out["mp_t"])
        for h in HORIZONS:
            ys[h].append(out[f"y{h}"])
            mps[f"mp_t{h}"].append(out[f"mp_t{h}"])
        n = len(out["X"])
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(out["t"])

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_sessions}] sessions; elapsed {elapsed:.1f}s", flush=True)

    out = {
        "X": np.concatenate(Xs, axis=0),
        "mp_t": np.concatenate(mp_t, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    for h in HORIZONS:
        out[f"y{h}"] = np.concatenate(ys[h], axis=0)
        out[f"mp_t{h}"] = np.concatenate(mps[f"mp_t{h}"], axis=0)
    print(f"  -> X shape {out['X'].shape} mem {out['X'].nbytes/1e9:.2f}GB", flush=True)
    return out


def feature_names(raw_cols: List[str], t3_cols: List[str]) -> List[str]:
    out = list(raw_cols)
    out.extend([f"{c}__zscore100" for c in raw_cols])
    out.extend(t3_cols)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default="data/features_v1")
    ap.add_argument("--out-dir", default="experiments/T10_schemeF/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_cols = get_default_feature_cols()
    t3_cols = feature_v1_columns()
    feat_names = feature_names(raw_cols, t3_cols)
    print(f"Scheme F: {len(feat_names)} features = "
          f"{len(raw_cols)} raw + {len(raw_cols)} zscore + {len(t3_cols)} T3", flush=True)
    assert "date" not in feat_names and "sym" not in feat_names and "time" not in feat_names

    amount_idx = raw_cols.index("amount_delta") if "amount_delta" in raw_cols else -1
    print(f"  amount_delta idx in raw: {amount_idx}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], raw_cols, t3_cols, args.cache_dir, amount_idx)
        out_path = os.path.join(args.out_dir, f"schemeF_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeF_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_names:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
