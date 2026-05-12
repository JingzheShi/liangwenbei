"""Build schemeP 370-d feature cache from raw parquet files.

Usage:
    python3 build_schemeP_cache.py --data_dir /path/to/parquets --out /path/to/cache

Input:
    Raw parquet files named snapshot_sym{sym}_date{date}_{sess}.parquet
    where sym in 0..4, date in 0..119, sess in {am, pm}.
    Each parquet has 2001 rows and 154+ raw LOB columns plus label_{5,10,20,40,60}.

Output (in --out directory):
    schemeP_train.npz   (date 0-79,   ~80 days x 5 sym x 2 sess)
    schemeP_val.npz     (date 80-95,  ~16 days)
    schemeP_test.npz    (date 96-119, ~24 days)
    schemeP_feat_names.txt
    schemeP_extra_feat_names.txt

Feature dims: 154 raw last-tick + 216 extras (fast_features_batch incl. Stage5) = 370
Note: fast_features_batch.py already integrates Stage5 features (216 = 196 baseline + 20 Stage5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fast_features_batch import (
    compute_batch_features,
    all_feature_names as extras_names_baseline,
)
# Note: fast_features_batch already includes Stage5 (216 = 196 + 20 stage5).
# stage5_features.py is kept in this directory for reference but not called here.

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}
N_SYMS = 5
N_DATES = 120
SESSIONS = ("am", "pm")

RAW_COLS: List[str] = [
    "open", "high", "low", "close", "volume_delta", "amount_delta",
] + [f"bid{k}" for k in range(1, 11)] + [f"bsize{k}" for k in range(1, 11)] \
  + [f"ask{k}" for k in range(1, 11)] + [f"asize{k}" for k in range(1, 11)] \
  + ["avgbid", "avgask", "totalbsize", "totalasize",
     "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
     "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
     "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc"] \
  + [f"midprice{k}" for k in range(1, 11)] \
  + [f"spread{k}" for k in range(1, 11)] \
  + [f"bid_diff{k}" for k in range(1, 11)] \
  + [f"ask_diff{k}" for k in range(1, 11)] \
  + ["bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance"] \
  + [f"bid_rate{k}" for k in range(1, 11)] \
  + [f"ask_rate{k}" for k in range(1, 11)] \
  + [f"bsize_rate{k}" for k in range(1, 11)] \
  + [f"asize_rate{k}" for k in range(1, 11)]
assert len(RAW_COLS) == 154, f"raw col count: {len(RAW_COLS)}"


def get_split() -> Dict[str, List[Tuple[int, int, str]]]:
    keys = [
        (sym, date, sess)
        for sym in range(N_SYMS)
        for date in range(N_DATES)
        for sess in SESSIONS
    ]
    return {
        "train": [(s, d, ss) for (s, d, ss) in keys if 0 <= d < 80],
        "val":   [(s, d, ss) for (s, d, ss) in keys if 80 <= d < 96],
        "test":  [(s, d, ss) for (s, d, ss) in keys if 96 <= d < 120],
    }


def build_one_session(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return {}

    A = df[RAW_COLS].to_numpy(dtype=np.float64, copy=False)
    sw = sliding_window_view(A, (WINDOW, A.shape[1])).squeeze(1)
    n_valid = valid_hi - valid_lo + 1
    X3d = sw[:n_valid]

    col_idx = {c: i for i, c in enumerate(RAW_COLS)}
    if "midprice" not in col_idx:
        col_idx["midprice"] = col_idx["midprice1"]

    # 216-d extras (196 baseline + 20 Stage5, both included in fast_features_batch)
    extras = compute_batch_features(X3d, col_idx)
    extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)

    raw_last = X3d[:, -1, :].astype(np.float32)
    amt_idx = col_idx["amount_delta"]
    v = raw_last[:, amt_idx]
    raw_last[:, amt_idx] = np.sign(v) * np.log1p(np.abs(v))

    X = np.concatenate([raw_last, extras], axis=1)  # (n_valid, 154 + 216 = 370)

    mid_idx = col_idx["midprice1"]
    mp_t = raw_last[:, mid_idx].astype(np.float32)

    rows = np.arange(valid_lo, valid_hi + 1)
    out: Dict[str, np.ndarray] = {
        "X": X,
        "mp_t": mp_t,
        "t": np.arange(valid_lo, valid_hi + 1, dtype=np.int16),
    }

    for h in HORIZONS:
        col = f"label_{h}"
        if col not in df.columns:
            raise KeyError(f"{col} missing in parquet")
        labels = df[col].to_numpy()
        out[f"y{h}"] = labels[rows].astype(np.int8)
        mid_arr = df["midprice1"].to_numpy(dtype=np.float32)
        out[f"mp_t{h}"] = mid_arr[rows + h]

    return out


def build_split(sym_dates, data_dir: str) -> Dict[str, np.ndarray]:
    Xs: List[np.ndarray] = []
    mp_ts: List[np.ndarray] = []
    syms_list: List[np.ndarray] = []
    dates_list: List[np.ndarray] = []
    sess_idxs: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    yhs: Dict[int, List[np.ndarray]] = {h: [] for h in HORIZONS}
    mph: Dict[int, List[np.ndarray]] = {h: [] for h in HORIZONS}

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    skipped = 0
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        if not os.path.isfile(path):
            skipped += 1
            continue
        df = pd.read_parquet(path)
        out = build_one_session(df)
        if not out:
            continue
        n = out["X"].shape[0]
        Xs.append(out["X"])
        mp_ts.append(out["mp_t"])
        ts_list.append(out["t"])
        syms_list.append(np.full(n, sym, dtype=np.int8))
        dates_list.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        for h in HORIZONS:
            yhs[h].append(out[f"y{h}"])
            mph[h].append(out[f"mp_t{h}"])

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (n_sessions - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1}/{n_sessions}] {elapsed:.1f}s  {rate:.1f} sess/s  eta {eta:.0f}s",
                  flush=True)

    if skipped > 0:
        print(f"  WARNING: skipped {skipped} missing parquet files", flush=True)

    if not Xs:
        raise RuntimeError(f"No valid sessions found in {data_dir}")

    res: Dict[str, np.ndarray] = {
        "X": np.concatenate(Xs, axis=0),
        "mp_t": np.concatenate(mp_ts, axis=0),
        "sym": np.concatenate(syms_list, axis=0),
        "date": np.concatenate(dates_list, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts_list, axis=0),
    }
    for h in HORIZONS:
        res[f"y{h}"] = np.concatenate(yhs[h], axis=0)
        res[f"mp_t{h}"] = np.concatenate(mph[h], axis=0)
    print(f"  total rows: {res['X'].shape[0]:,} | feats: {res['X'].shape[1]}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Build schemeP feature cache")
    ap.add_argument("--data_dir", default="./data",
                    help="Directory with snapshot_sym*_date*_*.parquet files")
    ap.add_argument("--out", default="./outputs/cache",
                    help="Output directory for .npz cache files")
    ap.add_argument("--which", default="train,val,test",
                    help="Which splits to build (comma-separated)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    extras_names = list(extras_names_baseline())  # 216 (includes Stage5 internally)
    full_names = list(RAW_COLS) + extras_names
    print(f"schemeP feature count: {len(full_names)} = "
          f"{len(RAW_COLS)} raw + {len(extras_names)} extras (incl. Stage5)", flush=True)

    extra_path = os.path.join(args.out, "schemeP_extra_feat_names.txt")
    full_path = os.path.join(args.out, "schemeP_feat_names.txt")
    with open(extra_path, "w") as f:
        f.write("\n".join(extras_names) + "\n")
    with open(full_path, "w") as f:
        f.write("\n".join(full_names) + "\n")
    print(f"wrote feature name lists -> {args.out}", flush=True)

    split = get_split()
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}")
            continue
        print(f"\n=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        t_start = time.time()
        out = build_split(split[key], args.data_dir)
        out_npz = os.path.join(args.out, f"schemeP_{key}.npz")
        np.savez(out_npz, **out)
        sz_mb = os.path.getsize(out_npz) / 1e6
        print(f"  saved {out_npz} ({sz_mb:.1f} MB)  {time.time()-t_start:.1f}s", flush=True)

    print(f"\nDONE: cache written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
