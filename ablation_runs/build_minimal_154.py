"""Minimal 154-d raw-last cache for P1 amount_delta ablation.

Produces two caches: one with raw amount_delta, one with sign(v)*log1p(|v|).
Same valid window (last tick of 100-tick window, labels at h=60 within bounds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PARENT, "final_submission_code", "01_build_features"))
from build_schemeP_cache import RAW_COLS, get_split, WINDOW, MAX_HORIZON  # noqa: E402

HORIZONS = (5, 10, 20, 40, 60)


def build_one_session(df: pd.DataFrame, apply_log1p: bool):
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return None

    A = df[RAW_COLS].to_numpy(dtype=np.float32, copy=False)
    rows = np.arange(valid_lo, valid_hi + 1)
    raw_last = A[rows].copy()

    amt_idx = RAW_COLS.index("amount_delta")
    if apply_log1p:
        v = raw_last[:, amt_idx]
        raw_last[:, amt_idx] = np.sign(v) * np.log1p(np.abs(v))

    out = {"X": raw_last, "t": rows.astype(np.int16)}
    mid_arr = df["midprice1"].to_numpy(dtype=np.float32)
    out["mp_t"] = mid_arr[rows]
    for h in HORIZONS:
        out[f"y{h}"] = df[f"label_{h}"].to_numpy()[rows].astype(np.int8)
        out[f"mp_t{h}"] = mid_arr[rows + h]
    return out


def build_split(sym_dates, data_dir, apply_log1p):
    Xs, ts, syms, dates, sess_idxs, mp_ts = [], [], [], [], [], []
    yh = {h: [] for h in HORIZONS}
    mph = {h: [] for h in HORIZONS}
    sess_to_idx = {"am": 0, "pm": 1}
    t0 = time.time()
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 10)
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        if not os.path.isfile(path):
            continue
        df = pd.read_parquet(path)
        out = build_one_session(df, apply_log1p)
        if out is None:
            continue
        n = out["X"].shape[0]
        Xs.append(out["X"])
        ts.append(out["t"])
        mp_ts.append(out["mp_t"])
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, sess_to_idx[sess], dtype=np.int8))
        for h in HORIZONS:
            yh[h].append(out[f"y{h}"])
            mph[h].append(out[f"mp_t{h}"])
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-3)
            eta = (n_sessions - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1}/{n_sessions}] {el:.1f}s rate {rate:.1f}/s eta {eta:.0f}s",
                  flush=True)
    res = {
        "X": np.concatenate(Xs, axis=0),
        "t": np.concatenate(ts, axis=0),
        "mp_t": np.concatenate(mp_ts, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
    }
    for h in HORIZONS:
        res[f"y{h}"] = np.concatenate(yh[h], axis=0)
        res[f"mp_t{h}"] = np.concatenate(mph[h], axis=0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log1p", action="store_true",
                    help="Apply sign(v)*log1p(|v|) to amount_delta column")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "feat_names.txt"), "w") as f:
        f.write("\n".join(RAW_COLS) + "\n")

    split = get_split()
    for key in args.which.split(","):
        key = key.strip()
        if key not in split:
            continue
        print(f"\n=== Building {key} (apply_log1p={args.log1p}) ===", flush=True)
        t0 = time.time()
        out = build_split(split[key], args.data_dir, args.log1p)
        out_path = os.path.join(args.out, f"raw154_{key}.npz")
        np.savez(out_path, **out)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} ({sz:.1f} MB)  {time.time()-t0:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
