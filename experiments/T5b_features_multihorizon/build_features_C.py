"""Build Scheme C1 last-tick features by reading T3 features_v1 cache.

Scheme C1 = 154 raw last-tick + 72 T3 last-tick = 226 features.

For each session, valid t range = [WINDOW-1, T-1-MAX_HORIZON] (so all 5 horizons valid).

Output (per split): .npz with keys
  X         (N, 226)  float32
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
import json
import os
import sys
import time
from typing import List

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
SESS_TO_IDX = {"am": 0, "pm": 1}


def get_scheme_c_columns() -> List[str]:
    raw = get_default_feature_cols()
    t3 = feature_v1_columns()
    return raw + t3


def build_split(sym_dates, feat_cols: List[str], cache_dir: str) -> dict:
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
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - MAX_HORIZON
        if valid_hi < valid_lo:
            continue

        sl = slice(valid_lo, valid_hi + 1)
        X_sl = df[feat_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)

        # log1p amount_delta (preserve sign)
        if "amount_delta" in feat_cols:
            j = feat_cols.index("amount_delta")
            col = X_sl[:, j].copy()
            X_sl = X_sl.copy()
            X_sl[:, j] = np.sign(col) * np.log1p(np.abs(col))

        Xs.append(X_sl)
        midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
        mp_t.append(midprice[sl])

        for h in HORIZONS:
            y = df[f"label_{h}"].iloc[sl].to_numpy(dtype=np.int8, copy=False)
            ys[h].append(y)
            mp_h = midprice[valid_lo + h : valid_hi + 1 + h]
            mps[f"mp_t{h}"].append(mp_h)

        n = len(X_sl)
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default="data/features_v1")
    ap.add_argument("--out-dir", default="experiments/T5b_features_multihorizon/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat_cols = get_scheme_c_columns()
    print(f"Scheme C1: {len(feat_cols)} features (154 raw + 72 T3)", flush=True)
    # NaN check on first session
    df0 = pd.read_parquet(os.path.join(args.cache_dir, "snapshot_sym0_date0_am.parquet"))
    has_nan = df0[feat_cols].iloc[WINDOW - 1 :].isna().any().any()
    print(f"  first session NaN check after WINDOW-1: {has_nan}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], feat_cols, args.cache_dir)
        out_path = os.path.join(args.out_dir, f"schemeC_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeC_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_cols:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
