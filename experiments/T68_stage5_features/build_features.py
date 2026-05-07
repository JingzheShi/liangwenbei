"""T68 Fast batched feature builder (schemeP).

Pipeline per (sym, date, sess) parquet of length T:
  1) Read raw 154 columns -> (T, 154) array
  2) sliding_window_view -> (n_valid, 100, 154) float64
     where n_valid = T - 100 - max_horizon + 1, with valid_lo=99
  3) Run fast_features_batch.compute_batch_features(X3d, col_idx)
     -> (n_valid, 196) = T3-no-time(69) + S1(54) + S2(59) + S3(14)
  4) Run stage5_features.compute_stage5_batch(X3d, col_idx)
     -> (n_valid, 20)
  5) raw last-tick: X3d[:, -1, :]  -> (n_valid, 154)
  6) Concat -> (n_valid, 154 + 196 + 20 = 370)
  7) Stash labels (label_h) and meta (sym, date, sess_idx, t)

Output:
  cache/schemeP_train.npz
  cache/schemeP_val.npz
  cache/schemeP_test.npz
  cache/schemeP_feat_names.txt   (370 names; baseline + Stage 5)
  cache/schemeP_extra_feat_names.txt  (216 = 196 baseline extras + 20 stage5)

Notes:
  - This pipeline is **batch-vectorized** end-to-end. No pandas rolling
    / EWM / per-session per-window Python loops.
  - amount_delta is **NOT** sign-preserving log1p'd here (matches T53/iter_010
    cache convention -> raw value direct from parquet).
  - sym/date never used as feature; only as meta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "submission", "iter_010_t61_batchvec"))

from src.data.split import get_split  # noqa: E402
from fast_features_batch import (  # noqa: E402
    compute_batch_features,
    all_feature_names as extras_names_baseline,
)
from stage5_features import (  # noqa: E402
    compute_stage5_batch,
    stage5_feature_names,
)


WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

# Raw columns for (T, K) array. Mirrors iter_010 config.json feature list.
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


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def build_one_session(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Returns dict with X (n_valid, 370), labels y_h, mp_t_h, t."""
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return {}

    A = df[RAW_COLS].to_numpy(dtype=np.float64, copy=False)
    sw = sliding_window_view(A, (WINDOW, A.shape[1])).squeeze(1)  # (T-W+1, W, K)
    n_valid = valid_hi - valid_lo + 1
    X3d = sw[:n_valid]  # rows ending at t = valid_lo .. valid_hi

    col_idx = {c: i for i, c in enumerate(RAW_COLS)}
    # add midprice alias used by Stage 1/2/3
    if "midprice" not in col_idx:
        col_idx["midprice"] = col_idx["midprice1"]

    # 196-d extras
    extras = compute_batch_features(X3d, col_idx)
    extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)

    # 20-d stage5
    s5 = compute_stage5_batch(X3d, col_idx)
    s5 = np.where(np.isfinite(s5), s5, 0.0).astype(np.float32)

    # 154-d raw last-tick (apply amount_delta log1p sign-preserving — matches iter_010 Predictor)
    raw_last = X3d[:, -1, :].astype(np.float32)
    amt_idx = col_idx["amount_delta"]
    v = raw_last[:, amt_idx]
    raw_last[:, amt_idx] = np.sign(v) * np.log1p(np.abs(v))

    X = np.concatenate([raw_last, extras, s5], axis=1)  # (n_valid, 370)

    # midprice at t (mp_t)
    mid_idx = col_idx["midprice1"]
    mp_t = raw_last[:, mid_idx].astype(np.float32)

    out: Dict[str, np.ndarray] = {
        "X": X,
        "mp_t": mp_t,
        "t": np.arange(valid_lo, valid_hi + 1, dtype=np.int16),
    }

    # labels and mp_t+h
    for h in HORIZONS:
        col = f"label_{h}"
        if col not in df.columns:
            raise KeyError(f"{col} missing in parquet")
        # iter_010 label encoding: 0/1/2 → cache uses int8 minus 1 ? Let's check schemeC.
        # schemeC y5/y10/etc dtype int8. labels in parquet are 0,1,2 floats.
        labels = df[col].to_numpy()
        rows = np.arange(valid_lo, valid_hi + 1)
        out[f"y{h}"] = labels[rows].astype(np.int8)

        # mp_t+h: midprice at t+h
        mid_arr = df["midprice1"].to_numpy(dtype=np.float32)
        out[f"mp_t{h}"] = mid_arr[rows + h]

    return out


def build_split(sym_dates, cache_dir: str) -> Dict[str, np.ndarray]:
    Xs: List[np.ndarray] = []
    mp_ts: List[np.ndarray] = []
    syms: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    sess_idxs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    yhs: Dict[int, List[np.ndarray]] = {h: [] for h in HORIZONS}
    mph: Dict[int, List[np.ndarray]] = {h: [] for h in HORIZONS}

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        out = build_one_session(df)
        if not out:
            continue
        Xs.append(out["X"])
        mp_ts.append(out["mp_t"])
        ts.append(out["t"])
        n = out["X"].shape[0]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        for h in HORIZONS:
            yhs[h].append(out[f"y{h}"])
            mph[h].append(out[f"mp_t{h}"])

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (n_sessions - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1}/{n_sessions}] sessions  {elapsed:.1f}s  {rate:.1f} sess/s  eta {eta:.0f}s",
                  flush=True)
            progress("building", done=i + 1, total=n_sessions, sec=elapsed)

    res: Dict[str, np.ndarray] = {
        "X": np.concatenate(Xs, axis=0),
        "mp_t": np.concatenate(mp_ts, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    for h in HORIZONS:
        res[f"y{h}"] = np.concatenate(yhs[h], axis=0)
        res[f"mp_t{h}"] = np.concatenate(mph[h], axis=0)
    print(f"  total rows: {res['X'].shape[0]:,} | feats: {res['X'].shape[1]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    extras_names = list(extras_names_baseline()) + list(stage5_feature_names())
    full_names = list(RAW_COLS) + extras_names
    print(f"schemeP feature count: {len(full_names)} = "
          f"{len(RAW_COLS)} raw + {len(extras_names_baseline())} extras + "
          f"{len(stage5_feature_names())} stage5", flush=True)

    extra_path = os.path.join(args.out_dir, "schemeP_extra_feat_names.txt")
    full_path = os.path.join(args.out_dir, "schemeP_feat_names.txt")
    with open(extra_path, "w") as f:
        f.write("\n".join(extras_names) + "\n")
    with open(full_path, "w") as f:
        f.write("\n".join(full_names) + "\n")
    print(f"wrote {extra_path} & {full_path}", flush=True)

    progress("loading_split", split=args.split)
    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}")
            continue
        print(f"=== {key} ({len(split[key])} sessions) ===", flush=True)
        progress("split", split=key, n=len(split[key]))
        out = build_split(split[key], args.cache_dir)
        out_npz = os.path.join(args.out_dir, f"schemeP_{key}.npz")
        np.savez(out_npz, **out)
        print(f"  saved {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB)", flush=True)

    progress("done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
