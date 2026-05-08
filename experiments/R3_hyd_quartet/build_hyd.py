"""R3 HYD Interaction Quartet — 16-dim within-window microstructure interactions.

Per (sym, date, sess) parquet, mirror T68/build_features.py windowing:
  X3d shape (n_valid, 100, 154) — sliding windows of 100 ticks
For each tick we compute 4 base interaction signals:
  pp  = (bsize1 - asize1) * (ask1 - bid1)             # price_pressure
  mu  = (ask1 - bid1) * liq_imb                        # market_urgency
        liq_imb = (totalbsize - totalasize) / (totalbsize + totalasize + eps)
  dp  = (totalasize - totalbsize) * (avgask - ask1)    # depth_pressure (depth × shape)
  sdr = (ask1 - bid1) / (totalbsize + totalasize + eps)# spread/depth ratio

Per-window summary (16 dims):
  4 last-tick (W=1) values
  4 mean over last W=5 ticks
  4 mean over last W=20 ticks
  4 mean over last W=50 ticks

Output: cache/hyd_{train,val,test}.npy float32 (N, 16)
Order: identical to T68 schemeP_*.npz X array (same get_split iteration).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from src.data.split import get_split  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

RAW_COLS = [
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
assert len(RAW_COLS) == 154

I_BID1 = RAW_COLS.index("bid1")
I_BSIZE1 = RAW_COLS.index("bsize1")
I_ASK1 = RAW_COLS.index("ask1")
I_ASIZE1 = RAW_COLS.index("asize1")
I_AVGASK = RAW_COLS.index("avgask")
I_TOTBS = RAW_COLS.index("totalbsize")
I_TOTAS = RAW_COLS.index("totalasize")

EPS = 1e-8

HYD_NAMES = [
    "hyd_pp_last", "hyd_mu_last", "hyd_dp_last", "hyd_sdr_last",
    "hyd_pp_W5",  "hyd_mu_W5",  "hyd_dp_W5",  "hyd_sdr_W5",
    "hyd_pp_W20", "hyd_mu_W20", "hyd_dp_W20", "hyd_sdr_W20",
    "hyd_pp_W50", "hyd_mu_W50", "hyd_dp_W50", "hyd_sdr_W50",
]
assert len(HYD_NAMES) == 16


def hyd_features(X3d: np.ndarray) -> np.ndarray:
    """X3d: (n_valid, 100, 154) float64. Returns (n_valid, 16) float32."""
    bid1 = X3d[:, :, I_BID1]
    ask1 = X3d[:, :, I_ASK1]
    bsize1 = X3d[:, :, I_BSIZE1]
    asize1 = X3d[:, :, I_ASIZE1]
    avgask = X3d[:, :, I_AVGASK]
    totbs = X3d[:, :, I_TOTBS]
    totas = X3d[:, :, I_TOTAS]

    spread1 = ask1 - bid1
    imb_size = bsize1 - asize1
    depth_sum = totbs + totas + EPS
    liq_imb = (totbs - totas) / depth_sum
    ask_slope = avgask - ask1
    depth_diff = totas - totbs

    pp = imb_size * spread1
    mu = spread1 * liq_imb
    dp = depth_diff * ask_slope
    sdr = spread1 / depth_sum

    n = X3d.shape[0]
    out = np.empty((n, 16), dtype=np.float32)
    out[:, 0] = pp[:, -1]
    out[:, 1] = mu[:, -1]
    out[:, 2] = dp[:, -1]
    out[:, 3] = sdr[:, -1]

    for j, W in enumerate((5, 20, 50), start=1):
        col = j * 4
        out[:, col + 0] = pp[:, -W:].mean(axis=1)
        out[:, col + 1] = mu[:, -W:].mean(axis=1)
        out[:, col + 2] = dp[:, -W:].mean(axis=1)
        out[:, col + 3] = sdr[:, -W:].mean(axis=1)

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def build_one_session(df: pd.DataFrame) -> np.ndarray | None:
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return None
    A = df[RAW_COLS].to_numpy(dtype=np.float64, copy=False)
    sw = sliding_window_view(A, (WINDOW, A.shape[1])).squeeze(1)
    n_valid = valid_hi - valid_lo + 1
    X3d = sw[:n_valid]
    return hyd_features(X3d)


def build_split(sym_dates, raw_dir):
    out_list = []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 10)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(raw_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        out = build_one_session(df)
        if out is None:
            continue
        out_list.append(out)
        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-3)
            print(f"  [{i+1}/{n_sessions}] {el:.1f}s {rate:.1f} sess/s", flush=True)
    X = np.concatenate(out_list, axis=0)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "hyd_feat_names.txt"), "w") as f:
        f.write("\n".join(HYD_NAMES) + "\n")

    split = get_split(args.split)
    for key in args.which.split(","):
        key = key.strip()
        if key not in split:
            continue
        print(f"=== {key} ({len(split[key])} sessions) ===", flush=True)
        X = build_split(split[key], args.raw_dir)
        out_path = os.path.join(args.out_dir, f"hyd_{key}.npy")
        np.save(out_path, X.astype(np.float32))
        print(f"  saved {out_path}  shape={X.shape}  size={os.path.getsize(out_path)/1e6:.1f}MB",
              flush=True)


if __name__ == "__main__":
    main()
