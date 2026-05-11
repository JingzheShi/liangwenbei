"""T191: build 5 trend / signed-momentum features per (sym,date,sess).

Output: cache/trend_feat_{train,val,test}.npz with key 'T' of shape (N, 5)
aligned 1:1 with the existing schemeP_*.npz rows (same iteration order).

Features (causal, last-tick aligned), computed on midprice1:
  0: mean_logret_W10   - mean of log(mid[t]/mid[t-1]) over last 10 ticks
  1: mean_logret_W20   - same, last 20 ticks
  2: mean_logret_W50   - last 50 ticks
  3: mean_logret_W100  - last 100 ticks (full window)
  4: mid_pct_change_W100 - (mid[t] - mid[t-99]) / mid[t-99]

All pure-NumPy vectorized; no per-window Python loops.
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

FEAT_NAMES = [
    "mean_logret_W10",
    "mean_logret_W20",
    "mean_logret_W50",
    "mean_logret_W100",
    "mid_pct_change_W100",
]


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


def compute_trend_one_session(mid: np.ndarray) -> np.ndarray:
    """mid : (T,) float64 midprice1 series. Returns (n_valid, 5).

    n_valid windows end at t = 99 .. T-1-MAX_HORIZON. We need full last-100 lookback,
    so valid_lo = 99. To match build_features.py: rows ending at t in
    [valid_lo, valid_hi] where valid_lo=99 and valid_hi=T-1-MAX_HORIZON.
    """
    T = len(mid)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return np.empty((0, 5), dtype=np.float32)
    n_valid = valid_hi - valid_lo + 1

    # midprice1 in raw parquet is already (mid - 1)? Let me check:
    # The parquet returns midprice1 dtype float64 with values ~-0.005 — so it is normalized
    # (centered around 0). The actual price = midprice1 + 1.0 (matches the fee formula
    # which uses (mp + 1.0) as denominator).
    # For log return we need actual prices, so use mid + 1.0
    mid_real = (mid + 1.0).astype(np.float64)
    # log return per tick: log(mid_real[t] / mid_real[t-1])
    logret = np.zeros(T, dtype=np.float64)
    logret[1:] = np.log(mid_real[1:] / mid_real[:-1])

    # Build sliding windows over logret: (T-W+1, W)
    sw_lr = sliding_window_view(logret, WINDOW)  # (T-W+1, W)
    sw_lr = sw_lr[:n_valid]  # only rows ending in valid range

    # mean over last K ticks within the W=100 window. Last K means columns [-K:].
    mean_lr_W10 = sw_lr[:, -10:].mean(axis=1)
    mean_lr_W20 = sw_lr[:, -20:].mean(axis=1)
    mean_lr_W50 = sw_lr[:, -50:].mean(axis=1)
    mean_lr_W100 = sw_lr.mean(axis=1)

    # mid_pct_change_W100: (mid[t] - mid[t-99]) / mid[t-99]   (use real price + 1.0)
    sw_mid = sliding_window_view(mid_real, WINDOW)  # (T-W+1, W)
    sw_mid = sw_mid[:n_valid]
    mid_now = sw_mid[:, -1]
    mid_99 = sw_mid[:, 0]
    pct_W100 = (mid_now - mid_99) / mid_99

    out = np.column_stack(
        [mean_lr_W10, mean_lr_W20, mean_lr_W50, mean_lr_W100, pct_W100]
    ).astype(np.float32)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def build_split(sym_dates: List[Tuple[int, int, str]], data_dir: str) -> np.ndarray:
    arrs: List[np.ndarray] = []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path, columns=["midprice1"])
        mid = df["midprice1"].to_numpy(dtype=np.float64, copy=False)
        T_feat = compute_trend_one_session(mid)
        arrs.append(T_feat)
        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (n_sessions - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1}/{n_sessions}] {elapsed:.1f}s  {rate:.1f} sess/s  eta {eta:.0f}s",
                  flush=True)
    return np.concatenate(arrs, axis=0)


def verify_alignment(split_key: str, T_arr: np.ndarray) -> None:
    """Sanity: row count must equal existing schemeP_{split}.npz row count."""
    sp_path = os.path.join(
        ROOT, "experiments", "T68_stage5_features", "cache", f"schemeP_{split_key}.npz"
    )
    d = np.load(sp_path)
    n_sp = len(d["mp_t"])
    d.close()
    assert len(T_arr) == n_sp, (
        f"alignment mismatch for {split_key}: T={len(T_arr)} vs schemeP={n_sp}"
    )
    print(f"  OK: {split_key} row count matches schemeP ({n_sp:,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Save feature names
    with open(os.path.join(args.out_dir, "trend_feat_names.txt"), "w") as f:
        f.write("\n".join(FEAT_NAMES) + "\n")

    progress("loading_split", split=args.split)
    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]

    for key in wanted:
        if key not in split:
            print(f"  skip unknown {key}", flush=True)
            continue
        print(f"=== {key} ({len(split[key])} sessions) ===", flush=True)
        progress("split", split=key, n=len(split[key]))
        T_arr = build_split(split[key], args.data_dir)
        print(f"  built {T_arr.shape}  dtype={T_arr.dtype}", flush=True)
        # Stats
        for i, name in enumerate(FEAT_NAMES):
            col = T_arr[:, i]
            print(f"    {name}: mean={col.mean():.3e} std={col.std():.3e} "
                  f"min={col.min():.3e} max={col.max():.3e}", flush=True)
        verify_alignment(key, T_arr)
        out_npz = os.path.join(args.out_dir, f"trend_feat_{key}.npz")
        np.savez(out_npz, T=T_arr, feat_names=np.array(FEAT_NAMES))
        print(f"  saved {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB)", flush=True)

    progress("done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
