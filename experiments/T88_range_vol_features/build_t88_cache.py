"""T88: Build augmented cache with 12 range-vol features per row.

Aligns row-for-row with schemeP cache (T68_stage5_features/cache/schemeP_*.npz).
Iterates same (sym, date, sess) order, same valid_lo=99 / valid_hi=T-1-60 window.

Output: cache/t88_range_vol_{train,val,test}.npz
   X_t88: (n_rows_in_schemeP, 12) float32  — same row order as schemeP cache
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

from src.data.split import get_split  # noqa: E402
from range_vol_features import (  # noqa: E402
    compute_range_vol_batch,
    feature_names as t88_feature_names,
)

WINDOW = 100  # same as schemeP
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}


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
    """Returns dict with X_t88 (n_valid, 12)."""
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return {}

    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    # sliding window: (T - W + 1, W) — each row i has close[i..i+W-1]
    sw = sliding_window_view(close, WINDOW)
    n_valid = valid_hi - valid_lo + 1
    # Window ending at t=valid_lo+i corresponds to sw[(valid_lo+i)-W+1] = sw[i] (since valid_lo=W-1)
    # Window covers indices [(valid_lo+i)-W+1 .. valid_lo+i] = [i .. i+W-1] = sw[i]
    close_3d = sw[:n_valid]  # (n_valid, 100)

    feats = compute_range_vol_batch(close_3d)  # (n_valid, 12)

    return {
        "X_t88": feats,
        "n": n_valid,
    }


def build_split(sym_dates, data_dir: str) -> Dict[str, np.ndarray]:
    Xs: List[np.ndarray] = []
    syms: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    sess_idxs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path, columns=["close"])
        out = build_one_session(df)
        if not out:
            continue
        Xs.append(out["X_t88"])
        n = out["n"]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(WINDOW - 1, WINDOW - 1 + n, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (n_sessions - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1}/{n_sessions}] sessions  {elapsed:.1f}s  {rate:.1f} sess/s  eta {eta:.0f}s",
                  flush=True)
            progress("building", done=i + 1, total=n_sessions, sec=elapsed)

    res = {
        "X_t88": np.concatenate(Xs, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    print(f"  total rows: {res['X_t88'].shape[0]:,} | feats: {res['X_t88'].shape[1]}",
          flush=True)
    return res


def verify_alignment(res, schemeP_cache):
    """Verify row order matches schemeP cache exactly."""
    n = len(schemeP_cache["sym"])
    if len(res["sym"]) != n:
        raise AssertionError(f"row count mismatch: t88={len(res['sym'])} vs schemeP={n}")
    if not (res["sym"] == schemeP_cache["sym"]).all():
        raise AssertionError("sym mismatch")
    if not (res["date"] == schemeP_cache["date"]).all():
        raise AssertionError("date mismatch")
    if not (res["sess_idx"] == schemeP_cache["sess_idx"]).all():
        raise AssertionError("sess_idx mismatch")
    if not (res["t"] == schemeP_cache["t"]).all():
        raise AssertionError("t mismatch")
    print(f"  ALIGNMENT VERIFIED: {n:,} rows match schemeP cache exactly", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    ap.add_argument("--schemeP-dir",
                    default=os.path.join(ROOT, "experiments", "T68_stage5_features", "cache"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = t88_feature_names()
    name_path = os.path.join(args.out_dir, "t88_feat_names.txt")
    with open(name_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {name_path}: {len(names)} names", flush=True)

    progress("loading_split", split=args.split)
    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}")
            continue
        print(f"\n=== {key} ({len(split[key])} sessions) ===", flush=True)
        progress("split", split=key, n=len(split[key]))
        out = build_split(split[key], args.data_dir)

        # Verify alignment vs schemeP cache
        schemeP_path = os.path.join(args.schemeP_dir, f"schemeP_{key}.npz")
        if os.path.exists(schemeP_path):
            schemeP = np.load(schemeP_path)
            verify_alignment(out, schemeP)
        else:
            print(f"  WARN no schemeP cache at {schemeP_path}; skipping alignment check")

        out_npz = os.path.join(args.out_dir, f"t88_range_vol_{key}.npz")
        np.savez(out_npz, **out)
        print(f"  saved {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB)", flush=True)

    progress("done")
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
