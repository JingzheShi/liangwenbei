"""T47: Build extended labels (h_30, h_120, h_240) aligned to T5b cache row order.

For each (sym, date, sess) parquet, compute:
  y_30   = label using threshold T=0.001 on (mp[t+30]-mp[t])
  y_120  = label using threshold T=0.001 on (mp[t+120]-mp[t])
  y_240  = label using threshold T=0.001 on (mp[t+240]-mp[t])
plus their respective mp_t* arrays.

The existing T5b cache uses t in [99, T-1-60] = [99, 1940] (per session).
For h=120, t+120 may exceed T-1; we set y=-1, mp_tH=NaN for invalid rows.
For h=240, similar.

Threshold reverse-engineered from existing labels:
  h=5,10:    T=0.0005
  h=20,40,60: T=0.001
We use T=0.001 for h=30 (between 20 and 40) and h=120, h=240 (extrapolation).

Output: experiments/T47_multihorizon_aux/cache/schemeC_{split}_extras.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split  # noqa: E402

WINDOW = 100
EXISTING_MAX_HORIZON = 60  # T5b cache valid_hi = T-1-60
NEW_HORIZONS = (30, 120, 240)
THRESHOLD = 0.001
SESS_TO_IDX = {"am": 0, "pm": 1}


def label_from_diff(diff: np.ndarray, T: float = THRESHOLD) -> np.ndarray:
    return np.where(diff > T, 2, np.where(diff < -T, 0, 1)).astype(np.int8)


def build_split(sym_dates, raw_dir: str) -> dict:
    ys = {h: [] for h in NEW_HORIZONS}
    mps = {f"mp_t{h}": [] for h in NEW_HORIZONS}

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()

    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(raw_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - EXISTING_MAX_HORIZON
        if valid_hi < valid_lo:
            continue
        mp = df["midprice"].to_numpy(dtype=np.float32, copy=False)

        for h in NEW_HORIZONS:
            n = valid_hi - valid_lo + 1
            y_h = np.full(n, -1, dtype=np.int8)
            mp_h = np.full(n, np.nan, dtype=np.float32)
            # valid rows: t + h <= T - 1, i.e. t <= T - 1 - h
            v_hi = T - 1 - h
            if v_hi >= valid_lo:
                lo = valid_lo
                hi = min(valid_hi, v_hi)
                # row indices in our slice (0-based offset from valid_lo)
                local_lo = lo - valid_lo
                local_hi = hi - valid_lo + 1
                base_t = np.arange(lo, hi + 1)
                diff = mp[base_t + h] - mp[base_t]
                y_h[local_lo:local_hi] = label_from_diff(diff)
                mp_h[local_lo:local_hi] = mp[base_t + h]
            ys[h].append(y_h)
            mps[f"mp_t{h}"].append(mp_h)

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_sessions}] sessions; elapsed {elapsed:.1f}s", flush=True)

    out = {}
    for h in NEW_HORIZONS:
        out[f"y{h}"] = np.concatenate(ys[h], axis=0)
        out[f"mp_t{h}"] = np.concatenate(mps[f"mp_t{h}"], axis=0)
        n_valid = (out[f"y{h}"] != -1).sum()
        cnts = np.bincount(out[f"y{h}"][out[f"y{h}"] != -1].astype(np.int64), minlength=3)
        print(f"  h{h}: total={len(out[f'y{h}']):,} valid={n_valid:,} "
              f"label0={cnts[0]:,} label1={cnts[1]:,} label2={cnts[2]:,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", default="local_debug")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split = get_split(args.split_mode)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} extras ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], args.raw_dir)
        out_path = os.path.join(args.out_dir, f"schemeC_{key}_extras.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
