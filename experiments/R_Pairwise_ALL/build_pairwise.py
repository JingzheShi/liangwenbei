"""R_Pairwise_ALL — 190 pairwise imbalance ratios across all 20 prices.

For each (sym, date, sess) parquet, mirror the schemeP/HYD windowing
(100-tick sliding window). For each *last tick* we compute pairwise
imbalances over 20 prices (bid1..bid10, ask1..ask10):

    imb_pair = (p_a - p_b) / (p_a + p_b + eps)   for all C(20,2)=190 ordered pairs (a<b)

Feature names: pair_<a>_<b>  where (a, b) ∈ all 20-choose-2 combinations.

CRITICAL_CONSTRAINTS-safe:
  - sym/date free
  - single-tick computation (last tick of window) — stateless
  - ratio scale-invariant — sym-agnostic

Output: cache/pairwise_{train,val,test}.npy float32 (N, 190)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations

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

# Match HYD raw column ordering. We only need the 20 prices though.
PRICE_NAMES = [f"bid{k}" for k in range(1, 11)] + [f"ask{k}" for k in range(1, 11)]
assert len(PRICE_NAMES) == 20

PAIRS = list(combinations(range(20), 2))
assert len(PAIRS) == 190

PAIR_FEAT_NAMES = [f"pair_{PRICE_NAMES[a]}_{PRICE_NAMES[b]}" for a, b in PAIRS]

EPS = 1e-12


def pairwise_features(prices_window: np.ndarray) -> np.ndarray:
    """prices_window: (n_valid, 100, 20) -> (n_valid, 190) float32 last-tick imbalances."""
    last = prices_window[:, -1, :]  # (n_valid, 20)
    out = np.empty((last.shape[0], len(PAIRS)), dtype=np.float32)
    a_idx = np.array([p[0] for p in PAIRS], dtype=np.int64)
    b_idx = np.array([p[1] for p in PAIRS], dtype=np.int64)
    pa = last[:, a_idx]
    pb = last[:, b_idx]
    out[:] = ((pa - pb) / (pa + pb + EPS)).astype(np.float32)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def build_one_session(df: pd.DataFrame) -> np.ndarray | None:
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - MAX_HORIZON
    if valid_hi < valid_lo:
        return None
    A = df[PRICE_NAMES].to_numpy(dtype=np.float64, copy=False)
    sw = sliding_window_view(A, (WINDOW, A.shape[1])).squeeze(1)
    n_valid = valid_hi - valid_lo + 1
    return pairwise_features(sw[:n_valid])


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
    return np.concatenate(out_list, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "pairwise_feat_names.txt"), "w") as f:
        f.write("\n".join(PAIR_FEAT_NAMES) + "\n")

    split = get_split(args.split)
    for key in args.which.split(","):
        key = key.strip()
        if key not in split:
            continue
        print(f"=== {key} ({len(split[key])} sessions) ===", flush=True)
        X = build_split(split[key], args.raw_dir)
        out_path = os.path.join(args.out_dir, f"pairwise_{key}.npy")
        np.save(out_path, X.astype(np.float32))
        sz_mb = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} shape={X.shape} size={sz_mb:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
