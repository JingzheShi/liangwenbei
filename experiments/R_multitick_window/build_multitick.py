"""Build multi-tick window snapshot features (the multi-tick trick).

For each row at position t in a session (t in [99, T-1-60]), extract LOB
features at lookback positions {5, 20, 50, 90} (raw values).

Per lookback: bid1..5, bsize1..5, ask1..5, asize1..5 = 20 features.
Total = 4 lookbacks * 20 = 80 features.

Output: multitick_extra_{train,val,test}.npz with:
  - X_multitick: (N, 80) float32
  - sym, date, sess_idx, t: alignment keys (matching schemeP)

Constraint: stateless within window — all features computed from a single
100-tick window of the source parquet (no cross-session/cross-batch state).
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split  # noqa: E402

WINDOW = 100
MAX_HORIZON = 60
SESS_TO_IDX = {"am": 0, "pm": 1}

# Lookback positions (ticks back from current row)
LOOKBACKS = [5, 20, 50, 90]

# Per-lookback features (raw LOB values, levels 1-5)
PER_LOOKBACK_COLS = [
    "bid1", "bid2", "bid3", "bid4", "bid5",
    "bsize1", "bsize2", "bsize3", "bsize4", "bsize5",
    "ask1", "ask2", "ask3", "ask4", "ask5",
    "asize1", "asize2", "asize3", "asize4", "asize5",
]
N_PER_LB = len(PER_LOOKBACK_COLS)  # 20
N_LB = len(LOOKBACKS)              # 4
TOTAL_DIM = N_LB * N_PER_LB        # 80


def feature_names() -> list[str]:
    names = []
    for lb in LOOKBACKS:
        for col in PER_LOOKBACK_COLS:
            names.append(f"{col}_lb{lb}")
    return names


def compute_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Vectorized: for rows t in [valid_lo, valid_hi], extract features at t-k for k in LOOKBACKS."""
    # Get the LOB cols once
    cols_arr = df[PER_LOOKBACK_COLS].to_numpy(dtype=np.float32, copy=False)
    n_rows = valid_hi - valid_lo + 1
    out = np.empty((n_rows, TOTAL_DIM), dtype=np.float32)
    for i, lb in enumerate(LOOKBACKS):
        # rows at position t-lb for t in [valid_lo, valid_hi]
        slice_start = valid_lo - lb
        slice_end = valid_hi - lb + 1
        out[:, i * N_PER_LB : (i + 1) * N_PER_LB] = cols_arr[slice_start:slice_end]
    return out


def build_split(sym_dates, cache_dir: str):
    Xs, syms, dates, sess_idxs, ts = [], [], [], [], []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 10)
    t0 = time.time()

    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - MAX_HORIZON
        if valid_hi < valid_lo:
            continue

        f = compute_session(df, valid_lo, valid_hi)
        if not np.all(np.isfinite(f)):
            f[~np.isfinite(f)] = 0.0
        Xs.append(f)
        n = f.shape[0]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [{i+1}/{n_sessions}] sessions; {elapsed:.1f}s; {rate:.1f} sess/s", flush=True)

    return {
        "X_multitick": np.concatenate(Xs, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }


def verify_alignment(extra: dict, ref_npz: str) -> None:
    print(f"  verify alignment vs {ref_npz} ...", flush=True)
    d = np.load(ref_npz)
    for k in ("sym", "date", "sess_idx", "t"):
        a = extra[k]
        b = d[k]
        assert a.shape == b.shape, f"{k} shape mismatch {a.shape} vs {b.shape}"
        if not np.array_equal(a, b):
            ne = int(np.sum(a != b))
            raise AssertionError(
                f"{k} mismatch: {ne} rows differ; first diff at idx {(a != b).argmax()}"
            )
    print(f"  alignment OK on {len(extra['sym']):,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "features_v1"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument(
        "--ref-cache-dir",
        default=os.path.join(ROOT, "experiments", "T55_stage4_extras", "cache"),
    )
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    names = feature_names()
    names_path = os.path.join(args.out_dir, "multitick_feat_names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"  wrote {len(names)} feat names -> {names_path}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], args.cache_dir)
        ref_path = os.path.join(args.ref_cache_dir, f"schemeP_{key}.npz")
        verify_alignment(out, ref_path)
        out_path = os.path.join(args.out_dir, f"multitick_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}  (X_multitick={out['X_multitick'].shape})", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
