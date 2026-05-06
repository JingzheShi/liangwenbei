"""T35: Build extra ReVol + SG features per Scheme C cache row.

Produces `cache/schemeK_extra_{train,val,test}.npz`:
    X_extra : (N, F_extra)  float32   F_extra = 12 (ReVol) + 20 (SG) = 32
    sym     : (N,)          int8     -- alignment sanity vs Scheme C cache
    date    : (N,)          int16
    sess_idx: (N,)          int8
    t       : (N,)          int16

Row ordering EXACTLY matches `experiments/T5b_features_multihorizon/cache/
schemeC_{split}.npz`, because we iterate sessions in the same get_split order
and use the same (valid_lo, valid_hi) per session. We assert this on load.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.data.split import get_split  # noqa: E402
from revol import compute_revol_session, revol_feature_names  # noqa: E402
from savgol import compute_sg_session, sg_feature_names  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

REVOL_SIGNALS = [("midprice", "mid"), ("wmp_lvl1", "wmp1")]
SG_SIGNALS = [
    ("midprice", "mid"),
    ("wmp_lvl1", "wmp1"),
    ("imbalance", "imb"),
    ("mlofi_W60_lvl1", "mlofi60"),
    ("spread1", "spr1"),
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


def build_split(sym_dates: List[tuple], cache_dir: str, split_name: str) -> dict:
    revol_cols = [c for c, _ in REVOL_SIGNALS]
    sg_cols = [c for c, _ in SG_SIGNALS]

    Xs = []
    syms, dates, sess_idxs, ts = [], [], [], []

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

        rev = compute_revol_session(df, revol_cols, valid_lo, valid_hi)
        sgf = compute_sg_session(df, sg_cols, valid_lo, valid_hi)
        X_sl = np.concatenate([rev, sgf], axis=1)

        Xs.append(X_sl)
        n = X_sl.shape[0]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [{i+1}/{n_sessions}] sessions; {elapsed:.1f}s; {rate:.1f} sess/s", flush=True)

    out = {
        "X_extra": np.concatenate(Xs, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    print(
        f"  -> X_extra shape {out['X_extra'].shape} mem {out['X_extra'].nbytes/1e6:.2f}MB",
        flush=True,
    )
    return out


def verify_alignment(extra: dict, scheme_c_npz: str) -> None:
    print(f"  verify alignment vs {scheme_c_npz} ...", flush=True)
    d = np.load(scheme_c_npz)
    for k in ("sym", "date", "sess_idx", "t"):
        a = extra[k]; b = d[k]
        assert a.shape == b.shape, f"{k} shape mismatch {a.shape} vs {b.shape}"
        if not np.array_equal(a, b):
            ne = np.sum(a != b)
            raise AssertionError(f"{k} mismatch: {ne} rows differ; first diff at index {(a != b).argmax()}")
    print(f"  alignment OK on {len(extra['sym']):,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "features_v1"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument(
        "--scheme-c-cache-dir",
        default=os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache"),
    )
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    revol_names = revol_feature_names([n for _, n in REVOL_SIGNALS])
    sg_names = sg_feature_names([n for _, n in SG_SIGNALS])
    feat_names_extra = revol_names + sg_names

    print(f"Scheme K extras: {len(feat_names_extra)} features", flush=True)
    print(f"  ReVol ({len(revol_names)}): {revol_names}", flush=True)
    print(f"  SG    ({len(sg_names)}): {sg_names}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeK_extra_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_names_extra:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)

    progress("loading_split", split=args.split)
    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]

    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        progress("building_split", split=key, n=len(split[key]))
        out = build_split(split[key], args.cache_dir, key)
        c_path = os.path.join(args.scheme_c_cache_dir, f"schemeC_{key}.npz")
        verify_alignment(out, c_path)
        out_path = os.path.join(args.out_dir, f"schemeK_extra_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    progress("done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
