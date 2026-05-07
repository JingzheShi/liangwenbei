"""T51 R34 Stage 2 feature builder.

Reads each session parquet, computes:
  - T44 R34 stage 1 (54 dims) — re-using r34_features.compute_all_session
  - T51 R34 stage 2 (59 dims) — r34_stage2_features.compute_all_session
Aligns to the existing schemeC cache row order, then concatenates:
  [226 schemeC | 54 stage1 | 59 stage2]  →  339 dims (schemeM)

Output:
    cache/schemeM_train.npz / schemeM_val.npz / schemeM_test.npz
    cache/schemeM_extra_feat_names.txt   (54+59 = 113 stage extras names)
    cache/schemeM_feat_names.txt         (339 names)

Constraint compliance:
  - sym/date never used for feature computation
  - causal rolling (no look-ahead)
  - per-session reset (no cross-session shift)
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
T44_DIR = os.path.join(ROOT, "experiments", "T44_r34_features")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, T44_DIR)

from src.data.split import get_split  # noqa: E402

# Stage 1 (T44)
from r34_features import (  # noqa: E402
    compute_all_session as compute_stage1_session,
    all_feature_names as stage1_feature_names,
)
# Stage 2 (T51, this dir)
from r34_stage2_features import (  # noqa: E402
    compute_all_session as compute_stage2_session,
    all_feature_names as stage2_feature_names,
)

WINDOW = 100
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


def build_split(sym_dates, cache_dir: str) -> dict:
    Xs1, Xs2 = [], []
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

        f1 = compute_stage1_session(df, valid_lo, valid_hi)
        f2 = compute_stage2_session(df, valid_lo, valid_hi)

        if not np.all(np.isfinite(f1)):
            f1 = np.where(np.isfinite(f1), f1, 0.0).astype(np.float32)
        if not np.all(np.isfinite(f2)):
            f2 = np.where(np.isfinite(f2), f2, 0.0).astype(np.float32)

        Xs1.append(f1)
        Xs2.append(f2)
        n = f1.shape[0]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [{i+1}/{n_sessions}] sessions; {elapsed:.1f}s; {rate:.1f} sess/s", flush=True)

    out = {
        "X_extra1": np.concatenate(Xs1, axis=0),
        "X_extra2": np.concatenate(Xs2, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    print(
        f"  -> X_extra1 {out['X_extra1'].shape} {out['X_extra1'].nbytes/1e6:.2f}MB; "
        f"X_extra2 {out['X_extra2'].shape} {out['X_extra2'].nbytes/1e6:.2f}MB",
        flush=True,
    )
    return out


def verify_alignment(extra: dict, scheme_c_npz: str) -> None:
    print(f"  verify alignment vs {scheme_c_npz} ...", flush=True)
    d = np.load(scheme_c_npz)
    for k in ("sym", "date", "sess_idx", "t"):
        a = extra[k]
        b = d[k]
        assert a.shape == b.shape, f"{k} shape mismatch {a.shape} vs {b.shape}"
        if not np.array_equal(a, b):
            ne = int(np.sum(a != b))
            raise AssertionError(
                f"{k} mismatch: {ne} rows differ; first diff at index {(a != b).argmax()}"
            )
    print(f"  alignment OK on {len(extra['sym']):,} rows", flush=True)


def merge_and_save(extra: dict, scheme_c_npz: str, out_npz: str) -> None:
    """Concat schemeC X with extra1 + extra2 and save under schemeM_{split}.npz."""
    d = dict(np.load(scheme_c_npz))
    X_full = np.concatenate(
        [d["X"], extra["X_extra1"].astype(np.float32), extra["X_extra2"].astype(np.float32)],
        axis=1,
    )
    d["X"] = X_full
    np.savez(out_npz, **d)
    print(f"  saved -> {out_npz}  (X={X_full.shape})", flush=True)


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

    stage1_names = stage1_feature_names()
    stage2_names = stage2_feature_names()
    print(f"Stage 1 extras: {len(stage1_names)} | Stage 2 extras: {len(stage2_names)}", flush=True)

    base_names_path = os.path.join(args.scheme_c_cache_dir, "schemeC_feat_names.txt")
    with open(base_names_path) as f:
        base_names = [line.strip() for line in f]
    extra_names = stage1_names + stage2_names
    full_names = base_names + extra_names
    print(
        f"schemeM: {len(full_names)} = {len(base_names)} + {len(stage1_names)} + {len(stage2_names)}",
        flush=True,
    )

    extra_path = os.path.join(args.out_dir, "schemeM_extra_feat_names.txt")
    full_path = os.path.join(args.out_dir, "schemeM_feat_names.txt")
    with open(extra_path, "w") as f:
        f.write("\n".join(extra_names) + "\n")
    with open(full_path, "w") as f:
        f.write("\n".join(full_names) + "\n")
    print(f"  wrote extra names -> {extra_path}", flush=True)
    print(f"  wrote full names  -> {full_path}", flush=True)

    progress("loading_split", split=args.split)
    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]

    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        progress("building_split", split=key, n=len(split[key]))
        out = build_split(split[key], args.cache_dir)
        c_path = os.path.join(args.scheme_c_cache_dir, f"schemeC_{key}.npz")
        verify_alignment(out, c_path)
        m_path = os.path.join(args.out_dir, f"schemeM_{key}.npz")
        merge_and_save(out, c_path, m_path)

    progress("done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
