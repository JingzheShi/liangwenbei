"""Verify the iter_007 Predictor's per-window 327-d feature vector matches
the training cache row-for-row at a fixed (sym, date, sess, t) point.

This is the safest sanity check: we compute the inference 327-d vector from a
100-tick window (only raw cols, no derived) and compare element-wise to the
cache row at the corresponding global index.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from Predictor import Predictor

CACHE = os.path.join(ROOT, "experiments", "T51_r34_stage2", "cache")


def load_keep_idx():
    base_dim = 226
    extras = open(os.path.join(CACHE, "schemeM_extra_feat_names.txt")).read().splitlines()
    report = json.load(open(os.path.join(ROOT, "experiments", "T51_r34_stage2", "sym_invariance_report.json")))
    fail = report["summary"]["fail_names"]
    fail_global = [base_dim + extras.index(n) for n in fail]
    drop = set(fail_global) | {223, 224, 225}
    total = base_dim + len(extras)
    keep = np.array([i for i in range(total) if i not in drop], dtype=np.int64)
    return keep


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    raw_cols = cfg["feature"]
    pred = Predictor()

    # Pick a session and a global tick index.
    # Use train npz for ground truth aligned to schemeM cache.
    train = np.load(os.path.join(CACHE, "schemeM_train.npz"))
    # Pick a row well inside the session: e.g. first row where t==150 in the cache.
    sym_arr = train["sym"]; date_arr = train["date"]; sess_arr = train["sess_idx"]; t_arr = train["t"]
    # find idx with t=150, sym=0, sess_idx=0
    mask = (sym_arr == 0) & (sess_arr == 0) & (t_arr == 150)
    idx = int(np.argmax(mask))
    if not mask[idx]:
        raise SystemExit("no matching cache row")
    sym = int(sym_arr[idx]); date = int(date_arr[idx]); sess_idx = int(sess_arr[idx]); t = int(t_arr[idx])
    sess = "am" if sess_idx == 0 else "pm"
    keep = load_keep_idx()
    cache_row = train["X"][idx, keep]
    print(f"  cache row idx={idx}: sym={sym} date={date} sess={sess} t={t} feat_dim={cache_row.shape[0]}")

    # Load full session parquet, slice [t-99, t] for the inference window.
    parquet_p = os.path.join(ROOT, "data", "features_v1", f"snapshot_sym{sym}_date{date}_{sess}.parquet")
    df = pd.read_parquet(parquet_p)
    # Restrict to ONLY raw 154 cols (simulating platform input).
    df_raw = df[raw_cols].iloc[t - 99:t + 1].reset_index(drop=True)
    assert len(df_raw) == 100

    f223, f327 = pred._compute_window_features(df_raw)
    print(f"  inference 327-d shape: {f327.shape}")

    # Compare element-wise
    diff = f327 - cache_row.astype(np.float32)
    abs_diff = np.abs(diff)
    print(f"  max |diff|: {abs_diff.max():.6e}")
    print(f"  median |diff|: {np.median(abs_diff):.6e}")

    # Top mismatches
    feat_names = open(os.path.join(CACHE, "schemeM_feat_names.txt")).read().splitlines()
    kept_names = [feat_names[i] for i in keep]
    top = np.argsort(-abs_diff)[:10]
    print("  top 10 mismatches:")
    for i in top:
        print(f"   [{i:3d}] {kept_names[i]:30s}  inf={f327[i]:+.6e}  cache={cache_row[i]:+.6e}  diff={diff[i]:+.6e}")


if __name__ == "__main__":
    main()
