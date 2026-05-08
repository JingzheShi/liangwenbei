"""Drill-down: per-feature cache-vs-fresh diff distribution.

The earlier audit showed kyle_inv max_abs_diff up to 0.056. Need to confirm:
  - Diff is concentrated in kyle_inv (which is numerically unstable: amt_last
    / cbrt(amt_last * sigma + EPS) — small sigma blows up).
  - For the other 214 features, diff is at float32 storage precision (~1e-5
    relative).
  - The diff is NOT a sign of window-shift or forward leak.

Also: re-run cache-vs-fresh in a regime where amt_last is large but sigma is
not vanishing, and verify diff drops.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUBDIR = os.path.join(ROOT, "submission", "iter_013_time_features")
sys.path.insert(0, SUBDIR)

import fast_features_batch as ffb  # noqa: E402

WINDOW = 100
RAW_COLS: List[str] = (
    ["open", "high", "low", "close", "volume_delta", "amount_delta"]
    + [f"bid{k}" for k in range(1, 11)]
    + [f"bsize{k}" for k in range(1, 11)]
    + [f"ask{k}" for k in range(1, 11)]
    + [f"asize{k}" for k in range(1, 11)]
    + ["avgbid", "avgask", "totalbsize", "totalasize",
       "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
       "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
       "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc"]
    + [f"midprice{k}" for k in range(1, 11)]
    + [f"spread{k}" for k in range(1, 11)]
    + [f"bid_diff{k}" for k in range(1, 11)]
    + [f"ask_diff{k}" for k in range(1, 11)]
    + ["bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance"]
    + [f"bid_rate{k}" for k in range(1, 11)]
    + [f"ask_rate{k}" for k in range(1, 11)]
    + [f"bsize_rate{k}" for k in range(1, 11)]
    + [f"asize_rate{k}" for k in range(1, 11)]
)
COL_IDX = {c: i for i, c in enumerate(RAW_COLS)}
COL_IDX["midprice"] = COL_IDX["midprice1"]
FEAT_NAMES = ffb.all_feature_names()
NF = len(FEAT_NAMES)


def _load_real_window(parquet_path: str, end_idx: int) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    arr = df[RAW_COLS].to_numpy(dtype=np.float64, copy=True)
    return arr[end_idx - WINDOW + 1: end_idx + 1].copy()


def main() -> int:
    cache = np.load(os.path.join(ROOT, "experiments", "T68_stage5_features",
                                  "cache", "schemeP_train.npz"))
    X = cache["X"]
    sym = cache["sym"]; date = cache["date"]; sess = cache["sess_idx"]; t_arr = cache["t"]
    n_total = X.shape[0]
    rng = np.random.default_rng(13)
    pick = rng.choice(n_total, size=128, replace=False)
    SESS_NAMES = {0: "am", 1: "pm"}

    abs_diff_accum = np.zeros(NF, dtype=np.float64)
    rel_diff_accum = np.zeros(NF, dtype=np.float64)
    n_seen = 0
    per_row_max = []

    for idx in pick:
        s_int = int(sym[idx]); d_int = int(date[idx])
        sn = SESS_NAMES[int(sess[idx])]; t_int = int(t_arr[idx])
        path = os.path.join(ROOT, "data", f"snapshot_sym{s_int}_date{d_int}_{sn}.parquet")
        if not os.path.isfile(path):
            continue
        win = _load_real_window(path, t_int)
        feats = ffb.compute_batch_features(win[None, :, :], COL_IDX)[0]
        cache_extras = X[idx, 154:154 + NF].astype(np.float64)
        d = np.abs(feats - cache_extras)
        denom = np.maximum(np.abs(cache_extras), 1.0)
        rel = d / denom
        abs_diff_accum = np.maximum(abs_diff_accum, d)
        rel_diff_accum = np.maximum(rel_diff_accum, rel)
        per_row_max.append(float(d.max()))
        n_seen += 1

    print(f"[drilldown] n_seen = {n_seen}", flush=True)
    print(f"[drilldown] global max abs diff = {abs_diff_accum.max():.6e}")
    print(f"[drilldown] global max rel diff = {rel_diff_accum.max():.6e}")
    print()
    print("Top 20 features by max-abs cache-vs-fresh diff:")
    order = np.argsort(-abs_diff_accum)
    for i in order[:20]:
        print(f"  {FEAT_NAMES[i]:<35} abs={abs_diff_accum[i]:.3e}  rel={rel_diff_accum[i]:.3e}")
    print()
    print("Top 20 features by max-rel cache-vs-fresh diff:")
    order_r = np.argsort(-rel_diff_accum)
    for i in order_r[:20]:
        print(f"  {FEAT_NAMES[i]:<35} abs={abs_diff_accum[i]:.3e}  rel={rel_diff_accum[i]:.3e}")
    print()

    # Drop kyle_inv & see remaining
    keep = [i for i, n in enumerate(FEAT_NAMES) if "kyle_inv" not in n]
    rem_abs = abs_diff_accum[keep].max()
    rem_rel = rel_diff_accum[keep].max()
    print(f"After dropping kyle_inv*: max abs = {rem_abs:.6e}, max rel = {rem_rel:.6e}")

    out_path = os.path.join(HERE, "audit_drilldown.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_seen": n_seen,
            "feat_names": FEAT_NAMES,
            "max_abs_diff_per_feature": abs_diff_accum.tolist(),
            "max_rel_diff_per_feature": rel_diff_accum.tolist(),
            "global_max_abs": float(abs_diff_accum.max()),
            "global_max_rel": float(rel_diff_accum.max()),
            "max_abs_excluding_kyle": float(rem_abs),
            "max_rel_excluding_kyle": float(rem_rel),
        }, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
