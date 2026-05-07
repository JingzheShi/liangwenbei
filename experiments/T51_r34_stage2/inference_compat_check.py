"""Verify T51 stage1+stage2 features are computable from a 100-tick inference window.

Loads a real session parquet, slices the last 100 ticks, computes all 113 extras,
and reports any NaN/Inf values per column. Prints diff vs full-session computation
at the same global index to confirm correctness at the boundary.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T44_DIR = os.path.join(ROOT, "experiments", "T44_r34_features")
sys.path.insert(0, HERE); sys.path.insert(0, T44_DIR)

from r34_features import compute_all_session as stage1_session, all_feature_names as s1_names
from r34_stage2_features import compute_all_session as stage2_session, all_feature_names as s2_names


def main():
    p = os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet")
    df_full = pd.read_parquet(p)
    T = len(df_full)
    print(f"session shape: {df_full.shape}")

    # Pick a row well inside the session (e.g. global index t=200) so a 100-tick
    # window ending at t fully overlaps real data.
    t_idx = 200
    win_lo = t_idx - 99
    win_hi = t_idx
    df_win = df_full.iloc[win_lo:win_hi + 1].reset_index(drop=True)
    assert len(df_win) == 100

    # Inference-window compute: valid_lo=valid_hi=99 (last tick of the window)
    f1_inf = stage1_session(df_win, 99, 99)
    f2_inf = stage2_session(df_win, 99, 99)
    extras_inf = np.concatenate([f1_inf, f2_inf], axis=1).reshape(-1)
    extras_names = s1_names() + s2_names()
    print(f"inference extras dim: {extras_inf.shape[0]} (expect 113)")

    # Full-session compute, take row at global t_idx
    f1_full = stage1_session(df_full, 99, T - 1 - 60)
    f2_full = stage2_session(df_full, 99, T - 1 - 60)
    # row in those arrays for global index t_idx is t_idx - 99
    row_full = t_idx - 99
    extras_full = np.concatenate([f1_full[row_full], f2_full[row_full]])

    # NaN/Inf
    nan_idx = np.where(~np.isfinite(extras_inf))[0]
    if len(nan_idx) > 0:
        print(f"!! NaN/Inf in inference compute: {len(nan_idx)} cols")
        for i in nan_idx[:10]:
            print(f"   {extras_names[i]} = {extras_inf[i]}")
    else:
        print(f"OK: all 113 extras finite at the inference boundary")

    # Diff
    diff = extras_inf - extras_full
    abs_diff = np.abs(diff)
    print(f"max |diff| inf vs full: {abs_diff.max():.6e}")
    top = np.argsort(-abs_diff)[:10]
    for i in top:
        print(f"  [{extras_names[i]}] inf={extras_inf[i]:+.6e} full={extras_full[i]:+.6e} diff={diff[i]:+.6e}")


if __name__ == "__main__":
    main()
