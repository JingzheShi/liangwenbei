"""Verify that submission/iter_001g/Predictor._build_feats_batch produces exactly
the same features for a real session window as build_features.build_features_one_session.

Critical test: training-inference parity. If features differ, the model is meaningless.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from build_features import (
    build_features_one_session, feature_names, get_feature_group_indices,
)
from src.data.dataset import get_default_feature_cols


def load_predictor():
    pred_path = os.path.join(ROOT, "submission", "iter_001g_lgbm_schemeE", "Predictor.py")
    spec = importlib.util.spec_from_file_location("iter1g_predictor", pred_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # Load real data
    feat_cols = get_default_feature_cols()
    df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot_sym0_date0_am.parquet"))

    # Build training features
    X_train, _, _, _, t_valid = build_features_one_session(df, feat_cols)
    print(f"training features: X_train={X_train.shape}, valid t range = [{t_valid[0]}, {t_valid[-1]}]")

    # For Predictor inference, simulate getting (100, 154) windows ending at each valid t
    arr_full = df[feat_cols].to_numpy(dtype=np.float32)
    # Pick 5 sample t's evenly spaced in valid range
    sample_ts = np.linspace(t_valid[0], t_valid[-1], 5).astype(int)
    windows = np.stack([arr_full[t - 99 : t + 1] for t in sample_ts], axis=0)  # (5, 100, 154)
    print(f"inference windows: {windows.shape}")

    # Use Predictor's batch builder
    Predictor = load_predictor().Predictor
    idx_map = {c: i for i, c in enumerate(feat_cols)}
    p = Predictor.__new__(Predictor)  # don't load model
    feats_pred = Predictor._build_feats_batch(windows, idx_map)
    print(f"predictor features: {feats_pred.shape}")

    # Compare with training features at same t's
    feats_train = X_train[sample_ts - t_valid[0]]
    print(f"training features at same t's: {feats_train.shape}")

    diff = np.abs(feats_pred - feats_train)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"\nmax abs diff: {max_diff:.6e}")
    print(f"mean abs diff: {mean_diff:.6e}")

    # Per-group diff to localize any discrepancy
    fnames = feature_names(feat_cols)
    groups = get_feature_group_indices(fnames)
    for g, idxs in groups.items():
        d = diff[:, idxs]
        print(f"  group {g} ({len(idxs)} cols): max={d.max():.6e}  mean={d.mean():.6e}")

    if max_diff > 1e-4:
        # Show the worst columns
        worst_cols = np.argsort(diff.max(axis=0))[-10:][::-1]
        print("\n  worst-diff columns:")
        for c in worst_cols:
            print(f"    [{c}] {fnames[c]}: train={feats_train[0, c]:.6f}  "
                  f"pred={feats_pred[0, c]:.6f}  diff={diff[0, c]:.6e}")
        raise AssertionError(f"Predictor / training feature mismatch (max_diff={max_diff})")

    print("\nOK: Predictor matches training features within float32 precision")


if __name__ == "__main__":
    main()
