"""Validate fast_features_batch.py output matches fast_features.py single-window."""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
ITER10 = os.path.join(ROOT, "submission", "iter_010_fast")
sys.path.insert(0, ITER10)

import fast_features as ff
import fast_features_batch as ffb

cfg = json.load(open(os.path.join(ROOT, "submission", "iter_009_t59_fullsym", "config.json")))
raw_cols = cfg["feature"]
col_idx = {c: i for i, c in enumerate(raw_cols)}
print(f"raw_cols count: {len(raw_cols)}")

# Load 2 sessions (one used for windows)
df1 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet"))
df2 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date100_am.parquet"))

# Pick N=20 random 100-row windows
rng = np.random.default_rng(42)
N = 20
windows = []
end_indices = []
for _ in range(N):
    use_df = df1 if rng.random() < 0.5 else df2
    end_i = int(rng.integers(99, len(use_df) - 1))
    df_w = use_df.iloc[end_i - 99:end_i + 1].reset_index(drop=True).copy()
    windows.append(df_w)
    end_indices.append(end_i)

# Build X3d
N = len(windows)
T = 100
K = len(raw_cols)
X3d = np.zeros((N, T, K), dtype=np.float64)
for n, w in enumerate(windows):
    for c in raw_cols:
        X3d[n, :, col_idx[c]] = w[c].to_numpy(dtype=np.float64, copy=False)

# Batch path
batch_feat = ffb.compute_batch_features(X3d, col_idx)
print(f"batch_feat shape: {batch_feat.shape}")  # (N, 196)

# Single-window path (loop)
sw_feat = np.zeros((N, 196), dtype=np.float64)
for n, w in enumerate(windows):
    df_arr = {}
    for c in w.columns:
        s = w[c]
        if pd.api.types.is_numeric_dtype(s):
            df_arr[c] = s.to_numpy(dtype=np.float64, copy=False)
    df_arr["midprice"] = df_arr["midprice1"]
    sw_feat[n] = ff.compute_window_features(df_arr)

diff = np.abs(batch_feat - sw_feat)
all_names = ffb.all_feature_names()
assert len(all_names) == 196

# Per-feature max diff
print("\n=== Per-feature max diff (top 20 worst) ===")
per_feat_max = diff.max(axis=0)
worst = np.argsort(-per_feat_max)[:20]
for i in worst:
    print(f"  [{i}] {all_names[i]}: max_diff={per_feat_max[i]:.3e}  "
          f"sw={sw_feat[diff[:,i].argmax(),i]:.4f}  batch={batch_feat[diff[:,i].argmax(),i]:.4f}")

print(f"\nOverall max diff: {diff.max():.3e}")
print(f"Median diff: {np.median(diff):.3e}")

# Tolerance check (relative)
rel = diff / (np.abs(sw_feat) + 1e-8)
print(f"Max relative diff: {rel.max():.3e}")

# Check for any feature with > 1e-3 absolute diff (should be small or only on huge values like kyle_inv)
big_diff = (per_feat_max > 1e-3)
if big_diff.any():
    print("\nFeatures with > 1e-3 max diff:")
    for i in np.where(big_diff)[0]:
        v = sw_feat[:, i]
        print(f"  [{i}] {all_names[i]}: max_diff={per_feat_max[i]:.3e}  scale={np.abs(v).max():.3e}")
