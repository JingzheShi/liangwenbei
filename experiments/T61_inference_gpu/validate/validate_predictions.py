"""Validate iter_010 predictions match iter_009 predictions on real windows."""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
ITER9 = os.path.join(ROOT, "submission", "iter_009_t59_fullsym")
ITER10 = os.path.join(ROOT, "submission", "iter_010_fast")

# Load each via importlib so they don't conflict
def load_pred(here, name):
    sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, "Predictor.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.remove(here)
    return m

m9 = load_pred(ITER9, "iter9_pred")
m10 = load_pred(ITER10, "iter10_pred")

p9 = m9.Predictor()
p10 = m10.Predictor()

cfg = json.load(open(os.path.join(ITER10, "config.json")))
raw_cols = cfg["feature"]

# Load real data, build 256 random windows
df1 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet"))
df2 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date100_am.parquet"))

rng = np.random.default_rng(2026)
N = 256
windows = []
for _ in range(N):
    use_df = df1 if rng.random() < 0.5 else df2
    end_i = int(rng.integers(99, len(use_df) - 1))
    df_w = use_df.iloc[end_i - 99:end_i + 1].reset_index(drop=True).copy()
    windows.append(df_w[raw_cols])

# Get features from each path
feats9 = p9._compute_batch_features(windows)
feats10 = p10._compute_batch_features(windows)

print(f"feats9 shape: {feats9.shape}, feats10 shape: {feats10.shape}")

# Compare
assert feats9.shape == feats10.shape
diff = np.abs(feats9 - feats10)
per_feat_max = diff.max(axis=0)
worst = np.argsort(-per_feat_max)[:10]
print("\n=== Top 10 worst feature diffs (iter_010 vs iter_009) ===")
for i in worst:
    print(f"  [{i}]: max_diff={per_feat_max[i]:.3e}  ref_scale={np.abs(feats9[:,i]).max():.3e}")

# Predictions
out9 = np.array(p9.predict(windows))
out10 = np.array(p10.predict(windows))

print(f"\nout9 shape: {out9.shape}, out10 shape: {out10.shape}")
print(f"Predictions identical (h=60): {np.array_equal(out9[:, 4], out10[:, 4])}")

# Per-horizon agreement
for h_idx, h in enumerate([5,10,20,40,60]):
    agree = (out9[:, h_idx] == out10[:, h_idx]).mean()
    print(f"  h={h}: agreement {agree*100:.2f}%  (active={(out9[:,h_idx]!=1).sum()} non-1 in iter9 vs {(out10[:,h_idx]!=1).sum()} in iter10)")

# Detailed prob comparison for h=60
probs9 = p9._ensemble_predict(p9._booster_lists[60], feats9)
probs10 = p10._ensemble_predict(p10._booster_lists[60], feats10)
prob_diff = np.abs(probs9 - probs10)
print(f"\nh=60 prob max diff: {prob_diff.max():.3e}")
print(f"h=60 prob median diff: {np.median(prob_diff):.3e}")
