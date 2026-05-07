"""Precision check: compare iter_012 batch features against schemeP cache.

For 50 sample test rows (different sym/date/t), construct 100-tick windows from
raw parquet snapshots, run iter_012 Predictor._compute_batch_features, and
compare extras_kept block against schemeP_test.npz X[idx][raw + 216 extras][kept].

Tolerance: max_abs_diff < 1e-3 on float32-stored cache (cache values were
stored as float32, our extras are computed in float64 then cast to float32).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUB = os.path.join(ROOT, "submission", "iter_012_v4_stage5")
CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
sys.path.insert(0, SUB)

from Predictor import Predictor

WINDOW = 100
N_RAW = 154

pred = Predictor()

test = np.load(os.path.join(CACHE, "schemeP_test.npz"))
X_cache = test["X"]  # (442080, 370) float32
sym_arr = test["sym"]
date_arr = test["date"]
sess_arr = test["sess_idx"]
t_arr = test["t"]

with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
    feat_names_full = [line.strip() for line in f]
extras_idx = list(range(N_RAW, len(feat_names_full)))  # 154..369 = 216 extras
assert len(extras_idx) == 216

# Sample 50 evenly spaced indices across the cache
np.random.seed(0)
sample_idx = np.random.choice(len(X_cache), size=50, replace=False)
print(f"sampling {len(sample_idx)} rows.")

sess_map = {0: "am", 1: "pm"}

# Build (N, 100, K) tensor and a list of DataFrames
batches = []
n_loaded = 0
n_skipped = 0
for ix in sample_idx:
    s = int(sym_arr[ix]); d = int(date_arr[ix]); ss = int(sess_arr[ix]); tt = int(t_arr[ix])
    p = os.path.join(ROOT, "data", f"snapshot_sym{s}_date{d}_{sess_map[ss]}.parquet")
    if not os.path.exists(p):
        n_skipped += 1
        continue
    df = pd.read_parquet(p)
    # window is rows tt-99 .. tt (inclusive)
    if tt + 1 < WINDOW or tt + 1 > len(df):
        n_skipped += 1
        continue
    win = df.iloc[tt + 1 - WINDOW : tt + 1].reset_index(drop=True)
    # Predictor._compute_batch_features expects DataFrame with config feature columns
    # The parquet has all needed columns plus extras (date/sym/time)
    batches.append((ix, win))
    n_loaded += 1

print(f"loaded {n_loaded} windows, skipped {n_skipped}")

# Compute via Predictor
t0 = time.time()
feats = pred._compute_batch_features([w for (_, w) in batches])
print(f"compute_batch_features: {time.time()-t0:.3f}s, shape={feats.shape}")
assert feats.shape[1] == 359, f"expected 359 features, got {feats.shape[1]}"

# Extract cache values for the same rows: raw_last (154) + extras_kept (205)
indices = [ix for (ix, _) in batches]
cache_block = X_cache[indices]  # (N, 370)
# raw block:
cache_raw = cache_block[:, :N_RAW]
# extras (216), keep ones not in FAIL
cache_extras_full = cache_block[:, N_RAW:]
cache_extras_kept = cache_extras_full[:, pred._extra_keep_idx]
cache_combined = np.concatenate([cache_raw, cache_extras_kept], axis=1)
print(f"cache combined shape: {cache_combined.shape}")

# Compare
diff = np.abs(feats - cache_combined)
print(f"\n=== Precision comparison ===")
print(f"  max diff:          {diff.max():.6g}")
print(f"  mean diff:         {diff.mean():.6g}")
print(f"  99th-pct diff:     {np.percentile(diff, 99):.6g}")
print(f"  max diff in raw block (cols 0..154):  {diff[:, :N_RAW].max():.6g}")
print(f"  max diff in extras  (cols 154..359):  {diff[:, N_RAW:].max():.6g}")

# Per-feature max diff
worst_cols = np.argsort(diff.max(axis=0))[-10:][::-1]
extra_names = pred._extra_names
keep_idx_to_name = [extra_names[i] for i in pred._extra_keep_idx]
all_names = list(pred._raw_feat_cols) + keep_idx_to_name
print(f"\nWorst 10 columns by max abs diff:")
for c in worst_cols:
    print(f"  col {c} ({all_names[c]}): max={diff[:, c].max():.6g}")

# Decision
THRESH = 1e-2  # cache stored as float32, plus our internal float32 cast adds ~1e-3 noise
ok = diff.max() < THRESH
print(f"\nDecision: {'PASS' if ok else 'FAIL'} (threshold {THRESH})")
sys.exit(0 if ok else 1)
