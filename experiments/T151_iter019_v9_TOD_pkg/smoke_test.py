"""Smoke test for iter_019 v9 TOD Predictor.

Verifies:
1. Predictor loads without errors
2. All 5 LGB models present and loadable
3. Feature vector is 364-d for LGB, 359-d for NN
4. predict() returns valid actions on real test data
5. No NaN in predictions
6. Stateless: same input → same output regardless of call order
"""
from __future__ import annotations
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- Load real test data ----
DATA_DIR = "/root/projects/liangwenbei_workdir/data"
PARQUET = os.path.join(DATA_DIR, "snapshot_sym0_date100_am.parquet")
PARQUET_PM = os.path.join(DATA_DIR, "snapshot_sym0_date0_pm.parquet")

print("=== iter_019 v9 TOD Smoke Test ===", flush=True)
print(f"PKG: {HERE}", flush=True)

# ---- Check models ----
print("\n[1] Checking model files...")
SEEDS = [1, 7, 13, 42, 100]
for s in SEEDS:
    p = os.path.join(HERE, f"model_h60_seed{s}.txt")
    if os.path.exists(p):
        sz = os.path.getsize(p) / 1e6
        print(f"  model_h60_seed{s}.txt  {sz:.1f} MB  OK")
    else:
        print(f"  model_h60_seed{s}.txt  MISSING!")
        sys.exit(1)

# ---- Load Predictor ----
print("\n[2] Loading Predictor...")
t0 = time.time()
from Predictor import Predictor
p = Predictor()
print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

# ---- Verify feature dimensions via inspection ----
print("\n[3] Verifying feature dimensions...")
import lightgbm as lgb
lgb_model = p._lgb_lists[60][0]
n_feat = lgb_model.num_feature()
print(f"  LGB model expects {n_feat} features (expected 364)")
assert n_feat == 364, f"LGB expects {n_feat} != 364"

nn_model = p._nn_lists[60][0]
print(f"  NN model in_dim = {nn_model.in_dim} (expected 359)")
assert nn_model.in_dim == 359, f"NN in_dim = {nn_model.in_dim} != 359"

# ---- Build test batches from real parquet ----
print("\n[4] Loading test data...")
df_raw = pd.read_parquet(PARQUET)
print(f"  shape: {df_raw.shape}")
print(f"  columns (first 5): {list(df_raw.columns[:5])}")
print(f"  'time' in columns: {'time' in df_raw.columns}")
print(f"  time sample: {df_raw['time'].head(3).tolist()}")

WINDOW = 100
n_rows = len(df_raw)
batches = []
for start in range(0, min(n_rows - WINDOW + 1, 1000), WINDOW):
    batch = df_raw.iloc[start:start + WINDOW].copy()
    batches.append(batch)

print(f"  Created {len(batches)} windows of size {WINDOW}")

# ---- Run predict ----
print("\n[5] Running predict()...")
t0 = time.time()
results = p.predict(batches)
elapsed = time.time() - t0
print(f"  Predicted {len(results)} windows in {elapsed:.2f}s ({elapsed/len(results)*1000:.1f} ms/window)")

# Verify results
actions_flat = [a for window_actions in results for a in window_actions]
actions_arr = np.array(actions_flat)
assert not np.any(np.isnan(actions_arr)), "NaN in actions!"
assert set(np.unique(actions_arr)).issubset({0, 1, 2}), f"Invalid actions: {np.unique(actions_arr)}"
print(f"  Action distribution: {dict(zip(*np.unique(actions_arr, return_counts=True)))}")
print(f"  All actions valid: OK")

# ---- Test PM session ----
print("\n[6] Testing PM session...")
if os.path.exists(PARQUET_PM):
    df_pm = pd.read_parquet(PARQUET_PM)
    batches_pm = []
    for start in range(0, min(len(df_pm) - WINDOW + 1, 200), WINDOW):
        batches_pm.append(df_pm.iloc[start:start + WINDOW].copy())
    results_pm = p.predict(batches_pm)
    actions_pm = np.array([a for r in results_pm for a in r])
    print(f"  PM actions: {dict(zip(*np.unique(actions_pm, return_counts=True)))}")
    print(f"  PM prediction: OK")
else:
    print(f"  PM parquet not found at {PARQUET_PM}, skipping")

# ---- Stateless test ----
print("\n[7] Statelessness check...")
b1 = [batches[0], batches[1]]
b2 = [batches[1], batches[0]]
r1 = p.predict(b1)
r2 = p.predict(b2)
r3 = p.predict(b1)  # repeat
assert r1 == r3, "Not stateless: same input gave different output!"
print(f"  Same-input consistency: OK (r1==r3)")
assert r1[0] == r2[1], "Window 0 of [0,1] != window 1 of [1,0]"
assert r1[1] == r2[0], "Window 1 of [0,1] != window 0 of [1,0]"
print(f"  Order-independence: OK")

# ---- TOD feature sanity ----
print("\n[8] TOD feature sanity...")
# Manually compute expected TOD for first batch
from Predictor import _time_to_secs, _AM_START_SEC, _PM_THRESH_SEC, _PM_START_SEC
t_val = batches[0]["time"].iloc[-1]
total_secs = _time_to_secs(t_val)
print(f"  Last tick time: {t_val} → {total_secs}s")
if total_secs < _PM_THRESH_SEC:
    sess_idx = 0
    mins_into = max(0.0, (total_secs - _AM_START_SEC) / 60.0)
else:
    sess_idx = 1
    mins_into = max(0.0, (total_secs - _PM_START_SEC) / 60.0)
mins_full = mins_into + 100.0 * sess_idx
print(f"  sess_idx={sess_idx}, mins_into={mins_into:.1f}, mins_full={mins_full:.1f}")
expected_tod_sin = float(np.sin(2.0 * np.pi * mins_full / 200.0))
print(f"  Expected tod_sin={expected_tod_sin:.4f}")

tod_result = p._compute_tod_features([batches[0]])
print(f"  Computed tod_sin={tod_result[0,0]:.4f}  {'OK' if abs(tod_result[0,0] - expected_tod_sin) < 1e-5 else 'MISMATCH'}")
assert abs(tod_result[0, 0] - expected_tod_sin) < 1e-5, "TOD sin mismatch"

print("\n=== ALL SMOKE TESTS PASSED ===")
print(f"  LGB features: 364-d  NN features: 359-d  Actions: valid")
