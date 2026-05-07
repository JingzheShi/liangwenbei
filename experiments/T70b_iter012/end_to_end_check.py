"""End-to-end: pass cache rows through T70 models directly vs through Predictor.

Loads ~256 rows from schemeP_test.npz, drops the 11 FAIL columns to get the
training feature matrix, runs each booster's predict to get reference probs.
Then runs the Predictor on reconstructed 100-tick windows from raw parquet,
and verifies output matches.

Goal: ensure the iter_012 inference reproduces T70 training-time predictions
(bit-exact since the feature pipeline is now confirmed identical).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUB = os.path.join(ROOT, "submission", "iter_012_v4_stage5")
CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
sys.path.insert(0, SUB)

from Predictor import Predictor

WINDOW = 100
N_RAW = 154

pred = Predictor()

test = np.load(os.path.join(CACHE, "schemeP_test.npz"))
X_cache = test["X"]
sym_arr = test["sym"]; date_arr = test["date"]
sess_arr = test["sess_idx"]; t_arr = test["t"]

with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
    feat_names_full = [line.strip() for line in f]

# Build the slicer matching T70 training (drop 11 FAIL names)
DROP = set(pred.__class__.__module__ and [])  # placeholder
from Predictor import FAIL_NAMES
keep_idx = np.array(
    [i for i, n in enumerate(feat_names_full) if n not in FAIL_NAMES],
    dtype=np.int64,
)
print(f"keep_idx len = {len(keep_idx)} (expect 359)")

# Sample 256 rows
np.random.seed(7)
sample = np.random.choice(len(X_cache), size=256, replace=False)
X_train_eq = X_cache[sample][:, keep_idx].astype(np.float32)

# Run all 5 T70 boosters
seeds = [1, 7, 13, 42, 100]
boosters = [lgb.Booster(model_file=os.path.join(SUB, f"model_h60_seed{s}.txt"))
            for s in seeds]
probs_ref = np.mean([b.predict(X_train_eq).astype(np.float32) for b in boosters], axis=0)
print(f"reference ensemble probs shape: {probs_ref.shape}")

# Build 100-tick windows from raw parquets
sess_map = {0: "am", 1: "pm"}
batches = []
n_loaded = 0
for ix in sample:
    s = int(sym_arr[ix]); d = int(date_arr[ix])
    ss = int(sess_arr[ix]); tt = int(t_arr[ix])
    p = os.path.join(ROOT, "data", f"snapshot_sym{s}_date{d}_{sess_map[ss]}.parquet")
    if not os.path.exists(p):
        continue
    df = pd.read_parquet(p)
    if tt + 1 < WINDOW or tt + 1 > len(df):
        continue
    win = df.iloc[tt + 1 - WINDOW : tt + 1].reset_index(drop=True)
    batches.append((ix, win))
    n_loaded += 1
print(f"loaded {n_loaded} windows")

# Reorder probs_ref to match the same indices as batches
sample_to_idx = {int(s): i for i, s in enumerate(sample)}
probs_ref_aligned = np.zeros((len(batches), 3), dtype=np.float32)
for k, (ix, _) in enumerate(batches):
    probs_ref_aligned[k] = probs_ref[sample_to_idx[int(ix)]]

# Run Predictor end-to-end
t0 = time.time()
preds_p = pred.predict([w for (_, w) in batches])
elapsed = time.time() - t0
print(f"Predictor.predict: {elapsed:.3f}s for {len(batches)} windows")

# Now we want to compare probs (not just argmax). Let's recompute via Predictor's internal pipeline:
feats = pred._compute_batch_features([w for (_, w) in batches])
ensemble_probs = pred._ensemble_predict(boosters, feats)
print(f"Predictor ensemble probs shape: {ensemble_probs.shape}")

diff = np.abs(probs_ref_aligned - ensemble_probs)
print(f"\n=== End-to-end probability match ===")
print(f"  max diff: {diff.max():.6g}")
print(f"  mean diff: {diff.mean():.6g}")

# Argmax agreement
argmax_ref = probs_ref_aligned.argmax(axis=1)
argmax_pred = ensemble_probs.argmax(axis=1)
agree = (argmax_ref == argmax_pred).sum()
print(f"  argmax agreement: {agree}/{len(argmax_ref)} = {agree/len(argmax_ref)*100:.2f}%")

ok = diff.max() < 1e-4
print(f"\nDecision: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
