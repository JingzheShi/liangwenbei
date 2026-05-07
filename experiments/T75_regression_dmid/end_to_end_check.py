"""End-to-end: pass cache rows through T75 regression boosters directly vs through Predictor.

Loads ~256 rows from schemeP_test.npz, drops the 11 FAIL columns, runs each
booster's predict to get reference Δmid_norm. Then runs the iter_013 Predictor
on reconstructed 100-tick windows from raw parquet, and compares the
ensemble pred_dmid_norm and the resulting EV-gate actions.

Goal: confirm iter_013 inference reproduces T75 training-time predictions.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG = os.path.join(os.path.dirname(__file__), "iter013_pkg")
CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
sys.path.insert(0, PKG)

from Predictor import Predictor, FAIL_NAMES  # type: ignore

WINDOW = 100
N_RAW = 154
SEEDS = [1, 7, 13, 42, 100]

pred = Predictor()

test = np.load(os.path.join(CACHE, "schemeP_test.npz"))
X_cache = test["X"]
sym_arr = test["sym"]; date_arr = test["date"]
sess_arr = test["sess_idx"]; t_arr = test["t"]

with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
    feat_names_full = [line.strip() for line in f]

keep_idx = np.array(
    [i for i, n in enumerate(feat_names_full) if n not in FAIL_NAMES],
    dtype=np.int64,
)
print(f"keep_idx len = {len(keep_idx)} (expect 359)")

np.random.seed(7)
sample = np.random.choice(len(X_cache), size=256, replace=False)
X_train_eq = X_cache[sample][:, keep_idx].astype(np.float32)

boosters = [lgb.Booster(model_file=os.path.join(PKG, f"model_h60_seed{s}.txt"))
            for s in SEEDS]
ref_pred = np.mean([b.predict(X_train_eq).astype(np.float32) for b in boosters], axis=0)
print(f"reference ensemble pred shape: {ref_pred.shape}  mean={ref_pred.mean():.6f} std={ref_pred.std():.6f}")

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

sample_to_idx = {int(s): i for i, s in enumerate(sample)}
ref_aligned = np.zeros(len(batches), dtype=np.float32)
for k, (ix, _) in enumerate(batches):
    ref_aligned[k] = ref_pred[sample_to_idx[int(ix)]]

# Run end-to-end Predictor (action output)
t0 = time.time()
preds_actions = pred.predict([w for (_, w) in batches])
elapsed = time.time() - t0
print(f"Predictor.predict: {elapsed:.3f}s for {len(batches)} windows")

# Internal: extract features then run boosters via Predictor's _ensemble_predict
feats = pred._compute_batch_features([w for (_, w) in batches])
ens_pred_dmid = pred._ensemble_predict(boosters, feats)
print(f"Predictor ensemble pred_dmid shape: {ens_pred_dmid.shape}")

diff = np.abs(ref_aligned - ens_pred_dmid)
print(f"\n=== End-to-end pred_dmid match ===")
print(f"  max abs diff: {diff.max():.6g}")
print(f"  mean abs diff: {diff.mean():.6g}")
print(f"  ref mean/std:  {ref_aligned.mean():.6f} / {ref_aligned.std():.6f}")
print(f"  pred mean/std: {ens_pred_dmid.mean():.6f} / {ens_pred_dmid.std():.6f}")

# Action agreement (using thresholds.json which Predictor loaded)
import json
with open(os.path.join(PKG, "thresholds.json")) as f:
    tcfg = json.load(f)
hcfg = next(h for h in tcfg["horizons"] if h.get("active"))
thr_up = float(hcfg["thr_up"]); thr_dn = float(hcfg["thr_dn"])
ref_actions = np.full(len(ref_aligned), 1, dtype=np.int64)
ref_actions[ref_aligned > thr_up] = 2
ref_actions[ref_aligned < -thr_dn] = 0

# extract h=60 column from preds_actions
preds_h60 = np.array([row[4] for row in preds_actions], dtype=np.int64)
agree = (ref_actions == preds_h60).sum()
print(f"  action agreement: {agree}/{len(preds_h60)} = {agree/len(preds_h60)*100:.2f}%")

ok = (diff.max() < 1e-4) and (agree == len(preds_h60))
print(f"\nDecision: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
