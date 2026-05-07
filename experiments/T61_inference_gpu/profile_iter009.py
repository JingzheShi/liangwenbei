"""Profile iter_009 inference: feature extraction + 5-model predict.

Goal: understand what dominates time for 442k samples (platform reality).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/root/projects/liangwenbei_workdir')
SUBMIT = ROOT / 'submission' / 'iter_009_t59_fullsym'
sys.path.insert(0, str(SUBMIT))

import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

t3 = _load('t3', SUBMIT / 'compute.py')
s1 = _load('s1', SUBMIT / 'r34_features.py')
s2 = _load('s2', SUBMIT / 'r34_stage2_features.py')
s3 = _load('s3', SUBMIT / 'r34_stage3_features.py')

import lightgbm as lgb

# load real data, build batches of 100-tick windows
print("Loading data...")
df_full = pd.read_parquet(ROOT / 'data' / 'snapshot_sym0_date0_am.parquet')
cfg = json.load(open(SUBMIT / 'config.json'))
feat_cols = cfg['feature']

# Make 100 batches of 100-tick windows from the data
N_BATCHES = 100
batches = []
for i in range(N_BATCHES):
    start = i * 10
    if start + 100 > len(df_full):
        start = i % (len(df_full) - 100)
    sub = df_full.iloc[start:start+100].reset_index(drop=True).copy()
    batches.append(sub)
print(f"Built {len(batches)} batches of 100 ticks each")

# Construct Predictor
from Predictor import Predictor
print("Loading Predictor...")
t0 = time.time()
P = Predictor()
print(f"  Predictor init: {time.time()-t0:.2f}s")
print(f"  Booster lists: { {h: len(v) for h,v in P._booster_lists.items()} }")

# Warmup
print("\n=== Warmup ===")
P.predict(batches[:8])
print("warmup done")

# Profile feature extraction alone
print("\n=== Profile feature extraction (1024 batches) ===")
# Make 1024 batches by repeating
big_batches = (batches * ((1024 // len(batches)) + 1))[:1024]

t0 = time.time()
feats = P._compute_batch_features(big_batches)
feat_t = time.time() - t0
print(f"  feature extract 1024 batches: {feat_t:.3f}s -> {feat_t*1000/1024:.3f}ms per batch")
print(f"  feats shape: {feats.shape}, dtype: {feats.dtype}")

# Profile predict alone (5-seed ensemble)
print("\n=== Profile 5-model ensemble predict ===")
boosters = P._booster_lists[60]
t0 = time.time()
probs = P._ensemble_predict(boosters, feats)
pred_t = time.time() - t0
print(f"  5-model predict 1024: {pred_t:.3f}s -> {pred_t*1000/1024:.3f}ms per sample")

# Profile single-model predict
print("\n=== Profile single-model predict ===")
t0 = time.time()
pp = boosters[0].predict(feats)
sm_t = time.time() - t0
print(f"  1-model predict 1024: {sm_t:.3f}s -> {sm_t*1000/1024:.3f}ms per sample")

# Full predict
print("\n=== Profile full predict ===")
t0 = time.time()
out = P.predict(big_batches)
full_t = time.time() - t0
print(f"  full predict 1024: {full_t:.3f}s")

# Estimate for 442k samples at the typical platform batch size
# Assume platform calls predict() many times with batch=1024
N_TOTAL = 442000
batches_per = N_TOTAL / 1024
est_total = full_t * batches_per
print(f"\n=== Estimate full inference for 442k samples ===")
print(f"  est total: {est_total:.0f}s = {est_total/60:.1f} min")

# breakdown per stage
results = {
    "n_batches_profiled": 1024,
    "feature_extract_s": feat_t,
    "feature_extract_ms_per_sample": feat_t*1000/1024,
    "ensemble5_predict_s": pred_t,
    "ensemble5_predict_ms_per_sample": pred_t*1000/1024,
    "single_model_predict_s": sm_t,
    "single_model_predict_ms_per_sample": sm_t*1000/1024,
    "full_predict_1024_s": full_t,
    "n_total_assumed": N_TOTAL,
    "est_total_s": est_total,
    "est_total_min": est_total/60,
    "feature_pct": feat_t/full_t*100,
    "predict_pct": pred_t/full_t*100,
}
out_p = ROOT / 'experiments' / 'T61_inference_gpu' / 'timing_cpu.json'
with open(out_p, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {out_p}")
print(json.dumps(results, indent=2))
