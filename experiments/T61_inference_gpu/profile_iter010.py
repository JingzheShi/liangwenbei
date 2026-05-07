"""Profile iter_010 inference time. Match the iter_009 profile harness.

The platform calls predict(batches) in chunks; we benchmark with a 1024-window
batch to measure wall-clock time. Then extrapolate to 442k samples.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
ITER10 = os.path.join(ROOT, "submission", "iter_010_fast")
sys.path.insert(0, ITER10)

# Load Predictor
from Predictor import Predictor as P10

# Make sure we don't accidentally have iter_009 imported
sys.path.insert(0, os.path.join(ROOT, "submission", "iter_009_t59_fullsym"))

p = P10()
print(f"Booster lists: {[(h, len(bs)) for h, bs in p._booster_lists.items()]}")

cfg = json.load(open(os.path.join(ITER10, "config.json")))
raw_cols = cfg["feature"]

# Load 2 sessions and build many windows
df1 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet"))
df2 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date100_am.parquet"))

# Build a list of 1024 100-tick windows
rng = np.random.default_rng(0)
N = 1024
windows = []
for _ in range(N):
    use_df = df1 if rng.random() < 0.5 else df2
    end_i = int(rng.integers(99, len(use_df) - 1))
    df_w = use_df.iloc[end_i - 99:end_i + 1].reset_index(drop=True).copy()
    windows.append(df_w[raw_cols])

# Warm-up
_ = p.predict(windows[:64])

# Time predict for 1024 windows
t0 = time.time()
out = p.predict(windows)
t1 = time.time() - t0
ms_per = t1 / N * 1000
print(f"\n=== 1024 windows ===")
print(f"  total: {t1:.3f}s   ms/sample: {ms_per:.3f}")

# Estimate for 442k
total_s = ms_per / 1000 * 442000
print(f"\n=== 442k extrapolation ===")
print(f"  est_total_s: {total_s:.1f}s   est_total_min: {total_s/60:.2f}")

# Save
out_d = {
    "n_windows": N,
    "total_s_per_1024": t1,
    "ms_per_sample": ms_per,
    "n_total_assumed": 442000,
    "est_total_s": total_s,
    "est_total_min": total_s / 60,
}
with open(os.path.join(ROOT, "experiments", "T61_inference_gpu", "timing_iter010.json"), "w") as f:
    json.dump(out_d, f, indent=2)
print(f"\nSaved timing_iter010.json")

# Also break out feature extraction vs predict
t0 = time.time()
feats = p._compute_batch_features(windows)
t_feat = time.time() - t0
t0 = time.time()
for hcfg in p._horizons:
    if not hcfg.get("active", True):
        continue
    H = int(hcfg["h"])
    boosters = p._booster_lists.get(H)
    if boosters:
        _ = p._ensemble_predict(boosters, feats)
t_pred = time.time() - t0
print(f"\nBreakdown 1024 windows:  feat={t_feat:.3f}s  predict={t_pred:.3f}s")

with open(os.path.join(ROOT, "experiments", "T61_inference_gpu", "timing_iter010.json"), "w") as f:
    out_d["feat_extract_s"] = t_feat
    out_d["predict_s"] = t_pred
    json.dump(out_d, f, indent=2)
