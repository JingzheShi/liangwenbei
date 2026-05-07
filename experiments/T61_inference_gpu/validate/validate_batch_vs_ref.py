"""Validate fast_features_batch.py output matches iter_009 reference."""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
ITER9 = os.path.join(ROOT, "submission", "iter_009_t59_fullsym")
ITER10 = os.path.join(ROOT, "submission", "iter_010_fast")

sys.path.insert(0, ITER10)
import fast_features_batch as ffb

def _load(here, fname, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(here, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

t3 = _load(ITER9, "compute.py", "i9_t3")
s1 = _load(ITER9, "r34_features.py", "i9_s1")
s2 = _load(ITER9, "r34_stage2_features.py", "i9_s2")
s3 = _load(ITER9, "r34_stage3_features.py", "i9_s3")

cfg = json.load(open(os.path.join(ITER9, "config.json")))
raw_cols = cfg["feature"]
col_idx = {c: i for i, c in enumerate(raw_cols)}
print(f"raw_cols count: {len(raw_cols)}")

df1 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet"))
df2 = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date100_am.parquet"))

rng = np.random.default_rng(123)
N = 16
windows = []
for _ in range(N):
    use_df = df1 if rng.random() < 0.5 else df2
    end_i = int(rng.integers(99, len(use_df) - 1))
    df_w = use_df.iloc[end_i - 99:end_i + 1].reset_index(drop=True).copy()
    windows.append(df_w)

# === Reference path: per-window ===
ref_feat = np.zeros((N, 196), dtype=np.float64)
all_t3_cols = list(t3.feature_v1_columns())
t3_no_time_cols = [c for c in all_t3_cols if not c.startswith("time_")]
for n, df_window in enumerate(windows):
    mlofi = t3.compute_mlofi(df_window)
    wmp = t3.compute_wmp(df_window)
    rv = t3.compute_rv(wmp["wmp_lvl1"])
    ewma = t3.compute_ewma_intst(df_window)
    t3_df = pd.concat([mlofi, wmp, rv, ewma], axis=1)
    t3_last = t3_df[t3_no_time_cols].iloc[-1].to_numpy(dtype=np.float64)

    derived = {"midprice": df_window["midprice1"].to_numpy(dtype=np.float64, copy=False)}
    for k in (1, 5, 10):
        col = f"mlofi_W20_lvl{k}"
        derived[col] = t3_df[col].to_numpy(dtype=np.float64, copy=False)
    df_ext = df_window.assign(**derived)

    last = 99
    s1_v = s1.compute_all_session(df_ext, last, last).reshape(-1)
    s2_v = s2.compute_all_session(df_ext, last, last).reshape(-1)
    s3_v = s3.compute_all_session(df_ext, last, last).reshape(-1)
    ref_feat[n] = np.concatenate([t3_last, s1_v, s2_v, s3_v])

ref_feat = np.where(np.isfinite(ref_feat), ref_feat, 0.0)

# === Batch path ===
T = 100; K = len(raw_cols)
X3d = np.zeros((N, T, K), dtype=np.float64)
for n, w in enumerate(windows):
    for c in raw_cols:
        X3d[n, :, col_idx[c]] = w[c].to_numpy(dtype=np.float64, copy=False)

batch_feat = ffb.compute_batch_features(X3d, col_idx)
batch_feat = np.where(np.isfinite(batch_feat), batch_feat, 0.0)

# === Compare ===
diff = np.abs(batch_feat - ref_feat)
all_names = ffb.all_feature_names()
per_feat_max = diff.max(axis=0)
worst = np.argsort(-per_feat_max)[:15]
print("\n=== Per-feature max diff (top 15 worst) ===")
for i in worst:
    n_at = diff[:, i].argmax()
    print(f"  [{i}] {all_names[i]}: max_diff={per_feat_max[i]:.3e}  "
          f"ref={ref_feat[n_at, i]:.6f}  batch={batch_feat[n_at, i]:.6f}")

print(f"\nOverall max diff: {diff.max():.3e}")

# Relative diff on non-zero values
rel = diff / (np.abs(ref_feat) + 1e-6)
print(f"Max relative diff: {rel.max():.3e}")

# Acceptance: max diff < 1e-3 OR (max diff / scale) < 1e-5
ok = True
for i in range(196):
    scale = max(np.abs(ref_feat[:, i]).max(), 1.0)
    if per_feat_max[i] > 1e-3 and per_feat_max[i] / scale > 1e-5:
        print(f"  FAIL: {all_names[i]} diff={per_feat_max[i]:.3e} scale={scale:.3e}")
        ok = False
print(f"\n{'PASS' if ok else 'FAIL'}: batch features match reference (relative tol 1e-5 on float32 scale)")
