"""Validate fast_features.py single-window output matches iter_009 reference."""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
ITER9 = os.path.join(ROOT, "submission", "iter_009_t59_fullsym")
ITER10 = os.path.join(ROOT, "submission", "iter_010_fast")

sys.path.insert(0, ITER9)
sys.path.insert(0, ITER10)

import fast_features as ff

def _load(here, fname, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(here, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

t3 = _load(ITER9, "compute.py", "i9_t3")
s1 = _load(ITER9, "r34_features.py", "i9_s1")
s2 = _load(ITER9, "r34_stage2_features.py", "i9_s2")
s3 = _load(ITER9, "r34_stage3_features.py", "i9_s3")

cfg = json.load(open(os.path.join(ITER9, "config.json")))
raw_cols = cfg["feature"]
print(f"raw_cols count: {len(raw_cols)}")

# Load test session
df_full = pd.read_parquet(os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet"))
print(f"session shape: {df_full.shape}")

# Pick 3 random 100-row windows
rng = np.random.default_rng(42)
N = 5
indices = rng.integers(99, len(df_full)-1, size=N).tolist()
indices.sort()
print(f"window end indices: {indices}")

max_diffs = []
for end_i in indices:
    df_window = df_full.iloc[end_i-99:end_i+1].reset_index(drop=True).copy()
    assert len(df_window) == 100

    # === Reference path (iter_009 r34_features etc.) ===
    # First compute T3
    mlofi = t3.compute_mlofi(df_window)
    wmp = t3.compute_wmp(df_window)
    rv = t3.compute_rv(wmp["wmp_lvl1"])
    ewma = t3.compute_ewma_intst(df_window)
    t3_df = pd.concat([mlofi, wmp, rv, ewma], axis=1)
    all_t3_cols = list(t3.feature_v1_columns())
    t3_no_time_cols = [c for c in all_t3_cols if not c.startswith("time_")]
    t3_last_ref = t3_df[t3_no_time_cols].iloc[-1].to_numpy(dtype=np.float64)

    # Now build df_ext for stage1/2/3
    derived = {"midprice": df_window["midprice1"].to_numpy(dtype=np.float64, copy=False)}
    for k in (1, 5, 10):
        col = f"mlofi_W20_lvl{k}"
        derived[col] = t3_df[col].to_numpy(dtype=np.float64, copy=False)
    df_ext = df_window.assign(**derived)

    last = 99
    s1_ref = s1.compute_all_session(df_ext, last, last).reshape(-1)
    s2_ref = s2.compute_all_session(df_ext, last, last).reshape(-1)
    s3_ref = s3.compute_all_session(df_ext, last, last).reshape(-1)
    extras_ref = np.concatenate([s1_ref, s2_ref, s3_ref])
    extras_ref = np.where(np.isfinite(extras_ref), extras_ref, 0.0).astype(np.float64)

    # === Fast path (fast_features.py) ===
    # Build df_arr dict (need 'midprice' alias + mlofi_W20_lvl{k} as derived)
    df_arr = {}
    for col in df_window.columns:
        s = df_window[col]
        if pd.api.types.is_numeric_dtype(s):
            df_arr[col] = s.to_numpy(dtype=np.float64, copy=False)
    df_arr["midprice"] = df_arr["midprice1"]

    t3_v, derived_fast = ff.compute_t3_no_time_last(df_arr)
    s1_v = ff.compute_stage1_last(df_arr, derived_fast)
    s2_v = ff.compute_stage2_last(df_arr)
    s3_v = ff.compute_stage3_last(df_arr)
    extras_fast = np.concatenate([s1_v, s2_v, s3_v])
    extras_fast = np.where(np.isfinite(extras_fast), extras_fast, 0.0)

    # Compare
    t3_diff = np.abs(t3_last_ref - t3_v)
    extras_diff = np.abs(extras_ref - extras_fast)

    max_t3 = t3_diff.max()
    max_extras = extras_diff.max()
    print(f"\n[end={end_i}] T3 max_diff: {max_t3:.3e}  Extras max_diff: {max_extras:.3e}")

    # Find offending features
    if max_extras > 1e-3:
        s1_names = list(s1.all_feature_names())
        s2_names = list(s2.all_feature_names())
        s3_names = list(s3.all_feature_names())
        all_names = s1_names + s2_names + s3_names
        bad_idx = np.argsort(-extras_diff)[:10]
        print(f"  Top 10 worst extras:")
        for i in bad_idx:
            print(f"    [{i}] {all_names[i]}: ref={extras_ref[i]:.6f}  fast={extras_fast[i]:.6f}  diff={extras_diff[i]:.3e}")

    if max_t3 > 1e-3:
        bad_idx = np.argsort(-t3_diff)[:10]
        print(f"  Top 10 worst T3:")
        for i in bad_idx:
            print(f"    [{i}] {t3_no_time_cols[i]}: ref={t3_last_ref[i]:.6f}  fast={t3_v[i]:.6f}  diff={t3_diff[i]:.3e}")

    max_diffs.append((max_t3, max_extras))

print("\n=== Summary ===")
print(f"Max T3 diff across {N} windows: {max(d[0] for d in max_diffs):.3e}")
print(f"Max extras diff across {N} windows: {max(d[1] for d in max_diffs):.3e}")
