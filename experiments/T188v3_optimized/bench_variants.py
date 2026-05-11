"""Benchmark all NN inference variants for T188v3.

Variants tested:
  V0: T188v2 baseline - sequential numpy (50 NNs in a loop)
  V1: Batched numpy - stack all 50 weight matrices, single np.matmul per layer
  V2: torch CPU batched bmm
  V3: torch CUDA batched bmm (T188v3 final implementation)

Also verifies numerical match between variants.
"""
import sys, time, json, os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PKG_V2 = "/root/projects/liangwenbei_workdir/experiments/T188v2_mega_proper/pkg_T188v2_50plus50"
PKG_V3 = "/root/projects/liangwenbei_workdir/experiments/T188v3_optimized_pkg"

sys.path.insert(0, PKG_V2)
import importlib.util

def load_mod(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ------------- load NN weights -------------
nn_paths = sorted(glob.glob(os.path.join(PKG_V2, "nn_h60_seed*.npz")))
N = len(nn_paths)
print(f"Loading {N} NN models...")

d0 = np.load(nn_paths[0], allow_pickle=False)
in_dim = int(d0["in_dim"][0])
hidden = list(d0["hidden"].tolist())
use_ln = bool(d0["use_layernorm"][0])

feat_means, feat_stds, clips, target_scales = [], [], [], []
lW = [[] for _ in hidden]
lb = [[] for _ in hidden]
lLNW = [[] for _ in hidden]
lLNb = [[] for _ in hidden]
WF_list, bF_list = [], []

for path in nn_paths:
    d = np.load(path, allow_pickle=False)
    feat_means.append(d["feat_mean"].astype(np.float32))
    feat_stds.append(d["feat_std"].astype(np.float32))
    clips.append(float(d["clip"][0]))
    target_scales.append(float(d["target_scale"][0]))
    for i in range(len(hidden)):
        lW[i].append(d[f"L{i}_W"].astype(np.float32))
        lb[i].append(d[f"L{i}_b"].astype(np.float32))
        if use_ln:
            lLNW[i].append(d[f"LN{i}_W"].astype(np.float32))
            lLNb[i].append(d[f"LN{i}_b"].astype(np.float32))
    WF_list.append(d["LF_W"].astype(np.float32))
    bF_list.append(d["LF_b"].astype(np.float32))

keep_idx = d0["keep_idx"].astype(np.int64)
clip_val = float(max(clips))

# Numpy stacked arrays
np_feat_mean = np.stack(feat_means, 0)    # (N, in_dim)
np_feat_std  = np.stack(feat_stds, 0)     # (N, in_dim)
np_ts = np.array(target_scales, np.float32).reshape(N, 1)  # (N, 1)
np_W  = [np.stack(lW[i], 0) for i in range(len(hidden))]   # list of (N, out, in)
np_b  = [np.stack(lb[i], 0) for i in range(len(hidden))]   # list of (N, out)
if use_ln:
    np_LNW = [np.stack(lLNW[i], 0) for i in range(len(hidden))]
    np_LNb = [np.stack(lLNb[i], 0) for i in range(len(hidden))]
np_WF = np.stack(WF_list, 0)   # (N, 1, last_dim)
np_bF = np.stack(bF_list, 0)   # (N, 1)

# Torch tensors (CPU)
dev_cpu = torch.device("cpu")
dev_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU device: {dev_gpu}")

def to_t(arr, dev): return torch.from_numpy(arr).to(dev)

# ------------- individual NN objects (for V0 baseline) -------------
from Predictor import _MLPNumpy
nn_objs = [_MLPNumpy(p) for p in nn_paths]

# ------------- synthetic benchmark data -------------
WINDOW = 100
cfg = json.load(open(os.path.join(PKG_V2, "config.json")))
feats_cols = cfg["feature"]
rng = np.random.default_rng(42)
B = 1024
batches = []
for i in range(B):
    df = pd.DataFrame(rng.standard_normal((WINDOW, len(feats_cols))).astype(np.float32), columns=feats_cols)
    df["sym"] = int(i % 5)
    batches.append(df)

# Build feature matrix (same logic as Predictor._compute_batch_features)
# Load T188v3 Predictor via importlib to avoid module cache conflict with V2
_pred_v3_spec = importlib.util.spec_from_file_location("pred_v3_mod", os.path.join(PKG_V3, "Predictor.py"))
_pred_v3_mod = importlib.util.module_from_spec(_pred_v3_spec)
_pred_v3_spec.loader.exec_module(_pred_v3_mod)
PredV3 = _pred_v3_mod.Predictor
_BatchedMLPEnsemble = _pred_v3_mod._BatchedMLPEnsemble
RAW_COLS = list(_pred_v3_mod.RAW_COLS_TRAIN_ORDER)
FAIL_NAMES = set(_pred_v3_mod.FAIL_NAMES)

ffb_mod = load_mod(os.path.join(PKG_V2, "fast_features_batch.py"), "ffb_bench")
raw_feat_cols = list(PredV3.__init__.__globals__["RAW_COLS_TRAIN_ORDER"])
raw_col_to_idx = {c: i for i, c in enumerate(raw_feat_cols)}
amount_delta_idx = raw_feat_cols.index("amount_delta")
col_idx = dict(raw_col_to_idx)
if "midprice1" in raw_col_to_idx:
    col_idx["midprice"] = raw_col_to_idx["midprice1"]
extra_names = list(ffb_mod.all_feature_names())
extra_keep_idx = np.array([i for i, n in enumerate(extra_names) if n not in FAIL_NAMES], dtype=np.int64)

K = len(raw_feat_cols)
X3d = np.empty((B, WINDOW, K), dtype=np.float64)
for n, df in enumerate(batches):
    X3d[n] = df[raw_feat_cols].to_numpy(dtype=np.float64, copy=False)
raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
v = raw_last[:, amount_delta_idx]
raw_last[:, amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))
extras_full = ffb_mod.compute_batch_features(X3d, col_idx)
extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32)
extras_kept = extras_full[:, extra_keep_idx]
feats = np.concatenate([raw_last, extras_kept], axis=1)  # (B, total_feats)
print(f"Feature matrix shape: {feats.shape}")

# Apply feature selection (keep_idx)
if feats.shape[1] != in_dim:
    Xs = feats[:, keep_idx].astype(np.float32)
else:
    Xs = feats.astype(np.float32, copy=True)
print(f"NN input shape: {Xs.shape}")

# ============================================================
# V0: Sequential numpy (baseline - T188v2)
# ============================================================
def _gelu_tanh_np(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

def _layernorm_np(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b

def v0_sequential_numpy(Xs_in):
    acc = None
    for nn in nn_objs:
        h = Xs_in.astype(np.float32, copy=True)
        h = (h - nn.feat_mean) / nn.feat_std
        h = np.where(np.isnan(h), 0.0, h)
        h = np.clip(h, -nn.clip, nn.clip).astype(np.float32)
        for i in range(len(nn.hidden)):
            h = h @ nn.W[i].T + nn.b[i]
            if nn.use_layernorm:
                h = _layernorm_np(h, nn.LN_W[i], nn.LN_b[i])
            h = _gelu_tanh_np(h)
        out = (h @ nn.WF.T + nn.bF).squeeze(-1) / nn.target_scale
        acc = out if acc is None else acc + out
    return acc / N

# ============================================================
# V1: Batched numpy (vectorized, all 50 NNs simultaneously)
# ============================================================
def v1_batched_numpy(Xs_in):
    # h: (B, in_dim) -> broadcast to (N, B, in_dim) via transpose tricks
    Xs_np = Xs_in.astype(np.float32, copy=True)
    # Normalize: np_feat_mean (N, in_dim), expand to (N, B, in_dim)
    h = (Xs_np[np.newaxis] - np_feat_mean[:, np.newaxis]) / np_feat_std[:, np.newaxis]  # (N, B, in_dim)
    h = np.where(np.isnan(h), 0.0, h)
    h = np.clip(h, -clip_val, clip_val).astype(np.float32)
    for i in range(len(hidden)):
        # np_W[i]: (N, out, in), h: (N, B, in) -> matmul -> (N, B, out)
        h = np.matmul(h, np_W[i].transpose(0, 2, 1)) + np_b[i][:, np.newaxis]
        if use_ln:
            mean = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = (h - mean) / np.sqrt(var + 1e-5) * np_LNW[i][:, np.newaxis] + np_LNb[i][:, np.newaxis]
        # GELU tanh
        c = 0.7978845608028654
        h = 0.5 * h * (1.0 + np.tanh(c * (h + 0.044715 * h * h * h)))
    # Final: np_WF (N, 1, last), h (N, B, last) -> (N, B, 1) -> (N, B)
    out = np.matmul(h, np_WF.transpose(0, 2, 1)).squeeze(-1) + np_bF  # (N, B)
    out = out / np_ts  # (N, B)
    return out.mean(0)  # (B,)

# ============================================================
# V2: torch CPU batched bmm
# ============================================================
def make_torch_batched(dev):
    def to_t(arr): return torch.from_numpy(arr).to(dev)
    fm = to_t(np_feat_mean)
    fs = to_t(np_feat_std)
    ts = to_t(np_ts)
    Ws = [to_t(np_W[i]) for i in range(len(hidden))]
    bs = [to_t(np_b[i]) for i in range(len(hidden))]
    if use_ln:
        LNWs = [to_t(np_LNW[i]) for i in range(len(hidden))]
        LNbs = [to_t(np_LNb[i]) for i in range(len(hidden))]
    WF_t = to_t(np_WF)
    bF_t = to_t(np_bF)

    @torch.no_grad()
    def predict(Xs_in):
        h = torch.from_numpy(Xs_in.astype(np.float32)).to(dev)
        h = h.unsqueeze(0).expand(N, -1, -1).contiguous()
        h = (h - fm.unsqueeze(1)) / fs.unsqueeze(1)
        h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
        h = torch.clamp(h, -clip_val, clip_val)
        for i in range(len(hidden)):
            h = torch.bmm(h, Ws[i].transpose(-1, -2)) + bs[i].unsqueeze(1)
            if use_ln:
                mean = h.mean(dim=-1, keepdim=True)
                var = h.var(dim=-1, keepdim=True, unbiased=False)
                h = (h - mean) / torch.sqrt(var + 1e-5)
                h = h * LNWs[i].unsqueeze(1) + LNbs[i].unsqueeze(1)
            h = F.gelu(h, approximate="tanh")
        out = torch.bmm(h, WF_t.transpose(-1, -2)).squeeze(-1) + bF_t
        out = out / ts
        return out.mean(0).cpu().numpy().astype(np.float32)

    return predict

v2_cpu = make_torch_batched(dev_cpu)
v3_gpu = make_torch_batched(dev_gpu) if str(dev_gpu) != "cpu" else None

# ============================================================
# Correctness check
# ============================================================
print("\n--- Correctness check ---")
ref = v0_sequential_numpy(Xs)
v1_out = v1_batched_numpy(Xs)
v2_out = v2_cpu(Xs)
v3_out = v3_gpu(Xs) if v3_gpu else None

print(f"V1 vs V0 max|diff|: {np.abs(v1_out - ref).max():.2e}")
print(f"V2 vs V0 max|diff|: {np.abs(v2_out - ref).max():.2e}")
if v3_out is not None:
    print(f"V3 vs V0 max|diff|: {np.abs(v3_out - ref).max():.2e}")

TOLERANCE = 1e-4
assert np.abs(v1_out - ref).max() < TOLERANCE, "V1 failed tolerance check!"
assert np.abs(v2_out - ref).max() < TOLERANCE, "V2 failed tolerance check!"
if v3_out is not None:
    assert np.abs(v3_out - ref).max() < TOLERANCE, "V3 failed tolerance check!"
print("All variants pass tolerance check (< 1e-4)")

# ============================================================
# Benchmark
# ============================================================
WARMUP = 2
RUNS = 5

def bench(fn, name, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn(Xs)
    if str(dev_gpu) != "cpu" and "GPU" in name:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(Xs)
        if str(dev_gpu) != "cpu" and "GPU" in name:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    best = min(times) * 1000
    print(f"  {name}: best={best:.1f}ms  avg={np.mean(times)*1000:.1f}ms  ({B} rows)")
    return best

print(f"\n--- Benchmark (B={B}) ---")
ms_v0 = bench(v0_sequential_numpy, "V0 Sequential numpy")
ms_v1 = bench(v1_batched_numpy,    "V1 Batched numpy   ")
ms_v2 = bench(v2_cpu,              "V2 torch CPU bmm   ")
ms_v3 = bench(v3_gpu, "V3 torch GPU bmm   ") if v3_gpu else None

print("\n--- Summary ---")
print(f"V0 baseline: {ms_v0:.1f} ms")
print(f"V1 batched numpy: {ms_v1:.1f} ms  ({ms_v0/ms_v1:.1f}x speedup)")
print(f"V2 torch CPU:     {ms_v2:.1f} ms  ({ms_v0/ms_v2:.1f}x speedup)")
if ms_v3:
    print(f"V3 torch CUDA:    {ms_v3:.1f} ms  ({ms_v0/ms_v3:.1f}x speedup)")

# Also benchmark full Predictor (end-to-end) for T188v3
print("\n--- End-to-end Predictor benchmark ---")
_spec_v3 = importlib.util.spec_from_file_location("pred_v3_full", os.path.join(PKG_V3, "Predictor.py"))
_mod_v3_full = importlib.util.module_from_spec(_spec_v3)
sys.path.insert(0, PKG_V3)  # needed for fast_features_batch import inside Predictor
_spec_v3.loader.exec_module(_mod_v3_full)
pred_v3 = _mod_v3_full.Predictor()
print(f"T188v3 Predictor device: {pred_v3._device}")

for _ in range(2):
    pred_v3.predict(batches[:4])

e2e_times = []
for run in range(RUNS):
    t0 = time.perf_counter()
    out_v3 = pred_v3.predict(batches)
    if str(pred_v3._device) != "cpu":
        torch.cuda.synchronize()
    e2e_times.append(time.perf_counter() - t0)
    print(f"  T188v3 run {run}: {e2e_times[-1]*1000:.1f}ms")

best_e2e_v3 = min(e2e_times) * 1000

# T188v2 baseline end-to-end
sys.path.insert(0, PKG_V2)
import importlib
spec = importlib.util.spec_from_file_location("pred_v2", os.path.join(PKG_V2, "Predictor.py"))
pred_mod_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred_mod_v2)
pred_v2 = pred_mod_v2.Predictor()

for _ in range(2):
    pred_v2.predict(batches[:4])
e2e_v2_times = []
for run in range(RUNS):
    t0 = time.perf_counter()
    out_v2 = pred_v2.predict(batches)
    e2e_v2_times.append(time.perf_counter() - t0)
best_e2e_v2 = min(e2e_v2_times) * 1000

print(f"\nT188v2 end-to-end best: {best_e2e_v2:.1f}ms")
print(f"T188v3 end-to-end best: {best_e2e_v3:.1f}ms")
print(f"End-to-end speedup: {best_e2e_v2/best_e2e_v3:.2f}x")

# Verify action match
h60_v2 = [x[4] for x in out_v2]
h60_v3 = [x[4] for x in out_v3]
n_match = sum(a == b for a, b in zip(h60_v2, h60_v3))
print(f"h60 action match: {n_match}/{B} ({100*n_match/B:.1f}%)")

n_batches_platform = 442000 // B + 1
est_v3_min = best_e2e_v3 / 1000 * n_batches_platform / 60
est_v2_min = best_e2e_v2 / 1000 * n_batches_platform / 60
print(f"\nEst. platform time T188v2: {est_v2_min:.1f} min ({n_batches_platform} batches)")
print(f"Est. platform time T188v3: {est_v3_min:.1f} min ({n_batches_platform} batches)")

# Write results
import json
results = {
    "task": "T188v3 optimize 50-NN ensemble inference",
    "baseline_T188v2_inference_ms_1024": ms_v0,
    "v0_sequential_numpy_ms_1024": ms_v0,
    "v1_batched_numpy_ms_1024": ms_v1,
    "v2_torch_cpu_ms_1024": ms_v2,
    "v3_torch_cuda_ms_1024": ms_v3,
    "v1_speedup_vs_baseline": round(ms_v0/ms_v1, 2),
    "v2_speedup_vs_baseline": round(ms_v0/ms_v2, 2),
    "v3_speedup_vs_baseline": round(ms_v0/ms_v3, 2) if ms_v3 else None,
    "best_variant": "V3 torch CUDA batched bmm" if ms_v3 else "V2 torch CPU batched bmm",
    "best_variant_nn_ms": ms_v3 if ms_v3 else ms_v2,
    "e2e_T188v2_ms_1024": best_e2e_v2,
    "e2e_T188v3_ms_1024": best_e2e_v3,
    "e2e_speedup": round(best_e2e_v2/best_e2e_v3, 2),
    "estimated_total_inference_min_for_442k_T188v2": round(est_v2_min, 1),
    "estimated_total_inference_min_for_442k_T188v3": round(est_v3_min, 1),
    "actions_match_T188v2_pct": round(100*n_match/B, 2),
    "actions_match_T188v2": n_match == B,
    "tolerance_check_passed": True,
    "max_abs_diff_v3_vs_v0": float(np.abs(v3_out - ref).max()) if v3_out is not None else None,
    "device": str(pred_v3._device),
    "note": "All 50 NNs run in single batched bmm per layer. CUDA 3090 available."
}

out_dir = "/root/projects/liangwenbei_workdir/experiments/T188v3_optimized"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "bench_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {out_dir}/bench_results.json")
