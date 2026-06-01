import sys, time, json, os
sys.path.insert(0, "/root/projects/liangwenbei_workdir/experiments/T188v2_mega_proper/pkg_T188v2_50plus50")
from Predictor import Predictor
import numpy as np, pandas as pd

WINDOW = 100
rng = np.random.default_rng(42)
cfg = json.load(open("/root/projects/liangwenbei_workdir/experiments/T188v2_mega_proper/pkg_T188v2_50plus50/config.json"))
feats = cfg["feature"]
B = 1024

batches = []
for i in range(B):
    df = pd.DataFrame(rng.standard_normal((WINDOW, len(feats))).astype(np.float32), columns=feats)
    df["sym"] = int(i % 5)
    batches.append(df)

print("Loading Predictor (50+50 ensemble)...", flush=True)
t_load = time.time()
p = Predictor()
print(f"Predictor loaded in {time.time()-t_load:.1f}s", flush=True)

# Warmup
_ = p.predict(batches[:4])

# Benchmark 3 runs
times = []
for run in range(3):
    t0 = time.time()
    out = p.predict(batches)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  run {run}: {elapsed*1000:.1f} ms for {B} rows = {elapsed*1000/B:.2f} ms/row", flush=True)
    assert len(out) == B, f"Expected {B} outputs, got {len(out)}"

best_ms = min(times) * 1000
ms_per_row = best_ms / B

print(f"\nBest inference time: {best_ms:.1f} ms for {B} rows")
print(f"Estimated platform time (442k rows, 1024 batch): {best_ms/1000 * (442000/B) / 60:.1f} min")
print(f"ms_per_1024: {best_ms:.1f}")

# Sanity check predictions
vals_h60 = [x[4] for x in out]
n0 = vals_h60.count(0); n1 = vals_h60.count(1); n2 = vals_h60.count(2)
n_active = n0 + n2
pct_active = n_active / B * 100
print(f"h60 actions: 0={n0} 1={n1} 2={n2}  active={pct_active:.1f}%")

# Platform time estimate: 1024 batches × 1024 rows
n_batches_platform = 442000 // 1024 + 1  # rough estimate
est_platform_sec = min(times) * n_batches_platform
print(f"Extrapolated platform time ({n_batches_platform} batches): {est_platform_sec/60:.1f} min")
if est_platform_sec > 3 * 3600:
    print("WARNING: Estimated time > 3h! May be too slow for platform.")
else:
    print("OK: Estimated time < 3h")
