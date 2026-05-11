"""T188v2 CORRECTED: Build submission package from 50 NN + 50 LGB models.

Uses CORRECT v2 thresholds:
  w_lgb=1.5 (NOT T150's 0.5)
  per_sym_beta = {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00} (NOT T150's 0.50 for sym1)
  conformal enabled = true
  thr_up=0.0003, thr_dn=0.000216

Base Predictor from iter_019 v2 (pkg_iter019_v2 directory).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
PKG_DIR = os.path.join(HERE, "pkg_T188v2_50plus50")
ZIP_NAME = "submission_050910_iter019_v2N_50plus50_proper.zip"
ZIP_OUT = os.path.join(ROOT, ZIP_NAME)

N_NN = 50
N_LGB = 50


def check_models():
    missing_nn = [s for s in range(1, N_NN + 1)
                  if not os.path.isfile(os.path.join(HERE, f"nn_h60_seed{s}.npz"))]
    missing_lgb = [s for s in range(1, N_LGB + 1)
                   if not os.path.isfile(os.path.join(HERE, f"model_h60_seed{s}.txt"))]
    if missing_nn:
        print(f"  MISSING NN npz: seeds {missing_nn[:10]}{'...' if len(missing_nn)>10 else ''}", flush=True)
    if missing_lgb:
        print(f"  MISSING LGB txt: seeds {missing_lgb[:10]}{'...' if len(missing_lgb)>10 else ''}", flush=True)
    return missing_nn, missing_lgb


def build():
    print("=== T188v2 CORRECTED Package Builder ===", flush=True)

    missing_nn, missing_lgb = check_models()
    n_nn = N_NN - len(missing_nn)
    n_lgb = N_LGB - len(missing_lgb)
    print(f"  NN available: {n_nn}/{N_NN}  LGB available: {n_lgb}/{N_LGB}", flush=True)
    if n_nn < N_NN or n_lgb < N_LGB:
        print(f"  WARNING: Using {n_nn} NN + {n_lgb} LGB (incomplete — consider waiting)", flush=True)

    if os.path.isdir(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    os.makedirs(PKG_DIR)
    print(f"  package dir: {PKG_DIR}", flush=True)

    # Copy static files from v2 base (Predictor handles N seeds dynamically)
    for fname in ["Predictor.py", "config.json", "fast_features.py",
                  "fast_features_batch.py", "requirements.txt"]:
        src = os.path.join(V2_PKG, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG_DIR, fname))
            print(f"  copied {fname}", flush=True)
        else:
            print(f"  MISSING BASE FILE: {fname}", flush=True)

    # Copy NN npz files
    nn_seeds = [s for s in range(1, N_NN + 1)
                if os.path.isfile(os.path.join(HERE, f"nn_h60_seed{s}.npz"))]
    for s in nn_seeds:
        shutil.copy2(
            os.path.join(HERE, f"nn_h60_seed{s}.npz"),
            os.path.join(PKG_DIR, f"nn_h60_seed{s}.npz"),
        )
    print(f"  copied {len(nn_seeds)} NN npz files (seeds {nn_seeds[:5]}...)", flush=True)

    # Copy LGB txt files
    lgb_seeds = [s for s in range(1, N_LGB + 1)
                 if os.path.isfile(os.path.join(HERE, f"model_h60_seed{s}.txt"))]
    for s in lgb_seeds:
        shutil.copy2(
            os.path.join(HERE, f"model_h60_seed{s}.txt"),
            os.path.join(PKG_DIR, f"model_h60_seed{s}.txt"),
        )
    print(f"  copied {len(lgb_seeds)} LGB txt files", flush=True)

    # Build thresholds.json: preserve EXACT v2 thresholds, only update ensemble_seeds
    with open(os.path.join(V2_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)

    # Apply CORRECT v2 thresholds (base thresholds.json may have T150 values; we override here)
    # Correct v2: w_lgb=1.5, per_sym_beta={0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00}
    tcfg["conformal_wrapper"]["per_sym_beta"] = {
        "0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00
    }
    for hcfg in tcfg["horizons"]:
        if hcfg.get("h") == 60 and hcfg.get("active", False):
            hcfg["w_lgb"] = 1.5
            hcfg["w_nn"] = 1.0
            hcfg["thr_up"] = 0.0003
            hcfg["thr_dn"] = 0.000216

            w_lgb = float(hcfg["w_lgb"])
            beta_1 = float(tcfg["conformal_wrapper"]["per_sym_beta"]["1"])
            print(f"  SET: w_lgb={w_lgb:.2f} beta_sym1={beta_1:.2f}", flush=True)

            # Update ensemble_seeds for all models (Predictor loads both NN and LGB by this list)
            all_seeds = sorted(set(nn_seeds) & set(lgb_seeds))
            if not all_seeds:
                all_seeds = sorted(set(nn_seeds) | set(lgb_seeds))
            hcfg["ensemble_seeds"] = all_seeds
            hcfg["_doc_t188v2"] = (
                f"T188v2 CORRECTED mega ensemble: {len(nn_seeds)} NN + {len(lgb_seeds)} LGB. "
                f"Phase1 pretrain on train(0-79), val(80-95). Phase2 SPO+ M7."
            )

    tcfg["_t188v2_build"] = {
        "n_nn": len(nn_seeds),
        "n_lgb": len(lgb_seeds),
        "nn_seeds": nn_seeds,
        "lgb_seeds": lgb_seeds,
        "thresholds_v2_correct": True,
        "w_lgb": 1.5,
        "per_sym_beta": {"0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00},
    }

    with open(os.path.join(PKG_DIR, "thresholds.json"), "w") as f:
        json.dump(tcfg, f, indent=2)
    print(f"  thresholds.json written ({len(all_seeds)} shared seeds)", flush=True)

    # Smoke test
    print("\n  Running smoke test...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(PKG_DIR, "Predictor.py")],
        capture_output=True, text=True, timeout=120, cwd=PKG_DIR,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  SMOKE TEST PASSED ({elapsed:.1f}s)", flush=True)
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}", flush=True)
    else:
        print(f"  SMOKE TEST FAILED:", flush=True)
        print(result.stderr[-2000:], flush=True)
        return False

    # Inference time benchmark (1024 batch)
    print("\n  Running inference benchmark (1024 rows)...", flush=True)
    bench_script = os.path.join(HERE, "bench_inference_v2.py")
    with open(bench_script, "w") as bf:
        bf.write(f'''import sys, time, json, os
sys.path.insert(0, "{PKG_DIR}")
from Predictor import Predictor
import numpy as np, pandas as pd

WINDOW = 100
rng = np.random.default_rng(42)
cfg = json.load(open("{PKG_DIR}/config.json"))
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
print(f"Predictor loaded in {{time.time()-t_load:.1f}}s", flush=True)

# Warmup
_ = p.predict(batches[:4])

# Benchmark 3 runs
times = []
for run in range(3):
    t0 = time.time()
    out = p.predict(batches)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  run {{run}}: {{elapsed*1000:.1f}} ms for {{B}} rows = {{elapsed*1000/B:.2f}} ms/row", flush=True)
    assert len(out) == B, f"Expected {{B}} outputs, got {{len(out)}}"

best_ms = min(times) * 1000
ms_per_row = best_ms / B

print(f"\\nBest inference time: {{best_ms:.1f}} ms for {{B}} rows")
print(f"Estimated platform time (442k rows, 1024 batch): {{best_ms/1000 * (442000/B) / 60:.1f}} min")
print(f"ms_per_1024: {{best_ms:.1f}}")

# Sanity check predictions
vals_h60 = [x[4] for x in out]
n0 = vals_h60.count(0); n1 = vals_h60.count(1); n2 = vals_h60.count(2)
n_active = n0 + n2
pct_active = n_active / B * 100
print(f"h60 actions: 0={{n0}} 1={{n1}} 2={{n2}}  active={{pct_active:.1f}}%")

# Platform time estimate: 1024 batches × 1024 rows
n_batches_platform = 442000 // 1024 + 1  # rough estimate
est_platform_sec = min(times) * n_batches_platform
print(f"Extrapolated platform time ({{n_batches_platform}} batches): {{est_platform_sec/60:.1f}} min")
if est_platform_sec > 3 * 3600:
    print("WARNING: Estimated time > 3h! May be too slow for platform.")
else:
    print("OK: Estimated time < 3h")
''')

    result2 = subprocess.run(
        [sys.executable, bench_script],
        capture_output=True, text=True, timeout=600,
    )
    bench_ok = result2.returncode == 0
    if bench_ok:
        print(f"  BENCHMARK:\n  {result2.stdout.strip()}", flush=True)
    else:
        print(f"  BENCHMARK FAILED: {result2.stderr[-1000:]}", flush=True)

    # File size
    total_sz = 0
    for r, dirs, files in os.walk(PKG_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            total_sz += os.path.getsize(os.path.join(r, fn))
    print(f"\n  Package size: {total_sz/1e6:.1f} MB", flush=True)
    if total_sz > 2e9:
        print(f"  ERROR: Package exceeds 2GB limit!", flush=True)
        return False

    # Build ZIP
    print(f"\n  Building zip: {ZIP_OUT}", flush=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for r, dirs, files in os.walk(PKG_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                fp = os.path.join(r, fn)
                arcname = os.path.relpath(fp, PKG_DIR)
                zf.write(fp, arcname)
    zip_sz = os.path.getsize(ZIP_OUT)
    print(f"  Zip size: {zip_sz/1e6:.1f} MB", flush=True)

    h = hashlib.md5()
    with open(ZIP_OUT, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    md5 = h.hexdigest()
    print(f"  MD5: {md5}", flush=True)

    # Parse benchmark output
    bench_ms = -1.0
    if bench_ok:
        for line in result2.stdout.split("\n"):
            if "ms_per_1024" in line:
                try:
                    bench_ms = float(line.split(":")[-1].strip())
                except:
                    pass

    results = {
        "task": "T188v2 CORRECTED mega ensemble 50+50 (proper T81 pretrain)",
        "phase1_val_split": "schemeP_val (dates 80-95)",
        "phase2_protocol": "SPO+ M7 11 epochs lambda_spo=30",
        "n_nn": len(nn_seeds),
        "n_lgb": len(lgb_seeds),
        "nn_seeds": nn_seeds,
        "lgb_seeds": lgb_seeds,
        "thresholds_v2_correct": True,
        "w_lgb": 1.5,
        "per_sym_beta": {"0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00},
        "pkg_dir": PKG_DIR,
        "zip": ZIP_OUT,
        "zip_size_mb": round(zip_sz / 1e6, 1),
        "pkg_size_mb": round(total_sz / 1e6, 1),
        "md5": md5,
        "smoke_test": "PASS",
        "inference_time_ms_per_1024": bench_ms,
    }
    res_path = os.path.join(HERE, "results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results.json -> {res_path}", flush=True)
    print(f"\n=== T188v2 BUILD COMPLETE ===", flush=True)
    print(f"  Zip: {ZIP_OUT}", flush=True)
    print(f"  MD5: {md5}", flush=True)
    print(f"  Size: {zip_sz/1e6:.1f} MB", flush=True)
    return True


if __name__ == "__main__":
    ok = build()
    sys.exit(0 if ok else 1)
