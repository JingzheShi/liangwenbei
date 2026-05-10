"""T188: Build submission package from 50 NN + 50 LGB models.

Copies v2 package structure, replaces 5-seed ensemble with 50+50,
updates thresholds.json with seeds 1-50, runs smoke test, zips.
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
PKG_DIR = os.path.join(HERE, "pkg_T188_50plus50")
ZIP_NAME = "submission_050910_iter019_v2N_50plus50_mega.zip"
ZIP_OUT = os.path.join(ROOT, ZIP_NAME)

N_NN = 50
N_LGB = 50


def check_models():
    missing_nn = [s for s in range(1, N_NN + 1)
                  if not os.path.isfile(os.path.join(HERE, f"nn_h60_seed{s}.npz"))]
    missing_lgb = [s for s in range(1, N_LGB + 1)
                   if not os.path.isfile(os.path.join(HERE, f"model_h60_seed{s}.txt"))]
    if missing_nn:
        print(f"  MISSING NN npz: {missing_nn}", flush=True)
    if missing_lgb:
        print(f"  MISSING LGB txt: {missing_lgb}", flush=True)
    return missing_nn, missing_lgb


def build():
    print("=== T188 Package Builder ===", flush=True)

    # Check models
    missing_nn, missing_lgb = check_models()
    n_nn = N_NN - len(missing_nn)
    n_lgb = N_LGB - len(missing_lgb)
    print(f"  NN available: {n_nn}/{N_NN}  LGB available: {n_lgb}/{N_LGB}", flush=True)
    if n_nn < N_NN or n_lgb < N_LGB:
        print(f"  WARNING: Using {n_nn} NN + {n_lgb} LGB (not full 50+50)", flush=True)

    # Build package directory
    if os.path.isdir(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    os.makedirs(PKG_DIR)
    print(f"  package dir: {PKG_DIR}", flush=True)

    # Copy static files from v2 (Predictor, features, config, requirements)
    for fname in ["Predictor.py", "config.json", "fast_features.py",
                  "fast_features_batch.py", "requirements.txt"]:
        src = os.path.join(V2_PKG, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG_DIR, fname))
            print(f"  copied {fname}", flush=True)
        else:
            print(f"  MISSING: {fname}", flush=True)

    # Copy NN npz files (seeds 1-50, skipping missing)
    nn_seeds = [s for s in range(1, N_NN + 1)
                if os.path.isfile(os.path.join(HERE, f"nn_h60_seed{s}.npz"))]
    for s in nn_seeds:
        shutil.copy2(
            os.path.join(HERE, f"nn_h60_seed{s}.npz"),
            os.path.join(PKG_DIR, f"nn_h60_seed{s}.npz"),
        )
    print(f"  copied {len(nn_seeds)} NN npz files", flush=True)

    # Copy LGB txt files (seeds 1-50, skipping missing)
    lgb_seeds = [s for s in range(1, N_LGB + 1)
                 if os.path.isfile(os.path.join(HERE, f"model_h60_seed{s}.txt"))]
    for s in lgb_seeds:
        shutil.copy2(
            os.path.join(HERE, f"model_h60_seed{s}.txt"),
            os.path.join(PKG_DIR, f"model_h60_seed{s}.txt"),
        )
    print(f"  copied {len(lgb_seeds)} LGB txt files", flush=True)

    # Build thresholds.json: copy v2 config but update ensemble_seeds to 1..50
    with open(os.path.join(V2_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    for hcfg in tcfg["horizons"]:
        if hcfg.get("h") == 60 and hcfg.get("active", True):
            hcfg["ensemble_seeds"] = nn_seeds  # same for NN and LGB (Predictor uses this for both)
            # Keep w_nn and w_lgb from v2 thresholds (T150-tuned: w_nn=1.0, w_lgb=0.5)
            # DO NOT touch these per user instruction "DO NOT touch thresholds"
            hcfg["_doc_t188"] = f"T188 mega ensemble: {len(nn_seeds)} NN + {len(lgb_seeds)} LGB"
    # Add T188 metadata
    tcfg["_t188_build"] = {
        "n_nn": len(nn_seeds),
        "n_lgb": len(lgb_seeds),
        "nn_seeds": nn_seeds,
        "lgb_seeds": lgb_seeds,
    }
    with open(os.path.join(PKG_DIR, "thresholds.json"), "w") as f:
        json.dump(tcfg, f, indent=2)
    print(f"  thresholds.json written (seeds 1-{max(nn_seeds) if nn_seeds else 0})", flush=True)

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
        print(f"  {result.stdout.strip()}", flush=True)
    else:
        print(f"  SMOKE TEST FAILED!", flush=True)
        print(result.stderr[-2000:], flush=True)
        return False

    # Inference time benchmark (1024 batch)
    print("\n  Running inference time benchmark...", flush=True)
    bench_script = os.path.join(HERE, "bench_inference.py")
    with open(bench_script, "w") as f:
        f.write(f'''import sys, time, json
sys.path.insert(0, "{PKG_DIR}")
from Predictor import Predictor
import numpy as np, pandas as pd

WINDOW = 100
rng = np.random.default_rng(0)
# Load config to get feature list
import json
cfg = json.load(open("{PKG_DIR}/config.json"))
feats = cfg["feature"]
B = 1024

batches = []
for _ in range(B):
    df = pd.DataFrame(rng.standard_normal((WINDOW, len(feats))).astype(np.float32), columns=feats)
    df["sym"] = int(rng.integers(0, 5))
    batches.append(df)

p = Predictor()
# Warmup
_ = p.predict(batches[:4])

# Benchmark 3 runs
times = []
for _ in range(3):
    t0 = time.time()
    out = p.predict(batches)
    times.append(time.time() - t0)
    assert len(out) == B

ms_per_batch = min(times) * 1000
print(f"Inference 1024-batch: {{min(times)*1000:.1f}} ms (best of 3)")
print(f"Estimated 442k rows: {{min(times) * 442000 / 1024 / 60:.1f}} min")

# Check prediction sanity
vals = [x[4] for x in out]  # h60 column
n0 = vals.count(0); n1 = vals.count(1); n2 = vals.count(2)
print(f"Actions h60: 0={n0} 1={n1} 2={n2}")
''')
    result2 = subprocess.run(
        [sys.executable, bench_script],
        capture_output=True, text=True, timeout=300,
    )
    if result2.returncode == 0:
        print(f"  BENCHMARK OK:", flush=True)
        print(f"  {result2.stdout.strip()}", flush=True)
    else:
        print(f"  BENCHMARK FAILED: {result2.stderr[-1000:]}", flush=True)

    # File size check
    total_sz = 0
    for root, dirs, files in os.walk(PKG_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            total_sz += os.path.getsize(os.path.join(root, fn))
    print(f"\n  Package size: {total_sz/1e6:.1f} MB", flush=True)
    if total_sz > 2e9:
        print(f"  ERROR: Package exceeds 2GB limit!", flush=True)
        return False

    # Build ZIP
    print(f"\n  Building zip: {ZIP_OUT}", flush=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for root, dirs, files in os.walk(PKG_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                fp = os.path.join(root, fn)
                arcname = os.path.relpath(fp, PKG_DIR)
                zf.write(fp, arcname)
    zip_sz = os.path.getsize(ZIP_OUT)
    print(f"  Zip size: {zip_sz/1e6:.1f} MB", flush=True)

    # MD5
    h = hashlib.md5()
    with open(ZIP_OUT, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    md5 = h.hexdigest()
    print(f"  MD5: {md5}", flush=True)

    # Write results
    results = {
        "task": "T188 mega ensemble 50+50 build",
        "n_nn": len(nn_seeds),
        "n_lgb": len(lgb_seeds),
        "nn_seeds": nn_seeds,
        "lgb_seeds": lgb_seeds,
        "pkg_dir": PKG_DIR,
        "zip": ZIP_OUT,
        "zip_size_mb": round(zip_sz / 1e6, 1),
        "pkg_size_mb": round(total_sz / 1e6, 1),
        "md5": md5,
        "smoke_test": "PASS",
    }
    if result2.returncode == 0:
        for line in result2.stdout.strip().split("\n"):
            if "1024-batch" in line:
                results["inference_bench"] = line
            if "Estimated" in line:
                results["inference_estimate"] = line

    res_path = os.path.join(HERE, "results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results.json written: {res_path}", flush=True)
    print(f"\n=== BUILD COMPLETE ===", flush=True)
    print(f"  Zip: {ZIP_OUT}", flush=True)
    print(f"  MD5: {md5}", flush=True)
    print(f"  Size: {zip_sz/1e6:.1f} MB", flush=True)
    return True


if __name__ == "__main__":
    ok = build()
    sys.exit(0 if ok else 1)
