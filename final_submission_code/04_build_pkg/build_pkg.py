"""Build submission package from 50 NN + 50 LGB models.

Assembles:
  - Predictor.py (fullhorizon version with share_with support)
  - thresholds.json (h=60 primary + h=5/10/20/40 share_with=60)
  - config.json, fast_features.py, fast_features_batch.py, requirements.txt
  - 50 nn_h60_seed{1..50}.npz  (NN weights)
  - 50 model_h60_seed{1..50}.txt  (LGB boosters)

Usage:
    python3 build_pkg.py --models ./outputs/models --pkg ./outputs/pkg
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
N_NN = 50
N_LGB = 50


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models(models_dir: str):
    missing_nn = [s for s in range(1, N_NN + 1)
                  if not os.path.isfile(os.path.join(models_dir, f"nn_h60_seed{s}.npz"))]
    missing_lgb = [s for s in range(1, N_LGB + 1)
                   if not os.path.isfile(os.path.join(models_dir, f"model_h60_seed{s}.txt"))]
    if missing_nn:
        print(f"  MISSING NN npz: seeds {missing_nn[:10]}{'...' if len(missing_nn) > 10 else ''}",
              flush=True)
    if missing_lgb:
        print(f"  MISSING LGB txt: seeds {missing_lgb[:10]}{'...' if len(missing_lgb) > 10 else ''}",
              flush=True)
    return missing_nn, missing_lgb


def build(models_dir: str, pkg_dir: str):
    print("=== T188v2 50+50 Full-Horizon Package Builder ===", flush=True)
    print(f"  models: {models_dir}", flush=True)
    print(f"  pkg:    {pkg_dir}", flush=True)

    missing_nn, missing_lgb = check_models(models_dir)
    n_nn = N_NN - len(missing_nn)
    n_lgb = N_LGB - len(missing_lgb)
    print(f"  NN available: {n_nn}/{N_NN}  LGB available: {n_lgb}/{N_LGB}", flush=True)
    if n_nn < N_NN or n_lgb < N_LGB:
        print(f"  WARNING: Using {n_nn} NN + {n_lgb} LGB (incomplete ensemble)", flush=True)

    if os.path.isdir(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(pkg_dir)

    # Copy static files from this directory (04_build_pkg/)
    static_files = ["Predictor.py", "config.json", "fast_features.py",
                    "fast_features_batch.py", "requirements.txt", "thresholds.json"]
    for fname in static_files:
        src = os.path.join(HERE, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(pkg_dir, fname))
            print(f"  copied {fname}", flush=True)
        else:
            print(f"  ERROR: missing static file {fname}", flush=True)

    # Copy NN npz files
    nn_seeds = [s for s in range(1, N_NN + 1)
                if os.path.isfile(os.path.join(models_dir, f"nn_h60_seed{s}.npz"))]
    for s in nn_seeds:
        shutil.copy2(
            os.path.join(models_dir, f"nn_h60_seed{s}.npz"),
            os.path.join(pkg_dir, f"nn_h60_seed{s}.npz"),
        )
    print(f"  copied {len(nn_seeds)} NN npz files", flush=True)

    # Copy LGB txt files
    lgb_seeds = [s for s in range(1, N_LGB + 1)
                 if os.path.isfile(os.path.join(models_dir, f"model_h60_seed{s}.txt"))]
    for s in lgb_seeds:
        shutil.copy2(
            os.path.join(models_dir, f"model_h60_seed{s}.txt"),
            os.path.join(pkg_dir, f"model_h60_seed{s}.txt"),
        )
    print(f"  copied {len(lgb_seeds)} LGB txt files", flush=True)

    # Verify the thresholds.json has share_with entries
    thr_path = os.path.join(pkg_dir, "thresholds.json")
    with open(thr_path) as f:
        thr = json.load(f)
    horizons_with_share = [h["h"] for h in thr["horizons"] if h.get("share_with")]
    print(f"  thresholds: horizons with share_with = {horizons_with_share}", flush=True)

    # Count files
    total_files = sum(1 for _ in os.scandir(pkg_dir))
    print(f"\n  pkg dir total files: {total_files}", flush=True)

    manifest = {
        "n_nn": len(nn_seeds),
        "n_lgb": len(lgb_seeds),
        "nn_seeds": nn_seeds,
        "lgb_seeds": lgb_seeds,
        "static_files": static_files,
    }
    with open(os.path.join(pkg_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written", flush=True)

    print(f"\n=== Package built: {pkg_dir} ===", flush=True)
    print("Next step:", flush=True)
    print(f"  cd {pkg_dir} && zip -qr ../submission.zip .", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="./outputs/models",
                    help="Directory with nn_h60_seed*.npz and model_h60_seed*.txt")
    ap.add_argument("--pkg", default="./outputs/pkg",
                    help="Output package directory")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.pkg) if os.path.dirname(args.pkg) else ".", exist_ok=True)
    build(args.models, args.pkg)


if __name__ == "__main__":
    main()
