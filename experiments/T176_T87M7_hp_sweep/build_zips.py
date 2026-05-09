"""T176: Build submission zips for top variants.

Copies v2 pkg, replaces 5 nn_h60_seed*.npz with variant's npz files.
Keeps all v2 thresholds, LGB models, and Predictor unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
T170_DIR = os.path.join(ROOT, "experiments", "T170_T87_M7")
VARIANTS_DIR = os.path.join(HERE, "variants")
ZIPS_DIR = os.path.join(HERE, "zips")
SEEDS = (1, 7, 13, 42, 100)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def smoke_test(pkg_dir):
    """Quick smoke test: import Predictor, instantiate, call predict."""
    test_script = f"""
import sys
sys.path.insert(0, '{pkg_dir}')
import numpy as np
from Predictor import Predictor

# Minimal smoke test: build small fake data
p = Predictor()
rng = np.random.default_rng(0)
# Predictor.predict(X, date=0) where X is (N, n_raw_features)
# Use 5 rows, sym=0-4
n = 5
# We need correct feature shape - check config
import json
with open('{pkg_dir}/config.json') as f:
    cfg = json.load(f)
n_feats = len(cfg['feature'])
X = rng.normal(size=(n, n_feats)).astype(np.float32)
syms = np.arange(n)
result = p.predict(X, date=0, sym=syms)
print(f'smoke test OK: result shape={{result.shape}} min={{result.min():.4f}} max={{result.max():.4f}}')
"""
    r = subprocess.run([sys.executable, "-c", test_script],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"  SMOKE TEST FAILED:\n{r.stdout}\n{r.stderr}")
        return False
    print(f"  {r.stdout.strip()}")
    return True


def build_zip_for_variant(variant_name, nn_dir, zip_name):
    """Build a zip from v2 pkg with replacement NNs."""
    tmp_dir = os.path.join(HERE, f"_tmp_pkg_{variant_name}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copytree(V2_PKG, tmp_dir)

    # Replace NN npz files
    for s in SEEDS:
        src = os.path.join(nn_dir, f"nn_h60_seed{s}.npz")
        dst = os.path.join(tmp_dir, f"nn_h60_seed{s}.npz")
        if not os.path.exists(src):
            print(f"  ERROR: missing {src}")
            shutil.rmtree(tmp_dir)
            return None
        shutil.copy2(src, dst)
        print(f"  replaced nn_h60_seed{s}.npz  ({os.path.getsize(dst)/1024:.1f} KB)")

    # Smoke test
    print("  running smoke test...")
    ok = smoke_test(tmp_dir)
    if not ok:
        print(f"  WARNING: smoke test failed for {variant_name}")

    # Build zip
    os.makedirs(ZIPS_DIR, exist_ok=True)
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(tmp_dir)):
            fpath = os.path.join(tmp_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("__"):
                zf.write(fpath, fname)

    shutil.rmtree(tmp_dir)

    size_mb = os.path.getsize(zip_path) / 1e6
    checksum = md5(zip_path)
    print(f"  zip: {zip_path}  ({size_mb:.2f} MB)  md5={checksum}")
    return {"path": zip_path, "size_mb": round(size_mb, 2), "md5": checksum}


def main():
    # Load eval results
    eval_path = os.path.join(HERE, "eval_results.json")
    if not os.path.exists(eval_path):
        print("ERROR: eval_results.json not found. Run eval_variants.py first.")
        sys.exit(1)

    with open(eval_path) as f:
        eval_results = json.load(f)

    top3 = eval_results.get("top_3_by_holdout_pnl", [])
    if not top3:
        print("ERROR: no top_3 found in eval_results.json")
        sys.exit(1)

    print(f"Building zips for top 3 variants: {top3}")

    zip_info = {}
    for variant_name in top3:
        print(f"\n=== Building zip for {variant_name} ===")

        # Determine NN directory
        if variant_name == "S1_ep11":
            # Use T170 dir (the baseline)
            nn_dir = T170_DIR
        else:
            nn_dir = os.path.join(VARIANTS_DIR, variant_name)

        # Build zip name from variant config
        vinfo = eval_results["variants"].get(variant_name, {})
        vcfg = vinfo.get("config", {})
        ep = vcfg.get("epochs", 11)
        lr = vcfg.get("lr", 3e-5)
        lam = vcfg.get("lambda_spo", 30)

        # Format LR nicely
        if lr == 1e-5:
            lr_str = "1e-5"
        elif lr == 3e-5:
            lr_str = "3e-5"
        elif lr == 1e-4:
            lr_str = "1e-4"
        else:
            lr_str = f"{lr:.0e}"

        zip_name = f"submission_050909_iter019_v2N_T87M7_ep{ep}_lr{lr_str}.zip"
        if lam != 30:
            zip_name = f"submission_050909_iter019_v2N_T87M7_ep{ep}_lr{lr_str}_lam{int(lam)}.zip"

        result = build_zip_for_variant(variant_name, nn_dir, zip_name)
        if result:
            zip_info[variant_name] = {
                "zip_name": zip_name,
                **result,
                "holdout_pnl": vinfo.get("holdout_pnl"),
                "delta_vs_v2": vinfo.get("delta_vs_v2"),
                "delta_vs_T170": vinfo.get("delta_vs_T170"),
                "config": vcfg,
            }

    # Update results.json with zip info
    with open(eval_path) as f:
        all_results = json.load(f)

    all_results["top_3_zips"] = zip_info
    all_results["task"] = "T176 T87 NN M7 HP variant sweep"

    results_path = os.path.join(HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results -> {results_path}")

    print("\n=== Summary ===")
    for vname, info in zip_info.items():
        print(f"  {vname}: pnl={info['holdout_pnl']:.4f}  delta_T170={info.get('delta_vs_T170', 0):+.4f}"
              f"  zip={info['zip_name']}")

    return zip_info


if __name__ == "__main__":
    main()
