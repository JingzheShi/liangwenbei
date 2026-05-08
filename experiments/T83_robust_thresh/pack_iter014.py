"""Pack iter_014_robust_thresh submission.

Reads winner thresholds from thresholds.json (in this dir) and rebuilds the
submission zip by:
  1) Copying iter_013's pkg directory contents (same boosters, fast_features,
     Predictor.py — only thresholds.json differs).
  2) Overwriting thresholds.json with the T83 winner.
  3) Re-zipping into the workdir as submission_<DATE>_iter014_robust_thresh.zip.

The iter_013 boosters are the 5-seed regression ensemble; we keep them
unchanged. Only the EV-gate threshold pair is replaced by the T83 winner.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ITER013_PKG = os.path.join(ROOT, "experiments", "T75_regression_dmid", "iter013_pkg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-name", default=None,
                    help="Output zip filename (default: submission_<MMDDHH>_iter014_robust_thresh.zip)")
    ap.add_argument("--thr-up", type=float, default=None,
                    help="Override thr_up; otherwise use thresholds.json from this dir")
    ap.add_argument("--thr-dn", type=float, default=None)
    args = ap.parse_args()

    # Load thresholds
    thr_path = os.path.join(HERE, "thresholds.json")
    if args.thr_up is not None and args.thr_dn is not None:
        thr_up, thr_dn = args.thr_up, args.thr_dn
        thr_doc = {
            "_doc": (f"iter_014 manual override thr_up={thr_up:.4e}, thr_dn={thr_dn:.4e}. "
                     "Same 5-seed regression boosters as iter_013."),
            "horizons": [
                {"h": 5, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
                {"h": 10, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
                {"h": 20, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
                {"h": 40, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
                {"h": 60, "thr_up": float(thr_up), "thr_dn": float(thr_dn),
                 "active": True, "ensemble_seeds": [1, 7, 13, 42, 100]}
            ]
        }
    else:
        if not os.path.exists(thr_path):
            print(f"[error] {thr_path} not found and no --thr-up/--thr-dn given")
            sys.exit(2)
        with open(thr_path) as f:
            thr_doc = json.load(f)

    # Sanity
    h60 = next(h for h in thr_doc["horizons"] if h["h"] == 60 and h["active"])
    print(f"thr_up={h60['thr_up']:.4e}  thr_dn={h60['thr_dn']:.4e}")

    # Build pkg in temp dir
    pkg_dir = os.path.join(HERE, "iter014_pkg")
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(pkg_dir)

    # Copy iter_013 pkg files (except thresholds.json)
    for fn in os.listdir(ITER013_PKG):
        if fn == "thresholds.json":
            continue
        src = os.path.join(ITER013_PKG, fn)
        dst = os.path.join(pkg_dir, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Write new thresholds.json
    with open(os.path.join(pkg_dir, "thresholds.json"), "w") as f:
        json.dump(thr_doc, f, indent=2)

    # Verify required files
    required = ["Predictor.py", "config.json", "fast_features.py", "fast_features_batch.py",
                "thresholds.json", "requirements.txt"]
    for fn in required:
        p = os.path.join(pkg_dir, fn)
        assert os.path.exists(p), f"missing {fn}"

    # Verify boosters present
    bcount = len([f for f in os.listdir(pkg_dir) if f.startswith("model_h60_seed")])
    assert bcount == 5, f"expected 5 boosters, got {bcount}"
    print(f"  pkg dir: {bcount} boosters + Predictor + features + thresholds")

    # Zip
    if args.zip_name is None:
        date_tag = datetime.datetime.now().strftime("%m%d%H")
        zip_name = f"submission_{date_tag}_iter014_robust_thresh.zip"
    else:
        zip_name = args.zip_name
    zip_path = os.path.join(ROOT, zip_name)

    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(pkg_dir):
            full = os.path.join(pkg_dir, fn)
            if os.path.isdir(full):
                # __pycache__ shouldn't be there but skip if so
                continue
            zf.write(full, fn)

    sz_mb = os.path.getsize(zip_path) / 1e6
    print(f"\n✓ Wrote {zip_path} ({sz_mb:.2f} MB)")

    # Quick smoke test on the pkg using sym=99 / shuffled
    print("\nRunning smoke test (random 100-row × 1000-batch, sym=99 unfamiliar)…")
    import importlib.util
    spec = importlib.util.spec_from_file_location("iter014_pred", os.path.join(pkg_dir, "Predictor.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, pkg_dir)
    spec.loader.exec_module(mod)
    Predictor = mod.Predictor

    import numpy as np
    import pandas as pd
    cfg = json.load(open(os.path.join(pkg_dir, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df_one = pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32),
                          columns=feats)
    p = Predictor()
    out = p.predict([df_one] * 4)
    print(f"  smoke test pass: 4 predictions, len={len(out)}, first={out[0]}")

    # Save record
    with open(os.path.join(HERE, "iter014_pack_record.json"), "w") as f:
        json.dump({"zip_path": zip_path, "size_mb": sz_mb,
                   "thr_up": h60["thr_up"], "thr_dn": h60["thr_dn"],
                   "n_boosters": bcount}, f, indent=2)

    sys.path.remove(pkg_dir)


if __name__ == "__main__":
    main()
