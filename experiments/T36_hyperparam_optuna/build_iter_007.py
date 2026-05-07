"""T36: Build iter_007 submission package — Optuna best hyperparam × 5-seed h_60.

Layout:
    submission/iter_007_optuna_h60/
        Predictor.py   (= iter_006 Predictor; supports asym thresholds)
        compute.py     (= iter_002)
        config.json    (= iter_002)
        requirements.txt (= iter_002)
        thresholds.json
        model_h5/h10/h20.txt   (= iter_002, INACTIVE)
        model_h40.txt          (= iter_002, ACTIVE)
        model_h60_seed{42,1,7,13,100}.txt  (NEW from train_final_optuna.py)
        submission.zip

Threshold:
    h_60: prefers OOF threshold from Optuna 5-fold OOF sweep (passed via
          --thresholds-json). Defaults to iter_006 asymmetric DE optimum
          if no override.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_006 = ROOT / "submission" / "iter_006_aug_a_5seed_h60_asymthresh"
SRC_002 = ROOT / "submission" / "iter_002_lgbm_schemeC"
T36 = ROOT / "experiments" / "T36_hyperparam_optuna"
DST = ROOT / "submission" / "iter_007_optuna_h60"
ZIP_OUT = ROOT / "submission_050631_iter007.zip"

DEFAULT_SEEDS = (42, 1, 7, 13, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--thresholds-json", default=None,
                    help="path to thresholds dict {h_60: {T_up,T_dn,d_up,d_dn} OR {T,delta}}")
    ap.add_argument("--inherit-iter006-thresh", action="store_true",
                    help="copy iter_006 asym DE thresholds (default if no thresholds-json)")
    ap.add_argument("--final-prefix", default="final_model_h60_optuna_seed",
                    help="prefix of trained model files")
    ap.add_argument("--keep-existing-zip", action="store_true",
                    help="don't overwrite existing zip if present")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_007 (Optuna {len(seeds)}-seed ensemble h_60) ===", flush=True)

    # Locate final h_60 models
    seed_models = []
    for s in seeds:
        p = T36 / f"{args.final_prefix}{s}.txt"
        if not p.exists():
            sys.exit(f"missing final model: {p}  (run train_final_optuna.py first)")
        seed_models.append((s, p))
    print(f"  {len(seed_models)} final models found", flush=True)

    # Resolve threshold
    if args.thresholds_json:
        with open(args.thresholds_json) as f:
            thresh = json.load(f)
        h60_thresh = thresh["h_60"]
        thresh_source = f"override from {args.thresholds_json}"
    else:
        # Inherit iter_006 asym DE thresholds
        with open(SRC_006 / "thresholds.json") as f:
            t006 = json.load(f)
        h60_thresh = next(h for h in t006["horizons"] if h["h"] == 60)
        thresh_source = "inherited iter_006 asym DE"
    print(f"  h_60 threshold: {h60_thresh} (source={thresh_source})", flush=True)

    # Stage destination
    DST.mkdir(parents=True, exist_ok=True)
    for old in DST.glob("model_h60*.txt"):
        old.unlink()

    # Copy iter_002 (h_5/h_10/h_20/h_40 models, compute, config, requirements)
    for fname in ("compute.py", "config.json", "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC_002 / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}", flush=True)

    # Predictor: reuse iter_006's (asym threshold support)
    shutil.copy2(SRC_006 / "Predictor.py", DST / "Predictor.py")
    print(f"  cp iter_006/Predictor.py -> Predictor.py (asym-capable)", flush=True)

    # Install h_60 models
    for s, p in seed_models:
        out = DST / f"model_h60_seed{s}.txt"
        shutil.copy2(p, out)
        print(f"  cp {p.name} -> {out.name}  ({p.stat().st_size:,} bytes)", flush=True)

    # Thresholds
    THRESHOLDS = {
        "_doc": (
            f"iter_007: Optuna-best hyperparam {len(seed_models)}-seed h_60 ensemble. "
            f"h_5/h_10/h_20 disabled. h_40 from iter_002 baseline."
        ),
        "_source": "T36 Optuna hyperparam search + " + thresh_source,
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged."},
            {**h60_thresh, "h": 60, "active": True,
             "ensemble_seeds": [s for s, _ in seed_models],
             "_source_T36": "Optuna best hyperparam"},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json", flush=True)

    # Sanity test (22 batches, shuffled, OOR-resistant)
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]
    spec = importlib.util.spec_from_file_location("iter007_predictor", DST / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    batches = []
    for i in range(22):
        df = pd.DataFrame(
            rng.standard_normal((100, len(feats))).astype(np.float32),
            columns=feats,
        )
        batches.append(df)
    perm = rng.permutation(len(batches))
    batches = [batches[i] for i in perm]

    p = mod.Predictor()
    out = p.predict(batches)
    assert isinstance(out, list) and len(out) == 22, f"expected 22, got {len(out)}"
    for row in out:
        assert isinstance(row, list) and len(row) == 5, f"row shape {row}"
        for v in row:
            assert v in (0, 1, 2), f"invalid pred {v}"
    for row in out:
        assert row[0] == 1 and row[1] == 1 and row[2] == 1, \
            f"expected idx 0/1/2 = 1 (inactive), got {row}"
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20 = 1 (inactive)",
          flush=True)

    # Zip
    files = ["Predictor.py", "compute.py", "config.json", "requirements.txt",
             "thresholds.json",
             "model_h5.txt", "model_h10.txt", "model_h20.txt",
             "model_h40.txt"]
    files += [f"model_h60_seed{s}.txt" for s, _ in seed_models]
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    if not args.keep_existing_zip or not ZIP_OUT.exists():
        shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)",
          flush=True)
    print("\n--- iter_007 contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
