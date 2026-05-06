"""Build iter_005 submission package: LightGBM + XGBoost ensemble at h=10
(Scheme C 223-d), single-model LightGBM at h=5/20/40/60 (reuse iter_002).

Reads:
    submission/iter_002_lgbm_schemeC/  (Predictor.py / config.json / compute.py /
                                        requirements.txt / model_h{5,20,40,60}.txt)
    experiments/T11_schemeC_multiseed/final_223_model_h10_seed{S}.txt  (LightGBM seeds)
    experiments/T18_xgboost/final_xgb_model_h10_seed{S}.ubj  (XGBoost seeds)
    submission/iter_005_lgbm_xgb_ensemble/Predictor.py  (must exist)
    submission/iter_005_lgbm_xgb_ensemble/thresholds.json (must be written)

Writes:
    submission/iter_005_lgbm_xgb_ensemble/
        Predictor.py            (LightGBM + XGBoost ensemble)
        compute.py              (T3 features, copied from iter_002)
        config.json             (copied from iter_002)
        requirements.txt        (numpy/pandas/lightgbm/xgboost pinned)
        thresholds.json         per-horizon (T, delta) + lgbm_seeds + xgb_seeds
        model_h{5,20,40,60}.txt (copied from iter_002)
        model_h10_seed{S}.txt   (LightGBM final 223-d seeds)
        xgb_model_h10_seed{S}.ubj (XGBoost final seeds)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUBMIT_DIR = os.path.join(ROOT, "submission", "iter_005_lgbm_xgb_ensemble")
ITER002_DIR = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC")
T11_DIR = os.path.join(ROOT, "experiments", "T11_schemeC_multiseed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm-seeds", default="42,1,7,13,100")
    ap.add_argument("--xgb-seeds", default="42")
    args = ap.parse_args()

    lgbm_seeds = [int(x) for x in args.lgbm_seeds.split(",")]
    xgb_seeds = [int(x) for x in args.xgb_seeds.split(",")]
    print(f"Building iter_005 - h=10 lgbm_seeds={lgbm_seeds}, xgb_seeds={xgb_seeds}",
          flush=True)

    os.makedirs(SUBMIT_DIR, exist_ok=True)

    predictor_path = os.path.join(SUBMIT_DIR, "Predictor.py")
    if not os.path.isfile(predictor_path):
        sys.exit(f"missing {predictor_path} - write iter_005 Predictor.py first")
    print(f"  Predictor.py OK")

    # Copy invariant aux from iter_002
    for fn in ("compute.py", "config.json"):
        src = os.path.join(ITER002_DIR, fn)
        dst = os.path.join(SUBMIT_DIR, fn)
        shutil.copy(src, dst)
        print(f"  copied {fn} from iter_002")

    # Custom requirements: add xgboost
    import xgboost as xgb_mod
    req_lines = [
        "numpy==2.4.4",
        "pandas==2.3.3",
        "lightgbm==4.6.0",
        f"xgboost=={xgb_mod.__version__}",
    ]
    with open(os.path.join(SUBMIT_DIR, "requirements.txt"), "w") as f:
        f.write("\n".join(req_lines) + "\n")
    print(f"  wrote requirements.txt with xgboost=={xgb_mod.__version__}")

    # Copy single-horizon LightGBM models from iter_002
    for H in (5, 20, 40, 60):
        src = os.path.join(ITER002_DIR, f"model_h{H}.txt")
        dst = os.path.join(SUBMIT_DIR, f"model_h{H}.txt")
        if not os.path.isfile(src):
            sys.exit(f"missing iter_002 model_h{H}.txt at {src}")
        shutil.copy(src, dst)
        print(f"  copied model_h{H}.txt from iter_002")

    # Copy LightGBM seed models for h=10
    for s in lgbm_seeds:
        src = os.path.join(T11_DIR, f"final_223_model_h10_seed{s}.txt")
        dst = os.path.join(SUBMIT_DIR, f"model_h10_seed{s}.txt")
        if not os.path.isfile(src):
            sys.exit(f"missing T11 final LightGBM model: {src}")
        shutil.copy(src, dst)
        print(f"  copied model_h10_seed{s}.txt (LightGBM)")

    # Copy XGBoost seed models for h=10
    for s in xgb_seeds:
        src = os.path.join(HERE, f"final_xgb_model_h10_seed{s}.ubj")
        dst = os.path.join(SUBMIT_DIR, f"xgb_model_h10_seed{s}.ubj")
        if not os.path.isfile(src):
            sys.exit(f"missing T18 final XGBoost model: {src}")
        shutil.copy(src, dst)
        print(f"  copied xgb_model_h10_seed{s}.ubj")

    # Remove any stale single-model h_10
    stale = os.path.join(SUBMIT_DIR, "model_h10.txt")
    if os.path.isfile(stale):
        os.remove(stale)
        print(f"  removed stale model_h10.txt")

    # Verify thresholds.json
    th_path = os.path.join(SUBMIT_DIR, "thresholds.json")
    if not os.path.isfile(th_path):
        sys.exit(f"missing {th_path} - write thresholds.json first")
    with open(th_path) as f:
        tcfg = json.load(f)
    h10 = next(h for h in tcfg["horizons"] if int(h["h"]) == 10)
    if list(h10.get("lgbm_seeds", [])) != list(lgbm_seeds):
        sys.exit(f"thresholds.json h=10 lgbm_seeds={h10.get('lgbm_seeds')} != {lgbm_seeds}")
    if list(h10.get("xgb_seeds", [])) != list(xgb_seeds):
        sys.exit(f"thresholds.json h=10 xgb_seeds={h10.get('xgb_seeds')} != {xgb_seeds}")
    print(f"  thresholds.json OK")

    print("\n=== iter_005 contents ===")
    for fn in sorted(os.listdir(SUBMIT_DIR)):
        if fn.startswith("__"):
            continue
        path = os.path.join(SUBMIT_DIR, fn)
        size = os.path.getsize(path) if os.path.isfile(path) else "(dir)"
        print(f"  {fn:35s} {size:>12} bytes")


if __name__ == "__main__":
    main()
