"""Build iter_003 submission package: 5-seed ensemble at h=10 (Scheme C 223-d),
single-model at h=5/20/40/60 (reuse iter_002).

Reads:
    experiments/T11_schemeC_multiseed/threshold_results.json   (h=10 ensemble best)
    experiments/T11_schemeC_multiseed/final_223_model_h10_seed{S}.txt  (5 seeds)
    submission/iter_002_lgbm_schemeC/  (Predictor.py / config.json / compute.py /
                                        requirements.txt / model_h{5,20,40,60}.txt /
                                        thresholds.json)
    submission/iter_003_lgbm_schemeC_5seed/Predictor.py  (must exist; writer's job)

Writes:
    submission/iter_003_lgbm_schemeC_5seed/
        Predictor.py            (multi-seed ensemble, supports `seeds` field)
        compute.py              (T3 features, copied from iter_002)
        config.json             (copied from iter_002 = 154 raw cols)
        requirements.txt        (numpy/pandas/lightgbm pinned)
        thresholds.json         per-horizon (T, delta) + active flag + h=10 seeds list
        model_h{5,20,40,60}.txt (copied from iter_002)
        model_h10_seed{42,1,7,13,100}.txt (final 223-d models)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUBMIT_DIR = os.path.join(ROOT, "submission", "iter_003_lgbm_schemeC_5seed")
ITER002_DIR = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"Building iter_003 — h=10 ensemble seeds={seeds}, other horizons reuse iter_002")

    # 1. iter_003 has Predictor.py already (manually written). Verify.
    predictor_path = os.path.join(SUBMIT_DIR, "Predictor.py")
    if not os.path.isfile(predictor_path):
        sys.exit(f"missing {predictor_path} — write iter_003 Predictor.py first")
    print(f"  Predictor.py OK")

    # 2. Copy invariant aux from iter_002
    for fn in ("compute.py", "config.json", "requirements.txt"):
        src = os.path.join(ITER002_DIR, fn)
        dst = os.path.join(SUBMIT_DIR, fn)
        shutil.copy(src, dst)
        print(f"  copied {fn} from iter_002")

    # 3. Copy single-horizon models from iter_002 (h=5, 20, 40, 60)
    for H in (5, 20, 40, 60):
        src = os.path.join(ITER002_DIR, f"model_h{H}.txt")
        dst = os.path.join(SUBMIT_DIR, f"model_h{H}.txt")
        if not os.path.isfile(src):
            sys.exit(f"missing iter_002 model_h{H}.txt at {src}")
        shutil.copy(src, dst)
        print(f"  copied model_h{H}.txt from iter_002")

    # 4. Copy 5 seed models for h=10
    for s in seeds:
        src = os.path.join(HERE, f"final_223_model_h10_seed{s}.txt")
        dst = os.path.join(SUBMIT_DIR, f"model_h10_seed{s}.txt")
        if not os.path.isfile(src):
            sys.exit(f"missing T11 final model: {src}")
        shutil.copy(src, dst)
        print(f"  copied model_h10_seed{s}.txt")

    # 5. Remove any stale single-model h_10 (from earlier copy)
    stale = os.path.join(SUBMIT_DIR, "model_h10.txt")
    if os.path.isfile(stale):
        os.remove(stale)
        print(f"  removed stale model_h10.txt (replaced by ensemble)")

    # 6. Verify thresholds.json (manually written) is consistent.
    th_path = os.path.join(SUBMIT_DIR, "thresholds.json")
    if not os.path.isfile(th_path):
        sys.exit(f"missing {th_path} — write thresholds.json first")
    with open(th_path) as f:
        tcfg = json.load(f)
    h10 = next(h for h in tcfg["horizons"] if int(h["h"]) == 10)
    if list(h10.get("seeds", [])) != list(seeds):
        sys.exit(f"thresholds.json h=10 seeds={h10.get('seeds')} != {seeds}")
    print(f"  thresholds.json OK (h=10 seeds match)")

    # 7. Final summary
    print("\n=== iter_003 contents ===")
    for fn in sorted(os.listdir(SUBMIT_DIR)):
        if fn.startswith("__"):
            continue
        path = os.path.join(SUBMIT_DIR, fn)
        size = os.path.getsize(path) if os.path.isfile(path) else "(dir)"
        print(f"  {fn:35s} {size:>12} bytes")


if __name__ == "__main__":
    main()
