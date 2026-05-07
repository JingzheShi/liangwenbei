"""Build iter_006 submission: Scheme J 583-d aug_a N-seed ensemble for h_60
(h_40 from iter_002 unchanged).

Layout:
    submission/iter_006_schemeJ_aug_a_Nseed_h60/
        Predictor.py             (= experiments/T33/Predictor_iter_006.py)
        compute_schemeJ.py       (= experiments/T33/compute_schemeJ.py)
        config.json              (= iter_002, raw feature col list)
        requirements.txt         (= iter_002)
        model_h5.txt             (= iter_002, INACTIVE)
        model_h10.txt            (= iter_002, INACTIVE)
        model_h20.txt            (= iter_002, INACTIVE)
        model_h40.txt            (= iter_002, ACTIVE — 223-d Scheme C)
        model_h60_seed{S}.txt    (T33 final aug_a, ACTIVE — 583-d Scheme J)
        thresholds.json
        submission.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submission" / "iter_002_lgbm_schemeC"
T33 = ROOT / "experiments" / "T33_long_window"
DST = ROOT / "submission" / "iter_006_schemeJ_aug_a_5seed_h60"
ZIP_OUT = ROOT / "submission_050629_iter006.zip"


def _load_threshold_from_sweep(seeds, label_prefix):
    p = T33 / "threshold_sweep_results.json"
    if not p.exists():
        sys.exit(f"missing {p}; run threshold_sweep.py first")
    with open(p) as f:
        res = json.load(f)
    # Find the ensemble entry
    seeds_str = "_".join(map(str, seeds))
    # Look for any key that contains 'ensemble' + matching seeds
    target_key = None
    for key in res:
        if "ensemble" in key and seeds_str in key:
            target_key = key; break
    if target_key is None:
        # fallback: pick the highest-ensemble result
        for key, val in res.items():
            if "ensemble" in key:
                target_key = key; break
    if target_key is None:
        sys.exit(f"no ensemble entry in {p}")
    b = res[target_key]["best"]
    return float(b["T"]), float(b["delta"]), float(b["sum_cum_pnl"]), \
           float(b["std_cum_pnl"]), int(b["n_pos_folds"]), target_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--T-override", type=float, default=None)
    ap.add_argument("--d-override", type=float, default=None)
    ap.add_argument("--variant", default="aug_a")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_006 ({args.variant} {len(seeds)}-seed Scheme J h_60) ===",
          flush=True)

    # Locate final model files (final_model_h60_{variant}_seed{S}.txt under T33)
    seed_models = []
    for s in seeds:
        p = T33 / f"final_model_h60_{args.variant}_seed{s}.txt"
        if not p.exists():
            sys.exit(f"missing final {args.variant} model: {p}")
        seed_models.append((s, p))
    print(f"  found {len(seed_models)} final models: seeds={[s for s,_ in seed_models]}",
          flush=True)

    # Resolve threshold
    if args.T_override is not None and args.d_override is not None:
        T_ens, d_ens = float(args.T_override), float(args.d_override)
        sweep_meta = {"T": T_ens, "delta": d_ens, "_source": "manual override"}
    else:
        T_ens, d_ens, sum_pnl, std_pnl, pos_folds, src_key = \
            _load_threshold_from_sweep(seeds, args.variant)
        sweep_meta = {"T": T_ens, "delta": d_ens,
                      "sum_cum_pnl": sum_pnl, "std_cum_pnl": std_pnl,
                      "n_pos_folds": pos_folds,
                      "_source": f"T33 ensemble OOF sweep ({src_key})"}
    print(f"  threshold: T={T_ens:.2f} d={d_ens:.2f} ({sweep_meta.get('_source')})",
          flush=True)

    DST.mkdir(parents=True, exist_ok=True)
    # Clean any stale h_60 model files
    for old in DST.glob("model_h60*.txt"):
        old.unlink()

    # Copy iter_002 config / requirements + h_5/h_10/h_20/h_40 models
    for fname in ("config.json", "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}", flush=True)

    # Copy iter_006 Predictor + compute_schemeJ
    shutil.copy2(T33 / "Predictor_iter_006.py", DST / "Predictor.py")
    shutil.copy2(T33 / "compute_schemeJ.py", DST / "compute_schemeJ.py")
    print(f"  cp Predictor_iter_006.py -> Predictor.py + compute_schemeJ.py",
          flush=True)

    # Install h_60 ensemble Scheme J models
    for s, p in seed_models:
        out = DST / f"model_h60_seed{s}.txt"
        shutil.copy2(p, out)
        print(f"  cp Scheme J h_60 seed{s} -> {out.name}  ({p.stat().st_size:,} bytes)",
              flush=True)

    # Thresholds (h_5/10/20 inactive; h_40 from iter_002; h_60 ensemble)
    THRESHOLDS = {
        "_doc": (
            f"iter_006 ({args.variant} {len(seed_models)}-seed Scheme J 583-d h_60 "
            "ensemble + h_40 Scheme C 223-d). h_5/h_10/h_20 disabled."
        ),
        "_source": "T33 OOF threshold sweep on Scheme J ensemble",
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged (Scheme C 223-d)."},
            {"h": 60, "T": T_ens, "delta": d_ens, "active": True,
             "_reason": (
                 f"T33 Scheme J 583-d {args.variant} {len(seed_models)}-seed ensemble. "
                 "Long-window features added on top of iter_005b. "
                 f"Best from OOF threshold sweep."
             ),
             "ensemble_meta": sweep_meta,
             "ensemble_seeds": [s for s, _ in seed_models]},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json", flush=True)

    # Sanity test (22 batches, shuffled, OOR sym)
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]

    spec = importlib.util.spec_from_file_location("iter006_predictor", DST / "Predictor.py")
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
    files = ["Predictor.py", "compute_schemeJ.py", "config.json", "requirements.txt",
             "thresholds.json",
             "model_h5.txt", "model_h10.txt", "model_h20.txt",
             "model_h40.txt"]
    files += [f"model_h60_seed{s}.txt" for s, _ in seed_models]
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)",
          flush=True)


if __name__ == "__main__":
    main()
